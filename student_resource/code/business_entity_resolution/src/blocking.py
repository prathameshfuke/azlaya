"""Step 2: blocking / candidate generation, built to scale to ~10M+ records.

For each S1 entity, finds its most similar S2/S3 records within the same `country` (an OPEN SET:
partitions are whatever country values appear in the data, never a hardcoded list), using two
TF-IDF views searched with a multithreaded sparse top-n matrix product (sparse_dot_topn):
  - name:    character 4-grams of name_norm (typo/abbreviation tolerant)
  - address: word tokens of address_norm (pin/postal codes, city, street -- this is the
             pin/city blocking signal)
Per S1 entity the two views' candidates are unioned, scored by name_sim + address_sim, and the
top K kept.

Why not Python inverted indices: the train split alone is ~12.5M records. Python dict/set/
Counter structures per record exhausted Kaggle's 30 GB and would take many hours to loop over
2.2M S1 entities. Sparse CSR matrices hold the same information in a few GB, and the top-n product
runs in C++ across every core. N-grams shared by more than MAX_DF records are dropped: they
carry no signal and would dominate both memory and compute.

TRAIN split: blocking (and everything downstream) uses a random sample of --max-s1 S1 entities
(default 300k of ~2.2M). Their candidates are still searched against the FULL S2/S3 pool, so hard
negatives stay as hard as at test time. The rest of the pipeline reads the sample back from
candidate_pairs_train.tsv (common.sampled_ground_truth). TEST split: always every S1 entity.

Run (from code/business_entity_resolution/):
    python -m src.blocking --split train --out ../../data_processed/candidate_pairs_train.tsv --report-recall
    python -m src.blocking --split test      # predict.py does this itself too
"""
from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd
from scipy import sparse
from tqdm.auto import tqdm

from src import common

DEFAULT_K = 30
DEFAULT_MAX_TRAIN_S1 = 300_000
NAME_TOP_N = 25
ADDRESS_TOP_N = 15
NAME_THRESHOLD = 0.1
ADDRESS_THRESHOLD = 0.1
MAX_DF = 10_000          # drop n-grams/tokens appearing in more records than this (per country)
S1_CHUNK = 20_000
BLOCKING_COLUMNS = ["entity_id", "country", "name_norm", "address_norm"]


def _vectorizer(view: str):
    from sklearn.feature_extraction.text import TfidfVectorizer

    if view == "name":
        return TfidfVectorizer(analyzer="char_wb", ngram_range=(4, 4), max_df=MAX_DF,
                               sublinear_tf=True, dtype=np.float32)
    return TfidfVectorizer(analyzer="word", token_pattern=r"(?u)\b\w+\b", max_df=MAX_DF,
                           sublinear_tf=True, dtype=np.float32)


def _topn_view(view: str, pool_texts: List[str], s1_texts: List[str], top_n: int,
               threshold: float, label: str) -> Optional[sparse.csr_matrix]:
    """(n_s1 x n_pool) sparse matrix of each S1 row's top_n cosine similarities against the pool,
    or None when the view has no usable vocabulary for this partition (e.g. every address empty)."""
    from sparse_dot_topn import sp_matmul_topn

    vec = _vectorizer(view)
    try:
        pool_m = vec.fit_transform(pool_texts)
    except ValueError:  # empty vocabulary / everything pruned by max_df
        return None
    pool_t = pool_m.T.tocsr()
    del pool_m
    s1_m = vec.transform(s1_texts).tocsr()
    n_threads = os.cpu_count() or 1
    parts = []
    for start in tqdm(range(0, s1_m.shape[0], S1_CHUNK), desc=f"{label} {view} top-{top_n}",
                      unit="chunk", leave=False):
        parts.append(sp_matmul_topn(s1_m[start:start + S1_CHUNK], pool_t, top_n=top_n,
                                    threshold=threshold, n_threads=n_threads).tocsr())
    return sparse.vstack(parts, format="csr") if parts else None


def _blocking_partition(s1_ids, s1_names, s1_addrs, pool_ids, pool_names, pool_addrs, k: int,
                        label: str) -> Dict[str, List[str]]:
    views = [
        _topn_view("name", pool_names, s1_names, NAME_TOP_N, NAME_THRESHOLD, label),
        _topn_view("address", pool_addrs, s1_addrs, ADDRESS_TOP_N, ADDRESS_THRESHOLD, label),
    ]
    views = [v for v in views if v is not None]
    out: Dict[str, List[str]] = {}
    if not views:
        return {s1: [] for s1 in s1_ids}
    combined = views[0] if len(views) == 1 else (views[0] + views[1]).tocsr()
    indptr, indices, data = combined.indptr, combined.indices, combined.data
    for row, s1 in enumerate(s1_ids):
        lo, hi = indptr[row], indptr[row + 1]
        if hi - lo > k:
            keep = np.argpartition(-data[lo:hi], k)[:k]
            order = keep[np.argsort(-data[lo:hi][keep])]
        else:
            order = np.argsort(-data[lo:hi])
        out[s1] = [pool_ids[j] for j in indices[lo:hi][order]]
    return out


def generate_candidates(s1_df: pd.DataFrame, s2_df: pd.DataFrame, s3_df: pd.DataFrame,
                        k: int = DEFAULT_K) -> Dict[str, List[str]]:
    """{s1_entity_id: [candidate_id, ...]} ranked by name_sim + address_sim, capped at k. Every S1
    row gets a key, even with no candidates (e.g. a country with no S2/S3 records)."""
    pool = pd.concat([s2_df[BLOCKING_COLUMNS], s3_df[BLOCKING_COLUMNS]], ignore_index=True)
    pool_by_country = {c: g for c, g in pool.groupby("country", sort=False)}
    results: Dict[str, List[str]] = {}
    for country, s1_group in tqdm(list(s1_df.groupby("country", sort=False)), desc="blocking countries",
                                  unit="country"):
        s1_ids = s1_group["entity_id"].tolist()
        pool_group = pool_by_country.get(country)
        if pool_group is None or pool_group.empty:
            results.update({s1: [] for s1 in s1_ids})
            continue
        print(f"[blocking] {country}: {len(s1_ids)} S1 vs {len(pool_group)} S2/S3 records")
        results.update(_blocking_partition(
            s1_ids, s1_group["name_norm"].tolist(), s1_group["address_norm"].tolist(),
            pool_group["entity_id"].to_numpy(), pool_group["name_norm"].tolist(),
            pool_group["address_norm"].tolist(), k, label=str(country)))
    return results


def sample_s1(s1_df: pd.DataFrame, max_s1: Optional[int], seed: int) -> pd.DataFrame:
    if max_s1 is None or len(s1_df) <= max_s1:
        return s1_df
    return s1_df.sample(n=max_s1, random_state=seed).reset_index(drop=True)


# ---------------------------------------------------------------------------- recall diagnostics

def report_blocking_recall(candidate_ids: Dict[str, List[str]], ground_truth: pd.DataFrame,
                           eval_ids: Set[str] = None) -> Dict[str, float]:
    """Fraction of true matches that survive into the candidate set -- the recall CEILING for
    everything downstream. `eval_ids` restricts the report to e.g. the held-out val split."""
    truth = common.ground_truth_map(ground_truth)
    total_true = total_recalled = 0
    per_entity_recall = []
    singleton_correct = singleton_total = 0
    for s1, true_ids in truth.items():
        if eval_ids is not None and s1 not in eval_ids:
            continue
        cand_ids = set(candidate_ids.get(s1, ()))
        if not true_ids:
            singleton_total += 1
            singleton_correct += not cand_ids
            continue
        recalled = len(true_ids & cand_ids)
        total_true += len(true_ids)
        total_recalled += recalled
        per_entity_recall.append(recalled / len(true_ids))

    return {
        "n_nonsingleton_entities": len(per_entity_recall),
        "overall_pair_recall": (total_recalled / total_true) if total_true else float("nan"),
        "macro_entity_recall": (sum(per_entity_recall) / len(per_entity_recall)) if per_entity_recall else float("nan"),
        "n_singleton_entities": singleton_total,
        "singleton_empty_candidate_rate": (singleton_correct / singleton_total) if singleton_total else float("nan"),
    }


# --------------------------------------------------------------------------------------- driver

def load_blocking_inputs(repo_root, split: str):
    return [common.load_normalized(repo_root, split, key, columns=BLOCKING_COLUMNS)
            for key in common.SOURCE_KEYS]


def run(repo_root, split: str, k: int, out_path, report_recall: bool, val_frac: float, seed: int,
        max_s1: Optional[int]):
    s1_df, s2_df, s3_df = load_blocking_inputs(repo_root, split)
    if split == "train":
        s1_df = sample_s1(s1_df, max_s1, seed)
    print(f"[blocking] {split}: {len(s1_df)} S1 (after sampling) / {len(s2_df)} S2 / {len(s3_df)} S3, k={k}")
    id_map = generate_candidates(s1_df, s2_df, s3_df, k=k)
    del s2_df, s3_df

    out_path = out_path or (common.output_dir(repo_root) / "candidate_pairs.tsv")
    common.write_id_list_tsv(id_map, out_path, "source1_entity_id", "candidate_entity_ids")
    print(f"[blocking] wrote {out_path} ({len(id_map)} S1 rows, {sum(len(v) for v in id_map.values())} pairs)")

    if report_recall:
        if split != "train":
            raise SystemExit("--report-recall needs ground truth, which only exists for --split train")
        ground_truth = common.sampled_ground_truth(repo_root, out_path)
        _, val_ids = common.stratified_split_by_s1(ground_truth, val_frac=val_frac, seed=seed)
        stats = report_blocking_recall(id_map, ground_truth, eval_ids=val_ids)
        print(f"[blocking] recall on held-out val split ({len(val_ids)} sampled S1 entities):")
        for key, value in stats.items():
            print(f"    {key}: {value}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    common.add_repo_root_arg(parser)
    parser.add_argument("--split", choices=list(common.SPLITS), required=True)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--max-s1", type=int, default=DEFAULT_MAX_TRAIN_S1,
                        help="[train only] S1 entities sampled for training. 0 = all.")
    parser.add_argument("--out", type=str, default=None,
                        help="Output path. Default: output/candidate_pairs.tsv")
    parser.add_argument("--report-recall", action="store_true")
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(args.repo_root, args.split, args.k, args.out, args.report_recall, args.val_frac, args.seed,
        args.max_s1 or None)


if __name__ == "__main__":
    main()

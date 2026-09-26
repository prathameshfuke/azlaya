"""Step 2: blocking / candidate generation.

For each S1 entity, generates up to K candidate S2/S3 records using:
  (a) a normalized-name word-token inverted index
  (b) a normalized-name character-trigram inverted index (typo tolerance)
  (c) a pin_code / city_guess secondary inverted index
all partitioned by `country` first (country is an OPEN SET: the partitioning code branches on
"whatever values are present in this data", never on a hardcoded {US, India} list).

Memory: an earlier version kept every record's trigram set plus uncapped postings lists, which
exhausted Kaggle's ~30 GB on the full dataset -- a trigram like "ted" or a token like "limited"
matches most of a partition. Now keys shared by more than MAX_POSTINGS records are dropped from
the index (they carry no discriminating signal anyway), postings are stored as compact uint32
arrays, and each query pre-ranks positions by how many index keys they share with the S1 record
(keeping the top PREFILTER) before exact Jaccard/TF-IDF scoring.

Run (from code/business_entity_resolution/):
    python -m src.blocking --split train --k 30 --out ../../data_processed/candidate_pairs_train.tsv --report-recall
    python -m src.blocking --split test --k 30        # predict.py does this itself too
"""
from __future__ import annotations

import argparse
from array import array
from collections import Counter, defaultdict
from typing import Dict, List, Set

import pandas as pd
from tqdm.auto import tqdm

from src import common

DEFAULT_K = 30
NGRAM_N = 3
# Index keys shared by more records than this (within one country/source partition) are dropped.
# Pin/city get a looser cap since a shared postal code is a much stronger hint than a trigram.
MAX_POSTINGS = 2000
MAX_LOCATION_POSTINGS = 5000
# Positions kept per query (by number of shared index keys) before exact scoring.
PREFILTER = 300


def char_ngrams(text: str, n: int = NGRAM_N) -> Set[str]:
    text = text.replace(" ", "")
    if len(text) < n:
        return {text} if text else set()
    return {text[i:i + n] for i in range(len(text) - n + 1)}


def jaccard(a, b) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / len(a | b) if inter else 0.0


def _freeze(index: Dict[str, List[int]], max_len: int) -> Dict[str, array]:
    return {k: array("I", v) for k, v in index.items() if len(v) <= max_len}


class _PartitionIndex:
    """Inverted indices over one (country, source) partition of records."""

    def __init__(self, ids: List[str], names: List[str], pins: List[str], cities: List[str]):
        self.ids = ids
        self.token_sets = [frozenset(n.split()) for n in names]
        token_index: Dict[str, List[int]] = defaultdict(list)
        ngram_index: Dict[str, List[int]] = defaultdict(list)
        pin_index: Dict[str, List[int]] = defaultdict(list)
        city_index: Dict[str, List[int]] = defaultdict(list)
        for i, name in enumerate(names):
            for t in self.token_sets[i]:
                token_index[t].append(i)
            for ng in char_ngrams(name):
                ngram_index[ng].append(i)
            if pins[i]:
                pin_index[pins[i]].append(i)
            if cities[i]:
                city_index[cities[i]].append(i)
        self.token_index = _freeze(token_index, MAX_POSTINGS)
        self.ngram_index = _freeze(ngram_index, MAX_POSTINGS)
        self.pin_index = _freeze(pin_index, MAX_LOCATION_POSTINGS)
        self.city_index = _freeze(city_index, MAX_LOCATION_POSTINGS)

    def candidate_positions(self, tokens, ngrams, pin: str, city: str, prefilter: int) -> List[int]:
        hits: Counter = Counter()
        for t in tokens:
            post = self.token_index.get(t)
            if post is not None:
                hits.update(post)
        for ng in ngrams:
            post = self.ngram_index.get(ng)
            if post is not None:
                hits.update(post)
        for key, index in ((pin, self.pin_index), (city, self.city_index)):
            if key:
                post = index.get(key)
                if post is not None:
                    hits.update(post)
        if len(hits) <= prefilter:
            return list(hits)
        return [pos for pos, _ in hits.most_common(prefilter)]


def _build_partition_indices(df: pd.DataFrame, label: str) -> Dict[str, _PartitionIndex]:
    """One inverted index per distinct `country` value found in `df` -- built dynamically from
    whatever countries are present, never from a fixed list."""
    indices: Dict[str, _PartitionIndex] = {}
    groups = list(df.groupby("country", sort=False))
    for country, group in tqdm(groups, desc=f"index {label}", unit="country"):
        indices[country] = _PartitionIndex(
            ids=group["entity_id"].tolist(),
            names=group["name_norm"].tolist(),
            pins=group["pin_code"].tolist(),
            cities=group["city_guess"].tolist(),
        )
    return indices


def generate_candidates(
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    k: int = DEFAULT_K,
    score_method: str = "jaccard",
    prefilter: int = PREFILTER,
) -> Dict[str, List[str]]:
    """Returns {s1_entity_id: [candidate_id, ...]} ranked by score desc, capped at `k`. Every S1
    row gets a key, even with no candidates.

    score_method: "jaccard" (default, word-token Jaccard on name_norm) or "tfidf" (TF-IDF cosine
    on word 1-2 grams, fit per country). Compare them with --report-recall before choosing.
    """
    if score_method not in ("jaccard", "tfidf"):
        raise ValueError(f"unknown score_method: {score_method!r}")

    s2_indices = _build_partition_indices(s2_df, "S2")
    s3_indices = _build_partition_indices(s3_df, "S3")
    tfidf_by_country = _build_tfidf_by_country(s2_df, s3_df) if score_method == "tfidf" else None

    results: Dict[str, List[str]] = {}
    cols = zip(s1_df["entity_id"], s1_df["country"], s1_df["name_norm"],
               s1_df["pin_code"], s1_df["city_guess"])
    for s1_id, country, name_norm, pin_code, city_guess in tqdm(
            cols, total=len(s1_df), desc="blocking S1", unit="entity", mininterval=2.0):
        tokens = frozenset(name_norm.split())
        ngrams = char_ngrams(name_norm)
        s1_vec = None
        if tfidf_by_country is not None and country in tfidf_by_country:
            s1_vec = tfidf_by_country[country]["vectorizer"].transform([name_norm])

        scored = []
        for indices in (s2_indices, s3_indices):
            index = indices.get(country)
            if index is None:
                continue
            for pos in index.candidate_positions(tokens, ngrams, pin_code, city_guess, prefilter):
                cand_id = index.ids[pos]
                if tfidf_by_country is None:
                    score = jaccard(tokens, index.token_sets[pos])
                else:
                    score = _tfidf_cosine(tfidf_by_country, country, cand_id, s1_vec)
                scored.append((score, cand_id))

        scored.sort(reverse=True)
        results[s1_id] = [cid for _, cid in scored[:k]]

    return results


# ----------------------------------------------------------------------- TF-IDF scoring backend

def _build_tfidf_by_country(s2_df: pd.DataFrame, s3_df: pd.DataFrame):
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer

    combined = pd.concat([s2_df[["entity_id", "name_norm", "country"]],
                          s3_df[["entity_id", "name_norm", "country"]]], ignore_index=True)
    out = {}
    for country, group in tqdm(list(combined.groupby("country", sort=False)), desc="tfidf fit", unit="country"):
        vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1, dtype=np.float32)
        matrix = vectorizer.fit_transform(group["name_norm"].tolist()).tocsr()
        id_to_row = {eid: i for i, eid in enumerate(group["entity_id"].tolist())}
        out[country] = {"vectorizer": vectorizer, "matrix": matrix, "id_to_row": id_to_row}
    return out


def _tfidf_cosine(tfidf_by_country, country: str, cand_id: str, s1_vec) -> float:
    """Dot product of L2-normalized TF-IDF rows == cosine."""
    if s1_vec is None:
        return 0.0
    bucket = tfidf_by_country.get(country)
    if bucket is None:
        return 0.0
    row_idx = bucket["id_to_row"].get(cand_id)
    if row_idx is None:
        return 0.0
    return float(bucket["matrix"][row_idx].multiply(s1_vec).sum())


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

def run(repo_root, split: str, k: int, score_method: str, out_path, report_recall: bool,
        val_frac: float, seed: int, prefilter: int):
    normalized = common.load_all_normalized(repo_root, split)
    s1_df, s2_df, s3_df = normalized["source1"], normalized["source2"], normalized["source3"]

    print(f"[blocking] {split}: {len(s1_df)} S1 / {len(s2_df)} S2 / {len(s3_df)} S3 rows, "
          f"k={k}, score_method={score_method}, prefilter={prefilter}")
    id_map = generate_candidates(s1_df, s2_df, s3_df, k=k, score_method=score_method, prefilter=prefilter)

    out_path = out_path or (common.output_dir(repo_root) / "candidate_pairs.tsv")
    common.write_id_list_tsv(id_map, out_path, "source1_entity_id", "candidate_entity_ids")
    print(f"[blocking] wrote {out_path} ({sum(len(v) for v in id_map.values())} pairs)")

    if report_recall:
        if split != "train":
            raise SystemExit("--report-recall needs ground truth, which only exists for --split train")
        ground_truth = common.load_split_sources(repo_root, "train")["ground_truth"]
        _, val_ids = common.stratified_split_by_s1(ground_truth, val_frac=val_frac, seed=seed)
        stats = report_blocking_recall(id_map, ground_truth, eval_ids=val_ids)
        print(f"[blocking] recall on held-out val split ({len(val_ids)} S1 entities):")
        for key, value in stats.items():
            print(f"    {key}: {value}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    common.add_repo_root_arg(parser)
    parser.add_argument("--split", choices=list(common.SPLITS), required=True)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--score-method", choices=["jaccard", "tfidf"], default="jaccard")
    parser.add_argument("--prefilter", type=int, default=PREFILTER)
    parser.add_argument("--out", type=str, default=None,
                        help="Output path. Default: output/candidate_pairs.tsv")
    parser.add_argument("--report-recall", action="store_true")
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(args.repo_root, args.split, args.k, args.score_method, args.out,
        args.report_recall, args.val_frac, args.seed, args.prefilter)


if __name__ == "__main__":
    main()

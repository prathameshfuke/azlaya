"""Step 2: blocking / candidate generation.

For each S1 entity, generates up to K candidate S2/S3 records using:
  (a) a normalized-name word-token inverted index
  (b) a normalized-name character-trigram inverted index (typo tolerance)
  (c) a pin_code / city_guess secondary inverted index
all partitioned by `country` first (country is an OPEN SET: the partitioning code branches on
"whatever values are present in this data", never on a hardcoded {US, India} list).

Candidates are unioned across (a)+(b)+(c), scored (Jaccard on name tokens by default, or TF-IDF
cosine with --score-method tfidf), and capped at the top K per S1 entity.

Not executed here. `report_blocking_recall` is ready to run on the GPU machine but is not run in
this environment; do that before trusting the recall numbers, and before picking a final K.

Run (from code/business_entity_resolution/):
    python -m src.blocking --split test --k 30
        -> output/candidate_pairs.tsv   (the file predict.py's test run also produces directly;
                                          running this standalone is mainly for inspection)
    python -m src.blocking --split train --k 30 --out data_processed/candidate_pairs_train.tsv
    python -m src.blocking --split train --report-recall
        -> builds train candidates, holds out a val split of S1 entities (same split logic as
           train_gbdt.py), and prints blocking recall on that held-out slice.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import pandas as pd

from src import common

DEFAULT_K = 30
NGRAM_N = 3


def char_ngrams(text: str, n: int = NGRAM_N) -> Set[str]:
    text = text.replace(" ", "")
    if len(text) < n:
        return {text} if text else set()
    return {text[i:i + n] for i in range(len(text) - n + 1)}


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / len(a | b)


class _PartitionIndex:
    """Inverted indices over one (country, source) partition of records."""

    def __init__(self, ids: List[str], names: List[str], pins: List[str], cities: List[str]):
        self.ids = ids
        self.token_sets: List[Set[str]] = [set(n.split()) for n in names]
        self.ngram_sets: List[Set[str]] = [char_ngrams(n) for n in names]
        self.token_index: Dict[str, List[int]] = defaultdict(list)
        self.ngram_index: Dict[str, List[int]] = defaultdict(list)
        self.pin_index: Dict[str, List[int]] = defaultdict(list)
        self.city_index: Dict[str, List[int]] = defaultdict(list)
        for i in range(len(ids)):
            for t in self.token_sets[i]:
                self.token_index[t].append(i)
            for ng in self.ngram_sets[i]:
                self.ngram_index[ng].append(i)
            if pins[i]:
                self.pin_index[pins[i]].append(i)
            if cities[i]:
                self.city_index[cities[i]].append(i)

    def candidate_positions(self, tokens: Set[str], ngrams: Set[str], pin: str, city: str) -> Set[int]:
        positions: Set[int] = set()
        for t in tokens:
            positions.update(self.token_index.get(t, ()))
        for ng in ngrams:
            positions.update(self.ngram_index.get(ng, ()))
        if pin:
            positions.update(self.pin_index.get(pin, ()))
        if city:
            positions.update(self.city_index.get(city, ()))
        return positions


def _build_partition_indices(df: pd.DataFrame) -> Dict[str, _PartitionIndex]:
    """One inverted index per distinct `country` value found in `df` -- built dynamically from
    whatever countries are present, never from a fixed list."""
    indices: Dict[str, _PartitionIndex] = {}
    for country, group in df.groupby("country", sort=False):
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
) -> Dict[str, List[Tuple[str, float, str]]]:
    """Returns {s1_entity_id: [(candidate_id, score, source_key), ...]} sorted by score desc,
    capped at `k`. `source_key` is "source2" or "source3", handy for features.py's source-pair
    feature without re-deriving it from the ID prefix.

    score_method: "jaccard" (default, cheap word-token Jaccard on name_norm) or "tfidf" (TF-IDF
    cosine on word n-grams, fit per country partition -- more expensive, potentially more
    precise; NEEDS GPU-MACHINE VERIFICATION of the recall/precision trade-off between the two,
    via report_blocking_recall, before picking one for the final submission).
    """
    if score_method not in ("jaccard", "tfidf"):
        raise ValueError(f"unknown score_method: {score_method!r}")

    s2_indices = _build_partition_indices(s2_df)
    s3_indices = _build_partition_indices(s3_df)

    tfidf_by_country = None
    if score_method == "tfidf":
        tfidf_by_country = _build_tfidf_by_country(s2_df, s3_df)

    results: Dict[str, List[Tuple[str, float, str]]] = {}
    for row in s1_df.itertuples(index=False):
        row_d = row._asdict()
        s1_id = row_d["entity_id"]
        country = row_d["country"]
        name_norm = row_d["name_norm"]
        pin_code = row_d["pin_code"]
        city_guess = row_d["city_guess"]
        tokens = set(name_norm.split())
        ngrams = char_ngrams(name_norm)

        s1_tfidf_vec = None
        if score_method == "tfidf" and country in tfidf_by_country:
            s1_tfidf_vec = tfidf_by_country[country]["vectorizer"].transform([name_norm])

        scored: List[Tuple[str, float, str]] = []
        for source_key, indices in (("source2", s2_indices), ("source3", s3_indices)):
            index = indices.get(country)
            if index is None:
                continue
            positions = index.candidate_positions(tokens, ngrams, pin_code, city_guess)
            for pos in positions:
                cand_id = index.ids[pos]
                if score_method == "jaccard":
                    score = jaccard(tokens, index.token_sets[pos])
                else:
                    score = _tfidf_cosine(tfidf_by_country, country, cand_id, s1_tfidf_vec)
                scored.append((cand_id, score, source_key))

        scored.sort(key=lambda t: t[1], reverse=True)
        results[s1_id] = scored[:k]

    return results


# ----------------------------------------------------------------------- TF-IDF scoring backend

def _build_tfidf_by_country(s2_df: pd.DataFrame, s3_df: pd.DataFrame):
    """Fits one TfidfVectorizer per country over the pooled S2+S3 name_norm text, and transforms
    every row once. Returns {country: {"vectorizer":.., "matrix":.., "id_to_row": {id: row_idx}}}.
    Fitting per country (not globally) keeps the vocabulary meaningful for open-set countries with
    very different naming conventions, at the cost of one extra pass per country."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    combined = pd.concat([s2_df[["entity_id", "name_norm", "country"]],
                           s3_df[["entity_id", "name_norm", "country"]]], ignore_index=True)
    out = {}
    for country, group in combined.groupby("country", sort=False):
        vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1)
        matrix = vectorizer.fit_transform(group["name_norm"].tolist())
        id_to_row = {eid: i for i, eid in enumerate(group["entity_id"].tolist())}
        out[country] = {"vectorizer": vectorizer, "matrix": matrix, "id_to_row": id_to_row}
    return out


def _tfidf_cosine(tfidf_by_country, country: str, cand_id: str, s1_vec) -> float:
    """Cosine between the already-transformed S1 query vector (`s1_vec`, computed once per S1
    row by the caller) and the candidate's precomputed row in the country's TF-IDF matrix.
    TfidfVectorizer output is L2-normalized by default, so a plain dot product is the cosine.
    This is the "cheap TF-IDF cosine" path and is expected to be slower than Jaccard at full
    test-set scale -- profile on the GPU machine before choosing it for the final run."""
    if s1_vec is None:
        return 0.0
    bucket = tfidf_by_country.get(country)
    if bucket is None:
        return 0.0
    row_idx = bucket["id_to_row"].get(cand_id)
    if row_idx is None:
        return 0.0
    return float(bucket["matrix"][row_idx].multiply(s1_vec).sum())


def candidates_to_id_map(candidates: Dict[str, List[Tuple[str, float, str]]]) -> Dict[str, Set[str]]:
    return {s1: {cid for cid, _score, _src in cands} for s1, cands in candidates.items()}


# ---------------------------------------------------------------------------- recall diagnostics

def report_blocking_recall(candidate_ids: Dict[str, Set[str]], ground_truth: pd.DataFrame,
                            eval_ids: Set[str] = None) -> Dict[str, float]:
    """Fraction of true matches (from ground truth) that survive into the candidate set. This is
    the recall CEILING for everything downstream -- the GBDT/Laya/ensemble stages can never
    recover a true match that blocking dropped. `eval_ids` restricts the report to a subset of S1
    entities (e.g. a held-out validation split), so blocking's own recall isn't measured on rows
    it was implicitly tuned against.

    NOT RUN HERE. Run on the GPU machine and use the printed numbers to decide whether K=30 (or
    the chosen score_method) needs adjusting before spending GBDT/Laya training time downstream.
    """
    total_true = total_recalled = 0
    per_entity_recall = []
    singleton_correct = singleton_total = 0
    for row in ground_truth.itertuples(index=False):
        row_d = row._asdict()
        s1 = row_d["source1_entity_id"]
        if eval_ids is not None and s1 not in eval_ids:
            continue
        raw = row_d["matched_entity_ids"]
        true_ids = set(raw.split(",")) if raw and raw.strip() else set()
        if not true_ids:
            singleton_total += 1
            if not candidate_ids.get(s1):
                singleton_correct += 1
            continue
        cand_ids = candidate_ids.get(s1, set())
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
        val_frac: float, seed: int):
    normalized = common.load_all_normalized(repo_root, split)
    s1_df, s2_df, s3_df = normalized["source1"], normalized["source2"], normalized["source3"]

    print(f"[blocking] {split}: {len(s1_df)} S1 / {len(s2_df)} S2 / {len(s3_df)} S3 rows, "
          f"k={k}, score_method={score_method}")
    candidates = generate_candidates(s1_df, s2_df, s3_df, k=k, score_method=score_method)
    id_map = candidates_to_id_map(candidates)

    out_path = out_path or (common.output_dir(repo_root) / "candidate_pairs.tsv")
    common.write_id_list_tsv(id_map, out_path, "source1_entity_id", "candidate_entity_ids")
    print(f"[blocking] wrote {out_path}")

    if report_recall:
        if split != "train":
            raise SystemExit("--report-recall needs ground truth, which only exists for --split train")
        ground_truth = common.load_split_sources(repo_root, "train")["ground_truth"]
        _, val_ids = common.stratified_split_by_s1(ground_truth, val_frac=val_frac, seed=seed)
        stats = report_blocking_recall(id_map, ground_truth, eval_ids=val_ids)
        print(f"[blocking] recall on held-out val split ({len(val_ids)} S1 entities, "
              f"val_frac={val_frac}, seed={seed}):")
        for key, value in stats.items():
            print(f"    {key}: {value}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    common.add_repo_root_arg(parser)
    parser.add_argument("--split", choices=list(common.SPLITS), required=True)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--score-method", choices=["jaccard", "tfidf"], default="jaccard")
    parser.add_argument("--out", type=str, default=None,
                         help="Output path. Default: output/candidate_pairs.tsv")
    parser.add_argument("--report-recall", action="store_true",
                         help="Also report blocking recall on a held-out val split (--split train only).")
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(args.repo_root, args.split, args.k, args.score_method, args.out,
        args.report_recall, args.val_frac, args.seed)


if __name__ == "__main__":
    main()

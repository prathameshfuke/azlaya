"""Step 3: pairwise feature engineering for every (S1, candidate) pair in candidate_pairs.tsv.

Not executed here. Read back / spot-check on the GPU machine: print `df.head()` and a couple of
known-positive / known-negative training pairs' feature rows before trusting this.

Run (from code/business_entity_resolution/):
    python -m src.features --split train --candidates output/candidate_pairs_train.tsv
    python -m src.features --split test  --candidates output/candidate_pairs.tsv

Writes data_processed/features_{split}.parquet, one row per (source1_entity_id,
candidate_entity_id), keyed by those two columns plus every feature column below.

Requires rapidfuzz, scikit-learn and pyarrow (see requirements.txt).
"""
from __future__ import annotations

import argparse
from typing import Dict, List

import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein, JaroWinkler

from src import common

WORD_NGRAM_RANGE = (1, 2)
CHAR_NGRAM_RANGE = (3, 5)

FEATURE_COLUMNS = [
    "source1_entity_id", "candidate_entity_id",
    "name_levenshtein_sim", "name_jaro_winkler", "name_token_jaccard",
    "name_tfidf_cosine_word", "name_tfidf_cosine_char",
    "address_levenshtein_sim", "address_jaro_winkler", "address_token_jaccard",
    "pin_exact_match", "city_exact_match", "country_match", "name_len_ratio",
    "is_source2", "is_source3", "s1_has_dba", "cand_has_dba",
    "dba_best_jaccard", "name_script_match",
]


# --------------------------------------------------------------------------------- entity lookup

def load_entity_lookup(repo_root, split: str) -> Dict[str, dict]:
    """entity_id -> normalized record dict, merged across source1/2/3 (IDs are globally unique
    by prefix, so a single flat dict is safe)."""
    lookup: Dict[str, dict] = {}
    for key in common.SOURCE_KEYS:
        df = common.load_normalized(repo_root, split, key)
        for row in df.itertuples(index=False):
            row_d = row._asdict()
            lookup[row_d["entity_id"]] = row_d
    return lookup


def build_pairs_frame(candidate_map: Dict[str, set]) -> pd.DataFrame:
    s1_ids, cand_ids = [], []
    for s1, cands in candidate_map.items():
        for c in cands:
            s1_ids.append(s1)
            cand_ids.append(c)
    return pd.DataFrame({"source1_entity_id": s1_ids, "candidate_entity_id": cand_ids})


# ----------------------------------------------------------------------------- string similarity

def token_jaccard(a: str, b: str) -> float:
    ta, tb = set(a.split()), set(b.split())
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    return inter / len(ta | tb) if inter else 0.0


def normalized_levenshtein(a: str, b: str) -> float:
    """1 - (edit distance / max length): 1.0 for identical strings, 0.0 for maximally different."""
    if not a and not b:
        return 1.0
    return Levenshtein.normalized_similarity(a, b)


def jaro_winkler(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    return JaroWinkler.normalized_similarity(a, b)


def _tfidf_cosine_columns(texts_a: List[str], texts_b: List[str], analyzer: str, ngram_range) -> np.ndarray:
    """Vectorized pairwise cosine for two equal-length lists of strings (texts_a[i] vs
    texts_b[i]): fit one vectorizer on the pooled vocabulary, transform both sides, then take the
    row-wise dot product. TfidfVectorizer output is L2-normalized, so the dot product IS the
    cosine similarity -- this avoids an O(n^2) similarity matrix entirely."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    non_empty = [t for t in texts_a + texts_b if t]
    if not non_empty:
        return np.zeros(len(texts_a))
    vectorizer = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram_range, min_df=1)
    vectorizer.fit(non_empty)
    A = vectorizer.transform(texts_a)
    B = vectorizer.transform(texts_b)
    return np.asarray(A.multiply(B).sum(axis=1)).ravel()


# --------------------------------------------------------------------------------------- driver

def compute_features(pairs_df: pd.DataFrame, lookup: Dict[str, dict]) -> pd.DataFrame:
    rows = []
    missing = 0
    for row in pairs_df.itertuples(index=False):
        s1_id, cand_id = row.source1_entity_id, row.candidate_entity_id
        s1 = lookup.get(s1_id)
        cand = lookup.get(cand_id)
        if s1 is None or cand is None:
            missing += 1
            continue
        rows.append((s1_id, cand_id, s1, cand))

    if missing:
        print(f"[features] WARNING: {missing} candidate pairs referenced an entity_id not found "
              f"in the normalized source files -- skipped. This usually means candidate_pairs.tsv "
              f"and the --split normalized files don't match; double-check before trusting output.")

    name_a = [r[2]["name_norm"] for r in rows]
    name_b = [r[3]["name_norm"] for r in rows]

    word_cos = _tfidf_cosine_columns(name_a, name_b, "word", WORD_NGRAM_RANGE)
    char_cos = _tfidf_cosine_columns(name_a, name_b, "char_wb", CHAR_NGRAM_RANGE)

    records = []
    for i, (s1_id, cand_id, s1, cand) in enumerate(rows):
        n_a, n_b = s1["name_norm"], cand["name_norm"]
        ad_a, ad_b = s1["address_norm"], cand["address_norm"]

        legal_a, trade_a = s1["legal_name_norm"], s1["trade_name_norm"]
        legal_b, trade_b = cand["legal_name_norm"], cand["trade_name_norm"]
        dba_variants = [
            token_jaccard(legal_a, legal_b), token_jaccard(legal_a, trade_b),
            token_jaccard(trade_a, legal_b), token_jaccard(trade_a, trade_b),
        ]

        pin_a, pin_b = s1["pin_code"], cand["pin_code"]
        city_a, city_b = s1["city_guess"], cand["city_guess"]

        records.append({
            "source1_entity_id": s1_id,
            "candidate_entity_id": cand_id,
            "name_levenshtein_sim": normalized_levenshtein(n_a, n_b),
            "name_jaro_winkler": jaro_winkler(n_a, n_b),
            "name_token_jaccard": token_jaccard(n_a, n_b),
            "name_tfidf_cosine_word": float(word_cos[i]),
            "name_tfidf_cosine_char": float(char_cos[i]),
            "address_levenshtein_sim": normalized_levenshtein(ad_a, ad_b),
            "address_jaro_winkler": jaro_winkler(ad_a, ad_b),
            "address_token_jaccard": token_jaccard(ad_a, ad_b),
            "pin_exact_match": bool(pin_a and pin_b and pin_a == pin_b),
            "city_exact_match": bool(city_a and city_b and city_a == city_b),
            "country_match": bool(s1["country"].strip().lower() == cand["country"].strip().lower()),
            "name_len_ratio": (min(len(n_a), len(n_b)) / max(len(n_a), len(n_b))) if max(len(n_a), len(n_b)) else 0.0,
            "is_source2": cand_id.startswith("S2-"),
            "is_source3": cand_id.startswith("S3-"),
            "s1_has_dba": s1["has_dba"] == "True",
            "cand_has_dba": cand["has_dba"] == "True",
            "dba_best_jaccard": max(dba_variants),
            "name_script_match": s1["name_script"] == cand["name_script"],
        })
    # Explicit `columns=` so an empty `records` (e.g. zero candidate pairs survived blocking)
    # still returns a frame with the right schema instead of a columnless one that breaks
    # every downstream `feature_columns(df)` / column-selection call.
    return pd.DataFrame.from_records(records, columns=FEATURE_COLUMNS)


def run(repo_root, split: str, candidates_path, out_path):
    candidates_path = candidates_path or (common.output_dir(repo_root) / "candidate_pairs.tsv")
    candidate_map = common.read_id_list_tsv(candidates_path)
    pairs_df = build_pairs_frame(candidate_map)
    print(f"[features] {split}: {len(candidate_map)} S1 entities, {len(pairs_df)} candidate pairs")

    lookup = load_entity_lookup(repo_root, split)
    features_df = compute_features(pairs_df, lookup)

    out_path = out_path or (common.data_processed_dir(repo_root) / f"features_{split}.parquet")
    features_df.to_parquet(out_path, index=False)
    print(f"[features] wrote {out_path} ({len(features_df)} rows, {features_df.shape[1]} columns)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    common.add_repo_root_arg(parser)
    parser.add_argument("--split", choices=list(common.SPLITS), required=True)
    parser.add_argument("--candidates", type=str, default=None,
                         help="Path to a candidate_pairs.tsv. Default: output/candidate_pairs.tsv")
    parser.add_argument("--out", type=str, default=None,
                         help="Output parquet path. Default: data_processed/features_{split}.parquet")
    args = parser.parse_args()
    run(args.repo_root, args.split, args.candidates, args.out)


if __name__ == "__main__":
    main()

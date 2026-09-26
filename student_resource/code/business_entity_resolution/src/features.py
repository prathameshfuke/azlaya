"""Step 3: pairwise feature engineering for every (S1, candidate) pair in candidate_pairs.tsv.

Vectorized and chunked so it scales to millions of pairs: entity records live in one indexed
DataFrame (not a dict per record), string metrics use rapidfuzz.process.cpdist (C-level,
multi-threaded), and TF-IDF vectorizers are fit ONCE on the unique names involved, then pairs are
scored by row-wise dot products on index arrays.

Run (from code/business_entity_resolution/):
    python -m src.features --split train --candidates ../../data_processed/candidate_pairs_train.tsv

Writes data_processed/features_{split}.parquet, one row per (source1_entity_id,
candidate_entity_id).
"""
from __future__ import annotations

import argparse
from typing import Dict, List

import numpy as np
import pandas as pd
from rapidfuzz import process
from rapidfuzz.distance import JaroWinkler, Levenshtein
from tqdm.auto import tqdm

from src import common

WORD_NGRAM_RANGE = (1, 2)
CHAR_NGRAM_RANGE = (3, 4)
CHUNK_SIZE = 500_000

ENTITY_COLS = [
    "entity_id", "business_name", "business_address", "country",
    "name_norm", "legal_name_norm", "trade_name_norm", "has_dba",
    "address_norm", "pin_code", "city_guess", "name_script",
]

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

def load_entity_table(repo_root, split: str, ids=None) -> pd.DataFrame:
    """Normalized S1/S2/S3 records in one frame indexed by entity_id (IDs are globally unique by
    prefix). Pass `ids` to keep only the records a step references -- the full train split is
    ~12.5M records, far more than any single step needs in memory."""
    ids = set(ids) if ids is not None else None
    frames = [common.load_normalized(repo_root, split, key, columns=ENTITY_COLS, ids=ids)
              for key in common.SOURCE_KEYS]
    return pd.concat(frames, ignore_index=True).set_index("entity_id")


def pair_entity_ids(pairs_df: pd.DataFrame) -> set:
    return set(pairs_df["source1_entity_id"]).union(pairs_df["candidate_entity_id"])


def load_entity_lookup(repo_root, split: str, ids) -> Dict[str, dict]:
    """entity_id -> record dict for just `ids` (the Laya fine-tuning examples' records)."""
    return load_entity_table(repo_root, split, ids=ids).to_dict("index")


def build_pairs_frame(candidate_map) -> pd.DataFrame:
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


def _jaccard_many(xs: List[str], ys: List[str]) -> np.ndarray:
    return np.fromiter((token_jaccard(a, b) for a, b in zip(xs, ys)), dtype=np.float32, count=len(xs))


def _pairwise(xs: List[str], ys: List[str], scorer) -> np.ndarray:
    return process.cpdist(xs, ys, scorer=scorer, workers=-1, dtype=np.float32)


def _fit_tfidf(texts: List[str], analyzer: str, ngram_range):
    from sklearn.feature_extraction.text import TfidfVectorizer

    non_empty = [t for t in texts if t]
    if not non_empty:
        return None
    vectorizer = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram_range, min_df=1, dtype=np.float32)
    vectorizer.fit(non_empty)
    return vectorizer.transform(texts).tocsr()


def _rowdot(matrix, ia: np.ndarray, ib: np.ndarray) -> np.ndarray:
    """Cosine between rows ia[i] and ib[i] of an L2-normalized TF-IDF matrix."""
    if matrix is None:
        return np.zeros(len(ia), dtype=np.float32)
    return np.asarray(matrix[ia].multiply(matrix[ib]).sum(axis=1), dtype=np.float32).ravel()


# --------------------------------------------------------------------------------------- driver

def compute_features(pairs_df: pd.DataFrame, table: pd.DataFrame, chunk_size: int = CHUNK_SIZE) -> pd.DataFrame:
    known = pairs_df["source1_entity_id"].isin(table.index) & pairs_df["candidate_entity_id"].isin(table.index)
    missing = int((~known).sum())
    if missing:
        print(f"[features] WARNING: {missing} candidate pairs referenced an entity_id not found in the "
              f"normalized source files -- skipped. Check that candidate_pairs.tsv matches --split.")
    pairs = pairs_df[known].reset_index(drop=True)
    if pairs.empty:
        return pd.DataFrame(columns=FEATURE_COLUMNS)

    involved = pd.unique(np.concatenate([pairs["source1_entity_id"].to_numpy(),
                                         pairs["candidate_entity_id"].to_numpy()]))
    row_of = pd.Series(np.arange(len(involved)), index=involved)
    names = table.loc[involved, "name_norm"].tolist()
    print(f"[features] fitting TF-IDF on {len(names)} unique names")
    word_m = _fit_tfidf(names, "word", WORD_NGRAM_RANGE)
    char_m = _fit_tfidf(names, "char_wb", CHAR_NGRAM_RANGE)

    out = []
    for start in tqdm(range(0, len(pairs), chunk_size), desc="features", unit="chunk"):
        chunk = pairs.iloc[start:start + chunk_size]
        s1_ids = chunk["source1_entity_id"].to_numpy()
        cand_ids = chunk["candidate_entity_id"].to_numpy()
        a = table.loc[s1_ids]
        b = table.loc[cand_ids]
        ia = row_of.loc[s1_ids].to_numpy()
        ib = row_of.loc[cand_ids].to_numpy()

        na, nb = a["name_norm"].tolist(), b["name_norm"].tolist()
        ada, adb = a["address_norm"].tolist(), b["address_norm"].tolist()
        la, lb = a["name_norm"].str.len().to_numpy(), b["name_norm"].str.len().to_numpy()
        longest = np.maximum(la, lb)
        pa, pb = a["pin_code"].to_numpy(), b["pin_code"].to_numpy()
        ca, cb = a["city_guess"].to_numpy(), b["city_guess"].to_numpy()

        dba_best = np.maximum.reduce([
            _jaccard_many(a["legal_name_norm"].tolist(), b["legal_name_norm"].tolist()),
            _jaccard_many(a["legal_name_norm"].tolist(), b["trade_name_norm"].tolist()),
            _jaccard_many(a["trade_name_norm"].tolist(), b["legal_name_norm"].tolist()),
            _jaccard_many(a["trade_name_norm"].tolist(), b["trade_name_norm"].tolist()),
        ])

        out.append(pd.DataFrame({
            "source1_entity_id": s1_ids,
            "candidate_entity_id": cand_ids,
            "name_levenshtein_sim": _pairwise(na, nb, Levenshtein.normalized_similarity),
            "name_jaro_winkler": _pairwise(na, nb, JaroWinkler.normalized_similarity),
            "name_token_jaccard": _jaccard_many(na, nb),
            "name_tfidf_cosine_word": _rowdot(word_m, ia, ib),
            "name_tfidf_cosine_char": _rowdot(char_m, ia, ib),
            "address_levenshtein_sim": _pairwise(ada, adb, Levenshtein.normalized_similarity),
            "address_jaro_winkler": _pairwise(ada, adb, JaroWinkler.normalized_similarity),
            "address_token_jaccard": _jaccard_many(ada, adb),
            "pin_exact_match": (pa != "") & (pa == pb),
            "city_exact_match": (ca != "") & (ca == cb),
            "country_match": (a["country"].str.strip().str.lower().to_numpy()
                              == b["country"].str.strip().str.lower().to_numpy()),
            "name_len_ratio": np.where(longest > 0, np.minimum(la, lb) / np.maximum(longest, 1), 0.0).astype(np.float32),
            "is_source2": chunk["candidate_entity_id"].str.startswith("S2-").to_numpy(),
            "is_source3": chunk["candidate_entity_id"].str.startswith("S3-").to_numpy(),
            "s1_has_dba": a["has_dba"].to_numpy() == "True",
            "cand_has_dba": b["has_dba"].to_numpy() == "True",
            "dba_best_jaccard": dba_best,
            "name_script_match": a["name_script"].to_numpy() == b["name_script"].to_numpy(),
        }))
    return pd.concat(out, ignore_index=True)[FEATURE_COLUMNS]


def run(repo_root, split: str, candidates_path, out_path):
    candidates_path = candidates_path or (common.output_dir(repo_root) / "candidate_pairs.tsv")
    candidate_map = common.read_id_list_tsv(common.require(candidates_path, "blocking.py (notebook Section 2)"))
    pairs_df = build_pairs_frame(candidate_map)
    print(f"[features] {split}: {len(candidate_map)} S1 entities, {len(pairs_df)} candidate pairs")

    table = load_entity_table(repo_root, split, ids=pair_entity_ids(pairs_df))
    features_df = compute_features(pairs_df, table)

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

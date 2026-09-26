"""Shared paths, IO helpers, and scoring utilities used by every script in this pipeline.

NOTE ON EXECUTION: authored on a machine with no GPU. The non-GPU stages (normalize/blocking/
features/train_gbdt) have since been smoke-tested end-to-end against a small synthetic dataset
shaped like the real challenge files, which is how the Unicode combining-mark bug in
`strip_punctuation` below was actually caught -- static reading alone missed it. Laya fine-tuning
and inference (laya_finetune.py's --stage train, ensemble.py/predict.py's Laya scoring) still have
not been run anywhere; those need a real GPU and the actual competition data. Keep spot-checking
outputs on the GPU machine (print a normalized sample, run report_blocking_recall) before trusting
numbers on the real dataset -- a 20-row synthetic smoke test proves the code paths run and do
something sensible, not that every heuristic is well-tuned at full scale.
"""
from __future__ import annotations

import argparse
import random
import re
import unicodedata
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

import pandas as pd

# --------------------------------------------------------------------------------------- paths
#
# src/common.py -> src -> business_entity_resolution -> code -> student_resource
# PROJECT_ROOT is the student_resource/ directory: it holds dataset/, and is where output/,
# data_processed/ and models/ get created. This matches where utils/validate_submission.py
# expects to be run from (its own docstring: "Run this ... from the student_resource/ directory").
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parents[2]

SPLITS = ("train", "test")
SOURCE_KEYS = ("source1", "source2", "source3")

SOURCE_FILENAMES = {
    "train": {"source1": "train_source1.tsv", "source2": "train_source2.tsv",
              "source3": "train_source3.tsv", "ground_truth": "train_ground_truth.tsv"},
    "test": {"source1": "test_source1.tsv", "source2": "test_source2.tsv",
             "source3": "test_source3.tsv"},
}

def add_repo_root_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--repo-root", type=Path, default=PROJECT_ROOT,
        help="Directory containing dataset/ (and where output/, data_processed/, models/ are "
             "created). Default: auto-detected student_resource/ directory.",
    )


def require(path, produced_by: str) -> Path:
    """Fail with a readable message when a step's input is missing, naming the earlier step that
    writes it. Without this, a step whose predecessor crashed (e.g. out of memory) fails with a
    bare FileNotFoundError that points at the wrong step."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. It is written by {produced_by} -- run that step first and check "
            f"its output for errors (a crashed earlier step is the usual cause).")
    return path


def dataset_dir(repo_root: Path, split: str) -> Path:
    assert split in SPLITS, split
    return Path(repo_root) / "dataset" / split


def data_processed_dir(repo_root: Path) -> Path:
    p = Path(repo_root) / "data_processed"
    p.mkdir(parents=True, exist_ok=True)
    return p


def output_dir(repo_root: Path) -> Path:
    p = Path(repo_root) / "output"
    p.mkdir(parents=True, exist_ok=True)
    return p


def models_dir(repo_root: Path) -> Path:
    p = Path(repo_root) / "models"
    p.mkdir(parents=True, exist_ok=True)
    return p


# --------------------------------------------------------------------------------------- raw IO

_READ_CHUNK_ROWS = 200_000


def _read_tsv_with_progress(path, columns=None, ids=None) -> pd.DataFrame:
    """Every column as `str`, no NaN coercion (an empty field stays ""), read in chunks so large
    files show a live row count instead of a silent multi-minute pause. `columns` limits which
    columns are kept and `ids` which entity_id rows are kept; at ~12.5M train records, loading
    only what a step needs is the difference between fitting in memory and not."""
    from tqdm.auto import tqdm

    path = Path(path)
    chunks = []
    with tqdm(desc=f"read {path.name}", unit="row", unit_scale=True, mininterval=1.0) as bar:
        for chunk in pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_filter=False,
                                 usecols=columns, chunksize=_READ_CHUNK_ROWS):
            bar.update(len(chunk))
            if ids is not None:
                chunk = chunk[chunk["entity_id"].isin(ids)]
            chunks.append(chunk)
    if chunks:
        df = pd.concat(chunks, ignore_index=True)
    else:
        df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, usecols=columns, nrows=0)
    df.columns = [c.strip() for c in df.columns]
    return df


def read_source_tsv(path) -> pd.DataFrame:
    """Read one source/ground-truth TSV. The "null"/"NaN"/"N/A" *strings* some address fields
    contain are left untouched here (they are real content; normalize.py strips them)."""
    return _read_tsv_with_progress(path)


def load_split_sources(repo_root: Path, split: str) -> Dict[str, pd.DataFrame]:
    d = dataset_dir(repo_root, split)
    names = SOURCE_FILENAMES[split]
    out = {key: read_source_tsv(require(d / names[key], "the dataset setup (notebook Section 0.5)"))
           for key in SOURCE_KEYS}
    if split == "train":
        out["ground_truth"] = read_source_tsv(d / names["ground_truth"])
    return out


# --------------------------------------------------------------------------------- processed IO

def normalized_path(repo_root: Path, split: str, key: str) -> Path:
    return data_processed_dir(repo_root) / f"{split}_{key}_normalized.tsv"


def load_normalized(repo_root: Path, split: str, key: str, columns=None, ids=None) -> pd.DataFrame:
    return _read_tsv_with_progress(require(normalized_path(repo_root, split, key),
                                           f"normalize.py (notebook Section 1) for --split {split}"),
                                   columns=columns, ids=ids)


def sampled_ground_truth(repo_root: Path, candidates_path=None) -> pd.DataFrame:
    """Train ground truth restricted to the S1 entities blocking sampled (every sampled S1 has a
    row in candidate_pairs_train.tsv, even with no candidates). Train, Laya, ensemble and the
    recall report all use this, so "val" and every metric refer to the same sampled entities --
    an unsampled entity would otherwise count as an empty prediction and skew F_0.5."""
    candidates_path = candidates_path or (data_processed_dir(repo_root) / "candidate_pairs_train.tsv")
    sampled = set(pd.read_csv(require(candidates_path, "blocking.py --split train (notebook Section 2)"),
                              sep="\t", dtype=str, usecols=["source1_entity_id"], keep_default_na=False)
                  ["source1_entity_id"])
    gt = read_source_tsv(require(dataset_dir(repo_root, "train") / SOURCE_FILENAMES["train"]["ground_truth"],
                                 "the dataset setup (notebook Section 0.5)"))
    return gt[gt["source1_entity_id"].isin(sampled)].reset_index(drop=True)


def write_normalized(df: pd.DataFrame, repo_root: Path, split: str, key: str) -> Path:
    path = normalized_path(repo_root, split, key)
    df.to_csv(path, sep="\t", index=False)
    return path


# ------------------------------------------------------- id-list TSV (the submission-file format)

def write_id_list_tsv(mapping: Dict[str, Iterable[str]], path, id_col: str, list_col: str) -> None:
    """Write the exact (source1_entity_id, comma-joined candidate/match IDs) TSV format that
    utils/validate_submission.py checks: one row per key of `mapping`, no quoting, IDs
    deduplicated, empty string (not "None"/"nan") for an entity with no matches."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(f"{id_col}\t{list_col}\n")
        for s1_id, ids in mapping.items():
            id_list = sorted(set(ids))
            f.write(f"{s1_id}\t{','.join(id_list)}\n")


def read_id_list_tsv(path) -> Dict[str, Set[str]]:
    path = Path(path)
    out: Dict[str, Set[str]] = {}
    with open(path, encoding="utf-8") as f:
        next(f, None)  # header
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            s1, _, rest = line.partition("\t")
            out[s1] = set(rest.split(",")) if rest.strip() else set()
    return out




# --------------------------------------------------------------------- stratified S1-level split

def stratified_split_by_s1(ground_truth: pd.DataFrame, val_frac: float = 0.2, seed: int = 42,
                            ) -> Tuple[Set[str], Set[str]]:
    """Split S1 entity IDs (whole entities, not candidate-pair rows) into train/val sets,
    stratified by singleton vs. has-a-match.

    Splitting by entity rather than by row is the point: every (S1, candidate) pair for a given
    S1 entity has to land on the same side of the split, otherwise the model implicitly sees part
    of a validation entity's own pairs during training (leakage), which inflates validation
    F_0.5. Stratifying by singleton/non-singleton keeps both classes represented in both splits
    even though most S1 entities in a challenge like this tend to be non-singletons or vice versa.
    """
    ids = ground_truth["source1_entity_id"].tolist()
    matched = ground_truth["matched_entity_ids"].fillna("")
    is_singleton = matched.str.strip().eq("")
    pos_ids = sorted(i for i, s in zip(ids, is_singleton) if not s)
    neg_ids = sorted(i for i, s in zip(ids, is_singleton) if s)

    rng = random.Random(seed)

    def _split(group: List[str]) -> Tuple[Set[str], Set[str]]:
        group = list(group)
        rng.shuffle(group)
        n_val = int(round(len(group) * val_frac))
        return set(group[n_val:]), set(group[:n_val])

    pos_train, pos_val = _split(pos_ids)
    neg_train, neg_val = _split(neg_ids)
    return pos_train | neg_train, pos_val | neg_val


# ---------------------------------------------------------------------------- F_0.5 (macro) score

def f_beta_per_entity(pred: Dict[str, Set[str]], truth: Dict[str, Set[str]], beta: float = 0.5,
                       ) -> Dict[str, float]:
    """Per-S1-entity F_beta, computed exactly as the challenge scores it: a correctly-predicted
    empty list scores 1.0, an incorrectly non-empty prediction on a true singleton scores 0.0.
    Iterates over `truth`'s keys, so every S1 entity in the ground truth / eval set gets a score
    even when `pred` has no row for it (treated as an empty prediction)."""
    beta2 = beta * beta
    scores: Dict[str, float] = {}
    for s1, true_ids in truth.items():
        pred_ids = pred.get(s1, set())
        if not true_ids and not pred_ids:
            scores[s1] = 1.0
            continue
        if not pred_ids or not true_ids:
            scores[s1] = 0.0
            continue
        tp = len(pred_ids & true_ids)
        if tp == 0:
            scores[s1] = 0.0
            continue
        precision = tp / len(pred_ids)
        recall = tp / len(true_ids)
        denom = beta2 * precision + recall
        scores[s1] = 0.0 if denom == 0 else (1 + beta2) * precision * recall / denom
    return scores


def macro_f_beta(pred: Dict[str, Set[str]], truth: Dict[str, Set[str]], beta: float = 0.5) -> float:
    scores = f_beta_per_entity(pred, truth, beta)
    return sum(scores.values()) / len(scores) if scores else 0.0


def ground_truth_map(ground_truth: pd.DataFrame) -> Dict[str, Set[str]]:
    """{source1_entity_id: set(matched_entity_ids)} from the ground-truth TSV, empty set for a
    singleton. Shared by train_gbdt.py, laya_finetune.py and ensemble.py so positive/negative
    labeling is defined identically everywhere it's used."""
    truth: Dict[str, Set[str]] = {}
    for row in ground_truth.itertuples(index=False):
        row_d = row._asdict()
        raw = row_d["matched_entity_ids"]
        truth[row_d["source1_entity_id"]] = set(raw.split(",")) if raw and raw.strip() else set()
    return truth


def precision_recall_macro(pred: Dict[str, Set[str]], truth: Dict[str, Set[str]]) -> Tuple[float, float]:
    """Micro-averaged precision/recall over all (S1, candidate) pairs, as a companion to the
    macro F_0.5 above (which the challenge actually scores on) — useful for error analysis."""
    tp = fp = fn = 0
    for s1, true_ids in truth.items():
        pred_ids = pred.get(s1, set())
        tp += len(pred_ids & true_ids)
        fp += len(pred_ids - true_ids)
        fn += len(true_ids - pred_ids)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return precision, recall


# --------------------------------------------------------------------------------- script detection

_DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")
_TAMIL_RE = re.compile(r"[஀-௿]")
_LATIN_RE = re.compile(r"[A-Za-z]")


def detect_script(text: str) -> str:
    """Cheap, dependency-free script classifier over Unicode ranges. Returns one of
    'latin', 'devanagari', 'tamil', 'mixed' (no script has >=60% of the letters seen), or
    'other' (no letters from any tracked script — e.g. pure digits/punctuation, or a script
    this challenge doesn't call out such as Chinese/Arabic, which is intentionally bucketed
    as 'other' rather than mis-labelled 'latin')."""
    if not text:
        return "other"
    counts = {
        "devanagari": len(_DEVANAGARI_RE.findall(text)),
        "tamil": len(_TAMIL_RE.findall(text)),
        "latin": len(_LATIN_RE.findall(text)),
    }
    total = sum(counts.values())
    if total == 0:
        return "other"
    dominant, n = max(counts.items(), key=lambda kv: kv[1])
    return dominant if n / total >= 0.6 else "mixed"


# --------------------------------------------------------------------------------- misc text util

_WS_RE = re.compile(r"\s+")


def collapse_ws(text: str) -> str:
    return _WS_RE.sub(" ", text).strip()


# Unicode general-category prefixes to KEEP when stripping punctuation: L* (letters), M* (marks --
# combining vowel signs and the virama/halant that Devanagari, Tamil, and most other Indic/complex
# scripts build words out of), N* (digits).
_KEEP_CATEGORY_PREFIXES = ("L", "M", "N")


def strip_punctuation(text: str, keep_chars: str = "") -> str:
    """Replace every character that is not a letter/mark/digit, whitespace, or in `keep_chars`
    with a single space.

    This is deliberately NOT `re.sub(r"[^\\w\\s]", " ", text)`: Python's `\\w` matches Unicode
    letters and digits but NOT combining marks (category Mn/Mc) -- and Devanagari/Tamil vowel
    signs and the virama/halter are combining marks, not standalone letters. A `\\w`-based strip
    silently deletes them, corrupting every word that uses one (i.e. most Devanagari/Tamil text)
    into fragments -- e.g. "शर्मा" (Sharma) becomes "शर म" (this was caught by actually running
    normalize.py against a synthetic Devanagari-name row, not by reading the regex).
    """
    return "".join(
        ch if ch.isspace() or ch in keep_chars or unicodedata.category(ch).startswith(_KEEP_CATEGORY_PREFIXES)
        else " "
        for ch in text
    )

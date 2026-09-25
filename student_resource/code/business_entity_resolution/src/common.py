"""Shared paths, IO helpers, and scoring utilities used by every script in this pipeline.

NOTE ON EXECUTION: this whole package was authored on a machine with no GPU and nothing here
has been run or imported. It is written to be correct by inspection; the first real run of any
of it should happen on the GPU machine (Colab/AWS), per README.md. Spot-check outputs there
(e.g. print a normalized sample, run report_blocking_recall) before trusting the numbers.
"""
from __future__ import annotations

import argparse
import random
import re
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

def read_source_tsv(path) -> pd.DataFrame:
    """Read one source/ground-truth TSV. Every column is read as `str` with no NaN coercion:
    a literal empty field stays "" rather than becoming float NaN, and the "null"/"NaN"/"N/A"
    *strings* some address fields contain are left untouched here (they are real content, and
    normalize.py is what's responsible for stripping them, not the reader)."""
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_filter=False)
    df.columns = [c.strip() for c in df.columns]
    return df


def load_split_sources(repo_root: Path, split: str) -> Dict[str, pd.DataFrame]:
    d = dataset_dir(repo_root, split)
    names = SOURCE_FILENAMES[split]
    out = {key: read_source_tsv(d / names[key]) for key in SOURCE_KEYS}
    if split == "train":
        out["ground_truth"] = read_source_tsv(d / names["ground_truth"])
    return out


# --------------------------------------------------------------------------------- processed IO

def normalized_path(repo_root: Path, split: str, key: str) -> Path:
    return data_processed_dir(repo_root) / f"{split}_{key}_normalized.tsv"


def load_normalized(repo_root: Path, split: str, key: str) -> pd.DataFrame:
    path = normalized_path(repo_root, split, key)
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_filter=False)


def write_normalized(df: pd.DataFrame, repo_root: Path, split: str, key: str) -> Path:
    path = normalized_path(repo_root, split, key)
    df.to_csv(path, sep="\t", index=False)
    return path


def load_all_normalized(repo_root: Path, split: str) -> Dict[str, pd.DataFrame]:
    out = {key: load_normalized(repo_root, split, key) for key in SOURCE_KEYS}
    return out


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

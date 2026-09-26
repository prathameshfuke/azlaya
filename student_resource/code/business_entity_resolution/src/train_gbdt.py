"""Step 4: GBDT (LightGBM) binary matcher over the feature table from features.py.

Labels: positive = candidate is in the S1 entity's matched_entity_ids (ground truth); negative =
candidate survived blocking but is NOT a true match (a "hard negative" -- it looks plausible
enough to have passed blocking, unlike a random pair, which is what makes it a useful negative).

Split: stratified by whole S1 entity (src/common.py:stratified_split_by_s1), not by row, so no
candidate pair from a validation entity is seen during training.

Not executed here. This script is written to run as-is on the GPU machine -- it does not need a
GPU (LightGBM here runs on CPU by default), but per the project constraints nothing is run in this
authoring environment. Read it back / spot-check the printed metrics format on the GPU machine.

Run (from code/business_entity_resolution/):
    python -m src.train_gbdt --features data_processed/features_train.parquet

Writes:
    models/gbdt_model.txt                      (LightGBM Booster, text format)
    models/gbdt_feature_columns.json            (the exact feature column order/list used)
    data_processed/gbdt_val_report.json         (precision/recall/macro F_0.5 + threshold sweep)
"""
from __future__ import annotations

import argparse
import json
from typing import Dict, List, Set

import numpy as np
import pandas as pd

from src import common

ID_COLS = ("source1_entity_id", "candidate_entity_id")
DEFAULT_THRESHOLD = 0.5


def feature_columns(features_df: pd.DataFrame) -> List[str]:
    return [c for c in features_df.columns if c not in ID_COLS]


def _bool_and_str_to_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """LightGBM wants numeric input; the feature table has a few boolean columns (pin_exact_match,
    is_source2, dba flags, ...) that pandas/pyarrow round-trip as Python bool -- cast everything
    to float32 uniformly rather than special-casing each column name."""
    out = df.copy()
    for c in cols:
        out[c] = out[c].astype("float32")
    return out


def label_pairs(features_df: pd.DataFrame, ground_truth: pd.DataFrame) -> pd.Series:
    truth = common.ground_truth_map(ground_truth)
    positive_keys = {f"{s1}|{c}" for s1, ids in truth.items() for c in ids}
    keys = features_df["source1_entity_id"] + "|" + features_df["candidate_entity_id"]
    return keys.isin(positive_keys)


def build_dataset(repo_root, features_path):
    features_path = features_path or (common.data_processed_dir(repo_root) / "features_train.parquet")
    features_df = pd.read_parquet(features_path)
    ground_truth = common.load_split_sources(repo_root, "train")["ground_truth"]
    labels = label_pairs(features_df, ground_truth)
    return features_df, labels, ground_truth


def score_with_gbdt(model, features_df: pd.DataFrame, cols: List[str] = None) -> np.ndarray:
    """Reusable by ensemble.py / predict.py: apply a trained Booster to any features_df with (at
    least) the same feature columns it was trained on."""
    cols = cols or feature_columns(features_df)
    X = _bool_and_str_to_numeric(features_df, cols)[cols]
    return model.predict(X)


def matches_from_scores(features_df: pd.DataFrame, scores: np.ndarray, threshold: float,
                         candidate_ids: Dict[str, Set[str]] = None) -> Dict[str, Set[str]]:
    """Aggregate per-pair scores into per-S1 predicted match sets at a probability threshold.
    `candidate_ids`, when given, is the full candidate map (so S1 entities with candidates but
    zero rows above threshold still get an explicit empty entry instead of being absent)."""
    out: Dict[str, Set[str]] = {s1: set() for s1 in candidate_ids} if candidate_ids else {}
    for s1, cand, score in zip(features_df["source1_entity_id"], features_df["candidate_entity_id"], scores):
        if score >= threshold:
            out.setdefault(s1, set()).add(cand)
    return out


def evaluate(features_df: pd.DataFrame, scores: np.ndarray, ground_truth: pd.DataFrame,
             val_ids: Set[str], thresholds=None) -> dict:
    """Precision/recall/macro-F0.5 at DEFAULT_THRESHOLD, plus a small threshold sweep (for
    diagnostics only -- final threshold selection is ensemble.py's job, on the ensemble's own
    output, not the bare GBDT's)."""
    thresholds = thresholds if thresholds is not None else [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    val_mask = features_df["source1_entity_id"].isin(val_ids)
    val_df = features_df[val_mask].reset_index(drop=True)
    val_scores = np.asarray(scores)[val_mask.to_numpy()]

    truth = common.ground_truth_map(ground_truth)
    val_truth = {s1: ids for s1, ids in truth.items() if s1 in val_ids}

    sweep = {}
    for t in thresholds:
        pred = matches_from_scores(val_df, val_scores, t, candidate_ids={s1: set() for s1 in val_ids})
        precision, recall = common.precision_recall_macro(pred, val_truth)
        macro_f05 = common.macro_f_beta(pred, val_truth, beta=0.5)
        sweep[str(t)] = {"precision": precision, "recall": recall, "macro_f0_5": macro_f05}

    return {"n_val_entities": len(val_ids), "n_val_pairs": int(val_mask.sum()), "threshold_sweep": sweep}


def train(repo_root, features_path, val_frac: float, seed: int, num_boost_round: int,
          early_stopping_rounds: int, extra_params: dict = None):
    import lightgbm as lgb

    features_df, labels, ground_truth = build_dataset(repo_root, features_path)
    cols = feature_columns(features_df)
    X_all = _bool_and_str_to_numeric(features_df, cols)[cols]
    y_all = labels.astype(int)

    train_ids, val_ids = common.stratified_split_by_s1(ground_truth, val_frac=val_frac, seed=seed)
    train_mask = features_df["source1_entity_id"].isin(train_ids).to_numpy()
    val_mask = features_df["source1_entity_id"].isin(val_ids).to_numpy()

    print(f"[train_gbdt] {train_mask.sum()} train pairs / {val_mask.sum()} val pairs "
          f"({y_all[train_mask].sum()} / {y_all[val_mask].sum()} positive)")

    train_set = lgb.Dataset(X_all[train_mask], label=y_all[train_mask], feature_name=cols)
    val_set = lgb.Dataset(X_all[val_mask], label=y_all[val_mask], feature_name=cols, reference=train_set)

    params = {
        "objective": "binary",
        "metric": ["binary_logloss", "auc"],
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 20,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "is_unbalance": True,  # true matches are a small minority of blocking candidates
        "verbose": -1,
        "seed": seed,
    }
    if extra_params:
        params.update(extra_params)

    from tqdm.auto import tqdm

    bar = tqdm(total=num_boost_round, desc="lightgbm", unit="iter")

    def _progress(env):
        bar.update(1)

    model = lgb.train(
        params, train_set, num_boost_round=num_boost_round,
        valid_sets=[val_set], valid_names=["val"],
        callbacks=[lgb.early_stopping(early_stopping_rounds), lgb.log_evaluation(100), _progress],
    )
    bar.close()

    val_scores_full = np.zeros(len(features_df), dtype="float32")
    val_scores_full[val_mask] = model.predict(X_all[val_mask], num_iteration=model.best_iteration)
    report = evaluate(features_df, val_scores_full, ground_truth, val_ids)

    models_path = common.models_dir(repo_root)
    model.save_model(str(models_path / "gbdt_model.txt"), num_iteration=model.best_iteration)
    with open(models_path / "gbdt_feature_columns.json", "w") as f:
        json.dump(cols, f, indent=2)
    report_path = common.data_processed_dir(repo_root) / "gbdt_val_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"[train_gbdt] best_iteration={model.best_iteration}")
    print(f"[train_gbdt] wrote {models_path / 'gbdt_model.txt'}")
    print(f"[train_gbdt] val report -> {report_path}")
    print(json.dumps(report, indent=2))
    return model, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    common.add_repo_root_arg(parser)
    parser.add_argument("--features", type=str, default=None,
                         help="Path to features_train.parquet. Default: data_processed/features_train.parquet")
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-boost-round", type=int, default=2000)
    parser.add_argument("--early-stopping-rounds", type=int, default=50)
    args = parser.parse_args()
    train(args.repo_root, args.features, args.val_frac, args.seed,
          args.num_boost_round, args.early_stopping_rounds)


if __name__ == "__main__":
    main()

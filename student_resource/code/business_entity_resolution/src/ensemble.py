"""Step 6: ensemble -- score candidates with fine-tuned Laya, stack with the GBDT features, refit
the F_0.5 threshold on the stack's own output, and report macro F_0.5 by country and singleton
status.

Not executed here (needs the fine-tuned Laya checkpoints from laya_finetune.py and a GPU for
reasonable Laya inference speed -- CPU works too, just slowly). Read back on the GPU machine and
compare the printed threshold-sweep / country-breakdown numbers against train_gbdt.py's plain-GBDT
val report before trusting that Laya + stacking actually helped: NEEDS GPU-MACHINE VERIFICATION,
this cannot be assumed to be an improvement without measuring it.

Run (from code/business_entity_resolution/), after train_gbdt.py and laya_finetune.py --stage
train have both produced their model artifacts:
    python -m src.ensemble --features data_processed/features_train.parquet

Writes:
    models/ensemble_lr.joblib                  (primary: LogisticRegression stacker)
    models/ensemble_gbdt_alt.txt                (alternative: shallow LightGBM stacker)
    models/ensemble_stack_columns.json          (exact stacking feature column order)
    models/ensemble_threshold.json              (F_0.5-optimal decision threshold, per model)
    data_processed/ensemble_val_report.json     (macro F_0.5 by country + singleton breakdown)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from src import common
from src import train_gbdt
from src.features import load_entity_lookup
from src.laya_finetune import record_text, SAME_ENTITY_INSTRUCTIONS, SAME_ENTITY_CRITERIA, build_router

DEFAULT_THRESHOLDS = [round(t, 2) for t in np.arange(0.05, 0.96, 0.05)]


# ------------------------------------------------------------------------------------- Laya pass

def score_with_laya(router, features_df: pd.DataFrame, lookup: Dict[str, dict],
                     batch_size: int = 64) -> np.ndarray:
    """Batched Laya scoring via `Router.predict_batch` (README: "predict_batch routes the full
    workload first, groups requests by checkpoint... dispatched to Agent.predict_batch() so
    states can share forward passes"). Returns P(same_entity) per row of `features_df`, in order.
    """
    questions = {"same_entity": {"type": "noul", "instructions": SAME_ENTITY_INSTRUCTIONS,
                                  "criteria": SAME_ENTITY_CRITERIA}}
    requests = []
    for row in features_df.itertuples(index=False):
        s1 = lookup.get(row.source1_entity_id, {})
        cand = lookup.get(row.candidate_entity_id, {})
        requests.append({"state": {"record_a": record_text(s1), "record_b": record_text(cand)},
                          "questions": questions})
    results = router.predict_batch(requests, batch_size=batch_size)
    return np.array([res["answers"]["same_entity"]["noul"] for res in results], dtype="float32")


# ------------------------------------------------------------------------------- stacking table

def build_stack_frame(features_df: pd.DataFrame, gbdt_prob: np.ndarray, laya_prob: np.ndarray,
                       base_cols: List[str]) -> pd.DataFrame:
    stack = features_df[base_cols].astype("float32").copy()
    stack["gbdt_prob"] = gbdt_prob.astype("float32")
    stack["laya_prob"] = laya_prob.astype("float32")
    return stack


def stack_columns(base_cols: List[str]) -> List[str]:
    return list(base_cols) + ["gbdt_prob", "laya_prob"]


# ---------------------------------------------------------------------------------- train stack

def train_logistic_stacker(X_train, y_train):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])
    pipe.fit(X_train, y_train)
    return pipe


def train_shallow_gbdt_stacker(X_train, y_train, X_val, y_val, cols: List[str], seed: int = 42):
    import lightgbm as lgb

    train_set = lgb.Dataset(X_train, label=y_train, feature_name=cols)
    val_set = lgb.Dataset(X_val, label=y_val, feature_name=cols, reference=train_set)
    params = {
        "objective": "binary", "metric": "binary_logloss", "learning_rate": 0.05,
        "num_leaves": 7, "max_depth": 3, "min_data_in_leaf": 30,
        "is_unbalance": True, "verbose": -1, "seed": seed,
    }
    model = lgb.train(params, train_set, num_boost_round=500, valid_sets=[val_set],
                       callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)])
    return model


def score_stack(model, kind: str, X) -> np.ndarray:
    if kind == "logistic":
        return model.predict_proba(X)[:, 1]
    return model.predict(X, num_iteration=getattr(model, "best_iteration", None))


# ---------------------------------------------------------------------- threshold + evaluation

def best_threshold(pairs_df: pd.DataFrame, scores: np.ndarray, truth: Dict[str, set],
                    eval_ids: set, thresholds=None) -> Dict:
    thresholds = thresholds if thresholds is not None else DEFAULT_THRESHOLDS
    empty_pred = {s1: set() for s1 in eval_ids}
    eval_truth = {s1: ids for s1, ids in truth.items() if s1 in eval_ids}

    best = {"threshold": 0.5, "macro_f0_5": -1.0}
    sweep = {}
    for t in thresholds:
        pred = dict(empty_pred)
        for s1, cand, score in zip(pairs_df["source1_entity_id"], pairs_df["candidate_entity_id"], scores):
            if s1 in pred and score >= t:
                pred[s1].add(cand)
        precision, recall = common.precision_recall_macro(pred, eval_truth)
        macro_f05 = common.macro_f_beta(pred, eval_truth, beta=0.5)
        sweep[str(t)] = {"precision": precision, "recall": recall, "macro_f0_5": macro_f05}
        if macro_f05 > best["macro_f0_5"]:
            best = {"threshold": t, "macro_f0_5": macro_f05, "precision": precision, "recall": recall}
    return {"best": best, "sweep": sweep}


def breakdown_report(pairs_df: pd.DataFrame, scores: np.ndarray, threshold: float,
                      truth: Dict[str, set], eval_ids: set, lookup: Dict[str, dict]) -> Dict:
    """Macro F_0.5 broken down by country (esp. France, which Laya never trains/calibrates on --
    see laya_finetune.py's script-based routing) and separately for singleton vs non-singleton
    entities."""
    pred = {s1: set() for s1 in eval_ids}
    for s1, cand, score in zip(pairs_df["source1_entity_id"], pairs_df["candidate_entity_id"], scores):
        if s1 in pred and score >= threshold:
            pred[s1].add(cand)
    eval_truth = {s1: ids for s1, ids in truth.items() if s1 in eval_ids}

    per_entity = common.f_beta_per_entity(pred, eval_truth, beta=0.5)

    by_country: Dict[str, List[float]] = {}
    singleton_scores, nonsingleton_scores = [], []
    for s1, score in per_entity.items():
        s1_rec = lookup.get(s1, {})
        country = s1_rec.get("country", "unknown") or "unknown"
        by_country.setdefault(country, []).append(score)
        (singleton_scores if not eval_truth.get(s1) else nonsingleton_scores).append(score)

    return {
        "overall_macro_f0_5": sum(per_entity.values()) / len(per_entity) if per_entity else float("nan"),
        "by_country": {c: {"n": len(v), "macro_f0_5": sum(v) / len(v)} for c, v in sorted(by_country.items())},
        "singletons": {"n": len(singleton_scores),
                       "macro_f0_5": (sum(singleton_scores) / len(singleton_scores)) if singleton_scores else float("nan")},
        "non_singletons": {"n": len(nonsingleton_scores),
                            "macro_f0_5": (sum(nonsingleton_scores) / len(nonsingleton_scores)) if nonsingleton_scores else float("nan")},
    }


# --------------------------------------------------------------------------------------- driver

def run(repo_root: Path, features_path, val_frac: float, seed: int, laya_batch_size: int):
    features_path = features_path or (common.data_processed_dir(repo_root) / "features_train.parquet")
    features_df = pd.read_parquet(features_path)
    ground_truth = common.load_split_sources(repo_root, "train")["ground_truth"]
    truth = common.ground_truth_map(ground_truth)
    labels = train_gbdt.label_pairs(features_df, ground_truth).astype(int).to_numpy()

    train_ids, val_ids = common.stratified_split_by_s1(ground_truth, val_frac=val_frac, seed=seed)
    train_mask = features_df["source1_entity_id"].isin(train_ids).to_numpy()
    val_mask = features_df["source1_entity_id"].isin(val_ids).to_numpy()

    models_path = common.models_dir(repo_root)
    with open(models_path / "gbdt_feature_columns.json") as f:
        gbdt_cols = json.load(f)
    import lightgbm as lgb
    gbdt_model = lgb.Booster(model_file=str(models_path / "gbdt_model.txt"))
    gbdt_prob = train_gbdt.score_with_gbdt(gbdt_model, features_df, cols=gbdt_cols)

    lookup = load_entity_lookup(repo_root, "train")
    print("[ensemble] scoring every candidate pair with the fine-tuned Laya Router "
          f"({len(features_df)} pairs, batch_size={laya_batch_size})...")
    router = build_router(repo_root)
    laya_prob = score_with_laya(router, features_df, lookup, batch_size=laya_batch_size)

    stack_df = build_stack_frame(features_df, gbdt_prob, laya_prob, gbdt_cols)
    cols = stack_columns(gbdt_cols)

    X_train, y_train = stack_df[train_mask][cols], labels[train_mask]
    X_val, y_val = stack_df[val_mask][cols], labels[val_mask]
    print(f"[ensemble] stacking on {len(cols)} columns ({len(gbdt_cols)} base features + "
          f"gbdt_prob + laya_prob): {X_train.shape[0]} train / {X_val.shape[0]} val rows")

    lr_model = train_logistic_stacker(X_train, y_train)
    gbdt_alt_model = train_shallow_gbdt_stacker(X_train, y_train, X_val, y_val, cols, seed=seed)

    results = {}
    for name, model, kind in (("logistic", lr_model, "logistic"), ("gbdt_alt", gbdt_alt_model, "gbdt")):
        val_scores = score_stack(model, kind, X_val)
        pairs_val = features_df[val_mask][["source1_entity_id", "candidate_entity_id"]].reset_index(drop=True)
        threshold_report = best_threshold(pairs_val, val_scores, truth, val_ids)
        breakdown = breakdown_report(pairs_val, val_scores, threshold_report["best"]["threshold"],
                                      truth, val_ids, lookup)
        results[name] = {"threshold_sweep": threshold_report, "breakdown": breakdown}
        print(f"[ensemble] {name}: best threshold={threshold_report['best']['threshold']} "
              f"macro_F0.5={threshold_report['best']['macro_f0_5']:.4f}")
        print(json.dumps(breakdown, indent=2))

    import joblib
    joblib.dump(lr_model, models_path / "ensemble_lr.joblib")
    gbdt_alt_model.save_model(str(models_path / "ensemble_gbdt_alt.txt"),
                               num_iteration=gbdt_alt_model.best_iteration)
    with open(models_path / "ensemble_stack_columns.json", "w") as f:
        json.dump(cols, f, indent=2)
    with open(models_path / "ensemble_threshold.json", "w") as f:
        json.dump({name: r["threshold_sweep"]["best"]["threshold"] for name, r in results.items()}, f, indent=2)
    report_path = common.data_processed_dir(repo_root) / "ensemble_val_report.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[ensemble] wrote {models_path / 'ensemble_lr.joblib'}, "
          f"{models_path / 'ensemble_gbdt_alt.txt'}, {report_path}")
    print("[ensemble] NOTE: 'logistic' is the primary/interpretable option per the challenge "
          "write-up; 'gbdt_alt' is provided to compare. Pick whichever scores higher macro F_0.5 "
          "on this val report for predict.py's --stack-model flag -- decide that on the GPU "
          "machine's real numbers, not here.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    common.add_repo_root_arg(parser)
    parser.add_argument("--features", type=str, default=None,
                         help="Path to features_train.parquet. Default: data_processed/features_train.parquet")
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--laya-batch-size", type=int, default=64)
    args = parser.parse_args()
    run(args.repo_root, args.features, args.val_frac, args.seed, args.laya_batch_size)


if __name__ == "__main__":
    main()

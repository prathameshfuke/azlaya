"""Step 6: ensemble -- score a shortlist of candidates with fine-tuned Laya, stack with the GBDT
features, refit the F_0.5 threshold on the stack's own output, and report macro F_0.5 by country
and singleton status.

Laya scores a SHORTLIST, not every candidate pair: with ~30 candidates per S1 entity the full
pair set is millions of rows, and a transformer forward pass per pair doesn't fit in a Kaggle
session. The shortlist is each S1 entity's top LAYA_TOP_N candidates by GBDT probability, restricted
to those with gbdt_prob >= LAYA_MIN_GBDT_PROB. Pairs below that bar are already confident GBDT
rejections; Laya's job is the semantic cases near the top of the list (DBA names, domain-as-name,
paraphrased legal names). Unscored pairs get laya_prob=0 and laya_scored=0, so the stacker can tell
"Laya said no" from "Laya never looked". The SAME shortlist rule is applied in predict.py.

Run (from code/business_entity_resolution/), after train_gbdt.py and laya_finetune.py --stage train:
    python -m src.ensemble --features ../../data_processed/features_train.parquet

Writes:
    models/ensemble_lr.joblib, models/ensemble_gbdt_alt.txt, models/ensemble_stack_columns.json,
    models/ensemble_threshold.json, models/ensemble_config.json (shortlist settings),
    data_processed/ensemble_val_report.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from src import common
from src import train_gbdt
from src.features import load_entity_table
from src.laya_finetune import SAME_ENTITY_CRITERIA, SAME_ENTITY_INSTRUCTIONS, build_routers

DEFAULT_THRESHOLDS = [round(t, 2) for t in np.arange(0.05, 0.96, 0.05)]
LAYA_TOP_N = 3
LAYA_MIN_GBDT_PROB = 0.05
LAYA_CHUNK = 20_000
# Records are short ("name | address | country"), so a large batch keeps a T4 busy; 64 left it
# mostly idle waiting on CPU-side tokenization between tiny forward passes.
LAYA_BATCH_SIZE = 256


# ------------------------------------------------------------------------------------- Laya pass

def laya_shortlist_mask(features_df: pd.DataFrame, gbdt_prob: np.ndarray, top_n: int,
                        min_prob: float) -> np.ndarray:
    frame = pd.DataFrame({"s1": features_df["source1_entity_id"].to_numpy(), "p": gbdt_prob})
    rank = frame.groupby("s1")["p"].rank(method="first", ascending=False)
    return ((rank <= top_n) & (frame["p"] >= min_prob)).to_numpy()


def record_texts(table: pd.DataFrame) -> pd.Series:
    """Same "name | address | country" format laya_finetune.record_text uses for training."""
    return table["business_name"] + " | " + table["business_address"] + " | " + table["country"]


def score_with_laya(routers, features_df: pd.DataFrame, table: pd.DataFrame, mask: np.ndarray,
                    batch_size: int = LAYA_BATCH_SIZE, chunk_size: int = LAYA_CHUNK) -> np.ndarray:
    """P(same_entity) for rows where `mask` is True (0.0 elsewhere). Chunks are spread across
    `routers` (one per GPU, from laya_finetune.build_routers) by a thread pool -- each worker
    checks a router out of a queue, so no two chunks share a GPU at once. torch releases the GIL
    during forward passes, so two T4s genuinely run concurrently."""
    import queue
    from concurrent.futures import ThreadPoolExecutor

    if not isinstance(routers, (list, tuple)):
        routers = [routers]
    questions = {"same_entity": {"type": "noul", "instructions": SAME_ENTITY_INSTRUCTIONS,
                                 "criteria": SAME_ENTITY_CRITERIA}}
    texts = record_texts(table)
    out = np.zeros(len(features_df), dtype=np.float32)
    idx = np.flatnonzero(mask)
    s1_col = features_df["source1_entity_id"].to_numpy()
    cand_col = features_df["candidate_entity_id"].to_numpy()
    chunks = [idx[start:start + chunk_size] for start in range(0, len(idx), chunk_size)]
    print(f"[laya] scoring {len(idx)} shortlisted pairs of {len(features_df)} on {len(routers)} device(s)")

    free = queue.Queue()
    for router in routers:
        free.put(router)

    def _score(rows):
        router = free.get()
        try:
            a = texts.loc[s1_col[rows]].tolist()
            b = texts.loc[cand_col[rows]].tolist()
            requests = [{"state": {"record_a": x, "record_b": y}, "questions": questions} for x, y in zip(a, b)]
            results = router.predict_batch(requests, batch_size=batch_size)
            return rows, [res["answers"]["same_entity"]["noul"] for res in results]
        finally:
            free.put(router)

    with ThreadPoolExecutor(max_workers=len(routers)) as pool, \
            tqdm(total=len(idx), desc="laya scoring", unit="pair", unit_scale=True) as bar:
        for rows, probs in pool.map(_score, chunks):
            out[rows] = probs
            bar.update(len(rows))
    return out


# ------------------------------------------------------------------------------- stacking table

def build_stack_frame(features_df: pd.DataFrame, gbdt_prob: np.ndarray, laya_prob: np.ndarray,
                      laya_scored: np.ndarray, base_cols: List[str]) -> pd.DataFrame:
    stack = features_df[base_cols].astype("float32").copy()
    stack["gbdt_prob"] = gbdt_prob.astype("float32")
    stack["laya_prob"] = laya_prob.astype("float32")
    stack["laya_scored"] = laya_scored.astype("float32")
    return stack


def stack_columns(base_cols: List[str]) -> List[str]:
    return list(base_cols) + ["gbdt_prob", "laya_prob", "laya_scored"]


# ---------------------------------------------------------------------------------- train stack

def train_logistic_stacker(X_train, y_train):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

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
    return lgb.train(params, train_set, num_boost_round=500, valid_sets=[val_set],
                     callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)])


def score_stack(model, kind: str, X) -> np.ndarray:
    if kind == "logistic":
        return model.predict_proba(X)[:, 1]
    return model.predict(X, num_iteration=getattr(model, "best_iteration", None))


# ---------------------------------------------------------------------- threshold + evaluation

def predicted_sets(pairs_df: pd.DataFrame, scores: np.ndarray, threshold: float,
                   eval_ids) -> Dict[str, set]:
    pred = {s1: set() for s1 in eval_ids}
    keep = scores >= threshold
    for s1, cand in zip(pairs_df["source1_entity_id"].to_numpy()[keep],
                        pairs_df["candidate_entity_id"].to_numpy()[keep]):
        if s1 in pred:
            pred[s1].add(cand)
    return pred


def best_threshold(pairs_df: pd.DataFrame, scores: np.ndarray, truth: Dict[str, set],
                   eval_ids: set, thresholds=None) -> Dict:
    thresholds = thresholds if thresholds is not None else DEFAULT_THRESHOLDS
    eval_truth = {s1: ids for s1, ids in truth.items() if s1 in eval_ids}
    best = {"threshold": 0.5, "macro_f0_5": -1.0}
    sweep = {}
    for t in tqdm(thresholds, desc="threshold sweep", unit="t"):
        pred = predicted_sets(pairs_df, scores, t, eval_ids)
        precision, recall = common.precision_recall_macro(pred, eval_truth)
        macro_f05 = common.macro_f_beta(pred, eval_truth, beta=0.5)
        sweep[str(t)] = {"precision": precision, "recall": recall, "macro_f0_5": macro_f05}
        if macro_f05 > best["macro_f0_5"]:
            best = {"threshold": t, "macro_f0_5": macro_f05, "precision": precision, "recall": recall}
    return {"best": best, "sweep": sweep}


def breakdown_report(pairs_df: pd.DataFrame, scores: np.ndarray, threshold: float,
                     truth: Dict[str, set], eval_ids: set, table: pd.DataFrame) -> Dict:
    """Macro F_0.5 by country (esp. France, which Laya never trains on) and by singleton status."""
    pred = predicted_sets(pairs_df, scores, threshold, eval_ids)
    eval_truth = {s1: ids for s1, ids in truth.items() if s1 in eval_ids}
    per_entity = common.f_beta_per_entity(pred, eval_truth, beta=0.5)
    countries = table["country"]

    by_country: Dict[str, List[float]] = {}
    singleton_scores, nonsingleton_scores = [], []
    for s1, score in per_entity.items():
        country = (countries.get(s1) or "unknown") if s1 in countries.index else "unknown"
        by_country.setdefault(country, []).append(score)
        (singleton_scores if not eval_truth.get(s1) else nonsingleton_scores).append(score)

    def _mean(v):
        return sum(v) / len(v) if v else float("nan")

    return {
        "overall_macro_f0_5": _mean(list(per_entity.values())),
        "by_country": {c: {"n": len(v), "macro_f0_5": _mean(v)} for c, v in sorted(by_country.items())},
        "singletons": {"n": len(singleton_scores), "macro_f0_5": _mean(singleton_scores)},
        "non_singletons": {"n": len(nonsingleton_scores), "macro_f0_5": _mean(nonsingleton_scores)},
    }


# --------------------------------------------------------------------------------------- driver

def run(repo_root: Path, features_path, val_frac: float, seed: int, laya_batch_size: int,
        laya_top_n: int, laya_min_gbdt_prob: float):
    import joblib
    import lightgbm as lgb

    features_path = features_path or (common.data_processed_dir(repo_root) / "features_train.parquet")
    features_df = pd.read_parquet(common.require(features_path, "features.py (notebook Section 3)"))
    ground_truth = common.load_split_sources(repo_root, "train")["ground_truth"]
    truth = common.ground_truth_map(ground_truth)
    labels = train_gbdt.label_pairs(features_df, ground_truth).astype(int).to_numpy()

    train_ids, val_ids = common.stratified_split_by_s1(ground_truth, val_frac=val_frac, seed=seed)
    train_mask = features_df["source1_entity_id"].isin(train_ids).to_numpy()
    val_mask = features_df["source1_entity_id"].isin(val_ids).to_numpy()

    models_path = common.models_dir(repo_root)
    with open(common.require(models_path / "gbdt_feature_columns.json", "train_gbdt.py (notebook Section 4)")) as f:
        gbdt_cols = json.load(f)
    gbdt_model = lgb.Booster(model_file=str(models_path / "gbdt_model.txt"))
    gbdt_prob = train_gbdt.score_with_gbdt(gbdt_model, features_df, cols=gbdt_cols)

    table = load_entity_table(repo_root, "train")
    shortlist = laya_shortlist_mask(features_df, gbdt_prob, laya_top_n, laya_min_gbdt_prob)
    routers = build_routers(repo_root)
    laya_prob = score_with_laya(routers, features_df, table, shortlist, batch_size=laya_batch_size)

    stack_df = build_stack_frame(features_df, gbdt_prob, laya_prob, shortlist, gbdt_cols)
    cols = stack_columns(gbdt_cols)
    X_train, y_train = stack_df[train_mask][cols], labels[train_mask]
    X_val, y_val = stack_df[val_mask][cols], labels[val_mask]
    print(f"[ensemble] stacking on {len(cols)} columns: {X_train.shape[0]} train / {X_val.shape[0]} val rows")

    lr_model = train_logistic_stacker(X_train, y_train)
    gbdt_alt_model = train_shallow_gbdt_stacker(X_train, y_train, X_val, y_val, cols, seed=seed)

    pairs_val = features_df.loc[val_mask, ["source1_entity_id", "candidate_entity_id"]].reset_index(drop=True)
    gbdt_only = best_threshold(pairs_val, gbdt_prob[val_mask], truth, val_ids)
    results = {"gbdt_only_baseline": {"threshold_sweep": gbdt_only}}
    print(f"[ensemble] GBDT-only baseline: best threshold={gbdt_only['best']['threshold']} "
          f"macro_F0.5={gbdt_only['best']['macro_f0_5']:.4f}")

    for name, model, kind in (("logistic", lr_model, "logistic"), ("gbdt_alt", gbdt_alt_model, "gbdt")):
        val_scores = score_stack(model, kind, X_val)
        threshold_report = best_threshold(pairs_val, val_scores, truth, val_ids)
        breakdown = breakdown_report(pairs_val, val_scores, threshold_report["best"]["threshold"],
                                     truth, val_ids, table)
        results[name] = {"threshold_sweep": threshold_report, "breakdown": breakdown}
        print(f"[ensemble] {name}: best threshold={threshold_report['best']['threshold']} "
              f"macro_F0.5={threshold_report['best']['macro_f0_5']:.4f}")
        print(json.dumps(breakdown, indent=2))

    joblib.dump(lr_model, models_path / "ensemble_lr.joblib")
    gbdt_alt_model.save_model(str(models_path / "ensemble_gbdt_alt.txt"),
                              num_iteration=gbdt_alt_model.best_iteration)
    with open(models_path / "ensemble_stack_columns.json", "w") as f:
        json.dump(cols, f, indent=2)
    with open(models_path / "ensemble_threshold.json", "w") as f:
        json.dump({name: results[name]["threshold_sweep"]["best"]["threshold"]
                   for name in ("logistic", "gbdt_alt")}, f, indent=2)
    with open(models_path / "ensemble_config.json", "w") as f:
        json.dump({"laya_top_n": laya_top_n, "laya_min_gbdt_prob": laya_min_gbdt_prob}, f, indent=2)
    report_path = common.data_processed_dir(repo_root) / "ensemble_val_report.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[ensemble] wrote models + {report_path}. Compare 'logistic'/'gbdt_alt' against "
          f"'gbdt_only_baseline' before trusting that Laya helped.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    common.add_repo_root_arg(parser)
    parser.add_argument("--features", type=str, default=None,
                        help="Path to features_train.parquet. Default: data_processed/features_train.parquet")
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--laya-batch-size", type=int, default=LAYA_BATCH_SIZE)
    parser.add_argument("--laya-top-n", type=int, default=LAYA_TOP_N)
    parser.add_argument("--laya-min-gbdt-prob", type=float, default=LAYA_MIN_GBDT_PROB)
    args = parser.parse_args()
    run(args.repo_root, args.features, args.val_frac, args.seed, args.laya_batch_size,
        args.laya_top_n, args.laya_min_gbdt_prob)


if __name__ == "__main__":
    main()

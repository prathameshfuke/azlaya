"""Step 7: full test-set inference -- blocking -> features -> GBDT -> Laya (shortlist) -> ensemble
-> threshold. Produces both submission files.

Run LAST, after train_gbdt.py, laya_finetune.py (prepare + train for both roles) and ensemble.py.

Run (from code/business_entity_resolution/):
    python -m src.predict --stack-model logistic

Writes (paths match what utils/validate_submission.py expects, run from student_resource/):
    output/candidate_pairs.tsv     -- the test-set candidates actually fed to the final model
    output/matching_results.tsv    -- final matches
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src import blocking, common, ensemble, features, train_gbdt
from src.laya_finetune import build_routers


def run(repo_root: Path, k: int, stack_model: str, laya_batch_size: int,
        threshold_override: float):
    import lightgbm as lgb

    models_path = common.models_dir(repo_root)

    # ---- 1. blocking ----
    s1_df, s2_df, s3_df = blocking.load_blocking_inputs(repo_root, "test")
    all_s1_ids = list(s1_df["entity_id"])
    print(f"[predict] blocking: {len(s1_df)} S1 test entities (all of them), k={k}")
    candidate_map = blocking.generate_candidates(s1_df, s2_df, s3_df, k=k)
    del s2_df, s3_df
    candidate_path = common.output_dir(repo_root) / "candidate_pairs.tsv"
    common.write_id_list_tsv(candidate_map, candidate_path, "source1_entity_id", "candidate_entity_ids")
    print(f"[predict] wrote {candidate_path} ({sum(len(v) for v in candidate_map.values())} pairs)")

    # ---- 2. features ----
    pairs_df = features.build_pairs_frame(candidate_map)
    del candidate_map
    table = features.load_entity_table(repo_root, "test", ids=features.pair_entity_ids(pairs_df))
    features_df = features.compute_features(pairs_df, table)
    del pairs_df

    if features_df.empty:
        print("[predict] WARNING: zero candidate pairs survived blocking -- every S1 test entity will "
              "be predicted as a singleton. This almost certainly means a blocking bug.")
        _write_matches(repo_root, {s1: set() for s1 in all_s1_ids})
        return

    # ---- 3. GBDT ----
    with open(common.require(models_path / "gbdt_feature_columns.json", "train_gbdt.py (notebook Section 4)")) as f:
        gbdt_cols = json.load(f)
    gbdt_model = lgb.Booster(model_file=str(models_path / "gbdt_model.txt"))
    gbdt_prob = train_gbdt.score_with_gbdt(gbdt_model, features_df, cols=gbdt_cols)
    print(f"[predict] GBDT scored {len(gbdt_prob)} pairs (mean prob {gbdt_prob.mean():.4f})")

    # ---- 4+5. Laya shortlist + ensemble stack, or GBDT alone ----
    if stack_model == "gbdt_only":
        # Chosen when the ensemble didn't beat the GBDT-only baseline on stack-eval: skips Laya
        # (and all its GPU time) entirely.
        final_prob = gbdt_prob
    else:
        with open(common.require(models_path / "ensemble_config.json", "ensemble.py (notebook Section 6)")) as f:
            cfg = json.load(f)
        shortlist = ensemble.laya_shortlist_mask(features_df, gbdt_prob, cfg["laya_top_n"], cfg["laya_min_gbdt_prob"])
        routers = build_routers(repo_root)
        laya_prob = ensemble.score_with_laya(routers, features_df, table, shortlist, batch_size=laya_batch_size)
        del routers
        with open(models_path / "ensemble_stack_columns.json") as f:
            stack_cols = json.load(f)
        stack_df = ensemble.build_stack_frame(features_df, gbdt_prob, laya_prob, shortlist, gbdt_cols)[stack_cols]
        if stack_model == "logistic":
            import joblib
            final_prob = ensemble.score_stack(joblib.load(models_path / "ensemble_lr.joblib"), "logistic", stack_df)
        else:
            alt_model = lgb.Booster(model_file=str(models_path / "ensemble_gbdt_alt.txt"))
            final_prob = ensemble.score_stack(alt_model, "gbdt", stack_df)
    print(f"[predict] {stack_model} mean prob {final_prob.mean():.4f}")

    # ---- 6. threshold ----
    if threshold_override is not None:
        threshold = threshold_override
    else:
        with open(common.require(models_path / "ensemble_threshold.json", "ensemble.py (notebook Section 6)")) as f:
            thresholds = json.load(f)
        threshold = thresholds["gbdt_only_baseline" if stack_model == "gbdt_only" else stack_model]
    print(f"[predict] decision threshold: {threshold}")

    # ---- 7. matching_results.tsv (every S1 seeded first -> one row each; subset of candidates) ----
    matches = ensemble.predicted_sets(features_df, final_prob, threshold, all_s1_ids)
    _write_matches(repo_root, matches)


def _write_matches(repo_root: Path, matches):
    matching_path = common.output_dir(repo_root) / "matching_results.tsv"
    common.write_id_list_tsv(matches, matching_path, "source1_entity_id", "matched_entity_ids")
    n_matched = sum(1 for ids in matches.values() if ids)
    print(f"[predict] wrote {matching_path}: {len(matches)} S1 entities, {n_matched} with >=1 match, "
          f"{len(matches) - n_matched} predicted singletons")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    common.add_repo_root_arg(parser)
    parser.add_argument("--k", type=int, default=blocking.DEFAULT_K)
    parser.add_argument("--stack-model", choices=["logistic", "gbdt_alt", "gbdt_only"], default="logistic")
    parser.add_argument("--laya-batch-size", type=int, default=ensemble.LAYA_BATCH_SIZE)
    parser.add_argument("--threshold", type=float, default=None,
                        help="Override the threshold saved by ensemble.py.")
    args = parser.parse_args()
    run(args.repo_root, args.k, args.stack_model, args.laya_batch_size, args.threshold)


if __name__ == "__main__":
    main()

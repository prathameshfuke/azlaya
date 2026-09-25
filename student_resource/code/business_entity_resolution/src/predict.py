"""Step 7: full test-set inference -- blocking -> features -> GBDT -> Laya -> ensemble -> threshold.

Not executed here. This is the script that actually produces the two submission files; run it
LAST, after train_gbdt.py, laya_finetune.py (--stage prepare, then --stage train for both roles)
and ensemble.py have all produced their model artifacts under models/. Validate its output with
utils/validate_submission.py on the GPU machine (see README.md) -- that step is also not run here.

Run (from code/business_entity_resolution/):
    python -m src.predict --stack-model logistic

Writes (paths match what utils/validate_submission.py expects by default, run from student_resource/):
    output/candidate_pairs.tsv     -- the test-set candidates actually fed to the final model
    output/matching_results.tsv    -- final matches
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src import common
from src import blocking
from src import features
from src import train_gbdt
from src import ensemble
from src.laya_finetune import build_router


def run(repo_root: Path, k: int, score_method: str, stack_model: str, laya_batch_size: int,
        threshold_override: float):
    # ---- 1. blocking ----
    normalized = common.load_all_normalized(repo_root, "test")
    s1_df, s2_df, s3_df = normalized["source1"], normalized["source2"], normalized["source3"]
    print(f"[predict] blocking: {len(s1_df)} S1 test entities, k={k}, score_method={score_method}")
    candidates = blocking.generate_candidates(s1_df, s2_df, s3_df, k=k, score_method=score_method)
    candidate_id_map = blocking.candidates_to_id_map(candidates)
    # Every S1 test entity gets a row, even with an empty candidate set: generate_candidates
    # iterates every row of s1_df, so candidate_id_map already has one key per S1 test entity.
    candidate_path = common.output_dir(repo_root) / "candidate_pairs.tsv"
    common.write_id_list_tsv(candidate_id_map, candidate_path, "source1_entity_id", "candidate_entity_ids")
    print(f"[predict] wrote {candidate_path} ({sum(len(v) for v in candidate_id_map.values())} pairs)")

    # ---- 2. features ----
    pairs_df = features.build_pairs_frame(candidate_id_map)
    lookup = features.load_entity_lookup(repo_root, "test")
    print(f"[predict] features: {len(pairs_df)} candidate pairs")
    features_df = features.compute_features(pairs_df, lookup)

    all_s1_ids = list(s1_df["entity_id"])
    if features_df.empty:
        print("[predict] WARNING: zero candidate pairs survived blocking -- every S1 test entity "
              "will be predicted as a singleton. This almost certainly means a blocking bug "
              "(e.g. a country-label mismatch between normalize.py's output and what blocking.py "
              "expects); investigate before trusting this submission.")
        matches = {s1: set() for s1 in all_s1_ids}
        _write_matches(repo_root, matches)
        return

    # ---- 3. GBDT ----
    models_path = common.models_dir(repo_root)
    with open(models_path / "gbdt_feature_columns.json") as f:
        gbdt_cols = json.load(f)
    import lightgbm as lgb
    gbdt_model = lgb.Booster(model_file=str(models_path / "gbdt_model.txt"))
    gbdt_prob = train_gbdt.score_with_gbdt(gbdt_model, features_df, cols=gbdt_cols)
    print(f"[predict] GBDT scored {len(gbdt_prob)} pairs "
          f"(mean prob {gbdt_prob.mean():.4f})")

    # ---- 4. Laya ----
    print(f"[predict] scoring with the fine-tuned Laya Router (batch_size={laya_batch_size})...")
    router = build_router(repo_root)
    laya_prob = ensemble.score_with_laya(router, features_df, lookup, batch_size=laya_batch_size)
    print(f"[predict] Laya scored {len(laya_prob)} pairs (mean prob {laya_prob.mean():.4f})")

    # ---- 5. ensemble stack ----
    with open(models_path / "ensemble_stack_columns.json") as f:
        stack_cols = json.load(f)
    stack_df = ensemble.build_stack_frame(features_df, gbdt_prob, laya_prob, gbdt_cols)[stack_cols]

    if stack_model == "logistic":
        import joblib
        model = joblib.load(models_path / "ensemble_lr.joblib")
        final_prob = ensemble.score_stack(model, "logistic", stack_df)
    else:
        alt_model = lgb.Booster(model_file=str(models_path / "ensemble_gbdt_alt.txt"))
        final_prob = ensemble.score_stack(alt_model, "gbdt", stack_df)
    print(f"[predict] ensemble ({stack_model}) mean prob {final_prob.mean():.4f}")

    # ---- 6. threshold ----
    if threshold_override is not None:
        threshold = threshold_override
    else:
        with open(models_path / "ensemble_threshold.json") as f:
            thresholds = json.load(f)
        threshold = thresholds[stack_model]
    print(f"[predict] decision threshold: {threshold}")

    # ---- 7. build + write matching_results.tsv ----
    matches = {s1: set() for s1 in all_s1_ids}
    for s1, cand, prob in zip(features_df["source1_entity_id"], features_df["candidate_entity_id"], final_prob):
        if prob >= threshold:
            matches[s1].add(cand)  # only ever adds IDs that were in candidate_id_map -> subset guaranteed
    _write_matches(repo_root, matches)


def _write_matches(repo_root: Path, matches):
    matching_path = common.output_dir(repo_root) / "matching_results.tsv"
    common.write_id_list_tsv(matches, matching_path, "source1_entity_id", "matched_entity_ids")
    n_matched = sum(1 for ids in matches.values() if ids)
    print(f"[predict] wrote {matching_path}: {len(matches)} S1 entities, "
          f"{n_matched} with >=1 match, {len(matches) - n_matched} predicted singletons")
    print("[predict] NOT RUN HERE. Validate on the GPU machine from student_resource/:\n"
          "    python3 utils/validate_submission.py --matching output/matching_results.tsv "
          "--candidate output/candidate_pairs.tsv --test-dir dataset/test")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    common.add_repo_root_arg(parser)
    parser.add_argument("--k", type=int, default=blocking.DEFAULT_K)
    parser.add_argument("--score-method", choices=["jaccard", "tfidf"], default="jaccard")
    parser.add_argument("--stack-model", choices=["logistic", "gbdt_alt"], default="logistic")
    parser.add_argument("--laya-batch-size", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=None,
                         help="Override the threshold saved by ensemble.py (models/ensemble_threshold.json).")
    args = parser.parse_args()
    run(args.repo_root, args.k, args.score_method, args.stack_model, args.laya_batch_size, args.threshold)


if __name__ == "__main__":
    main()

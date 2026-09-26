# Business Entity Resolution -- pipeline

Blocking + GBDT + fine-tuned Laya, stacked, for the ML Challenge 2026 Business Entity Resolution
task. Produces `output/candidate_pairs.tsv` and `output/matching_results.tsv`.

## Easiest way to run: the notebook

`notebooks/run_pipeline_colab_kaggle.ipynb` is self-contained: it writes every file in `src/`
(plus this README, `requirements.txt`, the challenge's `validate_submission.py` and
`Documentation_template.md`) from `%%writefile` cells, then runs each step in-kernel with live
progress bars. Upload it to Kaggle, attach the dataset, choose **GPU T4 x2**, and run top to bottom.
Nothing is cloned at runtime.

The notebook is generated from `src/` by `notebooks/build_notebook.py`. After editing anything in
`src/`, regenerate it from this directory with `python notebooks/build_notebook.py`.

## Layout

```
student_resource/
├── dataset/{train,test}/...
├── utils/validate_submission.py
├── output/                  <- candidate_pairs.tsv, matching_results.tsv (predict.py)
├── data_processed/          <- normalized TSVs, candidate/feature tables, Laya datasets, reports
├── models/                  <- GBDT, fine-tuned Laya checkpoints, ensemble stackers
└── code/business_entity_resolution/
    ├── src/*.py
    ├── notebooks/
    ├── requirements.txt
    └── README.md   (this file)
```

Scripts auto-detect `student_resource/` as three directories up from `src/` (`--repo-root`
overrides). Run CLI commands **from this directory** so `python -m src.<script>` resolves.

## Command-line run order (alternative to the notebook)

```bash
pip install -r requirements.txt

python -m src.normalize --split train --split test
python -m src.blocking --split train --out ../../data_processed/candidate_pairs_train.tsv --report-recall
python -m src.features --split train --candidates ../../data_processed/candidate_pairs_train.tsv
python -m src.train_gbdt --features ../../data_processed/features_train.parquet
python -m src.laya_finetune --stage prepare
torchrun --standalone --nproc_per_node=2 -m src.laya_finetune --stage train --role english
torchrun --standalone --nproc_per_node=2 -m src.laya_finetune --stage train --role multilingual
python -m src.ensemble --features ../../data_processed/features_train.parquet
python -m src.predict --stack-model logistic        # or gbdt_alt / gbdt_only, whichever ensemble.py reports best
```

With a single GPU, use `python -m src.laya_finetune --stage train --role <role>` instead of
`torchrun`.

Validate from `student_resource/`:

```bash
python3 utils/validate_submission.py --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

## How it scales to the real data (~12.5M train records)

- **Normalization** runs row-parallel across all CPU cores.
- **Blocking** uses TF-IDF vectors (name character 4-grams; address words, which carry pin, city
  and street) and a multithreaded sparse top-n matrix product (`sparse_dot_topn`), per country.
  N-grams shared by more than 10k records are dropped. An earlier Python inverted-index version
  exhausted Kaggle's 30 GB.
- **Training uses a sample** of S1 entities (`--max-s1`, default 300k of ~2.2M). Their candidates
  are still searched against the full S2/S3 pool, so hard negatives are realistic. Every training
  step reads the sample back via `common.sampled_ground_truth`, so splits and metrics all refer to
  the same entities. The test set is always processed in full.
- **Features** are computed in 500k-pair chunks, loading only the records those pairs reference.
- **Laya fine-tuning** uses `--max-finetune-entities` (default 6000) entities: all true matches
  plus up to 3 hard negatives each, in both orderings. DDP across both T4s.
- **Laya scoring** covers a shortlist (each S1's top 3 GBDT candidates with prob >= 0.05), with one
  Router per GPU run in parallel.

## Design notes

- **Output format rules** (one row per S1, matches subset of candidates, no duplicate IDs) are
  enforced structurally: `predict.py` seeds every test S1 id before filling matches, and
  `common.write_id_list_tsv` dedupes. Still confirm with the validator.
- **Country is an open set everywhere**: blocking partitions by whatever `country` values appear;
  only the legal-suffix dictionary branches France vs. everything else.
- **No leakage in stacking**: the GBDT trains on 80% of sampled entities. `ensemble.py` trains
  its stacker only on the other 20% (the GBDT never saw them), halved into stack-train and
  stack-eval. Laya fine-tuning uses only the GBDT's training pool, so `laya_prob` isn't overfit on
  those entities either.
- **GBDT-only fallback**: `ensemble.py` reports the GBDT-only baseline on the same stack-eval
  entities. If the Laya ensemble doesn't beat it, run `predict.py --stack-model gbdt_only`, which
  skips Laya entirely.
- **Laya calibration uses hard 0/1 labels.** The ground truth has no per-pair uncertainty to use
  as soft targets. The RLCD objective and post-hoc temperature fit still work on one-hot targets.
- **Heuristics worth eyeballing on real rows**: DBA splitting and `guess_city` in `normalize.py`.

## What this pipeline does NOT do

- No external data, API, or lookup of any kind (geocoding, business registries, etc.). Nothing
  here calls anything but Hugging Face Hub, to download the two base Laya checkpoints
  (Apache-2.0, 421M / 322M parameters, under the 8B cap).
- No hyperparameter search: LightGBM and RLCD settings are reasonable starting points exposed as
  arguments, not tuned against a score.

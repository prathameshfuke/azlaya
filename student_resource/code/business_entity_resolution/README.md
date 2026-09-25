# Business Entity Resolution -- pipeline

Blocking + GBDT + fine-tuned Laya, stacked, for the ML Challenge 2026 Business Entity Resolution
task. **Every script under `src/` was written on a machine with no GPU and has never been run.**
Nothing here has been executed, trained, or validated end-to-end -- run the steps below yourself,
in order, on a GPU machine (Colab / Kaggle / AWS), and read each script's own module docstring for
what it does and does not verify. Several places are flagged in code comments as
"NEEDS GPU-MACHINE VERIFICATION" -- treat those as a checklist, not an afterthought.

## Layout

This folder (`code/business_entity_resolution/`) is a self-contained copy of the pipeline, matching
the required final-submission zip structure. It expects to sit inside the challenge's
`student_resource/` directory, alongside `dataset/` and `utils/`:

```
student_resource/
├── dataset/{train,test}/...
├── utils/validate_submission.py
├── output/                  <- created by blocking.py / predict.py
├── data_processed/          <- created by normalize.py / features.py / laya_finetune.py
├── models/                  <- created by train_gbdt.py / laya_finetune.py / ensemble.py
└── code/business_entity_resolution/
    ├── src/*.py
    ├── requirements.txt
    └── README.md   (this file)
```

Every script auto-detects `student_resource/` as three directories up from `src/` and defaults
`--repo-root` to it; pass `--repo-root /some/other/path` to override (e.g. if you copy just this
folder somewhere else on the GPU machine and keep `dataset/` elsewhere).

All commands below are run **from this directory** (`code/business_entity_resolution/`) so that
`python -m src.<script>` resolves the `src` package correctly.

## Setup (GPU machine)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`laya_finetune.py` downloads two checkpoints from Hugging Face Hub on first use
(`convaiinnovations/laya`, `convaiinnovations/laya-multilingual`) -- needs network access. Both are
Apache-2.0 licensed and well under the challenge's 8B-parameter cap (421M / 322M).

## Run order

```bash
# 1. Normalize all six source files
python -m src.normalize --split train --split test

# 2. Blocking / candidate generation
#    - train candidates feed features.py (for GBDT + Laya training) and blocking's own recall report
python -m src.blocking --split train --k 30 --out ../../data_processed/candidate_pairs_train.tsv --report-recall
#    - test candidates: predict.py (step 7) regenerates these itself into output/candidate_pairs.tsv
#      using the SAME blocking.generate_candidates() function, so running this standalone for test
#      is only useful for inspecting blocking in isolation:
python -m src.blocking --split test --k 30

# 3. Feature engineering
python -m src.features --split train --candidates ../../data_processed/candidate_pairs_train.tsv

# 4. GBDT matcher (trains + reports val precision/recall/macro F_0.5)
python -m src.train_gbdt --features ../../data_processed/features_train.parquet

# 5. Laya fine-tuning -- two stages, two roles
python -m src.laya_finetune --stage prepare
python -m src.laya_finetune --stage train --role english
python -m src.laya_finetune --stage train --role multilingual
#    On Kaggle 2xT4, launch each role with torchrun instead of `python -m` for DDP:
#    torchrun --standalone --nproc_per_node=2 -m src.laya_finetune --stage train --role english
#    torchrun --standalone --nproc_per_node=2 -m src.laya_finetune --stage train --role multilingual

# 6. Ensemble: score with Laya, stack with GBDT features, refit threshold, report by country/singleton
python -m src.ensemble --features ../../data_processed/features_train.parquet

# 7. Full test-set inference -> output/candidate_pairs.tsv + output/matching_results.tsv
python -m src.predict --stack-model logistic
```

Step 6's printed report tells you whether `logistic` or `gbdt_alt` scored higher macro F_0.5 on
the held-out val split -- pass whichever won to step 7's `--stack-model`.

## 8. Validate before submitting

Not run in this environment. From `student_resource/` (one level above this folder):

```bash
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

## Design notes / where to look first if something's off

- **Every S1 entity gets exactly one row, matches are a subset of candidates, no duplicate IDs**:
  enforced structurally (`predict.py` seeds `matches` from every test S1 id up front;
  `common.write_id_list_tsv` dedupes), not just hoped for -- but confirm with the validator above
  before submitting, not by reading this bullet.
- **Country is an open set everywhere**: `normalize.py`'s suffix-dictionary lookup and
  `blocking.py`'s partitioning both key off *whatever* `country` values are present in the data
  (`df.groupby("country")` / a two-way `country_suffix_group()` bucket for France vs.
  everything-else), never a hardcoded `{US, India}` list. France only ever changes behavior in the
  legal-suffix dictionary (`normalize.py`) and in `ensemble.py`'s country breakdown report.
- **Splits are consistent across the whole pipeline**: `train_gbdt.py`, `laya_finetune.py`'s
  calibration slice, and `ensemble.py` all call `common.stratified_split_by_s1` with the same
  default `--val-frac 0.2 --seed 42`, so "val" means the same held-out S1 entities everywhere.
  Laya's fine-tuning and calibration data is built ONLY from the remaining train pool
  (`laya_finetune.py --stage prepare`), never from the val split those other scripts score against
  -- change `--val-frac`/`--seed` in lockstep across scripts if you touch them, or the split
  guarantee breaks silently.
- **Blocking recall is the ceiling for everything downstream.** Run `blocking.py --report-recall`
  first and look at `macro_entity_recall` before spending GPU time on GBDT/Laya training -- if it's
  low, raising `--k` or trying `--score-method tfidf` is cheaper than anything downstream.
- **Scale**: the full test set is large (validate_submission.py's own docstring notes ~1.7M
  entities). `data_processed/*.parquet` is used for the feature table specifically because of this;
  the normalized `data_processed/*.tsv` files stay plain TSV for consistency with the rest of the
  challenge's file format, but converting them to parquet too is a straightforward follow-up if
  `normalize.py`/`blocking.py` I/O becomes the bottleneck at full scale. `blocking.py`'s inverted
  indices and `features.py`'s per-pair loops are correctness-first, not throughput-tuned; profile
  before assuming they're fast enough at 1.7M rows.
- **Laya's calibration is on hard 0/1 labels**, not a soft teacher distribution (see the comment in
  `laya_finetune.py:make_example`) -- our ground truth doesn't have per-pair uncertainty to draw
  soft targets from, unlike the benchmark the upstream fine-tuning notebook was built for. The RLCD
  objective and post-hoc temperature fit still run correctly on one-hot targets; it just means less
  signal about calibrated *uncertainty*, more about calibrated *confidence*.
- **DBA / legal-vs-trade name matching, city_guess, and the DBA-pattern regex are all heuristics**
  written against the challenge's documented noise patterns, not against real sample rows (none
  were available to inspect in this environment). Each is flagged inline with a
  "NEEDS GPU-MACHINE VERIFICATION" comment at the point it's defined in `normalize.py`
  (`split_dba`, `guess_city`) -- print a sample of normalized rows and eyeball them before trusting
  these as blocking/feature signals.

## What this pipeline does NOT do

- No external data, API, or lookup of any kind (geocoding, business registries, etc.) -- the
  challenge explicitly prohibits this; nothing here calls out to anything but Hugging Face Hub for
  the two base Laya checkpoints.
- No hyperparameter search: LightGBM params in `train_gbdt.py` and the RLCD hyperparameters in
  `laya_finetune.py` (epochs, learning rates, sigma schedule) are the notebook's / a reasonable
  starting point, exposed as CLI flags, not tuned against any actual score (there is none yet).

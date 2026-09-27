# Business Entity Resolution — tokenmaxxing

**Amazon ML Challenge 2026** · Team **tokenmaxxing** · Public leaderboard **F0.5 = 0.98536** (rank 508)

For every Source-1 (S1) business record, find every Source-2 / Source-3 record that describes the
same real-world business — across three countries (India, US in train; **France, unseen, in test
only**), noisy names, inconsistent addresses, and transliterated scripts. Scored by macro F0.5 per
S1 entity, which weights precision 2x over recall and gives full credit for correctly predicting
"no match" on a singleton.

> Full methodology write-up (data, every model's hyperparameters, per-stage validation numbers,
> error analysis, and the France investigation): [`../../Documentation_template.md`](../../Documentation_template.md).

---

## Contents

- [Pipeline](#pipeline)
- [Why this design](#why-this-design)
- [Results](#results-8020-hold-out-of-train-by-s1)
- [Repository layout](#repository-layout)
- [Setup](#setup)
- [Reproducing end-to-end](#reproducing-end-to-end)
- [Compute and runtime](#compute-and-runtime)
- [Constraints compliance](#constraints-compliance)

## Pipeline

```
 raw TSVs
   │
   ▼
 preprocess.py ──► dataset_processed/
   Unicode/case normalisation · abbreviation expansion (position-aware) · Indic→Latin
   transliteration (1,347-word map learned from train) · US/India region codes + FR
   département→région map · legal-form/honorific stripping · house/unit/number parsing
   │
   ▼
 e5-large, contrastively fine-tuned on train pairs + mined hard negatives
   ──► FAISS index per country ──► top-20 S2/S3 candidates per S1
       (pair recall@20 = 99.9%, recall@50 = 100.0% on held-out split)
   │
   ▼
 mDeBERTa-v3-base cross-encoder reranker
   scores every (S1, candidate) pair jointly as a "name | address" text pair
   │
   ▼
 stage-2 decision ensemble (LightGBM / XGBoost / CatBoost, 50 features)
   reranker score + cosine · rank/gap within the S1's candidate list · cross-S1
   competition margin (63% of LightGBM's gain) · house/legal/name/address comparisons
   │
   ▼
 decision: threshold (grid-searched on macro F0.5) + one S1 per record + abstain-on-
 contested-record rule, with two additional France-only rules (descriptor-swap
 rejection, stricter house-number-mismatch threshold)
   │
   ▼
 output/matching_results.tsv  +  output/candidate_pairs.tsv
   │
   ▼
 validate_submission.py  ──►  PASS
```

## Why this design

Every stage boundary here is a measured decision, not a default:

- **Country blocking is exact** — country agrees on 100% of true training pairs, so restricting
  candidates to the same country before any embedding search loses no recall.
- **K=20 candidates** was chosen from a recall curve (@10/@20/@50/@100), not guessed — fine-tuned
  retrieval already hits 99.9% recall@20, so a larger K would only add compute, not matches.
- **A reranker on top of embeddings, not instead of them** — the fine-tuned cosine actually beats
  the reranker on raw pairwise AUC, but the reranker wins by a wide margin after assignment (macro
  F0.5 0.983 vs 0.958), because it's much better at ranking the correct S1 first for a record.
- **A second, tabular decision stage on top of the reranker** — because F0.5 penalises false merges
  2x harder than misses, the decision that matters most is "which of several plausible S1s should
  actually get this record," which needs cross-candidate context a per-pair score alone can't see.
  The single highest-value feature here (63% of LightGBM's gain) is exactly that: the reranker's
  margin over the next-best competing S1.
- **An abstain rule for contested records** — refusing a record two S1s both plausibly claim is the
  higher-expected-value move under F0.5's precision weighting; this one rule is the second-largest
  single score gain in the whole pipeline (0.9831 → 0.9914).
- **France gets two narrow, targeted rules, not a blanket stricter threshold** — manual inspection
  against independently-produced reference results showed the France gap was two specific, common
  patterns (a one-word "descriptor swap" in the name template, and over-lenient house-number
  mismatches), not general overmatching. See `Documentation_template.md` §8 for the full
  investigation and measured before/after agreement numbers.

## Results (80/20 hold-out of train, by S1)

**Retrieval** (pair recall %, within country):

| Model | @10 | @20 | @50 | @100 |
|---|---|---|---|---|
| e5-large, base | 96.7 | 97.3 | 97.9 | 98.2 |
| **e5-large, fine-tuned** | **99.8** | **99.9** | **100.0** | **100.0** |

**Decision stage** (macro F0.5):

| Stage | Macro F0.5 |
|---|---|
| Cosine threshold + one-S1-per-record | 0.9583 |
| Reranker + one-S1-per-record | 0.9831 |
| Reranker + one-S1-per-record + abstain rule | 0.9914 |
| Stage-2 ensemble, 50 features (deployed) | 0.9939 |

Full breakdown (per-model stage-2 comparison, error-pattern analysis, France before/after) is in
`Documentation_template.md`.

## Repository layout

```
code/business_entity_resolution/
├── README.md                    this file
├── requirements.txt              pinned dependencies
└── src/
    ├── preprocess.py             normalisation, abbreviation expansion, transliteration, parsing
    ├── retrieval_experiments.py  base embedding models, FAISS search, recall eval, train/val split
    ├── finetune_e5.py            hard-negative mining + contrastive fine-tuning of e5-large
    ├── train_reranker.py         mDeBERTa cross-encoder: candidates, training, prediction, eval
    ├── error_analysis.py         false-positive / false-negative pattern analysis
    ├── stack_lgbm.py             stage-2 feature engineering + F0.5-optimal threshold search
    ├── stack_models.py           LightGBM / XGBoost / CatBoost / MLP / logreg stage-2 models
    ├── full_pipeline_v1.py       test-set run: retrieval + reranker → submission
    ├── full_pipeline_v2.py       test-set run: + stage-2 ensemble
    ├── postprocess_v2.py         fix / merge / France-specific postprocessing variants
    ├── validate_submission.py    official submission-format validator
    └── analysis/                 report builders and submission-comparison / France scripts

../../output/
├── matching_results.tsv          final matches (this package's deliverable)
└── candidate_pairs.tsv           regenerated by the retrieval stage — see note below
```

## Setup

```bash
pip install -r requirements.txt
```

Place the challenge kit at `student_resource/dataset/{train,test}/...` next to this folder (not
included in this package — it's the organiser-provided data).

## Reproducing end-to-end

```bash
python src/preprocess.py                                   # normalisation → dataset_processed/
python src/retrieval_experiments.py                         # base-model retrieval + recall eval
python src/finetune_e5.py mine  && python src/finetune_e5.py train && python src/finetune_e5.py eval
python src/train_reranker.py cands_ft && python src/train_reranker.py train \
    && python src/train_reranker.py predict && python src/train_reranker.py eval
python src/stack_lgbm.py features && python src/stack_models.py
python src/full_pipeline_v1.py --gpus 0,1,2,3,4,5,6,7        # retrieval + reranker → submission
python src/full_pipeline_v2.py all                            # + stage-2 ensemble
python src/postprocess_v2.py france                           # France-specific rules
python src/validate_submission.py --matching ../../output/matching_results.tsv \
    --candidate ../../output/candidate_pairs.tsv --test-dir student_resource/dataset/test
```

Each stage caches its output and is safe to re-run independently once its inputs exist.

**Note on `candidate_pairs.tsv`:** this package ships the final `matching_results.tsv`.
`candidate_pairs.tsv` — the top-20-per-S1 candidate set produced by the retrieval stage and fed
directly into the reranker/stage-2 model, with nothing narrower computed afterward — is regenerated
by `retrieval_experiments.py` / `full_pipeline_v1.py` above, since it's an intermediate artifact
rather than a persisted standalone file in this snapshot.

## Compute and runtime

Developed and tested on 8x NVIDIA V100 32GB, 40 CPUs, 500GB RAM, Python 3.11, CUDA 12.

| Stage | Approx. time |
|---|---|
| Preprocessing (both train and test) | ~30 min, 40 CPUs |
| Retrieval + reranking, full test set (1.73M S1 × 9.97M S2/S3) | ~61 min, 8 GPUs |
| Stage-2 feature engineering + model training | ~16 min |

Fine-tuned model weights and stage-2 model files are not bundled in this package; every script
retrains or regenerates what it needs from the challenge data. Large intermediates (embeddings,
candidate lists, logits, features — roughly 80GB total) are cached locally and are not required to
be kept between runs.

## Constraints compliance

- **Models:** `intfloat/e5-large` (MIT, ~560M params) and `microsoft/mdeberta-v3-base` (MIT,
  ~280M params) are the only neural models; the stage-2 trees (LightGBM/XGBoost/CatBoost) are
  lightweight and open-source. Well under the 8B-parameter cap.
- **No external data or lookups:** every normalisation table (abbreviations, Indic transliteration
  map, region codes, France descriptor-word list) is learned solely from the provided training
  data — no external APIs, geocoding, or business-registry lookups anywhere in the pipeline.
- **Format:** both output files pass `validate_submission.py` — one row per test S1, empty list for
  predicted singletons, no duplicate IDs, and every matched ID present in that S1's candidate list.

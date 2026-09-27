# Amazon ML Challenge 2026 — Business Entity Resolution
## Methodology Document

**Team:** tokenmaxxing
**Overall public score (F0.5):** 0.98536 (rank 508)

| Player | Organisation |
|---|---|
| Prathamesh Surendra Fuke | Marathwada Mitra Mandal's College of Engineering (MMCOE), Pune |
| Aadi Patil | Marathwada Mitra Mandal's College of Engineering (MMCOE), Pune |
| Bhushan Amol Anokar | Marathwada Mitra Mandal's College of Engineering (MMCOE), Pune |
| Jalaja Utekar | Marathwada Mitra Mandal's College of Engineering (MMCOE), Pune |

---

## 1. Problem

For every Source-1 (S1) business record, find every Source-2 / Source-3 record describing the same
real-world business, across noisy names and addresses in three countries (India and US in train,
France appearing only in test). Scored by macro F0.5 per S1 entity — singletons count, and false
merges are penalised twice as heavily as misses.

## 2. Data and preprocessing

- Train: 2.21M S1, 5.03M S2, 5.29M S3. Test: 1.73M S1 (India 810k, US 663k, France 259k), plus
  9.97M S2/S3.
- Every S2/S3 record belongs to at most one S1; 5.6% of S1 are singletons; country agrees in 100%
  of true pairs, so blocking by country loses no recall.
- Preprocessing: Unicode/case normalisation, abbreviation expansion (Rd→Road, St→Street/Saint by
  position, Pvt→Private, Corp→Corporation, …), an Indic→Latin transliteration map for India-source
  names learned from the training pairs, US/India state and region codes plus French département→
  région mapping, legal-form/honorific stripping, and house/unit-number parsing. A combined
  `name_core | addr_clean` text field feeds the embedding models.
- Hard negatives for retrieval fine-tuning are generated synthetically: house number off by a small
  offset, and/or a changed legal suffix, so the model learns to separate near-duplicate businesses.

## 3. Candidate generation / blocking strategy

1. **Country as an exact block.** Since country agrees on 100% of true pairs, S2/S3 candidates are
   restricted to the same country as the S1 query before any embedding search — a large,
   loss-free reduction in the search space.
2. **Dense retrieval.** An e5-large sentence embedder, contrastively fine-tuned on the training
   pairs with the mined hard negatives above, embeds `name_core | addr_clean` for every record.
   A FAISS index per country retrieves the top-20 S2/S3 neighbours for each S1.
3. **Recall check.** Base e5-large already reaches pair recall@100 of 98.2%; after fine-tuning,
   recall@20 reaches 99.9% (recall@50 is 100.0%), so top-20 candidates are kept as the final
   candidate set fed to the matching stage — a small, near-lossless list per S1 entity.

This candidate set is what `candidate_pairs.tsv` records: at most 20 S2/S3 IDs per S1, all within
the same country.

## 4. Matching model and feature engineering

- **Cross-encoder reranker:** mDeBERTa-v3-base scores each `(S1, candidate)` pair as a
  "name | address" text pair (yes/no this is the same business). Its pair AUC (0.995) is slightly
  below the fine-tuned cosine similarity's, but it ranks the correct S1 first far more reliably —
  the main weakness is over-confidence on namesake businesses.
- **Assignment rule:** each S2/S3 record is given to at most one S1 (its best-scoring candidate).
  An **abstain rule** — if two or more S1s pass the acceptance threshold for the same record, it
  is given to none — removes most of the remaining false merges caused by namesakes and lifts
  macro F0.5 from 0.983 to 0.991 on the held-out validation split.
- **Stage-2 decision ensemble:** on top of the reranker, a second model layer (LightGBM + XGBoost +
  CatBoost, 5-fold out-of-fold) is trained on 50 engineered features per pair: the reranker score
  and logit, the fine-tuned cosine similarity, the reranker's margin over the *next-best* S1
  competing for the same record (the single most informative feature, ~63% of LightGBM's gain),
  each candidate's rank and score gap within its S1's list, and structured comparisons of house
  number, legal suffix, name tokens, and address tokens between the two records. This model learns
  when a contested record is still safe to assign, recovering recall that the abstain rule gives
  up, without new false merges — macro F0.5 reaches 0.994 on hold-out.
- **France-specific decision rules:** France is absent from training data and its business names
  follow a distinct template (core name + a generic descriptor word, e.g. "club/comité/lycée/
  amis/union/fédération" + legal form). The main France-specific failure mode is a **descriptor
  swap** — two otherwise-identical records (same address, same core name) that differ only in that
  one generic word, which the base model tends to over-merge. A manual inspection of disagreements
  against reference-quality baselines shaped a small set of stricter, France-only checks in the
  final decision stage (candidates with an empty address are treated more cautiously everywhere,
  since 91% of false positives and 82% of misses in error analysis involve an empty address on one
  side of the pair).

## 5. Validation results

80/20 split of the training ground truth by S1 (441,200 held-out S1 queries; the retriever and
reranker never trained on this split):

| Stage | Metric | Value |
|---|---|---|
| Retrieval, base e5-large (name+address) | pair recall @10 / @100 | 96.7 / 98.2 |
| Retrieval, fine-tuned e5-large | pair recall @10 / @20 / @50 | 99.8 / 99.9 / 100.0 |
| Cosine threshold + one-S1-per-record | macro F0.5 | 0.9583 |
| Reranker + one-S1-per-record | macro F0.5 | 0.9831 |
| Reranker + one-S1-per-record + abstain | macro F0.5 | 0.9914 |
| Stage-2 ensemble, 50 features (deployed) | macro F0.5 | 0.9939 |

**Error analysis** (reranker + assignment, threshold 0.5): of the F0.5 lost, 52.7% is non-singleton
false merges, 26.8% is missed matches, 15.7% is a false merge on a true singleton, and 4.8% is both.
Only 1,146 of the ~24k missed pairs were never retrieved at all — most of the remaining error is a
decision problem, not a recall problem, which motivated the stage-2 model and the abstain rule.

## 6. Constraints compliance

- **Model license/size:** e5-large (MIT, ~560M parameters) and mDeBERTa-v3-base (MIT, ~280M
  parameters); the stage-2 decision models (LightGBM/XGBoost/CatBoost) are lightweight and
  open-source. All well under the 8B-parameter cap.
- **No external data or lookups:** every normalisation table (abbreviations, Indic transliteration
  map, state/région codes) is derived from the provided training data only; no external APIs,
  geocoding services, or business registries are used anywhere in the pipeline.
- **Output format:** both `matching_results.tsv` and `candidate_pairs.tsv` pass the official
  `validate_submission.py` checks — one row per test S1 entity, empty list for predicted
  singletons, no duplicate IDs, and every matched ID drawn from the candidate set.

## 7. Reproducing

See `code/business_entity_resolution/README.md` for the exact run order, environment, and hardware
used (8x V100 32GB, Python 3.11, CUDA 12), and `code/business_entity_resolution/requirements.txt`
for pinned dependencies.

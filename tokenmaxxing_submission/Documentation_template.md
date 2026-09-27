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

## Table of contents

1. Problem framing
2. Data and preprocessing
3. Candidate generation / blocking strategy
4. Matching model and feature engineering
5. Decision policy
6. Validation results
7. Error analysis
8. Cross-checking against other systems, and the France investigation
9. Constraints compliance
10. Limitations and possible next steps
11. Reproducing

---

## 1. Problem framing

For every Source-1 (S1) business record, find every Source-2 / Source-3 record that describes the
same real-world business. S1 is the deduplicated reference source; a given S1 may have zero, one,
or many true matches in S2/S3, and every S2/S3 record belongs to at most one S1. Names and
addresses are noisy in both directions: abbreviations, legal-suffix inconsistencies, transliterated
scripts, missing address components, landmark references, and reordered fields.

The metric is **macro F0.5, averaged per S1 entity**, which shapes the whole design:

```
F0.5 = 1.25 · P · R / (0.25 · P + R)
```

Because precision is weighted 2x over recall, a wrong merge costs more than a missed one. Singletons
are scored too: an S1 with no true match gets 1.0 for predicting nothing, and 0.0 for predicting
anything. This means the system has to be as good at *refusing* a match as it is at *finding* one,
and it directly motivated the abstain rule and the two-stage decision architecture described below,
rather than optimising a single similarity threshold.

## 2. Data and preprocessing

**Scale.** Train: 2.21M S1, 5.03M S2, 5.29M S3, with ground-truth matches. Test: 1.73M S1
(India 810k, US 663k, **France 259k** — a country never seen in training), plus 9.97M S2/S3 records
to search over.

**Structural facts used to design the pipeline** (measured on train):
- Every S2/S3 record belongs to **at most one S1** — so the final decision stage can treat "which S1
  gets this record" as a competitive assignment, not an independent yes/no per pair.
- **5.6% of S1 are singletons** — frequent enough that getting singletons right materially moves the
  macro average, and rare enough that a system which merges too eagerly will be punished for it.
- **Country agrees on 100% of true pairs.** This makes country an exact, loss-free block: no
  cross-country candidate search is needed anywhere in the pipeline.
- **Indic script names.** 24% (S2) and 13% (S3) of India-source names are written in Indic script.
  The Indic vocabulary observed is small and closed (~170 words per script) and every word appears
  in train, so a learned transliteration lookup (rather than a generic transliteration library) is
  sufficient and exact.

**Normalisation pipeline** (`src/preprocess.py`), applied identically to all three sources:
- Unicode normalisation (NFKC) and case folding.
- **Abbreviation expansion**, position-aware where the expansion is ambiguous — e.g. `St` expands to
  `Street` mid-address but `Saint` at the start of a name; `Rd`→`Road`, `Pvt`→`Private`,
  `Corp`→`Corporation`, `Ltd`→`Limited`, and similar legal/administrative abbreviations.
  `&` and `and` are normalised to the same token.
- **Indic→Latin transliteration**, using a word-level map (1,347 entries) learned directly from
  cases in the training data where the same business appears in both Latin and Indic script across
  sources — not a general transliteration model, and not an external lookup.
- **Region/administrative normalisation:** US and Indian state codes, plus a French
  département→région mapping, so a business's regional descriptor is comparable regardless of which
  granularity the source used.
- **Legal-form and honorific stripping** into a separate `legal_form` field (Inc, LLC, Pvt Ltd, SAS,
  SARL, …), leaving a `name_core` that two records can be compared on without punishing legitimate
  legal-suffix variation.
- **House/unit/number parsing:** house number, unit/suite, and any other numeric tokens in the
  address are extracted into structured fields (`house_no`, `house_base`, `unit`, `addr_numbers`),
  handling formats like `N°36`, `Nº 20`, `#3`, `NO 47`, and zero-padded numbers like `0131`.
- The cleaned `name_core` and `addr_clean` are concatenated into a single `embed_text =
  name_core | addr_clean` field, which is what the embedding and reranking models actually see.

**Hard negatives for retriever fine-tuning** are generated synthetically rather than mined only from
random in-batch negatives: for a true pair, a near-duplicate negative is built by shifting the house
number by a small, specific offset (see the fingerprint set in §7) and/or substituting the legal
suffix, so the retriever is explicitly pushed to separate "almost the same business" pairs, which is
exactly the kind of confusion namesakes and clerical near-duplicates create in test.

## 3. Candidate generation / blocking strategy

Blocking has two stages, applied in order, and the design goal throughout is a **small candidate set
per S1 with near-zero recall loss** (the challenge explicitly ranks smaller candidate sets higher,
beyond the leaderboard score).

1. **Country as an exact block.** Since country agrees on 100% of true pairs (measured on train, and
   assumed to generalise — including to the unseen France split, where the label is still present
   even though no France records exist in train), the search space for each S1 is restricted to
   S2/S3 records with the same country label *before* any embedding computation. This alone cuts the
   average search space by roughly a factor equal to the number of countries, for free.
2. **Dense retrieval within country.** A sentence embedder (e5-large, see §4) encodes `embed_text`
   for every S1 and every S2/S3 record. A FAISS index is built per country over the S2/S3 embeddings,
   and the **top-20** nearest neighbours are retrieved for every S1.
3. **Recall-driven choice of K=20.** K is not arbitrary: recall was measured at @10/@20/@50/@100 on
   an 80/20 held-out split of train (§6). The base, off-the-shelf e5-large already reaches
   recall@100 = 98.2%. After contrastive fine-tuning with the hard negatives from §2, recall@20
   reaches 99.9% and recall@50 reaches 100.0% — so cutting the candidate list at 20 loses
   essentially no true matches (candidate pair recall of 99.92% end-to-end) while keeping the set an
   order of magnitude smaller than a naive top-100 cut would need.

This is exactly what `candidate_pairs.tsv` records: for each S1, up to 20 S2/S3 IDs, all sharing its
country, ranked by fine-tuned cosine similarity. It is the same list that is fed into the reranker
and the stage-2 model in §4 — nothing narrower is computed before matching, so
`candidate_pairs.tsv` is the true final candidate set, not an earlier, looser stage.

## 4. Matching model and feature engineering

Matching is a two-stage decision on top of the candidates from §3: a cross-encoder reranker scores
each pair, and a second, tabular model turns those scores (plus contextual and structured features)
into the final accept/reject decision. Splitting scoring from decision this way lets the decision
stage use *relative* information — how a candidate compares with its competitors — that a
per-pair-only score cannot see.

### 4.1 Retriever fine-tuning

- **Base model:** `intfloat/e5-large` (MIT license, XLM-RoBERTa-large backbone, ~560M parameters,
  multilingual — needed for the Indic-script and French-diacritic text).
- **Objective:** contrastive fine-tuning (in-batch negatives + the mined hard negatives from §2),
  temperature 0.02.
- **Training:** 1 epoch, batch size 128 (global, across 8 GPUs via DDP), AdamW, learning rate
  `2e-5` with linear warmup then linear decay to 0, weight decay 0.01.
- **Effect:** pair recall@10 on the held-out split rises from 70.3% (name-only, base e5-large) /
  96.7% (name+address, base e5-large) to 99.8% (name+address, fine-tuned) — the largest single
  improvement in the whole pipeline, and the reason retrieval is not the pipeline's bottleneck (see
  §7).

### 4.2 Cross-encoder reranker

- **Base model:** `microsoft/mdeberta-v3-base` (MIT license, ~280M parameters, multilingual).
- **Input:** the pair's two `name | address`-style texts, encoded jointly (`tokenizer(text_a,
  text_b)`) so the model can attend across the two records directly, rather than only comparing
  fixed embeddings.
- **Training:** 1 epoch (capped at 8,000 steps) over candidate pairs built from the fine-tuned
  retriever's top-K, batch size 64, AdamW, learning rate `3e-5` with linear warmup/decay, weight
  decay 0.01. Binary cross-entropy on the true/false match label.
- **Why a cross-encoder in addition to embeddings:** the reranker's raw pairwise AUC (0.995) is
  *slightly below* the fine-tuned embedding cosine's, but after assignment (§5) it outperforms the
  cosine by a wide margin (macro F0.5 0.983 vs 0.958 at threshold-only decision). It is much better
  at ranking the *correct* S1 first for a given record; its main weakness is over-confidence on
  namesake businesses, which the decision stage is specifically designed to catch.

### 4.3 Stage-2 decision ensemble

Rather than deciding directly on the reranker's score, a second model is trained on **50 deployed
features per (S1, candidate) pair** (with a 59-feature superset used during offline evaluation — see
§4.4), grouped into four families:

- **Model outputs:** the reranker probability/logit (from one or two independently trained reranker
  runs), and the fine-tuned e5 cosine similarity.
- **List context (within one S1's candidate list):** the candidate's retrieval rank; the score gap
  between this candidate and the best-scoring candidate for the same S1.
- **Cross-S1 competition (for one candidate record):** the reranker's margin over the *next-best S1*
  also claiming this same S2/S3 record — the single most informative feature, responsible for about
  **63% of LightGBM's total gain** — plus how many S1s pass a positive-score threshold for this
  record (`cand_n_s1_pos`), and how many candidates pass threshold for this S1 (`s1_n_pos`).
- **Structured field comparisons:** house-number equality/difference (exact, and a specific-offset
  "fingerprint" match, see §7); unit equality; legal-form equality/overlap; name-core and
  address-core exact-match and string-similarity features (edit distance / token overlap style);
  numeric-token Jaccard overlap between the two addresses; whether either side's address field is
  empty; name length difference; and name/candidate frequency in the corpus (how common that
  business name is, as an ambiguity signal).

**Models:** LightGBM, XGBoost and CatBoost, trained with 5-fold cross-validation grouped by S1 (so
no S1's candidates leak between folds), evaluated out-of-fold. A logistic regression and an MLP were
also trained as baselines. Ensembling the tree models adds nothing measurable over the single best
tree model (§6), so the deployed model is the single best-performing gradient-boosted tree model at
inference time, keeping the pipeline simpler without giving up score.

**What the stage-2 model buys:** it recovers most of the recall the abstain rule (§5) discards on
genuinely-contested records, by learning *when* a contested record is still safe to assign (e.g. one
S1 is a clear winner by a wide competition margin) versus when it truly is ambiguous — without
reintroducing the false merges the abstain rule was added to prevent.

### 4.4 Feature-set trade-off

The full 59-feature set (used for offline model comparison) includes the cosine similarity from
every base embedding model tried during retrieval experiments (6 models) plus two name-frequency
features. The **deployed 50-feature set** drops these: the 6 extra cosines would require embedding
all 11.7M test records with 6 different models purely to compute a feature, and the frequency
features scale with corpus size, so they're less stable between train and test. This trade costs
about 0.0003 macro F0.5 on the same hold-out fold (0.9942 → 0.9939) in exchange for a much cheaper
and simpler inference pipeline.

## 5. Decision policy

Two decision rules are layered on top of the model scores, both driven directly by the F0.5 metric's
precision weighting:

1. **One S1 per record.** Since every S2/S3 record can truly belong to at most one S1, each record
   is assigned to at most one S1: its highest-scoring candidate above threshold.
2. **Abstain rule.** If two or more S1s pass the acceptance threshold for the *same* record, the
   record is given to **none** of them, rather than to the highest scorer. Because a wrong merge
   costs 2x what a miss does under F0.5, refusing a genuinely ambiguous record is the higher-expected-
   value choice. This single rule lifts macro F0.5 from 0.9831 (reranker + one-S1-per-record) to
   0.9914 on hold-out — the second-largest single gain in the pipeline, after the retriever
   fine-tuning.
3. **Threshold search.** The final acceptance threshold (and the France-specific thresholds in §8)
   is chosen by grid search directly against the macro F0.5 metric on the hold-out fold, not by a
   generic operating point like 0.5.

## 6. Validation results

All numbers below are on an **80/20 split of the training ground truth by S1** (seed 42): 441,200
held-out S1 queries, against a corpus of 2,061,938 S2/S3 records (every true match for the held-out
S1s, plus 20% of the distractor records). Neither the retriever nor the reranker ever trained on
this split.

**Retrieval (pair recall %, within country):**

| Model | Input | @10 | @20 | @50 | @100 |
|---|---|---|---|---|---|
| e5-base | name only | 70.0 | 76.7 | 86.2 | 89.4 |
| e5-large | name only | 70.3 | 77.1 | 86.5 | 89.7 |
| bge-m3 | name only | 69.4 | 76.1 | 85.6 | 89.0 |
| e5-base | name + address | 96.2 | 96.9 | 97.5 | 97.9 |
| e5-large | name + address | 96.7 | 97.3 | 97.9 | 98.2 |
| bge-m3 | name + address | 96.4 | 97.0 | 97.5 | 97.8 |
| **e5-large, fine-tuned** | name + address | **99.8** | **99.9** | **100.0** | **100.0** |

The address field carries far more discriminating signal than the name alone (a ~27-point recall
jump at every cut-off), and fine-tuning closes almost all of the remaining gap.

**Decision stage (macro F0.5):**

| Stage | Precision | Recall | Macro F0.5 |
|---|---|---|---|
| Cosine (fine-tuned e5), threshold + one-S1-per-record | — | — | 0.9583 |
| Reranker, threshold-only | — | — | 0.8468 |
| Reranker, threshold + one-S1-per-record | — | — | 0.9831 |
| Reranker + one-S1-per-record + **abstain**, threshold 0.7 | — | — | **0.9914** |
| Stage-2, logistic regression, 59 features (out-of-fold) | 0.9946 | 0.9706 | 0.9866 |
| Stage-2, MLP, 59 features (out-of-fold) | 0.9988 | 0.9837 | 0.9940 |
| Stage-2, CatBoost, 59 features (out-of-fold) | 0.9992 | 0.9831 | 0.9941 |
| Stage-2, **LightGBM**, 59 features (out-of-fold) | 0.9988 | 0.9843 | **0.9942** |
| Stage-2, **XGBoost**, 59 features (out-of-fold) | 0.9992 | 0.9833 | **0.9942** |
| Stage-2, tree ensembles (mean / weighted) | 0.9989 | 0.9842 | 0.9942 |
| Stage-2, **deployed 50-feature set** (hold-out fold) | — | — | 0.9939 |

Note how the reranker alone (before any assignment logic) is *worse* on the pairwise-AUC-style
threshold decision (0.8468) than after the one-S1-per-record rule (0.9831) — the model's pairwise
scores are informative for ranking but not well calibrated as independent yes/no decisions, which is
exactly why decision logic on top of the raw score matters as much as the score itself here.

## 7. Error analysis

On the reranker + assignment configuration (threshold 0.5, before the stage-2 model), the F0.5 lost
on hold-out breaks down as:

| Error type | Share of lost F0.5 |
|---|---|
| Non-singleton S1s with false merges only | 52.7% |
| Non-singleton S1s with misses only | 26.8% |
| True singletons given a false merge | 15.7% |
| Both false merges and misses | 4.8% |

In absolute terms: 22,977 false positives and 24,038 missed pairs on 441,200 held-out S1s. Critically,
**only 1,146 of the missed pairs were never retrieved at all** — i.e. retrieval recall is not the
bottleneck; almost every miss is a decision the matching stage got wrong on a pair it did have
available, which is what motivated moving from a single reranker threshold to the stage-2 model.

**The largest single driver of both false positives and misses is an empty address on one side of
the pair:** candidates with no address information account for **91% of false positives and 82% of
misses**. A name-only record cannot reliably be told apart from a namesake business. If this failure
mode were solved perfectly, the estimated ceiling on this hold-out fold would be 0.9979 — the single
largest remaining opportunity in the pipeline (see §10).

**House-number fingerprints.** The hard-negative generation in §2 uses specific house-number offsets
(1, 2, 3, 4, 5, 7, 9, 11) chosen because these are the offsets that actually recur in clerical
near-duplicates in the data; a `house_fingerprint` feature flags pairs whose house-number difference
matches this set, and it is a useful discriminating signal precisely because it targets a
*specific, recurring* error pattern rather than "any numeric difference."

## 8. Cross-checking against other systems, and the France investigation

To sanity-check the pipeline beyond the held-out split, its test-set predictions were compared
against two independently produced result files that scored 0.986149 and 0.987337 on the public
leaderboard.

**Overall agreement.** Pair-level Jaccard similarity across all three systems is about 0.98. India
and US predictions are nearly identical (~3.39 matches per S1 on average, across systems) — the
disagreement is almost entirely concentrated in **France**, the country absent from training:

| Matches per S1 in France | |
|---|---|
| 0.987337-scoring system | 3.16 |
| 0.986149-scoring system | 3.24 |
| this pipeline, pre-France-rules | 3.21–3.35 |

The higher-scoring external result matched *less* in France, which was the signal that this
pipeline's France behaviour needed a closer look rather than assuming "more matches is safer."

**Manual inspection.** France pairs were grouped by which systems accepted them, and a sample from
each group was read by hand. This surfaced four concrete failure/behaviour patterns, in order of
impact:

1. **Descriptor swap — the dominant France-specific failure.** French business names follow a
   template of `core name + a generic descriptor word (club, comité, lycée, amis, union,
   fédération, …) + legal form`. False candidates keep the same core name and the same address, and
   swap only that one generic descriptor word — e.g. *"House Lycée SAS"* vs *"House Section SAS"*,
   both at the same address. This pattern is only 1.2% of pairs every system agrees on, but
   accounts for **50–88%** of the pairs a more cautious system rejects.
2. **Differing house numbers accepted in France more often than elsewhere.** This is a decision
   issue, not a parsing bug — formats like `N°36`, `Nº 20`, `#3`, `NO 47`, and zero-padded numbers
   like `0131` all parse correctly; the model is simply too lenient about small house-number
   differences specifically in France, where it has no training examples to calibrate against.
3. **Completely different names at the same address are usually genuine in France** — brand names,
   initials (e.g. *"Raid Compagnie SAS"* ↔ *"RC"*), and domain-style names. These should be kept,
   not filtered, so any France-specific fix has to be surgical rather than a blanket stricter name
   check.
4. **The remaining recall gap is empty-address namesakes and initials** — the same root cause as §7,
   compounded in France by initialism-heavy names (e.g. *"LC"* for *"Lille Compagnie"*), where the
   stage-2 model sometimes overrules a confident reranker score.

**France-specific decision rules deployed (`postprocess_v2.py france`, applied to France pairs
only):**
- **Reject descriptor swaps.** A list of the 114 most frequent name words among France S1 records is
  used to detect a single-word swap between two otherwise-matching names; known synonym pairs
  (cie/compagnie, ets/établissements, centre/center) are explicitly excluded so genuine synonym
  variation is not penalised.
- **Raise the bar on differing house numbers.** When both records have a parsed house number and
  they differ, the pair is only kept if the model's probability exceeds 0.98 (versus the general
  threshold), directly targeting failure pattern (2) without touching pattern (3).
- **Measured effect:** these two rules together remove 37,100 descriptor-swap pairs and 19,000
  differing-house pairs, bringing France down to 3.18 matches per S1 (from 3.35 pre-rules) — closer
  to the higher-scoring external system's 3.16.
- **Agreement improved measurably:** France-only pair Jaccard with the 0.987337-scoring system rose
  to 0.929 with these rules, versus 0.906 without them and 0.904 for the other external system —
  the highest agreement of any variant tested, used as independent (non-label) evidence that the
  rules were net corrections rather than new noise.

This is the version submitted (internally referred to as v3_france in development).

## 9. Constraints compliance

- **Model license and size:** `intfloat/e5-large` (MIT, ~560M parameters) and
  `microsoft/mdeberta-v3-base` (MIT, ~280M parameters) are the only neural models in the pipeline;
  the stage-2 decision models (LightGBM, XGBoost, CatBoost — all open-source) are lightweight
  gradient-boosted trees, not counted against the parameter cap. Total neural parameter count is
  well under the 8B limit.
- **No external data or lookups.** Every normalisation table used — abbreviation expansions, the
  Indic→Latin transliteration map, US/India region codes, the French département→région mapping, and
  the France descriptor-word list — is derived solely from patterns observed in the provided
  training data. No external APIs, geocoding services, business registries, or internet lookups are
  used anywhere in preprocessing, retrieval, matching, or decision logic.
- **Output format.** Both `matching_results.tsv` and `candidate_pairs.tsv` are checked against the
  official `validate_submission.py`: one row per test S1 entity, an empty list for predicted
  singletons, no duplicate entity IDs within a list or across rows, and every ID in
  `matching_results.tsv` drawn from that S1's row in `candidate_pairs.tsv`.

## 10. Limitations and possible next steps

- **Empty-address candidates are the largest remaining error source** (§7): a dedicated name-only
  retrieval path plus a namesake-aware decision rule for these specific candidates is the highest-
  leverage next step, since a correct fix here alone would close most of the gap to a near-perfect
  score on the hold-out fold.
- **France-specific hard negatives.** The France rules in §8 are hand-derived from manual
  inspection. A more systematic fix would synthesize France-flavoured hard negatives (descriptor
  swaps, small house-number shifts on French addresses) and fine-tune the reranker on them directly,
  rather than patching the decision stage after the fact.
- **Ensembling across independently-built systems.** The cross-checking work in §8 suggests a
  majority-vote or agreement-weighted ensemble across multiple independently trained pipelines could
  further improve precision on genuinely ambiguous pairs, at the cost of needing to maintain more
  than one pipeline.

## 11. Reproducing

See `code/business_entity_resolution/README.md` for the exact run order, environment, and hardware
used (8x V100 32GB, 40 CPUs, 500 GB RAM; Python 3.11, CUDA 12), and
`code/business_entity_resolution/requirements.txt` for pinned dependencies. In summary: preprocess →
retrieval experiments → hard-negative mining and e5 fine-tuning → reranker candidate-building,
training, and prediction → stage-2 feature engineering and model training → full pipeline run on
test → France postprocessing → validation against `validate_submission.py`.

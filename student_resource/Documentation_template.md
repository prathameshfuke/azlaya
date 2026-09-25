# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [Your Team Name]  
**Team Members:** [List all team members]  
**Submission Date:** [Date]

---

## 1. Executive Summary
*Provide a brief 2-3 sentence overview of your approach and key innovations.*

---

## 2. Methodology

### 2.1 Problem Analysis
*Key insights discovered during EDA — noise patterns, address variations, missing fields, etc.*

### 2.2 Solution Strategy
*Outline your high-level approach.*

**Approach Type:** [Blocking + Classifier / End-to-End / Graph-Based / Hybrid, etc]  
**Core Innovation:** [Brief description of your main technical contribution]

---

## 3. Candidate Generation (Blocking)
*Describe how you reduced the comparison space to a manageable candidate set.*

- **Blocking keys used:** [e.g., PIN code, phonetic name encoding, TF-IDF, etc.]
- **Candidate pairs generated:** [total]
- **How you ensured true matches were not lost:**

---

## 4. Matching Model

**Features used:**
- Name features: [e.g., Jaccard, Levenshtein, phonetic encoding]
- Address features: [e.g., token overlap, edit distance, PIN code matching]
- Other: []

**Model type:** [e.g., XGBoost, Siamese Network, Transformer, etc.]  
**Threshold selection method:** [e.g., F_0.5 optimization on validation set]

---

## 5. Laya Integration

### 5.1 Why Laya, and where it sits in the pipeline

Section 4's GBDT scores every blocked candidate pair on hand-engineered string-similarity
features: normalized Levenshtein and Jaro-Winkler distance, word- and character-n-gram TF-IDF
cosine, and address/pin-code/city overlap (see `features.py`). Those features are all functions
of surface string form, which makes the GBDT blind to pairs where two records describe the same
business but don't look alike as strings: a DBA record where the legal name and trade name are
unrelated strings (`Ectolumdrex dba X+ Madison Inc`), a record where the business name is a
domain rather than a name (`wilfordhancock.com`), or a legal name that's been reordered or
paraphrased between sources. To catch these, we run a second, heterogeneous scorer in parallel
over the same blocking output: **Laya** (`NandhaKishorM/laya`), a small (~322M–421M parameter),
Apache-2.0-licensed, non-autoregressive text classifier, fine-tuned to answer the matching
question directly in natural language rather than through engineered features. Because it reads
the two records as text and reasons about the question itself, it can recognize the DBA relation,
the domain-as-name case, or a paraphrased legal name as the same semantic content even when the
GBDT's string metrics see two dissimilar strings. Laya does not replace the GBDT; its output is
folded back into the GBDT's feature table as one more signal (Section 5.5).

### 5.2 How a candidate pair is scored

Every candidate pair is scored with a single `noul` (yes/no) question, `same_entity`, built from
the pair's two normalized records (`src/laya_finetune.py`):

```python
questions = {
    "same_entity": {
        "type": "noul",
        "instructions": (
            "record_a and record_b each describe one business as \"name | address | country\", "
            "taken from two different, independently-collected data sources. Decide whether "
            "they refer to the SAME real-world business."
        ),
        "criteria": {
            "true": "The same business despite surface noise between the two sources: legal-suffix "
                    "abbreviations, typos, transliteration differences, word-order changes, a "
                    "DBA/trade name standing in for the same legal entity, a missing address "
                    "component, or a landmark-based address reference for the same location.",
            "false": "Different businesses: distinct legal entities even when the names look "
                     "similar, a coincidental name match, or a clearly different address.",
        },
    }
}
state = {"record_a": "<name> | <address> | <country>", "record_b": "<name> | <address> | <country>"}
```

`same_entity`'s `noul` answer is `P(true)`, i.e. the model's probability that the two records
refer to the same business. Rather than scoring pairs one at a time, `ensemble.score_with_laya`
builds this state/question pair for every row of the candidate feature table and hands the whole
batch to `Router.predict_batch(requests, batch_size=...)`, which groups requests by checkpoint and
shares forward passes across them — the entire `candidate_pairs.tsv` set for a split is scored in
this one batched call rather than per-pair inference calls.

### 5.3 Fine-tuning approach

The Laya checkpoints (`convaiinnovations/laya`, `convaiinnovations/laya-multilingual`) are
general-purpose System-1 decision models, not trained on this task's records or vocabulary; the
project's working assumption is that they need fine-tuning to be useful here rather than being
applied zero-shot (this has not been measured, since nothing in this pipeline has been executed —
see the caveat in Section 5.6). Fine-tuning data is built entirely from our own pipeline's
intermediate output, not any external dataset: `laya_finetune.build_examples` pairs each S1
entity's blocking candidates (`candidate_pairs_train.tsv`) with the training ground truth
(`train_ground_truth.tsv`, via `common.ground_truth_map`). A pair is a **positive** example when
the candidate is in that S1 entity's `matched_entity_ids`, and a **negative** ("hard negative")
example when it survived blocking but is not a true match — i.e. it's a plausible-looking
non-match rather than a random, easily-distinguished pair, which is what makes it a useful
training signal. Every pair is duplicated with `record_a`/`record_b` swapped
(`make_example(rec_a, rec_b, ...)` and `make_example(cand_rec, s1_rec, ...)`) so the model isn't
implicitly trained to expect the S1 record first.

Training follows Laya's own RLCD (reinforcement learning against strictly proper scoring rules)
fine-tuning recipe, reproduced from the upstream repository's fine-tuning notebook: a noisy-logit
policy-gradient objective scored with `laya.common.proper_reward`, combined with a soft
cross-entropy term, optimized over several epochs (`stage_train` in `laya_finetune.py`). After
training, calibration temperatures are refit with `_fit_one_temp` on a calibration slice carved
out of **our own** training split — specifically, a further stratified split of the S1 entities
not held out for GBDT/ensemble validation (`--calib-frac`, disjoint from both the val split and
from any of Laya's own benchmark defaults) — so the model's reported confidence is calibrated to
this task's label distribution, not a general-purpose benchmark's.

### 5.4 Multilingual routing

Training data is split by script before fine-tuning (`route_by_script` in `laya_finetune.py`):
the `english` checkpoint is fine-tuned only on pairs where both records' business names are
Latin-script, while the `multilingual` checkpoint is fine-tuned on the full pair set, including
India's Devanagari- and Tamil-script business names (and any mixed-script record, since a record
that's part Latin and part Devanagari/Tamil doesn't route cleanly to either checkpoint alone at
train time). At inference, `build_router` assembles both fine-tuned checkpoints into a
`laya.Router`, which detects each request's script/language and dispatches non-English text to
the `multilingual` checkpoint automatically — this is how mixed-script Indian business names are
routed without any script-handling logic of our own beyond selecting which checkpoint each
training example belongs to.

### 5.5 Combining with the GBDT (stacking, not replacement)

Laya is not trusted standalone as the final decision-maker for a precision-heavy metric like
F_0.5: `noul` questions are known to be sensitive to how the criteria are phrased and can be
biased toward one answer on ambiguous inputs, and the `multilingual` checkpoint's fine-tuning data
never includes a France record (Section 5.4's routing is script-based, and France's business
names are Latin-script, so they train the `english` checkpoint — France is simply never a
fine-tuning example for the model that would otherwise be expected to generalize best to it). For
both reasons, Laya's output is stacked with the GBDT rather than substituted for it
(`ensemble.py`): `score_with_laya` produces `laya_prob` for every candidate pair, and
`build_stack_frame` appends it — alongside the GBDT's own `gbdt_prob` — as two extra columns onto
the GBDT's original feature table. A new meta-classifier is then trained on this combined table:
`train_logistic_stacker` (a scaled `LogisticRegression`, the primary/interpretable option) and
`train_shallow_gbdt_stacker` (a shallow LightGBM model, `num_leaves=7`, `max_depth=3`, kept as a
comparison alternative). Finally, `best_threshold` sweeps the decision threshold on the
**ensemble's own** output specifically, rather than reusing the plain GBDT's threshold from
Section 4 — the two models' probability distributions aren't the same, so their F_0.5-optimal
cutoffs aren't expected to match.

### 5.6 Evaluation discipline

Before trusting the Laya-augmented ensemble over the GBDT-only baseline, the two must be compared
on the same held-out validation split: `ensemble.py`'s `breakdown_report` computes macro F_0.5 on
that split broken down by country — with particular scrutiny on France, since (Section 5.5) no
France record ever appears in Laya's fine-tuning or calibration data, so its behavior there is
pure zero-shot generalization — and separately for singleton entities specifically, since under
F_0.5 correctly predicting "no match" is worth full credit (1.0) and a false merge on a singleton
scores 0.0. **As of this write-up, that comparison has not been run**: this entire integration was
authored and only statically checked (no GPU was available in the authoring environment), so
`ensemble_val_report.json` (Laya + stacking) has not yet been compared against
`gbdt_val_report.json` (GBDT-only). This section documents the comparison the pipeline is built
to make, not a result — the actual comparison, and whichever of `logistic`/`gbdt_alt` wins it, is
outstanding work before this can be called an improvement rather than an untested hypothesis.

### 5.7 Fair-play note

Laya runs entirely locally on our own fine-tuned checkpoint — there is no call to a hosted
inference API, and no external data source is consulted at inference or fine-tuning time. The
base checkpoints (`convaiinnovations/laya`, 421M parameters; `convaiinnovations/laya-multilingual`,
322M parameters) are Apache-2.0 licensed and both well under the competition's 8B-parameter cap.
Fine-tuning data is derived exclusively from `train_ground_truth.tsv` and this pipeline's own
blocking output (`candidate_pairs_train.tsv`) — no business registry, geocoding service, or other
external lookup is used anywhere in this integration, keeping it inside the challenge's
prohibition on external data lookup.

---

## 6. Results & Error Analysis

- **F_0.5 Score (macro):** [your best validation score]
- **Common false positives (wrong merges):** [brief description]
- **Common false negatives (missed matches):** [brief description]

---

## 7. Conclusion
*Summarize your approach, key achievements, and lessons learned in 2-3 sentences.*

---

## Appendix

### A. Code Artefacts
*Your complete, runnable code ships in the submission zip under
`code/business_entity_resolution/` (all source in `src/`, with a `README.md` and
`requirements.txt`). Summarise its structure and the entry point(s) to reproduce
`output/matching_results.tsv` and `output/candidate_pairs.tsv` here.*

### B. Additional Results
*Include any additional charts, graphs, or detailed results.*

---

**Note:** Teams can modify sections according to their approach while maintaining clarity and technical depth.

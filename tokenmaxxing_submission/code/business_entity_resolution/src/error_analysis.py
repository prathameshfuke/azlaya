#!/usr/bin/env python3
"""
Error analysis of the reranker (after "one S1 per record" assignment) on the 20% test split.

Reports:
  1. where macro F0.5 is lost (singletons / entities with false merges / entities with misses)
  2. false positives: whose record is it, and which patterns it shows
  3. false negatives: not retrieved / taken by another S1 in assignment / low score, and patterns
  4. examples of each pattern

Usage:  python code/error_analysis.py [--thr 0.5]
Writes: experiments/reranker/error_analysis.json and prints a summary.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import retrieval_experiments as rx  # noqa: E402

WORK = os.path.join(rx.ROOT, "experiments", "reranker")
FIELDS = ["entity_id", "country", "name_core", "legal_form", "house_base", "addr_clean", "addr_empty",
          "is_indic", "is_domain", "has_dba"]


def load_fields():
    parts = [rx.read_tsv(os.path.join(rx.PROC, "train", f"train_source{s}.tsv"), FIELDS) for s in (1, 2, 3)]
    return pd.concat(parts, ignore_index=True).set_index("entity_id")


def tag(df, F):
    """Attach structured comparison features for (s1_id, cand_id) pairs."""
    a = F.loc[df.s1_id].reset_index(drop=True)
    b = F.loc[df.cand_id].reset_index(drop=True)
    out = pd.DataFrame(index=df.index)
    out["country"] = a.country.values
    out["cand_addr_empty"] = b.addr_empty.values == "1"
    out["cand_indic"] = b.is_indic.values == "1"
    out["cand_domain"] = b.is_domain.values == "1"
    out["cand_dba"] = b.has_dba.values == "1"
    hb_a, hb_b = a.house_base.values, b.house_base.values
    both = (hb_a != "") & (hb_b != "")
    out["house_eq"] = both & (hb_a == hb_b)
    diff = np.full(len(df), -1)
    m = both & np.array([x.isdigit() and y.isdigit() and len(x) < 10 and len(y) < 10 for x, y in zip(hb_a, hb_b)])
    diff[m] = np.abs(hb_a[m].astype(np.int64) - hb_b[m].astype(np.int64))
    out["house_diff"] = diff
    out["house_diff_fingerprint"] = np.isin(diff, [1, 2, 3, 4, 5, 7, 9, 11])
    out["legal_eq"] = a.legal_form.values == b.legal_form.values
    out["name_sim"] = [fuzz.token_set_ratio(x, y) for x, y in zip(a.name_core.values, b.name_core.values)]
    out["addr_sim"] = [fuzz.token_set_ratio(x, y) if y else -1 for x, y in zip(a.addr_clean.values, b.addr_clean.values)]
    return out


def pattern(r):
    """One label per error, checked in priority order."""
    if r.cand_addr_empty:
        return "candidate address empty"
    if r.house_diff_fingerprint:
        return "house number off by 1-5/7/9/11 (distractor fingerprint)"
    if r.house_diff > 0:
        return "house number differs (other)"
    if r.name_sim < 50:
        return "name unrelated (sim<50)"
    if not r.legal_eq:
        return "legal form differs"
    if r.cand_indic:
        return "Indic-script name"
    if r.cand_domain:
        return "domain / handle name"
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thr", type=float, default=0.5)
    a = ap.parse_args()
    df = pd.read_parquet(os.path.join(WORK, "test_pairs.parquet"))
    df["p"] = 1 / (1 + np.exp(-np.load(os.path.join(WORK, "test_logits.npy"))))
    _, gt_test = rx.make_split()
    tp_all = gt_test.assign(m=gt_test.matched_entity_ids.str.split(",")).explode("m")
    tp_all = tp_all[tp_all.m != ""][["source1_entity_id", "m"]]
    owner = dict(zip(tp_all.m, tp_all.source1_entity_id))
    n_true = tp_all.groupby("source1_entity_id").size()

    # predictions with assignment
    cand = df[df.p > a.thr].sort_values("p", ascending=False)
    kept = cand.drop_duplicates("cand_id")
    df["pred"] = False
    df.loc[kept.index, "pred"] = True
    df["stolen"] = False                           # passed threshold but assigned to another S1
    df.loc[cand.index.difference(kept.index), "stolen"] = True

    # ---- 1. where F0.5 is lost
    g = df.groupby("s1_id").agg(tp=("pred", lambda s: 0), npred=("pred", "sum"))
    g["tp"] = df[df.pred & (df.label == 1)].groupby("s1_id").size().reindex(g.index).fillna(0)
    g["nt"] = n_true.reindex(g.index).fillna(0)
    P = np.where(g.npred > 0, g.tp / g.npred.clip(lower=1), 0)
    R = np.where(g.nt > 0, g.tp / g.nt.clip(lower=1), 0)
    f = np.where((0.25 * P + R) > 0, 1.25 * P * R / np.maximum(0.25 * P + R, 1e-12), 0)
    g["f"] = np.where(g.nt == 0, (g.npred == 0).astype(float), f)
    g["fp"] = g.npred - g.tp
    g["fn"] = g.nt - g.tp
    lost = 1 - g.f
    total_lost = lost.sum()
    single = g.nt == 0
    buckets = {
        "singletons with a false merge": lost[single].sum(),
        "non-singletons: false merges only": lost[~single & (g.fp > 0) & (g.fn == 0)].sum(),
        "non-singletons: misses only": lost[~single & (g.fp == 0) & (g.fn > 0)].sum(),
        "non-singletons: both": lost[~single & (g.fp > 0) & (g.fn > 0)].sum(),
    }
    print(f"macro F0.5 = {g.f.mean():.4f}  (lost {total_lost / len(g):.4f} per entity on {len(g):,} S1)")
    print("where the loss comes from:")
    for k, v in buckets.items():
        print(f"  {k:40s} {v / total_lost:6.1%}  (F0.5 points {v / len(g) * 100:.2f})")
    print(f"  singletons: {single.sum():,}, of which predicted non-empty: {(single & (g.npred > 0)).sum():,} "
          f"({(single & (g.npred > 0)).sum() / single.sum():.1%})")
    for c in sorted(df.country.unique()) if "country" in df else []:
        pass

    F = load_fields()

    # ---- 2. false positives
    fp = df[df.pred & (df.label == 0)].copy()
    fp = pd.concat([fp, tag(fp, F)], axis=1)
    fp["whose"] = np.where(fp.cand_id.map(owner).notna(), "another test S1", "distractor")
    fp["pattern"] = [pattern(r) for r in fp.itertuples()]
    fp["on_singleton"] = fp.s1_id.map(single)
    print(f"\nFALSE POSITIVES: {len(fp):,}  (on singletons: {fp.on_singleton.mean():.1%})")
    print(fp.whose.value_counts(normalize=True).round(3).to_string())
    print(pd.crosstab(fp.pattern, fp.whose, margins=True).sort_values("All", ascending=False).to_string())
    print("by country:", fp.country.value_counts(normalize=True).round(3).to_dict())

    # ---- 3. false negatives
    in_cands = set(zip(df.s1_id, df.cand_id))
    fn_all = tp_all[~tp_all.set_index(["source1_entity_id", "m"]).index.isin(df[df.pred & (df.label == 1)].set_index(["s1_id", "cand_id"]).index)]
    not_retrieved = sum((s, m) not in in_cands for s, m in zip(fn_all.source1_entity_id, fn_all.m))
    fn = df[(df.label == 1) & ~df.pred].copy()
    fn = pd.concat([fn, tag(fn, F)], axis=1)
    fn["reason"] = np.where(fn.stolen, "above threshold but assigned to another S1", "score below threshold")
    fn["pattern"] = [pattern(r) for r in fn.itertuples()]
    print(f"\nFALSE NEGATIVES: {len(fn_all):,} true pairs missed  (not retrieved in top-{df['rank'].max() + 1}: {not_retrieved:,})")
    print(fn.reason.value_counts().to_string())
    print(pd.crosstab(fn.pattern, fn.reason, margins=True).sort_values("All", ascending=False).to_string())
    print("by country:", fn.country.value_counts(normalize=True).round(3).to_dict())
    print(f"median score of missed pairs: {fn.p.median():.3f}; median retrieval rank: {fn['rank'].median():.0f}")

    # baseline rates of the same patterns among correctly predicted pairs, for context
    tp = df[df.pred & (df.label == 1)].sample(200_000, random_state=0)
    tp = pd.concat([tp, tag(tp, F)], axis=1)
    tp["pattern"] = [pattern(r) for r in tp.itertuples()]
    print("\npattern share among CORRECT matches (for comparison):")
    print(tp.pattern.value_counts(normalize=True).round(3).to_string())

    # ---- 4. examples
    pd.set_option("display.width", 250)
    print("\nEXAMPLES")
    for name, d, col in (("FP", fp, "pattern"), ("FN", fn, "pattern")):
        for pat, sub in d.groupby(col):
            if len(sub) < 50:
                continue
            print(f"\n[{name}] {pat}  (n={len(sub):,})")
            for r in sub.sample(min(4, len(sub)), random_state=1).itertuples():
                extra = f" whose={r.whose}" if name == "FP" else f" reason={r.reason}"
                print(f"   p={r.p:.3f} cos={r.cos:.3f} rank={r.rank}{extra}\n      S1  : {r.text_a}\n      cand: {r.text_b}")

    out = {"thr": a.thr, "macro_f05": float(g.f.mean()),
           "loss_share": {k: float(v / total_lost) for k, v in buckets.items()},
           "fp_total": int(len(fp)), "fp_whose": fp.whose.value_counts().to_dict(),
           "fp_patterns": fp.pattern.value_counts().to_dict(),
           "fn_total": int(len(fn_all)), "fn_not_retrieved": int(not_retrieved),
           "fn_reason": fn.reason.value_counts().to_dict(), "fn_patterns": fn.pattern.value_counts().to_dict(),
           "correct_patterns": tp.pattern.value_counts(normalize=True).to_dict()}
    json.dump(out, open(os.path.join(WORK, "error_analysis.json"), "w"), indent=1)


if __name__ == "__main__":
    main()

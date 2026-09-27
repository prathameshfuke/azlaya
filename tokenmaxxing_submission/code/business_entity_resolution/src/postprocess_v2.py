#!/usr/bin/env python3
"""
Post-processing variants of the v2 submission (needs the v1 + v2 caches; runs in minutes).

  fix     v2 with a conservative rule for France (the country absent from training):
            France: threshold --fr-thr (0.9) and "risky" pairs - house number off by 1-5/7/9/11
            (the distractor fingerprint) or unrelated names (token-set ratio < 60) - need p > --risky-p
            India / US: unchanged (on the validation split every stricter rule lowered F0.5 there)
          On the validation hold-out (India + US) the France rule would cost 0.0007 F0.5 if France
          behaved the same -> ~0.0001 overall, precision 0.9989 -> 0.9996.

  merge   agreement ensemble of v2 and another submission (--other, e.g. a friend's matching file):
            * every pair both submissions agree on
            * v2-only pairs with ensemble p >= --v2-only-p (0.98) and no house fingerprint
            * other-only pairs: kept unless v2 is confident they are wrong (p < --other-min-p, 0.05)
              when the pair is in v2's candidate list; pairs outside v2's top-20 are kept (v2 has no
              opinion on them - mostly retrieval misses on empty/partial addresses)
            * each S2/S3 record still goes to at most one S1 (agreed pairs first, then by v2 probability)

  france  v2 with targeted France rules from manual inspection (India / US unchanged):
            * descriptor swap - same core name, one generic French word replaced (club/comite/lycee/...) -> reject
            * house numbers both present and different -> need p > --house-p (0.98)

Usage:
    python code/postprocess_v2.py france                     # -> output/v3_france/
    python code/postprocess_v2.py fix                        # -> output/v2_fix/
    python code/postprocess_v2.py merge --other output/ensemble/matching_results.tsv   # -> output/merge/
"""
import argparse
import os
import subprocess
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import full_pipeline_v1 as v1  # noqa: E402
import retrieval_experiments as rx  # noqa: E402

ROOT = rx.ROOT
V1_CACHE, V2_CACHE = "/tmp/amazon_ml_cache/full_v1", "/tmp/amazon_ml_cache/full_v2"


def load():
    s1 = rx.read_tsv(os.path.join(v1.TEST_DIR, "test_source1.tsv"), ["entity_id", "country"])
    oth = pd.concat([rx.read_tsv(os.path.join(v1.TEST_DIR, f"test_source{s}.tsv"), ["entity_id"]) for s in (2, 3)],
                    ignore_index=True)
    z = np.load(os.path.join(V1_CACHE, "candidates.npz"))
    feats = pd.read_parquet(os.path.join(V2_CACHE, "test_features.parquet"), columns=["house_fingerprint", "name_tsr"])
    prob = np.load(os.path.join(V2_CACHE, "ensemble_prob.npy"))
    return s1, oth, z["qi"], z["ci"], prob, feats


def read_pairs(path):
    d = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    e = d.assign(c=d.matched_entity_ids.str.split(",")).explode("c")
    return e[e.c != ""][["source1_entity_id", "c"]].rename(columns={"source1_entity_id": "s"}).reset_index(drop=True)


def write(out, s1_ids, match, cands, s1_country):
    """match / cands: DataFrames with columns s, c (entity ids)."""
    os.makedirs(out, exist_ok=True)
    m = match.groupby("s").c.agg(list)
    c = cands.groupby("s").c.agg(list)
    m_lists = [m.get(s, []) for s in s1_ids]
    c_lists = [c.get(s, []) for s in s1_ids]
    mp, cp = os.path.join(out, "matching_results.tsv"), os.path.join(out, "candidate_pairs.tsv")
    v1.write_tsv(mp, ["source1_entity_id", "matched_entity_ids"], s1_ids, m_lists)
    v1.write_tsv(cp, ["source1_entity_id", "candidate_entity_ids"], s1_ids, c_lists)
    sizes = pd.Series([len(x) for x in m_lists])
    by_c = sizes.groupby(s1_country).agg(["mean", lambda x: (x == 0).mean()])
    by_c.columns = ["mean matches", "share empty"]
    print(f"wrote {os.path.relpath(mp, ROOT)}: {len(match):,} pairs, {(sizes == 0).mean():.1%} empty, "
          f"{sizes.mean():.3f} per S1\n{by_c.round(3).to_string()}")
    r = subprocess.run([sys.executable, "utils/validate_submission.py", "--matching", mp, "--candidate", cp,
                        "--test-dir", "dataset/test"], cwd=os.path.join(ROOT, "student_resource"),
                       capture_output=True, text=True)
    print("\n".join(l for l in r.stdout.splitlines() if l.startswith(("PASS", "FAIL", "WARNING: ", "ERROR", "  matching")))
          [:1500])


def compare(path, name, pairs):
    """Share of pairs in common with another submission (for the summary)."""
    o = read_pairs(path)
    a, b = set(pairs.s + "|" + pairs.c), set(o.s + "|" + o.c)
    print(f"  vs {name}: {len(a & b):,} common, {len(a - b):,} only here, {len(b - a):,} only there")


def fix(a):
    s1, oth, qi, ci, prob, F = load()
    fr = s1.country.values[qi] == "France"
    risky = (F.house_fingerprint.values == 1) | (F.name_tsr.values < 60)
    score = prob.copy()
    score[fr & risky & (prob <= a.risky_p)] = 0.0
    score[fr & (prob <= a.fr_thr)] = 0.0
    keep = v1.decide(qi, ci, score, a.thr, False)
    base = v1.decide(qi, ci, prob, a.thr, False)
    print(f"France rule: v2 {fr[base].sum():,} France pairs -> {fr[keep].sum():,} "
          f"({fr[base].sum() - fr[keep].sum():,} removed); India/US unchanged")
    match = pd.DataFrame({"s": s1.entity_id.values[qi[keep]], "c": oth.entity_id.values[ci[keep]]})
    cands = pd.DataFrame({"s": s1.entity_id.values[qi], "c": oth.entity_id.values[ci]})
    write(a.out or os.path.join(ROOT, "output", "v2_fix"), s1.entity_id.values, match, cands, s1.country.values)
    if a.other:
        compare(a.other, "other", match)


# French hard negatives seen when inspecting disagreements between submissions: same address, same core
# name, but one generic descriptor word swapped ("House Lycee SAS" vs "House Section SAS") - 1.2% of the
# pairs every submission agrees on, 50-88% of the pairs the best-scoring submission rejects.
FR_SYNONYMS = [{"cie", "compagnie", "co"}, {"ets", "etablissements", "etablissement"}, {"centre", "center"},
               {"st", "saint"}, {"ste", "sainte"}, {"and", "et"}, {"freres", "frere"}, {"assoc", "association"},
               {"federation", "fed"}, {"soc", "societe"}]


def fr_descriptor_vocab(names, min_share=0.002):
    from collections import Counter
    cnt = Counter(t for n in names for t in set(n.split()))
    return {t for t, c in cnt.items() if c / len(names) > min_share and len(t) > 2}


def descriptor_swap(na, nb, vocab):
    canon = {w: min(g) for g in FR_SYNONYMS for w in g}
    ta = {canon.get(t, t) for t in na.split()}
    tb = {canon.get(t, t) for t in nb.split()}
    oa, ob = ta - tb, tb - ta
    return bool(oa) and bool(ob) and (oa | ob) <= vocab and bool(ta & tb)


def france(a):
    s1, oth, qi, ci, prob, F = load()
    cols = ["entity_id", "name_core", "house_base"]
    n1 = rx.read_tsv(os.path.join(v1.TEST_DIR, "test_source1.tsv"), cols + ["country"])
    n2 = pd.concat([rx.read_tsv(os.path.join(v1.TEST_DIR, f"test_source{s}.tsv"), cols) for s in (2, 3)],
                   ignore_index=True)
    vocab = fr_descriptor_vocab(n1.name_core.values[n1.country.values == "France"])
    fr = s1.country.values[qi] == "France"
    cand = np.where(fr & (prob > a.thr))[0]               # only pairs that could be kept need the check
    na, nb = n1.name_core.values[qi[cand]], n2.name_core.values[ci[cand]]
    swap = np.array([descriptor_swap(x, y, vocab) for x, y in zip(na, nb)])
    ha, hb = n1.house_base.values[qi[cand]], n2.house_base.values[ci[cand]]
    hdiff = (ha != "") & (hb != "") & (ha != hb)
    score = prob.copy()
    score[cand[swap & (prob[cand] <= a.swap_p)]] = 0.0
    score[cand[hdiff & (prob[cand] <= a.house_p)]] = 0.0
    keep = v1.decide(qi, ci, score, a.thr, False)
    base = v1.decide(qi, ci, prob, a.thr, False)
    print(f"France: {len(vocab)} descriptor words; above thr {len(cand):,} pairs: descriptor swap {swap.sum():,}, "
          f"house differs {hdiff.sum():,}")
    print(f"France pairs kept: v2 {fr[base].sum():,} -> {fr[keep].sum():,}; India/US unchanged")
    match = pd.DataFrame({"s": s1.entity_id.values[qi[keep]], "c": oth.entity_id.values[ci[keep]]})
    cands = pd.DataFrame({"s": s1.entity_id.values[qi], "c": oth.entity_id.values[ci]})
    write(a.out or os.path.join(ROOT, "output", "v3_france"), s1.entity_id.values, match, cands, s1.country.values)


def merge(a):
    s1, oth, qi, ci, prob, F = load()
    V = read_pairs(os.path.join(ROOT, "output", "v2", "matching_results.tsv"))
    O = read_pairs(a.other)
    key_all = pd.Series(np.arange(len(qi)), index=s1.entity_id.values[qi] + "|" + oth.entity_id.values[ci])
    kv, ko = V.s + "|" + V.c, O.s + "|" + O.c
    sv, so = set(kv), set(ko)
    V["agree"] = kv.isin(so).values
    O["agree"] = ko.isin(sv).values
    V["p"] = prob[key_all.reindex(kv).values.astype(int)]
    V["fp"] = F.house_fingerprint.values[key_all.reindex(kv).values.astype(int)] == 1
    j = key_all.reindex(ko).values
    O["in_cands"] = ~np.isnan(j)
    O["p"] = np.where(O.in_cands, prob[np.nan_to_num(j).astype(int)], np.nan)

    agreed = V[V.agree].assign(src="agreed", pr=2.0)
    v2_only = V[~V.agree & (V.p >= a.v2_only_p) & ~V.fp].assign(src="v2-only", pr=lambda d: d.p)
    o_only = O[~O.agree & (~O.in_cands | (O.p >= a.other_min_p))].assign(
        src="other-only", pr=lambda d: d.p.fillna(0.5))
    allp = pd.concat([agreed, v2_only, o_only], ignore_index=True)
    # each S2/S3 record to at most one S1: agreed first, then higher v2 probability
    allp = allp.sort_values("pr", ascending=False, kind="stable").drop_duplicates("c")
    print("merged pairs by source:", allp.src.value_counts().to_dict())
    print(f"  dropped: v2-only {(~V.agree).sum() - len(v2_only):,} (p < {a.v2_only_p} or house fingerprint), "
          f"other-only {(~O.agree).sum() - len(o_only):,} (v2 p < {a.other_min_p}); "
          f"record conflicts {len(agreed) + len(v2_only) + len(o_only) - len(allp):,}")
    cands = pd.concat([pd.DataFrame({"s": s1.entity_id.values[qi], "c": oth.entity_id.values[ci]}),
                       allp[allp.src == "other-only"][["s", "c"]]], ignore_index=True).drop_duplicates()
    write(a.out or os.path.join(ROOT, "output", "merge"), s1.entity_id.values, allp[["s", "c"]], cands,
          s1.country.values)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["fix", "merge", "france"])
    ap.add_argument("--thr", type=float, default=0.75, help="v2 threshold (config.json)")
    ap.add_argument("--fr-thr", type=float, default=0.9)
    ap.add_argument("--risky-p", type=float, default=0.98)
    ap.add_argument("--other", default=os.path.join(ROOT, "output", "ensemble", "matching_results.tsv"))
    ap.add_argument("--v2-only-p", type=float, default=0.98)
    ap.add_argument("--other-min-p", type=float, default=0.05)
    ap.add_argument("--swap-p", type=float, default=1.1, help="france: descriptor swaps kept only above this p (1.1 = never)")
    ap.add_argument("--house-p", type=float, default=0.98, help="france: differing house numbers need p above this")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    {"fix": fix, "merge": merge, "france": france}[a.mode](a)


if __name__ == "__main__":
    main()

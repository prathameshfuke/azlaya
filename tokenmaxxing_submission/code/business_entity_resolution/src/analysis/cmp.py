import sys, os, numpy as np, pandas as pd
sys.path.insert(0, "code")
import retrieval_experiments as rx, full_pipeline_v1 as v1
def pairs(p):
    d = pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False)
    e = d.assign(c=d.matched_entity_ids.str.split(",")).explode("c")
    return d.source1_entity_id, e[e.c != ""][["source1_entity_id", "c"]].rename(columns={"source1_entity_id": "s"})
s1ids, F = pairs("output/ensemble/matching_results.tsv")
_, V = pairs("output/v2/matching_results.tsv")
_, V1 = pairs("output/matching_results.tsv")
print(f"pairs: friend {len(F):,}  v2 {len(V):,}  v1 {len(V1):,}")
print(f"friend: records assigned to >1 S1: {(F.c.value_counts() > 1).sum():,}; S1 empty {1 - F.s.nunique() / len(s1ids):.2%}")
key = lambda d: d.s + "|" + d.c
kf, kv = set(key(F)), set(key(V))
both = kf & kv
print(f"common {len(both):,}; friend-only {len(kf - kv):,}; v2-only {len(kv - kf):,}; Jaccard {len(both) / len(kf | kv):.4f}")

# per-S1 agreement and pseudo-F0.5 of v2 taking the friend as reference (and vice versa)
def per_s1(a, b, ids):
    ga, gb = a.groupby("s").c.apply(set), b.groupby("s").c.apply(set)
    out = []
    for s in ids:
        x, y = ga.get(s, set()), gb.get(s, set())
        if not y: out.append(1.0 if not x else 0.0); continue
        tp = len(x & y); P = tp / len(x) if x else 0; R = tp / len(y)
        out.append(1.25 * P * R / (0.25 * P + R) if tp else 0.0)
    return np.array(out)
f_v2_vs_friend = per_s1(V, F, s1ids)
print(f"S1 rows identical-ish: macro F0.5(v2 | friend as truth) = {f_v2_vs_friend.mean():.4f}; "
      f"exact-identical rows {np.mean(f_v2_vs_friend == 1):.2%}")
print(f"macro F0.5(v1 | friend as truth) = {per_s1(V1, F, s1ids).mean():.4f}")

# attach test info + v2 scores to disagreement pairs
cols = ["entity_id", "country", "name_clean", "addr_clean", "addr_empty", "house_base", "legal_form"]
s1 = rx.read_tsv(os.path.join(v1.TEST_DIR, "test_source1.tsv"), cols)
oth = pd.concat([rx.read_tsv(os.path.join(v1.TEST_DIR, f"test_source{i}.tsv"), cols) for i in (2, 3)], ignore_index=True)
z = np.load("/tmp/amazon_ml_cache/full_v1/candidates.npz"); qi, ci = z["qi"], z["ci"]
prob = np.load("/tmp/amazon_ml_cache/full_v2/ensemble_prob.npy")
rr = 1 / (1 + np.exp(-np.load("/tmp/amazon_ml_cache/full_v1/logits.npy")))
cand_key = pd.Series(np.arange(len(qi)), index=s1.entity_id.values[qi] + "|" + oth.entity_id.values[ci])
qpos = pd.Series(np.arange(len(s1)), index=s1.entity_id); cpos = pd.Series(np.arange(len(oth)), index=oth.entity_id)
country = dict(zip(s1.entity_id, s1.country))
V2owner = dict(zip(V.c, V.s))
for name, ks in (("friend-only", kf - kv), ("v2-only", kv - kf)):
    d = pd.DataFrame({"k": list(ks)}); d[["s", "c"]] = d.k.str.split("|", expand=True)
    j = cand_key.reindex(d.k).values
    d["in_cands"] = ~np.isnan(j); jj = np.where(d.in_cands, j, 0).astype(int)
    d["p_v2"] = np.where(d.in_cands, prob[jj], np.nan); d["p_rr"] = np.where(d.in_cands, rr[jj], np.nan)
    d["country"] = d.s.map(country)
    b = oth.iloc[cpos.reindex(d.c).fillna(0).astype(int).values]; a = s1.iloc[qpos.reindex(d.s).values]
    d["cand_addr_empty"] = b.addr_empty.values == "1"
    d["house_eq"] = (a.house_base.values == b.house_base.values) & (a.house_base.values != "")
    d["legal_eq"] = a.legal_form.values == b.legal_form.values
    d["name_eq"] = a.name_clean.values == b.name_clean.values
    if name == "friend-only":
        d["v2_gave_it_to_other_s1"] = d.c.map(V2owner).notna()
    print(f"\n=== {name}: {len(d):,} pairs")
    print(f"  in v2 top-20 candidates: {d.in_cands.mean():.2%}")
    print("  by country:", d.country.value_counts(normalize=True).round(3).to_dict())
    print("  share of all pairs of that country type -> see below")
    print(f"  cand address empty {d.cand_addr_empty.mean():.2%}; house equal {d.house_eq.mean():.2%}; "
          f"legal equal {d.legal_eq.mean():.2%}; name_clean equal {d.name_eq.mean():.2%}")
    if name == "friend-only":
        print(f"  record assigned by v2 to a different S1: {d.v2_gave_it_to_other_s1.mean():.2%}")
    print("  v2 ensemble prob quantiles:", d.p_v2.quantile([.1, .25, .5, .75, .9]).round(3).to_dict())
    print("  reranker prob quantiles:   ", d.p_rr.quantile([.1, .25, .5, .75, .9]).round(3).to_dict())
    d.to_parquet(f"/tmp/amazon_ml_scratch/{name}.parquet")
    for r in d.sample(6, random_state=0).itertuples():
        A = s1.iloc[qpos[r.s]]; B = oth.iloc[cpos[r.c]]
        print(f"   [{r.country}] p_v2={r.p_v2:.3f} p_rr={r.p_rr:.3f}\n     S1  : {A.name_clean} | {A.addr_clean}\n     cand: {B.name_clean} | {B.addr_clean}")
# per-country pair counts
for n, d in (("friend", F), ("v2", V)):
    print(n, "pairs per S1 by country:", (d.s.map(country).value_counts() / s1.country.value_counts()).round(3).to_dict())

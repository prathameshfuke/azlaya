import sys, os, numpy as np, pandas as pd
sys.path.insert(0, "code")
import retrieval_experiments as rx, full_pipeline_v1 as v1
files = {"new_0.987337": "output/matching_results_0.987337.tsv", "friend_0.986149": "output/ensemble/matching_results.tsv",
         "v1": "output/matching_results.tsv", "v2": "output/v2/matching_results.tsv",
         "v2_fix": "output/v2_fix/matching_results.tsv", "merge": "output/merge/matching_results.tsv"}
def read(p):
    d = pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False)
    e = d.assign(c=d.matched_entity_ids.str.split(",")).explode("c"); e = e[e.c != ""]
    return d, e.source1_entity_id.values + "|" + e.c.values, e
D, K, E = {}, {}, {}
for n, p in files.items():
    D[n], k, E[n] = read(p); K[n] = set(k)
N = D["new_0.987337"]
print(f"new file: {len(N):,} rows, header {list(N.columns)}, empty {(N.matched_entity_ids == '').mean():.2%}, pairs {len(K['new_0.987337']):,}")
print(f"  records assigned to >1 S1: {(E['new_0.987337'].c.value_counts() > 1).sum():,}")
names = list(files)
J = pd.DataFrame(index=names, columns=names, dtype=float)
for a in names:
    for b in names:
        J.loc[a, b] = len(K[a] & K[b]) / len(K[a] | K[b])
print("\npair Jaccard:"); print(J.round(4).to_string())
s1 = rx.read_tsv(os.path.join(v1.TEST_DIR, "test_source1.tsv"), ["entity_id", "country"])
ids = s1.entity_id.values; country = dict(zip(s1.entity_id, s1.country))
def macro(pred, truth):   # macro F0.5 of pred with truth = another file (singletons included)
    gp, gt = pred.groupby("source1_entity_id").c.apply(set), truth.groupby("source1_entity_id").c.apply(set)
    f = []
    for s in ids:
        x, y = gp.get(s, set()), gt.get(s, set())
        if not y: f.append(float(not x)); continue
        tp = len(x & y)
        if not tp: f.append(0.0); continue
        P, R = tp / len(x), tp / len(y); f.append(1.25 * P * R / (0.25 * P + R))
    return np.mean(f)
print("\nmacro F0.5 of each file measured against the NEW file as reference:")
for n in names[1:]:
    print(f"  {n:16s} {macro(E[n], E['new_0.987337']):.4f}")
print("\npairs per S1 by country:")
rows = {n: (pd.Series([country[s.split('|')[0]] for s in K[n]]).value_counts() / s1.country.value_counts()).round(3) for n in names}
print(pd.DataFrame(rows).T.to_string())
# who is new closest to on the disagreements between v2 and friend?
kv, kf, kn = K["v2"], K["friend_0.986149"], K["new_0.987337"]
v2o, fo = kv - kf, kf - kv
print(f"\nv2-only vs friend ({len(v2o):,} pairs): new file contains {len(v2o & kn) / len(v2o):.1%}")
print(f"friend-only vs v2 ({len(fo):,} pairs): new file contains {len(fo & kn) / len(fo):.1%}")
for c in ("France", "India", "US"):
    a = {k for k in v2o if country[k.split('|')[0]] == c}; b = {k for k in fo if country[k.split('|')[0]] == c}
    print(f"  {c:7s} v2-only kept by new {len(a & kn) / max(1, len(a)):.1%} (n={len(a):,}) | friend-only kept by new {len(b & kn) / max(1, len(b)):.1%} (n={len(b):,})")
nn = kn - kv - kf
print(f"pairs only in new (in neither v2 nor friend): {len(nn):,}; missing from new but in both v2 and friend: {len((kv & kf) - kn):,}")
import pickle; pickle.dump({"K": K, "country": country}, open("/tmp/amazon_ml_cache/cmp2.pkl", "wb"))

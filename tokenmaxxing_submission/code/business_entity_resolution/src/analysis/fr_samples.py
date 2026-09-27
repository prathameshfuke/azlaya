import sys, os, pickle, numpy as np, pandas as pd
sys.path.insert(0, "code")
import retrieval_experiments as rx, full_pipeline_v1 as v1
d = pickle.load(open("/tmp/amazon_ml_cache/cmp2.pkl", "rb")); K, country = d["K"], d["country"]
cols = ["entity_id", "business_name", "business_address", "country"]
s1 = rx.read_tsv(os.path.join(v1.TEST_DIR, "test_source1.tsv"), cols).set_index("entity_id")
oth = pd.concat([rx.read_tsv(os.path.join(v1.TEST_DIR, f"test_source{i}.tsv"), cols) for i in (2, 3)]).set_index("entity_id")
z = np.load("/tmp/amazon_ml_cache/full_v1/candidates.npz"); prob = np.load("/tmp/amazon_ml_cache/full_v2/ensemble_prob.npy")
rr = 1 / (1 + np.exp(-np.load("/tmp/amazon_ml_cache/full_v1/logits.npy")))
s1_ids = s1.index.values; o_ids = oth.index.values
key = pd.Series(np.arange(len(z["qi"])), index=s1_ids[z["qi"]] + "|" + o_ids[z["ci"]])
fr = lambda S: {k for k in S if country[k.split("|")[0]] == "France"}
new, fri, v2 = fr(K["new_0.987337"]), fr(K["friend_0.986149"]), fr(K["v2"])
groups = {
  "A. v2 says YES, new+friend say NO": v2 - new - fri,
  "B. v2+new say YES, friend NO": (v2 & new) - fri,
  "C. new says YES only": new - v2 - fri,
  "D. v2+friend YES, new NO": (v2 & fri) - new,
  "E. friend YES only": fri - v2 - new,
  "F. all three agree (reference)": v2 & fri & new,
}
rng = np.random.default_rng(7)
for g, S in groups.items():
    S = sorted(S); print(f"\n{'=' * 110}\n{g}: {len(S):,} pairs")
    for k in rng.choice(S, min(14, len(S)), replace=False):
        a, b = k.split("|"); j = key.get(k)
        sc = f"p_v2={prob[j]:.2f} rr={rr[j]:.2f}" if j is not None else "not in v2 top-20"
        # how many S2/S3 share this S1's name exactly / how many S1 claim
        print(f"  [{sc}]\n    S1  : {s1.at[a, 'business_name']}  ||  {s1.at[a, 'business_address']}\n    {b[:2]}  : {oth.at[b, 'business_name']}  ||  {oth.at[b, 'business_address']}")

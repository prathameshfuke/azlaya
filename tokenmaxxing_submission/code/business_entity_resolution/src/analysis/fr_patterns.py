import sys, os, pickle, numpy as np, pandas as pd
from collections import Counter
sys.path.insert(0, "code")
import retrieval_experiments as rx, full_pipeline_v1 as v1
d = pickle.load(open("/tmp/amazon_ml_cache/cmp2.pkl", "rb")); K, country = d["K"], d["country"]
cols = ["entity_id", "country", "name_core", "legal_form", "house_base", "addr_clean"]
s1 = rx.read_tsv(os.path.join(v1.TEST_DIR, "test_source1.tsv"), cols).set_index("entity_id")
oth = pd.concat([rx.read_tsv(os.path.join(v1.TEST_DIR, f"test_source{i}.tsv"), cols) for i in (2, 3)]).set_index("entity_id")
# generic French descriptor words: frequent name tokens among France S1 names
fr1 = s1[s1.country == "France"]
cnt = Counter(t for n in fr1.name_core for t in set(n.split()))
N = len(fr1)
vocab = {t for t, c in cnt.items() if c / N > 0.002 and len(t) > 2}
print(f"{len(vocab)} frequent France name tokens, top:", [t for t, _ in cnt.most_common(45)])
def classify(k):
    a, b = k.split("|"); A, B = s1.loc[a], oth.loc[b]
    ta, tb = set(A.name_core.split()), set(B.name_core.split())
    oa, ob = ta - tb, tb - ta
    hd = A.house_base != "" and B.house_base != "" and A.house_base != B.house_base
    swap = len(oa) >= 1 and len(ob) >= 1 and (oa | ob) <= vocab and len(ta & tb) >= 1
    return {"house_diff": hd, "descriptor_swap": swap,
            "desc_added_or_dropped": (not swap) and (len(oa) + len(ob) >= 1) and (oa | ob) <= vocab and len(ta & tb) >= 1,
            "name_disjoint": len(ta & tb) == 0, "cand_addr_empty": B.addr_clean == ""}
fr = lambda S: {k for k in S if country[k.split("|")[0]] == "France"}
new, fri, v2 = fr(K["new_0.987337"]), fr(K["friend_0.986149"]), fr(K["v2"])
groups = {"A v2 only": v2 - new - fri, "B v2+new": (v2 & new) - fri, "C new only": new - v2 - fri,
          "D v2+friend": (v2 & fri) - new, "E friend only": fri - v2 - new, "F all agree": v2 & fri & new}
rng = np.random.default_rng(0); rows = {}
for g, S in groups.items():
    S = sorted(S); samp = rng.choice(S, min(5000, len(S)), replace=False)
    rows[g] = pd.DataFrame([classify(k) for k in samp]).mean().round(3); rows[g]["n pairs"] = len(S)
print(pd.DataFrame(rows).T.to_string())

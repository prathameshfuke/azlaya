#!/usr/bin/env python3
"""
Exploratory data analysis for the Amazon ML Challenge 2026 business entity
resolution task (S1 -> S2/S3 matching).

Runs every analysis used for EDA.pdf, prints the findings, saves the numbers to
<out>/results.json, draws the figures to <out>/fig/ and builds <out>/EDA.pdf.

Usage (from the project root):
    python code/eda.py                       # all steps + figures + PDF
    python code/eda.py --steps overview gt   # only some steps
    python code/eda.py --no-report           # skip figures and PDF

Requires: pandas, pyarrow, numpy, rapidfuzz; matplotlib + reportlab for the report.
Runtime: roughly 15-25 minutes on the full data (the blocking step is the slowest).
"""
import argparse
import csv
import json
import os
import re
import time
import unicodedata

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILES = [f"{sp}_s{s}" for sp in ("train", "test") for s in (1, 2, 3)]
SEED = 0

# --------------------------------------------------------------------------------------
# Normalisation helpers
# --------------------------------------------------------------------------------------
LEGAL = set(
    "inc incorporated llc l l c ltd limited pvt private corp corporation co company llp pc "
    "pllc plc lp the sarl sas sasu eurl sci sa public group holdings dba".split()
)
LEGAL_RE = re.compile(
    r"\b(inc|llc|l\.l\.c|ltd|limited|pvt|private|corp|corporation|co|company|llp|pc|pllc|plc|lp|holdings|group)\b",
    re.I,
)
STREET_ABBR = {
    "street": "st", "road": "rd", "drive": "dr", "avenue": "ave", "lane": "ln", "boulevard": "blvd",
    "court": "ct", "place": "pl", "circle": "cir", "trail": "trl", "highway": "hwy", "parkway": "pkwy",
    "terrace": "ter", "saint": "st", "north": "n", "south": "s", "east": "e", "west": "w", "rue": "r",
}
INDIC_RE = re.compile(r"[ऀ-෿]")


def strip_accents(s):
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def norm_name(s):
    """Lower-case, strip accents/punctuation/'(ID: n)', drop legal and generic tokens."""
    s = strip_accents(s.lower()).replace("&", " and ")
    s = re.sub(r"\(id:?\s*\d+\)", "", s)
    s = re.sub(r"[^\w\s]", " ", s)
    return " ".join(t for t in s.split() if t not in LEGAL)


def is_indic(s):
    return bool(INDIC_RE.search(s))


def legal_suffixes(s):
    return frozenset(m.lower().replace(".", "") for m in LEGAL_RE.findall(s))


def first_number(s):
    """First integer in the address (house number proxy), or None."""
    m = re.findall(r"\d+", s)
    return int(m[0]) if m else None


def first_number_str(s):
    m = re.search(r"\d+", s)
    return m.group().lstrip("0") if m else ""


def addr_tokens(s):
    s = strip_accents(s.lower())
    s = re.sub(r"[^\w\s/]", " ", s)
    return [STREET_ABBR.get(t, t.lstrip("0") if t.isdigit() else t) for t in s.split()]


def blocking_keys(addr, name):
    t = addr_tokens(addr)
    nums = [x for x in t if re.fullmatch(r"\d+[a-z]?(/\d+)*", x)]
    words = [x for x in t if x.isalpha() and len(x) > 2]
    nn = norm_name(name).split()
    return {
        "k_name": " ".join(nn),
        "k_num_word": f"{nums[0]}|{words[0]}" if nums and words else None,
        "k_num_2words": f"{nums[0]}|{' '.join(sorted(words[:2]))}" if nums and len(words) > 1 else None,
        "k_nametok_num": f"{nn[0]}|{nums[0]}" if nn and nums else None,
    }


# --------------------------------------------------------------------------------------
# Data loading (TSV -> cached parquet)
# --------------------------------------------------------------------------------------
class Data:
    def __init__(self, data_dir, cache_dir):
        self.data_dir, self.cache_dir = data_dir, cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self._mem = {}

    def _read_tsv(self, path):
        # dtype=str + keep_default_na=False keeps names like "NULL"/"NA" as text.
        return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE)

    def get(self, key):
        if key in self._mem:
            return self._mem[key]
        pq = os.path.join(self.cache_dir, f"{key}.parquet")
        if os.path.exists(pq):
            df = pd.read_parquet(pq)
        else:
            if key == "gt":
                path = os.path.join(self.data_dir, "train", "train_ground_truth.tsv")
            else:
                sp, s = key.split("_s")
                path = os.path.join(self.data_dir, sp, f"{sp}_source{s}.tsv")
            t = time.time()
            df = self._read_tsv(path)
            df.to_parquet(pq)
            print(f"  loaded {key}: {df.shape} in {time.time() - t:.1f}s")
        self._mem[key] = df
        return df

    def pairs(self):
        """Exploded ground truth: one row per (source1_entity_id, matched id m)."""
        if "pairs" not in self._mem:
            gt = self.get("gt")
            p = gt.assign(m=gt.matched_entity_ids.str.split(",")).explode("m")
            self._mem["pairs"] = p[p.m != ""][["source1_entity_id", "m"]].reset_index(drop=True)
        return self._mem["pairs"]

    def others(self, split="train"):
        """S2 and S3 concatenated."""
        k = f"{split}_others"
        if k not in self._mem:
            self._mem[k] = pd.concat([self.get(f"{split}_s2"), self.get(f"{split}_s3")], ignore_index=True)
        return self._mem[k]


def header(t):
    print("\n" + "=" * 90 + f"\n{t}\n" + "=" * 90)


# --------------------------------------------------------------------------------------
# Step 1: per-file overview & field quality
# --------------------------------------------------------------------------------------
def step_overview(D, R):
    header("1. File overview and field quality")
    out = {}
    for k in FILES:
        df = D.get(k)
        n, a, nm = df.business_name, df.business_address, df.business_name
        rec = {
            "rows": len(df),
            "unique_ids": int(df.entity_id.nunique()),
            "country": df.country.value_counts().to_dict(),
            "name_empty": float((n.str.strip() == "").mean()),
            "name_len_mean": float(n.str.len().mean()),
            "addr_empty": float((a.str.strip() == "").mean()),
            "addr_null_literal": float(a.str.lower().str.contains(r"\bnull\b|\bnan\b|\bnone\b", regex=True).mean()),
            "addr_len_mean": float(a.str.len().mean()),
            "addr_len_p50": float(a.str.len().median()),
            "addr_len_p99": float(a.str.len().quantile(0.99)),
            "dup_name_addr": float(df.duplicated(["business_name", "business_address"]).mean()),
            "dup_name": float(df.duplicated(["business_name"]).mean()),
            "name_non_ascii": float(nm.str.contains(r"[^\x00-\x7f]").mean()),
            "name_devanagari": float(nm.str.contains(r"[ऀ-ॿ]").mean()),
            "name_other_indic": float(nm.str.contains(r"[ঀ-෿]").mean()),
            "addr_non_ascii": float(a.str.contains(r"[^\x00-\x7f]").mean()),
            "by_country": {},
        }
        for c in df.country.unique():
            aa = a[df.country == c]
            rec["by_country"][c] = {
                "pin6": float(aa.str.contains(r"\b\d{6}\b").mean()),
                "zip5": float(aa.str.contains(r"\b\d{5}(?:-\d{4})?\b").mean()),
                "n_commas": float(aa.str.count(",").mean()),
                "all_upper": float((aa == aa.str.upper()).mean()),
                "landmark": float(aa.str.contains(r"(?i)\b(?:near|opp|opposite|behind|next to)\b").mean()),
                "unit": float(aa.str.contains(r"(?i)\bunit\b").mean()),
                "po_box": float(aa.str.contains(r"(?i)po box").mean()),
                "hash": float(aa.str.contains("#").mean()),
            }
        out[k] = rec
        print(f"\n--- {k}: rows={rec['rows']:,} unique_ids={rec['unique_ids']:,} countries={rec['country']}")
        print(f"  name : empty={rec['name_empty']:.4f} len={rec['name_len_mean']:.1f} non-ascii={rec['name_non_ascii']:.4f} "
              f"devanagari={rec['name_devanagari']:.4f} other-indic={rec['name_other_indic']:.4f} dup={rec['dup_name']:.4f}")
        print(f"  addr : empty={rec['addr_empty']:.4f} NULL-literal={rec['addr_null_literal']:.4f} len={rec['addr_len_mean']:.1f} "
              f"p50={rec['addr_len_p50']} p99={rec['addr_len_p99']} non-ascii={rec['addr_non_ascii']:.4f}")
        for c, v in rec["by_country"].items():
            print(f"  {c:7s}: " + " ".join(f"{kk}={vv:.3f}" for kk, vv in v.items()))
    s1 = D.get("train_s1")
    us = s1[s1.country == "US"].business_address
    R["us_s1_starts_with_state"] = float(us.str.match(r"^[A-Z]{2},").mean())
    R["us_s1_starts_with_digit"] = float(us.str.match(r"^\d").mean())
    print(f"\nUS S1 address starts with 2-letter state: {R['us_s1_starts_with_state']:.3f}, with a digit: {R['us_s1_starts_with_digit']:.3f}")
    t1 = D.get("test_s1")
    print("\nFrance samples (test S1):")
    print(t1[t1.country == "France"].sample(12, random_state=SEED).to_string())
    t2 = D.get("test_s2")
    print("\nFrance samples (test S2):")
    print(t2[t2.country == "France"].sample(8, random_state=SEED).to_string())
    R["overview"] = out


# --------------------------------------------------------------------------------------
# Step 2: ground-truth structure & leakage checks
# --------------------------------------------------------------------------------------
def step_gt(D, R):
    header("2. Ground-truth structure and leakage checks")
    gt, s1 = D.get("gt"), D.get("train_s1")
    pairs = D.pairs()
    r = {}
    r["gt_ids_equal_s1_ids"] = set(gt.source1_entity_id) == set(s1.entity_id)
    lists = gt.matched_entity_ids.str.split(",").apply(lambda l: [x for x in l if x])
    n = lists.str.len()
    n2 = lists.apply(lambda l: sum(x.startswith("S2") for x in l))
    n3 = n - n2
    r["singleton_frac"] = float((n == 0).mean())
    r["n_singletons"] = int((n == 0).sum())
    r["mean_matches"] = float(n.mean())
    r["n_matches_dist"] = {int(k): int(v) for k, v in n.value_counts().sort_index().items()}
    r["n_s2_dist"] = {int(k): int(v) for k, v in n2.value_counts().sort_index().items()}
    r["n_s3_dist"] = {int(k): int(v) for k, v in n3.value_counts().sort_index().items()}
    r["total_pairs"] = len(pairs)
    r["unique_matched_ids"] = int(pairs.m.nunique())
    s2ids, s3ids = set(D.get("train_s2").entity_id), set(D.get("train_s3").entity_id)
    u = set(pairs.m)
    r["s2_matched_share"] = len(u & s2ids) / len(s2ids)
    r["s3_matched_share"] = len(u & s3ids) / len(s3ids)
    r["matched_ids_missing"] = len(u - s2ids - s3ids)
    c1 = s1.set_index("entity_id").country
    gtc = pd.DataFrame({"n": n.values, "country": gt.source1_entity_id.map(c1).values})
    r["by_country"] = gtc.groupby("country").agg(singleton=("n", lambda x: float((x == 0).mean())),
                                                  mean_n=("n", "mean"), count=("n", "size")).to_dict("index")
    oc = D.others().set_index("entity_id").country
    r["country_agreement"] = float((pairs.source1_entity_id.map(c1).values == pairs.m.map(oc).values).mean())
    # Leakage: numeric-ID and row-position correlation within true pairs.
    i1 = pairs.source1_entity_id.str[3:].astype(np.int64)
    i2 = pairs.m.str[3:].astype(np.int64)
    r["id_corr"] = float(np.corrcoef(i1, i2)[0, 1])
    pos1 = pd.Series(np.arange(len(s1)), index=s1.entity_id)
    pos_o = pd.concat([pd.Series(np.arange(len(D.get(k))), index=D.get(k).entity_id) for k in ("train_s2", "train_s3")])
    r["row_pos_corr"] = float(np.corrcoef(pairs.source1_entity_id.map(pos1), pairs.m.map(pos_o))[0, 1])
    r["train_test_id_overlap"] = {s: len(set(D.get(f"train_s{s}").entity_id) & set(D.get(f"test_s{s}").entity_id)) for s in (1, 2, 3)}
    r["test_s1_names_in_train_s1"] = float(D.get("test_s1").business_name.isin(set(s1.business_name)).mean())
    r["size_ratio"] = {sp: {"s2_over_s1": len(D.get(f"{sp}_s2")) / len(D.get(f"{sp}_s1")),
                            "s3_over_s1": len(D.get(f"{sp}_s3")) / len(D.get(f"{sp}_s1"))} for sp in ("train", "test")}
    for k, v in r.items():
        print(f"  {k}: {v}")
    print(f"  -> each S2/S3 record maps to at most one S1: {r['total_pairs'] == r['unique_matched_ids']}")
    R["gt"] = r


# --------------------------------------------------------------------------------------
# Step 3: sample matched clusters
# --------------------------------------------------------------------------------------
def step_examples(D, R, n_per_country=6):
    header("3. Sample matched clusters")
    s1 = D.get("train_s1").set_index("entity_id")
    o = D.others().set_index("entity_id")
    by_s1 = D.pairs().groupby("source1_entity_id").m.apply(list)
    for c in ("US", "India"):
        for i in s1[s1.country == c].sample(n_per_country, random_state=1).index:
            r = s1.loc[i]
            print(f"\n[{c}] {i} | {r.business_name} | {r.business_address}")
            for m in by_s1.get(i, []):
                q = o.loc[m]
                print(f"    {m[:2]} | {q.business_name} | {q.business_address}")


# --------------------------------------------------------------------------------------
# Step 4: similarity of true pairs (200k sample)
# --------------------------------------------------------------------------------------
def pair_features(D, n=200_000):
    from rapidfuzz import fuzz
    s1 = D.get("train_s1").set_index("entity_id")
    o = D.others().set_index("entity_id")
    p = D.pairs().sample(n, random_state=SEED)
    A, B = s1.loc[p.source1_entity_id], o.loc[p.m]
    df = pd.DataFrame({"c": A.country.values, "src": p.m.str[:2].values,
                       "n1": A.business_name.values, "n2": B.business_name.values,
                       "a1": A.business_address.values, "a2": B.business_address.values})
    df["nn1"], df["nn2"] = df.n1.map(norm_name), df.n2.map(norm_name)
    df["indic"] = df.n2.map(is_indic)
    df["name_exact_raw"] = df.n1 == df.n2
    df["name_exact_norm"] = df.nn1 == df.nn2
    df["name_tsr"] = [fuzz.token_set_ratio(a, b) for a, b in zip(df.nn1, df.nn2)]
    df["a2_empty"] = df.a2.str.strip() == ""
    df["addr_tsr"] = [fuzz.token_set_ratio(a.lower(), b.lower()) for a, b in zip(df.a1, df.a2)]
    df["house_eq"] = [first_number_str(a) == first_number_str(b) and first_number_str(a) != "" for a, b in zip(df.a1, df.a2)]
    df["domain"] = df.n2.str.contains(r"\.(?:com|in|net|org|fr|co)\b", regex=True)
    df["dba"] = df.n2.str.contains(r"(?i)\b(?:dba|doing business as|d/b/a|t/a|trading as)\b", regex=True)
    return df


def step_pair_similarity(D, R):
    header("4. Similarity of true matched pairs (200k sample)")
    df = pair_features(D)
    D._mem["pairfeat"] = df
    g = df.groupby(["c", "src"])
    rates = g[["name_exact_raw", "name_exact_norm", "indic", "a2_empty", "house_eq", "domain", "dba"]].mean()
    q = g[["name_tsr", "addr_tsr"]].quantile([0.05, 0.25, 0.5]).unstack()
    print(rates.round(3).to_string())
    print(q.round(1).to_string())
    low = (df[~df.indic].name_tsr < 60).mean()
    print(f"\nnon-Indic true pairs with name token-set < 60: {low:.4f}")
    print("\nLow name-similarity true pairs (non-Indic):")
    print(df[(~df.indic) & (df.name_tsr < 50)].sample(15, random_state=SEED)[["n1", "n2", "a1", "a2"]].to_string())
    # histogram for figure 3
    bins = np.arange(0, 105, 5)
    hist = {c: (np.histogram(df[df.c == c].name_tsr, bins=bins)[0] / (df.c == c).sum() * 100).tolist() for c in ("US", "India")}
    R["pair_similarity"] = {
        "rates": {f"{c}|{s}": v for (c, s), v in rates.to_dict("index").items()},
        "quantiles": {f"{c}|{s}": {f"{m}_q{p}": float(q.loc[(c, s), (m, p)]) for m in ("name_tsr", "addr_tsr") for p in (0.05, 0.25, 0.5)}
                      for (c, s) in q.index},
        "non_indic_name_tsr_lt60": float(low),
        "name_tsr_hist": hist, "name_tsr_bins": bins.tolist(),
    }


# --------------------------------------------------------------------------------------
# Step 5: distractors & S1 duplicate names
# --------------------------------------------------------------------------------------
def step_distractors(D, R):
    header("5. Distractors and duplicate names in S1")
    s1 = D.get("train_s1").copy()
    o = D.others().copy()
    matched = set(D.pairs().m)
    o["matched"] = o.entity_id.isin(matched)
    s1["nn"], o["nn"] = s1.business_name.map(norm_name), o.business_name.map(norm_name)
    o["indic"] = o.business_name.map(is_indic)
    nnset = set(s1.nn)
    r = {
        "indic_share_matched": float(o[o.matched].indic.mean()),
        "indic_share_unmatched": float(o[~o.matched].indic.mean()),
        "addr_empty_matched": float((o[o.matched].business_address == "").mean()),
        "addr_empty_unmatched": float((o[~o.matched].business_address == "").mean()),
        "normname_in_s1_matched": float(o[o.matched].nn.isin(nnset).mean()),
        "normname_in_s1_unmatched": float(o[~o.matched].nn.isin(nnset).mean()),
        "unmatched_share": float((~o.matched).mean()),
        "s1_shares_normname": float(s1.nn.duplicated(keep=False).mean()),
        "s1_normname_group_sizes": {int(k): int(v) for k, v in s1.nn.value_counts().value_counts().sort_index().head(10).items()},
        "largest_s1_name_groups": s1.nn.value_counts().head(10).to_dict(),
    }
    for k, v in r.items():
        print(f"  {k}: {v}")
    dup = s1[s1.nn.duplicated(keep=False)]
    print("\nExample S1 groups sharing a normalised name:")
    pick = dup.nn.drop_duplicates().sample(6, random_state=3)
    print(dup[dup.nn.isin(pick)].sort_values("nn")[["entity_id", "business_name", "business_address", "country"]].to_string())
    s1i = s1.set_index("nn")
    print("\nDistractors (unmatched S2/S3) with their S1 name twins:")
    for _, row in o[(~o.matched) & (~o.indic) & (o.nn.isin(nnset))].sample(10, random_state=5).iterrows():
        print(f"\nDISTRACTOR {row.entity_id} | {row.business_name} | {row.business_address}")
        for _, t in s1i.loc[[row.nn]].head(3).iterrows():
            print(f"    S1 twin {t.entity_id} | {t.business_name} | {t.business_address}")
    R["distractors"] = r


# --------------------------------------------------------------------------------------
# Step 6: hard negatives - house-number and legal-suffix signals (US)
# --------------------------------------------------------------------------------------
def step_hard_negatives(D, R, n=400_000, addr_min=70):
    header("6. Hard negatives: house number and legal suffix (US)")
    from rapidfuzz import fuzz
    s1 = D.get("train_s1")
    o = D.others()
    owner = dict(zip(D.pairs().m, D.pairs().source1_entity_id))
    s1u = s1[s1.country == "US"][["entity_id", "business_name", "business_address"]].rename(
        columns={"entity_id": "s1id", "business_name": "n1", "business_address": "a1"})
    s1u["nn"] = s1u.n1.map(norm_name)
    ou = o[o.country == "US"].sample(n, random_state=SEED).copy()
    ou["owner"] = ou.entity_id.map(owner)
    ou["nn"] = ou.business_name.map(norm_name)
    m = ou.merge(s1u, on="nn")
    m["pos"] = m.owner == m.s1id
    m["atsr"] = [fuzz.token_set_ratio(a.lower(), b.lower()) for a, b in zip(m.a1, m.business_address)]
    m["hdiff"] = (m.a1.map(first_number) - m.business_address.map(first_number)).abs()
    m["leg_eq"] = [legal_suffixes(a) == legal_suffixes(b) for a, b in zip(m.n1, m.business_name)]
    close = m[m.atsr >= addr_min]
    r = {"n_close_pairs": len(close), "close_pos_rate": float(close.pos.mean())}
    print(f"same-norm-name pairs with address token-set >= {addr_min}: n={len(close):,}, positive rate={close.pos.mean():.3f}")
    for lab, g in close.groupby("pos"):
        hd = g.hdiff.dropna()
        k = "pos" if lab else "neg"
        r[k] = {"n": len(g), "hdiff_eq0": float((hd == 0).mean()), "hdiff_1_10": float(((hd > 0) & (hd <= 10)).mean()),
                "hdiff_gt10": float((hd > 10).mean()), "legal_eq": float(g.leg_eq.mean()), "addr_tsr_median": float(g.atsr.median())}
        print(f"  {k}: " + " ".join(f"{a}={b:.3f}" if isinstance(b, float) else f"{a}={b}" for a, b in r[k].items()))
    hdn = close[~close.pos].hdiff.dropna()
    r["neg_hdiff_counts_0_12"] = {int(k): int(v) for k, v in hdn[hdn <= 12].value_counts().sort_index().items()}
    # House-number difference over all US true pairs in the 200k sample.
    pf = D._mem.get("pairfeat")
    if pf is None:
        pf = pair_features(D)
        D._mem["pairfeat"] = pf
    pf = pf[pf.c == "US"]
    hd = (pf.a1.map(first_number) - pf.a2.map(first_number)).abs().dropna()
    r["true_us_hdiff_eq0"] = float((hd == 0).mean())
    r["true_us_hdiff_quantiles"] = {str(k): float(v) for k, v in hd.quantile([0.5, 0.8, 0.9, 0.95, 0.97, 0.99]).items()}
    r["true_us_hdiff_counts_1_12"] = {int(k): int(v) for k, v in hd[(hd > 0) & (hd <= 12)].value_counts().sort_index().items()}
    print(f"  true US pairs: diff==0 {r['true_us_hdiff_eq0']:.3f}, quantiles {r['true_us_hdiff_quantiles']}")
    print(f"  true diff counts 1..12: {r['true_us_hdiff_counts_1_12']}")
    print(f"  NEG diff counts 0..12 : {r['neg_hdiff_counts_0_12']}")
    R["hard_negatives"] = r


# --------------------------------------------------------------------------------------
# Step 7: blocking-key recall and volume
# --------------------------------------------------------------------------------------
def step_blocking(D, R):
    header("7. Blocking: recall and candidate volume of simple exact keys")
    s1, o = D.get("train_s1"), D.others()

    def K(df):
        k = pd.DataFrame([blocking_keys(a, nm) for a, nm in zip(df.business_address, df.business_name)])
        k["id"], k["c"] = df.entity_id.values, df.country.values
        return k

    t = time.time()
    k1, ko = K(s1), K(o)
    print(f"  keys built in {time.time() - t:.0f}s")
    p = D.pairs().merge(k1.add_suffix("_1"), left_on="source1_entity_id", right_on="id_1") \
                 .merge(ko.add_suffix("_2"), left_on="m", right_on="id_2")
    cols = ["k_name", "k_num_word", "k_num_2words", "k_nametok_num"]
    hit = pd.DataFrame({c: (p[c + "_1"] == p[c + "_2"]) & p[c + "_1"].notna() for c in cols})
    r = {"recall": hit.mean().to_dict(), "recall_union_all": float(hit.any(axis=1).mean()),
         "recall_union_name_numword": float((hit.k_name | hit.k_num_word).mean()),
         "recall_union_3": float((hit.k_name | hit.k_num_word | hit.k_nametok_num).mean()), "volume": {}}
    for c in cols:
        a = k1.groupby(["c", c]).size().rename("a")
        b = ko.groupby(["c", c]).size().rename("b")
        j = pd.concat([a, b], axis=1, join="inner")
        prod = j.a * j.b
        r["volume"][c] = {"pairs": int(prod.sum()), "per_s1": float(prod.sum() / len(s1)), "max_block": int(prod.max())}
    # Upper-bound macro F0.5 with the 3-key union and a perfect matcher on its candidates.
    p["hit"] = hit.k_name | hit.k_num_word | hit.k_nametok_num
    per = p.groupby("source1_entity_id").hit.mean()
    gt = D.get("gt")
    rec = gt.source1_entity_id.map(per)
    f = np.where(rec.isna(), 1.0, np.where(rec > 0, 1.25 * rec / (0.25 + rec), 0.0))
    r["upper_bound_f05_union3"] = float(f.mean())
    r["baseline_all_empty_f05"] = float((gt.matched_entity_ids == "").mean())
    for k, v in r.items():
        print(f"  {k}: {v}")
    R["blocking"] = r


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


def make_figures(R, fig_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(fig_dir, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                         "figure.dpi": 200, "savefig.bbox": "tight"})

    def save(ax, title, fn, ylab=None):
        ax.set_title(title, loc="left", color=INK, fontsize=10, fontweight="bold")
        if ylab:
            ax.set_ylabel(ylab)
        ax.figure.savefig(os.path.join(fig_dir, fn))
        plt.close(ax.figure)

    def bars(ax, x, y, c, w=0.8, **k):
        return ax.bar(x, y, width=w, color=c, edgecolor="white", linewidth=1.5, **k)

    made = []
    if "gt" in R:  # matches per S1
        d = {int(k): v for k, v in R["gt"]["n_matches_dist"].items()}
        tot = sum(d.values())
        fig, ax = plt.subplots(figsize=(6.4, 2.6))
        x = list(d)
        y = np.array(list(d.values())) / tot * 100
        bars(ax, x, y, BLUE)
        ax.set_xticks(x)
        ax.set_xlabel("Number of S2+S3 matches per S1 entity")
        ax.grid(axis="x", visible=False)
        for xi, yi in zip(x, y):
            if xi in (0, 3):
                ax.text(xi, yi + 0.5, f"{yi:.1f}%", ha="center", color=INK, fontsize=8)
        ax.text(0, y[0] + 3, "singletons", ha="center", color=INK2, fontsize=7.5)
        save(ax, f"Matches per S1 entity (train, share of {tot / 1e6:.2f}M entities)", "matches.png", "% of S1 entities")
        made.append("matches.png")

    if "overview" in R:  # country mix
        labs = ["Train S1", "Train S2", "Train S3", "Test S1", "Test S2", "Test S3"]
        v = np.array([[R["overview"][k]["country"].get(c, 0) for c in ("US", "India", "France")] for k in FILES], float)
        v = v / v.sum(1, keepdims=True) * 100
        fig, ax = plt.subplots(figsize=(6.4, 2.6))
        left = np.zeros(len(labs))
        for i, (nme, col) in enumerate(zip(["US", "India", "France"], [BLUE, ORANGE, AQUA])):
            ax.barh(labs, v[:, i], left=left, color=col, edgecolor="white", linewidth=2, label=nme, height=0.65)
            for j in range(len(labs)):
                if v[j, i] > 6:
                    ax.text(left[j] + v[j, i] / 2, j, f"{v[j, i]:.0f}%", ha="center", va="center", color="white", fontsize=7.5, fontweight="bold")
            left += v[:, i]
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("% of records")
        ax.legend(ncol=3, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=8)
        ax.set_title("Country mix by file", loc="left", color=INK, fontsize=10, fontweight="bold", pad=18)
        fig.savefig(os.path.join(fig_dir, "country.png"))
        plt.close(fig)
        made.append("country.png")

    if "pair_similarity" in R:  # name similarity of true pairs
        ps = R["pair_similarity"]
        b = np.array(ps["name_tsr_bins"])
        fig, ax = plt.subplots(figsize=(6.4, 2.6))
        for c, col in (("US", BLUE), ("India", ORANGE)):
            ax.plot(b[:-1] + 2.5, ps["name_tsr_hist"][c], color=col, lw=2, marker="o", ms=4, label=c)
        ax.set_yscale("log")
        ax.set_xlabel("Name token-set similarity (0-100) after normalisation")
        ax.legend(frameon=False)
        save(ax, "Name similarity of TRUE matched pairs (200k sample)", "namesim.png", "% of pairs (log scale)")
        made.append("namesim.png")

    if "hard_negatives" in R:
        h = R["hard_negatives"]
        neg = {int(k): v for k, v in h["neg_hdiff_counts_0_12"].items()}
        pos = {int(k): v for k, v in h["true_us_hdiff_counts_1_12"].items()}
        fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.5))
        bars(axs[0], list(neg), list(neg.values()), ORANGE)
        axs[0].set_xticks(range(0, 13))
        axs[0].set_title("Hard negatives", loc="left", fontsize=9, color=INK)
        axs[0].set_xlabel("|house no. difference|")
        axs[0].set_ylabel("pairs")
        axs[0].grid(axis="x", visible=False)
        bars(axs[1], list(pos), list(pos.values()), BLUE)
        axs[1].set_xticks(range(0, 13))
        axs[1].set_title(f"True matches (diff > 0 only; {h['true_us_hdiff_eq0'] * 100:.1f}% have diff = 0)", loc="left", fontsize=9, color=INK)
        axs[1].set_xlabel("|house no. difference|")
        axs[1].grid(axis="x", visible=False)
        fig.suptitle("House-number difference: the distractor fingerprint (US, same-name & similar-address pairs)",
                     x=0.02, ha="left", fontsize=10, fontweight="bold", color=INK, y=1.04)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "housediff.png"))
        plt.close(fig)
        made.append("housediff.png")

        fig, ax = plt.subplots(figsize=(6.4, 2.5))
        cats = ["House no. equal", "House no. differs by 1-10", "Legal suffix identical"]
        p = [h["pos"][k] * 100 for k in ("hdiff_eq0", "hdiff_1_10", "legal_eq")]
        n = [h["neg"][k] * 100 for k in ("hdiff_eq0", "hdiff_1_10", "legal_eq")]
        xx = np.arange(3)
        bars(ax, xx - 0.2, p, BLUE, w=0.38, label="True matches")
        bars(ax, xx + 0.2, n, ORANGE, w=0.38, label="Hard negatives")
        for i in range(3):
            ax.text(xx[i] - 0.2, p[i] + 2, f"{p[i]:.1f}%", ha="center", fontsize=8, color=INK)
            ax.text(xx[i] + 0.2, n[i] + 2, f"{n[i]:.1f}%", ha="center", fontsize=8, color=INK)
        ax.set_xticks(xx, cats)
        ax.set_ylim(0, 115)
        ax.set_yticks(range(0, 101, 20))
        ax.grid(axis="x", visible=False)
        ax.legend(frameon=False, ncol=2, loc="upper center")
        save(ax, "Separating true matches from hard negatives", "signals.png", "% of pairs")
        made.append("signals.png")

    if "blocking" in R:
        bl = R["blocking"]
        k = ["Normalised name (exact)", "House no. + 1st street word", "House no. + 2 street words", "1st name token + house no.", "Union of all four"]
        r = [bl["recall"][c] * 100 for c in ("k_name", "k_num_word", "k_num_2words", "k_nametok_num")] + [bl["recall_union_all"] * 100]
        fig, ax = plt.subplots(figsize=(6.4, 2.4))
        ax.barh(k, r, color=[BLUE] * 4 + [AQUA], edgecolor="white", height=0.6)
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        for i, v in enumerate(r):
            ax.text(v + 1, i, f"{v:.1f}%", va="center", fontsize=8, color=INK)
        ax.axvline(97, color=INK2, lw=1)
        ax.text(96, -0.55, "target ≥97%", ha="right", fontsize=7.5, color=INK2)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel(f"Pair recall on {R['gt']['total_pairs'] / 1e6:.2f}M true train pairs" if "gt" in R else "Pair recall on true train pairs")
        save(ax, "Recall of simple exact blocking keys", "blocking.png")
        made.append("blocking.png")
    print(f"figures written to {fig_dir}: {made}")


# --------------------------------------------------------------------------------------
# PDF report
# --------------------------------------------------------------------------------------
def build_report(R, fig_dir, pdf_path):
    need = ("overview", "gt", "pair_similarity", "distractors", "hard_negatives", "blocking")
    missing = [k for k in need if k not in R]
    if missing:
        print(f"report skipped: results missing for steps {missing} (run all steps)")
        return
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.pdfmetrics import registerFontFamily
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import HRFlowable, Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    F = "/usr/share/fonts/truetype/dejavu/"
    for nm, fn in (("DV", "DejaVuSans.ttf"), ("DVB", "DejaVuSans-Bold.ttf"), ("DVI", "DejaVuSans-Oblique.ttf"), ("DVM", "DejaVuSansMono.ttf")):
        pdfmetrics.registerFont(TTFont(nm, F + fn))
    registerFontFamily("DV", normal="DV", bold="DVB", italic="DVI", boldItalic="DVB")
    C_INK, C_SUB, C_ACC = colors.HexColor(INK), colors.HexColor(INK2), colors.HexColor(BLUE)
    C_LINE, C_BG = colors.HexColor("#d9d8d3"), colors.HexColor("#f4f3ef")
    body = ParagraphStyle("b", fontName="DV", fontSize=9.2, leading=13.2, textColor=C_INK, spaceAfter=5)
    bul = ParagraphStyle("bu", parent=body, leftIndent=12, bulletIndent=2, spaceAfter=2.5)
    h1 = ParagraphStyle("h1", fontName="DVB", fontSize=14, leading=18, textColor=C_INK, spaceBefore=10, spaceAfter=6, keepWithNext=1)
    h2 = ParagraphStyle("h2", fontName="DVB", fontSize=10.5, leading=14, textColor=C_ACC, spaceBefore=8, spaceAfter=4, keepWithNext=1)
    cap = ParagraphStyle("c", parent=body, fontSize=8, leading=11, textColor=C_SUB, spaceAfter=10)
    cell = ParagraphStyle("cell", fontName="DV", fontSize=7.8, leading=10, textColor=C_INK)
    cellb = ParagraphStyle("cellb", parent=cell, fontName="DVB")
    title = ParagraphStyle("t", fontName="DVB", fontSize=22, leading=27, textColor=C_INK)
    S = []

    def P(t, s=body):
        S.append(Paragraph(t, s))

    def BUL(items):
        for t in items:
            S.append(Paragraph(t, bul, bulletText="•"))

    def T(rows, widths):
        data = [[Paragraph(str(c), cellb if i == 0 else cell) for c in r] for i, r in enumerate(rows)]
        t = Table(data, colWidths=[w * mm for w in widths], repeatRows=1)
        t.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, -1), 0.4, C_LINE), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                               ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                               ("BACKGROUND", (0, 0), (-1, 0), C_BG), ("LINEBELOW", (0, 0), (-1, 0), 0.8, C_SUB)]))
        S.append(t)
        S.append(Spacer(1, 8))

    def IMG(fn, w=170, c=None):
        path = os.path.join(fig_dir, fn)
        iw, ih = ImageReader(path).getSize()
        S.append(Image(path, width=w * mm, height=w * mm * ih / iw))
        if c:
            P(c, cap)
        else:
            S.append(Spacer(1, 8))

    def KEY(t):
        k = Table([[Paragraph(t, ParagraphStyle("k", parent=body, spaceAfter=0))]], colWidths=[170 * mm])
        k.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), C_BG), ("LINEBEFORE", (0, 0), (0, -1), 2.5, C_ACC),
                               ("LEFTPADDING", (0, 0), (-1, -1), 8), ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
        S.append(k)
        S.append(Spacer(1, 8))

    pct = lambda x, d=1: f"{x * 100:.{d}f}%"
    ov, gt, ps, di, hn, bl = (R[k] for k in need)
    t1 = ov["test_s1"]["country"]
    fr_share = t1.get("France", 0) / sum(t1.values())
    ind = lambda k: ov[k]["name_devanagari"] + ov[k]["name_other_indic"]
    ind_lo = min(ind(k) / (ov[k]["country"].get("India", 0) / ov[k]["rows"]) for k in ("train_s2", "train_s3"))
    ind_hi = max(ind(k) / (ov[k]["country"].get("India", 0) / ov[k]["rows"]) for k in ("train_s2", "train_s3"))
    maxm = max(int(k) for k in gt["n_matches_dist"])

    S.append(Spacer(1, 6))
    P("Exploratory Data Analysis", title)
    P("Amazon ML Challenge 2026 — Business Entity Resolution", ParagraphStyle("st", parent=body, fontSize=12, leading=16, textColor=C_SUB))
    tot_rows = sum(v["rows"] for v in ov.values())
    P(f"Scope: all six source files and the training ground truth (~{tot_rows / 1e6:.0f}M records). Every figure in this report was computed from the provided data by code/eda.py.", cap)
    S.append(HRFlowable(width="100%", color=C_LINE, thickness=0.8, spaceAfter=8))
    P("Key findings", h1)
    BUL([
        f"<b>Heavily linked data.</b> Only {pct(gt['singleton_frac'])} of Source 1 entities are singletons; the average entity has {gt['mean_matches']:.2f} matches (up to {maxm}). Recall matters as well as precision.",
        f"<b>Every S2/S3 record belongs to at most one S1 entity</b> ({gt['total_pairs'] / 1e6:.2f}M matched IDs, all unique). About {pct(di['unmatched_share'], 0)} of S2/S3 records match nothing and act as distractors.",
        f"<b>Country agrees in {pct(gt['country_agreement'], 0)} of true pairs</b>, so it is a free blocking key. The test set adds France ({pct(fr_share, 0)} of test S1), which has no labels in training.",
        f"<b>Name alone is unreliable.</b> {pct(di['s1_shares_normname'], 0)} of S1 entities share a normalised name with another S1 entity, and about {pct(ps['non_indic_name_tsr_lt60'], 0)} of non-Indic true matches have a name similarity below 60 (a brand, an acronym or a domain).",
        f"<b>Distractors are adversarial near-copies</b>: same name and street, a house number off by exactly 1–5, 7, 9 or 11, and a changed legal suffix. House-number equality ({pct(hn['pos']['hdiff_eq0'])} of true pairs vs {pct(hn['neg']['hdiff_eq0'])} of hard negatives) and legal-suffix equality ({pct(hn['pos']['legal_eq'], 0)} vs {pct(hn['neg']['legal_eq'])}) are the strongest precision signals.",
        f"<b>Simple exact blocking keys reach only {pct(bl['recall_union_all'])} pair recall</b> (an upper-bound macro F0.5 of about {bl['upper_bound_f05_union3']:.2f}). Fuzzy or embedding-based candidate generation is required.",
        f"<b>India names in S2/S3 are {ind_lo * 100:.0f}–{ind_hi * 100:.0f}% non-Latin script</b> (Devanagari, Telugu, Tamil, Kannada, Bengali): transliterations of the English name. Transliteration is a necessary preprocessing step.",
        "<b>No leakage</b>: entity IDs and row order are uncorrelated with matches, and train and test share no IDs.",
    ])

    P("1. Problem and evaluation", h1)
    P("Source 1 is a deduplicated reference list of businesses. For every S1 entity we must return all S2 and S3 records that refer to the same real-world business, which may be none, one or many. The three sources share no identifiers; each record has only <i>business_name</i>, <i>business_address</i> and <i>country</i>.")
    P('The metric is <b>F0.5, computed per S1 entity and then macro-averaged</b> over all entities, singletons included. F0.5 = 1.25·P·R / (0.25·P + R) weights precision twice as heavily as recall. A singleton scores 1.0 for an empty prediction and 0.0 for any prediction. Two files are submitted: <font name="DVM">matching_results.tsv</font> (scored) and <font name="DVM">candidate_pairs.tsv</font> (used to audit blocking recall and reduction ratio). Constraints: MIT/Apache-2.0 models of at most 8B parameters, and no external data lookup (including geocoding).')
    KEY(f"<b>Implication.</b> Because singletons are only {pct(gt['singleton_frac'])} and the average entity has {gt['mean_matches']:.1f} matches, predicting only the most confident match leaves a lot of F0.5 on the table. With perfect precision and 50% recall an entity scores 0.83; with 100% recall it scores 1.0.")

    P("2. Dataset overview", h1)
    rows = [["File", "Rows", "US", "India", "France", "Name mean len", "Addr mean len"]]
    for k in FILES:
        o = ov[k]
        c = o["country"]
        rows.append([k.replace("_s", "_source"), f"{o['rows']:,}", f"{c.get('US', 0):,}", f"{c.get('India', 0):,}",
                     f"{c['France']:,}" if c.get("France") else "—", f"{o['name_len_mean']:.1f}", f"{o['addr_len_mean']:.1f}"])
    T(rows, [30, 22, 22, 22, 20, 27, 27])
    IMG("country.png", 160, "Figure 1. The country mix shifts between train and test: France (unseen in training) appears only in the test files. S2 and S3 are each about 2.3–2.9× the size of S1.")
    P('File integrity: read every column as a string with <font name="DVM">keep_default_na=False</font>; otherwise names such as "NULL" or "NA" are silently turned into missing values.')

    P("3. Ground-truth structure", h1)
    IMG("matches.png", 160, f"Figure 2. Distribution of matches per S1 entity; only {pct(gt['singleton_frac'])} are singletons.")
    bc = gt["by_country"]
    T([["Statistic", "Value"],
       ["Singleton S1 entities", f"{gt['n_singletons']:,} ({pct(gt['singleton_frac'], 2)}); " + ", ".join(f"{c} {pct(v['singleton'], 2)}" for c, v in bc.items())],
       ["Mean matches per S1", f"{gt['mean_matches']:.2f} (" + ", ".join(f"{c} {v['mean_n']:.2f}" for c, v in bc.items()) + ")"],
       ["Max matches from S2 / S3 for one entity", f"{max(int(k) for k in gt['n_s2_dist'])} / {max(int(k) for k in gt['n_s3_dist'])}, so S2 and S3 are not deduplicated"],
       ["Total true pairs", f"{gt['total_pairs']:,}"],
       ["S2/S3 IDs matched to more than one S1", f"{gt['total_pairs'] - gt['unique_matched_ids']:,}"],
       ["Share of S2 / S3 records matched to some S1", f"{pct(gt['s2_matched_share'])} / {pct(gt['s3_matched_share'])}"],
       ["Country agreement in true pairs", pct(gt["country_agreement"])],
       ["Correlation of numeric ID / row position between pair members", f"{gt['id_corr']:.4f} / {gt['row_pos_corr']:.4f} (no leakage)"]], [85, 85])

    P("4. Field quality and script mix", h1)
    rows = [["File", "Empty address", "Literal NULL / None in address", "Non-ASCII name", "Devanagari name", "Other Indic name", "US address all-caps"]]
    for k in FILES:
        o = ov[k]
        rows.append([k.replace("_s", " S"), pct(o["addr_empty"]), pct(o["addr_null_literal"]), pct(o["name_non_ascii"]),
                     pct(o["name_devanagari"]), pct(o["name_other_indic"]), pct(o["by_country"]["US"]["all_upper"])])
    T(rows, [20, 22, 30, 22, 24, 24, 28])
    P("Non-ASCII names in test S1 are French accented names. S1 is otherwise clean, title-cased Latin text.", cap)
    ti, tu = ov["train_s1"]["by_country"]["India"], ov["train_s1"]["by_country"]["US"]
    BUL([f"<b>Postal codes are nearly absent</b>: 6-digit PINs appear in {pct(ti['pin6'])} of India S1 addresses and 5-digit ZIPs in {pct(tu['zip5'], 0)} of US S1 addresses, so postal-code blocking is not viable.",
         f"<b>India addresses are long and unstructured</b> ({ti['n_commas']:.1f} commas on average vs {tu['n_commas']:.1f} for the US), and {pct(ti['landmark'], 0)} of India S1 addresses contain landmark phrases (Near, Opp, Behind, Next to).",
         f"<b>Empty addresses concentrate in true matches</b> ({pct(di['addr_empty_matched'])} of matched S2/S3 records vs {pct(di['addr_empty_unmatched'])} of unmatched ones). For these pairs the model must rely on the name.",
         f"<b>Duplicate names within S1 are common</b>: {pct(ov['train_s1']['dup_name'], 0)} of S1 rows repeat an exact raw name, and {pct(di['s1_shares_normname'], 0)} share a normalised name."])

    P("5. Noise patterns (observed examples)", h1)
    P("Real S1 → S2/S3 pairs taken from the ground truth (see step 3 output for more):")
    T([["Pattern", "S1 value", "Matched S2/S3 value"],
       ["Case, prefix and legal suffix", "CH Dynamic Auto Glass", "The CH Dynamic Auto Glass · CH Dynamic Auto Glass  Inc."],
       ["DBA / trade name", "CH Dynamic Auto Glass", "Rizatavo Co doing business as CH Dynamic Auto Glass"],
       ["Injected accents, punctuation, &amp;", "Bureau of Parks and Recreation", "Bureau óf Parks and-Recreation · Bureau of Parks &amp; Recreation Authority"],
       ["Typos", "Blue Alliance / First Networks Inc", "Blue Alliatnae / FIRST NELWKSB INC"],
       ["Domain / handle as name", "Phoenix Vidyalaya · Youth Unified Program", "phoenixvidyalaya.com, ph0enixvidyalaya.com · #youthunified"],
       ["Junk tokens", "Phoenix Vidyalaya", "Phoenix  Center (ID: 47290) · Dr Phoenix Vídyalaya"],
       ["Word-order change", "Jay It Private Limited", "Private Jay It Limited"],
       ["Script change (India)", "Premier Foundation Private Limited", "[same name written in Devanagari]"],
       ["Unrelated name", "White Nuclear Inc", "Jaxorbi (identical address)"],
       ["House-number noise", "11265 Sunrise Gold Circle", "011265 SUNRISE GOLD CIRCLE · ##15140 ELDERFLOWER LANE"],
       ["Component reorder, full state", "5212 Tinkers Creek Place, Clinton, MD", "MD, 5212 TINKERS CREEK PLACE, CLINTON · …, Clnton, Maryland"],
       ["City alias / misspelling", "Pune, Maharashtra · Penfield, NY", "Poona, Pune, MH · Pennfield, New York"],
       ["State in native script", "…, Karnataka", "…, [Karnataka in Kannada script] / KA"],
       ["Partial / empty address", "Sno 32/2/1 Hno 1048, Gulabnagar, …, Pune", "BLOCK A-825 SNO 32/2/1 HNO 1048, PUNE, … · (empty)"],
       ["French abbreviations (test)", "Rue de Dieppe, Lille", "63 R. DE DIEPPE, LILLE · 51 R DE LA PLANCHE AU GUE"]], [38, 55, 77])
    groups = ["India|S2", "India|S3", "US|S2", "US|S3"]
    rt = ps["rates"]
    rows = [["Rate among true pairs (200k sample)", "India S2", "India S3", "US S2", "US S3"]]
    for lab, key in (("Raw name exactly equal", "name_exact_raw"), ("Normalised name exactly equal", "name_exact_norm"),
                     ("Name in Indic script", "indic"), ("First house number equal", "house_eq"), ("Empty address", "a2_empty"),
                     ("Domain-style name", "domain"), ("DBA pattern", "dba")):
        rows.append([lab] + [pct(rt[g][key]) for g in groups])
    T(rows, [70, 25, 25, 25, 25])
    IMG("namesim.png", 160, f"Figure 3. Name similarity of true pairs. India has a heavy low tail, mostly Indic-script names that raw string metrics cannot compare. About {pct(ps['non_indic_name_tsr_lt60'], 0)} of non-Indic true pairs score below 60.")
    qq = ps["quantiles"]
    rows = [["Similarity (true pairs)", "India S2", "India S3", "US S2", "US S3"]]
    for lab, key in (("Name token-set, 5th percentile", "name_tsr_q0.05"), ("Name token-set, median", "name_tsr_q0.5"),
                     ("Address token-set, 5th percentile", "addr_tsr_q0.05"), ("Address token-set, median", "addr_tsr_q0.5")):
        rows.append([lab] + [f"{qq[g][key]:.0f}" for g in groups])
    T(rows, [70, 25, 25, 25, 25])

    P("6. Hard negatives: how distractors are built", h1)
    P(f"US S2/S3 records were joined to every US S1 entity with the same normalised name, and pairs with an address token-set similarity of at least 70 were kept ({hn['n_close_pairs']:,} pairs, {pct(hn['close_pos_rate'], 0)} of them true). The rest are near-copies that differ only in small details:")
    T([["Distractor (S2)", "Closest S1 (not its true entity)"],
       ["Schulman's Industries Center Corp · 18641 MANN LN, FAIRHOPE, AL", "Schulman's Industries Center · 18634 Mann Lane, AL, Fairhope"],
       ["Golden Electronics Dynamics, Llc · 142 LINCOLN ST, LEOMINSTER, MA", "Golden Electronics Dynamics, Corp · 135 Lincoln Street, Leominster, MA"],
       ["Mann Safe Activate Inc · 615 Eye St, HARLINGEN, TX", "Mann Safe Activate LLC · 606 Eye Street, Harlingen, TX"],
       ["Bobby Banuelos Quality Diagnostics Ltd · 2813- YANCEYVILLE STREET", "Bobby Banuelos Quality Diagnostics Inc · 2811 Yanceyville Street, Unit B"]], [85, 85])
    IMG("housediff.png", 170, "Figure 4. Among hard negatives the house-number difference is almost always exactly 1, 2, 3, 4, 5, 7, 9 or 11; 6, 8, 10 and 12 almost never occur. True pairs mostly keep the same number; their non-zero differences are mostly 1–2 (digit noise) or large (a unit or plot number parsed as the house number).")
    IMG("signals.png", 150, "Figure 5. Exact house-number equality and exact legal-suffix equality separate the classes well. Hard negatives almost always change the legal suffix (LLC → Corp, Inc → Ltd).")
    KEY(f"<b>Implication.</b> A name-plus-address fuzzy score will rank these distractors as matches, and on a singleton a single false merge costs the full 1.0. Feature engineering must include a robust house-number parser, the absolute difference and membership of the difference in {{1,2,3,4,5,7,9,11}}, and legal-suffix extraction with equality and compatibility flags. A changed suffix alone is not decisive: {pct(1 - hn['pos']['legal_eq'], 0)} of true pairs also change it.")

    P("7. Blocking (candidate generation) analysis", h1)
    P(f"Within-country exact keys were evaluated against all {gt['total_pairs'] / 1e6:.2f}M true training pairs. Addresses were lower-cased, accents removed, common street words abbreviated and leading zeros stripped.")
    IMG("blocking.png", 160, "Figure 6. Pair recall of simple exact keys.")
    rows = [["Key (within country)", "Pair recall", "Candidate pairs", "Per S1", "Largest block"]]
    for lab, key in (("Normalised name", "k_name"), ("House no. + first street word", "k_num_word"),
                     ("House no. + two street words", "k_num_2words"), ("First name token + house no.", "k_nametok_num")):
        v = bl["volume"][key]
        rows.append([lab, pct(bl["recall"][key]), f"{v['pairs'] / 1e6:.1f}M", f"{v['per_s1']:.1f}", f"{v['max_block']:,}"])
    rows.append(["Union of all four", pct(bl["recall_union_all"]), "—", "—", "—"])
    T(rows, [60, 25, 30, 22, 33])
    P(f"With the name, house+street and name-token+house keys and a perfect matcher, the best achievable macro F0.5 is <b>{bl['upper_bound_f05_union3']:.3f}</b>. Predicting an empty list for every entity scores {bl['baseline_all_empty_f05']:.3f}.")
    KEY("<b>Implication.</b> The blocking stage caps recall and must combine several methods: (a) the exact keys above; (b) TF-IDF character n-gram (3–4) nearest neighbours on name + address within each country; (c) nearest neighbours on a small multilingual text encoder (MIT/Apache) to catch Indic-script and unrelated-name cases, which also generalises to France. Target at least 97% pair recall at about 20–50 candidates per S1, measured on a held-out split. Frequent-token keys need block-size caps.")

    P("8. Recommended preprocessing", h1)
    P("Reading and cleaning", h2)
    BUL(['Read with <font name="DVM">sep="\\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE</font>.',
         "Treat NULL, None, N/A and null inside addresses as empty tokens; add a flag for an empty address.",
         "Apply Unicode NFKD, strip combining accents and collapse repeated whitespace."])
    P("Names", h2)
    BUL(['Lower-case; &amp; → "and"; remove junk prefixes and suffixes (--, ***, #, The, Dr, Smt, "(ID: n)", "| www…", brackets).',
         'Split DBA names ("X doing business as Y", "dba", "d/b/a", "t/a") and keep both parts.',
         "Recover words from domains and handles (firstnetworks.com → firstnetworks) and use them for character-level comparison.",
         "<b>Extract the legal suffix as a separate field</b> rather than just deleting it, and map variants to one form: Pvt/Private, Ltd/Limited, L.L.C./LLC, Corp/Corporation, Co/Company, plus SARL/SAS/SASU/EURL/SCI/SA for France.",
         "Transliterate Indic scripts to Latin, either with an MIT/Apache transliteration library or with a character mapping learned from the aligned training pairs."])
    P("Addresses", h2)
    BUL(['Parse into house/plot number, street, unit, city, state and landmark. Strip leading zeros and "#"/"##" from numbers.',
         "Abbreviation tables per country: US (Street→St, Road→Rd …), India (H.No, Plot No, Opp, Nr), France (Rue→R, Boulevard→Bd, Avenue→Av, bis/ter).",
         "Map state names, codes and native-script state names to one canonical code, using a dictionary learned from the data.",
         'Build a city alias table from training pairs (Poona↔Pune, Calcutta↔Kolkata, "City of X"↔X) and use typo-tolerant city matching.',
         "Compare addresses as sorted component sets, since component order varies."])
    P("Generalising to France", h2)
    BUL(["Treat country as an open set. Every normalisation rule should be country-generic, with an optional country-specific table on top.",
         "Prefer features that do not depend on country (numeric equality, character n-gram similarity, embedding similarity) so the matcher transfers to unlabelled France."])

    P("9. Suggested modelling pipeline", h1)
    T([["Stage", "Approach"],
       ["Blocking", "Union of exact keys, TF-IDF character n-gram kNN and multilingual-embedding kNN (FAISS on GPU), all within country; cap block sizes. Write the final union to candidate_pairs.tsv."],
       ["Pair features", "Name: token-set / Jaro-Winkler / TF-IDF cosine, legal-suffix equal, domain and DBA flags, name frequency in S1. Address: house number equal, absolute difference, difference in {1,2,3,4,5,7,9,11}, street / city / state / unit similarity, empty flags. Source (S2/S3), country, embedding cosine."],
       ["Matcher", "Gradient-boosted trees (LightGBM/XGBoost) trained on candidate pairs, with hard negatives taken from the blocking output. Optionally, a small fine-tuned cross-encoder for uncertain pairs."],
       ["Post-processing", "Assign each S2/S3 record only to its best-scoring S1 (the ground truth is one-to-many). Tune the threshold for macro F0.5 on validation. Add transitive support: an S3 record almost identical to an accepted S2 record is also accepted."],
       ["Validation", "Hold out about 10% of S1 entities with all their candidates, compute macro F0.5 exactly as specified, and report blocking recall and reduction ratio separately."]], [30, 140])

    P("Appendix: method notes", h1)
    BUL(["Code: code/eda.py (this report is generated by it). Results are saved to results.json next to the PDF.",
         "Name normalisation: NFKD accent removal, lower-case, &amp;→and, punctuation removed, \"(ID: n)\" removed, legal and generic tokens dropped (inc, llc, ltd, pvt, private, corp, co, the, holdings, group, sarl, sas …).",
         f"Similarity statistics use a 200,000-pair random sample of true pairs (seed {SEED}). The hard-negative analysis uses a 400,000-record US sample of S2 ∪ S3 joined to S1 on normalised name.",
         "House number is the first integer in the address string, which explains the long tail of large differences for true pairs where a unit or plot number comes first.",
         "Examples are verbatim records from the training data. Indic-script strings are described in words because the report font lacks those glyphs."])

    def deco(c, d):
        c.saveState()
        c.setFont("DV", 7.5)
        c.setFillColor(C_SUB)
        c.drawString(20 * mm, 12 * mm, "Amazon ML Challenge 2026 · EDA")
        c.drawRightString(190 * mm, 12 * mm, f"{d.page}")
        c.restoreState()

    doc = SimpleDocTemplate(pdf_path, pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm, topMargin=18 * mm, bottomMargin=20 * mm,
                            title="EDA — Amazon ML Challenge 2026 Entity Resolution")
    doc.build(S, onFirstPage=deco, onLaterPages=deco)
    print(f"report written to {pdf_path}")


# --------------------------------------------------------------------------------------
STEPS = {
    "overview": step_overview,
    "gt": step_gt,
    "examples": step_examples,
    "similarity": step_pair_similarity,
    "distractors": step_distractors,
    "hard_negatives": step_hard_negatives,
    "blocking": step_blocking,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=os.path.join(ROOT, "student_resource", "dataset"))
    ap.add_argument("--out", default=os.path.join(ROOT, "eda_output"), help="results.json, fig/ and EDA.pdf go here")
    ap.add_argument("--cache", default=None, help="parquet cache dir (default: <out>/cache)")
    ap.add_argument("--steps", nargs="+", choices=list(STEPS), default=list(STEPS))
    ap.add_argument("--no-report", action="store_true", help="skip figures and PDF")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    D = Data(a.data_dir, a.cache or os.path.join(a.out, "cache"))
    res_path = os.path.join(a.out, "results.json")
    R = json.load(open(res_path)) if os.path.exists(res_path) else {}
    for s in a.steps:
        t = time.time()
        STEPS[s](D, R)
        print(f"[{s} done in {time.time() - t:.0f}s]")
        with open(res_path, "w") as f:
            json.dump(R, f, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x))
    if not a.no_report:
        fig_dir = os.path.join(a.out, "fig")
        make_figures(R, fig_dir)
        build_report(R, fig_dir, os.path.join(a.out, "EDA.pdf"))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Stage-2 decision model: LightGBM stacked on top of the retrievers and rerankers.

For every (S1, candidate) pair of the 20% test split (top-20 of the fine-tuned retriever)
it builds features from
  * model scores (the ensemble): mDeBERTa reranker logits (v1, v2 if present) and cosine
    similarities from every embedding run (e5-large-ft, e5-large, e5-base, bge-m3; name-only
    and name+address);
  * ranks / context inside the S1's candidate list;
  * competition between S1s for the same record (how many S1s claim it, margin to the best
    other S1) - the learned version of the "abstain" rule;
  * structured comparisons: house number, legal form, names, addresses, flags.

The test split was never used to train the retriever or the rerankers, so it is used here with
5-fold cross-validation grouped by S1 entity; all reported numbers are out-of-fold.
Decision = probability threshold + each S2/S3 record to at most one S1 (optionally abstaining
when several S1s pass), tuned for per-S1 macro F0.5.

Usage:
    python code/stack_lgbm.py features          # build / refresh the feature table
    python code/stack_lgbm.py cv                # train + evaluate (all feature sets)
Outputs in experiments/stacker/.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import retrieval_experiments as rx  # noqa: E402

RR = os.path.join(rx.ROOT, "experiments", "reranker")
WORK = os.path.join(rx.ROOT, "experiments", "stacker")
FEAT = os.path.join(WORK, "features.parquet")
EMB_RUNS = [("e5-large-ft", "embed_text"), ("e5-large", "embed_text"), ("e5-large", "name_core"),
            ("e5-base", "embed_text"), ("e5-base", "name_core"), ("bge-m3", "embed_text"), ("bge-m3", "name_core")]
FIELDS = ["entity_id", "name_core", "name_clean", "legal_form", "house_no", "house_base", "unit", "addr_clean",
          "addr_numbers", "addr_empty", "is_indic", "is_domain", "has_dba"]
FINGERPRINT = [1, 2, 3, 4, 5, 7, 9, 11]


# --------------------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------------------
def rowwise_cos(qe, ce, qi, ci, chunk=1_000_000):
    out = np.empty(len(qi), dtype=np.float32)
    for s in range(0, len(qi), chunk):
        a = qe[qi[s:s + chunk]].astype(np.float32)
        b = ce[ci[s:s + chunk]].astype(np.float32)
        out[s:s + chunk] = np.einsum("ij,ij->i", a, b)
    return out


def group_rank(key, score, ascending=False):
    """Rank of each row's score inside its group (0 = best)."""
    return pd.Series(score).groupby(key).rank(ascending=ascending, method="first").values - 1


def competition(df, col, prefix):
    """For each candidate record: margin to the best OTHER S1 that also retrieved it."""
    s = df[col].values.astype(np.float64)
    codes, _ = pd.factorize(df.cand_id.values)
    order = np.lexsort((-s, codes))                       # by candidate, best score first
    sc, ss = codes[order], s[order]
    first = np.r_[True, sc[1:] != sc[:-1]]
    pos = np.arange(len(order)) - np.maximum.accumulate(np.where(first, np.arange(len(order)), 0))
    ng = codes.max() + 1
    top1 = np.full(ng, -30.0); top2 = np.full(ng, -30.0)
    top1[sc[pos == 0]] = ss[pos == 0]
    top2[sc[pos == 1]] = ss[pos == 1]
    rank = np.empty(len(s)); rank[order] = pos
    best_other = np.where(rank == 0, top2[codes], top1[codes])
    return {f"{prefix}_margin_other": s - best_other,
            f"{prefix}_best_other": best_other,
            f"{prefix}_rank_in_cand": rank}


def string_feats(a, b, prefix):
    from rapidfuzz import fuzz, process
    from rapidfuzz.distance import JaroWinkler
    f = {}
    for name, scorer in (("tsr", fuzz.token_set_ratio), ("ratio", fuzz.ratio), ("partial", fuzz.partial_ratio),
                         ("tsort", fuzz.token_sort_ratio)):
        f[f"{prefix}_{name}"] = process.cpdist(a, b, scorer=scorer, workers=-1).astype(np.float32)
    f[f"{prefix}_jw"] = process.cpdist(a, b, scorer=JaroWinkler.normalized_similarity, workers=-1).astype(np.float32)
    return f


def build_features():
    os.makedirs(WORK, exist_ok=True)
    t0 = time.time()
    df = pd.read_parquet(os.path.join(RR, "test_pairs.parquet"), columns=["s1_id", "cand_id", "rank", "cos", "label"])
    X = pd.DataFrame({"s1_id": df.s1_id.values, "cand_id": df.cand_id.values, "label": df.label.values.astype(np.int8)})
    X["ret_rank"] = df["rank"].values.astype(np.int16)

    # ---- reranker logits (ensemble members)
    for tag in ("", "_v2"):
        p = os.path.join(RR, f"test_logits{tag}.npy")
        if os.path.exists(p):
            X[f"rr{tag or '_v1'}"] = np.load(p)
    rr_cols = [c for c in X if c.startswith("rr_")]
    print(f"rerankers: {rr_cols}")

    # ---- cosine similarities from every embedding run
    gt_train, gt_test = rx.make_split()
    queries, corpus, _ = rx.build_eval_set(gt_train, gt_test, ["embed_text"])
    qpos = pd.Series(np.arange(len(queries)), index=queries.entity_id)
    cpos = pd.Series(np.arange(len(corpus)), index=corpus.entity_id)
    qi, ci = qpos.loc[X.s1_id].values, cpos.loc[X.cand_id].values
    for m, f in EMB_RUNS:
        qp, cp = (os.path.join(rx.OUT, "emb", f"{m}_{f}_{w}.npy") for w in ("queries", "corpus"))
        if os.path.exists(qp) and os.path.exists(cp):
            X[f"cos_{m}_{f}"] = rowwise_cos(np.load(qp, mmap_mode="r"), np.load(cp, mmap_mode="r"), qi, ci)
    cos_cols = [c for c in X if c.startswith("cos_")]
    X["cos_mean"] = X[cos_cols].mean(axis=1)
    print(f"cosines: {len(cos_cols)} runs ({time.time() - t0:.0f}s)")

    # ---- context inside the S1 list and competition across S1s
    for c in rr_cols + ["cos_e5-large-ft_embed_text"]:
        short = c.replace("cos_e5-large-ft_embed_text", "cosft")
        X[f"{short}_rank_in_s1"] = group_rank(X.s1_id.values, X[c].values)
        X[f"{short}_gap_to_s1_max"] = X[c].values - X.groupby("s1_id")[c].transform("max").values
        for k, v in competition(X, c, short).items():
            X[k] = v
    X["cand_degree"] = X.groupby("cand_id").s1_id.transform("size").values
    main_rr = rr_cols[0]
    X["cand_n_s1_pos"] = X.groupby("cand_id")[main_rr].transform(lambda s: (s > 0).sum()).values
    X["s1_n_pos"] = X.groupby("s1_id")[main_rr].transform(lambda s: (s > 0).sum()).values
    print(f"context features ({time.time() - t0:.0f}s)")

    # ---- structured comparisons
    parts = [rx.read_tsv(os.path.join(rx.PROC, "train", f"train_source{s}.tsv"), FIELDS) for s in (1, 2, 3)]
    Fd = pd.concat(parts, ignore_index=True).set_index("entity_id")
    A = Fd.loc[X.s1_id].reset_index(drop=True)
    B = Fd.loc[X.cand_id].reset_index(drop=True)
    X["cand_is_s3"] = X.cand_id.str.startswith("S3").astype(np.int8).values
    for c in ("addr_empty", "is_indic", "is_domain", "has_dba"):
        X[f"cand_{c}"] = (B[c].values == "1").astype(np.int8)
    X["s1_addr_ncomp"] = A.addr_clean.str.count(",").values + (A.addr_clean.values != "")
    X["cand_addr_ncomp"] = B.addr_clean.str.count(",").values + (B.addr_clean.values != "")
    ha, hb = A.house_base.values, B.house_base.values
    both = (ha != "") & (hb != "")
    X["house_both"] = both.astype(np.int8)
    X["house_eq"] = (both & (ha == hb)).astype(np.int8)
    X["house_full_eq"] = (both & (A.house_no.values == B.house_no.values)).astype(np.int8)
    num = both & np.array([len(x) < 10 and len(y) < 10 for x, y in zip(ha, hb)])
    d = np.full(len(X), -1.0)
    d[num] = np.abs(ha[num].astype(np.int64) - hb[num].astype(np.int64))
    X["house_diff"] = d
    X["house_fingerprint"] = np.isin(d, FINGERPRINT).astype(np.int8)
    X["house_len_eq"] = (both & np.array([len(x) == len(y) for x, y in zip(ha, hb)])).astype(np.int8)
    na = A.addr_numbers.str.split().values
    nb = B.addr_numbers.str.split().values
    inter = np.array([len(set(x) & set(y)) for x, y in zip(na, nb)])
    union = np.array([len(set(x) | set(y)) for x, y in zip(na, nb)])
    X["num_overlap"] = inter
    X["num_jaccard"] = np.divide(inter, union, out=np.zeros(len(X)), where=union > 0)
    X["unit_eq"] = ((A.unit.values != "") & (A.unit.values == B.unit.values)).astype(np.int8)
    X["unit_both"] = ((A.unit.values != "") & (B.unit.values != "")).astype(np.int8)
    la, lb = A.legal_form.values, B.legal_form.values
    X["legal_eq"] = (la == lb).astype(np.int8)
    X["legal_a_empty"] = (la == "").astype(np.int8)
    X["legal_b_empty"] = (lb == "").astype(np.int8)
    X["legal_overlap"] = [len(set(x.split()) & set(y.split())) for x, y in zip(la, lb)]
    X["name_core_eq"] = (A.name_core.values == B.name_core.values).astype(np.int8)
    X["name_clean_eq"] = (A.name_clean.values == B.name_clean.values).astype(np.int8)
    for k, v in string_feats(list(A.name_core.values), list(B.name_core.values), "name").items():
        X[k] = v
    for k, v in string_feats(list(A.addr_clean.values), list(B.addr_clean.values), "addr").items():
        X[k] = v
    X["name_len_diff"] = np.abs(A.name_core.str.len().values - B.name_core.str.len().values)
    # how common the name is among the S1 queries / in the candidate corpus (ambiguity)
    q_counts = Fd.loc[queries.entity_id].name_core.value_counts()
    c_counts = Fd.loc[corpus.entity_id].name_core.value_counts()
    X["s1_name_freq"] = A.name_core.map(q_counts).fillna(0).values
    X["cand_name_freq"] = B.name_core.map(c_counts).fillna(0).values
    X.to_parquet(FEAT)
    print(f"features: {X.shape} -> {os.path.relpath(FEAT, rx.ROOT)} ({time.time() - t0:.0f}s)")


# --------------------------------------------------------------------------------------
# Decision + metric
# --------------------------------------------------------------------------------------
def prepare(X, n_true):
    """Integer codes for S1 / candidate ids and the true-match count per S1 (fast metric)."""
    s1c, s1u = pd.factorize(X.s1_id.values)
    cc, _ = pd.factorize(X.cand_id.values)
    nt = n_true.reindex(s1u).fillna(0).values
    return s1c, cc, nt


def decide(cand_codes, score, thr, abstain):
    """Threshold, then each candidate record to at most one S1 (abstain: to none if >=2 pass)."""
    ok = np.where(score > thr)[0]
    order = ok[np.argsort(-score[ok], kind="stable")]
    _, first, counts = np.unique(cand_codes[order], return_index=True, return_counts=True)
    keep = order[first] if not abstain else order[first[counts == 1]]
    pred = np.zeros(len(score), bool)
    pred[keep] = True
    return pred


def macro_f05(s1_codes, label, pred, nt):
    """Per-S1 F0.5 averaged over all S1 (singletons: 1 if nothing predicted, else 0)."""
    tp = np.bincount(s1_codes, weights=(pred & (label == 1)), minlength=len(nt))
    n = np.bincount(s1_codes, weights=pred, minlength=len(nt))
    P = np.divide(tp, n, out=np.zeros(len(tp)), where=n > 0)
    R = np.divide(tp, nt, out=np.zeros(len(tp)), where=nt > 0)
    f = np.divide(1.25 * P * R, 0.25 * P + R, out=np.zeros(len(tp)), where=(0.25 * P + R) > 0)
    f = np.where(nt == 0, (n == 0).astype(float), f)
    return float(f.mean()), float(P[n > 0].mean()), float(R[nt > 0].mean())


def best_decision(X, score, n_true, grid, prep=None):
    s1c, cc, nt = prep if prep is not None else prepare(X, n_true)
    y = X.label.values
    res = []
    for abstain in (False, True):
        for thr in grid:
            pred = decide(cc, score, thr, abstain)
            f, p, r = macro_f05(s1c, y, pred, nt)
            res.append({"abstain": abstain, "thr": float(thr), "f05": f, "P": p, "R": r})
    return max(res, key=lambda d: d["f05"]), res


# --------------------------------------------------------------------------------------
# Cross-validated LightGBM
# --------------------------------------------------------------------------------------
def feature_sets(cols):
    rr = [c for c in cols if c.startswith("rr_") or c.startswith("rr")]
    comp = [c for c in cols if "margin_other" in c or "best_other" in c or "rank_in_cand" in c
            or c in ("cand_degree", "cand_n_s1_pos", "s1_n_pos")]
    base = [c for c in cols if c not in ("s1_id", "cand_id", "label")]
    return {
        "full": base,
        "no_competition": [c for c in base if c not in comp],
        "no_reranker": [c for c in base if not c.startswith("rr")],
    }


def cv(folds, sets, rounds):
    import lightgbm as lgb
    X = pd.read_parquet(FEAT)
    _, gt_test = rx.make_split()
    n_true = gt_test.set_index("source1_entity_id").matched_entity_ids.map(lambda s: len([x for x in s.split(",") if x]))
    # fold by S1 entity
    u = X.s1_id.unique()
    fold_of = pd.Series(np.random.default_rng(0).integers(0, folds, len(u)), index=u)
    fid = X.s1_id.map(fold_of).values
    y = X.label.values
    params = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_data_in_leaf=200,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  num_threads=36, verbose=-1)
    out = {}
    grid = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    # reference: the raw reranker(s) and a simple average ensemble, same decision procedure
    refs = {c: X[c].values for c in X if c.startswith("rr_")}
    if len(refs) > 1:
        refs["rr_mean"] = np.mean([1 / (1 + np.exp(-v)) for v in refs.values()], axis=0)
        refs["rr_mean"] = np.log(refs["rr_mean"] / (1 - refs["rr_mean"] + 1e-9) + 1e-9)
    for name, logit in refs.items():
        prob = 1 / (1 + np.exp(-logit))
        best, _ = best_decision(X, prob, n_true, grid)
        out[f"ref:{name}"] = best
        print(f"[reference] {name:12s} F0.5 {best['f05']:.4f} (P {best['P']:.4f} R {best['R']:.4f}) thr {best['thr']} abstain {best['abstain']}")
    imps = {}
    for sname in sets:
        cols = feature_sets(list(X.columns))[sname]
        oof = np.zeros(len(X))
        t = time.time()
        for k in range(folds):
            tr, va = fid != k, fid == k
            dtr = lgb.Dataset(X.loc[tr, cols], y[tr], free_raw_data=True)
            dva = lgb.Dataset(X.loc[va, cols], y[va], reference=dtr)
            m = lgb.train(params, dtr, rounds, valid_sets=[dva], callbacks=[lgb.early_stopping(50, verbose=False)])
            oof[va] = m.predict(X.loc[va, cols], num_iteration=m.best_iteration)
            if k == 0:
                imps[sname] = dict(sorted(zip(cols, m.feature_importance("gain").tolist()), key=lambda x: -x[1])[:25])
        best, allres = best_decision(X, oof, n_true, grid)
        out[f"lgbm:{sname}"] = best
        np.save(os.path.join(WORK, f"oof_{sname}.npy"), oof)
        print(f"[lgbm] {sname:15s} ({len(cols)} feats, {time.time() - t:.0f}s) F0.5 {best['f05']:.4f} "
              f"(P {best['P']:.4f} R {best['R']:.4f}) thr {best['thr']} abstain {best['abstain']}")
        for r in allres:
            if r["thr"] == best["thr"]:
                print(f"        thr {r['thr']} abstain={r['abstain']}: {r['f05']:.4f}")
    json.dump({"results": out, "importance_fold0": imps}, open(os.path.join(WORK, "cv_results.json"), "w"), indent=1)
    print("\ntop features (full, gain):")
    for k, v in list(imps.get("full", {}).items())[:20]:
        print(f"   {k:32s} {v:,.0f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["features", "cv", "all"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--sets", nargs="+", default=["full", "no_competition", "no_reranker"])
    ap.add_argument("--rounds", type=int, default=1500)
    a = ap.parse_args()
    if a.stage in ("features", "all"):
        build_features()
    if a.stage in ("cv", "all"):
        cv(a.folds, a.sets, a.rounds)


if __name__ == "__main__":
    main()

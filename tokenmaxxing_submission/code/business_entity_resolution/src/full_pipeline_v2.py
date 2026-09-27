#!/usr/bin/env python3
"""
Full inference pipeline v2 on the real test set: v1 + an ensemble of stage-2 decision models.

  1-3. retrieval (e5-large-ft top-K) and reranking (mDeBERTa) are taken from the v1 cache
       (run code/full_pipeline_v1.py first; it leaves candidates.npz and logits.npy there)
  4.   features  the stacker features of code/stack_lgbm.py for every test (S1, candidate) pair:
                 reranker / retriever scores, rank and gap inside the S1 list, competition between
                 S1s for the same record, structured house / legal / unit / name / address comparisons
  5.   ensemble  LightGBM + XGBoost + CatBoost + MLP, trained on the 20% validation split
                 (the split never seen by the retriever or the reranker); ensemble = mean of logits
  6.   decide    threshold + each S2/S3 record to at most one S1 (+ optional abstain), with the
                 threshold / abstain chosen on the held-out validation fold
  7.   write     output/v2/matching_results.tsv, candidate_pairs.tsv, and run the validator

Deployment feature set = the stacker features minus
  * the cosines of the base (non-fine-tuned) embedders and cos_mean - they would need 6 more
    embedding passes over 11.7M test records and carry < 0.5% of the LightGBM gain;
  * s1_name_freq / cand_name_freq - raw counts that scale with the size of the corpus (the test
    set has ~4x more S1s than the validation split), so their values would shift.

Stages:
    python code/full_pipeline_v2.py train      # fit the 4 models on validation folds 1-4, pick
                                               # the ensemble + threshold on fold 0 -> models/stacker_v2/
    python code/full_pipeline_v2.py predict    # test features -> ensemble -> submission files
    python code/full_pipeline_v2.py all
    python code/full_pipeline_v2.py predict --thr 0.6 --abstain   # override the decision
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import full_pipeline_v1 as v1  # noqa: E402
import retrieval_experiments as rx  # noqa: E402
import stack_lgbm as sl  # noqa: E402

ROOT = rx.ROOT
MODEL_DIR = os.path.join(ROOT, "models", "stacker_v2")
V1_CACHE = "/tmp/amazon_ml_cache/full_v1"
OUT_DIR = os.path.join(ROOT, "output", "v2")
DROP = {"cos_e5-large_embed_text", "cos_e5-large_name_core", "cos_e5-base_embed_text", "cos_e5-base_name_core",
        "cos_bge-m3_embed_text", "cos_bge-m3_name_core", "cos_mean", "s1_name_freq", "cand_name_freq"}
MEMBERS = ["lgbm", "xgb", "cat", "mlp"]
GRID = np.array([0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9])


def logit(p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


# --------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------
def fit_lgbm(Xtr, ytr, Xva, yva):
    import lightgbm as lgb
    params = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_data_in_leaf=200,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  num_threads=int(os.environ.get("LGBM_THREADS", 24)), verbose=-1)
    dtr = lgb.Dataset(Xtr, ytr)
    m = lgb.train(params, dtr, 2000, valid_sets=[lgb.Dataset(Xva, yva, reference=dtr)],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
    m.save_model(os.path.join(MODEL_DIR, "lgbm.txt"), num_iteration=m.best_iteration)
    print(f"    lgbm: {m.best_iteration} trees", flush=True)


def fit_xgb(Xtr, ytr, Xva, yva, gpu):
    import xgboost as xgb
    params = dict(objective="binary:logistic", eval_metric="logloss", tree_method="hist",
                  device=f"cuda:{gpu}" if gpu >= 0 else "cpu", learning_rate=0.05, max_depth=9,
                  min_child_weight=20, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, max_bin=256)
    dtr = xgb.QuantileDMatrix(Xtr, ytr, max_bin=256)
    dva = xgb.DMatrix(Xva, yva)
    m = xgb.train(params, dtr, 3000, evals=[(dva, "va")], early_stopping_rounds=50, verbose_eval=False)
    m = m[: m.best_iteration + 1]
    m.save_model(os.path.join(MODEL_DIR, "xgb.json"))
    print(f"    xgb: {m.num_boosted_rounds()} trees", flush=True)


def fit_cat(Xtr, ytr, Xva, yva, gpu):
    from catboost import CatBoostClassifier, Pool
    m = CatBoostClassifier(iterations=4000, learning_rate=0.08, depth=8, l2_leaf_reg=3, border_count=254,
                           task_type="GPU" if gpu >= 0 else "CPU", devices=str(max(gpu, 0)),
                           od_type="Iter", od_wait=100, verbose=0)
    m.fit(Pool(Xtr, ytr), eval_set=Pool(Xva, yva), use_best_model=True)
    m.save_model(os.path.join(MODEL_DIR, "cat.cbm"))
    print(f"    cat: {m.tree_count_} trees", flush=True)


def _mlp_net(d, hidden=(512, 256, 128)):
    import torch
    layers = []
    for h in hidden:
        layers += [torch.nn.Linear(d, h), torch.nn.BatchNorm1d(h), torch.nn.GELU(), torch.nn.Dropout(0.1)]
        d = h
    return torch.nn.Sequential(*layers, torch.nn.Linear(d, 1))


def _mlp_matrix(M, big):
    M = np.nan_to_num(M.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    M[:, big] = np.sign(M[:, big]) * np.log1p(np.abs(M[:, big]))
    return M


def fit_mlp(Xtr, ytr, gpu, epochs=6):
    import torch
    dev = f"cuda:{gpu}" if gpu >= 0 else "cpu"
    raw = np.nan_to_num(Xtr.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    big = np.abs(raw).max(0) > 50                        # signed log for heavy-tailed columns
    M = _mlp_matrix(Xtr, big)
    mu, sd = M.mean(0), M.std(0) + 1e-6
    Xt = torch.from_numpy((M - mu) / sd).to(dev)
    Yt = torch.from_numpy(ytr.astype(np.float32)).to(dev)
    net = _mlp_net(M.shape[1]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    bs = 8192
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=epochs * ((len(M) + bs - 1) // bs))
    lossf = torch.nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(len(M), device=dev)
        for s in range(0, len(M), bs):
            b = perm[s:s + bs]
            loss = lossf(net(Xt[b]).squeeze(-1), Yt[b])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    torch.save({"state": net.state_dict(), "mu": mu, "sd": sd, "big": big, "d": M.shape[1]},
               os.path.join(MODEL_DIR, "mlp.pt"))
    print("    mlp: trained", flush=True)


def predict_members(M, gpu):
    """M: float32 feature matrix (deploy columns, in order) -> {member: probability}."""
    import catboost
    import lightgbm as lgb
    import torch
    import xgboost as xgb
    out = {}
    t = time.time()
    out["lgbm"] = lgb.Booster(model_file=os.path.join(MODEL_DIR, "lgbm.txt")).predict(
        M, num_threads=int(os.environ.get("LGBM_THREADS", 24)))
    bx = xgb.Booster()
    bx.load_model(os.path.join(MODEL_DIR, "xgb.json"))
    bx.set_param({"device": f"cuda:{gpu}" if gpu >= 0 else "cpu"})
    out["xgb"] = np.concatenate([bx.inplace_predict(M[s:s + 4_000_000]) for s in range(0, len(M), 4_000_000)])
    cm = catboost.CatBoostClassifier()
    cm.load_model(os.path.join(MODEL_DIR, "cat.cbm"))
    out["cat"] = cm.predict_proba(M, thread_count=-1)[:, 1]
    ck = torch.load(os.path.join(MODEL_DIR, "mlp.pt"), weights_only=False)
    dev = f"cuda:{gpu}" if gpu >= 0 else "cpu"
    net = _mlp_net(ck["d"]).to(dev)
    net.load_state_dict(ck["state"])
    net.eval()
    ps = []
    with torch.no_grad():
        for s in range(0, len(M), 1_000_000):
            z = (_mlp_matrix(M[s:s + 1_000_000], ck["big"]) - ck["mu"]) / ck["sd"]
            ps.append(torch.sigmoid(net(torch.from_numpy(z).to(dev)).squeeze(-1)).cpu().numpy())
    out["mlp"] = np.concatenate(ps)
    print(f"  predicted {len(M):,} pairs with {list(out)} ({time.time() - t:.0f}s)", flush=True)
    return {k: v.astype(np.float64) for k, v in out.items()}


def combine(probs, cfg):
    z = sum(w * logit(probs[m]) for m, w in cfg["weights"].items()) / sum(cfg["weights"].values())
    return 1 / (1 + np.exp(-z))


# --------------------------------------------------------------------------------------
# Stage: train (on the validation-split features)
# --------------------------------------------------------------------------------------
def train(gpu):
    os.makedirs(MODEL_DIR, exist_ok=True)
    t0 = time.time()
    X = pd.read_parquet(sl.FEAT)
    cols = [c for c in X.columns if c not in ("s1_id", "cand_id", "label") and c not in DROP]
    u = X.s1_id.unique()                                  # same folds as code/stack_models.py
    fold_of = pd.Series(np.random.default_rng(0).integers(0, 5, len(u)), index=u)
    va = X.s1_id.map(fold_of).values == 0
    y = X.label.values
    M = X[cols].to_numpy(dtype=np.float32, na_value=np.nan)
    print(f"train: {(~va).sum():,} pairs (folds 1-4), hold-out {va.sum():,} pairs (fold 0), {len(cols)} features")
    fit_lgbm(M[~va], y[~va], M[va], y[va])
    fit_xgb(M[~va], y[~va], M[va], y[va], gpu)
    fit_cat(M[~va], y[~va], M[va], y[va], gpu)
    fit_mlp(M[~va], y[~va], gpu)

    # choose the ensemble and the decision on the hold-out fold
    probs = predict_members(M[va], gpu)
    Xv = X.loc[va, ["s1_id", "cand_id", "label"]].reset_index(drop=True)
    _, gt_test = rx.make_split()
    n_true = gt_test.set_index("source1_entity_id").matched_entity_ids.map(lambda s: len([x for x in s.split(",") if x]))
    prep = sl.prepare(Xv, n_true)
    res = {}
    cands = {m: {"weights": {m: 1.0}} for m in MEMBERS}
    cands["mean(lgbm+xgb+cat+mlp)"] = {"weights": {m: 1.0 for m in MEMBERS}}
    cands["mean(lgbm+xgb+cat)"] = {"weights": {m: 1.0 for m in ("lgbm", "xgb", "cat")}}
    for name, c in cands.items():
        best, _ = sl.best_decision(Xv, combine(probs, c), n_true, GRID, prep)
        res[name] = {**c, **best}
        print(f"  {name:24s} F0.5 {best['f05']:.4f} (P {best['P']:.4f} R {best['R']:.4f}) thr {best['thr']} abstain {best['abstain']}")
    rr, _ = sl.best_decision(Xv, 1 / (1 + np.exp(-X.loc[va, "rr_v1"].values)), n_true, GRID, prep)
    print(f"  {'reference: reranker v1':24s} F0.5 {rr['f05']:.4f} thr {rr['thr']} abstain {rr['abstain']}")
    ens = {k: v for k, v in res.items() if k.startswith("mean")}
    chosen = max(ens, key=lambda k: ens[k]["f05"])
    cfg = {"features": cols, "ensemble": chosen, "weights": ens[chosen]["weights"],
           "thr": ens[chosen]["thr"], "abstain": ens[chosen]["abstain"], "holdout": res,
           "holdout_reranker_v1": rr}
    json.dump(cfg, open(os.path.join(MODEL_DIR, "config.json"), "w"), indent=1)
    print(f"chosen: {chosen}, thr {cfg['thr']}, abstain {cfg['abstain']} -> models/stacker_v2/config.json "
          f"({(time.time() - t0) / 60:.1f} min)")


# --------------------------------------------------------------------------------------
# Stage: predict (test features -> submission)
# --------------------------------------------------------------------------------------
def group_rank_desc(codes, s):
    """0 = highest score inside each group."""
    order = np.lexsort((-s, codes))
    c = codes[order]
    first = np.r_[True, c[1:] != c[:-1]]
    pos = np.arange(len(order)) - np.maximum.accumulate(np.where(first, np.arange(len(order)), 0))
    r = np.empty(len(s))
    r[order] = pos
    return r


def test_features(s1, oth, qi, ci, cos, rr):
    """Same definitions as stack_lgbm.build_features, on index arrays (qi -> s1 row, ci -> oth row)."""
    t0 = time.time()
    X = pd.DataFrame({"s1_id": qi, "cand_id": ci})     # integer ids: grouping only needs equality
    X["ret_rank"] = group_rank_desc(qi, cos).astype(np.int16)
    X["rr_v1"] = rr
    X["cos_e5-large-ft_embed_text"] = cos
    ng_q = qi.max() + 1
    for c, short in (("rr_v1", "rr_v1"), ("cos_e5-large-ft_embed_text", "cosft")):
        s = X[c].values.astype(np.float64)
        X[f"{short}_rank_in_s1"] = group_rank_desc(qi, s)
        mx = np.full(ng_q, -np.inf)
        np.maximum.at(mx, qi, s)
        X[f"{short}_gap_to_s1_max"] = s - mx[qi]
        for k, v in sl.competition(X, c, short).items():
            X[k] = v
    X["cand_degree"] = np.bincount(ci)[ci]
    X["cand_n_s1_pos"] = np.bincount(ci, weights=rr > 0)[ci]
    X["s1_n_pos"] = np.bincount(qi, weights=rr > 0)[qi]
    print(f"  score/context features ({time.time() - t0:.0f}s)", flush=True)

    A = {c: s1[c].values[qi] for c in sl.FIELDS if c != "entity_id"}
    B = {c: oth[c].values[ci] for c in sl.FIELDS if c != "entity_id"}
    X["cand_is_s3"] = oth.entity_id.str.startswith("S3").values[ci].astype(np.int8)
    for c in ("addr_empty", "is_indic", "is_domain", "has_dba"):
        X[f"cand_{c}"] = (B[c] == "1").astype(np.int8)
    ncomp = lambda d: (pd.Series(d.addr_clean.values).str.count(",").values + (d.addr_clean.values != ""))  # noqa: E731
    X["s1_addr_ncomp"] = ncomp(s1)[qi]
    X["cand_addr_ncomp"] = ncomp(oth)[ci]
    ha, hb = A["house_base"], B["house_base"]
    both = (ha != "") & (hb != "")
    X["house_both"] = both.astype(np.int8)
    X["house_eq"] = (both & (ha == hb)).astype(np.int8)
    X["house_full_eq"] = (both & (A["house_no"] == B["house_no"])).astype(np.int8)
    num = both & np.array([len(x) < 10 and len(y) < 10 and x.isdigit() and y.isdigit() for x, y in zip(ha, hb)])
    d = np.full(len(X), -1.0)
    d[num] = np.abs(ha[num].astype(np.int64) - hb[num].astype(np.int64))
    X["house_diff"] = d
    X["house_fingerprint"] = np.isin(d, sl.FINGERPRINT).astype(np.int8)
    X["house_len_eq"] = (both & np.array([len(x) == len(y) for x, y in zip(ha, hb)])).astype(np.int8)
    na, nb = A["addr_numbers"], B["addr_numbers"]
    inter = np.empty(len(X), np.int32)
    union = np.empty(len(X), np.int32)
    for i, (x, y) in enumerate(zip(na, nb)):
        sx, sy = set(x.split()), set(y.split())
        inter[i], union[i] = len(sx & sy), len(sx | sy)
    X["num_overlap"] = inter
    X["num_jaccard"] = np.divide(inter, union, out=np.zeros(len(X)), where=union > 0)
    X["unit_eq"] = ((A["unit"] != "") & (A["unit"] == B["unit"])).astype(np.int8)
    X["unit_both"] = ((A["unit"] != "") & (B["unit"] != "")).astype(np.int8)
    la, lb = A["legal_form"], B["legal_form"]
    X["legal_eq"] = (la == lb).astype(np.int8)
    X["legal_a_empty"] = (la == "").astype(np.int8)
    X["legal_b_empty"] = (lb == "").astype(np.int8)
    X["legal_overlap"] = [len(set(x.split()) & set(y.split())) if x and y else 0 for x, y in zip(la, lb)]
    X["name_core_eq"] = (A["name_core"] == B["name_core"]).astype(np.int8)
    X["name_clean_eq"] = (A["name_clean"] == B["name_clean"]).astype(np.int8)
    print(f"  structured features ({time.time() - t0:.0f}s)", flush=True)
    for k, v in sl.string_feats(list(A["name_core"]), list(B["name_core"]), "name").items():
        X[k] = v
    for k, v in sl.string_feats(list(A["addr_clean"]), list(B["addr_clean"]), "addr").items():
        X[k] = v
    X["name_len_diff"] = np.abs(pd.Series(A["name_core"]).str.len().values - pd.Series(B["name_core"]).str.len().values)
    print(f"  string features ({time.time() - t0:.0f}s)", flush=True)
    return X


def predict(a):
    t0 = time.time()
    cfg = json.load(open(os.path.join(MODEL_DIR, "config.json")))
    thr = a.thr if a.thr is not None else cfg["thr"]
    abstain = cfg["abstain"] if a.abstain is None else a.abstain
    need = [os.path.join(a.v1_cache, f) for f in ("candidates.npz", "logits.npy")]
    missing = [p for p in need if not os.path.exists(p)]
    if missing:
        sys.exit(f"missing v1 cache files {missing}: run code/full_pipeline_v1.py first")
    z = np.load(need[0])
    qi, ci, cos = z["qi"], z["ci"], z["cos"]
    rr = np.load(need[1])

    cols = ["entity_id", "country"] + [c for c in sl.FIELDS if c != "entity_id"]
    s1 = rx.read_tsv(os.path.join(v1.TEST_DIR, "test_source1.tsv"), cols)
    oth = pd.concat([rx.read_tsv(os.path.join(v1.TEST_DIR, f"test_source{s}.tsv"), cols) for s in (2, 3)],
                    ignore_index=True)            # same order as v1 (the indices in candidates.npz)
    print(f"test: {len(s1):,} S1, {len(oth):,} S2+S3, {len(qi):,} candidate pairs")

    feat_path = os.path.join(a.cache, "test_features.parquet")
    if os.path.exists(feat_path):
        X = pd.read_parquet(feat_path)
    else:
        os.makedirs(a.cache, exist_ok=True)
        X = test_features(s1, oth, qi, ci, cos, rr)
        X.to_parquet(feat_path)
    M = X[cfg["features"]].to_numpy(dtype=np.float32, na_value=np.nan)
    del X
    probs = predict_members(M, a.gpu)
    prob = combine(probs, cfg)
    np.save(os.path.join(a.cache, "ensemble_prob.npy"), prob)

    keep = v1.decide(qi, ci, prob, thr, abstain)
    print(f"decision: {cfg['ensemble']}, thr {thr}, abstain {abstain} -> {len(keep):,} matched pairs")
    os.makedirs(a.out, exist_ok=True)
    s1_ids, c_ids = s1.entity_id.values, oth.entity_id.values
    match = pd.Series(c_ids[ci[keep]]).groupby(qi[keep]).agg(list)
    cands = pd.Series(c_ids[ci]).groupby(qi).agg(list)
    m_lists = [match.get(i, []) for i in range(len(s1))]
    c_lists = [cands.get(i, []) for i in range(len(s1))]
    mp, cp = os.path.join(a.out, "matching_results.tsv"), os.path.join(a.out, "candidate_pairs.tsv")
    v1.write_tsv(mp, ["source1_entity_id", "matched_entity_ids"], s1_ids, m_lists)
    v1.write_tsv(cp, ["source1_entity_id", "candidate_entity_ids"], s1_ids, c_lists)
    sizes = pd.Series([len(l) for l in m_lists])
    by_c = sizes.groupby(s1.country.values).agg(["mean", lambda x: (x == 0).mean()])
    by_c.columns = ["mean matches", "share empty"]
    print(f"wrote {os.path.relpath(mp, ROOT)} ({len(s1):,} rows, {(sizes == 0).mean():.1%} empty, "
          f"mean {sizes.mean():.2f} matches per S1)\n{by_c.round(3).to_string()}")
    v1_path = os.path.join(v1.OUT_DIR, "matching_results.tsv")
    if os.path.exists(v1_path):                     # how different is v2 from the v1 submission?
        o = pd.read_csv(v1_path, sep="\t", dtype=str, keep_default_na=False)
        old = dict(zip(o.source1_entity_id, o.matched_entity_ids))
        same = np.mean([sorted(x for x in old.get(s, "").split(",") if x) == sorted(l) for s, l in zip(s1_ids, m_lists)])
        print(f"vs v1 submission: {same:.2%} of S1 rows identical")
    r = subprocess.run([sys.executable, "utils/validate_submission.py", "--matching", mp, "--candidate", cp,
                        "--test-dir", "dataset/test"], cwd=os.path.join(ROOT, "student_resource"),
                       capture_output=True, text=True)
    print(r.stdout[-1500:])
    print(f"total time {(time.time() - t0) / 60:.1f} min")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["train", "predict", "all"])
    ap.add_argument("--gpu", type=int, default=0, help="GPU for xgb / cat / mlp (-1 = CPU)")
    ap.add_argument("--thr", type=float, default=None, help="override the threshold from config.json")
    ap.add_argument("--abstain", type=lambda s: s.lower() in ("1", "true", "yes"), default=None,
                    help="override abstain (true/false)")
    ap.add_argument("--v1-cache", default=V1_CACHE)
    ap.add_argument("--cache", default="/tmp/amazon_ml_cache/full_v2")
    ap.add_argument("--out", default=OUT_DIR)
    a = ap.parse_args()
    if a.stage in ("train", "all"):
        train(a.gpu)
    if a.stage in ("predict", "all"):
        predict(a)


if __name__ == "__main__":
    main()

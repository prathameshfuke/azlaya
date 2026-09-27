#!/usr/bin/env python3
"""
Train several stage-2 decision models on the stacker features and ensemble them.

Models (same features, same 5 folds grouped by S1 entity, out-of-fold predictions):
  lgbm     LightGBM (CPU)
  xgb      XGBoost (GPU, hist)
  cat      CatBoost (GPU)
  mlp      3-layer MLP on transformed + standardised features (GPU)
  logreg   logistic regression baseline (GPU, same pipeline as the MLP without hidden layers)
Ensembles over the OOF probabilities:
  mean     average of the model logits
  rank     average of rank-normalised scores
  weighted non-negative weights fitted on OOF logits (few parameters -> low overfitting risk)

Every score goes through the same decision (threshold + each S2/S3 record to at most one S1,
optionally abstaining when several S1s pass) and the challenge metric (per-S1 macro F0.5).
Features come from code/stack_lgbm.py (experiments/stacker/features.parquet).

Usage:
    python code/stack_models.py                          # all models + ensembles
    python code/stack_models.py --models lgbm xgb        # subset (cached OOFs are reused)
    python code/stack_models.py --refit lgbm             # retrain even if an OOF is cached
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
import stack_lgbm as sl  # noqa: E402

WORK = sl.WORK
ALL_MODELS = ["lgbm", "xgb", "cat", "mlp", "logreg"]
GRID = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])


def load():
    X = pd.read_parquet(sl.FEAT)
    cols = [c for c in X.columns if c not in ("s1_id", "cand_id", "label")]
    u = X.s1_id.unique()
    fold_of = pd.Series(np.random.default_rng(0).integers(0, 5, len(u)), index=u)
    fid = X.s1_id.map(fold_of).values
    _, gt_test = rx.make_split()
    n_true = gt_test.set_index("source1_entity_id").matched_entity_ids.map(lambda s: len([x for x in s.split(",") if x]))
    return X, cols, fid, n_true


# --------------------------------------------------------------------------------------
# Models: each returns out-of-fold probabilities
# --------------------------------------------------------------------------------------
def fit_lgbm(X, cols, y, fid, folds):
    import lightgbm as lgb
    params = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_data_in_leaf=200,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  num_threads=int(os.environ.get("LGBM_THREADS", 36)), verbose=-1)
    oof = np.zeros(len(y))
    imp = None
    for k in range(folds):
        tr, va = fid != k, fid == k
        dtr = lgb.Dataset(X.loc[tr, cols], y[tr])
        dva = lgb.Dataset(X.loc[va, cols], y[va], reference=dtr)
        m = lgb.train(params, dtr, 2000, valid_sets=[dva], callbacks=[lgb.early_stopping(50, verbose=False)])
        oof[va] = m.predict(X.loc[va, cols], num_iteration=m.best_iteration)
        if imp is None:
            imp = dict(sorted(zip(cols, m.feature_importance("gain").tolist()), key=lambda x: -x[1]))
        print(f"    lgbm fold {k}: {m.best_iteration} trees", flush=True)
    return oof, imp


def fit_xgb(X, cols, y, fid, folds):
    import xgboost as xgb
    params = dict(objective="binary:logistic", eval_metric="logloss", tree_method="hist", device="cuda",
                  learning_rate=0.05, max_depth=9, min_child_weight=20, subsample=0.8, colsample_bytree=0.8,
                  reg_lambda=1.0, max_bin=256)
    oof = np.zeros(len(y))
    for k in range(folds):
        tr, va = fid != k, fid == k
        dtr = xgb.QuantileDMatrix(X.loc[tr, cols].values, y[tr], max_bin=256)
        dva = xgb.DMatrix(X.loc[va, cols].values, y[va])
        m = xgb.train(params, dtr, 3000, evals=[(dva, "va")], early_stopping_rounds=50, verbose_eval=False)
        oof[va] = m.predict(dva, iteration_range=(0, m.best_iteration + 1))
        print(f"    xgb fold {k}: {m.best_iteration + 1} trees", flush=True)
    return oof, None


def fit_cat(X, cols, y, fid, folds):
    from catboost import CatBoostClassifier, Pool
    oof = np.zeros(len(y))
    for k in range(folds):
        tr, va = fid != k, fid == k
        m = CatBoostClassifier(iterations=4000, learning_rate=0.08, depth=8, l2_leaf_reg=3, border_count=254,
                               task_type="GPU", devices="0", od_type="Iter", od_wait=100, verbose=0)
        m.fit(Pool(X.loc[tr, cols].values, y[tr]), eval_set=Pool(X.loc[va, cols].values, y[va]), use_best_model=True)
        oof[va] = m.predict_proba(X.loc[va, cols].values)[:, 1]
        print(f"    cat fold {k}: {m.get_best_iteration() + 1} trees", flush=True)
    return oof, None


def _nn_matrix(X, cols):
    """Signed log for heavy-tailed columns, NaN -> 0; standardised later per fold."""
    M = X[cols].to_numpy(dtype=np.float32, na_value=0.0)
    M = np.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)
    big = np.abs(M).max(0) > 50
    M[:, big] = np.sign(M[:, big]) * np.log1p(np.abs(M[:, big]))
    return M


def fit_nn(X, cols, y, fid, folds, hidden, epochs, gpu=0):
    import torch
    dev = f"cuda:{gpu}"
    M = _nn_matrix(X, cols)
    oof = np.zeros(len(y))
    for k in range(folds):
        tr, va = np.where(fid != k)[0], np.where(fid == k)[0]
        mu, sd = M[tr].mean(0), M[tr].std(0) + 1e-6
        Xt = torch.from_numpy((M[tr] - mu) / sd).to(dev)
        Yt = torch.from_numpy(y[tr].astype(np.float32)).to(dev)
        Xv = torch.from_numpy((M[va] - mu) / sd).to(dev)
        layers, d = [], M.shape[1]
        for h in hidden:
            layers += [torch.nn.Linear(d, h), torch.nn.BatchNorm1d(h), torch.nn.GELU(), torch.nn.Dropout(0.1)]
            d = h
        net = torch.nn.Sequential(*layers, torch.nn.Linear(d, 1)).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=2e-3 if hidden else 1e-2, weight_decay=1e-4)
        steps_per_ep = (len(tr) + 8191) // 8192
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=opt.param_groups[0]["lr"], total_steps=epochs * steps_per_ep)
        lossf = torch.nn.BCEWithLogitsLoss()
        for ep in range(epochs):
            net.train()
            perm = torch.randperm(len(tr), device=dev)
            for s in range(0, len(tr), 8192):
                b = perm[s:s + 8192]
                loss = lossf(net(Xt[b]).squeeze(-1), Yt[b])
                opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        net.eval()
        with torch.no_grad():
            oof[va] = torch.cat([torch.sigmoid(net(Xv[s:s + 65536]).squeeze(-1)) for s in range(0, len(va), 65536)]).cpu().numpy()
        print(f"    {'mlp' if hidden else 'logreg'} fold {k} done", flush=True)
        del Xt, Yt, Xv
        torch.cuda.empty_cache()
    return oof, None


FITTERS = {
    "lgbm": fit_lgbm,
    "xgb": fit_xgb,
    "cat": fit_cat,
    "mlp": lambda X, c, y, f, k: fit_nn(X, c, y, f, k, hidden=[512, 256, 128], epochs=6),
    "logreg": lambda X, c, y, f, k: fit_nn(X, c, y, f, k, hidden=[], epochs=3),
}


# --------------------------------------------------------------------------------------
# Ensembles
# --------------------------------------------------------------------------------------
def logit(p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def fit_weights(Z, y, fid, folds):
    """Non-negative logit weights, fitted out-of-fold (weights for fold k learned on the others)."""
    import torch
    out = np.zeros(len(y))
    ws = []
    for k in range(folds):
        tr, va = fid != k, fid == k
        z = torch.tensor(Z[tr], dtype=torch.float32)
        t = torch.tensor(y[tr], dtype=torch.float32)
        w = torch.zeros(Z.shape[1], requires_grad=True)
        b = torch.zeros(1, requires_grad=True)
        opt = torch.optim.LBFGS([w, b], max_iter=200)

        def closure():
            opt.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(z @ torch.nn.functional.softplus(w) + b, t)
            loss.backward()
            return loss
        opt.step(closure)
        wk = torch.nn.functional.softplus(w).detach().numpy()
        out[va] = 1 / (1 + np.exp(-(Z[va] @ wk + b.item())))
        ws.append(wk)
    return out, np.mean(ws, 0)


PREP = None      # integer codes for the fast metric, set in main()


def evaluate(name, X, score, n_true, results):
    best, allres = sl.best_decision(X, score, n_true, GRID, PREP)
    no_abs = max((r for r in allres if not r["abstain"]), key=lambda r: r["f05"])
    results[name] = {"best": best, "best_no_abstain": no_abs}
    print(f"  {name:22s} F0.5 {best['f05']:.4f} (P {best['P']:.4f}, R {best['R']:.4f}, thr {best['thr']}, "
          f"abstain {best['abstain']}) | without abstain {no_abs['f05']:.4f}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=ALL_MODELS, choices=ALL_MODELS)
    ap.add_argument("--refit", nargs="*", default=[])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--no-ensemble", action="store_true", help="only train/evaluate the listed models (for parallel runs)")
    a = ap.parse_args()
    X, cols, fid, n_true = load()
    y = X.label.values
    global PREP
    PREP = sl.prepare(X, n_true)
    print(f"features: {len(X):,} pairs x {len(cols)} features; {X.s1_id.nunique():,} S1; positives {y.mean():.3f}")
    results, oofs = {}, {}

    if not a.no_ensemble:
        print("\nreference (no stage-2 model):")
        evaluate("reranker v1 only", X, 1 / (1 + np.exp(-X["rr_v1"].values)), n_true, results)
        evaluate("retriever cosine only", X, X["cos_e5-large-ft_embed_text"].values, n_true, results)

    print("\nstage-2 models (out-of-fold):")
    imps = {}
    for m in a.models:
        path = os.path.join(WORK, f"oof_{m}.npy")
        if os.path.exists(path) and m not in a.refit:
            oofs[m] = np.load(path)
        else:
            t = time.time()
            oofs[m], imp = FITTERS[m](X, cols, y, fid, a.folds)
            np.save(path, oofs[m])
            if imp:
                imps[m] = imp
            print(f"    {m} trained in {(time.time() - t) / 60:.1f} min")
        evaluate(m, X, oofs[m], n_true, results)

    if len(oofs) > 1 and not a.no_ensemble:
        print("\nensembles:")
        names = list(oofs)
        Z = np.stack([logit(oofs[m]) for m in names], 1)
        evaluate("mean(" + "+".join(names) + ")", X, 1 / (1 + np.exp(-Z.mean(1))), n_true, results)
        R = np.stack([pd.Series(oofs[m]).rank(pct=True).values for m in names], 1)
        evaluate("rank-mean(all)", X, R.mean(1), n_true, results)
        trees = [m for m in names if m in ("lgbm", "xgb", "cat")]
        if len(trees) > 1:
            Zt = np.stack([logit(oofs[m]) for m in trees], 1)
            evaluate("mean(" + "+".join(trees) + ")", X, 1 / (1 + np.exp(-Zt.mean(1))), n_true, results)
        wp, w = fit_weights(Z, y, fid, a.folds)
        print("    weights: " + ", ".join(f"{m}={x:.2f}" for m, x in zip(names, w)))
        evaluate("weighted(all)", X, wp, n_true, results)
        np.save(os.path.join(WORK, "oof_weighted.npy"), wp)

    prev = os.path.join(WORK, "stack_results.json" if not a.no_ensemble else f"stack_results_{'_'.join(a.models)}.json")
    old = json.load(open(prev)) if os.path.exists(prev) else {}
    old.update({"results": {**old.get("results", {}), **results}})
    if imps:
        old["importance"] = {**old.get("importance", {}), **{k: dict(list(v.items())[:30]) for k, v in imps.items()}}
    json.dump(old, open(prev, "w"), indent=1)
    lines = ["| Scorer | Macro F0.5 | Precision | Recall | Thr | Abstain | F0.5 without abstain |", "|---|---|---|---|---|---|---|"]
    for n, r in old["results"].items():
        b = r["best"]
        lines.append(f"| {n} | {b['f05']:.4f} | {b['P']:.4f} | {b['R']:.4f} | {b['thr']} | {b['abstain']} | {r['best_no_abstain']['f05']:.4f} |")
    open(os.path.join(WORK, "stack_results.md"), "w").write("\n".join(lines) + "\n")
    if "lgbm" in imps:
        print("\ntop LightGBM features (gain):")
        for k, v in list(imps["lgbm"].items())[:20]:
            print(f"   {k:34s} {v:,.0f}")
    print(f"\nresults -> {os.path.relpath(prev, rx.ROOT)} and stack_results.md")


if __name__ == "__main__":
    main()

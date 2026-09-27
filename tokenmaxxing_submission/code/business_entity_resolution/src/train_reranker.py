#!/usr/bin/env python3
"""
Cross-encoder reranker (mDeBERTa-v3-base): given an S1 record and a retrieved S2/S3
candidate, predict match / no match.

Stages (run all, or one at a time):
  cands    Build pairs.
           train: 80% split. Per S1 anchor up to 2 true matches (label 1) and 3 hard negatives
                  mined by the base e5-large (label 0; from code/finetune_e5.py). Train-split
                  singletons get 3 nearest neighbours as negatives only.
           test : 20% split. Top-20 candidates of the fine-tuned retriever (e5-large-ft) for every
                  test S1, singletons included - exactly what the reranker sees at inference.
  train    DDP fine-tuning on all GPUs, binary cross-entropy, fp16.
  predict  Score the test pairs.
  eval     Pair AUC / AP, and the challenge metric: per-S1 macro F0.5 (singletons included),
           threshold swept, with and without "each S2/S3 goes to at most one S1", compared
           with a retrieval-cosine baseline.

Input text for each side: "name_clean | addr_clean" (keeps the legal form, unlike embed_text).

Usage (from the project root):
    python code/train_reranker.py                     # all stages
    python code/train_reranker.py --stage train --max-steps 50
"""
import argparse
import datetime
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.multiprocessing as tmp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import retrieval_experiments as rx  # noqa: E402

ROOT = rx.ROOT
FT_WORK = os.path.join(ROOT, "experiments", "finetune")          # mined data from finetune_e5.py
WORK = os.path.join(ROOT, "experiments", "reranker")
BASE_MODEL = "microsoft/mdeberta-v3-base"
OUT_MODEL = os.path.join(ROOT, "models", "mdeberta-reranker")
MAX_LEN = 160
TEST_K = 20
TAG = ""        # "" = v1 (base-model hard negatives); "_v2" = negatives from the fine-tuned retriever


def p_train_pairs():
    return os.path.join(WORK, f"train_pairs{TAG}.parquet")


def p_model():
    return OUT_MODEL + TAG.replace("_", "-")


def p_logits():
    return os.path.join(WORK, f"test_logits{TAG}.npy")


# --------------------------------------------------------------------------------------
# Stage 1: candidate pairs
# --------------------------------------------------------------------------------------
def load_texts():
    """entity_id -> 'name_clean | addr_clean' for train S1, S2, S3."""
    cols = ["entity_id", "name_clean", "addr_clean"]
    parts = [rx.read_tsv(os.path.join(rx.PROC, "train", f"train_source{s}.tsv"), cols) for s in (1, 2, 3)]
    d = pd.concat(parts, ignore_index=True)
    txt = d.name_clean + " | " + d.addr_clean
    return pd.Series(txt.values, index=d.entity_id.values)


def stage_cands(gpus, seed=0):
    os.makedirs(WORK, exist_ok=True)
    rng = np.random.default_rng(seed)
    texts = load_texts()

    # ---- train pairs from the mined data
    anchors = pd.read_parquet(os.path.join(FT_WORK, "anchors.parquet"))
    corpus = pd.read_parquet(os.path.join(FT_WORK, "corpus.parquet"))
    z = np.load(os.path.join(FT_WORK, "train_index.npz"))
    offsets, pos, hard = z["offsets"], z["pos"], z["hard"]
    a_id, c_id = anchors.entity_id.values, corpus.entity_id.values
    s1s, cands, labels = [], [], []
    for i in range(len(anchors)):
        p = pos[offsets[i]:offsets[i + 1]]
        for c in rng.choice(p, min(2, len(p)), replace=False):
            s1s.append(a_id[i]); cands.append(c_id[c]); labels.append(1)
        h = hard[i][hard[i] >= 0][:10]
        for c in rng.choice(h, min(3, len(h)), replace=False) if len(h) else []:
            s1s.append(a_id[i]); cands.append(c_id[c]); labels.append(0)

    # ---- train-split singletons: negatives only (their nearest neighbours)
    gt_train, gt_test = rx.make_split()
    single = gt_train[gt_train.matched_entity_ids == ""].source1_entity_id
    s1 = rx.read_tsv(os.path.join(rx.PROC, "train", "train_source1.tsv"), ["entity_id", "country", "embed_text"])
    s1 = s1[s1.entity_id.isin(set(single))].reset_index(drop=True)
    cfg = dict(rx.MODELS["e5-large"], path=rx.model_path(rx.MODELS["e5-large"]["hf"]))
    se = rx.embed(s1.embed_text.values, cfg, 96, gpus, os.path.join(WORK, "base_singletons.npy"))
    ce = np.load(os.path.join(FT_WORK, "base_corpus.npy"), mmap_mode="r")
    top = rx.search(se, np.asarray(ce), s1.country.values, corpus.country.values, 8, gpus)
    for i in range(len(s1)):
        h = top[i][top[i] >= 0]
        for c in rng.choice(h, min(3, len(h)), replace=False):
            s1s.append(s1.entity_id[i]); cands.append(c_id[c]); labels.append(0)
    tr = pd.DataFrame({"s1_id": s1s, "cand_id": cands, "label": labels})
    tr["text_a"] = texts.loc[tr.s1_id].values
    tr["text_b"] = texts.loc[tr.cand_id].values
    tr = tr.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    tr.to_parquet(p_train_pairs())
    print(f"train pairs: {len(tr):,} ({tr.label.mean():.1%} positive; {len(s1):,} singleton anchors)")

    # ---- test pairs: top-K of the fine-tuned retriever
    queries, ecorpus, tpairs = rx.build_eval_set(gt_train, gt_test, ["embed_text"])
    qe = np.load(os.path.join(rx.OUT, "emb", "e5-large-ft_embed_text_queries.npy"))
    cee = np.load(os.path.join(rx.OUT, "emb", "e5-large-ft_embed_text_corpus.npy"))
    top = rx.search(qe, cee, queries.country.values, ecorpus.country.values, TEST_K, gpus)
    qi = np.repeat(np.arange(len(queries)), TEST_K)
    ci = top.reshape(-1)
    keep = ci >= 0
    qi, ci = qi[keep], ci[keep]
    te = pd.DataFrame({"s1_id": queries.entity_id.values[qi], "cand_id": ecorpus.entity_id.values[ci],
                       "rank": np.tile(np.arange(TEST_K), len(queries))[keep]})
    te["cos"] = np.einsum("ij,ij->i", qe[qi].astype(np.float32), cee[ci].astype(np.float32))
    truth = set(zip(tpairs.source1_entity_id, tpairs.m))
    te["label"] = [int((a, b) in truth) for a, b in zip(te.s1_id, te.cand_id)]
    te["text_a"] = texts.loc[te.s1_id].values
    te["text_b"] = texts.loc[te.cand_id].values
    te.to_parquet(os.path.join(WORK, "test_pairs.parquet"))
    print(f"test pairs: {len(te):,} for {te.s1_id.nunique():,} S1 ({te.label.sum():,} true of {len(tpairs):,} "
          f"-> candidate recall {te.label.sum() / len(tpairs):.4f})")


def stage_cands_ft(gpus, seed=0, k=20):
    """v2 training pairs: negatives from the FINE-TUNED retriever's top-k on the train split.

    These are the candidates the reranker actually faces at inference, including
    empty-address records that belong to a namesake S1 (missing from the v1 negatives).
    """
    rng = np.random.default_rng(seed)
    texts = load_texts()
    anchors = pd.read_parquet(os.path.join(FT_WORK, "anchors.parquet"))
    corpus = pd.read_parquet(os.path.join(FT_WORK, "corpus.parquet"))
    z = np.load(os.path.join(FT_WORK, "train_index.npz"))
    offsets, pos = z["offsets"], z["pos"]
    cfg = dict(rx.MODELS["e5-large-ft"], path=rx.MODELS["e5-large-ft"]["hf"])
    ae = rx.embed(anchors.embed_text.values, cfg, 128, gpus, os.path.join(WORK, "ft_anchors.npy"))
    ce = rx.embed(corpus.embed_text.values, cfg, 128, gpus, os.path.join(WORK, "ft_corpus.npy"))
    top = rx.search(ae, ce, anchors.country.values, corpus.country.values, k, gpus)
    a_id, c_id = anchors.entity_id.values, corpus.entity_id.values
    s1s, cands, labels = [], [], []
    for i in range(len(anchors)):
        p = pos[offsets[i]:offsets[i + 1]]
        for c in rng.choice(p, min(2, len(p)), replace=False):
            s1s.append(a_id[i]); cands.append(c_id[c]); labels.append(1)
        ps = set(p.tolist())
        neg = [c for c in top[i] if c >= 0 and c not in ps]
        for c in rng.choice(neg, min(4, len(neg)), replace=False) if neg else []:
            s1s.append(a_id[i]); cands.append(c_id[c]); labels.append(0)
    gt_train, _ = rx.make_split()
    single = gt_train[gt_train.matched_entity_ids == ""].source1_entity_id
    s1 = rx.read_tsv(os.path.join(rx.PROC, "train", "train_source1.tsv"), ["entity_id", "country", "embed_text"])
    s1 = s1[s1.entity_id.isin(set(single))].reset_index(drop=True)
    se = rx.embed(s1.embed_text.values, cfg, 128, gpus, os.path.join(WORK, "ft_singletons.npy"))
    top = rx.search(se, ce, s1.country.values, corpus.country.values, k, gpus)
    for i in range(len(s1)):
        h = top[i][top[i] >= 0]
        for c in rng.choice(h, min(4, len(h)), replace=False):
            s1s.append(s1.entity_id[i]); cands.append(c_id[c]); labels.append(0)
    tr = pd.DataFrame({"s1_id": s1s, "cand_id": cands, "label": labels})
    tr["text_a"] = texts.loc[tr.s1_id].values
    tr["text_b"] = texts.loc[tr.cand_id].values
    tr = tr.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    tr.to_parquet(p_train_pairs())
    empty = tr.text_b.str.rstrip().str.endswith("|")
    print(f"v2 train pairs: {len(tr):,} ({tr.label.mean():.1%} positive); empty-address share: "
          f"positives {empty[tr.label == 1].mean():.1%}, negatives {empty[tr.label == 0].mean():.1%}")


# --------------------------------------------------------------------------------------
# Stage 2: training
# --------------------------------------------------------------------------------------
def _train_worker(rank, gpus, args):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    global TAG
    TAG = args.tag                                  # spawned workers re-import the module
    world = len(gpus)
    torch.cuda.set_device(gpus[rank])
    dev = torch.device(f"cuda:{gpus[rank]}")
    dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{args.port}", rank=rank, world_size=world,
                            timeout=datetime.timedelta(minutes=60))
    df = pd.read_parquet(p_train_pairs(), columns=["label", "text_a", "text_b"])
    A, Bt, Y = df.text_a.values, df.text_b.values, df.label.values.astype(np.float32)
    del df
    path = rx.model_path(BASE_MODEL)
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path, num_labels=1, dtype=torch.float32).to(dev)
    ddp = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpus[rank]])
    opt = torch.optim.AdamW(ddp.parameters(), lr=args.lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda")
    steps = len(Y) // (args.batch * world)
    if args.max_steps:
        steps = min(steps, args.max_steps)
    total = steps * args.epochs
    warm = max(1, int(0.06 * total))

    def lr_lambda(s):
        return (s + 1) / warm if s < warm else max(0.0, (total - s) / max(1, total - warm))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    lossf = torch.nn.BCEWithLogitsLoss()
    step, t0, log, run_loss, run_acc = 0, time.time(), [], 0.0, 0.0
    model.train()
    lens = np.array([len(a) + len(b) for a, b in zip(A, Bt)])
    gb = args.batch * world

    def global_batches(ep):
        """Length-bucketed batches (less padding), identical on every rank."""
        r = np.random.default_rng(7 + ep)
        order = r.permutation(len(Y))
        out = []
        for c in range(0, len(order) - gb + 1, gb * 50):
            chunk = order[c:c + gb * 50]
            chunk = chunk[np.argsort(lens[chunk], kind="stable")]
            out += [chunk[i:i + gb] for i in range(0, len(chunk) - gb + 1, gb)]
        return [out[i] for i in r.permutation(len(out))]

    for ep in range(args.epochs):
        batches = global_batches(ep)
        for s in range(steps):
            b = batches[s]
            ids = b[rank::world]                      # interleave so every rank gets similar lengths
            enc = tok(list(A[ids]), list(Bt[ids]), padding=True, truncation=True, max_length=MAX_LEN,
                      return_tensors="pt").to(dev)
            y = torch.from_numpy(Y[ids]).to(dev)
            with torch.autocast("cuda", dtype=torch.float16):
                logit = ddp(**enc).logits.squeeze(-1)
            loss = lossf(logit.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(ddp.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            run_loss += loss.item()
            run_acc += ((logit > 0).float() == y).float().mean().item()
            if rank == 0 and step % 100 == 0:
                el = time.time() - t0
                print(f"  ep {ep + 1} step {step}/{total} loss {run_loss / 100:.4f} acc {run_acc / 100:.4f} "
                      f"lr {sched.get_last_lr()[0]:.2e} | {el / step:.2f}s/step, eta {(total - step) * el / step / 60:.0f} min",
                      flush=True)
                log.append({"step": step, "loss": run_loss / 100, "acc": run_acc / 100})
                run_loss = run_acc = 0.0
    if rank == 0:
        out = p_model()
        os.makedirs(out, exist_ok=True)
        model.save_pretrained(out, safe_serialization=True)
        tok.save_pretrained(out)
        json.dump({"args": vars(args), "log": log, "base": BASE_MODEL, "max_len": MAX_LEN},
                  open(os.path.join(out, "training_log.json"), "w"), indent=1)
        print(f"saved reranker to {os.path.relpath(out, ROOT)}")
    dist.barrier()
    dist.destroy_process_group()


def stage_train(gpus, args):
    tmp.spawn(_train_worker, args=(gpus, args), nprocs=len(gpus), join=True)


# --------------------------------------------------------------------------------------
# Stage 3: scoring
# --------------------------------------------------------------------------------------
def _predict_worker(rank, gpus, pairs_path, model_dir, batch, out_dir):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    dev = f"cuda:{gpus[rank]}"
    df = pd.read_parquet(pairs_path, columns=["text_a", "text_b"])
    idx = np.arange(rank, len(df), len(gpus))
    A, Bt = df.text_a.values, df.text_b.values
    idx = idx[np.argsort([len(A[i]) + len(Bt[i]) for i in idx])]
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, dtype=torch.float16).to(dev).eval()
    out = np.zeros(len(idx), dtype=np.float32)
    with torch.inference_mode():
        for s in range(0, len(idx), batch):
            b = idx[s:s + batch]
            enc = tok(list(A[b]), list(Bt[b]), padding=True, truncation=True, max_length=MAX_LEN,
                      return_tensors="pt").to(dev)
            out[s:s + len(b)] = model(**enc).logits.squeeze(-1).float().cpu().numpy()
            if rank == 0 and (s // batch) % 500 == 0:
                print(f"    gpu{gpus[rank]}: {s + len(b):,}/{len(idx):,}", flush=True)
    np.save(os.path.join(out_dir, f"shard{rank}_idx.npy"), idx)
    np.save(os.path.join(out_dir, f"shard{rank}_score.npy"), out)


def stage_predict(gpus, batch=1024):
    t = time.time()
    pairs_path = os.path.join(WORK, "test_pairs.parquet")
    parts = os.path.join(WORK, "pred_parts")
    os.makedirs(parts, exist_ok=True)
    tmp.spawn(_predict_worker, args=(gpus, pairs_path, p_model(), batch, parts), nprocs=len(gpus), join=True)
    n = len(pd.read_parquet(pairs_path, columns=["label"]))
    logits = np.zeros(n, dtype=np.float32)
    for r in range(len(gpus)):
        logits[np.load(os.path.join(parts, f"shard{r}_idx.npy"))] = np.load(os.path.join(parts, f"shard{r}_score.npy"))
    np.save(p_logits(), logits)
    print(f"scored {n:,} pairs in {time.time() - t:.0f}s")


# --------------------------------------------------------------------------------------
# Stage 4: evaluation with the challenge metric
# --------------------------------------------------------------------------------------
def macro_f05(df, score, thr, n_true, one_s1_per_record):
    """Per-S1 F0.5 averaged over all S1 (singletons: 1 if nothing predicted, else 0)."""
    pred = score > thr
    if one_s1_per_record:
        sub = df.loc[pred, ["cand_id"]].assign(sc=score[pred])
        best = sub.sort_values("sc", ascending=False).drop_duplicates("cand_id").index
        pred = np.zeros(len(df), bool)
        pred[best] = True
    g = pd.DataFrame({"q": df.s1_id.values, "tp": pred & (df.label.values == 1), "np": pred})
    agg = g.groupby("q").agg(tp=("tp", "sum"), npred=("np", "sum"))
    nt = n_true.reindex(agg.index).fillna(0).values
    tp, npred = agg.tp.values, agg.npred.values
    P = np.divide(tp, npred, out=np.zeros(len(tp)), where=npred > 0)
    R = np.divide(tp, nt, out=np.zeros(len(tp)), where=nt > 0)
    f = np.divide(1.25 * P * R, 0.25 * P + R, out=np.zeros(len(tp)), where=(0.25 * P + R) > 0)
    f = np.where(nt == 0, (npred == 0).astype(float), f)
    return float(f.mean()), float(P[npred > 0].mean()) if (npred > 0).any() else 0.0, float(R[nt > 0].mean())


def stage_eval():
    from sklearn.metrics import average_precision_score, roc_auc_score
    df = pd.read_parquet(os.path.join(WORK, "test_pairs.parquet"), columns=["s1_id", "cand_id", "rank", "cos", "label"])
    logits = np.load(p_logits())
    prob = 1 / (1 + np.exp(-logits))
    gt_train, gt_test = rx.make_split()
    n_true = gt_test.set_index("source1_entity_id").matched_entity_ids.map(lambda s: len([x for x in s.split(",") if x]))
    y = df.label.values
    res = {"n_pairs": len(df), "n_s1": int(df.s1_id.nunique()), "candidate_recall": float(y.sum() / n_true.sum()),
           "auc_reranker": float(roc_auc_score(y, prob)), "ap_reranker": float(average_precision_score(y, prob)),
           "auc_cosine": float(roc_auc_score(y, df.cos.values)), "ap_cosine": float(average_precision_score(y, df.cos.values))}
    print(f"pairs {res['n_pairs']:,}, candidate recall {res['candidate_recall']:.4f}")
    print(f"pair AUC  reranker {res['auc_reranker']:.4f}  cosine {res['auc_cosine']:.4f}")
    print(f"pair AP   reranker {res['ap_reranker']:.4f}  cosine {res['ap_cosine']:.4f}")
    rows = []
    for name, sc, grid in (("cosine", df.cos.values, np.quantile(df.cos.values, np.linspace(0.5, 0.995, 40))),
                           ("reranker", prob, np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.97, 0.99]))):
        for assign in (False, True):
            best = max(((thr,) + macro_f05(df, sc, thr, n_true, assign) for thr in grid), key=lambda x: x[1])
            at5 = macro_f05(df, sc, 0.5, n_true, assign) if name == "reranker" else None
            rows.append({"scorer": name, "one_s1_per_record": assign, "best_thr": float(best[0]),
                         "macro_f05": best[1], "precision": best[2], "recall": best[3],
                         "macro_f05_at_0.5": at5[0] if at5 else None})
            print(f"  {name:9s} assign={str(assign):5s} best thr {best[0]:.4f}: macro F0.5 {best[1]:.4f} "
                  f"(P {best[2]:.4f}, R {best[3]:.4f})" + (f" | at 0.5: {at5[0]:.4f}" if at5 else ""))
    oracle = macro_f05(df, y.astype(float), 0.5, n_true, False)
    print(f"  oracle (perfect classifier on these candidates): macro F0.5 {oracle[0]:.4f}")
    res["f05"] = rows
    res["oracle_f05"] = oracle[0]
    json.dump(res, open(os.path.join(WORK, f"results{TAG}.json"), "w"), indent=1)
    print(f"results -> {os.path.relpath(os.path.join(WORK, f'results{TAG}.json'), ROOT)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="all", choices=["all", "cands", "cands_ft", "train", "predict", "eval"])
    ap.add_argument("--tag", default="", help='"" = v1 reranker, "_v2" = negatives from the fine-tuned retriever')
    ap.add_argument("--gpus", default=",".join(str(i) for i in range(torch.cuda.device_count())))
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=64, help="pairs per GPU per step")
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--port", type=int, default=29521)
    a = ap.parse_args()
    gpus = [int(g) for g in a.gpus.split(",")]
    global TAG
    TAG = a.tag
    if a.stage == "cands_ft":
        stage_cands_ft(gpus)
    if a.stage in ("all", "cands"):
        stage_cands(gpus)
    if a.stage in ("all", "train"):
        stage_train(gpus, a)
    if a.stage in ("all", "predict"):
        stage_predict(gpus)
    if a.stage in ("all", "eval"):
        stage_eval()


if __name__ == "__main__":
    main()

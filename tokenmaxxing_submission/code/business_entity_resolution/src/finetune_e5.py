#!/usr/bin/env python3
"""
Fine-tune multilingual-e5-large for S1 -> S2/S3 retrieval with contrastive learning.

Stages (run all, or one at a time):
  mine   Embed the 80% train split with the base model and mine hard negatives:
         for every train S1 entity, its top-32 FAISS neighbours in the training corpus
         that are not true matches (the near-copy distractors).
  train  DDP on all GPUs. Each example = (S1 text, one random true match, one mined hard
         negative). Symmetric InfoNCE loss with in-batch negatives gathered across GPUs.
  eval   Same evaluation as code/retrieval_experiments.py (20% test split, same corpus,
         same FAISS search), so results compare directly with the base model.

Leakage control: test-split S1 records, their matches, and the distractors sampled into
the evaluation corpus are all excluded from mining and training.

Usage (from the project root):
    python code/finetune_e5.py                 # mine + train + eval
    python code/finetune_e5.py --stage train --epochs 1 --batch 128
    python code/finetune_e5.py --stage eval
"""
import argparse
import datetime
import json
import math
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
WORK = os.path.join(ROOT, "experiments", "finetune")
OUT_MODEL = os.path.join(ROOT, "models", "e5-large-ft")
BASE = "e5-large"
FIELD = "embed_text"
MAX_LEN = 96
N_MINE = 32          # neighbours retrieved per anchor when mining
N_HARD = 16          # hard negatives kept per anchor


# --------------------------------------------------------------------------------------
# Stage 1: training data + hard-negative mining
# --------------------------------------------------------------------------------------
def build_training_data():
    """Anchors (train S1 with >=1 match), their positives, and the training corpus."""
    gt_train, gt_test = rx.make_split()
    # evaluation corpus (to exclude its distractors from training)
    _, eval_corpus, _ = rx.build_eval_set(gt_train, gt_test, [FIELD])
    cols = ["entity_id", "country", FIELD]
    s1 = rx.read_tsv(os.path.join(rx.PROC, "train", "train_source1.tsv"), cols)
    o = pd.concat([rx.read_tsv(os.path.join(rx.PROC, "train", f"train_source{s}.tsv"), cols) for s in (2, 3)],
                  ignore_index=True)
    p = gt_train.assign(m=gt_train.matched_entity_ids.str.split(",")).explode("m")
    p = p[p.m != ""][["source1_entity_id", "m"]]
    excluded = set(eval_corpus.entity_id)                  # test matches + eval distractors
    corpus = o[~o.entity_id.isin(excluded)].reset_index(drop=True)
    anchors = s1[s1.entity_id.isin(set(p.source1_entity_id))].reset_index(drop=True)
    c_pos = pd.Series(np.arange(len(corpus)), index=corpus.entity_id)
    a_pos = pd.Series(np.arange(len(anchors)), index=anchors.entity_id)
    p = p.assign(ai=p.source1_entity_id.map(a_pos).values, ci=p.m.map(c_pos).values).sort_values("ai")
    assert p.ci.notna().all(), "a training positive is missing from the training corpus"
    offsets = np.zeros(len(anchors) + 1, dtype=np.int64)
    np.add.at(offsets, p.ai.values + 1, 1)
    offsets = np.cumsum(offsets)
    pos = p.ci.values.astype(np.int64)
    print(f"training data: {len(anchors):,} anchors, {len(pos):,} positive pairs, corpus {len(corpus):,} "
          f"(excluded {len(excluded):,} eval-corpus records)")
    return anchors, corpus, offsets, pos


def stage_mine(gpus):
    os.makedirs(WORK, exist_ok=True)
    anchors, corpus, offsets, pos = build_training_data()
    cfg = dict(rx.MODELS[BASE], path=rx.model_path(rx.MODELS[BASE]["hf"]))
    ae = rx.embed(anchors[FIELD].values, cfg, MAX_LEN, gpus, os.path.join(WORK, "base_anchors.npy"))
    ce = rx.embed(corpus[FIELD].values, cfg, MAX_LEN, gpus, os.path.join(WORK, "base_corpus.npy"))
    t = time.time()
    top = rx.search(ae, ce, anchors.country.values, corpus.country.values, N_MINE, gpus)
    print(f"  mined top-{N_MINE} in {time.time() - t:.0f}s")
    hard = np.full((len(anchors), N_HARD), -1, dtype=np.int64)
    found = 0
    for i in range(len(anchors)):
        p = set(pos[offsets[i]:offsets[i + 1]].tolist())
        found += len(p & set(top[i].tolist()))
        neg = [c for c in top[i] if c >= 0 and c not in p][:N_HARD]
        hard[i, :len(neg)] = neg
    print(f"  base-model recall@{N_MINE} on training anchors: {found / len(pos):.3f}")
    anchors[["entity_id", "country", FIELD]].to_parquet(os.path.join(WORK, "anchors.parquet"))
    corpus[["entity_id", "country", FIELD]].to_parquet(os.path.join(WORK, "corpus.parquet"))
    np.savez(os.path.join(WORK, "train_index.npz"), offsets=offsets, pos=pos, hard=hard)
    print(f"  saved training data to {os.path.relpath(WORK, ROOT)}")


# --------------------------------------------------------------------------------------
# Stage 2: DDP contrastive training
# --------------------------------------------------------------------------------------
def _encode(model, enc):
    h = model(**enc).last_hidden_state
    m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
    v = (h * m).sum(1) / m.sum(1).clamp(min=1)          # mean pooling, as e5
    return torch.nn.functional.normalize(v.float(), dim=-1)


def _train_worker(rank, gpus, args):
    from transformers import AutoModel, AutoTokenizer
    from torch.distributed.nn.functional import all_gather
    world = len(gpus)
    torch.cuda.set_device(gpus[rank])
    dev = torch.device(f"cuda:{gpus[rank]}")
    dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{args.port}", rank=rank, world_size=world,
                            timeout=datetime.timedelta(minutes=60))
    anchors = pd.read_parquet(os.path.join(WORK, "anchors.parquet"))[FIELD].values
    corpus = pd.read_parquet(os.path.join(WORK, "corpus.parquet"))[FIELD].values
    z = np.load(os.path.join(WORK, "train_index.npz"))
    offsets, pos, hard = z["offsets"], z["pos"], z["hard"]

    path = rx.model_path(rx.MODELS[BASE]["hf"])
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModel.from_pretrained(path, add_pooling_layer=False).to(dev)   # pooler unused (mean pooling)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    ddp = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpus[rank]])
    opt = torch.optim.AdamW(ddp.parameters(), lr=args.lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda")
    prefix = rx.MODELS[BASE]["prefix"]

    steps_per_epoch = len(anchors) // (args.batch * world)
    if args.max_steps:
        steps_per_epoch = min(steps_per_epoch, args.max_steps)
    total = steps_per_epoch * args.epochs
    warm = max(1, int(0.05 * total))
    def lr_lambda(s):                                   # linear warmup, then linear decay to 0
        if s < warm:
            return (s + 1) / warm
        return max(0.0, (total - s) / max(1, total - warm))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    rng = np.random.default_rng(1234 + rank)

    def tokenize(texts):
        return tok([prefix + (t or " ") for t in texts], padding=True, truncation=True, max_length=MAX_LEN,
                   return_tensors="pt").to(dev)

    step, t0, log = 0, time.time(), []
    for ep in range(args.epochs):
        order = np.random.default_rng(99 + ep).permutation(len(anchors))   # same order on every rank
        for s in range(steps_per_epoch):
            ids = order[(s * world + rank) * args.batch:(s * world + rank + 1) * args.batch]
            a_txt, p_txt, n_txt = [], [], []
            for i in ids:
                a_txt.append(anchors[i])
                p_txt.append(corpus[pos[rng.integers(offsets[i], offsets[i + 1])]])
                h = hard[i][hard[i] >= 0]
                n_txt.append(corpus[h[rng.integers(len(h))]] if len(h) else corpus[rng.integers(len(corpus))])
            with torch.autocast("cuda", dtype=torch.float16):
                # one forward pass for anchors, positives and negatives (DDP-safe with checkpointing)
                q = _encode(ddp, tokenize(a_txt + p_txt + n_txt))
                qa, qp, qn = q.split(len(ids))
            # gather across GPUs (differentiable) -> global in-batch negatives
            ga = torch.cat(all_gather(qa)); gp = torch.cat(all_gather(qp)); gn = torch.cat(all_gather(qn))
            labels = torch.arange(len(qa), device=dev) + rank * len(qa)
            logits_a = qa @ torch.cat([gp, gn]).T / args.temp       # anchor -> {positives, hard negatives}
            logits_p = qp @ ga.T / args.temp                         # positive -> anchors
            loss = (torch.nn.functional.cross_entropy(logits_a, labels) +
                    torch.nn.functional.cross_entropy(logits_p, labels)) / 2
            opt.zero_grad(set_to_none=True)
            # all_gather's backward sums gradients over ranks and DDP averages them,
            # so this gives the gradient of the mean loss across GPUs.
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(ddp.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            if rank == 0 and (step % 50 == 0 or step == 1):
                acc = (logits_a.argmax(1) == labels).float().mean().item()
                el = time.time() - t0
                print(f"  ep {ep + 1} step {step}/{total} loss {loss.item():.4f} in-batch acc {acc:.3f} "
                      f"lr {sched.get_last_lr()[0]:.2e} | {el / step:.2f}s/step, eta {(total - step) * el / step / 60:.0f} min",
                      flush=True)
                log.append({"step": step, "loss": loss.item(), "acc": acc})
    if rank == 0:
        os.makedirs(OUT_MODEL, exist_ok=True)
        model.save_pretrained(OUT_MODEL, safe_serialization=True)
        tok.save_pretrained(OUT_MODEL)
        json.dump({"args": vars(args), "log": log, "base": rx.MODELS[BASE]["hf"]},
                  open(os.path.join(OUT_MODEL, "training_log.json"), "w"), indent=1)
        print(f"saved fine-tuned model to {os.path.relpath(OUT_MODEL, ROOT)}")
    dist.barrier()
    dist.destroy_process_group()


def stage_train(gpus, args):
    tmp.spawn(_train_worker, args=(gpus, args), nprocs=len(gpus), join=True)


# --------------------------------------------------------------------------------------
# Stage 3: evaluation (same protocol as retrieval_experiments.py)
# --------------------------------------------------------------------------------------
def stage_eval(gpus):
    out = rx.OUT
    emb = os.path.join(out, "emb")
    for f in os.listdir(emb) if os.path.isdir(emb) else []:
        if f.startswith("e5-large-ft_"):              # the model changed -> drop stale embeddings
            os.remove(os.path.join(emb, f))
    sys.argv = [sys.argv[0], "--models", "e5-large-ft", "--fields", FIELD, "--gpus", ",".join(map(str, gpus))]
    rx.main()
    res = json.load(open(os.path.join(out, "results.json")))
    print("\n  pair recall (all)      " + "  ".join(f"@{k}" for k in rx.KS))
    for m in (BASE, "e5-large-ft"):
        r = res.get(f"{m}|{FIELD}")
        if r:
            print(f"  {m:14s} " + "  ".join(f"{r['all']['pair_recall'][str(k)] * 100:.1f}" for k in rx.KS))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="all", choices=["all", "mine", "train", "eval"])
    ap.add_argument("--gpus", default=",".join(str(i) for i in range(torch.cuda.device_count())))
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=128, help="anchors per GPU per step")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--temp", type=float, default=0.02)
    ap.add_argument("--max-steps", type=int, default=0, help="cap steps per epoch (0 = full epoch)")
    ap.add_argument("--port", type=int, default=29511)
    a = ap.parse_args()
    gpus = [int(g) for g in a.gpus.split(",")]
    if a.stage in ("all", "mine"):
        stage_mine(gpus)
    if a.stage in ("all", "train"):
        stage_train(gpus, a)
    if a.stage in ("all", "eval"):
        stage_eval(gpus)


if __name__ == "__main__":
    main()

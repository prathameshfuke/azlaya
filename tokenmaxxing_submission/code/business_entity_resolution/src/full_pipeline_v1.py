#!/usr/bin/env python3
"""
Full inference pipeline v1 on the real test set -> submission files.

  1. embed    test S1 and test S2+S3 `embed_text` with the fine-tuned e5-large (models/e5-large-ft)
  2. retrieve top-K S2/S3 candidates per S1 with FAISS, within country (open set of countries)
  3. rerank   every (S1, candidate) pair with the mDeBERTa cross-encoder (models/mdeberta-reranker)
              on "name_clean | addr_clean"
  4. decide   keep pairs with probability > thr; each S2/S3 record goes to at most one S1
              (the highest-scoring one), and to none if >= 2 S1s pass (abstain) -
              the best rule on the 20% validation split (macro F0.5 0.9914)
  5. write    output/matching_results.tsv  (one row per test S1, empty if no match)
              output/candidate_pairs.tsv   (the exact top-K fed to the reranker)
              and run utils/validate_submission.py

Every step is cached under --cache (large files; kept outside the home quota by default),
so a re-run resumes where it stopped. Change --thr / --no-abstain to re-decide in seconds.

Usage (from the project root):
    python code/full_pipeline_v1.py --gpus 4,5,6,7
    python code/full_pipeline_v1.py --thr 0.5 --no-abstain      # re-decide only (uses caches)
"""
import argparse
import csv
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as tmp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import retrieval_experiments as rx  # noqa: E402

ROOT = rx.ROOT
TEST_DIR = os.path.join(rx.PROC, "test")
RETRIEVER = "e5-large-ft"
RERANKER_DIR = os.path.join(ROOT, "models", "mdeberta-reranker")
RERANK_MAX_LEN = 160
OUT_DIR = os.path.join(ROOT, "output")


def load_test():
    cols = ["entity_id", "country", "embed_text", "name_clean", "addr_clean"]
    s1 = rx.read_tsv(os.path.join(TEST_DIR, "test_source1.tsv"), cols)
    oth = pd.concat([rx.read_tsv(os.path.join(TEST_DIR, f"test_source{s}.tsv"), cols) for s in (2, 3)],
                    ignore_index=True)
    for d in (s1, oth):
        d["rr_text"] = d.name_clean + " | " + d.addr_clean
    print(f"test: {len(s1):,} S1, {len(oth):,} S2+S3; countries S1 {s1.country.value_counts().to_dict()}")
    return s1, oth


# --------------------------------------------------------------------------------------
# Reranker scoring on index pairs (texts are not duplicated per pair)
# --------------------------------------------------------------------------------------
def _score_worker(rank, gpus, cache, batch):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    dev = f"cuda:{gpus[rank]}"
    qt = pd.read_parquet(os.path.join(cache, "s1_text.parquet")).rr_text.values
    ct = pd.read_parquet(os.path.join(cache, "cand_text.parquet")).rr_text.values
    z = np.load(os.path.join(cache, "candidates.npz"))
    qi, ci = z["qi"], z["ci"]
    idx = np.arange(rank, len(qi), len(gpus))
    idx = idx[np.argsort([len(qt[qi[i]]) + len(ct[ci[i]]) for i in idx], kind="stable")]
    tok = AutoTokenizer.from_pretrained(RERANKER_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(RERANKER_DIR, dtype=torch.float16).to(dev).eval()
    out = np.zeros(len(idx), dtype=np.float32)
    t = time.time()
    with torch.inference_mode():
        for s in range(0, len(idx), batch):
            b = idx[s:s + batch]
            enc = tok(list(qt[qi[b]]), list(ct[ci[b]]), padding=True, truncation=True,
                      max_length=RERANK_MAX_LEN, return_tensors="pt").to(dev)
            out[s:s + len(b)] = model(**enc).logits.squeeze(-1).float().cpu().numpy()
            if rank == 0 and (s // batch) % 1000 == 0:
                el = time.time() - t
                print(f"    gpu{gpus[rank]}: {s + len(b):,}/{len(idx):,} pairs, eta {(len(idx) - s) * el / max(1, s) / 60:.0f} min",
                      flush=True)
    np.save(os.path.join(cache, "score_parts", f"shard{rank}_idx.npy"), idx)
    np.save(os.path.join(cache, "score_parts", f"shard{rank}_logit.npy"), out)


def score_pairs(cache, gpus, batch=1024):
    path = os.path.join(cache, "logits.npy")
    if os.path.exists(path):
        return np.load(path)
    os.makedirs(os.path.join(cache, "score_parts"), exist_ok=True)
    t = time.time()
    tmp.spawn(_score_worker, args=(gpus, cache, batch), nprocs=len(gpus), join=True)
    n = len(np.load(os.path.join(cache, "candidates.npz"))["qi"])
    logits = np.zeros(n, dtype=np.float32)
    for r in range(len(gpus)):
        logits[np.load(os.path.join(cache, "score_parts", f"shard{r}_idx.npy"))] = \
            np.load(os.path.join(cache, "score_parts", f"shard{r}_logit.npy"))
    np.save(path, logits)
    print(f"  reranked {n:,} pairs in {(time.time() - t) / 60:.1f} min")
    return logits


# --------------------------------------------------------------------------------------
def decide(qi, ci, prob, thr, abstain):
    """Threshold; each candidate record to its best S1 only; optionally to none if >= 2 S1s pass."""
    ok = np.where(prob > thr)[0]
    order = ok[np.argsort(-prob[ok], kind="stable")]
    c = pd.Series(ci[order])
    keep = order[~c.duplicated().values]
    if abstain:
        multi = c[c.duplicated()].unique()
        keep = keep[~np.isin(ci[keep], multi)]
    return keep


def write_tsv(path, header, s1_ids, lists):
    with open(path, "w", encoding="utf-8") as f:
        f.write("\t".join(header) + "\n")
        for s, l in zip(s1_ids, lists):
            f.write(f"{s}\t{','.join(l)}\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpus", default=",".join(str(i) for i in range(torch.cuda.device_count())))
    ap.add_argument("--k", type=int, default=20, help="candidates per S1")
    ap.add_argument("--thr", type=float, default=0.7)
    ap.add_argument("--no-abstain", action="store_true")
    ap.add_argument("--cache", default="/tmp/amazon_ml_cache/full_v1")
    ap.add_argument("--out", default=OUT_DIR)
    a = ap.parse_args()
    gpus = [int(g) for g in a.gpus.split(",")]
    os.makedirs(a.cache, exist_ok=True)
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()

    s1, oth = load_test()
    s1[["entity_id", "rr_text"]].to_parquet(os.path.join(a.cache, "s1_text.parquet"))
    oth[["entity_id", "rr_text"]].to_parquet(os.path.join(a.cache, "cand_text.parquet"))

    # 1-2. retrieval
    cand_path = os.path.join(a.cache, "candidates.npz")
    if not os.path.exists(cand_path):
        cfg = dict(rx.MODELS[RETRIEVER], path=rx.model_path(rx.MODELS[RETRIEVER]["hf"]))
        qe = rx.embed(s1.embed_text.values, cfg, 128, gpus, os.path.join(a.cache, "emb_s1.npy"))
        ce = rx.embed(oth.embed_text.values, cfg, 128, gpus, os.path.join(a.cache, "emb_cand.npy"))
        t = time.time()
        top = rx.search(qe, ce, s1.country.values, oth.country.values, a.k, gpus)
        qi = np.repeat(np.arange(len(s1)), a.k)
        ci = top.reshape(-1)
        keep = ci >= 0
        qi, ci = qi[keep], ci[keep]
        cos = np.empty(len(qi), dtype=np.float32)          # chunked: all pairs at once would need ~300 GB
        for s in range(0, len(qi), 1_000_000):
            cos[s:s + 1_000_000] = np.einsum("ij,ij->i", qe[qi[s:s + 1_000_000]].astype(np.float32),
                                             ce[ci[s:s + 1_000_000]].astype(np.float32))
        np.savez(cand_path, qi=qi, ci=ci, cos=cos)
        print(f"  retrieved {len(qi):,} candidate pairs in {(time.time() - t) / 60:.1f} min")
    z = np.load(cand_path)
    qi, ci = z["qi"], z["ci"]

    # 3. reranking
    prob = 1 / (1 + np.exp(-score_pairs(a.cache, gpus)))

    # 4. decision
    keep = decide(qi, ci, prob, a.thr, not a.no_abstain)
    print(f"decision: thr {a.thr}, abstain {not a.no_abstain} -> {len(keep):,} matched pairs")

    # 5. outputs
    s1_ids, c_ids = s1.entity_id.values, oth.entity_id.values
    match = pd.Series(c_ids[ci[keep]]).groupby(qi[keep]).agg(list)
    cands = pd.Series(c_ids[ci]).groupby(qi).agg(list)
    empty = []
    m_lists = [match.get(i, empty) for i in range(len(s1))]
    c_lists = [cands.get(i, empty) for i in range(len(s1))]
    mp, cp = os.path.join(a.out, "matching_results.tsv"), os.path.join(a.out, "candidate_pairs.tsv")
    write_tsv(mp, ["source1_entity_id", "matched_entity_ids"], s1_ids, m_lists)
    write_tsv(cp, ["source1_entity_id", "candidate_entity_ids"], s1_ids, c_lists)
    n_empty = sum(len(l) == 0 for l in m_lists)
    print(f"wrote {os.path.relpath(mp, ROOT)} ({len(s1):,} rows, {n_empty:,} empty = {n_empty / len(s1):.1%}, "
          f"mean {len(keep) / len(s1):.2f} matches per S1) and {os.path.relpath(cp, ROOT)}")
    by_c = pd.Series([len(l) for l in m_lists]).groupby(s1.country.values).agg(["mean", lambda x: (x == 0).mean()])
    by_c.columns = ["mean matches", "share empty"]
    print(by_c.round(3).to_string())

    # validation with the official checker
    r = subprocess.run([sys.executable, "utils/validate_submission.py", "--matching", mp, "--candidate", cp,
                        "--test-dir", "dataset/test"], cwd=os.path.join(ROOT, "student_resource"),
                       capture_output=True, text=True)
    print(r.stdout[-1500:])
    print(f"total time {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()

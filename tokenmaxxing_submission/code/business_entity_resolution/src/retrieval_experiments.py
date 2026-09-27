#!/usr/bin/env python3
"""
Embedding-retrieval experiments for the Amazon ML Challenge 2026 entity-resolution task.

Experiment 1: embed only the business name   (column `name_core`)
Experiment 2: embed name + address           (column `embed_text`)

Setup
  * The training ground truth is split 80/20 by Source-1 entity (seed 42) and saved to
    dataset_processed/split/gt_train.tsv and gt_test.tsv (reused if they exist).
  * Queries  = S1 entities in the 20% test split.
  * Corpus   = S2/S3 records matched to those entities + a random 20% of the unmatched
               (distractor) S2/S3 records, so the distractor ratio matches the full data.
               Records owned by the 80% train split are left out.
  * Search   = FAISS top-k by cosine (inner product on L2-normalised vectors), one index per
               country: exact GPU flat index by default, or --index ivf for approximate search.
  * Metrics  = pair recall@k (share of true S1-S2/S3 pairs retrieved in the query's top k),
               entity full recall@k (share of non-singleton S1 entities with all matches in top k),
               reported overall and per country.

Usage (from the project root):
    python code/retrieval_experiments.py                               # all models, both experiments
    python code/retrieval_experiments.py --models e5-base --fields name_core
    python code/retrieval_experiments.py --gpus 0,1,2,3 --index ivf

Outputs go to <out> (default: experiments/retrieval/): cached embeddings (emb/),
results.json and results.md.
"""
import argparse
import csv
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as tmp

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, "student_resource", "dataset_processed")
OUT = os.path.join(ROOT, "experiments", "retrieval")
KS = (1, 5, 10, 20, 50, 100)
SEED = 42

# All MIT-licensed and under 600M parameters.
MODELS = {
    "e5-base": {"hf": "intfloat/multilingual-e5-base", "pool": "mean", "prefix": "query: ", "params": "278M"},
    "e5-large": {"hf": "intfloat/multilingual-e5-large", "pool": "mean", "prefix": "query: ", "params": "560M"},
    "bge-m3": {"hf": "BAAI/bge-m3", "pool": "cls", "prefix": "", "params": "568M"},
}
MODEL_DIR = os.path.join(ROOT, "models")
# e5-large fine-tuned on the 80% train split (code/finetune_e5.py); local path, not a hub id.
MODELS["e5-large-ft"] = {"hf": os.path.join(MODEL_DIR, "e5-large-ft"), "pool": "mean", "prefix": "query: ", "params": "560M"}
BASE_MODELS = ["e5-base", "e5-large", "bge-m3"]


def model_path(hf_id):
    """Local copy of the model with safetensors weights.

    Some repos (BAAI/bge-m3) only ship pytorch_model.bin, which recent transformers refuses to
    load on torch < 2.6. Convert it once with torch.load(weights_only=True) and reuse the copy.
    """
    if os.path.isdir(hf_id):                       # local checkpoint
        return hf_id
    from huggingface_hub import list_repo_files, snapshot_download
    files = list_repo_files(hf_id)
    if any(f.endswith(".safetensors") and "/" not in f for f in files):
        return snapshot_download(hf_id, allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt"],
                                 ignore_patterns=["onnx/*", "openvino/*"])
    out = os.path.join(MODEL_DIR, hf_id.replace("/", "__"))
    if os.path.exists(os.path.join(out, "model.safetensors")):
        return out
    from safetensors.torch import save_file
    src = snapshot_download(hf_id, allow_patterns=["*.json", "pytorch_model.bin", "*.model", "*.txt"],
                            ignore_patterns=["onnx/*", "openvino/*"])
    os.makedirs(out, exist_ok=True)
    for f in os.listdir(src):
        if f != "pytorch_model.bin" and os.path.isfile(os.path.join(src, f)):
            with open(os.path.join(src, f), "rb") as fi, open(os.path.join(out, f), "wb") as fo:
                fo.write(fi.read())
    sd = torch.load(os.path.join(src, "pytorch_model.bin"), map_location="cpu", weights_only=True)
    save_file({k: v.contiguous() for k, v in sd.items()}, os.path.join(out, "model.safetensors"))
    print(f"  converted {hf_id} weights to safetensors -> {os.path.relpath(out, ROOT)}")
    return out
FIELDS = {"name_core": {"exp": "Experiment 1: name only", "max_len": 48},
          "embed_text": {"exp": "Experiment 2: name + address", "max_len": 128}}


def read_tsv(path, usecols=None):
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE,
                       escapechar="\\", usecols=usecols)


# --------------------------------------------------------------------------------------
# Split and evaluation set
# --------------------------------------------------------------------------------------
def make_split(frac_test=0.2, seed=SEED):
    d = os.path.join(PROC, "split")
    tr, te = os.path.join(d, "gt_train.tsv"), os.path.join(d, "gt_test.tsv")
    if os.path.exists(tr) and os.path.exists(te):
        return read_tsv(tr), read_tsv(te)
    os.makedirs(d, exist_ok=True)
    gt = read_tsv(os.path.join(PROC, "train", "train_ground_truth.tsv"))
    rng = np.random.default_rng(seed)
    is_test = rng.random(len(gt)) < frac_test
    gtr, gte = gt[~is_test], gt[is_test]
    for df, p in ((gtr, tr), (gte, te)):
        df.to_csv(p, sep="\t", index=False, quoting=csv.QUOTE_NONE, escapechar="\\")
    print(f"split saved: train {len(gtr):,} / test {len(gte):,} S1 entities -> {os.path.relpath(d, ROOT)}")
    return gtr, gte


def build_eval_set(gt_train, gt_test, fields, distractor_frac=0.2, seed=SEED):
    cols = ["entity_id", "country"] + list(fields)
    s1 = read_tsv(os.path.join(PROC, "train", "train_source1.tsv"), cols)
    o = pd.concat([read_tsv(os.path.join(PROC, "train", f"train_source{s}.tsv"), cols) for s in (2, 3)], ignore_index=True)

    def explode(g):
        p = g.assign(m=g.matched_entity_ids.str.split(",")).explode("m")
        return p[p.m != ""][["source1_entity_id", "m"]]

    p_test, p_train = explode(gt_test), explode(gt_train)
    owned_test, owned_train = set(p_test.m), set(p_train.m)
    unmatched = ~o.entity_id.isin(owned_test | owned_train)
    rng = np.random.default_rng(seed)
    keep = o.entity_id.isin(owned_test) | (unmatched & (rng.random(len(o)) < distractor_frac))
    corpus = o[keep].reset_index(drop=True)
    queries = s1[s1.entity_id.isin(set(gt_test.source1_entity_id))].reset_index(drop=True)
    print(f"eval set: {len(queries):,} queries (test S1), corpus {len(corpus):,} "
          f"({int(corpus.entity_id.isin(owned_test).sum()):,} true matches + "
          f"{int((~corpus.entity_id.isin(owned_test)).sum()):,} distractors), {len(p_test):,} true pairs")
    return queries, corpus, p_test


# --------------------------------------------------------------------------------------
# Multi-GPU embedding
# --------------------------------------------------------------------------------------
def _embed_worker(rank, gpus, texts, cfg, max_len, batch, tmp_dir):
    from transformers import AutoModel, AutoTokenizer
    dev = f"cuda:{gpus[rank]}"
    tok = AutoTokenizer.from_pretrained(cfg["path"])
    model = AutoModel.from_pretrained(cfg["path"], dtype=torch.float16).to(dev).eval()
    idx = np.arange(rank, len(texts), len(gpus))
    # sort by length so batches have little padding
    idx = idx[np.argsort([len(texts[i]) for i in idx])]
    out = np.zeros((len(idx), model.config.hidden_size), dtype=np.float16)
    with torch.inference_mode():
        for s in range(0, len(idx), batch):
            b = [cfg["prefix"] + (texts[i] or " ") for i in idx[s:s + batch]]
            enc = tok(b, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(dev)
            h = model(**enc).last_hidden_state
            if cfg["pool"] == "cls":
                v = h[:, 0]
            else:
                m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
                v = (h * m).sum(1) / m.sum(1).clamp(min=1)
            v = torch.nn.functional.normalize(v.float(), dim=-1)
            out[s:s + len(b)] = v.cpu().numpy().astype(np.float16)
            if rank == 0 and (s // batch) % 200 == 0:
                print(f"    gpu{gpus[rank]}: {s + len(b):,}/{len(idx):,}", flush=True)
    np.save(os.path.join(tmp_dir, f"shard{rank}_idx.npy"), idx)
    np.save(os.path.join(tmp_dir, f"shard{rank}_emb.npy"), out)


def embed(texts, cfg, max_len, gpus, cache_path, batch=512):
    if os.path.exists(cache_path):
        return np.load(cache_path)
    t = time.time()
    tmp_dir = cache_path + ".parts"
    os.makedirs(tmp_dir, exist_ok=True)
    tmp.spawn(_embed_worker, args=(gpus, list(texts), cfg, max_len, batch, tmp_dir), nprocs=len(gpus), join=True)
    embs = None
    for r in range(len(gpus)):
        idx = np.load(os.path.join(tmp_dir, f"shard{r}_idx.npy"))
        e = np.load(os.path.join(tmp_dir, f"shard{r}_emb.npy"))
        if embs is None:
            embs = np.zeros((len(texts), e.shape[1]), dtype=np.float16)
        embs[idx] = e
    np.save(cache_path, embs)
    for f in os.listdir(tmp_dir):
        os.remove(os.path.join(tmp_dir, f))
    os.rmdir(tmp_dir)
    print(f"  embedded {len(texts):,} texts in {time.time() - t:.0f}s -> {os.path.relpath(cache_path, ROOT)}")
    return embs


# --------------------------------------------------------------------------------------
# Search and metrics
# --------------------------------------------------------------------------------------
def build_index(vecs, kind, gpus):
    """FAISS inner-product index (vectors are L2-normalised, so IP = cosine).

    flat: exact search, sharded over the GPUs in fp16.
    ivf : approximate IVF-Flat (nlist ~ 4*sqrt(N), nprobe 64) for very large corpora.
    """
    import faiss
    d = vecs.shape[1]
    if kind == "flat":
        cpu = faiss.IndexFlatIP(d)
    else:
        nlist = int(4 * np.sqrt(len(vecs)))
        cpu = faiss.IndexIVFFlat(faiss.IndexFlatIP(d), d, nlist, faiss.METRIC_INNER_PRODUCT)
    if gpus and faiss.get_num_gpus() > 0:
        co = faiss.GpuMultipleClonerOptions()
        co.shard = True
        co.useFloat16 = True
        res = [faiss.StandardGpuResources() for _ in gpus]
        index = faiss.index_cpu_to_gpu_multiple_py(res, cpu, co, gpus)
    else:
        index = cpu
    if kind != "flat":
        rng = np.random.default_rng(SEED)
        train = vecs[rng.choice(len(vecs), min(len(vecs), 50 * nlist), replace=False)]
        index.train(train)
        faiss.ParameterSpace().set_index_parameter(index, "nprobe", 64)
    index.add(vecs)                      # sharded indexes accept a single add() call
    return index


def search(q_emb, c_emb, q_country, c_country, k, gpus, kind="flat", chunk=65536):
    """Top-k retrieval with FAISS, one index per country. Returns corpus indices (n_q, k), -1 = none."""
    top = np.full((len(q_emb), k), -1, dtype=np.int64)
    for c in np.unique(q_country):
        qi = np.where(q_country == c)[0]
        ci = np.where(c_country == c)[0]
        if len(ci) == 0:
            continue
        index = build_index(c_emb[ci].astype(np.float32), kind, gpus)
        kk = min(k, len(ci))
        for s in range(0, len(qi), chunk):
            _, I = index.search(q_emb[qi[s:s + chunk]].astype(np.float32), kk)
            top[qi[s:s + chunk], :kk] = np.where(I >= 0, ci[np.clip(I, 0, None)], -1)
        del index
    return top


def evaluate(top, queries, corpus, pairs):
    q_pos = {e: i for i, e in enumerate(queries.entity_id)}
    c_pos = {e: i for i, e in enumerate(corpus.entity_id)}
    qi = pairs.source1_entity_id.map(q_pos).values
    ci = pairs.m.map(c_pos).values
    # rank of each true pair inside its query's top-k list (k_max if not found)
    kmax = top.shape[1]
    hits = top[qi] == ci[:, None]
    rank = np.where(hits.any(1), hits.argmax(1), kmax)
    country = queries.country.values[qi]
    res = {}
    for scope in ["all"] + sorted(set(country)):
        m = np.ones(len(rank), bool) if scope == "all" else country == scope
        r = rank[m]
        q = qi[m]
        ent = {}
        for k in KS:
            found = pd.Series(r < k).groupby(q).all()
            ent[k] = float(found.mean())
        res[scope] = {"n_pairs": int(m.sum()),
                      "pair_recall": {k: float((r < k).mean()) for k in KS},
                      "entity_full_recall": ent}
    return res


# --------------------------------------------------------------------------------------
def write_markdown(all_res, path):
    lines = ["# Retrieval experiments", "",
             "Pair recall@k = share of true S1–S2/S3 pairs found in the S1 query's top-k (within country).",
             "Entity full recall@k = share of S1 entities with **all** their matches in the top-k.", ""]
    for field, fcfg in FIELDS.items():
        rows = [(m, r) for (m, f), r in all_res.items() if f == field]
        if not rows:
            continue
        lines += [f"## {fcfg['exp']} (`{field}`)", "",
                  "| Model | Params | Scope | " + " | ".join(f"R@{k}" for k in KS) + " | " + " | ".join(f"Full@{k}" for k in (10, 50, 100)) + " |",
                  "|---|---|---|" + "---|" * (len(KS) + 3)]
        for m, r in rows:
            for scope, v in r.items():
                lines.append(f"| {m} | {MODELS[m]['params']} | {scope} | " +
                             " | ".join(f"{v['pair_recall'][k] * 100:.1f}" for k in KS) + " | " +
                             " | ".join(f"{v['entity_full_recall'][k] * 100:.1f}" for k in (10, 50, 100)) + " |")
        lines.append("")
    open(path, "w").write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=BASE_MODELS, choices=list(MODELS))
    ap.add_argument("--fields", nargs="+", default=list(FIELDS), choices=list(FIELDS))
    ap.add_argument("--gpus", default=",".join(str(i) for i in range(torch.cuda.device_count())))
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--index", default="flat", choices=["flat", "ivf"], help="FAISS index: exact flat or approximate IVF")
    a = ap.parse_args()
    gpus = [int(g) for g in a.gpus.split(",")]
    os.makedirs(os.path.join(a.out, "emb"), exist_ok=True)

    gt_train, gt_test = make_split()
    queries, corpus, pairs = build_eval_set(gt_train, gt_test, a.fields)
    res_path = os.path.join(a.out, "results.json")
    saved = json.load(open(res_path)) if os.path.exists(res_path) else {}
    all_res = {tuple(k.split("|")): {s: {kk: {int(x): y for x, y in vv.items()} if isinstance(vv, dict) else vv
                                          for kk, vv in v.items()} for s, v in r.items()}
               for k, r in saved.items()}
    for model in a.models:
        cfg = dict(MODELS[model], path=model_path(MODELS[model]["hf"]))
        for field in a.fields:
            fc = FIELDS[field]
            print(f"\n=== {fc['exp']} | {model} ({cfg['hf']}) | field={field} | index={a.index}")
            qe = embed(queries[field].values, cfg, fc["max_len"], gpus, os.path.join(a.out, "emb", f"{model}_{field}_queries.npy"), a.batch)
            ce = embed(corpus[field].values, cfg, fc["max_len"], gpus, os.path.join(a.out, "emb", f"{model}_{field}_corpus.npy"), a.batch)
            t = time.time()
            top = search(qe, ce, queries.country.values, corpus.country.values, max(KS), gpus, a.index)
            r = evaluate(top, queries, corpus, pairs)
            print(f"  search + eval in {time.time() - t:.0f}s")
            for scope, v in r.items():
                print(f"  {scope:6s} pair recall " + " ".join(f"@{k}={v['pair_recall'][k]:.3f}" for k in KS) +
                      f" | entity full recall @10={v['entity_full_recall'][10]:.3f} @100={v['entity_full_recall'][100]:.3f}")
            all_res[(model, field)] = r
            json.dump({f"{m}|{f}": v for (m, f), v in all_res.items()}, open(res_path, "w"), indent=1)
            write_markdown(all_res, os.path.join(a.out, "results.md"))
    print(f"\nresults: {os.path.relpath(res_path, ROOT)} and results.md")


if __name__ == "__main__":
    main()

"""Step 5: Laya fine-tuning -- dataset build + RLCD training for the "same_entity" noul question.

Follows the recipe in NandhaKishorM/laya's own fine-tuning notebook
(notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb): gold-distribution soft targets,
a noisy-logit policy-gradient objective scored with `laya.common.proper_reward` (a strictly
proper scoring rule) plus a soft cross-entropy term, then post-hoc temperature calibration on a
held-out slice. That notebook fine-tunes on an existing HF benchmark dataset; this script builds
the *same* training-item shape from our own candidate pairs + ground truth instead.

NOT RUN HERE -- needs a GPU. This needs `pip install laya transformers datasets safetensors
huggingface_hub` (see requirements.txt) and network access to Hugging Face Hub for the base
checkpoints, both of which this authoring environment intentionally does not exercise. Verify on
the GPU machine: sanity-check a handful of built examples' `state`/`gold` (the --stage prepare
output), confirm `laya.common.build_sequence` accepts them without a length mismatch (the "markers
!= k" skip in build_training_item silently drops malformed items -- if a large fraction get
dropped, that's a bug worth investigating before spending GPU time on the rest).

Two checkpoints are fine-tuned separately, per the challenge's Router usage:
  - "english":       only on pairs where BOTH sides are Latin-script (its natural domain).
  - "multilingual":  on the FULL pair set (every script), so it also covers India's Devanagari/
                      Tamil business names -- the Router then sends non-Latin text there.
Train/calibration entities are drawn only from the S1 entities NOT held out for GBDT/ensemble
validation (src/common.py:stratified_split_by_s1's train_ids), and the calibration slice is a
further split of THAT train set (never the val set, and never a benchmark's own holdout) --
satisfying "held-out slice of your own train split" for the calibration-fitting step.

Run (from code/business_entity_resolution/), two stages:
    # 1. Build the jsonl datasets (fast, CPU-only, needs no GPU/laya import):
    python -m src.laya_finetune --stage prepare

    # 2. Fine-tune each role (GPU; single process or `torchrun --nproc_per_node=2` for 2xT4):
    python -m src.laya_finetune --stage train --role english
    torchrun --standalone --nproc_per_node=2 -m src.laya_finetune --stage train --role multilingual

Writes:
    data_processed/laya_{role}_{train,calib}.jsonl   (--stage prepare)
    models/laya_{role}/                               (--stage train: fine-tuned checkpoint dir)
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List

from tqdm.auto import tqdm

from src import common
from src.features import load_entity_lookup

ROLES = ("english", "multilingual")
BASE_CHECKPOINTS = {"english": "convaiinnovations/laya", "multilingual": "convaiinnovations/laya-multilingual"}

SAME_ENTITY_INSTRUCTIONS = (
    "record_a and record_b each describe one business as \"name | address | country\", taken "
    "from two different, independently-collected data sources. Decide whether they refer to the "
    "SAME real-world business."
)
SAME_ENTITY_CRITERIA = {
    "true": (
        "The same business despite surface noise between the two sources: legal-suffix "
        "abbreviations (Corp/Corporation, Ltd/Limited, Pvt/Private, Inc, or French SARL/SAS/SASU/"
        "SA), typos, transliteration differences, word-order changes, punctuation differences, a "
        "DBA/trade name standing in for the same legal entity, a missing address component (no "
        "PIN/postal code, no state), or a landmark-based address reference (e.g. \"near SBI ATM\") "
        "for what is otherwise the same location."
    ),
    "false": (
        "Different businesses: distinct legal entities even when the names look similar (e.g. "
        "two different branches of the same chain at different addresses), a coincidental name "
        "match, or an address that is clearly a different location."
    ),
}


def record_text(rec: dict) -> str:
    return f"{rec.get('business_name', '')} | {rec.get('business_address', '')} | {rec.get('country', '')}"


def make_example(rec_a: dict, rec_b: dict, label_true: bool) -> dict:
    return {
        "state": {"record_a": record_text(rec_a), "record_b": record_text(rec_b)},
        "questions": {"same_entity": {"type": "noul", "instructions": SAME_ENTITY_INSTRUCTIONS,
                                       "criteria": SAME_ENTITY_CRITERIA}},
        # Hard 0/1 targets: our ground truth has no soft/teacher distribution to draw
        # probabilities from (unlike the notebook's RLCD-teacher-labeled benchmark), so
        # "probabilities" is a one-hot vector rather than a genuinely soft label. proper_reward
        # and the soft-CE term still work correctly on one-hot targets; it just means the
        # RLCD objective has less to teach about calibrated *uncertainty* than a soft-labeled
        # dataset would, only about calibrated *confidence* via the post-hoc temperature fit.
        "gold": {"same_entity": {
            "label": "true" if label_true else "false",
            "probabilities": {"true": 1.0 if label_true else 0.0, "false": 0.0 if label_true else 1.0},
        }},
        "meta": {"script_a": rec_a.get("name_script", "other"), "script_b": rec_b.get("name_script", "other")},
    }


def build_examples(candidate_map: Dict[str, set], lookup: Dict[str, dict], truth: Dict[str, set],
                    s1_ids: set, max_negatives_per_entity: int, seed: int) -> List[dict]:
    """Every positive pair, plus up to `max_negatives_per_entity` randomly-chosen hard negatives
    per S1 entity. Using all ~K negatives per entity on the full training set produces millions of
    examples -- more than a Kaggle session can tokenize in RAM, let alone train on in time -- and
    heavily over-weights "false" anyway."""
    rng = random.Random(seed)
    examples = []
    for s1 in tqdm(sorted(s1_ids), desc="laya examples", unit="entity", mininterval=2.0):
        s1_rec = lookup.get(s1)
        if s1_rec is None:
            continue
        true_ids = truth.get(s1, set())
        cands = sorted(c for c in candidate_map.get(s1, ()) if c in lookup)
        positives = [c for c in cands if c in true_ids]
        negatives = [c for c in cands if c not in true_ids]
        if len(negatives) > max_negatives_per_entity:
            negatives = rng.sample(negatives, max_negatives_per_entity)
        for cand, label_true in [(c, True) for c in positives] + [(c, False) for c in negatives]:
            cand_rec = lookup[cand]
            examples.append(make_example(s1_rec, cand_rec, label_true))
            examples.append(make_example(cand_rec, s1_rec, label_true))  # swapped-order copy
    return examples


def route_by_script(examples: List[dict]) -> Dict[str, List[dict]]:
    english = [e for e in examples if e["meta"]["script_a"] == "latin" and e["meta"]["script_b"] == "latin"]
    return {"english": english, "multilingual": list(examples)}  # multilingual sees everything


def write_jsonl(examples: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps({k: v for k, v in ex.items() if k != "meta"}, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ------------------------------------------------------------------------------------- prepare

def stage_prepare(repo_root: Path, candidates_path, val_frac: float, calib_frac: float, seed: int,
                  max_negatives_per_entity: int):
    ground_truth = common.load_split_sources(repo_root, "train")["ground_truth"]
    truth = common.ground_truth_map(ground_truth)
    train_ids, val_ids = common.stratified_split_by_s1(ground_truth, val_frac=val_frac, seed=seed)
    print(f"[laya_finetune/prepare] {len(train_ids)} train-pool / {len(val_ids)} held-out-val S1 "
          f"entities (val entities are NEVER used for fine-tuning or calibration)")

    # Calibration slice carved out of the TRAIN pool only, with a different seed offset so it's
    # not the same split as train_gbdt's/ensemble's val split.
    train_gt = ground_truth[ground_truth["source1_entity_id"].isin(train_ids)]
    finetune_ids, calib_ids = common.stratified_split_by_s1(train_gt, val_frac=calib_frac, seed=seed + 1)
    print(f"[laya_finetune/prepare] of the train pool: {len(finetune_ids)} fine-tune / "
          f"{len(calib_ids)} calibration S1 entities")

    candidates_path = candidates_path or (common.data_processed_dir(repo_root) / "candidate_pairs_train.tsv")
    candidate_map = common.read_id_list_tsv(candidates_path)
    lookup = load_entity_lookup(repo_root, "train")

    finetune_examples = build_examples(candidate_map, lookup, truth, finetune_ids, max_negatives_per_entity, seed)
    calib_examples = build_examples(candidate_map, lookup, truth, calib_ids, max_negatives_per_entity, seed + 1)
    print(f"[laya_finetune/prepare] built {len(finetune_examples)} fine-tune / "
          f"{len(calib_examples)} calibration examples (each pair contributes 2: original + "
          f"swapped-order copy)")

    finetune_by_role = route_by_script(finetune_examples)
    calib_by_role = route_by_script(calib_examples)

    out_dir = common.data_processed_dir(repo_root)
    for role in ROLES:
        train_path = out_dir / f"laya_{role}_train.jsonl"
        calib_path = out_dir / f"laya_{role}_calib.jsonl"
        write_jsonl(finetune_by_role[role], train_path)
        write_jsonl(calib_by_role[role], calib_path)
        print(f"[laya_finetune/prepare] {role}: {len(finetune_by_role[role])} train / "
              f"{len(calib_by_role[role])} calib -> {train_path.name}, {calib_path.name}")


# --------------------------------------------------------------------------------------- train
#
# Mirrors laya's own fine-tuning notebook (see module docstring) closely enough to be a drop-in
# reproduction of its recipe, generalized to (a) read our jsonl datasets instead of the
# LocalLLaMA/typed-decisions HF dataset, (b) run as a plain script instead of notebook cells, and
# (c) work single-process (Colab, 1 GPU) as well as under `torchrun` DDP (Kaggle 2xT4) -- the
# notebook is DDP-only. Everything below this point needs torch/transformers/laya installed and a
# GPU; none of it is imported or run by --stage prepare.

def _build_training_item(tok, cfg, example: dict):
    from laya.common import build_sequence, render_options, QTYPES

    state = example["state"]
    q = example["questions"]["same_entity"]
    gold = example["gold"]["same_entity"]["probabilities"]
    target = [gold["false"], gold["true"]]
    label = int(target[1] > target[0])
    qdict = {"t": "noul", "ins": q["instructions"], "crit": q.get("criteria", {})}
    k = len(render_options(qdict))
    seq, markers = build_sequence(tok, state, qdict, cfg["max_len"], cfg["head_max_len"])
    if len(markers) != k:
        return None
    return {"ids": seq, "markers": markers, "qtype": QTYPES["noul"], "target": target, "label": label}


def _collate_train_batch(items, pad_id):
    import torch

    n, L = len(items), max(len(it["ids"]) for it in items)
    kmax = max(len(it["markers"]) for it in items)
    ids = torch.full((n, L), pad_id, dtype=torch.long)
    att = torch.zeros((n, L), dtype=torch.long)
    mpos = torch.zeros((n, kmax), dtype=torch.long)
    mmask = torch.zeros((n, kmax), dtype=torch.bool)
    target = torch.zeros((n, kmax), dtype=torch.float32)
    for i, it in enumerate(items):
        ids[i, :len(it["ids"])] = torch.tensor(it["ids"])
        att[i, :len(it["ids"])] = 1
        k = len(it["markers"])
        mpos[i, :k] = torch.tensor(it["markers"])
        mmask[i, :k] = True
        target[i, :len(it["target"])] = torch.tensor(it["target"], dtype=torch.float32)
    return {
        "input_ids": ids, "attention_mask": att, "marker_pos": mpos, "marker_mask": mmask,
        "target": target,
        "qtype": torch.tensor([it["qtype"] for it in items]),
        "label": torch.tensor([it["label"] for it in items]),
    }


def _fit_one_temp(sel):
    import torch

    if len(sel) < 10:
        return 1.0
    kmax = max(len(z) for z, _ in sel)
    Z = torch.full((len(sel), kmax), -1e4)
    T = torch.zeros((len(sel), kmax))
    for i, (z, t) in enumerate(sel):
        Z[i, :len(z)] = torch.tensor(z)
        T[i, :len(t)] = torch.tensor(t, dtype=torch.float32)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = -(T * torch.log_softmax(Z / log_t.exp(), -1)).sum(-1).mean()
        loss.backward()
        return loss

    opt.step(closure)
    return float(torch.clamp(log_t.exp(), 0.1, 10.0).item())


def stage_train(repo_root: Path, role: str, epochs: int, micro_batch: int, grad_accum: int,
                 lr_encoder: float, lr_head: float, calib_max: int, seed: int):
    import time
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from safetensors.torch import load_file, save_file
    from transformers import AutoTokenizer
    from huggingface_hub import snapshot_download
    from laya.agent import _fix_tokenizer_config
    from laya.common import build_model, proper_reward, QTYPES

    ddp_mode = "WORLD_SIZE" in os.environ and int(os.environ.get("WORLD_SIZE", "1")) > 1
    if ddp_mode:
        dist.init_process_group("nccl")
        rank, world_size = dist.get_rank(), dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        rank, world_size, local_rank = 0, 1, 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type != "cuda" and rank == 0:
            print("[laya_finetune/train] WARNING: no CUDA device visible -- this will be very "
                  "slow or fail outright. This script must be run on the GPU machine, not here.")

    model_repo = BASE_CHECKPOINTS[role]
    model_dir = snapshot_download(model_repo)
    _fix_tokenizer_config(model_dir)
    tok = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
    with open(os.path.join(model_dir, "rl_agent_config.json")) as f:
        cfg = json.load(f)
    cfg["gradient_checkpointing"] = True
    cfg["max_tokens_per_batch"] = 4096

    data_dir = common.data_processed_dir(repo_root)
    train_examples = read_jsonl(data_dir / f"laya_{role}_train.jsonl")
    calib_examples = read_jsonl(data_dir / f"laya_{role}_calib.jsonl")
    if rank == 0:
        print(f"[laya_finetune/train] role={role} model={model_repo} device={device} "
              f"ddp={ddp_mode} world_size={world_size}")
        print(f"[laya_finetune/train] {len(train_examples)} train / {len(calib_examples)} "
              f"calibration examples (calibration held out of training, never seen by any rank)")

    quiet = rank != 0
    train_items = [it for ex in tqdm(train_examples, desc="tokenize train", disable=quiet, mininterval=2.0)
                   if (it := _build_training_item(tok, cfg, ex)) is not None]
    calib_items = [it for ex in tqdm(calib_examples, desc="tokenize calib", disable=quiet, mininterval=2.0)
                   if (it := _build_training_item(tok, cfg, ex)) is not None]
    dropped = (len(train_examples) - len(train_items)) + (len(calib_examples) - len(calib_items))
    if dropped and rank == 0:
        print(f"[laya_finetune/train] WARNING: {dropped} example(s) dropped by build_sequence "
              f"(marker/option count mismatch) -- inspect on the GPU machine if this is a large "
              f"fraction of the dataset.")

    if len(calib_items) > calib_max:
        random.Random(seed).shuffle(calib_items)
        calib_items = calib_items[:calib_max]

    my_items = train_items[rank::world_size]
    model = build_model(cfg, encoder_dir=os.path.join(model_dir, "encoder"))
    weights = load_file(os.path.join(model_dir, "model.safetensors"))
    model.load_state_dict(weights, strict=True)
    model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.head_checkpointing = True
    model.to(device)
    model.train()

    ddp_model = DDP(model, device_ids=[local_rank], find_unused_parameters=True) if ddp_mode else model

    sigma_start, sigma_end = 0.4, 0.1
    enc_params = [p for n, p in ddp_model.named_parameters() if "encoder." in n]
    head_params = [p for n, p in ddp_model.named_parameters() if "encoder." not in n]
    optimizer = torch.optim.AdamW(
        [{"params": enc_params, "lr": lr_encoder}, {"params": head_params, "lr": lr_head}],
        weight_decay=0.01,
    )
    total_updates = max(1, (len(my_items) // (micro_batch * grad_accum)) * epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_updates, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    output_dir = common.models_dir(repo_root) / f"laya_{role}"
    if rank == 0:
        print(f"[laya_finetune/train] {len(train_items)} usable train items "
              f"({len(calib_items)} held out for calibration) | {len(my_items)} on this rank | "
              f"{epochs} epochs -> {output_dir}")
    t0 = time.time()

    for epoch in range(epochs):
        random.seed(seed + epoch + rank)
        random.shuffle(my_items)
        epoch_loss, n_batches, accum_step = 0.0, 0, 0
        optimizer.zero_grad(set_to_none=True)
        sigma = sigma_start + (sigma_end - sigma_start) * (epoch / max(1, epochs - 1))

        for b_idx in tqdm(range(0, len(my_items), micro_batch), desc=f"epoch {epoch + 1}/{epochs}",
                          unit="batch", disable=quiet, mininterval=5.0):
            chunk = my_items[b_idx:b_idx + micro_batch]
            if not chunk:
                continue
            batch = _collate_train_batch(chunk, tok.pad_token_id)
            amp_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
            with torch.autocast(device.type, dtype=amp_dtype, enabled=device.type in ("cuda", "cpu")):
                logits, act = ddp_model(
                    batch["input_ids"].to(device), batch["attention_mask"].to(device),
                    batch["marker_pos"].to(device), batch["marker_mask"].to(device),
                    batch["qtype"].to(device),
                )
            logits = logits.float()
            mask = batch["marker_mask"].to(device)
            k = mask.sum(-1, keepdim=True).float()
            target = batch["target"].to(device)

            group_size = 4
            eps = torch.randn((group_size,) + logits.shape, device=device) * sigma * mask
            eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
            z = logits.detach().unsqueeze(0) + eps
            q = torch.softmax(z.masked_fill(~mask, -1e4), -1)
            with torch.no_grad():
                r = proper_reward(q, target.unsqueeze(0), batch["qtype"].to(device), mask, w_sph=0.75, w_rps=1.0)
                adv = r - r.mean(0, keepdim=True)
                adv = adv / (adv.std() + 1e-6)

            logp = -(((z - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma ** 2)
            loss_rl = -(adv * logp).mean()
            loss_ce = -(target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1).mean()
            loss = (loss_rl + 1.0 * loss_ce) / grad_accum + 0.0 * act.sum()

            scaler.scale(loss).backward()
            accum_step += 1
            if accum_step % grad_accum == 0 or (b_idx + micro_batch) >= len(my_items):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            epoch_loss += loss.item() * grad_accum
            n_batches += 1
            if rank == 0 and n_batches % 50 == 0:
                print(f"  epoch {epoch + 1}/{epochs} step {n_batches} loss={loss.item() * grad_accum:.4f} "
                      f"reward={r.mean().item():.3f} lr={scheduler.get_last_lr()[0]:.2e}")

        if rank == 0:
            print(f"=== epoch {epoch + 1}/{epochs} done in {time.time() - t0:.1f}s | "
                  f"avg loss {epoch_loss / max(1, n_batches):.4f} ===")
        if ddp_mode:
            dist.barrier()
        if rank == 0:
            ckpt_dir = output_dir / "checkpoint_latest"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            ckpt_sd = {k: v.half().contiguous().cpu() for k, v in model.state_dict().items()}
            save_file(ckpt_sd, str(ckpt_dir / "model.safetensors"))
            model.encoder.config.save_pretrained(str(ckpt_dir / "encoder"))
            tok.save_pretrained(str(ckpt_dir / "tokenizer"))
            with open(ckpt_dir / "checkpoint_meta.json", "w") as f:
                json.dump({"epoch": epoch + 1, "total_epochs": epochs,
                           "avg_loss": epoch_loss / max(1, n_batches)}, f, indent=2)

    if ddp_mode:
        dist.barrier()

    if rank == 0:
        print("\n[laya_finetune/train] fitting post-training calibration temperature on the "
              "held-out calibration slice (never trained on, above)...")
        del optimizer, scaler, scheduler
        if device.type == "cuda":
            torch.cuda.empty_cache()
        model.eval()
        calib_preds = []
        with torch.no_grad():
            for c_idx in range(0, len(calib_items), 16):
                c_chunk = calib_items[c_idx:c_idx + 16]
                if not c_chunk:
                    continue
                cb = _collate_train_batch(c_chunk, tok.pad_token_id)
                amp_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
                with torch.autocast(device.type, dtype=amp_dtype, enabled=device.type in ("cuda", "cpu")):
                    l_sub, _ = model(
                        cb["input_ids"].to(device), cb["attention_mask"].to(device),
                        cb["marker_pos"].to(device), cb["marker_mask"].to(device),
                        cb["qtype"].to(device),
                    )
                l_np = l_sub.float().cpu().numpy()
                for r_idx, it in enumerate(c_chunk):
                    k = len(it["markers"])
                    calib_preds.append((it["qtype"], l_np[r_idx, :k], it["target"]))

        fitted_temps = [1.2, 1.2, 1.2]  # (choice, score, noul) -- QTYPES order; only "noul" is used here
        try:
            sel = [(z, t) for qtype, z, t in calib_preds if qtype == QTYPES["noul"]]
            if sel:
                fitted_temps[QTYPES["noul"]] = _fit_one_temp(sel)
            print(f"[laya_finetune/train] fitted noul temperature: {fitted_temps[QTYPES['noul']]:.3f} "
                  f"(on {len(sel)} held-out calibration items)")
        except Exception as exc:
            print(f"[laya_finetune/train] temperature fitting failed, keeping default 1.2: {exc}")

        output_dir.mkdir(parents=True, exist_ok=True)
        sd = {k: v.half().contiguous().cpu() for k, v in model.state_dict().items()}
        save_file(sd, str(output_dir / "model.safetensors"))
        model.encoder.config.save_pretrained(str(output_dir / "encoder"))
        tok.save_pretrained(str(output_dir / "tokenizer"))
        cfg["fine_tuned"] = True
        cfg["model_name"] = f"laya-entity-resolution-{role}"
        cfg["temperature"] = fitted_temps
        cfg.pop("temperature_by_options", None)
        with open(output_dir / "rl_agent_config.json", "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"[laya_finetune/train] saved fine-tuned {role} checkpoint to {output_dir}")

    if ddp_mode:
        dist.destroy_process_group()


# ---------------------------------------------------------------------------------- Router glue

def build_routers(repo_root: Path) -> list:
    """One Router per visible GPU (both T4s on Kaggle's T4x2), each pinned to its own device, so
    ensemble.score_with_laya can run them in parallel. Falls back to a single CPU router. Prints
    where each checkpoint actually landed: laya silently moves a checkpoint to CPU if it doesn't
    fit in GPU memory, which would otherwise show up only as "CPU busy, GPU idle"."""
    import torch

    n_gpus = torch.cuda.device_count()
    devices = [f"cuda:{i}" for i in range(n_gpus)] or ["cpu"]
    routers = []
    for device in devices:
        router = build_router(repo_root, device=device)
        for name, agent in router._agents.items():
            print(f"[laya] router for {device}: checkpoint '{name}' loaded on {agent.device}")
            if device != "cpu" and agent.device.type != "cuda":
                print(f"[laya] WARNING: '{name}' fell back to CPU on {device} (likely GPU out of memory) "
                      f"-- scoring will be very slow. Free GPU memory (restart the kernel after fine-tuning) and retry.")
        routers.append(router)
    return routers


def build_router(repo_root: Path, device: str = None):
    """Assembles both fine-tuned checkpoints into a laya.Router, exactly the mechanism the
    challenge's fine-tuning step asks for: 'Use the Router (English vs multilingual checkpoint)
    so India records with Devanagari/Tamil business names route to the multilingual checkpoint.'
    `Router(models=...)` accepts local directories directly (laya/router.py: `Agent(repo, ...)`
    where `repo` is whatever string was passed -- a local dir short-circuits the Hub download).
    Falls back to the base (non-fine-tuned) checkpoints with a warning if a fine-tuned dir is
    missing, so ensemble.py/predict.py can still run end-to-end before fine-tuning is done."""
    from laya import Router

    models = {}
    for role in ROLES:
        local_dir = common.models_dir(repo_root) / f"laya_{role}"
        if local_dir.exists():
            models[role] = str(local_dir)
        else:
            print(f"[laya_finetune] WARNING: no fine-tuned '{role}' checkpoint at {local_dir}; "
                  f"falling back to the base '{BASE_CHECKPOINTS[role]}' checkpoint (zero-shot).")
            models[role] = BASE_CHECKPOINTS[role]
    return Router(models=models, device=device, preload=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    common.add_repo_root_arg(parser)
    parser.add_argument("--stage", choices=["prepare", "train"], required=True)
    # prepare
    parser.add_argument("--candidates", type=str, default=None,
                         help="[prepare] Path to the train candidate_pairs.tsv. "
                              "Default: data_processed/candidate_pairs_train.tsv")
    parser.add_argument("--val-frac", type=float, default=0.2,
                         help="[prepare] Must match train_gbdt.py's --val-frac so 'val' means "
                              "the same held-out entities everywhere in the pipeline.")
    parser.add_argument("--calib-frac", type=float, default=0.15,
                         help="[prepare] Fraction of the TRAIN pool (not val) carved out for "
                              "Laya's post-training calibration fit.")
    parser.add_argument("--max-negatives-per-entity", type=int, default=3,
                         help="[prepare] Hard negatives kept per S1 entity (all positives are kept).")
    # train
    parser.add_argument("--role", choices=list(ROLES), default=None,
                         help="[train] Which checkpoint to fine-tune.")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--micro-batch", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr-encoder", type=float, default=2.5e-5)
    parser.add_argument("--lr-head", type=float, default=1.0e-4)
    parser.add_argument("--calib-max", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.stage == "prepare":
        stage_prepare(args.repo_root, args.candidates, args.val_frac, args.calib_frac, args.seed,
                      args.max_negatives_per_entity)
    else:
        if not args.role:
            raise SystemExit("--stage train requires --role {english,multilingual}")
        stage_train(args.repo_root, args.role, args.epochs, args.micro_batch, args.grad_accum,
                    args.lr_encoder, args.lr_head, args.calib_max, args.seed)


if __name__ == "__main__":
    main()

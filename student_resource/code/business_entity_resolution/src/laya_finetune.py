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
import math
import os
import random
import shutil
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps({k: v for k, v in ex.items() if k != "meta"}, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def read_jsonl(path: Path) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ------------------------------------------------------------------------------------- prepare

def _cap(ids, limit: int, seed: int) -> set:
    ids = sorted(ids)
    if limit and len(ids) > limit:
        ids = random.Random(seed).sample(ids, limit)
    return set(ids)


def stage_prepare(repo_root: Path, candidates_path, val_frac: float, calib_frac: float, seed: int,
                  max_negatives_per_entity: int, max_finetune_entities: int = 6000,
                  max_calib_entities: int = 1000):
    """max_finetune_entities: the upstream recipe fine-tunes on ~6k items in minutes on 2xT4;
    6000 entities x (positives + 3 negatives) x 2 orderings is ~60-80k examples, roughly an hour
    per role. Raise it if you have GPU time to spare."""
    candidates_path = candidates_path or (common.data_processed_dir(repo_root) / "candidate_pairs_train.tsv")
    ground_truth = common.sampled_ground_truth(repo_root, candidates_path)
    truth = common.ground_truth_map(ground_truth)
    train_ids, val_ids = common.stratified_split_by_s1(ground_truth, val_frac=val_frac, seed=seed)
    print(f"[laya_finetune/prepare] {len(train_ids)} train-pool / {len(val_ids)} held-out-val S1 "
          f"entities (val entities are NEVER used for fine-tuning or calibration)")

    # Calibration slice carved out of the TRAIN pool only, with a different seed offset so it's
    # not the same split as train_gbdt's/ensemble's val split.
    train_gt = ground_truth[ground_truth["source1_entity_id"].isin(train_ids)]
    finetune_ids, calib_ids = common.stratified_split_by_s1(train_gt, val_frac=calib_frac, seed=seed + 1)
    finetune_ids = _cap(finetune_ids, max_finetune_entities, seed)
    calib_ids = _cap(calib_ids, max_calib_entities, seed + 1)
    print(f"[laya_finetune/prepare] using {len(finetune_ids)} fine-tune / {len(calib_ids)} calibration "
          f"S1 entities (capped)")

    candidate_map = common.read_id_list_tsv(candidates_path)
    chosen = finetune_ids | calib_ids
    needed_ids = set(chosen)
    for s1 in chosen:
        needed_ids.update(candidate_map.get(s1, ()))
    lookup = load_entity_lookup(repo_root, "train", needed_ids)

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


# ------------------------------------------------------------------------- checkpoints / resume
#
# A role is DONE when models/laya_<role>/rl_agent_config.json exists (written last, after the
# calibrated weights). While training, models/laya_<role>/checkpoint_latest/ holds the newest
# weights plus checkpoint_meta.json = {"epoch": completed epochs, "batches_done_in_epoch": batches
# already trained in the next epoch, ...}. Checkpoints are written to a .tmp dir and swapped in,
# so a kill mid-save never leaves a half-written checkpoint as the only copy.

FINAL_MARKER = "rl_agent_config.json"


def role_output_dir(repo_root: Path, role: str) -> Path:
    return common.models_dir(repo_root) / f"laya_{role}"


def role_is_done(repo_root: Path, role: str) -> bool:
    return (role_output_dir(repo_root, role) / FINAL_MARKER).exists()


def find_checkpoint(output_dir: Path) -> Optional[Path]:
    for name in ("checkpoint_latest", "checkpoint_latest.old"):
        d = Path(output_dir) / name
        if (d / "checkpoint_meta.json").exists() and (d / "model.safetensors").exists():
            return d
    return None


def read_resume_point(ckpt_dir: Path, epochs: int) -> Tuple[int, int, dict]:
    """(start_epoch, batches_done_in_that_epoch, meta). Checkpoints written before mid-epoch
    saving existed only carry "epoch", so they resume at the start of the next epoch."""
    meta = json.loads((Path(ckpt_dir) / "checkpoint_meta.json").read_text())
    start_epoch = int(meta.get("epoch", 0))
    batches_done = int(meta.get("batches_done_in_epoch", 0))
    if start_epoch >= epochs:
        return epochs, 0, meta
    return start_epoch, batches_done, meta


def checkpoint_status(repo_root: Path, role: str, epochs: int) -> str:
    """One-line, human-readable fine-tuning status for the notebook."""
    if role_is_done(repo_root, role):
        return "done (fine-tuned + calibrated)"
    ckpt = find_checkpoint(role_output_dir(repo_root, role))
    if ckpt is None:
        return "not started"
    start_epoch, batches_done, _ = read_resume_point(ckpt, epochs)
    if start_epoch >= epochs:
        return f"all {epochs} epochs trained; calibration + final save still to do"
    return (f"resumable: {start_epoch}/{epochs} epochs complete"
            + (f" + {batches_done} batches into epoch {start_epoch + 1}" if batches_done else ""))


def _save_checkpoint(model, tok, output_dir: Path, meta: dict) -> None:
    from safetensors.torch import save_file

    output_dir = Path(output_dir)
    tmp, final, old = (output_dir / "checkpoint_latest.tmp", output_dir / "checkpoint_latest",
                       output_dir / "checkpoint_latest.old")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    # Only floating-point tensors go to fp16; integer buffers keep their dtype, so the checkpoint
    # round-trips exactly into the training model on resume.
    sd = {k: (v.half() if v.is_floating_point() else v).contiguous().cpu() for k, v in model.state_dict().items()}
    save_file(sd, str(tmp / "model.safetensors"))
    model.encoder.config.save_pretrained(str(tmp / "encoder"))
    tok.save_pretrained(str(tmp / "tokenizer"))
    (tmp / "checkpoint_meta.json").write_text(json.dumps(meta, indent=2))
    shutil.rmtree(old, ignore_errors=True)
    if final.exists():
        os.replace(final, old)
    os.replace(tmp, final)
    shutil.rmtree(old, ignore_errors=True)


def stage_train(repo_root: Path, role: str, epochs: int, micro_batch: int, grad_accum: int,
                 lr_encoder: float, lr_head: float, calib_max: int, seed: int, resume: bool = True,
                 checkpoint_every_updates: int = 200, max_hours: Optional[float] = None) -> str:
    """Returns "done", "paused" (time budget reached; re-run to continue) or "skipped" (already
    fine-tuned). resume=True continues from models/laya_<role>/checkpoint_latest when present,
    including mid-epoch; a checkpoint is saved every `checkpoint_every_updates` optimizer updates
    and at every epoch end. max_hours stops cleanly (after saving) once exceeded."""
    import time
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from safetensors.torch import load_file, save_file
    from transformers import AutoTokenizer
    from huggingface_hub import snapshot_download
    from laya.agent import _fix_tokenizer_config
    from laya.common import build_model, proper_reward, QTYPES

    output_dir = role_output_dir(repo_root, role)
    if resume and role_is_done(repo_root, role):
        if int(os.environ.get("RANK", "0")) == 0:
            print(f"[laya_finetune/train] {role}: already fine-tuned ({output_dir / FINAL_MARKER} "
                  f"exists) -- skipping. Pass resume=False / --no-resume to retrain from scratch.")
        return "skipped"

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
    prepared_by = "laya_finetune.py --stage prepare (notebook Section 5a)"
    train_examples = read_jsonl(common.require(data_dir / f"laya_{role}_train.jsonl", prepared_by))
    calib_examples = read_jsonl(common.require(data_dir / f"laya_{role}_calib.jsonl", prepared_by))
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

    # Equal-length shards: under DDP every rank must run the same number of batches, or the
    # longer rank blocks forever in its last all-reduce.
    n_per_rank = len(train_items) // world_size
    my_items = train_items[rank::world_size][:n_per_rank]

    start_epoch, batches_done = 0, 0
    weights_path = os.path.join(model_dir, "model.safetensors")
    ckpt = find_checkpoint(output_dir) if resume else None
    if ckpt is not None:
        start_epoch, batches_done, meta = read_resume_point(ckpt, epochs)
        if meta.get("n_train_items") not in (None, len(train_items)):
            if rank == 0:
                print(f"[laya_finetune/train] WARNING: {ckpt} was trained on {meta['n_train_items']} items "
                      f"but the current dataset has {len(train_items)} (Section 5a was re-run with "
                      f"different settings) -- ignoring it and starting from the base checkpoint.")
            start_epoch, batches_done = 0, 0
        else:
            weights_path = str(ckpt / "model.safetensors")
            same_layout = all(meta.get(k) in (None, v) for k, v in
                              (("micro_batch", micro_batch), ("grad_accum", grad_accum), ("world_size", world_size)))
            if batches_done and not same_layout:
                # A mid-epoch position is only meaningful with the same batching and GPU count.
                if rank == 0:
                    print("[laya_finetune/train] batch size / grad-accum / GPU count changed since the "
                          "checkpoint -- keeping its weights but restarting that epoch from its first batch.")
                batches_done = 0
            if rank == 0:
                print(f"[laya_finetune/train] RESUMING {role} from {ckpt}: {start_epoch}/{epochs} epochs "
                      f"complete" + (f" + {batches_done} batches into epoch {start_epoch + 1}" if batches_done else ""))

    model = build_model(cfg, encoder_dir=os.path.join(model_dir, "encoder"))
    weights = load_file(weights_path)
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
    n_batches_per_epoch = math.ceil(len(my_items) / micro_batch)
    updates_per_epoch = math.ceil(n_batches_per_epoch / grad_accum)
    total_updates = max(1, updates_per_epoch * epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_updates, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    # Resume: fast-forward the LR schedule to where the interrupted run was, and replay the
    # per-epoch in-place shuffles so the resumed epoch sees exactly the same batch order (the
    # already-trained batches are then skipped). AdamW's moment estimates aren't checkpointed, so
    # they restart from zero -- a brief, harmless re-warm-up.
    updates_done = start_epoch * updates_per_epoch + batches_done // grad_accum
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # "scheduler.step() before optimizer.step()"
        for _ in range(updates_done):
            scheduler.step()
    for e in range(start_epoch):
        random.seed(seed + e + rank)
        random.shuffle(my_items)

    if rank == 0:
        print(f"[laya_finetune/train] {len(train_items)} usable train items "
              f"({len(calib_items)} held out for calibration) | {len(my_items)} on this rank | "
              f"{epochs} epochs -> {output_dir}")
        if max_hours:
            print(f"[laya_finetune/train] time budget {max_hours:.1f}h: stops cleanly (after saving a "
                  f"checkpoint) once exceeded; re-run to continue.")
    t0 = time.time()
    max_seconds = max_hours * 3600 if max_hours else None

    def _should_stop() -> bool:
        # Called at the same update count on every rank, so the all-reduce lines up; any rank
        # over budget stops all of them together.
        local = max_seconds is not None and (time.time() - t0) > max_seconds
        if ddp_mode:
            flag = torch.tensor([1 if local else 0], device=device)
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            return bool(flag.item())
        return local

    def _ckpt_meta(completed_epochs: int, batches_in_next: int) -> dict:
        return {"epoch": completed_epochs, "batches_done_in_epoch": batches_in_next,
                "total_epochs": epochs, "updates_done": updates_done, "n_train_items": len(train_items),
                "micro_batch": micro_batch, "grad_accum": grad_accum, "world_size": world_size}

    paused = False
    for epoch in range(start_epoch, epochs):
        random.seed(seed + epoch + rank)
        random.shuffle(my_items)
        skip = batches_done if epoch == start_epoch else 0
        epoch_loss, n_batches, accum_step = 0.0, 0, skip
        optimizer.zero_grad(set_to_none=True)
        sigma = sigma_start + (sigma_end - sigma_start) * (epoch / max(1, epochs - 1))

        for b_idx in tqdm(range(skip * micro_batch, len(my_items), micro_batch), desc=f"epoch {epoch + 1}/{epochs}",
                          unit="batch", disable=quiet, mininterval=5.0, initial=skip, total=n_batches_per_epoch):
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
            last_batch = (b_idx + micro_batch) >= len(my_items)
            stepped = False
            if accum_step % grad_accum == 0 or last_batch:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                updates_done += 1
                stepped = True

            epoch_loss += loss.item() * grad_accum
            n_batches += 1
            if rank == 0 and n_batches % 50 == 0:
                print(f"  epoch {epoch + 1}/{epochs} step {n_batches} loss={loss.item() * grad_accum:.4f} "
                      f"reward={r.mean().item():.3f} lr={scheduler.get_last_lr()[0]:.2e}")

            # Mid-epoch checkpoint, only right after an optimizer update (no half-accumulated
            # gradients are lost). The epoch-end checkpoint below covers the last batch.
            if stepped and not last_batch and checkpoint_every_updates and updates_done % checkpoint_every_updates == 0:
                if rank == 0:
                    _save_checkpoint(model, tok, output_dir, _ckpt_meta(epoch, b_idx // micro_batch + 1))
                if _should_stop():
                    paused = True
                    break

        if paused:
            break
        if rank == 0:
            print(f"=== epoch {epoch + 1}/{epochs} done in {time.time() - t0:.1f}s | "
                  f"avg loss {epoch_loss / max(1, n_batches):.4f} ===")
        if ddp_mode:
            dist.barrier()
        if rank == 0:
            _save_checkpoint(model, tok, output_dir, {**_ckpt_meta(epoch + 1, 0),
                                                      "avg_loss": epoch_loss / max(1, n_batches)})
        if epoch + 1 < epochs and _should_stop():
            paused = True
            break

    if paused:
        if ddp_mode:
            dist.barrier()
            dist.destroy_process_group()
        if rank == 0:
            print(f"\n[laya_finetune/train] PAUSED {role}: time budget of {max_hours}h reached after "
                  f"{(time.time() - t0) / 3600:.2f}h. Progress is saved in {output_dir / 'checkpoint_latest'} "
                  f"-- re-run the same step to continue from there.")
        return "paused"

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
    return "done"


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
        local_dir = role_output_dir(repo_root, role)
        # The final marker, not just the directory: models/laya_<role>/ also exists while a
        # fine-tune is only partway done (it holds checkpoint_latest/), and that isn't loadable.
        if role_is_done(repo_root, role):
            models[role] = str(local_dir)
        else:
            print(f"[laya_finetune] WARNING: no finished fine-tuned '{role}' checkpoint at {local_dir} "
                  f"({checkpoint_status(repo_root, role, epochs=4)}); falling back to the base "
                  f"'{BASE_CHECKPOINTS[role]}' checkpoint (zero-shot).")
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
    parser.add_argument("--max-finetune-entities", type=int, default=6000,
                         help="[prepare] S1 entities used for fine-tuning examples (0 = all).")
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
    parser.add_argument("--no-resume", action="store_true",
                        help="[train] Ignore models/laya_<role>/checkpoint_latest and any finished "
                             "checkpoint; retrain from the base model.")
    parser.add_argument("--checkpoint-every-updates", type=int, default=200,
                        help="[train] Save a resumable checkpoint every N optimizer updates (0 = epoch ends only).")
    parser.add_argument("--max-hours", type=float, default=None,
                        help="[train] Stop cleanly (after saving) once this many hours have passed; "
                             "re-run to continue.")
    args = parser.parse_args()

    if args.stage == "prepare":
        stage_prepare(args.repo_root, args.candidates, args.val_frac, args.calib_frac, args.seed,
                      args.max_negatives_per_entity, args.max_finetune_entities)
    else:
        if not args.role:
            raise SystemExit("--stage train requires --role {english,multilingual}")
        status = stage_train(args.repo_root, args.role, args.epochs, args.micro_batch, args.grad_accum,
                             args.lr_encoder, args.lr_head, args.calib_max, args.seed,
                             resume=not args.no_resume, checkpoint_every_updates=args.checkpoint_every_updates,
                             max_hours=args.max_hours)
        if int(os.environ.get("RANK", "0")) == 0:
            print(f"[laya_finetune/train] {args.role}: {status}")


if __name__ == "__main__":
    main()

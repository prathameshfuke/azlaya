"""Generates run_pipeline_colab_kaggle.ipynb: a self-contained notebook that writes every pipeline
source file itself (one %%writefile cell per file, embedded verbatim from src/) and then runs
the pipeline step by step. No git clone or pull is needed at runtime, and the notebook can't
drift from src/ because it's regenerated from it.

Regenerate after editing anything in src/ (from code/business_entity_resolution/):
    python notebooks/build_notebook.py
"""
import hashlib
import json
from pathlib import Path

PIPELINE = Path(__file__).resolve().parents[1]          # code/business_entity_resolution
STUDENT_RESOURCE = PIPELINE.parents[1]                   # student_resource
OUT = PIPELINE / "notebooks" / "run_pipeline_colab_kaggle.ipynb"

# (notebook-side path relative to a root variable, repo-side source file)
EMBEDDED = [
    ("$PIPELINE_DIR/requirements.txt", PIPELINE / "requirements.txt"),
    ("$PIPELINE_DIR/README.md", PIPELINE / "README.md"),
    ("$PIPELINE_DIR/src/__init__.py", PIPELINE / "src" / "__init__.py"),
    ("$PIPELINE_DIR/src/common.py", PIPELINE / "src" / "common.py"),
    ("$PIPELINE_DIR/src/normalize.py", PIPELINE / "src" / "normalize.py"),
    ("$PIPELINE_DIR/src/blocking.py", PIPELINE / "src" / "blocking.py"),
    ("$PIPELINE_DIR/src/features.py", PIPELINE / "src" / "features.py"),
    ("$PIPELINE_DIR/src/train_gbdt.py", PIPELINE / "src" / "train_gbdt.py"),
    ("$PIPELINE_DIR/src/laya_finetune.py", PIPELINE / "src" / "laya_finetune.py"),
    ("$PIPELINE_DIR/src/ensemble.py", PIPELINE / "src" / "ensemble.py"),
    ("$PIPELINE_DIR/src/predict.py", PIPELINE / "src" / "predict.py"),
    ("$STUDENT_RESOURCE_DIR/utils/validate_submission.py", STUDENT_RESOURCE / "utils" / "validate_submission.py"),
    ("$STUDENT_RESOURCE_DIR/Documentation_template.md", STUDENT_RESOURCE / "Documentation_template.md"),
]

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)})


def code(text):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                  "source": text.splitlines(keepends=True)})


# ---------------------------------------------------------------------------------------- title
md("""\
# Business Entity Resolution: full pipeline

Runs the ML Challenge 2026 Business Entity Resolution pipeline end to end: normalize, blocking,
features, GBDT, Laya fine-tuning, ensemble, test-set inference, validate, package.

**Self-contained:** Section 0.3 writes every pipeline source file (`src/*.py`,
`requirements.txt`, `README.md`, the challenge's `validate_submission.py` and
`Documentation_template.md`) from `%%writefile` cells, so all the code is visible here. Nothing is
cloned or pulled at runtime. To change the pipeline, edit the `%%writefile` cell, re-run it, then
**restart the kernel** before re-running later steps so the new code is loaded.

**Hardware:** Sections 1-4 are CPU work (text processing and gradient boosting; a GPU doesn't
speed them up). Section 5 (Laya fine-tuning) and the Laya scoring in Sections 6-7 use the GPU.
On Kaggle choose **GPU T4 x2**: fine-tuning runs DDP across both GPUs, and scoring runs one
model copy per GPU in parallel.

Run cells **top to bottom**; each step reads what the previous one wrote. Every step checks its
inputs and names the earlier step to re-run if something is missing.

**Resuming after an interruption** (session timeout, crash, closed tab): just **Run All** again.
With `RESUME = True` (Section 0.6, the default) every step whose output already exists is skipped
in seconds, and Laya fine-tuning continues from its last checkpoint (saved every ~200 updates and
at each epoch end), mid-epoch included. Section 0.6 prints exactly what's done. Don't run a single
later cell on its own after a restart: the setup cells in Section 0 define what it needs.

The files have to survive for this to work. On Kaggle, `/kaggle/working` is kept between sessions
only with **Settings → Persistence → Files** turned on. Otherwise, save a version, attach its
output to the next session (**+ Add Data → Your Work**), and Section 0.5b copies it back in.
""")

# ---------------------------------------------------------------------------------------- setup
md("## 0. Setup\n\n### 0.1 Check the GPU\n")

code("""\
import subprocess

try:
    result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
    print(result.stdout or result.stderr)
except FileNotFoundError:
    print("nvidia-smi not found -- no GPU accelerator is selected for this session.")
    print("Kaggle: Settings (right sidebar) -> Accelerator -> GPU T4 x2, then re-run this cell.")
    print("Sections 1-4 run on CPU; only Sections 5-7 need the GPU, so you can add it before Section 5.")
""")

md("### 0.2 Check internet access\n\nRequired for `pip install` below and (in Section 5) downloading Laya's base checkpoints from\nHugging Face Hub. On Kaggle, internet is OFF by default for new notebooks -- this is the single\nmost common reason \"Run All\" silently goes wrong partway through.\n")

code("""\
import urllib.request

_INTERNET_HOSTS = ["https://pypi.org", "https://huggingface.co"]
_unreachable = []
for _host in _INTERNET_HOSTS:
    try:
        urllib.request.urlopen(_host, timeout=8)
    except Exception as e:
        _unreachable.append((_host, str(e)))

if _unreachable:
    detail = "\\n".join(f"  {h}: {e}" for h, e in _unreachable)
    raise RuntimeError(
        "No internet access from this session -- can't reach:\\n" + detail +
        "\\n\\nKaggle: Settings (right sidebar) -> Internet -> turn ON, then Save Version / restart "
        "the session and Run All again. (A phone number must be verified on the Kaggle account for "
        "this toggle to be available.) Colab: internet is on by default; check your network/proxy "
        "if this still fails there."
    )
print("Internet access OK:", ", ".join(_INTERNET_HOSTS))
""")

md("""\
### 0.3 Write the pipeline code

Creates the submission-package layout and writes every source file into it. Re-running these
cells overwrites the files with the version shown here.
""")

code("""\
import os

# /kaggle/working on Kaggle, /content on Colab, else the notebook's own directory.
if os.path.isdir("/kaggle/working"):
    WORKDIR = "/kaggle/working"
elif os.path.isdir("/content"):
    WORKDIR = "/content"
else:
    WORKDIR = os.getcwd()

STUDENT_RESOURCE_DIR = f"{WORKDIR}/azlaya/student_resource"
PIPELINE_DIR = f"{STUDENT_RESOURCE_DIR}/code/business_entity_resolution"
for d in (f"{PIPELINE_DIR}/src", f"{STUDENT_RESOURCE_DIR}/utils"):
    os.makedirs(d, exist_ok=True)
print("STUDENT_RESOURCE_DIR:", STUDENT_RESOURCE_DIR)
print("PIPELINE_DIR:        ", PIPELINE_DIR)
""")

def _as_fstring_path(nb_path: str) -> str:
    return nb_path.replace("$PIPELINE_DIR", "{PIPELINE_DIR}").replace("$STUDENT_RESOURCE_DIR", "{STUDENT_RESOURCE_DIR}")


_hash_lines_parts = []
for nb_path, src_path in EMBEDDED:
    body = src_path.read_text(encoding="utf-8")
    if not body.strip():
        # `%%writefile` is a cell magic that refuses a genuinely empty cell body ("UsageError:
        # %%writefile is a cell magic, but the cell body is empty") -- src/__init__.py is 0
        # bytes, which hit exactly that. This halted "Run All" at that cell in practice, which
        # then skipped every cell after it (including src/common.py's own %%writefile) with no
        # error of its own -- so common.py on disk stayed whatever old version was already
        # there, which is what actually caused a later "module 'src.common' has no attribute
        # ..." error, not a stale in-memory module. A one-line comment keeps the file's meaning
        # (an empty Python package marker) while giving %%writefile a non-empty body.
        body = "# (intentionally empty)\n"
    elif not body.endswith("\n"):
        body += "\n"
    code(f"%%writefile {nb_path}\n{body}")
    # Hash the exact `body` that gets written (not the raw source file), so the empty-file
    # substitution above doesn't cause a spurious permanent mismatch for __init__.py.
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    _hash_lines_parts.append(f'_expected[f"{_as_fstring_path(nb_path)}"] = "{digest}"\n')
_hash_lines = "".join(_hash_lines_parts)

code(f"""\
# Verify every file above actually landed on disk with current content, instead of trusting that
# each %%writefile cell ran. A %%writefile cell errors and halts "Run All" on some inputs (e.g. a
# genuinely empty file used to do this); every cell after the failure is then silently skipped,
# leaving old file content in place with no error of its own -- that's what caused a later,
# confusing "module 'src.common' has no attribute ..." several sections down, not a code bug.
import hashlib
import py_compile

_expected = {{}}
{_hash_lines}
problems = []
for path, expected_hash in _expected.items():
    if not os.path.isfile(path):
        problems.append(f"{{path}}: missing -- its %%writefile cell above didn't run")
        continue
    actual = hashlib.sha256(open(path, "rb").read()).hexdigest()
    if actual != expected_hash:
        problems.append(f"{{path}}: on-disk content doesn't match this notebook -- re-run its %%writefile cell above")
    if path.endswith(".py"):
        try:
            py_compile.compile(path, doraise=True)
        except py_compile.PyCompileError as e:
            problems.append(f"{{path}}: does not compile -- {{e}}")

if problems:
    raise RuntimeError(
        "Section 0.3 did not finish writing the pipeline correctly:\\n  " + "\\n  ".join(problems)
        + "\\n\\nRe-run the listed %%writefile cell(s) above (scroll up), confirm each one's output "
          "starts with 'Writing' or 'Overwriting', then re-run this cell."
    )
print(f"Verified {{len(_expected)}} files on disk match this notebook.")
""")

md("""\
### 0.4 Install dependencies

Installs exactly `requirements.txt` (written above) and prints it first. Runs pip as a checked
subprocess (not `!pip`, whose failure `!`-shell syntax would NOT stop "Run All" -- it would just
print an error and silently move on to cells that then fail confusingly with missing imports) so
a real install failure stops here with pip's own error text. Also self-heals a numpy/pandas binary
mismatch if the install ever causes one.
""")

code("""\
import subprocess as _subprocess

%cd $PIPELINE_DIR
print(open("requirements.txt").read())
_pip_result = _subprocess.run(["pip", "install", "-q", "-r", "requirements.txt"],
                              capture_output=True, text=True)
print(_pip_result.stdout)
if _pip_result.returncode != 0:
    raise RuntimeError(
        "pip install failed (exit code " + str(_pip_result.returncode) + "):\\n" + _pip_result.stderr +
        "\\n\\nCommon causes: internet is off (see 0.2 above), Kaggle disk quota, or a genuinely "
        "unavailable package version."
    )

try:
    import pandas as pd
    import numpy as np
except ValueError as e:
    print(f"numpy/pandas ABI mismatch detected ({e}) -- reinstalling both together...")
    _subprocess.run(["pip", "install", "-q", "--upgrade", "--force-reinstall", "numpy", "pandas"], check=True)
    import pandas as pd
    import numpy as np

import laya, sparse_dot_topn, torch, transformers
print("laya", laya.__version__, "| torch", torch.__version__, "| transformers", transformers.__version__,
      "| numpy", np.__version__, "| pandas", pd.__version__, "| CUDA available:", torch.cuda.is_available())
""")

md("""\
### 0.5 Provide the dataset

Attach **[prathameshfuke/dataset-ml](https://www.kaggle.com/datasets/prathameshfuke/dataset-ml)**
(private: be signed into the `prathameshfuke` Kaggle account) via **+ Add Data** in the right
sidebar. The next cell finds `train_source1.tsv` / `test_source1.tsv` anywhere under
`/kaggle/input/` and symlinks their folders into `student_resource/dataset/`. The fallback cells
after it are only for running outside Kaggle.
""")

code("""\
import glob
import shutil


def find_dataset_under(root):
    train_hits = sorted(glob.glob(f"{root}/**/train_source1.tsv", recursive=True))
    test_hits = sorted(glob.glob(f"{root}/**/test_source1.tsv", recursive=True))
    for name, hits in (("train_source1.tsv", train_hits), ("test_source1.tsv", test_hits)):
        if len(hits) > 1:
            print(f"WARNING: multiple {name} found, using {hits[0]}:\\n  " + "\\n  ".join(hits))
    return (os.path.dirname(train_hits[0]) if train_hits else None,
            os.path.dirname(test_hits[0]) if test_hits else None)


def link_dataset(train_dir, test_dir):
    os.makedirs(f"{STUDENT_RESOURCE_DIR}/dataset", exist_ok=True)
    for name, src in (("train", train_dir), ("test", test_dir)):
        dst = f"{STUDENT_RESOURCE_DIR}/dataset/{name}"
        if os.path.islink(dst):
            os.remove(dst)
        elif os.path.isdir(dst):
            shutil.rmtree(dst)
        os.symlink(src, dst)


train_dir, test_dir = find_dataset_under("/kaggle/input") if os.path.isdir("/kaggle/input") else (None, None)
if train_dir and test_dir:
    link_dataset(train_dir, test_dir)
    print(f"train -> {train_dir}\\ntest  -> {test_dir}\\nsymlinked into {STUDENT_RESOURCE_DIR}/dataset/")
else:
    print("No train_source1.tsv/test_source1.tsv under /kaggle/input.")
    print("On Kaggle: + Add Data -> search 'dataset-ml' -> add prathameshfuke/dataset-ml, then re-run this cell.")
""")

md("Fallbacks, only if the cell above found nothing (e.g. on Colab).")

code("""\
# Fallback 1 -- a direct download URL to a zip with train/ and test/ at its root.
RUN_FALLBACK_URL = False
DATASET_URL = ""

if RUN_FALLBACK_URL:
    assert DATASET_URL, "Set DATASET_URL first."
    os.makedirs(f"{STUDENT_RESOURCE_DIR}/dataset", exist_ok=True)
    zip_path = f"{WORKDIR}/dataset_download.zip"
    subprocess.run(["curl", "-fL", "-o", zip_path, DATASET_URL], check=True)
    subprocess.run(["unzip", "-q", "-o", zip_path, "-d", f"{STUDENT_RESOURCE_DIR}/dataset"], check=True)
""")

code("""\
# Fallback 2 -- Google Drive (Colab): folder containing train/ and test/.
RUN_FALLBACK_DRIVE = False
DRIVE_DATASET_DIR = "/content/drive/MyDrive/azlaya-dataset"

if RUN_FALLBACK_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    link_dataset(f"{DRIVE_DATASET_DIR}/train", f"{DRIVE_DATASET_DIR}/test")
""")

code("""\
required = [
    "dataset/train/train_source1.tsv", "dataset/train/train_source2.tsv",
    "dataset/train/train_source3.tsv", "dataset/train/train_ground_truth.tsv",
    "dataset/test/test_source1.tsv", "dataset/test/test_source2.tsv", "dataset/test/test_source3.tsv",
]
missing = [f for f in required if not os.path.isfile(f"{STUDENT_RESOURCE_DIR}/{f}")]
if missing:
    raise FileNotFoundError("Missing dataset files -- run a dataset cell above first:\\n  " + "\\n  ".join(missing))
print("All dataset files found.")
""")

md("""\
### 0.5b Restore a previous run's files (optional)

If a previous session's `/kaggle/working` didn't survive, attach that run's saved output
(**+ Add Data → Your Work →** this notebook). This cell finds its
`azlaya/student_resource/{data_processed,models,output}` and copies back any file that isn't
already here, so Section 0.6 sees those steps as done. With nothing attached it does nothing.
""")

code("""\
from pathlib import Path

RESTORE_FROM_INPUT = True


def find_previous_run(root="/kaggle/input"):
    if not os.path.isdir(root):
        return None
    for p in sorted(Path(root).glob("**/azlaya/student_resource")):
        if (p / "data_processed").is_dir() or (p / "models").is_dir():
            return p
    return None


_prev = find_previous_run() if RESTORE_FROM_INPUT else None
if _prev is None:
    print("No previous run attached -- nothing to restore (normal for a first run, or when "
          "/kaggle/working persisted).")
else:
    _copied = _skipped = 0
    for _sub in ("data_processed", "models", "output"):
        for _f in sorted((_prev / _sub).rglob("*")) if (_prev / _sub).is_dir() else []:
            # skip half-written files/dirs (<name>.tmp) and the transient checkpoint backup (.old)
            if not _f.is_file() or any(p.endswith((".tmp", ".old")) for p in _f.relative_to(_prev).parts):
                continue
            _dst = Path(STUDENT_RESOURCE_DIR) / _sub / _f.relative_to(_prev / _sub)
            if _dst.exists():
                _skipped += 1
                continue
            _dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(_f, _dst)
            _copied += 1
    print(f"Restored {_copied} file(s) from {_prev} ({_skipped} already present, left untouched).")
""")

md("""\
### 0.6 Load the pipeline modules + run settings

Steps run **in this kernel** (not as `!python` subprocesses) so progress bars are live. Re-running
this cell reloads `src/` from disk.

**Run settings** (edit here):

- `RESUME = True` skips every step whose output already exists, and continues Laya from its
  checkpoint. Set it to `False` to recompute everything, e.g. after changing a step's code or
  settings: its old output would otherwise be reused.
- `LAYA_EPOCHS`: epochs per checkpoint. `multilingual` defaults to 2. The Router only sends it
  records with non-Latin names (~4% of fine-tuning pairs in this data), yet it trains on all pairs,
  so 4 epochs would cost as much GPU time as `english` for a small share of predictions.
- `LAYA_TIME_BUDGET_HOURS`: Section 5b stops cleanly (checkpoint saved) after this long, which
  leaves time in a 12-hour Kaggle GPU session for Sections 6-9. Re-run to continue.
""")

code("""\
import gc
import importlib
import json
import sys
from pathlib import Path

if PIPELINE_DIR not in sys.path:
    sys.path.insert(0, PIPELINE_DIR)

_PIPELINE_MODULE_NAMES = ["common", "features", "blocking", "train_gbdt", "laya_finetune", "ensemble", "normalize", "predict"]


def reload_pipeline():
    \"\"\"Re-sync every pipeline module against what is CURRENTLY on disk in src/. Call this
    (every step cell below does, automatically) before using any pipeline function.

    Why this exists: a %%writefile cell only writes a file to disk -- it does not update code
    already loaded into this running kernel. normalize.py's own `from src import common`, for
    example, bound a specific `common` module OBJECT the first time normalize.py was imported;
    rewriting common.py on disk afterwards does not change that object. importlib.reload()
    re-executes a module's code INTO its existing object (unlike deleting it from sys.modules
    and re-importing, which creates a new object other modules keep no reference to), so every
    already-imported module sees the update immediately. This is what fixes
    "module 'src.common' has no attribute 'require'"-style errors after editing a %%writefile
    cell without a full kernel restart.

    If reload() itself ever errors (rare -- can happen after a large structural edit, e.g.
    removing a module-level name other code still imports by name), the reliable fallback is
    Kaggle's Run menu -> Restart Session, then Run All from the top.
    \"\"\"
    mods = {}
    for name in _PIPELINE_MODULE_NAMES:
        full = f"src.{name}"
        mods[name] = importlib.reload(sys.modules[full]) if full in sys.modules else importlib.import_module(full)
    return mods


globals().update(reload_pipeline())

ROOT = Path(STUDENT_RESOURCE_DIR)
DP = ROOT / "data_processed"


def free_memory():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def run_streaming(cmd):
    \"\"\"Run a subprocess (only torchrun) and stream its output live.\"\"\"
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    proc = subprocess.Popen(cmd, cwd=PIPELINE_DIR, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for chunk in iter(lambda: os.read(proc.stdout.fileno(), 4096), b""):
        sys.stdout.write(chunk.decode(errors="replace"))
        sys.stdout.flush()
    if proc.wait() != 0:
        raise RuntimeError(f"{' '.join(cmd)} exited with code {proc.returncode} -- see output above")


print("modules loaded | CPU cores:", os.cpu_count())

# ---------------------------------------------------------------- run settings
RESUME = True
LAYA_EPOCHS = {"english": 4, "multilingual": 2}
LAYA_TIME_BUDGET_HOURS = 7.0
CHECKPOINT_EVERY_UPDATES = 200

MODELS = ROOT / "models"
STEP_OUTPUTS = {
    "1. normalize": [DP / f"{s}_{k}_normalized.tsv" for s in ("train", "test")
                     for k in ("source1", "source2", "source3")],
    "2. blocking": [DP / "candidate_pairs_train.tsv"],
    "3. features": [DP / "features_train.parquet"],
    "4. GBDT": [MODELS / "gbdt_model.txt", DP / "gbdt_val_report.json"],
    "5a. Laya datasets": [DP / f"laya_{r}_{p}.jsonl" for r in ("english", "multilingual")
                          for p in ("train", "calib")],
    "6. ensemble": [MODELS / "ensemble_config.json", DP / "ensemble_val_report.json"],
}


def step_done(step):
    return RESUME and all(p.exists() for p in STEP_OUTPUTS[step])


def skip_note(step):
    print(f"RESUME: '{step}' output already exists -- skipped. (Set RESUME = False in 0.6 to redo it.)")


def prediction_status():
    marker = DP / "prediction_done.json"
    return json.loads(marker.read_text()) if marker.exists() else None


def print_pipeline_status():
    print("Pipeline status" + ("" if RESUME else "  (RESUME = False: every step will re-run)"))
    for step, paths in STEP_OUTPUTS.items():
        if step == "6. ensemble":
            for role in laya_finetune.ROLES:
                print(f"  5b. Laya {role:13s} {laya_finetune.checkpoint_status(ROOT, role, LAYA_EPOCHS[role])}")
        missing = [p.name for p in paths if not p.exists()]
        print(f"  {step:22s} {'done' if not missing else 'to do' + (f' (missing: {missing[0]})' if len(missing) < len(paths) else '')}")
    pred = prediction_status()
    print(f"  {'7. test predictions':22s} " + (f"done with '{pred['stack_model']}'" if pred else "to do"))


print_pipeline_status()
""")

# ------------------------------------------------------------------------------------ section 1
md("""\
## 1. Normalize

Cleans names and addresses in all six source files (legal suffixes, DBA splitting, pin/postal
code, city guess, script detection), in parallel across every CPU core. With `RESUME`, files
already normalized are reused one by one.
""")

code("""\
reload_pipeline()
normalize.run(ROOT, ["train", "test"], skip_existing=RESUME)
free_memory()
""")

code("""\
import pandas as pd
sample = pd.read_csv(DP / "train_source1_normalized.tsv", sep="\\t", dtype=str, nrows=10, keep_default_na=False)
sample[["business_name", "name_norm", "has_dba", "trade_name_norm", "address_norm", "pin_code", "city_guess", "name_script"]]
""")

# ------------------------------------------------------------------------------------ section 2
md("""\
## 2. Blocking (candidate generation)

For each S1 entity, finds up to K=30 similar S2/S3 records in the same country. It combines a
name view (character 4-gram TF-IDF) and an address view (word TF-IDF, which carries pin, city and
street), searched with a multithreaded sparse top-n matrix product.

**Training uses a sample of `MAX_TRAIN_S1` S1 entities** (default 300k of the ~2.2M). Every later
training step uses the same sample. Their candidates are still searched against the full S2/S3
pool, so hard negatives stay realistic. The test set is always blocked in full (Section 7).

**Read the recall report:** `macro_entity_recall` is the ceiling for everything downstream.
""")

code("""\
reload_pipeline()
MAX_TRAIN_S1 = 300_000
if step_done("2. blocking"):
    skip_note("2. blocking")
else:
    blocking.run(ROOT, "train", k=30, out_path=DP / "candidate_pairs_train.tsv", report_recall=True,
                 val_frac=0.2, seed=42, max_s1=MAX_TRAIN_S1)
    free_memory()
""")

# ------------------------------------------------------------------------------------ section 3
md("""\
## 3. Feature engineering

Levenshtein, Jaro-Winkler, word/char TF-IDF cosine, address similarity, pin/city/country match,
DBA-aware name match. Computed in chunks for every candidate pair.
""")

code("""\
import pyarrow.parquet as pq

reload_pipeline()
if step_done("3. features"):
    skip_note("3. features")
else:
    features.run(ROOT, "train", DP / "candidate_pairs_train.tsv", None)
    free_memory()
_pf = pq.ParquetFile(DP / "features_train.parquet")   # peek without loading all rows into memory
print(f"features_train.parquet: {_pf.metadata.num_rows:,} rows x {_pf.metadata.num_columns} columns")
_pf.read_row_group(0).to_pandas().head()
""")

# ------------------------------------------------------------------------------------ section 4
md("""\
## 4. GBDT matcher

LightGBM on the feature table, split by whole S1 entity so no validation entity leaks into
training. Reports precision / recall / macro F_0.5 on the held-out entities.
""")

code("""\
reload_pipeline()
if step_done("4. GBDT"):
    skip_note("4. GBDT")
else:
    _ = train_gbdt.train(ROOT, DP / "features_train.parquet", val_frac=0.2, seed=42,
                         num_boost_round=2000, early_stopping_rounds=50)
    free_memory()
print(json.dumps(json.load(open(DP / "gbdt_val_report.json")), indent=2))
""")

md("""\
### 4b. Safety submission: GBDT only (optional, recommended)

Laya fine-tuning takes many GPU hours, and a session can end before it's done. This produces a
**complete, valid submission from the GBDT alone** now, and keeps a copy in
`output_gbdt_only/` and `submission_gbdt_only.zip`. Nothing here is wasted: the test-set
blocking and features it computes (the slow CPU part of Section 7) are cached and reused by
Section 7. Skipped when a submission already exists.
""")

code("""\
import shutil

MAKE_SAFETY_SUBMISSION = True

reload_pipeline()
if not MAKE_SAFETY_SUBMISSION:
    print("MAKE_SAFETY_SUBMISSION = False -- skipped.")
elif RESUME and prediction_status() is not None:
    print(f"RESUME: a submission already exists (made with '{prediction_status()['stack_model']}') -- skipped.")
else:
    predict.run(ROOT, k=30, stack_model="gbdt_only", laya_batch_size=ensemble.LAYA_BATCH_SIZE,
                threshold_override=None)
    free_memory()
    _safe = ROOT / "output_gbdt_only"
    shutil.rmtree(_safe, ignore_errors=True)
    shutil.copytree(ROOT / "output", _safe)
    _r = subprocess.run([sys.executable, "utils/validate_submission.py", "--matching", "output/matching_results.tsv",
                         "--candidate", "output/candidate_pairs.tsv", "--test-dir", "dataset/test"],
                        cwd=STUDENT_RESOURCE_DIR, capture_output=True, text=True)
    print(_r.stdout[-3000:], _r.stderr[-3000:])
    _stage = Path(WORKDIR) / "submission_gbdt_only_package"
    shutil.rmtree(_stage, ignore_errors=True)
    shutil.copytree(_safe, _stage / "output")
    shutil.copytree(PIPELINE_DIR, _stage / "code/business_entity_resolution",
                    ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy(ROOT / "Documentation_template.md", _stage / "Documentation_template.md")
    print("Safety submission:", shutil.make_archive(f"{WORKDIR}/submission_gbdt_only", "zip", _stage))
""")

# ------------------------------------------------------------------------------------ section 5
md("""\
## 5. Laya fine-tuning

Two checkpoints fine-tuned with Laya's RLCD recipe: `english` on Latin-script pairs,
`multilingual` on all pairs (covers Devanagari/Tamil). Examples come from `MAX_FINETUNE_ENTITIES`
training entities: all their true matches, plus up to 3 hard negatives each, in both orderings.
The default of 6000 entities (~66-69k examples) takes **hours per epoch** on T4 x2; a 4-epoch
`english` run did not finish in one overnight session. That's why 5b is resumable and
time-budgeted, and why 4b makes a GBDT-only safety submission first.

### 5a. Build the fine-tuning datasets (CPU)
""")

code("""\
reload_pipeline()
MAX_FINETUNE_ENTITIES = 6000
if step_done("5a. Laya datasets"):
    skip_note("5a. Laya datasets")
else:
    laya_finetune.stage_prepare(ROOT, None, val_frac=0.2, calib_frac=0.15, seed=42,
                                max_negatives_per_entity=3, max_finetune_entities=MAX_FINETUNE_ENTITIES)
    free_memory()
with open(DP / "laya_english_train.jsonl") as f:
    print(json.dumps(json.loads(next(f)), indent=2))
""")

md("""\
### 5b. Fine-tune both checkpoints

With 2+ GPUs this runs `torchrun` DDP across all of them, streaming output live. DDP needs
separate processes, so this step can't run in-kernel. With one GPU it runs in-kernel.

**Resumable.** Each role saves `models/laya_<role>/checkpoint_latest/` every
`CHECKPOINT_EVERY_UPDATES` optimizer updates and at each epoch end. Re-running this cell continues
from there (same batch order, LR schedule fast-forwarded), skips a role that's already finished,
and stops cleanly after `LAYA_TIME_BUDGET_HOURS` (both set in 0.6). A role is finished once
`models/laya_<role>/rl_agent_config.json` exists. Until both are, Sections 6-7 fall back to the
base (not fine-tuned) checkpoint for the missing one, with a warning.

Fine-tuning uses the `%%writefile` version of `src/laya_finetune.py` on disk, so re-run its cell
in 0.3 first if you edited it.
""")

code("""\
import time
import torch

reload_pipeline()
N_GPUS = torch.cuda.device_count()
print(f"visible GPUs: {N_GPUS}", [torch.cuda.get_device_name(i) for i in range(N_GPUS)])
if N_GPUS == 0:
    raise RuntimeError("No GPU visible -- turn on an accelerator (Kaggle: Settings -> Accelerator) before Section 5.")

_t_start = time.time()
for role in ("english", "multilingual"):
    print(f"\\n========== {role}: {laya_finetune.checkpoint_status(ROOT, role, LAYA_EPOCHS[role])} ==========")
    if RESUME and laya_finetune.role_is_done(ROOT, role):
        continue
    hours_left = LAYA_TIME_BUDGET_HOURS - (time.time() - _t_start) / 3600
    if hours_left < 0.25:
        print(f"Time budget ({LAYA_TIME_BUDGET_HOURS}h) used up -- re-run this cell to continue with {role}.")
        break
    if N_GPUS >= 2:
        run_streaming(["torchrun", "--standalone", f"--nproc_per_node={N_GPUS}",
                       "-m", "src.laya_finetune", "--stage", "train", "--role", role,
                       "--epochs", str(LAYA_EPOCHS[role]),
                       "--checkpoint-every-updates", str(CHECKPOINT_EVERY_UPDATES),
                       "--max-hours", f"{hours_left:.3f}"] + ([] if RESUME else ["--no-resume"]))
    else:
        laya_finetune.stage_train(ROOT, role, epochs=LAYA_EPOCHS[role], micro_batch=8, grad_accum=4,
                                  lr_encoder=2.5e-5, lr_head=1.0e-4, calib_max=400, seed=42,
                                  resume=RESUME, checkpoint_every_updates=CHECKPOINT_EVERY_UPDATES,
                                  max_hours=hours_left)
    free_memory()
    if not laya_finetune.role_is_done(ROOT, role):
        print(f"\\n{role} is paused ({laya_finetune.checkpoint_status(ROOT, role, LAYA_EPOCHS[role])}). "
              f"Re-run this cell to continue -- in this session, or in a new one with the files kept.")
        break

print()
print_pipeline_status()
""")

# ------------------------------------------------------------------------------------ section 6
md("""\
## 6. Ensemble

Trains the stacker **only on entities the GBDT never saw** (its held-out split, capped at
`MAX_STACK_ENTITIES`, halved into stack-train / stack-eval). On the GBDT's own training entities
its probabilities are overfit, and a stacker trained there would over-trust them.

Laya scores a shortlist: each entity's top 3 GBDT candidates with probability >= 0.05. The
stacker combines the GBDT features, `gbdt_prob` and `laya_prob`, then the F_0.5 threshold is
re-fit. The report compares logistic / shallow-GBDT stackers against a **GBDT-only baseline** on
the same entities, broken down by country and singleton status.
""")

code("""\
reload_pipeline()
MAX_STACK_ENTITIES = 60_000
_unfinished = [r for r in laya_finetune.ROLES if not laya_finetune.role_is_done(ROOT, r)]
if _unfinished:
    print(f"WARNING: Laya {_unfinished} not fine-tuned yet (Section 5b) -- the base checkpoint is used for "
          f"{'it' if len(_unfinished) == 1 else 'them'}. Finish 5b first unless you're out of GPU time.")
if step_done("6. ensemble"):
    skip_note("6. ensemble")
else:
    free_memory()
    ensemble.run(ROOT, DP / "features_train.parquet", val_frac=0.2, seed=42,
                 laya_batch_size=ensemble.LAYA_BATCH_SIZE, laya_top_n=ensemble.LAYA_TOP_N,
                 laya_min_gbdt_prob=ensemble.LAYA_MIN_GBDT_PROB, max_entities=MAX_STACK_ENTITIES)
    free_memory()
""")

code("""\
ens_report = json.load(open(DP / "ensemble_val_report.json"))
scores = {name: ens_report[name]["threshold_sweep"]["best"] for name in ("gbdt_only_baseline", "logistic", "gbdt_alt")}
for name, best in scores.items():
    print(f"{name:20s} threshold={best['threshold']:<5} macro_F0.5={best['macro_f0_5']:.4f} "
          f"precision={best['precision']:.4f} recall={best['recall']:.4f}")

best_name = max(scores, key=lambda n: scores[n]["macro_f0_5"])
BEST_MODEL = "gbdt_only" if best_name == "gbdt_only_baseline" else best_name
print(f"\\nUsing for the test set: {BEST_MODEL}"
      + ("  (Laya didn't beat GBDT alone here, so Section 7 skips Laya)" if BEST_MODEL == "gbdt_only" else ""))
print(json.dumps(ens_report[best_name]["breakdown"], indent=2))
""")

# ------------------------------------------------------------------------------------ section 7
md("""\
## 7. Full test-set inference

Blocks **every** test S1 entity, then runs features, GBDT, Laya (unless `BEST_MODEL` is
`gbdt_only`) and the ensemble with its threshold. Writes `output/candidate_pairs.tsv` and
`output/matching_results.tsv`. If Section 4b already blocked the test set and computed its
features, those are reused, and only the scoring runs.
""")

code("""\
reload_pipeline()
_pred = prediction_status()
if RESUME and _pred is not None and _pred["stack_model"] == BEST_MODEL:
    print(f"RESUME: output/ already holds predictions from '{BEST_MODEL}' -- skipped.")
else:
    free_memory()
    predict.run(ROOT, k=30, stack_model=BEST_MODEL, laya_batch_size=ensemble.LAYA_BATCH_SIZE,
                threshold_override=None)
    free_memory()
""")

# ------------------------------------------------------------------------------------ section 8
md("## 8. Validate before submitting\n\nThe challenge's own validator, run from `student_resource/`.\n")

code("""\
%cd $STUDENT_RESOURCE_DIR
!python3 utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
%cd $PIPELINE_DIR
""")

# ------------------------------------------------------------------------------------ section 9
md("""\
## 9. Package the submission zip

`output/`, `code/business_entity_resolution/` (src, README, requirements) and
`Documentation_template.md`, in the layout the challenge requires.
""")

code("""\
zip_root = f"{WORKDIR}/submission_package"
if os.path.isdir(zip_root):
    shutil.rmtree(zip_root)
os.makedirs(zip_root)
shutil.copytree(f"{STUDENT_RESOURCE_DIR}/output", f"{zip_root}/output")
shutil.copytree(PIPELINE_DIR, f"{zip_root}/code/business_entity_resolution",
                ignore=shutil.ignore_patterns("__pycache__"))
shutil.copy(f"{STUDENT_RESOURCE_DIR}/Documentation_template.md", f"{zip_root}/Documentation_template.md")
archive_path = shutil.make_archive(f"{WORKDIR}/submission", "zip", zip_root)
print("Wrote", archive_path, "(Kaggle: Output pane after saving a version)")
""")

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        "accelerator": "GPU",
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("wrote", OUT, f"({len(cells)} cells)")

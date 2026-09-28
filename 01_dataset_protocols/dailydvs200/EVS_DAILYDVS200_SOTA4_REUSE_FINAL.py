#!/usr/bin/env python3
"""
Event-Security Study — DailyDVS-200 (headless cluster run).

Fourth dataset, following the validated N-MNIST / DVS-Gesture / CIFAR10-DVS
protocol. Same attack definition, same controls, same invariance columns,
same attack/control pipeline; the victim registry is upgraded to four
protocol-scale variants of strong DailyDVS-200 architectures with public code.

WHAT IS DIFFERENT ABOUT THIS DATASET (read before launching)
------------------------------------------------------------
1. Format: recordings are AEDAT 4.0 (`.aedat4`), a compressed FlatBuffer
   container — NOT the raw AEDAT2 byte stream used by DVS-Gesture/CIFAR.
   It cannot be parsed with a short native reader, so this script uses a
   reader backend, tried in order:
       dv_processing  (pip install dv-processing)   [preferred]
       aedat          (pip install aedat)           [rust-backed fallback]
       dv             (pip install dv)              [legacy fallback]
   A cached .npz layer means the backend is needed once, at preprocessing.
2. Split: the authors publish train/val/test.txt split files. We use them
   verbatim, then audit participant ids parsed from filenames and record any
   overlap in split_subject_audit.json. Report the split according to that
   audit rather than assuming subject-disjointness.
3. Scale: 22,046 recordings across 200 classes at 320x240. Full 5 models x
   3 seeds (one protected consumer + four strong-architecture variants) is expensive
   (see RUNTIME below). Two
   pre-declared knobs keep it tractable; both are recorded in run_config:
       NUM_CLASSES_SUBSET  keep the first K classes by label id (0 = all)
       MAX_TRAIN_PER_CLASS cap recordings per class in the train split
   Either changes the task, so fix them BEFORE looking at any result.
4. Accuracy expectations: published/later comparison results on DailyDVS-200
   report Video Swin-T 48.06, TimeSformer 44.25, MVFNet 48.30, and ACTION-Net
   42.61 top-1. The first two are from the official DailyDVS-200 benchmark;
   MVFNet and ACTION-Net are reported by later DailyDVS-200 comparisons. The
   models below are protocol-scale architecture variants trained under the SAME
   64x64 / 80-bin pipeline as the previous DailyDVS run. They are not the
   released 224x224/pretrained benchmark configurations, so their clean
   accuracies are not expected to reproduce those published numbers. ASR on a victim whose
   clean accuracy is near chance is not interpretable; every row carries
   `asr_interpretable` against MIN_CLEAN_INTERPRET, and the honest reading
   of a weak victim is "excluded from interpretation", not "very
   vulnerable".

RAW-EVENT MATERIALIZATION (new here; N-MNIST had it, CIFAR/DVS did not)
  Attacked count tensors are converted back into real timestamp-edited
  event streams with (x, y, p) untouched, verified to reproduce the
  attacked tensor bitwise, and exported with their grid anchor t0_us.
  This is what allows physical-realizability discussion and evaluation
  through an external model's own preprocessing.  RAW_EXPORT_N=0 disables.

PROTOCOL (extension of the completed five-model DailyDVS run; the
attack, controls, representation, consumer and attack subset are
reused unchanged, and only the victim registry differs)
--------------------------------------------------------
  T_FINE=80, S_PER_FRAME=8, N_COARSE=10, 64x64 letterbox from 320x240
  five models: coarse_frameformer (consumer) + swin_t, timesformer,
               mvfnet, actionnet (victims)
  victim choice: protocol-scale variants of strong DailyDVS-200 architectures
               with public code/reference results. The attack, controls, seeds,
               budgets and representation remain unchanged; these are not
               claimed to be the released benchmark configurations.
  budgets 1/2/5/10/20% of unique original events
  controls: uniform random + exact signed-displacement-matched (max-flow)
  diagnostics use the SAME progressive optimization path as the main
  attack (ABLATION_PATH / COST_PATH) — the mismatch found in the earlier
  datasets is fixed here from the start
  attack subset: proportional stratified, ATTACK_TOTAL streams, no
  per-class ASR (200 classes x few clean-correct samples is noise)

DATA — fully automatic (download -> verify -> join -> extract -> cache)
  Source: huggingface.co/datasets/Charlieqaq/DailyDVS-200
          9 archive parts (91.2 GB) + per-part .sha256 + all_data_label.json
  The script downloads (resumable), checks every published sha256, joins the
  parts, extracts, then decodes to tensors. Each stage is guarded by an
  on-disk marker, so re-running after an interruption resumes rather than
  repeats. Disk needed: ~91 GB parts + ~91 GB joined + ~95 GB extracted +
  ~7 GB tensor cache. Set DELETE_ARCHIVE_AFTER_EXTRACT=1 to drop the joined
  copy once extraction succeeds, or stage parts on a roomier filesystem via
  DAILYDVS_HF_PARTS.

DATA (paths)
  DAILYDVS_DATA        directory holding the extracted .aedat4 recordings
                       (paths must match the split files, e.g.
                        action_50/C49P2M1S1_20231111_15_48_00.aedat4)
  DAILYDVS_SPLIT_DIR   directory holding the authors' train/val/test.txt
  No download is performed. A preflight verifies the layout and DECODES ONE
  REAL RECORDING before any expensive work starts. (Set DAILYDVS_HF_PARTS or
  DAILYDVS_HF_DOWNLOAD=1 only if you ever need the archive path again.)

RUN
---
  # preprocess + train (first launch also builds the .npz cache)
  DAILYDVS_DATA=/path/to/event_raw DAILYDVS_SPLIT_DIR=/path/to/splits \
  DAILYDVS_RUN=/path/to/run RUN_ATTACKS=0 python3 EVS_DAILYDVS200_SOTA4_REUSE_FINAL.py
  # attacks + diagnostics
  DAILYDVS_DATA=... DAILYDVS_SPLIT_DIR=... DAILYDVS_RUN=... \
  python3 EVS_DAILYDVS200_SOTA4_REUSE_FINAL.py
Chain identical commands with --dependency=afterany; everything resumes.

RUNTIME (one H100, full 200-way, consumer + four strong-architecture variants, three seeds)
  preprocessing  ~4-8 h (22k aedat4 decodes, once)
  training       ~5-8 days
  attacks        ~1-2 days
Use NUM_CLASSES_SUBSET / MAX_TRAIN_PER_CLASS to fit a real budget.
"""

import os, sys, gc, json, math, time, random, shutil
from pathlib import Path
from collections import Counter, defaultdict, deque

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("torch:", torch.__version__, "| device:", DEVICE)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
torch.backends.cudnn.benchmark = True


# ============================================================
# 0. Configuration
# ============================================================
NUM_CLASSES_SUBSET = int(os.environ.get("NUM_CLASSES_SUBSET", "0"))
MAX_TRAIN_PER_CLASS = int(os.environ.get("MAX_TRAIN_PER_CLASS", "0"))
ATTACK_TOTAL = int(os.environ.get("ATTACK_TOTAL", "1000"))

DATA_ROOT = Path(os.environ.get(
    "DAILYDVS_DATA", "/path/to/workspace/EVS_DailyDVS200/data"))
SPLIT_DIR = Path(os.environ.get(
    "DAILYDVS_SPLIT_DIR", str(DATA_ROOT / "splits")))
ROOT = Path(os.environ.get(
    "DAILYDVS_RUN", "/path/to/workspace/EVS_DailyDVS200/run"))
# Cache/output identity encodes every knob that changes the TASK, so a
# 50-class pilot can never leave tensors, manifests, checkpoints or attack
# units that a later 200-class run silently reuses.
_TASK_ID = (f"c{NUM_CLASSES_SUBSET if NUM_CLASSES_SUBSET else 'all'}"
            f"_cap{MAX_TRAIN_PER_CLASS if MAX_TRAIN_PER_CLASS else 'all'}")
TENSOR_ROOT = ROOT / f"tensor_cache_{_TASK_ID}"
CKPT_ROOT = ROOT / f"checkpoints_{_TASK_ID}"
# Keep the previous DailyDVS results immutable. The SOTA-victim extension
# writes to its own result directory while reusing the same tensor cache.
# Deliberate scoped/debug runs are additionally namespaced so they cannot
# overwrite or create checkpoints that a later canonical run would reuse.
BASE_OUT = ROOT / f"results_{_TASK_ID}"
_SCOPED_RUN = os.environ.get("ALLOW_SCOPED_RUN", "0") == "1"
_SCOPED_RUN_NAME = os.environ.get("SCOPED_RUN_NAME", "scoped").strip() or "scoped"
_SCOPED_RUN_NAME = "".join(c if (c.isalnum() or c in "-_") else "_"
                           for c in _SCOPED_RUN_NAME)
OUT = (ROOT / f"results_sota4_scoped_{_SCOPED_RUN_NAME}_{_TASK_ID}"
       if _SCOPED_RUN else ROOT / f"results_sota4_{_TASK_ID}")
for p in [ROOT, TENSOR_ROOT, CKPT_ROOT, OUT]:
    p.mkdir(parents=True, exist_ok=True)
if not DATA_ROOT.exists():
    raise FileNotFoundError(
        f"DailyDVS data root does not exist: {DATA_ROOT}. Set DAILYDVS_DATA.")
if not SPLIT_DIR.exists():
    raise FileNotFoundError(
        f"DailyDVS split directory does not exist: {SPLIT_DIR}. "
        "Set DAILYDVS_SPLIT_DIR.")

_SCOPE_TAG = (f"_scoped-{_SCOPED_RUN_NAME}" if _SCOPED_RUN else "")
PROTOCOL_TAG = (f"dailydvs200_sota4_exact_v3{_SCOPE_TAG}_"
                f"{_TASK_ID}_n{ATTACK_TOTAL}")
DIAG_TAG = f"{PROTOCOL_TAG}_diagv3"

# Reuse the protected consumer and exact attack subset from the last completed
# DailyDVS run. This keeps the new SOTA-victim extension identical to the
# previous experiment except for the victim architecture.
BASE_PROTOCOL_TAG = os.environ.get(
    "DAILYDVS_BASE_PROTOCOL_TAG",
    f"dailydvs200_locked5_exact_v2_{_TASK_ID}_n{ATTACK_TOTAL}")
REUSE_BASE_OBSERVER = os.environ.get("REUSE_BASE_OBSERVER", "1") == "1"
REUSE_BASE_ATTACK_SUBSET = os.environ.get(
    "REUSE_BASE_ATTACK_SUBSET", "1") == "1"
BASE_ATTACK_MANIFEST = Path(os.environ.get(
    "DAILYDVS_BASE_ATTACK_MANIFEST",
    str(BASE_OUT / "attack_subset_manifest.csv")))
BASE_ATTACK_QUOTAS = Path(os.environ.get(
    "DAILYDVS_BASE_ATTACK_QUOTAS",
    str(BASE_OUT / "attack_subset_quotas.csv")))

PREPROCESS_VERSION = "dailydvs_letterbox_int32accum_v2"
EXPECTED_RECORDINGS = 22046

SENSOR_W, SENSOR_H = 320, 240          # DVXplorer Lite
MODEL_H = MODEL_W = 64
T_FINE = 80
S_PER_FRAME = 8
N_COARSE = T_FINE // S_PER_FRAME
assert T_FINE % S_PER_FRAME == 0

NUM_CLASSES_FULL = 200
NUM_CLASSES = NUM_CLASSES_SUBSET if NUM_CLASSES_SUBSET > 0 else NUM_CLASSES_FULL

ATTACK_BUDGETS = [0.01, 0.02, 0.05, 0.10, 0.20]
SHIFT_ABLATION_BUDGET = 0.10
SHIFT_ABLATION_BINS = [1, 2, 4, None]
# Diagnostics walk the SAME progressive path as the main attack.
ABLATION_PATH = [0.01, 0.02, 0.05, 0.10]
COST_BUDGETS = [0.10, 0.20]
COST_PATH = [0.01, 0.02, 0.05, 0.10, 0.20]
COST_SUBSET = int(os.environ.get("COST_SUBSET", "250"))

EVAL_BATCH = int(os.environ.get("EVAL_BATCH", "16"))
MIN_CLEAN_INTERPRET = float(os.environ.get("MIN_CLEAN_INTERPRET", "0.10"))
CLEAN_WARN = float(os.environ.get("CLEAN_WARN", "0.15"))

# Agreement gate for the reused protected consumer, in percentage points.
# Evaluation runs under bfloat16 autocast, which is not bit-deterministic
# between runs, so the same checkpoint on the same cached split moves by a
# few clips (0.1 pp is 4 clips out of 4,093). The gate is set above that
# noise floor and far below any real change of observer. Split/cache
# identity is verified exactly and separately, so this tolerance does not
# weaken the "same observer" guarantee.
CONSUMER_IDENTITY_TOL_PT = float(
    os.environ.get("CONSUMER_IDENTITY_TOL_PT", "0.5"))
CONSUMER_IDENTITY_TOL = CONSUMER_IDENTITY_TOL_PT / 100.0
CONSUMER_IDENTITY_WARN_PT = float(
    os.environ.get("CONSUMER_IDENTITY_WARN_PT", "0.1"))

# The completed DailyDVS run (EVS_DAILYDVS200_HF_FULL*.py lineage)
# hardcoded EVAL_BATCH = 48, while this extension defaults to 16 to fit the
# larger SOTA victims. Under bfloat16 the reduction order depends on the
# batch, so the same consumer weights score slightly differently at a
# different batch size. The identity check therefore re-evaluates the reused
# consumer at the baseline's batch, so it compares like with like instead of
# absorbing a systematic offset into the tolerance.
CONSUMER_EVAL_BATCH = int(os.environ.get("CONSUMER_EVAL_BATCH", "48"))

# SOTA4 registry (internal name): protocol-scale variants of strong
# DailyDVS-200 architectures with public implementations. Published/later
# comparison Top-1 references (context only, not targets for our 64x64
# from-scratch protocol): Swin-T 48.06, TimeSformer 44.25, MVFNet 48.30
# (later comparison; distinct from MVF-Net at 43.98), ACTION-Net 42.61.
OFFICIAL_MODELS = ["coarse_frameformer", "swin_t", "timesformer",
                   "mvfnet", "actionnet"]
TEMPORAL_MODELS = ["swin_t", "timesformer", "mvfnet", "actionnet"]

RUN_SEEDS = [int(s.strip()) for s in os.environ.get("RUN_SEEDS", "0,1,2").split(",") if s.strip()]
if len(set(RUN_SEEDS)) != len(RUN_SEEDS):
    raise ValueError(f"RUN_SEEDS contains duplicates: {RUN_SEEDS}")
DIAGNOSTIC_SEED = 0
TRAIN_MODELS = [x.strip() for x in os.environ.get("TRAIN_MODELS", ",".join(OFFICIAL_MODELS)).split(",") if x.strip()]
ATTACK_MODELS = [x.strip() for x in os.environ.get("ATTACK_MODELS", ",".join(TEMPORAL_MODELS)).split(",") if x.strip()]
ALLOW_SCOPED_RUN = _SCOPED_RUN
FORCE_RETRAIN = os.environ.get("FORCE_RETRAIN", "0") == "1"
FORCE_REBUILD_CACHE = os.environ.get("FORCE_REBUILD_CACHE", "0") == "1"
RUN_ATTACKS = os.environ.get("RUN_ATTACKS", "1") == "1"
RUN_SHIFT_ABLATION = os.environ.get("RUN_SHIFT_ABLATION", "1") == "1"
RUN_COST_OF_STEALTH = os.environ.get("RUN_COST_OF_STEALTH", "1") == "1"

if not ALLOW_SCOPED_RUN:
    if RUN_SEEDS != [0, 1, 2] or ATTACK_TOTAL != 1000 or \
       TRAIN_MODELS != OFFICIAL_MODELS or ATTACK_MODELS != TEMPORAL_MODELS:
        raise SystemExit(
            "official protocol requires RUN_SEEDS=0,1,2 with the full model "
            "registry; set ALLOW_SCOPED_RUN=1 for a deliberate partial run. "
            "Scoped runs use a separate result/checkpoint namespace.")

print("protocol:", PROTOCOL_TAG, "| classes:", NUM_CLASSES,
      "| seeds:", RUN_SEEDS, "| attack total:", ATTACK_TOTAL)
print("base protocol:", BASE_PROTOCOL_TAG,
      "| reuse observer:", REUSE_BASE_OBSERVER,
      "| reuse attack subset:", REUSE_BASE_ATTACK_SUBSET)
print("baseline results dir:", BASE_OUT)
print("new SOTA4 results dir:", OUT)
print("base attack manifest:", BASE_ATTACK_MANIFEST)
print("ablation path:", ABLATION_PATH, "| cost path:", COST_PATH)

if FORCE_REBUILD_CACHE and not FORCE_RETRAIN:
    existing_ckpts = list(CKPT_ROOT.glob("*.pt"))
    if existing_ckpts:
        raise RuntimeError(
            "FORCE_REBUILD_CACHE=1 with existing checkpoints requires "
            "FORCE_RETRAIN=1.")


# ============================================================
# 1. AEDAT4 reader backends
# ============================================================
_BACKEND = None


def _init_backend():
    global _BACKEND
    if _BACKEND is not None:
        return _BACKEND
    try:
        import dv_processing as dvp          # noqa: F401
        _BACKEND = "dv_processing"
    except Exception:
        try:
            import aedat                     # noqa: F401
            _BACKEND = "aedat"
        except Exception:
            try:
                import dv                    # noqa: F401
                _BACKEND = "dv"
            except Exception:
                raise RuntimeError(
                    "no AEDAT4 backend found. Install one of:\n"
                    "  pip install dv-processing   (preferred)\n"
                    "  pip install aedat\n"
                    "  pip install dv")
    print("aedat4 backend:", _BACKEND)
    return _BACKEND


def read_aedat4(path):
    """Return dict of int arrays t (us), x, y, p in {0,1}."""
    backend = _init_backend()
    ts, xs, ys, ps = [], [], [], []

    if backend == "dv_processing":
        import dv_processing as dvp
        reader = dvp.io.MonoCameraRecording(str(path))
        while reader.isRunning():
            batch = reader.getNextEventBatch()
            if batch is None:
                continue
            arr = batch.numpy()               # structured: timestamp,x,y,polarity
            ts.append(arr["timestamp"].astype(np.int64))
            xs.append(arr["x"].astype(np.int16))
            ys.append(arr["y"].astype(np.int16))
            ps.append(arr["polarity"].astype(np.int8))

    elif backend == "aedat":
        import aedat
        decoder = aedat.Decoder(str(path))
        for packet in decoder:
            if "events" not in packet:
                continue
            e = packet["events"]
            ts.append(np.asarray(e["t"], dtype=np.int64))
            xs.append(np.asarray(e["x"], dtype=np.int16))
            ys.append(np.asarray(e["y"], dtype=np.int16))
            ps.append(np.asarray(e["on"], dtype=np.int8))

    else:  # dv
        from dv import AedatFile
        with AedatFile(str(path)) as f:
            for arr in f["events"].numpy():
                ts.append(arr["timestamp"].astype(np.int64))
                xs.append(arr["x"].astype(np.int16))
                ys.append(arr["y"].astype(np.int16))
                ps.append(arr["polarity"].astype(np.int8))

    if not ts:
        raise RuntimeError(f"no events decoded from {path}")
    t = np.concatenate(ts); x = np.concatenate(xs)
    y = np.concatenate(ys); p = np.concatenate(ps)
    order = np.argsort(t, kind="stable")
    t, x, y, p = t[order], x[order], y[order], p[order]
    p = (p > 0).astype(np.int8)
    keep = (x >= 0) & (x < SENSOR_W) & (y >= 0) & (y < SENSOR_H)
    return {"t": t[keep], "x": x[keep], "y": y[keep], "p": p[keep]}


def validate_recording(ev, path, min_events=2000,
                       min_dur_s=0.3, max_dur_s=30.0):
    n = len(ev["t"])
    if n < min_events:
        return False, f"only {n} events"
    dur = (int(ev["t"].max()) - int(ev["t"].min())) / 1e6
    if not (min_dur_s < dur < max_dur_s):
        return False, f"duration {dur:.2f}s"
    if int(ev["x"].max()) >= SENSOR_W or int(ev["y"].max()) >= SENSOR_H:
        return False, "coordinate out of range"
    if not set(np.unique(ev["p"]).tolist()) <= {0, 1}:
        return False, "polarity out of range"
    return True, ""



# ============================================================
# 1b. Hugging Face source preparation
# ------------------------------------------------------------
# The release at huggingface.co/datasets/Charlieqaq/DailyDVS-200 ships
# DailyDvs-200.zip.part-001 ... part-009 (91.2 GB total) plus per-part
# .sha256 files and all_data_label.json. This helper joins, verifies and
# extracts them once; it is a no-op if the recordings are already present.
#   DAILYDVS_HF_PARTS  directory holding the .part-XXX files
#   VERIFY_SHA256      default 1
# ============================================================
HF_REPO = os.environ.get("DAILYDVS_HF_REPO", "Charlieqaq/DailyDVS-200")
HF_PARTS_DIR = os.environ.get("DAILYDVS_HF_PARTS", "")
HF_DOWNLOAD = os.environ.get("DAILYDVS_HF_DOWNLOAD", "0") == "1"
DELETE_ARCHIVE_AFTER_EXTRACT = os.environ.get(
    "DELETE_ARCHIVE_AFTER_EXTRACT", "0") == "1"
# Default assumes the recordings are ALREADY on disk under
# DAILYDVS_DATA. Archive handling only runs if explicitly asked.
VERIFY_SHA256 = os.environ.get("VERIFY_SHA256", "1") == "1"


def _human(n):
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or u == "TB":
            return f"{n:.1f} {u}"
        n /= 1024


def _sha256(path, chunk=1 << 24):
    import hashlib
    h = hashlib.sha256()
    size = path.stat().st_size
    done = 0
    t0 = time.time()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(chunk), b""):
            h.update(blk)
            done += len(blk)
            if done % (1 << 32) < chunk:
                print(f"    hashing {path.name}: {100*done/size:.0f}% "
                      f"({done/max(1e-9, time.time()-t0)/1e6:.0f} MB/s)")
    return h.hexdigest()


def download_from_hf(dest):
    """Fetch the release from huggingface.co/datasets/Charlieqaq/DailyDVS-200.

    The repo stores the recordings as nine split-archive parts
    (DailyDvs-200.zip.part-001 ... part-009, 91.2 GB total) with a .sha256
    beside each, plus all_data_label.json / all_data.json. hf_hub_download
    is resumable, so an interrupted job simply continues; re-running after
    completion costs only a metadata check per file.
    """
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except ImportError:
        raise SystemExit(
            "pip install -U huggingface_hub\n"
            "(a token is only needed if the repo is gated: huggingface-cli login)")
    dest.mkdir(parents=True, exist_ok=True)

    try:
        available = set(list_repo_files(HF_REPO, repo_type="dataset"))
        print(f"  repo listing: {len(available)} files")
    except Exception as e:
        print(f"  WARNING: could not list repo ({e!r}); using expected names")
        available = None

    part_names = [f"DailyDvs-200.zip.part-{i:03d}" for i in range(1, 10)]
    wanted = list(part_names)
    wanted += [f"{n}.sha256" for n in part_names]
    wanted += ["all_data_label.json", "all_data.json"]
    if available is not None:
        extra = sorted(q for q in available
                       if q.startswith("DailyDvs-200.zip.part-")
                       and q not in wanted)
        wanted += extra
        wanted = [q for q in wanted if q in available]
        missing = [q for q in part_names if q not in available]
        if missing:
            print(f"  NOTE: parts absent from the repo listing: {missing}")

    got = []
    for n in wanted:
        try:
            fp = Path(hf_hub_download(repo_id=HF_REPO, filename=n,
                                      repo_type="dataset",
                                      local_dir=str(dest)))
            got.append(fp)
            print(f"  have {n} ({_human(fp.stat().st_size)})")
        except Exception as e:
            lvl = "ERROR" if n in part_names else "note"
            print(f"  {lvl}: could not fetch {n}: {e!r}")
            if n in part_names:
                raise SystemExit(
                    f"required archive part {n} could not be downloaded; "
                    "fix connectivity/auth and rerun (downloads resume)")
    total = sum(q.stat().st_size for q in got if q.suffix != ".sha256")
    print(f"  downloaded/verified {len(got)} files, {_human(total)}")
    return dest


def prepare_dataset():
    """Download -> checksum -> join -> extract, each step skipped if already
    done. Safe to re-run: every stage is guarded by an on-disk marker."""
    if (DATA_ROOT / ".hf_extracted").exists():
        print("  dataset already extracted")
        return
    n_have = (sum(1 for _ in DATA_ROOT.rglob("*.aedat4"))
              if DATA_ROOT.exists() else 0)
    if n_have >= EXPECTED_RECORDINGS:
        print(f"  {n_have} .aedat4 recordings already present; "
              "skipping archive handling")
        (DATA_ROOT / ".hf_extracted").write_text("pre-existing complete tree\n")
        return
    if n_have:
        print(f"  WARNING: {n_have} recordings present but the extraction "
              f"marker is absent (expected {EXPECTED_RECORDINGS}). Treating "
              "the tree as INCOMPLETE and (re)running extraction; existing "
              "files are overwritten in place.")

    parts_dir = Path(HF_PARTS_DIR) if HF_PARTS_DIR else (ROOT / "hf_parts")
    if HF_DOWNLOAD:
        print(f"\n=== downloading {HF_REPO} -> {parts_dir} ===")
        download_from_hf(parts_dir)

    parts = sorted(q for q in parts_dir.rglob("DailyDvs-200.zip.part-*")
                   if q.suffix != ".sha256")
    if not parts:
        raise SystemExit(
            f"no DailyDvs-200.zip.part-* under {parts_dir}. Set "
            "DAILYDVS_HF_DOWNLOAD=1 to fetch them, or stage them yourself.")
    print(f"\n=== preparing archive ({len(parts)} parts) ===")

    if VERIFY_SHA256:
        vmark = parts_dir / ".sha256_ok"
        done_before = set(vmark.read_text().split()) if vmark.exists() else set()
        newly = []
        for q in parts:
            if q.name in done_before:
                continue
            sfp = Path(str(q) + ".sha256")
            if not sfp.exists():
                print(f"  WARNING: no checksum published for {q.name}")
                continue
            want = sfp.read_text().split()[0].strip().lower()
            got = _sha256(q).lower()
            if got != want:
                raise SystemExit(
                    f"CHECKSUM MISMATCH for {q.name}\n  expected {want}\n"
                    f"  got      {got}\nDelete the file and rerun to "
                    "re-download that part.")
            print(f"  sha256 ok: {q.name}")
            newly.append(q.name)
        if newly:
            vmark.write_text("\n".join(sorted(done_before | set(newly))))
    else:
        print("  checksum verification disabled (VERIFY_SHA256=0)")

    joined = parts_dir / "DailyDvs-200.zip"
    if not joined.exists():
        need = sum(q.stat().st_size for q in parts)
        free = shutil.disk_usage(parts_dir).free
        if free < need * 1.05:
            raise SystemExit(
                f"joining needs {_human(need)} free in {parts_dir} but only "
                f"{_human(free)} available. Free space, or point "
                "DAILYDVS_HF_PARTS at a larger filesystem.")
        print(f"  joining -> {joined} ({_human(need)})")
        t0 = time.time()
        with open(joined, "wb") as out:
            for q in parts:
                with open(q, "rb") as fh:
                    for blk in iter(lambda: fh.read(1 << 24), b""):
                        out.write(blk)
                print(f"    appended {q.name}")
        print(f"  joined in {(time.time()-t0)/60:.1f} min")

    import zipfile
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(DATA_ROOT).free
    with zipfile.ZipFile(joined) as z:
        need = sum(i.file_size for i in z.infolist())
        print(f"  archive holds {len(z.infolist())} entries, "
              f"{_human(need)} uncompressed")
        if free < need * 1.05:
            raise SystemExit(
                f"extraction needs {_human(need)} free in {DATA_ROOT} but "
                f"only {_human(free)} available.")
        t0 = time.time()
        z.extractall(DATA_ROOT)
    print(f"  extracted in {(time.time()-t0)/60:.1f} min")
    (DATA_ROOT / ".hf_extracted").write_text(
        f"extracted from {joined} on {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    if DELETE_ARCHIVE_AFTER_EXTRACT:
        joined.unlink()
        print("  removed the joined archive (DELETE_ARCHIVE_AFTER_EXTRACT=1)")

def preflight_local_data():
    """Data is already staged on the cluster: verify it before anything expensive.

    Checks, in order of how fast they fail:
      1. DAILYDVS_DATA exists and contains .aedat4 recordings
      2. the split files are present and parse
      3. a random sample of the paths named in the splits actually resolve
      4. ONE real recording decodes through the chosen AEDAT4 backend, and
         its tensor is built end to end (counts, duration, peak cell count)
    Step 4 is the only test that can catch a broken/missing backend, so it
    runs here rather than 20 hours into preprocessing.
    """
    print("\n=== preflight: local dataset ===")
    if not DATA_ROOT.exists():
        raise SystemExit(f"DAILYDVS_DATA does not exist: {DATA_ROOT}")
    sample = []
    for q in DATA_ROOT.rglob("*.aedat4"):
        sample.append(q)
        if len(sample) >= 5:
            break
    if not sample:
        raise SystemExit(
            f"no .aedat4 recordings found under {DATA_ROOT}. If the archive "
            "is still zipped, set DAILYDVS_HF_PARTS to the parts directory "
            "(or extract it yourself) and rerun.")
    n_files = sum(1 for _ in DATA_ROOT.rglob("*.aedat4"))
    print(f"  recordings on disk: {n_files} (expect {EXPECTED_RECORDINGS})")
    if n_files < EXPECTED_RECORDINGS:
        print("  WARNING: fewer recordings than the published count")

    for sp in ("train", "val", "test"):
        fp = SPLIT_DIR / f"{sp}.txt"
        if not fp.exists():
            raise SystemExit(
                f"{fp} missing. Put the authors' train/val/test.txt in "
                f"DAILYDVS_SPLIT_DIR ({SPLIT_DIR}).")
    print(f"  split files present in {SPLIT_DIR}")

    probe = load_official_split("train")
    miss = 0
    rng = np.random.default_rng(0)
    idx = rng.choice(len(probe), size=min(50, len(probe)), replace=False)
    for j in idx:
        if resolve_path(probe.iloc[int(j)]["rel_path"]) is None:
            miss += 1
    if miss:
        raise SystemExit(
            f"{miss}/50 sampled split entries could not be resolved under "
            f"{DATA_ROOT}. Check that the extracted layout matches the "
            "relative paths in train.txt (e.g. action_50/C49P2M1S1_...aedat4).")
    print("  path resolution: 50/50 sampled split entries found")

    src = resolve_path(probe.iloc[int(idx[0])]["rel_path"])
    print(f"  decoding one recording: {Path(src).name}")
    ev = read_aedat4(src)
    n_ev = len(ev["t"])
    dur_s = (int(ev["t"].max()) - int(ev["t"].min())) / 1e6
    print(f"    events={n_ev} duration={dur_s:.2f}s "
          f"x=[{int(ev['x'].min())},{int(ev['x'].max())}] "
          f"y=[{int(ev['y'].min())},{int(ev['y'].max())}] "
          f"p={sorted(set(np.unique(ev['p']).tolist()))}")
    X, duration_us, peak, t0 = raw_to_tensor(src)
    print(f"    tensor={X.shape} {X.dtype} events={int(X.sum())} "
          f"peak_cell={peak} (int16 limit 32000)")
    if int(X.sum()) != n_ev:
        print("    NOTE: some events fell outside the sensor window and were "
              "dropped by validation")
    print("=== preflight passed ===\n")


# ============================================================
# 1c. Official split reconstruction from the published subject lists
# ------------------------------------------------------------
# The Hugging Face release does NOT ship train/val/test.txt (they are on
# Baidu Netdisk / the GitHub repo), but the dataset card publishes the
# participant IDs of each split verbatim. Since every filename encodes its
# participant (e.g. C0P3M0S1_20231111_09_11_23.aedat4 -> P3), the official
# partition can be reconstructed exactly from all_data_label.json.
#
# The published lists OVERLAP: participant 45 is listed in both train and
# test, participant 4 in both test and validation. We resolve this with one
# deterministic, disclosed rule -- a subject claimed by test or validation is
# never used for training (test > val > train) -- and write the full
# resolution to official_split_reconstruction.json so the paper can state
# precisely what was done.
# ============================================================
OFFICIAL_SUBJECTS = {
    "train": [0, 1, 2, 6, 8, 9, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23,
              25, 26, 28, 29, 30, 32, 34, 35, 36, 38, 39, 40, 44, 45, 46],
    "test": [4, 7, 10, 11, 16, 33, 37, 42, 45],
    "val": [3, 4, 5, 24, 27, 31, 41, 43],
}


def resolve_official_subjects():
    """test > val > train; returns disjoint sets plus the conflict report."""
    test = set(OFFICIAL_SUBJECTS["test"])
    val = set(OFFICIAL_SUBJECTS["val"]) - test
    train = set(OFFICIAL_SUBJECTS["train"]) - test - val
    conflicts = {
        "in_train_and_test": sorted(set(OFFICIAL_SUBJECTS["train"]) & test),
        "in_test_and_val": sorted(set(OFFICIAL_SUBJECTS["test"])
                                  & set(OFFICIAL_SUBJECTS["val"])),
        "in_train_and_val": sorted(set(OFFICIAL_SUBJECTS["train"])
                                   & set(OFFICIAL_SUBJECTS["val"])),
    }
    return {"train": train, "val": val, "test": test}, conflicts


def splits_from_label_json(path):
    """Reconstruct the authors' partition from published subject IDs."""
    with open(path) as f:
        raw = json.load(f)
    rows = []

    def add(rel, lab):
        if rel is None or lab is None:
            return
        rows.append({"rel_path": str(rel), "label": int(lab)})

    if isinstance(raw, dict):
        for k, v in raw.items():
            if isinstance(v, (int, str)) and str(v).lstrip("-").isdigit():
                add(k, int(v))
            elif isinstance(v, dict):
                add(v.get("FilePath", v.get("path", k)),
                    v.get("Action", v.get("label")))
    else:
        for r in raw:
            add(r.get("FilePath", r.get("path", r.get("FileName"))),
                r.get("Action", r.get("label")))
    df = pd.DataFrame(rows)
    if not len(df):
        raise RuntimeError(f"could not parse recordings from {path}")
    df["subject"] = df["rel_path"].map(
        lambda q: (lambda m: int(m.group(1)) if m else -1)(
            SUBJECT_RE.search(Path(q).name)))
    unparsed = int((df["subject"] < 0).sum())
    if unparsed:
        raise RuntimeError(
            f"{unparsed} recordings have no parseable P<id> participant tag; "
            "the official partition cannot be reconstructed from them")

    sets, conflicts = resolve_official_subjects()
    known = sets["train"] | sets["val"] | sets["test"]
    unknown_subs = sorted(set(df["subject"]) - known)
    tr = df[df["subject"].isin(sets["train"])].reset_index(drop=True)
    va = df[df["subject"].isin(sets["val"])].reset_index(drop=True)
    te = df[df["subject"].isin(sets["test"])].reset_index(drop=True)

    report = {
        "source": str(path),
        "published_subject_lists": OFFICIAL_SUBJECTS,
        "published_list_conflicts": conflicts,
        "conflict_rule": "a subject claimed by test or val is never used for "
                         "training (test > val > train)",
        "resolved_subjects": {k: sorted(v) for k, v in sets.items()},
        "subjects_not_in_any_published_list": unknown_subs,
        "n_train": len(tr), "n_val": len(va), "n_test": len(te),
        "n_unassigned_recordings": int(len(df) - len(tr) - len(va) - len(te)),
        "subject_disjoint_after_resolution": True,
    }
    with open(OUT / "official_split_reconstruction.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"  reconstructed official split: {len(tr)} train / {len(va)} val / "
          f"{len(te)} test  (conflicts resolved: {conflicts})")
    if unknown_subs:
        print(f"  NOTE: subjects in no published list, dropped: {unknown_subs}")
    if NUM_CLASSES_SUBSET:
        tr = tr[tr.label < NUM_CLASSES_SUBSET].reset_index(drop=True)
        va = va[va.label < NUM_CLASSES_SUBSET].reset_index(drop=True)
        te = te[te.label < NUM_CLASSES_SUBSET].reset_index(drop=True)
    return tr, va, te, "reconstructed_official_subject_split"


# ============================================================
# 2. Authors' released split + participant-overlap audit
# ============================================================
SUBJECT_RE = __import__("re").compile(r"P(\d+)")


def subjects_of(df):
    """Participant ids parsed from the relative paths (best effort)."""
    out = set()
    for rel in df["rel_path"]:
        m = SUBJECT_RE.search(Path(rel).name)
        if m:
            out.add(int(m.group(1)))
    return out


def audit_split_disjointness(splits):
    """The repository README lists participant ids that appear to overlap
    between splits. Rather than assert a property we have not verified, we
    parse ids from the released files and RECORD what we find; the paper then
    describes the split by evidence, not by assumption."""
    subs = {k: subjects_of(v) for k, v in splits.items()}
    report = {f"n_subjects_{k}": len(v) for k, v in subs.items()}
    overlaps = {}
    keys = sorted(subs)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            inter = sorted(subs[a] & subs[b])
            if inter:
                overlaps[f"{a}&{b}"] = inter
    report["subject_overlaps"] = overlaps
    report["subject_disjoint"] = (len(overlaps) == 0 and
                                  all(len(v) for v in subs.values()))
    with open(OUT / "split_subject_audit.json", "w") as f:
        json.dump({k: (sorted(v) if isinstance(v, set) else v)
                   for k, v in {**report, **{f"subjects_{k}": s
                                             for k, s in subs.items()}}.items()},
                  f, indent=2)
    if not any(len(v) for v in subs.values()):
        print("WARNING: could not parse participant ids from paths; "
              "split described as 'official released split'")
    elif overlaps:
        train_side = {k: v for k, v in overlaps.items() if k.startswith("train")}
        if train_side:
            print(f"WARNING: TRAINING shares participants with evaluation: "
                  f"{train_side}. Do not describe the split as subject-disjoint.")
        else:
            print(f"participant overlap confined to evaluation splits "
                  f"({overlaps}); training is subject-disjoint from both "
                  f"val and test, which is the property that matters.")
    else:
        print("split audit: no participant overlap detected "
              "(subject-disjoint supported by evidence)")
    return report


def load_official_split(split):
    """train.txt / val.txt / test.txt: '<relative path> <action id>'."""
    fp = SPLIT_DIR / f"{split}.txt"
    assert fp.exists(), (
        f"{fp} missing. Download train/val/test.txt from the DailyDVS-200 "
        "repository and point DAILYDVS_SPLIT_DIR at them; we use the "
        "authors' subject-disjoint split rather than inventing one.")
    rows = []
    for lineno, line in enumerate(fp.read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        parts = line.rsplit(None, 1)
        if len(parts) != 2:
            raise RuntimeError(f"{fp}:{lineno}: malformed split line: {line!r}")
        rel, label_text = parts
        try:
            label = int(label_text)
        except ValueError as e:
            raise RuntimeError(
                f"{fp}:{lineno}: non-integer action label {label_text!r}") from e
        if not (0 <= label < NUM_CLASSES_FULL):
            raise RuntimeError(
                f"{fp}:{lineno}: label {label} outside [0,{NUM_CLASSES_FULL-1}]")
        if NUM_CLASSES_SUBSET and label >= NUM_CLASSES_SUBSET:
            continue
        rows.append({"rel_path": rel, "label": label})
    df = pd.DataFrame(rows)
    if not len(df):
        raise RuntimeError(f"no usable rows in {fp}")
    if df["rel_path"].duplicated().any():
        dup = df.loc[df["rel_path"].duplicated(), "rel_path"].iloc[0]
        raise RuntimeError(f"{fp}: duplicate recording path in split: {dup}")
    expected = set(range(NUM_CLASSES))
    got = set(df["label"].astype(int).unique())
    if got != expected:
        raise RuntimeError(
            f"{fp}: selected class coverage mismatch; "
            f"missing={sorted(expected-got)[:20]}, extra={sorted(got-expected)[:20]}")
    return df


def resolve_path(rel):
    cand = DATA_ROOT / rel
    if cand.exists():
        return cand
    name = Path(rel).name
    hits = list(DATA_ROOT.rglob(name))
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        raise RuntimeError(
            f"relative path {rel!r} was not found directly and basename "
            f"{name!r} matched {len(hits)} files; refusing ambiguous fallback")
    return None


# ============================================================
# 3. Tensor cache (letterbox 320x240 -> 64x64, fixed mapping)
# ============================================================
PAD_X = 0
PAD_Y = (SENSOR_W - SENSOR_H) // 2          # 40; letterbox into a 320x320 square


def fine_bin_index(t_us, t0_us, duration_us):
    """THE canonical binning. Used by the cache and by raw materialization so
    the two can never disagree about which fine bin an event belongs to."""
    rel = t_us.astype(np.int64) - int(t0_us)
    return np.clip(np.floor(rel.astype(np.float64) * T_FINE / int(duration_us)
                            ).astype(np.int64), 0, T_FINE - 1)


def map_xy(x_raw, y_raw):
    """Fixed aspect-preserving letterbox; identical for every recording."""
    x = np.minimum(MODEL_W - 1, (x_raw.astype(np.int64) * MODEL_W) // SENSOR_W)
    y = np.minimum(MODEL_H - 1,
                   ((y_raw.astype(np.int64) + PAD_Y) * MODEL_H) // SENSOR_W)
    return x, y


def raw_to_tensor(path):
    ev = read_aedat4(path)
    ok, why = validate_recording(ev, path)
    if not ok:
        raise RuntimeError(f"invalid recording ({why})")
    t_abs = ev["t"].astype(np.int64)
    t0_us = int(t_abs.min())
    t = t_abs - t0_us
    duration_us = max(1, int(t.max()) + 1)
    tb = fine_bin_index(t_abs, t0_us, duration_us)
    # fixed, aspect-preserving mapping identical for every recording
    x, y = map_xy(ev["x"], ev["y"])
    p = ev["p"].astype(np.int64)
    # Accumulate in int32: np.add.at on a uint16 buffer WRAPS SILENTLY, so a
    # post-hoc "< 32000" check would pass on an already-corrupted tensor
    # (70000 increments into one uint16 cell -> 4464). DailyDVS recordings are
    # dense and seconds long, so this is a live risk, not a theoretical one.
    X32 = np.zeros((T_FINE, 2, MODEL_H, MODEL_W), dtype=np.int32)
    np.add.at(X32, (tb, p, y, x), 1)
    peak = int(X32.max())
    if peak >= 32000:
        raise RuntimeError(
            f"cell count {peak} would overflow int16 — the attack pipeline "
            "stores tensors as int16; re-bin (larger T_FINE) or downsample")
    return X32.astype(np.uint16), duration_us, peak, t0_us


PREP_WORKERS = int(os.environ.get("PREP_WORKERS", "8"))


def _prep_one(args):
    """Worker: decode one recording -> cached tensor. Returns a status tuple."""
    i, rel_path, label, outp = args
    outp = Path(outp)
    if outp.exists():
        return ("ok", i, rel_path, label, str(outp), -1)
    src = resolve_path(rel_path)
    if src is None:
        return ("bad", i, rel_path, label, "", "not found")
    try:
        X, duration_us, peak, t0_us = raw_to_tensor(src)
    except Exception as e:
        return ("bad", i, rel_path, label, "", repr(e))
    tmp = outp.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, X=X, label=np.int16(int(label)),
                        duration_us=np.int64(duration_us),
                        t0_us=np.int64(t0_us),
                        peak_cell_count=np.int32(peak),
                        spatial_mapping=np.array(
                            "fixed_letterbox_320x240_to_64x64"),
                        preprocess_version=np.array(PREPROCESS_VERSION),
                        source=np.array(str(src)))
    os.replace(tmp, outp)
    return ("ok", i, rel_path, label, str(outp), peak)


def build_tensor_cache(df, split):
    """Decoding 22k AEDAT4 containers is the wall-clock bottleneck, and it is
    embarrassingly parallel, so it runs in a process pool. Writes are atomic
    (.tmp + os.replace), so an interrupted run never leaves a partial .npz."""
    folder = TENSOR_ROOT / split
    folder.mkdir(parents=True, exist_ok=True)
    df = df.reset_index(drop=True)
    jobs = []
    for i, r in df.iterrows():
        outp = folder / f"{int(r['label']):03d}_{i:06d}.npz"
        if FORCE_REBUILD_CACHE and outp.exists():
            outp.unlink()
        jobs.append((i, r["rel_path"], int(r["label"]), str(outp)))
    rows, bad, peaks = [], [], []
    t0 = time.time()
    if PREP_WORKERS > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        done = 0
        with ProcessPoolExecutor(max_workers=PREP_WORKERS) as ex:
            futs = [ex.submit(_prep_one, j) for j in jobs]
            for fut in as_completed(futs):
                status, i, rel, lab, path, extra = fut.result()
                done += 1
                if status == "ok":
                    rows.append({"path": path, "label": lab})
                    if isinstance(extra, int) and extra >= 0:
                        peaks.append(extra)
                else:
                    bad.append({"rel_path": rel, "reason": extra})
                if done % 500 == 0:
                    rate = done / max(1e-9, time.time() - t0)
                    eta = (len(jobs) - done) / max(1e-9, rate) / 3600
                    print(f"  cache {split}: {done}/{len(jobs)} "
                          f"({rate:.1f}/s, ETA {eta:.1f} h, {len(bad)} bad)")
        rows.sort(key=lambda d: d["path"])
    else:
        for j in jobs:
            status, i, rel, lab, path, extra = _prep_one(j)
            if status == "ok":
                rows.append({"path": path, "label": lab})
                if isinstance(extra, int) and extra >= 0:
                    peaks.append(extra)
            else:
                bad.append({"rel_path": rel, "reason": extra})

    frac_bad = len(bad) / max(1, len(jobs))
    if bad:
        pd.DataFrame(bad).to_csv(OUT / f"bad_files_{split}.csv", index=False)
        print(f"  {split}: skipped {len(bad)} files ({100*frac_bad:.2f}%)")
    if frac_bad > 0.01:
        raise RuntimeError(
            f"{split}: {100*frac_bad:.1f}% of recordings failed to decode — "
            "check the AEDAT4 backend and DAILYDVS_DATA layout before running")
    if peaks:
        print(f"  {split}: peak cell count max={max(peaks)} "
              f"p99={int(np.percentile(peaks, 99))} (int16 limit 32000)")
    res = pd.DataFrame(rows)
    if MAX_TRAIN_PER_CLASS and split == "train":
        res = (res.groupby("label", group_keys=False)
                  .apply(lambda g: g.head(MAX_TRAIN_PER_CLASS))
                  .reset_index(drop=True))
        print(f"  train capped to {MAX_TRAIN_PER_CLASS}/class -> {len(res)}")
    res.to_csv(OUT / f"{split}_tensor_manifest.csv", index=False)
    return res


# Reuse the exact tensor manifests from the completed baseline run. The CSVs
# are tiny metadata files pointing into TENSOR_ROOT; copying them does not
# duplicate or rebuild any event tensors. This also keeps the old result
# directory untouched.
if not FORCE_REBUILD_CACHE:
    for _name in ("train_tensor_manifest.csv", "test_tensor_manifest.csv",
                  "val_tensor_manifest.csv", "split_subject_audit.json",
                  "split_recording_overlap_audit.json", "val_test_overlap.json",
                  "split_train_raw.csv", "split_test_raw.csv"):
        _src = BASE_OUT / _name
        _dst = OUT / _name
        if (not _dst.exists()) and _src.exists():
            shutil.copy2(_src, _dst)
            print(f"reused baseline metadata: {_src.name}")

# Preserve the split provenance from the completed baseline run. When cached
# manifests are reused, the split-building branch below is intentionally skipped,
# so SPLIT_KIND would otherwise be recorded as "unknown".
SPLIT_KIND = "unknown"
_base_run_config = BASE_OUT / "run_config.json"
if not FORCE_REBUILD_CACHE and _base_run_config.exists():
    try:
        with open(_base_run_config) as _f:
            _base_cfg = json.load(_f)
        SPLIT_KIND = str(_base_cfg.get("split_kind", "unknown"))
        print("reused baseline split_kind:", SPLIT_KIND)
    except Exception as _e:
        print(f"WARNING: could not read baseline split metadata: {_e!r}")

if (OUT / "train_tensor_manifest.csv").exists() and not FORCE_REBUILD_CACHE:
    train_manifest = pd.read_csv(OUT / "train_tensor_manifest.csv")
    test_manifest = pd.read_csv(OUT / "test_tensor_manifest.csv")
    _probe = Path(train_manifest.iloc[0]["path"])
    if not _probe.exists():
        raise FileNotFoundError(f"cached tensor missing: {_probe}")
    with np.load(_probe) as _pr:
        _pv = str(_pr["preprocess_version"]) if "preprocess_version" in _pr.files else ""
        _has_anchor = "t0_us" in _pr.files
    if _pv != PREPROCESS_VERSION or not _has_anchor:
        raise RuntimeError(
            "legacy DailyDVS tensor cache detected (wrong preprocess_version "
            "or missing t0_us anchor). The current code fixes uint16 "
            "accumulation-before-check and stores the grid anchor needed for "
            "raw-event export. Rebuild once with FORCE_REBUILD_CACHE=1 "
            "(and FORCE_RETRAIN=1 if checkpoints exist).")
    required = {"path", "label"}
    if not required.issubset(train_manifest.columns) or not required.issubset(test_manifest.columns):
        raise RuntimeError("cached DailyDVS manifests are malformed")
    expected = set(range(NUM_CLASSES))
    if set(train_manifest["label"].astype(int).unique()) != expected or set(test_manifest["label"].astype(int).unique()) != expected:
        raise RuntimeError(
            "cached DailyDVS manifests do not match the requested class set")
    print("tensor caches found:", len(train_manifest), "/", len(test_manifest))
else:
    prepare_dataset()
    preflight_local_data()
    SPLIT_KIND = "official_released_txt"
    if (SPLIT_DIR / "train.txt").exists():
        tr_raw = load_official_split("train")
        te_raw = load_official_split("test")
    else:
        label_json = next(iter([q for q in [DATA_ROOT / "all_data_label.json",
                                            Path(HF_PARTS_DIR or ".") / "all_data_label.json"]
                                if q.exists()]), None)
        if label_json is None:
            raise FileNotFoundError(
                "neither train/test.txt (DAILYDVS_SPLIT_DIR) nor "
                "all_data_label.json were found. The Hugging Face release "
                "ships all_data_label.json; the official partition files "
                "live in the GitHub repository. Provide one of them.")
        print(f"official split files absent; reconstructing the authors' "
              f"partition from published subject IDs + {label_json.name}")
        tr_raw, _va_reconstructed, te_raw, SPLIT_KIND = \
            splits_from_label_json(label_json)
    try:
        va_raw = (load_official_split("val")
                  if SPLIT_KIND == "official_released_txt"
                  else _va_reconstructed)
    except AssertionError:
        va_raw = None
        print("no val.txt found; validation monitoring disabled")
    path_overlap = set(tr_raw["rel_path"]) & set(te_raw["rel_path"])
    if path_overlap:
        raise RuntimeError(
            f"released train/test files overlap by {len(path_overlap)} recordings")
    # The authors' released val.txt and test.txt SHARE participant 4
    # (555 recordings). Training is unaffected -- train.txt is strictly
    # subject-disjoint from both -- but a monitoring split that overlaps the
    # test set is untidy, so the shared recordings are dropped from val and
    # the action is recorded.
    if va_raw is not None and te_raw is not None:
        shared = set(va_raw["rel_path"]) & set(te_raw["rel_path"])
        if shared:
            va_raw = va_raw[~va_raw["rel_path"].isin(shared)].reset_index(drop=True)
            shared_subs = sorted({
                (lambda m: int(m.group(1)) if m else -1)(
                    SUBJECT_RE.search(Path(q).name)) for q in shared})
            print(f"  val/test share {len(shared)} recordings "
                  f"(participants {shared_subs}); dropped from the monitoring "
                  f"split -> val now {len(va_raw)} recordings. Test is "
                  f"unchanged and train remains subject-disjoint from both.")
            with open(OUT / "val_test_overlap.json", "w") as f:
                json.dump({"n_shared_recordings": len(shared),
                           "shared_participants": shared_subs,
                           "action": "removed from val; test unchanged",
                           "train_subject_disjoint_from_eval": True}, f, indent=2)
    SPLIT_AUDIT = audit_split_disjointness(
        {"train": tr_raw, "test": te_raw, **({"val": va_raw} if va_raw is not None else {})})
    with open(OUT / "split_recording_overlap_audit.json", "w") as f:
        json.dump({"train_test_recording_overlap_count": 0}, f, indent=2)
    tr_raw.to_csv(OUT / "split_train_raw.csv", index=False)
    te_raw.to_csv(OUT / "split_test_raw.csv", index=False)
    print("official split:", len(tr_raw), "train /", len(te_raw), "test")
    train_manifest = build_tensor_cache(tr_raw, "train")
    test_manifest = build_tensor_cache(te_raw, "test")
    if va_raw is not None:
        build_tensor_cache(va_raw, "val")

present = sorted(train_manifest["label"].unique().tolist())
print("classes present in train:", len(present))
d0 = np.load(train_manifest.iloc[0]["path"])
print("tensor:", d0["X"].shape, d0["X"].dtype, "| events:", int(d0["X"].sum()),
      "| duration ms:", int(d0["duration_us"]) / 1000)


# ============================================================
# 4. Dataset / loaders
# ============================================================
def augment_event_tensor(X):
    if np.random.rand() < 0.5:
        X = X[..., ::-1].copy()
    dy, dx = np.random.randint(-3, 4), np.random.randint(-3, 4)
    if dx or dy:
        X = np.roll(X, shift=(dy, dx), axis=(-2, -1))
        if dy > 0: X[..., :dy, :] = 0
        elif dy < 0: X[..., dy:, :] = 0
        if dx > 0: X[..., :dx] = 0
        elif dx < 0: X[..., dx:] = 0
    return X


class DailyDVSDataset(Dataset):
    def __init__(self, manifest, train=False):
        self.df = manifest.reset_index(drop=True)
        self.train = train
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        d = np.load(self.df.iloc[idx]["path"])
        X = d["X"].astype(np.float32)
        if self.train:
            X = augment_event_tensor(X)
        return (torch.from_numpy(X),
                torch.tensor(int(d["label"]), dtype=torch.long),
                torch.tensor(int(d["duration_us"]), dtype=torch.long))


train_ds = DailyDVSDataset(train_manifest, train=True)
test_ds = DailyDVSDataset(test_manifest, train=False)
# Official val split is monitored per epoch; test is touched only for the
# final frozen checkpoint. No model selection uses either.
val_manifest = None
if (OUT / "val_tensor_manifest.csv").exists():
    val_manifest = pd.read_csv(OUT / "val_tensor_manifest.csv")
val_ds = DailyDVSDataset(val_manifest, train=False) if val_manifest is not None else None


def make_loader(ds, batch, shuffle):
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=6,
                      pin_memory=True, persistent_workers=True)


test_loader = make_loader(test_ds, EVAL_BATCH, False)
val_loader = make_loader(val_ds, EVAL_BATCH, False) if val_ds is not None else None


def autocast_context():
    if DEVICE.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return torch.autocast(device_type="cpu", enabled=False)


# ============================================================
# 5. Grouping / invariance / attack primitives (validated set)
# ============================================================
def grouped(X):
    return (X.reshape(N_COARSE, S_PER_FRAME, 2, MODEL_H, MODEL_W)
             .transpose(0, 2, 3, 4, 1).reshape(-1, S_PER_FRAME))


def ungrouped(Xg):
    return (Xg.reshape(N_COARSE, 2, MODEL_H, MODEL_W, S_PER_FRAME)
              .transpose(0, 4, 1, 2, 3).reshape(T_FINE, 2, MODEL_H, MODEL_W))


def coarse_np(X):
    return X.reshape(N_COARSE, S_PER_FRAME, 2, MODEL_H, MODEL_W).sum(axis=1)


def coarse_equal(a, b):
    return np.array_equal(coarse_np(a), coarse_np(b))


def batch_coarse_metrics(A, B):
    Ac = A.reshape(len(A), N_COARSE, S_PER_FRAME, 2, MODEL_H, MODEL_W).sum(axis=2)
    Bc = B.reshape(len(B), N_COARSE, S_PER_FRAME, 2, MODEL_H, MODEL_W).sum(axis=2)
    return {"frame_exact_equal": bool(np.array_equal(Ac, Bc)),
            "max_integer_frame_difference":
                int(np.abs(Ac.astype(np.int64) - Bc.astype(np.int64)).max())}


@torch.no_grad()
def predict_logits(model, X_np, batch=EVAL_BATCH):
    model.eval()
    outs = []
    for i in range(0, len(X_np), batch):
        xb = torch.from_numpy(X_np[i:i + batch].astype(np.float32)).to(DEVICE)
        with autocast_context():
            z = model(xb)
        outs.append(z.float().cpu().numpy())
    return np.concatenate(outs)


def _has_rnn(model):
    return any(isinstance(m, nn.RNNBase) for m in model.modules())


def input_gradient(model, X, y):
    model.eval()
    x = torch.from_numpy(X[None].astype(np.float32)).to(DEVICE)
    x.requires_grad_(True)
    yt = torch.tensor([int(y)], device=DEVICE, dtype=torch.long)
    if _has_rnn(model):
        with torch.backends.cudnn.flags(enabled=False):
            g = torch.autograd.grad(F.cross_entropy(model(x), yt), x)[0][0]
    else:
        g = torch.autograd.grad(F.cross_entropy(model(x), yt), x)[0][0]
    return g.detach().cpu().numpy()


def _select_moves(Xg, Gg, Ag, n_moves, S, max_shift_bins=None):
    g_idx, s_idx = np.nonzero(Ag > 0)
    if len(g_idx) == 0:
        return Xg, Ag, np.asarray([], dtype=np.int16)
    s_all = np.arange(S)
    if max_shift_bins is None:
        band = np.ones((S, S), dtype=bool)
    else:
        band = np.abs(s_all[None, :] - s_all[:, None]) <= max_shift_bins
    np.fill_diagonal(band, False)
    Gmask = np.where(band[s_idx], Gg[g_idx], -np.inf)
    dst = Gmask.argmax(axis=1)
    gains = Gmask[np.arange(len(dst)), dst] - Gg[g_idx, s_idx]
    keep = np.isfinite(gains) & (gains > 0)
    g_idx, s_idx, dst, gains = g_idx[keep], s_idx[keep], dst[keep], gains[keep]
    if len(g_idx) == 0:
        return Xg, Ag, np.asarray([], dtype=np.int16)
    order = np.argsort(-gains, kind="stable")
    g_idx, s_idx, dst = g_idx[order], s_idx[order], dst[order]
    caps = Ag[g_idx, s_idx].astype(np.int64)
    csum = np.cumsum(caps)
    take = caps.copy()
    cut = int(np.searchsorted(csum, n_moves))
    if cut < len(caps):
        take[cut] = max(0, n_moves - (csum[cut - 1] if cut > 0 else 0))
        take[cut + 1:] = 0
    Xg[g_idx, s_idx] -= take
    np.add.at(Xg, (g_idx, dst), take)
    Ag[g_idx, s_idx] -= take
    return Xg, Ag, np.repeat((dst - s_idx), take).astype(np.int16)


def best_unique_moves(Xcur, grad, available, n_moves, max_shift_bins=None):
    Xg = grouped(Xcur).astype(np.int32)
    Gg = grouped(grad).astype(np.float32)
    Xg, Ag, shifts = _select_moves(Xg, Gg, available, n_moves,
                                   S_PER_FRAME, max_shift_bins)
    Xnext = ungrouped(Xg).astype(np.int16)
    assert Xnext.min() >= 0
    assert int(Xnext.sum()) == int(Xcur.sum())
    assert coarse_equal(Xcur, Xnext)
    return Xnext, Ag, shifts.astype(np.int8)


def progressive_unique_attack(model, X0, y, budgets, max_shift_bins=None):
    budgets = sorted(float(b) for b in budgets)
    total = int(X0.sum())
    Xcur = X0.copy().astype(np.int16)
    available = grouped(X0).astype(np.int32)
    moved, shifts_all = 0, []
    ck = {}
    for b in budgets:
        need = int(round(b * total)) - moved
        if need > 0:
            grad = input_gradient(model, Xcur, y)
            Xcur, available, sh = best_unique_moves(
                Xcur, grad, available, need, max_shift_bins)
            moved += len(sh)
            shifts_all.extend(sh.tolist())
        ck[b] = {"X": Xcur.copy(), "moved": int(moved),
                 "shifts": np.asarray(shifts_all, dtype=np.int8).copy()}
    return ck


def grouped_free(X):
    return X.reshape(T_FINE, 2 * MODEL_H * MODEL_W).T.copy()


def ungrouped_free(Xg):
    return Xg.T.reshape(T_FINE, 2, MODEL_H, MODEL_W)


def progressive_free_unique_attack(model, X0, y, budgets):
    """Free retiming, one gradient recomputation per checkpoint — identical
    optimization effort to progressive_unique_attack."""
    budgets = sorted(float(b) for b in budgets)
    total = int(X0.sum())
    Xcur = X0.copy().astype(np.int16)
    Ag = grouped_free(X0).astype(np.int32)
    moved, shifts_all = 0, []
    ck = {}
    for b in budgets:
        need = int(round(b * total)) - moved
        if need > 0:
            grad = input_gradient(model, Xcur, y)
            Xg = grouped_free(Xcur).astype(np.int32)
            Gg = grouped_free(grad).astype(np.float32)
            Xg, Ag, sh = _select_moves(Xg, Gg, Ag, need, T_FINE, None)
            Xcur = ungrouped_free(Xg).astype(np.int16)
            moved += len(sh)
            shifts_all.extend(sh.tolist())
        assert Xcur.min() >= 0 and int(Xcur.sum()) == total
        ck[b] = {"X": Xcur.copy(), "moved": int(moved),
                 "shifts": np.asarray(shifts_all, dtype=np.int16).copy()}
    return ck


def event_units_from_original(X0):
    Xg = grouped(X0).astype(np.int32)
    g_idx, s_idx = np.nonzero(Xg > 0)
    counts = Xg[g_idx, s_idx]
    return (np.repeat(g_idx, counts).astype(np.int32),
            np.repeat(s_idx, counts).astype(np.int8))


def _exact_shift_source_flow(source_counts, shift_counts, S, rng):
    """Integral max-flow: allocate each signed shift to feasible source bins.
    Guarantees the adversarial signed-shift multiset is reproduced exactly."""
    shifts = sorted(shift_counts)
    n_sh, n_src = len(shifts), S
    N = n_sh + n_src + 2
    src_node, snk_node = N - 2, N - 1
    cap = np.zeros((N, N), dtype=np.int64)
    for i, d in enumerate(shifts):
        cap[src_node, i] = shift_counts[d]
        for b in range(n_src):
            if 0 <= b + d < S and source_counts[b] > 0:
                cap[i, n_sh + b] = shift_counts[d]
    for b in range(n_src):
        cap[n_sh + b, snk_node] = source_counts[b]
    need = int(sum(shift_counts.values()))
    flow = np.zeros_like(cap)
    while True:
        parent = [-1] * N
        parent[src_node] = src_node
        q = deque([src_node])
        while q and parent[snk_node] == -1:
            u = q.popleft()
            nxt = list(range(N))
            rng.shuffle(nxt)
            for v in nxt:
                if parent[v] == -1 and cap[u, v] - flow[u, v] > 0:
                    parent[v] = u
                    q.append(v)
        if parent[snk_node] == -1:
            break
        v, aug = snk_node, math.inf
        while v != src_node:
            u = parent[v]
            aug = min(aug, cap[u, v] - flow[u, v])
            v = u
        v = snk_node
        while v != src_node:
            u = parent[v]
            flow[u, v] += aug
            flow[v, u] -= aug
            v = u
    if int(flow[src_node, :n_sh].sum()) != need:
        return None
    out = {}
    for i, d in enumerate(shifts):
        for b in range(n_src):
            f = int(flow[i, n_sh + b])
            if f > 0:
                out[(b, d)] = f
    return out


def exact_displacement_matched_random_attack(X0, target_shifts, rng):
    target = np.asarray(target_shifts, dtype=np.int16)
    target = target[target != 0]
    if len(target) == 0:
        return X0.copy().astype(np.int16), np.empty(0, dtype=np.int8), True
    if int(np.abs(target).max()) >= S_PER_FRAME:
        raise ValueError("target shift exceeds the protected coarse window")
    unit_g, unit_s = event_units_from_original(X0)
    source_counts = np.bincount(unit_s.astype(np.int64), minlength=S_PER_FRAME)
    shift_counts = Counter(int(d) for d in target.tolist())
    alloc = _exact_shift_source_flow(source_counts, shift_counts,
                                     S_PER_FRAME, rng)
    if alloc is None:
        raise RuntimeError("no feasible exact displacement matching")
    Xg = grouped(X0).astype(np.int32)
    avail = np.ones(len(unit_g), dtype=bool)
    realized = []
    by_src = defaultdict(list)
    for (b, d), n in alloc.items():
        by_src[b].append((d, n))
    for b, items in by_src.items():
        pool = np.flatnonzero((unit_s == b) & avail)
        rng.shuffle(pool)
        pos = 0
        for d, n in items:
            chosen = pool[pos:pos + n]
            pos += n
            avail[chosen] = False
            gsel = unit_g[chosen]
            ssel = unit_s[chosen].astype(np.int64)
            np.subtract.at(Xg, (gsel, ssel), 1)
            np.add.at(Xg, (gsel, ssel + d), 1)
            realized.extend([d] * len(chosen))
    realized = np.asarray(realized, dtype=np.int8)
    Xatt = ungrouped(Xg).astype(np.int16)
    assert Xatt.min() >= 0
    assert int(Xatt.sum()) == int(X0.sum())
    assert coarse_equal(X0, Xatt)
    assert Counter(realized.tolist()) == Counter(target.astype(np.int8).tolist())
    return Xatt, realized, True


def wilson95(k, n):
    if n == 0:
        return float("nan"), float("nan")
    z, ph = 1.959963984540054, k / n
    den = 1 + z * z / n
    cen = (ph + z * z / (2 * n)) / den
    hw = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return max(0.0, cen - hw), min(1.0, cen + hw)


def attack_metrics(pred, clean_pred, y, clean_correct):
    n_cc = int(clean_correct.sum())
    k = int((pred[clean_correct] != y[clean_correct]).sum()) if n_cc else 0
    lo, hi = wilson95(k, n_cc)
    return {"attacked_accuracy": float((pred == y).mean()),
            "asr_clean_correct": (k / n_cc) if n_cc else float("nan"),
            "asr_ci95_low": lo, "asr_ci95_high": hi,
            "n_clean_correct": n_cc,
            "prediction_flip_rate": float((pred != clean_pred).mean())}



# ============================================================
# 5b. Raw-event materialization  (the N-MNIST bridge)
# ------------------------------------------------------------
# N-MNIST exported attacked COUNT TENSORS back to real timestamp-edited
# event streams; CIFAR10-DVS and DVS128 Gesture stopped at the tensor.
# DailyDVS-200 is a real recorded dataset with absolute microsecond
# timestamps, so the bridge applies here again — and it is what makes
# (a) physical-realizability claims and (b) evaluation through somebody
# else's preprocessing possible at all.
#
# Guarantees, all asserted per sample:
#   * event count preserved exactly
#   * (x, y, p) multiset preserved exactly (only t changes)
#   * every event stays inside its ORIGINAL protected coarse window
#   * rebuilding the tensor from the exported stream reproduces the
#     attacked tensor bit for bit, and the coarse frames are unchanged
#   * clean and attacked streams are saved on the SAME anchor t0, with
#     t0/duration/T_FINE stored, so an auditor can re-derive the grid
#     without guessing (this is the N-MNIST export bug, fixed here).
# ============================================================
RAW_EXPORT_N = int(os.environ.get("RAW_EXPORT_N", "8"))
RAW_EXPORT_BUDGET = float(os.environ.get("RAW_EXPORT_BUDGET", "0.10"))

EVENT_DTYPE = np.dtype([("x", "<i2"), ("y", "<i2"),
                        ("t", "<i8"), ("p", "i1")])


def _ceil_div_nonnegative(a, b):
    """Exact ceil(a/b) for non-negative integer a and positive integer b."""
    return (a + b - 1) // b


def _fine_bin_bounds(bin_idx, duration_us):
    """Inclusive integer timestamp-offset bounds for the canonical binning.

    fine_bin_index uses floor(rel*T_FINE/duration_us). Therefore integer rel
    belongs to bin k iff
        ceil(k*D/T) <= rel <= ceil((k+1)*D/T) - 1.
    Using these exact bounds avoids the old floor(D/T)-width approximation,
    which is wrong whenever duration_us is not divisible by T_FINE.
    """
    k = np.asarray(bin_idx, dtype=np.int64)
    D = int(duration_us)
    lo = (k * D + T_FINE - 1) // T_FINE
    hi = ((k + 1) * D + T_FINE - 1) // T_FINE - 1
    hi = np.minimum(hi, D - 1)
    return lo.astype(np.int64), hi.astype(np.int64)


def materialize_retimed_events(ev, Xatt, t0_us, duration_us):
    """Rewrite original event timestamps so canonical binning yields Xatt.

    Events are matched within (model pixel, polarity, protected coarse window),
    so (x, y, p) are untouched and every event stays inside its original
    protected coarse window. Destination timestamps are chosen with the exact
    integer boundaries implied by fine_bin_index, not by an approximate fixed
    bin width.
    """
    t_abs = ev["t"].astype(np.int64)
    src_bin = fine_bin_index(t_abs, t0_us, duration_us)
    mx, my = map_xy(ev["x"], ev["y"])
    pol = ev["p"].astype(np.int64)

    # cell key on the MODEL grid, and the coarse window each event sits in
    cell = (pol * MODEL_H + my) * MODEL_W + mx
    win = src_bin // S_PER_FRAME
    group = cell * N_COARSE + win

    new_t = t_abs.copy()
    order = np.lexsort((t_abs, group))
    gsorted = group[order]
    starts = np.flatnonzero(np.r_[True, gsorted[1:] != gsorted[:-1]])
    ends = np.r_[starts[1:], len(gsorted)]

    for a, b in zip(starts, ends):
        idx = order[a:b]                      # one (cell, coarse window)
        g = int(gsorted[a])
        c, w = divmod(g, N_COARSE)
        pl, rest = divmod(c, MODEL_H * MODEL_W)
        yy, xx = divmod(rest, MODEL_W)
        lo_bin = w * S_PER_FRAME
        target_counts = Xatt[lo_bin:lo_bin + S_PER_FRAME, pl, yy, xx].astype(np.int64)
        if int(target_counts.sum()) != len(idx):
            raise RuntimeError(
                "materialization mismatch: attacked tensor changes the "
                "per-window count, which the attack must never do")

        target_bins = np.repeat(
            np.arange(lo_bin, lo_bin + S_PER_FRAME, dtype=np.int64),
            target_counts)
        if len(target_bins) != len(idx):
            raise RuntimeError("materialization target count mismatch")

        # idx and target_bins are ordered. Clamp each original relative
        # timestamp to the nearest valid timestamp in its assigned target bin;
        # this minimizes displacement for that monotone assignment.
        rel_orig = t_abs[idx] - int(t0_us)
        lo_rel, hi_rel = _fine_bin_bounds(target_bins, duration_us)
        if np.any(lo_rel > hi_rel):
            raise RuntimeError(
                "empty fine bin encountered during raw-event materialization; "
                "duration_us is too short for T_FINE")
        new_rel = np.minimum(np.maximum(rel_orig, lo_rel), hi_rel)
        new_t[idx] = int(t0_us) + new_rel

        # Local assertion catches any future disagreement with canonical binning.
        got_bins = fine_bin_index(new_t[idx], t0_us, duration_us)
        if not np.array_equal(got_bins, target_bins):
            raise RuntimeError(
                "exact timestamp construction disagrees with fine_bin_index")

    out = np.empty(len(t_abs), dtype=EVENT_DTYPE)
    out["x"] = ev["x"]; out["y"] = ev["y"]; out["p"] = ev["p"]; out["t"] = new_t
    out = out[np.argsort(out["t"], kind="stable")]

    # --- verification against the attacked tensor
    chk = np.zeros((T_FINE, 2, MODEL_H, MODEL_W), dtype=np.int32)
    tb = fine_bin_index(out["t"], t0_us, duration_us)
    cx, cy = map_xy(out["x"], out["y"])
    np.add.at(chk, (tb, out["p"].astype(np.int64), cy, cx), 1)
    if not np.array_equal(chk, Xatt.astype(np.int32)):
        raise RuntimeError("materialized stream does not reproduce Xatt")

    # Every event must remain in its original protected coarse window. Because
    # sorting changes event order, compare the group-level coarse counts rather
    # than event indices. Exact tensor reproduction plus coarse_equal is the
    # authoritative invariant.
    if not coarse_equal(chk.astype(np.int16), Xatt.astype(np.int16)):
        raise RuntimeError("materialized stream violated coarse invariance")
    return out


def export_raw_examples(model, name, seed, Xattack_raw_paths, yattack,
                        budget=RAW_EXPORT_BUDGET, n=RAW_EXPORT_N):
    """Export n clean/attacked RAW EVENT pairs for physical-realizability and
    external-model evaluation. Re-reads the source recordings so the exported
    streams carry true sensor coordinates and microsecond timestamps."""
    outdir = OUT / f"raw_retimed_{DIAG_TAG}_seed{seed}_{name}"
    outdir.mkdir(parents=True, exist_ok=True)
    written = []
    for i in range(min(n, len(Xattack_raw_paths))):
        npz_path, src_path = Xattack_raw_paths[i]
        d = np.load(npz_path)
        X0 = d["X"].astype(np.int16)
        t0_us = int(d["t0_us"]) if "t0_us" in d.files else None
        duration_us = int(d["duration_us"])
        if t0_us is None:
            raise RuntimeError(
                "cached tensor predates the t0_us anchor; rebuild the cache "
                "with FORCE_REBUILD_CACHE=1 before exporting raw streams")
        ck = progressive_unique_attack(model, X0, int(yattack[i]),
                                       [b for b in ATTACK_BUDGETS
                                        if b <= budget])
        Xatt = ck[float(budget)]["X"]
        ev = read_aedat4(src_path)
        clean = np.empty(len(ev["t"]), dtype=EVENT_DTYPE)
        clean["x"] = ev["x"]; clean["y"] = ev["y"]
        clean["p"] = ev["p"]; clean["t"] = ev["t"].astype(np.int64)
        clean = clean[np.argsort(clean["t"], kind="stable")]
        att = materialize_retimed_events(ev, Xatt, t0_us, duration_us)
        assert len(att) == len(clean)
        assert Counter(zip(att["x"].tolist(), att["y"].tolist(),
                           att["p"].tolist())) == \
               Counter(zip(clean["x"].tolist(), clean["y"].tolist(),
                           clean["p"].tolist()))
        fp = outdir / f"raw_retimed_example_{i}.npz"
        np.savez_compressed(
            fp, clean=clean, attacked=att, label=np.int16(int(yattack[i])),
            t0_us=np.int64(t0_us), duration_us=np.int64(duration_us),
            t_fine=np.int16(T_FINE), s_per_frame=np.int16(S_PER_FRAME),
            budget=np.float32(budget),
            sensor_wh=np.array([SENSOR_W, SENSOR_H], dtype=np.int32),
            spatial_mapping=np.array("fixed_letterbox_320x240_to_64x64"),
            note=np.array("clean and attacked share the anchor t0_us; "
                          "bin k = floor((t-t0)*T_FINE/duration)"))
        written.append(str(fp))
    print(f"  raw export: {len(written)} clean/attacked event pairs -> {outdir}")
    return written


# ============================================================
# 6. Models (legacy registry retained + DailyDVS SOTA4 victim registry)
# ============================================================
class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x > 0).to(x.dtype)
    @staticmethod
    def backward(ctx, grad):
        (x,) = ctx.saved_tensors
        return grad / (1.0 + 10.0 * x.abs()) ** 2


spike_fn = SurrogateSpike.apply


def lif_sequence(current, beta=0.90, threshold=1.0):
    mem = torch.zeros_like(current[:, 0])
    spikes = []
    for t in range(current.shape[1]):
        mem = beta * mem + current[:, t]
        s = spike_fn(mem - threshold)
        mem = mem - s * threshold
        spikes.append(s)
    return torch.stack(spikes, dim=1)


def td_apply(module, x):
    B, T = x.shape[:2]
    z = module(x.reshape(B * T, *x.shape[2:]))
    return z.reshape(B, T, *z.shape[1:])


class TDConvBN(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.op = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch))
    def forward(self, x):
        return td_apply(self.op, x)


class ConvSNN(nn.Module):
    def __init__(self, beta=0.90):
        super().__init__()
        self.beta = beta
        self.c1 = TDConvBN(2, 32, 2); self.c2 = TDConvBN(32, 64, 2)
        self.c3 = TDConvBN(64, 128, 2); self.c4 = TDConvBN(128, 256, 2)
        self.head = nn.Linear(256, NUM_CLASSES)
    def forward(self, X):
        x = X * 0.15
        for c in (self.c1, self.c2, self.c3, self.c4):
            x = lif_sequence(c(x), self.beta)
        return self.head(x.mean(dim=(-1, -2)).mean(dim=1))


class SEWBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, beta=0.90):
        super().__init__()
        self.beta = beta
        self.conv1 = TDConvBN(in_ch, out_ch, stride)
        self.conv2 = TDConvBN(out_ch, out_ch, 1)
        self.skip = (nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
            nn.BatchNorm2d(out_ch)) if (stride != 1 or in_ch != out_ch) else None)
    def forward(self, x):
        residual = x
        z = lif_sequence(self.conv1(x), self.beta)
        z = lif_sequence(self.conv2(z), self.beta)
        if self.skip is not None:
            residual = lif_sequence(td_apply(self.skip, residual), self.beta)
        return z + residual


class SEWResNet18(nn.Module):
    def __init__(self, beta=0.90):
        super().__init__()
        self.beta = beta
        self.stem = TDConvBN(2, 32, 2)
        self.s1 = nn.Sequential(SEWBlock(32, 32, 1, beta), SEWBlock(32, 32, 1, beta))
        self.s2 = nn.Sequential(SEWBlock(32, 64, 2, beta), SEWBlock(64, 64, 1, beta))
        self.s3 = nn.Sequential(SEWBlock(64, 128, 2, beta), SEWBlock(128, 128, 1, beta))
        self.s4 = nn.Sequential(SEWBlock(128, 256, 2, beta), SEWBlock(256, 256, 1, beta))
        self.fc = nn.Linear(256, NUM_CLASSES)
    def forward(self, X):
        x = lif_sequence(self.stem(X * 0.15), self.beta)
        x = self.s1(x); x = self.s2(x); x = self.s3(x); x = self.s4(x)
        return self.fc(x.mean(dim=(-1, -2)).mean(dim=1))


class GNResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        g1 = min(16, out_ch)
        while out_ch % g1 != 0:
            g1 -= 1
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(g1, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(g1, out_ch)
        self.skip = (nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False)
                     if (stride != 1 or in_ch != out_ch) else nn.Identity())
    def forward(self, x):
        r = self.skip(x)
        x = F.gelu(self.gn1(self.conv1(x)))
        x = self.gn2(self.conv2(x))
        return F.gelu(x + r)


class EventTemporalTransformerV2(nn.Module):
    def __init__(self, d_model=384, nhead=8, depth=8, mlp_ratio=4, dropout=0.10):
        super().__init__()
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 48, 5, stride=2, padding=2, bias=False),
            nn.GroupNorm(8, 48), nn.GELU(),
            GNResBlock(48, 64, 1), GNResBlock(64, 96, 2),
            GNResBlock(96, 128, 2), GNResBlock(128, 192, 2),
            nn.AdaptiveAvgPool2d(1))
        self.token_proj = nn.Sequential(nn.Linear(192, d_model), nn.LayerNorm(d_model))
        self.temporal_dwconv = nn.Conv1d(d_model, d_model, 5, padding=2,
                                         groups=d_model, bias=False)
        self.temporal_pwconv = nn.Conv1d(d_model, d_model, 1, bias=False)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos = nn.Parameter(torch.zeros(1, T_FINE + 1, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * mlp_ratio,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.out_norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Dropout(0.20),
            nn.Linear(d_model, NUM_CLASSES))
        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.pos, std=0.02)
    def forward(self, X):
        B, T = X.shape[:2]
        z = torch.log1p(X).reshape(B * T, 2, MODEL_H, MODEL_W)
        z = self.spatial(z).flatten(1)
        z = self.token_proj(z).reshape(B, T, -1)
        local = self.temporal_pwconv(F.gelu(self.temporal_dwconv(z.transpose(1, 2))))
        z = z + local.transpose(1, 2)
        z = torch.cat([self.cls.expand(B, -1, -1), z], dim=1)
        z = z + self.pos[:, :T + 1]
        z = self.out_norm(self.encoder(z))
        return self.head(torch.cat([z[:, 0], z[:, 1:].mean(dim=1)], dim=1))


class TemporalGRU_DVS(nn.Module):
    def __init__(self, hidden=256):
        super().__init__()
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 32, 5, stride=2, padding=2), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1))
        self.gru = nn.GRU(128, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, NUM_CLASSES)
    def forward(self, X):
        z = td_apply(self.spatial, X * 0.15).flatten(2)
        _, h = self.gru(z)
        return self.fc(h[-1])


class CoarseSpatialBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.skip = (nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
            nn.BatchNorm2d(out_ch)) if (stride != 1 or in_ch != out_ch)
            else nn.Identity())
    def forward(self, x):
        r = self.skip(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + r)


class CoarseFrameFormer(nn.Module):
    """Consumes ONLY the protected coarse tensor."""
    def __init__(self, d_model=320, nhead=8, depth=4, dropout=0.10):
        super().__init__()
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 64, 5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(),
            CoarseSpatialBlock(64, 64, 1), CoarseSpatialBlock(64, 128, 2),
            CoarseSpatialBlock(128, 192, 2), CoarseSpatialBlock(192, 256, 2),
            nn.AdaptiveAvgPool2d(1))
        self.proj = nn.Linear(256, d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos = nn.Parameter(torch.zeros(1, N_COARSE + 1, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4 * d_model,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(d_model, NUM_CLASSES))
        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.pos, std=0.02)
    def forward(self, X):
        B = X.shape[0]
        coarse = X.reshape(B, N_COARSE, S_PER_FRAME, 2, MODEL_H, MODEL_W).sum(dim=2)
        z = torch.log1p(coarse).reshape(B * N_COARSE, 2, MODEL_H, MODEL_W)
        z = self.spatial(z).flatten(1)
        z = self.proj(z).reshape(B, N_COARSE, -1)
        z = torch.cat([self.cls.expand(B, -1, -1), z], dim=1) + self.pos
        z = self.norm(self.encoder(z))
        return self.head(torch.cat([z[:, 0], z[:, 1:].mean(dim=1)], dim=1))



# --- DailyDVS strong-architecture victims (SOTA4 internal registry) ---------
# Video Swin-T (48.06 Top-1) and TimeSformer (44.25 Top-1) are reported in
# the DailyDVS-200 benchmark. These are protocol-scale adaptations that keep
# the models' defining temporal mechanisms while consuming the same frozen
# 80x2x64x64 representation used in our previous DailyDVS experiments.
# They are trained from scratch here; the published scores are context only,
# not expected reproduction targets.

def _swin3d_slices(length, win, shift):
    if shift > 0:
        return (slice(0, length - win), slice(length - win, length - shift),
                slice(length - shift, length))
    return (slice(0, length),)


def _swin3d_partition(x, w):
    B, D, H, W, C = x.shape
    x = x.reshape(B, D // w[0], w[0], H // w[1], w[1], W // w[2], w[2], C)
    return (x.permute(0, 1, 3, 5, 2, 4, 6, 7)
             .reshape(-1, w[0] * w[1] * w[2], C))


def _swin3d_reverse(win, w, B, D, H, W):
    x = win.reshape(B, D // w[0], H // w[1], W // w[2], w[0], w[1], w[2], -1)
    return (x.permute(0, 1, 4, 2, 5, 3, 6, 7)
             .reshape(B, D, H, W, -1))


def _swin3d_mask(grid, win, shift):
    D, H, W = grid
    img = torch.zeros(1, D, H, W, 1)
    cnt = 0
    for ds in _swin3d_slices(D, win[0], shift[0]):
        for hs in _swin3d_slices(H, win[1], shift[1]):
            for ws in _swin3d_slices(W, win[2], shift[2]):
                img[:, ds, hs, ws, :] = cnt
                cnt += 1
    mw = _swin3d_partition(img, win).squeeze(-1)
    m = mw.unsqueeze(1) - mw.unsqueeze(2)
    return m.masked_fill(m != 0, -100.0).masked_fill(m == 0, 0.0)


class WindowAttention3D(nn.Module):
    """W-MSA over a 3D (T,H,W) window with learned relative position bias."""
    def __init__(self, dim, window, heads):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        wd, wh, ww = window
        self.rpb = nn.Parameter(
            torch.zeros((2 * wd - 1) * (2 * wh - 1) * (2 * ww - 1), heads))
        coords = torch.stack(torch.meshgrid(
            torch.arange(wd), torch.arange(wh), torch.arange(ww),
            indexing="ij")).flatten(1)
        rel = (coords[:, :, None] - coords[:, None, :]).permute(1, 2, 0).contiguous()
        rel[..., 0] += wd - 1
        rel[..., 1] += wh - 1
        rel[..., 2] += ww - 1
        rel[..., 0] *= (2 * wh - 1) * (2 * ww - 1)
        rel[..., 1] *= (2 * ww - 1)
        self.register_buffer("rp_index", rel.sum(-1), persistent=False)
        self.qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.proj = nn.Linear(dim, dim)
        nn.init.trunc_normal_(self.rpb, std=0.02)
    def forward(self, x, mask=None):
        Bn, N, C = x.shape
        qkv = (self.qkv(x).reshape(Bn, N, 3, self.heads, C // self.heads)
                          .permute(2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q * self.scale) @ k.transpose(-2, -1)
        bias = (self.rpb[self.rp_index.reshape(-1)]
                .reshape(N, N, self.heads).permute(2, 0, 1))
        attn = attn + bias.unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            attn = (attn.reshape(Bn // nW, nW, self.heads, N, N)
                    + mask.unsqueeze(1).unsqueeze(0).to(attn.dtype))
            attn = attn.reshape(Bn, self.heads, N, N)
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(Bn, N, C)
        return self.proj(x)


class SwinBlock3D(nn.Module):
    def __init__(self, dim, heads, window, shift, mlp_ratio=4, dropout=0.10):
        super().__init__()
        self.window, self.shift = window, shift
        self.n1 = nn.LayerNorm(dim)
        self.attn = WindowAttention3D(dim, window, heads)
        self.n2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mlp_ratio * dim, dim), nn.Dropout(dropout))
    def forward(self, x, mask):
        B, D, H, W, C = x.shape
        z = self.n1(x)
        if any(self.shift):
            z = torch.roll(z, shifts=tuple(-s for s in self.shift), dims=(1, 2, 3))
        z = _swin3d_reverse(self.attn(_swin3d_partition(z, self.window), mask),
                            self.window, B, D, H, W)
        if any(self.shift):
            z = torch.roll(z, shifts=self.shift, dims=(1, 2, 3))
        x = x + z
        return x + self.mlp(self.n2(x))


class SwinStage3D(nn.Module):
    def __init__(self, dim, depth, heads, grid, window=(8, 4, 4), dropout=0.10):
        super().__init__()
        win = tuple(min(w, g) for w, g in zip(window, grid))
        shift = tuple(w // 2 if g > w else 0 for w, g in zip(win, grid))
        for w, g in zip(win, grid):
            assert g % w == 0, (grid, win)
        self.blocks = nn.ModuleList(
            SwinBlock3D(dim, heads, win,
                        shift if (i % 2 == 1 and any(shift)) else (0, 0, 0),
                        dropout=dropout)
            for i in range(depth))
        if any(shift):
            self.register_buffer("attn_mask", _swin3d_mask(grid, win, shift),
                                 persistent=False)
        else:
            self.attn_mask = None
    def forward(self, x):
        for blk in self.blocks:
            x = blk(x, self.attn_mask if any(blk.shift) else None)
        return x


class PatchMerging3D(nn.Module):
    """Spatial 2x2 merging (temporal length preserved), as in Video Swin."""
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(4 * dim)
        self.reduce = nn.Linear(4 * dim, 2 * dim, bias=False)
    def forward(self, x):
        x = torch.cat([x[:, :, 0::2, 0::2], x[:, :, 1::2, 0::2],
                       x[:, :, 0::2, 1::2], x[:, :, 1::2, 1::2]], dim=-1)
        return self.reduce(self.norm(x))


class VideoSwinDVS(nn.Module):
    """Video Swin-T (Liu et al., CVPR'22) at protocol scale: 3D patch embed
    (2,4,4) on the fine 80-bin grid, 3D shifted-window attention with
    relative position bias, spatial patch merging between stages."""
    def __init__(self, embed=96, depths=(2, 2, 4), heads=(3, 6, 12),
                 window=(8, 4, 4), dropout=0.10):
        super().__init__()
        self.patch = nn.Conv3d(2, embed, (2, 4, 4), stride=(2, 4, 4))
        self.pnorm = nn.LayerNorm(embed)
        D0, H0, W0 = T_FINE // 2, MODEL_H // 4, MODEL_W // 4
        self.stages, self.merges = nn.ModuleList(), nn.ModuleList()
        dim = embed
        for i, (dep, hd) in enumerate(zip(depths, heads)):
            grid = (D0, H0 // (2 ** i), W0 // (2 ** i))
            self.stages.append(SwinStage3D(dim, dep, hd, grid, window, dropout))
            if i < len(depths) - 1:
                self.merges.append(PatchMerging3D(dim))
                dim *= 2
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, NUM_CLASSES)
    def forward(self, X):
        z = torch.log1p(X).permute(0, 2, 1, 3, 4)          # (B,2,T,H,W)
        z = self.patch(z).permute(0, 2, 3, 4, 1)           # (B,D,H',W',C)
        z = self.pnorm(z)
        for i, stage in enumerate(self.stages):
            z = stage(z)
            if i < len(self.merges):
                z = self.merges[i](z)
        z = self.norm(z)
        return self.head(z.mean(dim=(1, 2, 3)))


class DividedSTBlock(nn.Module):
    """Divided space-time attention block (TimeSformer, ICML'21). The
    temporal-attention residual enters through a zero-initialised fc, as in
    the reference implementation."""
    def __init__(self, d, heads, mlp_ratio=4, dropout=0.10):
        super().__init__()
        self.nt = nn.LayerNorm(d)
        self.attn_t = nn.MultiheadAttention(d, heads, dropout=dropout,
                                            batch_first=True)
        self.temporal_fc = nn.Linear(d, d)
        self.ns = nn.LayerNorm(d)
        self.attn_s = nn.MultiheadAttention(d, heads, dropout=dropout,
                                            batch_first=True)
        self.nm = nn.LayerNorm(d)
        self.mlp = nn.Sequential(
            nn.Linear(d, mlp_ratio * d), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mlp_ratio * d, d), nn.Dropout(dropout))
        nn.init.zeros_(self.temporal_fc.weight)
        nn.init.zeros_(self.temporal_fc.bias)
    def forward(self, z, T, N):
        B = z.shape[0]
        zt = (self.nt(z).reshape(B, T, N, -1).transpose(1, 2)
                        .reshape(B * N, T, -1))
        zt = self.attn_t(zt, zt, zt, need_weights=False)[0]
        zt = (zt.reshape(B, N, T, -1).transpose(1, 2).reshape(B, T * N, -1))
        z = z + self.temporal_fc(zt)
        zs = self.ns(z).reshape(B * T, N, -1)
        zs = self.attn_s(zs, zs, zs, need_weights=False)[0].reshape(B, T * N, -1)
        z = z + zs
        return z + self.mlp(self.nm(z))


class TimeSformerDVS(nn.Module):
    """TimeSformer (Bertasius et al., ICML'21) at protocol scale: each fine
    bin is a frame, patchified 8x8 -> 64 spatial tokens x 80 time steps,
    divided space-time attention throughout."""
    def __init__(self, d_model=256, depth=8, heads=8, patch=8, dropout=0.10):
        super().__init__()
        self.n_patch = (MODEL_H // patch) * (MODEL_W // patch)
        self.embed = nn.Conv2d(2, d_model, patch, stride=patch)
        self.pos_sp = nn.Parameter(torch.zeros(1, 1, self.n_patch, d_model))
        self.pos_tm = nn.Parameter(torch.zeros(1, T_FINE, 1, d_model))
        self.blocks = nn.ModuleList(
            DividedSTBlock(d_model, heads, 4, dropout) for _ in range(depth))
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, NUM_CLASSES)
        nn.init.trunc_normal_(self.pos_sp, std=0.02)
        nn.init.trunc_normal_(self.pos_tm, std=0.02)
    def forward(self, X):
        B, T = X.shape[:2]
        z = torch.log1p(X).reshape(B * T, 2, MODEL_H, MODEL_W)
        z = self.embed(z).flatten(2).transpose(1, 2)
        z = z.reshape(B, T, self.n_patch, -1) + self.pos_sp + self.pos_tm[:, :T]
        z = z.reshape(B, T * self.n_patch, -1)
        for blk in self.blocks:
            z = blk(z, T, self.n_patch)
        return self.head(self.norm(z).mean(dim=1))



class MVFModule(nn.Module):
    """Multi-view fusion module (MVFNet, Wu et al., AAAI'21): split channels,
    apply channel-wise 1D convs along the T, H and W axes to the multi-view
    part, fuse by summation into the identity path."""
    def __init__(self, ch, alpha=0.5):
        super().__init__()
        self.cm = max(1, int(ch * alpha))
        cm = self.cm
        self.conv_t = nn.Conv3d(cm, cm, (3, 1, 1), padding=(1, 0, 0),
                                groups=cm, bias=False)
        self.conv_h = nn.Conv3d(cm, cm, (1, 3, 1), padding=(0, 1, 0),
                                groups=cm, bias=False)
        self.conv_w = nn.Conv3d(cm, cm, (1, 1, 3), padding=(0, 0, 1),
                                groups=cm, bias=False)
        self.bn = nn.BatchNorm3d(cm)
    def forward(self, x, T):
        BT, C, H, W = x.shape
        xm, xr = x[:, :self.cm], x[:, self.cm:]
        v = xm.reshape(BT // T, T, self.cm, H, W).transpose(1, 2)
        v = self.bn(self.conv_t(v) + self.conv_h(v) + self.conv_w(v))
        v = v.transpose(1, 2).reshape(BT, self.cm, H, W)
        return torch.cat([xm + v, xr], dim=1)


class MVFBasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, alpha=0.5):
        super().__init__()
        self.mvf = MVFModule(in_ch, alpha)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.skip = (nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
            nn.BatchNorm2d(out_ch)) if (stride != 1 or in_ch != out_ch)
            else nn.Identity())
    def forward(self, x, T):
        x = self.mvf(x, T)
        r = self.skip(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + r)


class MVFNetDVS(nn.Module):
    """MVFNet (Wu et al., AAAI'21) at protocol scale: 2D ResNet trunk over
    the 80 fine bins with an MVF module in every residual block, TSN-style
    average consensus over time."""
    def __init__(self, widths=(48, 96, 192, 384), alpha=0.5, dropout=0.20):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(2, widths[0], 5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(widths[0]), nn.ReLU(inplace=True))
        blocks = []
        in_ch = widths[0]
        for si, w in enumerate(widths):
            stride = 1 if si == 0 else 2
            blocks.append(MVFBasicBlock(in_ch, w, stride, alpha))
            blocks.append(MVFBasicBlock(w, w, 1, alpha))
            in_ch = w
        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Sequential(nn.Dropout(dropout),
                                  nn.Linear(widths[-1], NUM_CLASSES))
    def forward(self, X):
        B, T = X.shape[:2]
        z = torch.log1p(X).reshape(B * T, 2, MODEL_H, MODEL_W)
        z = self.stem(z)
        for blk in self.blocks:
            z = blk(z, T)
        feat = z.mean(dim=(-1, -2)).reshape(B, T, -1).mean(dim=1)
        return self.head(feat)




class ACTIONModule(nn.Module):
    """Protocol-scale ACTION module (Wang et al., CVPR'21).

    The original ACTION block combines three complementary excitation paths:
    spatio-temporal excitation (STE), channel excitation (CE), and motion
    excitation (ME).  This implementation keeps those three mechanisms while
    operating on our fixed fine-bin tensor. Input/output are flattened
    frame features with shape (B*T,C,H,W), and only the victim architecture
    changes; the DailyDVS representation and attack protocol do not.
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(4, channels // reduction)
        # STE: compress channels, apply a 3D spatio-temporal filter, gate.
        self.ste_reduce = nn.Conv2d(channels, 1, 1, bias=False)
        self.ste_conv = nn.Conv3d(1, 1, (3, 3, 3), padding=1, bias=False)
        self.ste_bn = nn.BatchNorm3d(1)
        # CE: spatial squeeze + temporal mixing + channel re-calibration.
        self.ce_reduce = nn.Conv1d(channels, hidden, 1, bias=False)
        self.ce_temporal = nn.Conv1d(hidden, hidden, 3, padding=1,
                                     groups=hidden, bias=False)
        self.ce_expand = nn.Conv1d(hidden, channels, 1, bias=False)
        # ME: reduced-channel temporal difference followed by channel gate.
        self.me_reduce = nn.Conv2d(channels, hidden, 1, bias=False)
        self.me_expand = nn.Conv1d(hidden, channels, 1, bias=False)

    def forward(self, x, T):
        BT, C, H, W = x.shape
        if BT % T != 0:
            raise ValueError(f"ACTIONModule got BT={BT} not divisible by T={T}")
        B = BT // T
        xt = x.reshape(B, T, C, H, W)

        # Spatio-temporal excitation.
        ste = self.ste_reduce(x).reshape(B, T, 1, H, W).permute(0, 2, 1, 3, 4)
        ste = torch.sigmoid(self.ste_bn(self.ste_conv(ste)))
        ste = ste.permute(0, 2, 1, 3, 4)                  # B,T,1,H,W

        # Channel excitation with explicit temporal context.
        ce = xt.mean(dim=(-1, -2)).transpose(1, 2)         # B,C,T
        ce = F.relu(self.ce_reduce(ce), inplace=False)
        ce = F.relu(self.ce_temporal(ce), inplace=False)
        ce = torch.sigmoid(self.ce_expand(ce)).transpose(1, 2)
        ce = ce.unsqueeze(-1).unsqueeze(-1)                # B,T,C,1,1

        # Motion excitation from consecutive feature differences.
        mr = self.me_reduce(x).reshape(B, T, -1, H, W)
        diff = torch.zeros_like(mr)
        diff[:, 1:] = mr[:, 1:] - mr[:, :-1]
        me = diff.abs().mean(dim=(-1, -2)).transpose(1, 2) # B,hidden,T
        me = torch.sigmoid(self.me_expand(me)).transpose(1, 2)
        me = me.unsqueeze(-1).unsqueeze(-1)                # B,T,C,1,1

        # Residual excitation, preserving the 2D backbone's identity path.
        out = xt * (1.0 + ste) * (1.0 + ce) * (1.0 + me)
        return out.reshape(BT, C, H, W)


class ACTIONBasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.action = ACTIONModule(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.skip = (nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
            nn.BatchNorm2d(out_ch)) if (stride != 1 or in_ch != out_ch)
            else nn.Identity())

    def forward(self, x, T):
        r = self.skip(x)
        z = self.action(x, T)
        z = F.relu(self.bn1(self.conv1(z)), inplace=False)
        z = self.bn2(self.conv2(z))
        return F.relu(z + r, inplace=False)


class ACTIONNetDVS(nn.Module):
    """ACTION-Net-style victim at the frozen DailyDVS protocol scale.

    ACTION-Net reports 42.61 Top-1 in the later DailyDVS-200 comparison.
    The public ACTION-Net implementation is ResNet-based; here we keep its
    STE/CE/ME multipath excitation in a compact residual trunk so it can be
    trained and attacked under exactly the same 80-bin, 64x64 protocol as the
    previous DailyDVS victims.
    """
    def __init__(self, widths=(48, 96, 192, 384), dropout=0.20):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(2, widths[0], 5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(widths[0]), nn.ReLU(inplace=True))
        blocks = []
        in_ch = widths[0]
        for si, w in enumerate(widths):
            stride = 1 if si == 0 else 2
            blocks.append(ACTIONBasicBlock(in_ch, w, stride))
            blocks.append(ACTIONBasicBlock(w, w, 1))
            in_ch = w
        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Sequential(nn.Dropout(dropout),
                                  nn.Linear(widths[-1], NUM_CLASSES))

    def forward(self, X):
        B, T = X.shape[:2]
        z = torch.log1p(X).reshape(B * T, 2, MODEL_H, MODEL_W)
        z = self.stem(z)
        for blk in self.blocks:
            z = blk(z, T)
        feat = z.mean(dim=(-1, -2)).reshape(B, T, -1).mean(dim=1)
        return self.head(feat)


# Optimizer/scheduler families transferred unchanged from the validated runs;
# epochs/batches are dataset-scaled (DailyDVS train split is ~2x CIFAR).
MODEL_SPECS = {
    "coarse_frameformer": dict(builder=CoarseFrameFormer, epochs=40, batch=48,
                               lr=4e-4, wd=3e-3, clip=2.0, label_smoothing=0.05,
                               betas=(0.9, 0.95), schedule="warmup_cosine_steps",
                               warmup_frac=0.05),
    "conv_snn": dict(builder=ConvSNN, epochs=35, batch=24, lr=2e-3, wd=1e-4,
                     clip=5.0, label_smoothing=0.0, betas=(0.9, 0.999),
                     schedule="epoch_cosine", warmup_frac=0.0),
    "sew_resnet18": dict(builder=SEWResNet18, epochs=40, batch=12, lr=2e-3,
                         wd=1e-4, clip=5.0, label_smoothing=0.0,
                         betas=(0.9, 0.999), schedule="epoch_cosine",
                         warmup_frac=0.0),
    "event_transformer_v2": dict(builder=EventTemporalTransformerV2, epochs=40,
                                 batch=24, lr=3e-4, wd=5e-3, clip=1.0,
                                 label_smoothing=0.05, betas=(0.9, 0.95),
                                 schedule="warmup_cosine_steps", warmup_frac=0.08),
    "temporal_gru": dict(builder=TemporalGRU_DVS, epochs=25, batch=32, lr=1e-3,
                         wd=1e-4, clip=5.0, label_smoothing=0.0,
                         betas=(0.9, 0.999), schedule="warmup_cosine_steps",
                         warmup_frac=0.02),

    # SOTA4 internal registry: protocol-scale DailyDVS architecture variants.
    # Hyperparameters follow the same optimizer/scheduler families used by the
    # previous run; only the victim architecture is new.
    "swin_t": dict(builder=VideoSwinDVS, epochs=40, batch=16, lr=3e-4,
                   wd=2e-2, clip=1.0, label_smoothing=0.10,
                   betas=(0.9, 0.95), schedule="warmup_cosine_steps",
                   warmup_frac=0.08),
    "timesformer": dict(builder=TimeSformerDVS, epochs=40, batch=16, lr=3e-4,
                        wd=5e-3, clip=1.0, label_smoothing=0.05,
                        betas=(0.9, 0.95), schedule="warmup_cosine_steps",
                        warmup_frac=0.08),
    "mvfnet": dict(builder=MVFNetDVS, epochs=35, batch=16, lr=6e-4,
                   wd=1e-4, clip=5.0, label_smoothing=0.0,
                   betas=(0.9, 0.999), schedule="warmup_cosine_steps",
                   warmup_frac=0.02),
    "actionnet": dict(builder=ACTIONNetDVS, epochs=35, batch=16, lr=6e-4,
                      wd=1e-4, clip=5.0, label_smoothing=0.0,
                      betas=(0.9, 0.999), schedule="warmup_cosine_steps",
                      warmup_frac=0.02),
}

print("\nsmoke test at DailyDVS constants "
      f"(T_FINE={T_FINE}, {NUM_CLASSES} classes):")
smoke_x = torch.zeros(2, T_FINE, 2, MODEL_H, MODEL_W, device=DEVICE)
for name in OFFICIAL_MODELS:
    m = MODEL_SPECS[name]["builder"]().to(DEVICE)
    with torch.no_grad(), autocast_context():
        out = m(smoke_x)
    assert out.shape == (2, NUM_CLASSES), (name, out.shape)
    print(f"  {name}: ok ({sum(p.numel() for p in m.parameters())/1e6:.1f}M)")
    del m
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()


# ============================================================
# 7. Training (resumable, no accuracy gate)
# ============================================================
def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    for X, y, _ in loader:
        X = X.to(DEVICE, non_blocking=True); y = y.to(DEVICE, non_blocking=True)
        with autocast_context():
            logits = model(X)
        correct += int((logits.argmax(1) == y).sum()); total += len(y)
    return correct / total


def warmup_cosine_step(step, total_steps, warmup_steps):
    if step < warmup_steps:
        return max(1e-6, step / max(1, warmup_steps))
    p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, p)))


def epoch_cosine_multiplier(epoch, total_epochs):
    return 0.5 * (1.0 + math.cos(math.pi * epoch / max(1, total_epochs)))


def checkpoint_path(name, seed):
    """Resolve the checkpoint used by this extension.

    The protected CoarseFrameFormer is intentionally reused from the last
    completed DailyDVS experiment so only the victim architecture changes.
    New SOTA victims use the new SOTA4 protocol namespace.
    """
    if name == "coarse_frameformer" and REUSE_BASE_OBSERVER:
        return CKPT_ROOT / f"{name}_seed{seed}_{BASE_PROTOCOL_TAG}.pt"
    return CKPT_ROOT / f"{name}_seed{seed}_{PROTOCOL_TAG}.pt"


def train_model(name, seed, force=False):
    spec = MODEL_SPECS[name]
    set_seed(seed)
    ckpt = checkpoint_path(name, seed)
    part = CKPT_ROOT / f"{name}_seed{seed}_{PROTOCOL_TAG}_partial.pt"
    hist = CKPT_ROOT / f"{name}_seed{seed}_{PROTOCOL_TAG}_history.csv"
    model = spec["builder"]().to(DEVICE)
    if name == "coarse_frameformer" and REUSE_BASE_OBSERVER:
        if not ckpt.exists():
            raise FileNotFoundError(
                f"requested reuse of the previous protected consumer, but "
                f"its checkpoint is missing: {ckpt}. Point "
                f"DAILYDVS_BASE_PROTOCOL_TAG at the completed run, or set "
                f"REUSE_BASE_OBSERVER=0 to deliberately retrain it.")
        print("reusing protected consumer:", ckpt.name)
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        return model
    if ckpt.exists() and not force:
        print("loading final:", ckpt.name)
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        return model
    loader = make_loader(train_ds, spec["batch"], True)
    opt = torch.optim.AdamW(model.parameters(), lr=spec["lr"],
                            weight_decay=spec["wd"], betas=spec["betas"])
    total_steps = spec["epochs"] * len(loader)
    warmup_steps = (int(spec["warmup_frac"] * total_steps)
                    if spec["schedule"] == "warmup_cosine_steps" else 0)
    start_ep, gstep, history = 0, 0, []
    if part.exists() and not force:
        try:
            st = torch.load(part, map_location=DEVICE)
            model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
            start_ep, gstep, history = st["epoch"], st["gstep"], st.get("history", [])
            print(f"resuming {name} seed {seed} at epoch {start_ep}")
        except Exception as e:
            print(f"partial unreadable ({e!r}); from scratch")
    for ep in range(start_ep, spec["epochs"]):
        model.train()
        losses, correct, total = [], 0, 0
        t0 = time.time()
        if spec["schedule"] == "epoch_cosine":
            lr_now = spec["lr"] * epoch_cosine_multiplier(ep, spec["epochs"])
            for pg in opt.param_groups:
                pg["lr"] = lr_now
        for X, y, _ in loader:
            X = X.to(DEVICE, non_blocking=True); y = y.to(DEVICE, non_blocking=True)
            if spec["schedule"] == "warmup_cosine_steps":
                lr_now = spec["lr"] * warmup_cosine_step(gstep, total_steps, warmup_steps)
                for pg in opt.param_groups:
                    pg["lr"] = lr_now
            opt.zero_grad(set_to_none=True)
            with autocast_context():
                logits = model(X)
                loss = F.cross_entropy(logits, y,
                                       label_smoothing=spec["label_smoothing"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), spec["clip"])
            opt.step(); gstep += 1
            losses.append(float(loss.detach().cpu()))
            correct += int((logits.detach().argmax(1) == y).sum()); total += len(y)
        monitor_acc = evaluate(model, val_loader if val_loader is not None
                               else test_loader)
        row = {"epoch": ep + 1, "loss": float(np.mean(losses)),
               "train_acc": correct / total,
               "monitor_split": "val" if val_loader is not None else "test",
               "monitor_acc": monitor_acc,
               "lr": lr_now, "seconds": time.time() - t0}
        history.append(row)
        print(f"{name} seed={seed} ep={ep+1:02d}/{spec['epochs']} "
              f"loss={row['loss']:.4f} train={row['train_acc']:.4f} "
              f"{row['monitor_split']}={monitor_acc:.4f} "
              f"t={row['seconds']:.0f}s")
        pd.DataFrame(history).to_csv(hist, index=False)
        tmp = part.with_suffix(".pt.tmp")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "epoch": ep + 1, "gstep": gstep, "history": history}, tmp)
        os.replace(tmp, part)
    torch.save(model.state_dict(), ckpt)
    if part.exists():
        part.unlink()
    return model


clean_rows = []
for name in TRAIN_MODELS:
    for seed in RUN_SEEDS:
        model = train_model(name, seed, force=FORCE_RETRAIN)
        acc = evaluate(model, test_loader)
        clean_rows.append({"seed": seed, "model": name,
                           "clean_test_accuracy": float(acc),
                           "asr_interpretable": bool(acc >= MIN_CLEAN_INTERPRET)})
        print("FINAL", name, "seed", seed, "acc", round(acc, 4))
        if acc < CLEAN_WARN:
            print(f"*** WARNING ONLY: {name} seed {seed} clean {acc:.3f} < "
                  f"{CLEAN_WARN}. Protocol is frozen: the model is kept and "
                  f"reported. ASR is flagged uninterpretable below "
                  f"{MIN_CLEAN_INTERPRET}.")
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

clean_df = pd.DataFrame(clean_rows)

# Observer-identity check: the reused protected consumer must be the same
# observer, on the same data, as in the completed run. Two things are
# verified, with different strictness, because they have different noise
# floors:
#
#   (1) Checkpoint and split/cache identity -- exact. The consumer weights
#       are hashed, and the test tensor manifest (paths and labels, in
#       order) must hash-match the baseline's. This is what actually
#       catches a wrong checkpoint, a rebuilt split or a different cache,
#       and it is bit-exact, so it stays a hard gate.
#   (2) Clean accuracy agreement -- tolerant. Evaluation runs under
#       bfloat16 autocast, which is not bit-deterministic between runs, so
#       the same weights on the same clips can disagree by a few clips.
#       The gate is CONSUMER_IDENTITY_TOL_PT percentage points; any
#       difference above CONSUMER_IDENTITY_WARN_PT is still printed, so
#       drift stays visible instead of being silently absorbed.
def _manifest_fingerprint(path):
    """Content fingerprint of a tensor manifest: sha256 over the (path, label)
    pairs sorted by path, plus a separate hash of the row order. Content is
    what defines the split; row order differs harmlessly between a parallel
    build (sorted by path) and a serial one (job order), so order is reported
    rather than enforced."""
    import hashlib
    _m = pd.read_csv(path)
    if not {"path", "label"}.issubset(_m.columns):
        raise RuntimeError(f"{path}: manifest is malformed (need path,label)")
    _pairs = [f"{_r}\t{int(_l)}" for _r, _l in
              zip(_m["path"].astype(str), _m["label"])]
    _content = hashlib.sha256("\n".join(sorted(_pairs)).encode("utf-8")).hexdigest()
    _order = hashlib.sha256("\n".join(_pairs).encode("utf-8")).hexdigest()
    return _content, _order, len(_m)


if REUSE_BASE_OBSERVER and "coarse_frameformer" in TRAIN_MODELS:
    _base_clean = BASE_OUT / "clean_accuracy_by_seed.csv"
    if not _base_clean.exists():
        raise FileNotFoundError(
            f"cannot verify the reused consumer: {_base_clean} is missing")

    # (1) exact: the same test clips, from the same cache, as the baseline
    _base_manifest = BASE_OUT / "test_tensor_manifest.csv"
    _this_manifest = OUT / "test_tensor_manifest.csv"
    if _base_manifest.exists() and _this_manifest.exists():
        _bh, _bo, _bn = _manifest_fingerprint(_base_manifest)
        _nh, _no, _nn = _manifest_fingerprint(_this_manifest)
        print(f"test split fingerprint: baseline {_bh[:12]} (n={_bn}) | "
              f"current {_nh[:12]} (n={_nn})")
        if _bh != _nh:
            raise RuntimeError(
                "the reused consumer would be evaluated on a different test "
                f"split or tensor cache than the baseline: {_this_manifest} "
                f"does not match {_base_manifest}. Refusing to attack "
                "against a different observer.")
        if _bo != _no:
            print("*** WARNING ONLY: the test manifest holds the same clips "
                  "as the baseline but in a different row order (a serial vs "
                  "parallel cache build). Content is identical, so the "
                  "observer is unchanged.")
    else:
        _bh = _nh = None
        print("WARNING: baseline test manifest unavailable; split identity "
              "verified only through the accuracy agreement below")

    # (2) tolerant: clean accuracy agreement, seed by seed, at the baseline's
    # evaluation batch so the comparison is like with like
    _b = pd.read_csv(_base_clean)
    _b = _b[_b["model"] == "coarse_frameformer"].set_index("seed")[
        "clean_test_accuracy"]
    _n = clean_df[clean_df["model"] == "coarse_frameformer"].set_index("seed")[
        "clean_test_accuracy"]
    _n_test = int(len(test_manifest))
    _match_batch = CONSUMER_EVAL_BATCH != EVAL_BATCH
    if _match_batch:
        print(f"re-evaluating the reused consumer at the baseline batch "
              f"{CONSUMER_EVAL_BATCH} (this run evaluates at {EVAL_BATCH})")
        _base_batch_loader = make_loader(test_ds, CONSUMER_EVAL_BATCH, False)
    _rows = []
    for _seed in RUN_SEEDS:
        if _seed not in _b.index or _seed not in _n.index:
            raise RuntimeError(f"consumer seed {_seed} missing from the "
                               "baseline or the current evaluation")
        _acc_run = float(_n[_seed])
        if _match_batch:
            _cm = train_model("coarse_frameformer", _seed)
            _acc_cmp = float(evaluate(_cm, _base_batch_loader))
            del _cm
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            _acc_cmp = _acc_run
        _d = abs(float(_b[_seed]) - _acc_cmp)
        _ck = checkpoint_path("coarse_frameformer", _seed)
        _rows.append({"seed": _seed, "baseline_acc": float(_b[_seed]),
                      "reused_acc_at_base_batch": _acc_cmp,
                      "reused_acc_at_run_batch": _acc_run,
                      "base_eval_batch": CONSUMER_EVAL_BATCH,
                      "run_eval_batch": EVAL_BATCH,
                      "abs_diff_pct_points": 100 * _d,
                      "abs_diff_clips": int(round(_d * _n_test)),
                      "tolerance_pct_points": CONSUMER_IDENTITY_TOL_PT,
                      "within_tolerance": bool(_d <= CONSUMER_IDENTITY_TOL + 1e-12),
                      "checkpoint": _ck.name,
                      "checkpoint_sha256": _sha256(_ck) if _ck.exists() else "",
                      "test_manifest_sha256": _nh or "",
                      "baseline_test_manifest_sha256": _bh or ""})
    _chk = pd.DataFrame(_rows)
    _chk.to_csv(OUT / "reused_consumer_identity_check.csv", index=False)
    print("reused consumer identity check "
          f"(tolerance {CONSUMER_IDENTITY_TOL_PT:.2f} pt on {_n_test} clips):")
    print(_chk.drop(columns=["checkpoint_sha256", "test_manifest_sha256",
                             "baseline_test_manifest_sha256"])
              .round(4).to_string(index=False))
    for _r in _rows:
        if (_r["abs_diff_pct_points"] > CONSUMER_IDENTITY_WARN_PT
                and _r["within_tolerance"]):
            print(f"*** WARNING ONLY: consumer seed {_r['seed']} differs from "
                  f"the baseline by {_r['abs_diff_pct_points']:.3f} pt "
                  f"({_r['abs_diff_clips']} clips), above "
                  f"{CONSUMER_IDENTITY_WARN_PT:.2f} pt but within the "
                  f"{CONSUMER_IDENTITY_TOL_PT:.2f} pt tolerance. This is the "
                  "expected scale of bfloat16 evaluation non-determinism.")
    if not _chk["within_tolerance"].all():
        _bad = _chk[~_chk["within_tolerance"]]
        raise RuntimeError(
            "reused CoarseFrameFormer does not reproduce the baseline clean "
            f"accuracy within {CONSUMER_IDENTITY_TOL_PT:.2f} pt "
            f"(worst: seed {int(_bad.iloc[0]['seed'])}, "
            f"{float(_bad.iloc[0]['abs_diff_pct_points']):.3f} pt = "
            f"{int(_bad.iloc[0]['abs_diff_clips'])} clips): wrong checkpoint, "
            "cache, or test split. Refusing to attack against a different "
            "observer. If this is known evaluation noise, raise "
            "CONSUMER_IDENTITY_TOL_PT deliberately and record it.")
clean_df.to_csv(OUT / "clean_accuracy_by_seed.csv", index=False)
print(clean_df.to_string(index=False))
if len(clean_df):
    print(clean_df.groupby("model")["clean_test_accuracy"]
          .agg(["mean", "std"]).round(4).to_string())


# ============================================================
# 8. Attack phase (resumable) + diagnostics
# ============================================================
def proportional_attack_quotas(manifest, total):
    counts = manifest["label"].value_counts().sort_index()
    labels = counts.index.to_numpy()
    avail = counts.to_numpy()
    if total > int(avail.sum()):
        raise ValueError(f"ATTACK_TOTAL={total} exceeds test size {avail.sum()}")
    raw = avail / avail.sum() * total
    q = np.floor(raw).astype(np.int64)
    q = np.minimum(q, avail)
    rem = total - int(q.sum())
    frac_order = np.argsort(-(raw - np.floor(raw)))
    i = 0
    while rem > 0:
        j = frac_order[i % len(frac_order)]
        if q[j] < avail[j]:
            q[j] += 1; rem -= 1
        i += 1
        if i > 50 * len(frac_order):
            raise RuntimeError("quota allocation failed")
    return labels, q


if RUN_ATTACKS:
    if REUSE_BASE_ATTACK_SUBSET:
        if not BASE_ATTACK_MANIFEST.exists():
            raise FileNotFoundError(
                f"requested reuse of the previous attack subset, but the "
                f"manifest is missing: {BASE_ATTACK_MANIFEST}. Point "
                f"DAILYDVS_BASE_ATTACK_MANIFEST at the completed run, or set "
                f"REUSE_BASE_ATTACK_SUBSET=0 to deliberately regenerate it.")
        attack_manifest = pd.read_csv(BASE_ATTACK_MANIFEST).reset_index(drop=True)
        required_cols = {"path", "label"}
        if not required_cols.issubset(attack_manifest.columns):
            raise RuntimeError(
                f"base attack manifest is malformed; expected columns "
                f"{sorted(required_cols)}, got {attack_manifest.columns.tolist()}")
        if len(attack_manifest) != ATTACK_TOTAL:
            raise RuntimeError(
                f"base attack manifest has {len(attack_manifest)} rows but "
                f"ATTACK_TOTAL={ATTACK_TOTAL}; refusing to change the sample set")
        # Verify every reused sample is still in the current cached test split
        # and that its label has not changed.
        test_lookup = {str(r.path): int(r.label)
                       for r in test_manifest[["path", "label"]].itertuples(index=False)}
        missing = []
        mismatched = []
        for r in attack_manifest[["path", "label"]].itertuples(index=False):
            p, lab = str(r.path), int(r.label)
            if p not in test_lookup:
                missing.append(p)
            elif test_lookup[p] != lab:
                mismatched.append((p, lab, test_lookup[p]))
        if missing or mismatched:
            raise RuntimeError(
                f"reused attack subset does not match the current test cache: "
                f"missing={len(missing)}, label_mismatch={len(mismatched)}")
        print(f"reusing exact attack subset: {BASE_ATTACK_MANIFEST} "
              f"({len(attack_manifest)} samples)")
        # Preserve provenance without rewriting the original manifest.
        with open(OUT / f"attack_subset_reuse_{PROTOCOL_TAG}.json", "w") as f:
            json.dump({
                "reused": True,
                "source_manifest": str(BASE_ATTACK_MANIFEST),
                "source_quotas": str(BASE_ATTACK_QUOTAS),
                "n_samples": int(len(attack_manifest)),
                "base_protocol_tag": BASE_PROTOCOL_TAG,
                "new_protocol_tag": PROTOCOL_TAG,
            }, f, indent=2)
    else:
        labels, quotas = proportional_attack_quotas(test_manifest, ATTACK_TOTAL)
        rng = np.random.default_rng(9090)
        parts = []
        for lab, q in zip(labels, quotas):
            if q <= 0:
                continue
            sub = test_manifest[test_manifest["label"] == lab].reset_index(drop=True)
            pick = rng.choice(len(sub), size=int(q), replace=False)
            parts.append(sub.iloc[pick])
        attack_manifest = (pd.concat(parts).sample(frac=1, random_state=9091)
                           .reset_index(drop=True))
        attack_manifest.to_csv(OUT / f"attack_subset_manifest_{PROTOCOL_TAG}.csv",
                               index=False)
        pd.DataFrame({"label": labels, "available":
                      test_manifest["label"].value_counts().sort_index().to_numpy(),
                      "quota": quotas}).to_csv(
            OUT / f"attack_subset_quotas_{PROTOCOL_TAG}.csv", index=False)
        print("generated a NEW attack subset because "
              "REUSE_BASE_ATTACK_SUBSET=0")

    Xattack, yattack, dur_us = [], [], []
    for _, r in attack_manifest.iterrows():
        d = np.load(r["path"])
        Xattack.append(d["X"].astype(np.int16))
        yattack.append(int(d["label"])); dur_us.append(int(d["duration_us"]))
    Xattack = np.stack(Xattack)
    yattack = np.asarray(yattack, np.int64)
    dur_us = np.asarray(dur_us, np.int64)
    bin_ms = dur_us / T_FINE / 1000.0
    print("attack set:", Xattack.shape, "| classes covered:",
          len(np.unique(yattack)))

    UNIT_DIR = OUT / f"attack_partial_{PROTOCOL_TAG}"
    UNIT_DIR.mkdir(parents=True, exist_ok=True)

    def _fp(tag):
        return UNIT_DIR / f"{tag}.json"

    def _load_unit(tag):
        fp = _fp(tag)
        if not fp.exists():
            return None
        try:
            return json.load(open(fp))
        except Exception:
            return None

    def _save_unit(tag, payload):
        tmp = _fp(tag).with_suffix(".json.tmp")
        json.dump(payload, open(tmp, "w"))
        os.replace(tmp, _fp(tag))

    def load_model(name, seed):
        m = MODEL_SPECS[name]["builder"]().to(DEVICE)
        ckpt = checkpoint_path(name, seed)
        if not ckpt.exists():
            raise FileNotFoundError(f"checkpoint missing for {name} seed {seed}: {ckpt}")
        m.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        m.eval()
        return m

    clean_map = {(r["model"], r["seed"]): r["clean_test_accuracy"]
                 for r in clean_rows}
    main_rows, verify_rows = [], []

    for seed in RUN_SEEDS:
        coarse2 = load_model("coarse_frameformer", seed)
        c2_clean = predict_logits(coarse2, Xattack)
        c2_clean_pred = c2_clean.argmax(1)

        for name in ATTACK_MODELS:
            tag = f"unit_seed{seed}_{name}"
            cached = _load_unit(tag)
            if cached is not None:
                print("resume:", tag)
                main_rows += cached["main_rows"]
                verify_rows += cached.get("verify_rows", [])
                continue
            u_main, u_verify = [], []
            model = load_model(name, seed)
            clean_pred = predict_logits(model, Xattack).argmax(1)
            clean_correct = clean_pred == yattack
            full_clean = clean_map[(name, seed)]
            print(f"\n[{name} seed {seed}] subset clean "
                  f"{clean_correct.mean():.4f} | full-test {full_clean:.4f}")

            adv, shf = {b: [] for b in ATTACK_BUDGETS}, {b: [] for b in ATTACK_BUDGETS}
            t0 = time.time()
            for i in range(len(Xattack)):
                ck = progressive_unique_attack(model, Xattack[i], yattack[i],
                                               ATTACK_BUDGETS)
                for b in ATTACK_BUDGETS:
                    adv[b].append(ck[float(b)]["X"])
                    shf[b].append(ck[float(b)]["shifts"])
                if (i + 1) % 100 == 0:
                    print(f"  gradient {i+1}/{len(Xattack)} ({time.time()-t0:.0f}s)")

            for b in ATTACK_BUDGETS:
                b = float(b)
                Xadv = np.stack(adv[b])
                pred = predict_logits(model, Xadv).argmax(1)
                mean_ms = [float(np.abs(s).mean()) * bin_ms[i] if len(s) else 0.0
                           for i, s in enumerate(shf[b])]
                row = {"seed": seed, "model": name, "attack": "gradient_unique",
                       "budget": b, "n_attack": len(Xattack),
                       "clean_accuracy_full_test": full_clean,
                       "asr_interpretable": bool(full_clean >= MIN_CLEAN_INTERPRET),
                       **attack_metrics(pred, clean_pred, yattack, clean_correct),
                       "mean_realized_unique_fraction": float(np.mean(
                           [len(s) / max(1, int(Xattack[i].sum()))
                            for i, s in enumerate(shf[b])])),
                       "mean_abs_shift_ms": float(np.mean(mean_ms)),
                       **batch_coarse_metrics(Xattack, Xadv)}
                c2a = predict_logits(coarse2, Xadv)
                row["coarse_frameformer_logits_exact"] = bool(
                    np.array_equal(c2a, c2_clean))
                row["coarse_frameformer_flip_rate"] = float(
                    (c2a.argmax(1) != c2_clean_pred).mean())
                u_main.append(row)

                # uniform control
                r_u = np.random.default_rng(310000 + seed * 1000 + int(b * 10000))
                Xu = []
                for i in range(len(Xattack)):
                    n_mv = len(shf[b][i])
                    ug, us = event_units_from_original(Xattack[i])
                    n_mv = min(n_mv, len(ug))
                    chosen = r_u.permutation(len(ug))[:n_mv]
                    Xg = grouped(Xattack[i]).astype(np.int32)
                    gs, ss = ug[chosen], us[chosen].astype(np.int64)
                    ds = (ss + r_u.integers(1, S_PER_FRAME, size=n_mv)) % S_PER_FRAME
                    np.subtract.at(Xg, (gs, ss), 1)
                    np.add.at(Xg, (gs, ds), 1)
                    Xr = ungrouped(Xg).astype(np.int16)
                    assert Xr.min() >= 0 and coarse_equal(Xattack[i], Xr)
                    Xu.append(Xr)
                Xu = np.stack(Xu)
                pu = predict_logits(model, Xu).argmax(1)
                c2u = predict_logits(coarse2, Xu)
                u_main.append({"seed": seed, "model": name,
                               "attack": "random_uniform", "budget": b,
                               "n_attack": len(Xattack),
                               "clean_accuracy_full_test": full_clean,
                               "asr_interpretable": bool(full_clean >= MIN_CLEAN_INTERPRET),
                               **attack_metrics(pu, clean_pred, yattack, clean_correct),
                               "mean_realized_unique_fraction": float(np.mean(
                                   [min(len(shf[b][i]), int(Xattack[i].sum()))
                                    / max(1, int(Xattack[i].sum()))
                                    for i in range(len(Xattack))])),
                               **batch_coarse_metrics(Xattack, Xu),
                               "coarse_frameformer_logits_exact": bool(
                                   np.array_equal(c2u, c2_clean)),
                               "coarse_frameformer_flip_rate": float(
                                   (c2u.argmax(1) != c2_clean_pred).mean())})

                # exact displacement-matched control
                r_m = np.random.default_rng(320000 + seed * 1000 + int(b * 10000))
                Xm, n_exact = [], 0
                for i in range(len(Xattack)):
                    xm, sm, ok = exact_displacement_matched_random_attack(
                        Xattack[i], shf[b][i], r_m)
                    n_exact += int(ok)
                    Xm.append(xm)
                Xm = np.stack(Xm)
                pm = predict_logits(model, Xm).argmax(1)
                c2m = predict_logits(coarse2, Xm)
                u_main.append({"seed": seed, "model": name,
                               "attack": "random_displacement_matched_exact",
                               "budget": b, "n_attack": len(Xattack),
                               "clean_accuracy_full_test": full_clean,
                               "asr_interpretable": bool(full_clean >= MIN_CLEAN_INTERPRET),
                               **attack_metrics(pm, clean_pred, yattack, clean_correct),
                               "mean_abs_shift_ms": float(np.mean(
                                   [float(np.abs(s_).mean()) * bin_ms[i] if len(s_) else 0.0
                                    for i, s_ in enumerate(shf[b])])),
                               **batch_coarse_metrics(Xattack, Xm),
                               "exact_match_fraction": float(n_exact / len(Xattack)),
                               "coarse_frameformer_logits_exact": bool(
                                   np.array_equal(c2m, c2_clean)),
                               "coarse_frameformer_flip_rate": float(
                                   (c2m.argmax(1) != c2_clean_pred).mean())})
                u_verify.append({"seed": seed, "model": name, "budget": b,
                                 "samples": len(Xattack),
                                 "signed_histogram_exact": n_exact,
                                 "fallback": len(Xattack) - n_exact})
                print(f"  b={b}: grad {u_main[-3]['asr_clean_correct']:.3f} | "
                      f"unif {u_main[-2]['asr_clean_correct']:.3f} | "
                      f"exact {u_main[-1]['asr_clean_correct']:.3f}")
                del Xadv, Xu, Xm
                gc.collect()

            # N-MNIST-style raw export: a few clean/attacked EVENT pairs
            if RAW_EXPORT_N > 0 and seed == DIAGNOSTIC_SEED:
                try:
                    pairs = []
                    for j in range(min(RAW_EXPORT_N, len(attack_manifest))):
                        npz_p = attack_manifest.iloc[j]["path"]
                        with np.load(npz_p) as dd:
                            src_p = str(dd["source"]) if "source" in dd.files else None
                        if src_p:
                            pairs.append((npz_p, src_p))
                    if pairs:
                        export_raw_examples(model, name, seed, pairs, yattack)
                except Exception as e:
                    print(f"  raw export skipped ({e!r})")

            _save_unit(tag, {"main_rows": u_main, "verify_rows": u_verify})
            main_rows += u_main; verify_rows += u_verify
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        del coarse2
        gc.collect()

    mv = pd.DataFrame(main_rows)
    mv.to_csv(OUT / "main_attack_results_by_seed.csv", index=False)
    pd.DataFrame(verify_rows).to_csv(OUT / "exact_match_verification.csv",
                                     index=False)
    agg = (mv.groupby(["model", "attack", "budget"])
           [["attacked_accuracy", "asr_clean_correct"]].agg(["mean", "std"]))
    agg.to_csv(OUT / "main_attack_aggregate.csv")
    print(agg.round(4).to_string())
    print("frame_exact all:", bool(mv.frame_exact_equal.all()),
          "| maxdiff:", int(mv.max_integer_frame_difference.max()),
          "| consumer logits exact all:",
          bool(mv.coarse_frameformer_logits_exact.all()))

    # ---- max-shift ablation (progressive path, + main-table check)
    if RUN_SHIFT_ABLATION:
        abl_rows = []
        for name in ATTACK_MODELS:
            tag = f"ablation_{DIAG_TAG}_seed{DIAGNOSTIC_SEED}_{name}"
            cached = _load_unit(tag)
            if cached is not None:
                abl_rows += cached["rows"]; continue
            model = load_model(name, DIAGNOSTIC_SEED)
            clean_pred = predict_logits(model, Xattack).argmax(1)
            clean_correct = clean_pred == yattack
            rows = []
            for ms in SHIFT_ABLATION_BINS:
                Xa, sh = [], []
                for i in range(len(Xattack)):
                    ck = progressive_unique_attack(
                        model, Xattack[i], yattack[i], ABLATION_PATH,
                        max_shift_bins=ms)
                    Xa.append(ck[SHIFT_ABLATION_BUDGET]["X"])
                    sh.append(ck[SHIFT_ABLATION_BUDGET]["shifts"])
                Xa = np.stack(Xa)
                pred = predict_logits(model, Xa).argmax(1)
                mean_ms = [float(np.abs(s).mean()) * bin_ms[i] if len(s) else 0.0
                           for i, s in enumerate(sh)]
                rows.append({"seed": DIAGNOSTIC_SEED, "model": name,
                             "budget": SHIFT_ABLATION_BUDGET,
                             "n_attack": len(Xattack),
                             "clean_accuracy_full_test": clean_map[(name, DIAGNOSTIC_SEED)],
                             "asr_interpretable": bool(
                                 clean_map[(name, DIAGNOSTIC_SEED)] >= MIN_CLEAN_INTERPRET),
                             "optimization_path": "->".join(f"{x:g}" for x in ABLATION_PATH),
                             "max_shift_label": "full" if ms is None else str(ms),
                             "mean_abs_shift_ms": float(np.mean(mean_ms)),
                             **attack_metrics(pred, clean_pred, yattack, clean_correct),
                             **batch_coarse_metrics(Xattack, Xa)})
                print(f"  ablation {name} shift={rows[-1]['max_shift_label']}: "
                      f"ASR {rows[-1]['asr_clean_correct']:.3f}")
                del Xa
                gc.collect()
            _save_unit(tag, {"rows": rows})
            abl_rows += rows
            del model
            gc.collect()
        abl_df = pd.DataFrame(abl_rows)
        abl_df.to_csv(OUT / "max_shift_ablation.csv", index=False)
        checks = []
        for name in abl_df["model"].unique():
            old = mv[(mv.model == name) & (mv.seed == DIAGNOSTIC_SEED)
                     & (mv.attack == "gradient_unique")
                     & np.isclose(mv.budget, SHIFT_ABLATION_BUDGET)]
            new = abl_df[(abl_df.model == name) & (abl_df.max_shift_label == "full")]
            if len(old) == 1 and len(new) == 1:
                a = float(old.iloc[0]["asr_clean_correct"])
                bb = float(new.iloc[0]["asr_clean_correct"])
                checks.append({"model": name, "main_10pct_asr": a,
                               "ablation_full_asr": bb,
                               "absolute_difference_pct_points": 100 * abs(a - bb),
                               "consistent_within_0p5pt": bool(abs(a - bb) <= 0.005)})
        if checks:
            cdf = pd.DataFrame(checks)
            cdf.to_csv(OUT / "max_shift_ablation_main_check.csv", index=False)
            print(cdf.round(4).to_string(index=False))

    # ---- cost of stealth (both arms on the same progressive path)
    if RUN_COST_OF_STEALTH:
        cost_rows = []
        n_sub = min(COST_SUBSET, len(Xattack))
        Xs, ys, bms = Xattack[:n_sub], yattack[:n_sub], bin_ms[:n_sub]
        for name in ATTACK_MODELS:
            tag = f"cost_{DIAG_TAG}_seed{DIAGNOSTIC_SEED}_{name}"
            cached = _load_unit(tag)
            if cached is not None:
                cost_rows += cached["rows"]; continue
            model = load_model(name, DIAGNOSTIC_SEED)
            coarse2_l = load_model("coarse_frameformer", DIAGNOSTIC_SEED)
            clean_pred = predict_logits(model, Xs).argmax(1)
            clean_correct = clean_pred == ys
            c2c = predict_logits(coarse2_l, Xs)
            arms = {k: {b: [] for b in COST_BUDGETS} for k in ("stealth", "free")}
            arm_sh = {k: {b: [] for b in COST_BUDGETS} for k in ("stealth", "free")}
            for i in range(n_sub):
                ck_s = progressive_unique_attack(model, Xs[i], ys[i], COST_PATH)
                ck_f = progressive_free_unique_attack(model, Xs[i], ys[i], COST_PATH)
                for b in COST_BUDGETS:
                    arms["stealth"][b].append(ck_s[float(b)]["X"])
                    arm_sh["stealth"][b].append(ck_s[float(b)]["shifts"].astype(np.int16))
                    arms["free"][b].append(ck_f[float(b)]["X"])
                    arm_sh["free"][b].append(ck_f[float(b)]["shifts"])
                if (i + 1) % 50 == 0:
                    print(f"  cost {name}: {i+1}/{n_sub}")
            rows = []
            for b in COST_BUDGETS:
                for kind in ("stealth", "free"):
                    Xa = np.stack(arms[kind][b]); sh = arm_sh[kind][b]
                    pred = predict_logits(model, Xa).argmax(1)
                    c2a = predict_logits(coarse2_l, Xa)
                    mean_ms = [float(np.abs(s).mean()) * bms[i] if len(s) else 0.0
                               for i, s in enumerate(sh)]
                    rows.append({"seed": DIAGNOSTIC_SEED, "model": name,
                                 "kind": kind, "budget": float(b), "n": n_sub,
                                 "clean_accuracy_full_test": clean_map[(name, DIAGNOSTIC_SEED)],
                                 "asr_interpretable": bool(
                                     clean_map[(name, DIAGNOSTIC_SEED)] >= MIN_CLEAN_INTERPRET),
                                 "optimization_path": "->".join(f"{x:g}" for x in COST_PATH),
                                 **attack_metrics(pred, clean_pred, ys, clean_correct),
                                 "mean_abs_shift_ms": float(np.mean(mean_ms)),
                                 **batch_coarse_metrics(Xs, Xa),
                                 "coarse_frameformer_logits_exact": bool(
                                     np.array_equal(c2a, c2c)),
                                 "coarse_frameformer_flip_rate": float(
                                     (c2a.argmax(1) != c2c.argmax(1)).mean())})
                    print(f"  cost {name} {kind} b={b}: acc "
                          f"{rows[-1]['attacked_accuracy']:.3f} maxdiff "
                          f"{rows[-1]['max_integer_frame_difference']}")
                    del Xa
                    gc.collect()
            _save_unit(tag, {"rows": rows})
            cost_rows += rows
            del model, coarse2_l, arms, arm_sh
            gc.collect()
        pd.DataFrame(cost_rows).to_csv(OUT / "cost_of_stealth.csv", index=False)

with open(OUT / "run_config.json", "w") as f:
    json.dump({"dataset": "DailyDVS-200", "protocol_tag": PROTOCOL_TAG,
               "data_source": str(DATA_ROOT), "split_dir": str(SPLIT_DIR),
               "split_kind": SPLIT_KIND,
               "split": "official released split when train/test.txt are supplied; otherwise derived from all_data_label.json and marked as ours. See split_subject_audit.json for the participant-overlap audit.",
               "raw_event_materialization": True,
               "sensor": [SENSOR_W, SENSOR_H], "model_hw": MODEL_H,
               "spatial_mapping": "fixed_letterbox_320x240_to_64x64",
               "preprocess_version": PREPROCESS_VERSION,
               "split_subject_audit": str(OUT / "split_subject_audit.json"),
               "split_recording_overlap_audit": str(
                   OUT / "split_recording_overlap_audit.json"),
               "t_fine": T_FINE, "s_per_frame": S_PER_FRAME, "n_coarse": N_COARSE,
               "num_classes": NUM_CLASSES,
               "num_classes_subset": NUM_CLASSES_SUBSET,
               "max_train_per_class": MAX_TRAIN_PER_CLASS,
               "attack_budgets": ATTACK_BUDGETS, "attack_total": ATTACK_TOTAL,
               "ablation_path": ABLATION_PATH, "cost_path": COST_PATH,
               "per_class_asr_reported": False,
               "min_clean_interpret": MIN_CLEAN_INTERPRET,
               "seeds": RUN_SEEDS, "models": OFFICIAL_MODELS,
               "attack_models": TEMPORAL_MODELS,
               "eval_batch": EVAL_BATCH,
               "scoped_run": ALLOW_SCOPED_RUN,
               "scoped_run_name": _SCOPED_RUN_NAME if ALLOW_SCOPED_RUN else None,
               "reuse_previous_dailyDVS": {
                   "reuse_tensor_cache": True,
                   "base_results_dir": str(BASE_OUT),
                   "new_results_dir": str(OUT),
                   "reuse_base_observer": REUSE_BASE_OBSERVER,
                   "base_protocol_tag": BASE_PROTOCOL_TAG,
                   "consumer_identity_tol_pct_points": CONSUMER_IDENTITY_TOL_PT,
                   "consumer_identity_warn_pct_points": CONSUMER_IDENTITY_WARN_PT,
                   "consumer_eval_batch": CONSUMER_EVAL_BATCH,
                   "reuse_base_attack_subset": REUSE_BASE_ATTACK_SUBSET,
                   "base_attack_manifest": str(BASE_ATTACK_MANIFEST),
                   "base_attack_quotas": str(BASE_ATTACK_QUOTAS),
               },
               "sota4_reference_top1": {
                   "swin_t": 48.06, "timesformer": 44.25,
                   "mvfnet": 48.30, "actionnet": 42.61},
               "sota4_note": "protocol-scale architecture variants trained from scratch; they are not the released benchmark configurations, and published DailyDVS scores are reference context only",
               "aedat4_backend": _BACKEND}, f, indent=2)

print("\nDONE. Outputs under:", OUT)

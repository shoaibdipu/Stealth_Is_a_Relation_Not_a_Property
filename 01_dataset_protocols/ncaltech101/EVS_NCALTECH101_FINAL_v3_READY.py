"""
Event-Security Study — N-Caltech101 (headless cluster run).

PRE-RESULT N-CALTECH101 CONTROLLED REPLICATION — v3
====================================================

This v3 file adapts the frozen CIFAR10-DVS protocol to N-Caltech101 without
looking at N-Caltech101 attack results.  It keeps the attack machinery from
the CIFAR protocol unchanged while fixing the N-Caltech-specific sampling,
spatial mapping, archive-robustness, and interpretation issues before launch.

Run design
----------
* One normal full run over seeds 0,1,2; no seed-0 selection gate.
* Five models, identical architecture families to the CIFAR10-DVS run:
    1) coarse_frameformer       — protected coarse-frame consumer
    2) conv_snn                 — temporal/spiking victim
    3) sew_resnet18             — temporal/spiking victim
    4) event_transformer_v2     — temporal ANN victim
    5) temporal_gru             — temporal recurrent victim
* Four temporal victims are attacked for all three seeds.
* Max-shift and cost-of-stealth remain seed-0 diagnostics.

Dataset / representation
------------------------
* Raw N-Caltech101 40-bit .bin events are parsed directly. Timestamp-overflow
  marker events (y=240) are handled and removed following the standard reader.
* Canonical N-Caltech101 is treated as 101 classes: 100 object classes plus
  BACKGROUND_Google, with the duplicated Faces class excluded. A 102-folder
  archive containing both Faces and Faces_easy is canonicalized automatically
  by dropping Faces. Non-canonical class counts require an explicit override.
* The dataset has no single official train/test split; this protocol uses a
  deterministic stratified 80/20 split with SPLIT_SEED=2027.
* Spatial preprocessing is FIXED across recordings: the 240x180 sensor plane is
  centered in a 240x240 square and uniformly rescaled to 64x64. This preserves
  aspect ratio and avoids the sample-dependent stretch used in v1.
* T_FINE=80; S_PER_FRAME=8; N_COARSE=10, matching the CIFAR10-DVS attack grid.
* The attack subset is ATTACK_TOTAL=1000 recordings, sampled without replacement
  using deterministic class-proportional stratification. It is not forced to be
  class-balanced. Per-class ASR is intentionally not reported because rare
  classes still contain too few attacked clean-correct examples for stable
  classwise estimates.
* Unique-event budgets: 1%, 2%, 5%, 10%, 20%.
* Constrained attacks preserve the coarse integer tensor exactly.
* Uniform-random and strict signed-displacement-matched random controls are
  retained; the latter uses the same exact integral max-flow matcher.
* Main attack rows include the clean-correct denominator and Wilson 95% CI for
  ASR. Models below the predeclared clean-accuracy interpretation floor remain
  in output tables but are flagged as unsuitable for vulnerability inference.

Dataset robustness
------------------
* Every raw recording is validated before splitting. Bad/truncated/outlier files
  are skipped and logged rather than aborting on the first failure.
* The run aborts only if the skipped-file fraction exceeds MAX_BAD_FILE_FRACTION
  (default 1%) or a class becomes unusable.

Training recipes
----------------
The optimizer, regularization, scheduler family, epochs, and batch sizes are
carried unchanged from the frozen CIFAR10-DVS code. This is a deliberately
predeclared transfer, not an assertion that the schedules are optimal for
N-Caltech101. Low clean accuracy is reported; it is not used to retune models
or change attack inclusion after results are observed.

Expected raw layout
-------------------
Either:
    $NCALTECH101_DATA/Caltech101/<class_name>/*.bin
or:
    $NCALTECH101_DATA/<class_name>/*.bin
A direct Caltech101.zip under $NCALTECH101_DATA is also supported and is
extracted once into the run directory.

Normal launch
-------------
  python3 EVS_NCALTECH101_FINAL_v3_READY.py

Training first, attacks later (recommended on a cluster):
  RUN_ATTACKS=0 python3 EVS_NCALTECH101_FINAL_v3_READY.py
  python3 EVS_NCALTECH101_FINAL_v3_READY.py

Environment knobs
-----------------
  NCALTECH101_DATA      dataset root; default
      /path/to/workspace/EVS_NCALTECH101/data
  NCALTECH101_RUN       work root; default
      /path/to/workspace/EVS_NCALTECH101/run
  RUN_SEEDS             default "0,1,2"
  TRAIN_MODELS          full five-model set by default
  ATTACK_MODELS         all four temporal victims by default
  ATTACK_TOTAL          default 1000 (official protocol)
  CLEAN_WARN            default 0.50; warning only
  MIN_CLEAN_INTERPRET   default 0.10; below this ASR is retained but flagged
  MAX_BAD_FILE_FRACTION default 0.01
  ALLOW_NONCANONICAL_CLASS_COUNT default 0
  NCALTECH101_EXCLUDE_CLASSES optional comma list for unusual archives
  RUN_ATTACKS           default 1
  RUN_SHIFT_ABLATION    default 1
  RUN_COST_OF_STEALTH   default 1
  DIAGNOSTIC_SEED       default 0
  FORCE_RETRAIN         default 0
  FORCE_REBUILD_CACHE   default 0; rebuild any legacy tensor cache once
  ALLOW_SCOPED_RUN      default 0; debugging/non-paper runs only
"""

import sys, os
import gc, json, math, time, random, shutil, zipfile
from pathlib import Path
from collections import Counter, deque

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
DATA_ROOT = Path(os.environ.get(
    "NCALTECH101_DATA", "/path/to/workspace/EVS_NCALTECH101/data"))
ROOT = Path(os.environ.get(
    "NCALTECH101_RUN", "/path/to/workspace/EVS_NCALTECH101/run"))
EXTRACT_ROOT = ROOT / "raw_extracted"
# v2 paths are isolated so v1 stretched caches/results can never be reused silently.
TENSOR_ROOT = ROOT / "tensor_cache_letterbox240_v2"
CKPT_ROOT = ROOT / "checkpoints"
OUT = ROOT / "results_v2"
for p in [ROOT, EXTRACT_ROOT, TENSOR_ROOT, CKPT_ROOT, OUT]:
    p.mkdir(parents=True, exist_ok=True)
assert DATA_ROOT.exists(), f"dataset root missing: {DATA_ROOT}"

CANONICAL_NUM_CLASSES = 101
NUM_CLASSES = CANONICAL_NUM_CLASSES  # may change only under explicit non-canonical override
SENSOR_W, SENSOR_H = 240, 180
MODEL_H = MODEL_W = 64

T_FINE = 80
S_PER_FRAME = 8
N_COARSE = T_FINE // S_PER_FRAME
assert T_FINE % S_PER_FRAME == 0

SPLIT_SEED = 2027
TRAIN_FRACTION = 0.80
TEST_FRACTION = 1.0 - TRAIN_FRACTION

ATTACK_BUDGETS = [0.01, 0.02, 0.05, 0.10, 0.20]
ATTACK_TOTAL = int(os.environ.get("ATTACK_TOTAL", "1000"))
MAX_BAD_FILE_FRACTION = float(os.environ.get("MAX_BAD_FILE_FRACTION", "0.01"))
ALLOW_NONCANONICAL_CLASS_COUNT = os.environ.get("ALLOW_NONCANONICAL_CLASS_COUNT", "0") == "1"
EXCLUDE_CLASSES = [x.strip() for x in os.environ.get("NCALTECH101_EXCLUDE_CLASSES", "").split(",") if x.strip()]
MIN_CLEAN_INTERPRET = float(os.environ.get("MIN_CLEAN_INTERPRET", "0.10"))
SHIFT_ABLATION_BUDGET = 0.10
SHIFT_ABLATION_BINS = [1, 2, 4, None]
# Diagnostics must reach their read-out budget along the SAME progressive
# path as the main attack, otherwise they use fewer gradient recomputations
# and are not comparable with the headline table (see DIAGNOSTIC_PATHS note).
ABLATION_PATH = [0.01, 0.02, 0.05, 0.10]
COST_BUDGETS = [0.10, 0.20]
COST_PATH = [0.01, 0.02, 0.05, 0.10, 0.20]
COST_SUBSET = int(os.environ.get("COST_SUBSET", "250"))
EVAL_BATCH = 48

RUN_SEEDS = [int(x.strip()) for x in os.environ.get("RUN_SEEDS", "0,1,2").split(",") if x.strip()]
if len(set(RUN_SEEDS)) != len(RUN_SEEDS):
    raise ValueError(f"RUN_SEEDS contains duplicates: {RUN_SEEDS}")

OFFICIAL_MODELS = [
    "coarse_frameformer",
    "conv_snn",
    "sew_resnet18",
    "event_transformer_v2",
    "temporal_gru",
]
TEMPORAL_MODELS = [
    "conv_snn",
    "sew_resnet18",
    "event_transformer_v2",
    "temporal_gru",
]
DEFAULT_TRAIN = ",".join(OFFICIAL_MODELS)
DEFAULT_ATTACK = ",".join(TEMPORAL_MODELS)
TRAIN_MODELS = [x.strip() for x in os.environ.get("TRAIN_MODELS", DEFAULT_TRAIN).split(",") if x.strip()]
ATTACK_MODELS = [x.strip() for x in os.environ.get("ATTACK_MODELS", DEFAULT_ATTACK).split(",") if x.strip()]

PROTOCOL_TAG_BASE = "ncaltech101_3seed_cifarlocked5_exact_prop1000_letterbox_v2"
CACHE_TAG = "ncaltech101_letterbox240_t80_v2"
# Final protocol tag is suffixed with the realized class count after class-map resolution.
PROTOCOL_TAG = None
CLEAN_WARN = float(os.environ.get("CLEAN_WARN", "0.50"))
DIAGNOSTIC_SEED = int(os.environ.get("DIAGNOSTIC_SEED", "0"))
FORCE_RETRAIN = os.environ.get("FORCE_RETRAIN", "0") == "1"
FORCE_REBUILD_CACHE = os.environ.get("FORCE_REBUILD_CACHE", "0") == "1"
ALLOW_SCOPED_RUN = os.environ.get("ALLOW_SCOPED_RUN", "0") == "1"
RUN_ATTACKS = os.environ.get("RUN_ATTACKS", "1") == "1"
RUN_SHIFT_ABLATION = os.environ.get("RUN_SHIFT_ABLATION", "1") == "1"
RUN_COST_OF_STEALTH = os.environ.get("RUN_COST_OF_STEALTH", "1") == "1"

print("RUN_SEEDS:", RUN_SEEDS)
print("TRAIN_MODELS:", TRAIN_MODELS)
print("ATTACK_MODELS:", ATTACK_MODELS)
print("CLEAN_WARN (warning only):", CLEAN_WARN)
print("DIAGNOSTIC_SEED:", DIAGNOSTIC_SEED)
print("ALLOW_SCOPED_RUN:", ALLOW_SCOPED_RUN)
print("ATTACK_TOTAL:", ATTACK_TOTAL, "| budgets:", ATTACK_BUDGETS)
print("MAX_BAD_FILE_FRACTION:", MAX_BAD_FILE_FRACTION)
print("MIN_CLEAN_INTERPRET:", MIN_CLEAN_INTERPRET)
print("ALLOW_NONCANONICAL_CLASS_COUNT:", ALLOW_NONCANONICAL_CLASS_COUNT)

if FORCE_REBUILD_CACHE and not FORCE_RETRAIN:
    existing_ckpts = list(CKPT_ROOT.glob("*.pt"))
    if existing_ckpts:
        raise RuntimeError(
            "FORCE_REBUILD_CACHE=1 with existing checkpoints is unsafe unless "
            "FORCE_RETRAIN=1 as well.")

# ============================================================
# 1. Locate/extract N-Caltech101
# ============================================================
def _class_dirs_under(root):
    if not root.exists() or not root.is_dir():
        return []
    out = []
    for d in sorted([p for p in root.iterdir() if p.is_dir()]):
        if any(d.rglob("*.bin")):
            out.append(d)
    return out


def _plausible_event_root(root):
    """Accept common 100/101/102-folder variants; class policy is applied later."""
    dirs = _class_dirs_under(root)
    return dirs if 100 <= len(dirs) <= 102 else []


def resolve_event_root():
    candidates = [
        DATA_ROOT / "Caltech101",
        DATA_ROOT / "N-Caltech101" / "Caltech101",
        DATA_ROOT / "NCaltech101" / "Caltech101",
        DATA_ROOT,
    ]
    for cand in candidates:
        dirs = _plausible_event_root(cand)
        if dirs:
            print("event root:", cand, f"({len(dirs)} raw class folders)")
            return cand

    zips = [DATA_ROOT / "Caltech101.zip", DATA_ROOT / "N-Caltech101-archive.zip"]
    zips += sorted(DATA_ROOT.glob("*Caltech101*.zip"))
    seen = set()
    for zp in zips:
        if not zp.exists() or zp.resolve() in seen:
            continue
        seen.add(zp.resolve())
        target = EXTRACT_ROOT / zp.stem
        marker = target / ".extracted_ok"
        if not marker.exists():
            target.mkdir(parents=True, exist_ok=True)
            print("extracting", zp, "->", target)
            with zipfile.ZipFile(zp, "r") as zf:
                zf.extractall(target)
            marker.touch()
        nested = [target / "Caltech101", target]
        nested += [p for p in target.rglob("Caltech101") if p.is_dir()]
        for cand in nested:
            if _plausible_event_root(cand):
                print("event root:", cand,
                      f"({len(_class_dirs_under(cand))} raw class folders)")
                return cand

        # Some downloads are outer archives containing Caltech101.zip.
        for inner in target.rglob("Caltech101.zip"):
            inner_target = target / "Caltech101_inner"
            inner_marker = inner_target / ".extracted_ok"
            if not inner_marker.exists():
                inner_target.mkdir(parents=True, exist_ok=True)
                print("extracting nested", inner, "->", inner_target)
                with zipfile.ZipFile(inner, "r") as zf:
                    zf.extractall(inner_target)
                inner_marker.touch()
            nested2 = [inner_target / "Caltech101", inner_target]
            nested2 += [q for q in inner_target.rglob("Caltech101") if q.is_dir()]
            for cand in nested2:
                if _plausible_event_root(cand):
                    print("event root:", cand,
                          f"({len(_class_dirs_under(cand))} raw class folders)")
                    return cand

    raise FileNotFoundError(
        "Could not locate a plausible N-Caltech101 event tree (100-102 class "
        "folders containing .bin files). Expected DATA_ROOT/Caltech101/<class>/*.bin "
        "or DATA_ROOT/<class>/*.bin.")


def select_class_dirs(event_root):
    """Resolve known archive variants without silently changing the paper task."""
    global NUM_CLASSES
    dirs = _class_dirs_under(event_root)
    by_name = {d.name: d for d in dirs}

    # Explicit user exclusions are applied first for unusual mirrors.
    if EXCLUDE_CLASSES:
        missing = [c for c in EXCLUDE_CLASSES if c not in by_name]
        if missing:
            raise RuntimeError(
                f"NCALTECH101_EXCLUDE_CLASSES requested absent folders: {missing}; "
                f"found={sorted(by_name)}")
        dirs = [d for d in dirs if d.name not in set(EXCLUDE_CLASSES)]
        by_name = {d.name: d for d in dirs}
        print("explicitly excluded classes:", EXCLUDE_CLASSES)

    # The original N-Caltech101 removes the duplicated Faces class, retaining
    # Faces_easy and BACKGROUND_Google. Some mirrors restore Faces, yielding 102.
    if (len(dirs) == 102 and "Faces" in by_name and
            "Faces_easy" in by_name and "BACKGROUND_Google" in by_name):
        print("102-folder mirror detected: dropping duplicated 'Faces' to recover "
              "canonical 101-class N-Caltech101")
        dirs = [d for d in dirs if d.name != "Faces"]
        by_name = {d.name: d for d in dirs}

    canonical_ok = (len(dirs) == CANONICAL_NUM_CLASSES and
                    "BACKGROUND_Google" in by_name and "Faces" not in by_name)
    if not canonical_ok:
        msg = (
            f"N-Caltech101 class layout is non-canonical after normalization: "
            f"found {len(dirs)} folders; BACKGROUND_Google="
            f"{'present' if 'BACKGROUND_Google' in by_name else 'missing'}; "
            f"Faces={'present' if 'Faces' in by_name else 'absent'}. "
            f"Classes={sorted(by_name)}")
        if not ALLOW_NONCANONICAL_CLASS_COUNT:
            raise RuntimeError(
                msg + "\nFor a deliberate 100-class/background-free or other variant, "
                "set ALLOW_NONCANONICAL_CLASS_COUNT=1 (and optionally "
                "NCALTECH101_EXCLUDE_CLASSES=...). This keeps the canonical paper "
                "run from changing silently.")
        print("*** NON-CANONICAL CLASS OVERRIDE ***")
        print(msg)

    dirs = sorted(dirs, key=lambda p: p.name)
    NUM_CLASSES = len(dirs)
    if NUM_CLASSES < 2:
        raise RuntimeError(f"only {NUM_CLASSES} usable classes after normalization")
    return dirs


# ============================================================
# 2. N-Caltech101 40-bit event parser + validation
# ============================================================
def read_ncaltech101(path):
    raw = np.fromfile(path, dtype=np.uint8)
    if raw.size == 0:
        raise RuntimeError(f"empty event file: {path}")
    if raw.size % 5 != 0:
        raise RuntimeError(
            f"{path}: byte length {raw.size} is not divisible by 5; "
            "not a canonical N-Caltech101 40-bit stream")
    raw = raw.reshape(-1, 5)

    x = raw[:, 0].astype(np.int16)
    y = raw[:, 1].astype(np.int16)
    p = ((raw[:, 2] >> 7) & 1).astype(np.int8)
    t = (((raw[:, 2] & 0x7F).astype(np.int64) << 16)
         | (raw[:, 3].astype(np.int64) << 8)
         | raw[:, 4].astype(np.int64))

    # Standard N-MNIST/N-Caltech reader semantics: y==240 is a timestamp
    # overflow marker, not a spatial event. Each marker increments following
    # timestamps by 2^13 us. Remove marker rows after applying the offset.
    overflow = (y == 240)
    if overflow.any():
        t = t + np.cumsum(overflow, dtype=np.int64) * (2 ** 13)
        keep = ~overflow
        x, y, p, t = x[keep], y[keep], p[keep], t[keep]

    if len(t) > 1 and np.any(t[1:] < t[:-1]):
        order = np.argsort(t, kind="stable")
        x, y, p, t = x[order], y[order], p[order], t[order]
    return {"t": t, "x": x, "y": y, "p": p}


def validate_recording(path):
    ev = read_ncaltech101(path)
    n = len(ev["t"])
    if n <= 100:
        raise ValueError(f"only {n} real events")
    if n == 0:
        raise ValueError("no real events")
    dur_s = (int(ev["t"].max()) - int(ev["t"].min())) / 1e6
    if not (0.10 < dur_s < 1.0):
        raise ValueError(f"duration {dur_s:.3f}s outside expected 0.10-1.0s range")
    if int(ev["x"].min()) < 0 or int(ev["x"].max()) >= SENSOR_W:
        raise ValueError(
            f"x range {int(ev['x'].min())}..{int(ev['x'].max())} outside 0..{SENSOR_W-1}")
    if int(ev["y"].min()) < 0 or int(ev["y"].max()) >= SENSOR_H:
        raise ValueError(
            f"y range {int(ev['y'].min())}..{int(ev['y'].max())} outside 0..{SENSOR_H-1}")
    if not set(np.unique(ev["p"]).tolist()) <= {0, 1}:
        raise ValueError("polarity outside {0,1}")
    return {"n_events": n, "duration_s": dur_s,
            "x_min": int(ev["x"].min()), "x_max": int(ev["x"].max()),
            "y_min": int(ev["y"].min()), "y_max": int(ev["y"].max())}


# ============================================================
# 3. Manifest, deterministic 80/20 split, fixed letterbox cache
# ============================================================
def build_raw_manifest():
    event_root = resolve_event_root()
    class_dirs = select_class_dirs(event_root)

    class_names = [d.name for d in class_dirs]
    class_to_id = {c: i for i, c in enumerate(class_names)}
    pd.DataFrame({"label": range(NUM_CLASSES), "class_name": class_names}).to_csv(
        OUT / "class_map.csv", index=False)

    rows, bad_rows = [], []
    n_discovered = 0
    t0 = time.time()
    for d in class_dirs:
        cls = d.name
        files = sorted(d.rglob("*.bin"))
        n_discovered += len(files)
        if len(files) < 20:
            print(f"WARNING: {cls} has only {len(files)} raw .bin files before validation")
        good_here = 0
        for fp in files:
            try:
                meta = validate_recording(fp)
                rows.append({"path": str(fp), "label": class_to_id[cls],
                             "class_name": cls, **meta})
                good_here += 1
            except Exception as e:
                bad_rows.append({"path": str(fp), "class_name": cls,
                                 "error": repr(e)})
        if good_here < 2:
            raise RuntimeError(
                f"{cls}: only {good_here} valid recordings after validation")
        print(f"  validated {cls}: {good_here}/{len(files)} good")

    bad_df = pd.DataFrame(bad_rows, columns=["path", "class_name", "error"])
    bad_df.to_csv(OUT / "bad_files_skipped.csv", index=False)
    bad_fraction = len(bad_rows) / max(1, n_discovered)
    print(f"raw validation: {len(rows)} good / {n_discovered} discovered; "
          f"skipped {len(bad_rows)} ({100*bad_fraction:.3f}%) in "
          f"{time.time()-t0:.0f}s")
    if bad_fraction > MAX_BAD_FILE_FRACTION:
        raise RuntimeError(
            f"bad-file fraction {bad_fraction:.4f} exceeds "
            f"MAX_BAD_FILE_FRACTION={MAX_BAD_FILE_FRACTION:.4f}; inspect "
            f"{OUT / 'bad_files_skipped.csv'}")

    m = pd.DataFrame(rows)
    if len(m) < 7000:
        raise RuntimeError(
            f"only {len(m)} valid N-Caltech101 recordings remain; inspect dataset")
    print("total valid recordings:", len(m), "| classes:", m["label"].nunique())
    counts = m.groupby(["label", "class_name"]).size().rename("n").reset_index()
    counts.to_csv(OUT / "class_counts_raw.csv", index=False)
    return m


def fixed_split(manifest):
    rng = np.random.default_rng(SPLIT_SEED)
    tr, te = [], []
    split_rows = []
    for cls in range(NUM_CLASSES):
        sub = manifest[manifest["label"] == cls].reset_index(drop=True)
        n = len(sub)
        if n < 2:
            raise RuntimeError(f"class {cls}: only {n} valid samples")
        order = rng.permutation(n)
        n_test = max(1, int(math.ceil(TEST_FRACTION * n)))
        n_train = n - n_test
        tr.append(sub.iloc[order[:n_train]])
        te.append(sub.iloc[order[n_train:]])
        split_rows.append({"label": cls, "class_name": sub.iloc[0]["class_name"],
                           "total": n, "train": n_train, "test": n_test})

    tr = pd.concat(tr).sample(frac=1, random_state=SPLIT_SEED).reset_index(drop=True)
    te = pd.concat(te).sample(frac=1, random_state=SPLIT_SEED + 1).reset_index(drop=True)
    tr.to_csv(OUT / "split_train_raw.csv", index=False)
    te.to_csv(OUT / "split_test_raw.csv", index=False)
    pd.DataFrame(split_rows).to_csv(OUT / "split_counts_by_class.csv", index=False)
    print("split:", len(tr), "train /", len(te), "test (stratified 80/20)")
    return tr, te


def raw_to_tensor(path):
    ev = read_ncaltech101(path)
    t = ev["t"].astype(np.int64)
    x = ev["x"].astype(np.int64)
    y = ev["y"].astype(np.int64)
    p = ev["p"].astype(np.int64)
    if len(t) == 0:
        raise RuntimeError(f"no real events in {path}")
    if x.min() < 0 or x.max() >= SENSOR_W or y.min() < 0 or y.max() >= SENSOR_H:
        raise RuntimeError(f"spatial coordinate outside canonical {SENSOR_W}x{SENSOR_H} sensor")

    # Fixed, sample-independent, aspect-preserving mapping:
    # 240x180 sensor -> vertically centered 240x240 square -> 64x64.
    # This is equivalent to adding 30 sensor pixels above and below the 180px
    # height before a uniform 240->64 resize. Retiming never changes x/y/p.
    pad_y = (SENSOR_W - SENSOR_H) // 2  # 30
    x = np.minimum(MODEL_W - 1, (x * MODEL_W) // SENSOR_W)
    y = np.minimum(MODEL_H - 1, ((y + pad_y) * MODEL_H) // SENSOR_W)

    t = t - int(t.min())
    duration_us = max(1, int(t.max()) + 1)
    tb = np.clip(np.floor(t.astype(np.float64) * T_FINE / duration_us
                          ).astype(np.int64), 0, T_FINE - 1)

    # Accumulate in int32: np.add.at on uint16 WRAPS SILENTLY, so a
    # post-hoc "< 32000" check on a uint16 buffer can pass after the
    # tensor is already corrupted (70000 increments -> 4464).
    X32 = np.zeros((T_FINE, 2, MODEL_H, MODEL_W), dtype=np.int32)
    np.add.at(X32, (tb, p, y, x), 1)
    if int(X32.max()) >= 32000:
        raise RuntimeError(
            f"{path}: cell count {int(X32.max())} would overflow int16")
    X = X32.astype(np.uint16)
    return X, duration_us


def build_tensor_cache(df, split):
    folder = TENSOR_ROOT / split
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    t0 = time.time()
    for i, r in df.reset_index(drop=True).iterrows():
        outp = folder / f"{int(r['label']):03d}_{i:05d}.npz"
        if FORCE_REBUILD_CACHE and outp.exists():
            outp.unlink()
        if not outp.exists():
            X, duration_us = raw_to_tensor(r["path"])
            np.savez_compressed(
                outp, X=X, label=np.int16(int(r["label"])),
                duration_us=np.int64(duration_us),
                spatial_mapping=np.array("fixed_letterbox_240x180_to_64x64"),
                preprocess_version=np.array("ncaltech_letterbox_int32accum_v3"),
                source=np.array(str(r["path"])))
        rows.append({"path": str(outp), "label": int(r["label"]),
                     "class_name": str(r["class_name"])})
        if (i + 1) % 500 == 0:
            print(f"  cache {split}: {i+1}/{len(df)} ({time.time()-t0:.0f}s)")
    res = pd.DataFrame(rows)
    res.to_csv(OUT / f"{split}_tensor_manifest.csv", index=False)
    return res


if (OUT / "train_tensor_manifest.csv").exists() and not FORCE_REBUILD_CACHE:
    train_manifest = pd.read_csv(OUT / "train_tensor_manifest.csv")
    test_manifest = pd.read_csv(OUT / "test_tensor_manifest.csv")
    required_cols = {"path", "label", "class_name"}
    if not required_cols.issubset(train_manifest.columns) or not required_cols.issubset(test_manifest.columns):
        raise RuntimeError(
            "cached N-Caltech manifests are incompatible with v3; "
            "set FORCE_REBUILD_CACHE=1 or use a fresh NCALTECH101_RUN")
    probe_path = Path(train_manifest.iloc[0]["path"])
    if not probe_path.exists():
        raise FileNotFoundError(f"cached tensor missing: {probe_path}")
    with np.load(probe_path) as probe:
        pv = str(probe["preprocess_version"]) if "preprocess_version" in probe.files else ""
    if pv != "ncaltech_letterbox_int32accum_v3":
        raise RuntimeError(
            "legacy N-Caltech tensor cache detected. The current code fixes "
            "uint16 accumulation-before-check; rebuild once with "
            "FORCE_REBUILD_CACHE=1 (and FORCE_RETRAIN=1 if checkpoints exist), "
            "or use a fresh NCALTECH101_RUN.")
    print("tensor caches found:", len(train_manifest), "/", len(test_manifest))
else:
    raw_manifest = build_raw_manifest()
    tr_raw, te_raw = fixed_split(raw_manifest)
    train_manifest = build_tensor_cache(tr_raw, "train")
    test_manifest = build_tensor_cache(te_raw, "test")

# Recover class names even when the tensor cache already exists.
if (OUT / "class_map.csv").exists():
    _cm = pd.read_csv(OUT / "class_map.csv").sort_values("label")
    CLASS_NAMES = _cm["class_name"].astype(str).tolist()
elif "class_name" in train_manifest.columns:
    _cm = (pd.concat([train_manifest[["label", "class_name"]],
                     test_manifest[["label", "class_name"]]])
           .drop_duplicates().sort_values("label"))
    CLASS_NAMES = _cm["class_name"].astype(str).tolist()
    _cm.to_csv(OUT / "class_map.csv", index=False)
else:
    raise RuntimeError("class_map.csv missing and cached manifests have no class_name")
if len(CLASS_NAMES) != NUM_CLASSES:
    # Cached manifests can be used under an explicit non-canonical override.
    if not ALLOW_NONCANONICAL_CLASS_COUNT:
        raise RuntimeError(
            f"cached class map contains {len(CLASS_NAMES)} classes but configured "
            f"NUM_CLASSES={NUM_CLASSES}; remove stale v2 cache or set the explicit override")
    NUM_CLASSES = len(CLASS_NAMES)

PROTOCOL_TAG = f"{PROTOCOL_TAG_BASE}_c{NUM_CLASSES}"
# Corrected (progressive-path) diagnostics get their own identity so that
# under-optimized single-round partials from an earlier v2 run can never be
# resumed in place of the corrected computation. Training/headline attack
# checkpoints are unchanged and stay reusable.
DIAG_TAG = f"{PROTOCOL_TAG}_diagv3"
print("PROTOCOL_TAG:", PROTOCOL_TAG)
print("DIAG_TAG:", DIAG_TAG)
print("NUM_CLASSES:", NUM_CLASSES)

d0 = np.load(train_manifest.iloc[0]["path"])
print("tensor:", d0["X"].shape, d0["X"].dtype,
      "| events:", int(d0["X"].sum()),
      "| duration ms:", int(d0["duration_us"]) / 1000,
      "| spatial mapping: fixed 240x180 -> 240x240 letterbox -> 64x64")


# ============================================================
# 4. Dataset (+ train-time augmentation; attacks use raw cache)
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


class NCaltech101Dataset(Dataset):
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


train_ds = NCaltech101Dataset(train_manifest, train=True)
test_ds = NCaltech101Dataset(test_manifest, train=False)


def make_loader(ds, batch, shuffle):
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=4,
                      pin_memory=True, persistent_workers=True)


test_loader = make_loader(test_ds, EVAL_BATCH, False)


def autocast_context():
    if DEVICE.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return torch.autocast(device_type="cpu", enabled=False)


# ============================================================
# 5. Grouping / invariance / inference helpers (validated set)
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
    """Vectorized greedy move selection on an (M, S) grouping."""
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
        ck[b] = {"X": Xcur.copy(), "moved": moved,
                 "shifts": np.asarray(shifts_all, dtype=np.int8).copy()}
    return ck


# Free attack: same greedy, but the group is the whole recording per
# (pol, pixel) — moves may cross coarse frames, frames change.
def grouped_free(X):
    return X.reshape(T_FINE, 2 * MODEL_H * MODEL_W).T.copy()


def ungrouped_free(Xg):
    return Xg.T.reshape(T_FINE, 2, MODEL_H, MODEL_W)


def progressive_free_unique_attack(model, X0, y, budgets):
    """Free retiming with exactly one gradient recomputation per progressive
    budget checkpoint -- the same optimization effort as
    progressive_unique_attack, so cost-of-stealth compares like with like.
    Events may move anywhere in the recording, so frames change."""
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


# ---- exact displacement-matched control (validated: 6336/6336 exact)
def event_units_from_original(X0):
    Xg = grouped(X0).astype(np.int32)
    g_idx, s_idx = np.nonzero(Xg > 0)
    counts = Xg[g_idx, s_idx]
    return (np.repeat(g_idx, counts).astype(np.int32),
            np.repeat(s_idx, counts).astype(np.int8))


def _exact_shift_source_flow(source_counts, shift_counts, S, rng):
    """Solve exact source-bin allocation for a signed shift histogram.

    Returns flow[(src_bin, shift)] = integer event count. This is a tiny
    integral max-flow problem (S<=8 here), implemented locally so the run
    does not depend on scipy/networkx. Randomized node order keeps the
    matched control stochastic while preserving the histogram exactly.
    """
    shifts = [int(d) for d, n in shift_counts.items() if n > 0]
    if not shifts:
        return {}

    src_order = [int(x) for x in rng.permutation(np.arange(S))]
    shift_order = [int(x) for x in rng.permutation(np.asarray(shifts, dtype=np.int16))]

    # Node layout: source=0, source-bin nodes 1..S, shift nodes next, sink last.
    source = 0
    src_node = {s: 1 + i for i, s in enumerate(src_order)}
    shift_node = {d: 1 + S + i for i, d in enumerate(shift_order)}
    sink = 1 + S + len(shift_order)
    n_nodes = sink + 1
    graph = [[] for _ in range(n_nodes)]

    def add_edge(u, v, cap):
        graph[u].append([v, int(cap), len(graph[v])])
        graph[v].append([u, 0, len(graph[u]) - 1])

    for s0 in src_order:
        add_edge(source, src_node[s0], int(source_counts[s0]))
    edge_refs = {}
    inf_cap = int(sum(shift_counts.values()))
    for s0 in src_order:
        for d in shift_order:
            if 0 <= s0 + d < S:
                u, v = src_node[s0], shift_node[d]
                idx = len(graph[u])
                add_edge(u, v, inf_cap)
                edge_refs[(s0, d)] = (u, idx)
    for d in shift_order:
        add_edge(shift_node[d], sink, int(shift_counts[d]))

    total_need = int(sum(shift_counts.values()))
    total_flow = 0
    while total_flow < total_need:
        level = [-1] * n_nodes
        q = deque([source]); level[source] = 0
        while q:
            u = q.popleft()
            for v, cap, rev in graph[u]:
                if cap > 0 and level[v] < 0:
                    level[v] = level[u] + 1
                    q.append(v)
        if level[sink] < 0:
            break
        it = [0] * n_nodes

        def dfs(u, pushed):
            if u == sink:
                return pushed
            while it[u] < len(graph[u]):
                ei = it[u]
                v, cap, rev = graph[u][ei]
                if cap > 0 and level[v] == level[u] + 1:
                    got = dfs(v, min(pushed, cap))
                    if got:
                        graph[u][ei][1] -= got
                        graph[v][rev][1] += got
                        return got
                it[u] += 1
            return 0

        while total_flow < total_need:
            pushed = dfs(source, total_need - total_flow)
            if not pushed:
                break
            total_flow += pushed

    if total_flow != total_need:
        raise RuntimeError(
            f"exact displacement match infeasible: need {total_need}, got {total_flow}; "
            f"source_counts={source_counts.tolist()}, shifts={dict(shift_counts)}")

    alloc = {}
    for key, (u, ei) in edge_refs.items():
        v, residual, rev = graph[u][ei]
        used = graph[v][rev][1]  # reverse capacity equals realized flow
        if used:
            alloc[key] = int(used)
    return alloc


def exact_displacement_matched_random_attack(X0, target_shifts, rng):
    target = np.asarray(target_shifts, dtype=np.int16)
    target = target[target != 0]
    if len(target) == 0:
        return X0.copy().astype(np.int16), np.empty(0, dtype=np.int8), True
    if int(np.abs(target).max()) >= S_PER_FRAME:
        raise ValueError("target shift exceeds the protected coarse window")

    unit_g, unit_s = event_units_from_original(X0)
    shift_counts = Counter(int(d) for d in target.tolist())
    source_counts = np.bincount(unit_s.astype(np.int64), minlength=S_PER_FRAME)
    alloc = _exact_shift_source_flow(source_counts, shift_counts, S_PER_FRAME, rng)

    Xg = grouped(X0).astype(np.int32)
    realized_parts = []
    for src in range(S_PER_FRAME):
        idx = np.flatnonzero(unit_s == src)
        if len(idx) == 0:
            continue
        idx = rng.permutation(idx)
        cursor = 0
        ds = [d for (s0, d), n in alloc.items() if s0 == src and n > 0]
        if ds:
            ds = [int(x) for x in rng.permutation(np.asarray(ds, dtype=np.int16))]
        for d in ds:
            need = alloc[(src, d)]
            chosen = idx[cursor:cursor + need]
            cursor += need
            if len(chosen) != need:
                raise RuntimeError("internal exact-matcher allocation error")
            gsel = unit_g[chosen]
            ssel = unit_s[chosen].astype(np.int64)
            np.subtract.at(Xg, (gsel, ssel), 1)
            np.add.at(Xg, (gsel, ssel + d), 1)
            realized_parts.append(np.full(need, d, dtype=np.int8))

    realized = (np.concatenate(realized_parts) if realized_parts
                else np.empty(0, dtype=np.int8))
    Xatt = ungrouped(Xg).astype(np.int16)
    assert Xatt.min() >= 0
    assert int(Xatt.sum()) == int(X0.sum())
    assert coarse_equal(X0, Xatt)
    assert Counter(realized.tolist()) == Counter(target.astype(np.int8).tolist())
    return Xatt, realized, True


# ============================================================
# 6. Models — final five-model registry, all definitions embedded
#    ConvSNN / SEWResNet18 are carried from the DVS main run;
#    TransformerV2 / TemporalGRU / CoarseFrameFormer from the DVS add-on.
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
        self.c1 = TDConvBN(2, 32, 2)
        self.c2 = TDConvBN(32, 64, 2)
        self.c3 = TDConvBN(64, 128, 2)
        self.c4 = TDConvBN(128, 256, 2)
        self.head = nn.Linear(256, NUM_CLASSES)
    def forward(self, X):
        x = X * 0.15
        x = lif_sequence(self.c1(x), self.beta)
        x = lif_sequence(self.c2(x), self.beta)
        x = lif_sequence(self.c3(x), self.beta)
        x = lif_sequence(self.c4(x), self.beta)
        x = x.mean(dim=(-1, -2))
        x = x.mean(dim=1)
        return self.head(x)


class SEWBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, beta=0.90):
        super().__init__()
        self.beta = beta
        self.conv1 = TDConvBN(in_ch, out_ch, stride)
        self.conv2 = TDConvBN(out_ch, out_ch, 1)
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch))
        else:
            self.skip = None
    def forward(self, x):
        residual = x
        z = lif_sequence(self.conv1(x), self.beta)
        z = lif_sequence(self.conv2(z), self.beta)
        if self.skip is not None:
            residual = td_apply(self.skip, residual)
            residual = lif_sequence(residual, self.beta)
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
        x = X * 0.15
        x = lif_sequence(self.stem(x), self.beta)
        x = self.s1(x); x = self.s2(x); x = self.s3(x); x = self.s4(x)
        x = x.mean(dim=(-1, -2))
        x = x.mean(dim=1)
        return self.fc(x)


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
                     if stride != 1 or in_ch != out_ch else nn.Identity())
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
            nn.BatchNorm2d(out_ch))
            if stride != 1 or in_ch != out_ch else nn.Identity())
    def forward(self, x):
        r = self.skip(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + r)


class CoarseFrameFormer(nn.Module):
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
        coarse = torch.log1p(coarse)
        z = coarse.reshape(B * N_COARSE, 2, MODEL_H, MODEL_W)
        z = self.spatial(z).flatten(1)
        z = self.proj(z).reshape(B, N_COARSE, -1)
        z = torch.cat([self.cls.expand(B, -1, -1), z], dim=1)
        z = z + self.pos
        z = self.norm(self.encoder(z))
        return self.head(torch.cat([z[:, 0], z[:, 1:].mean(dim=1)], dim=1))


# Epochs/batches are carried unchanged from the frozen CIFAR10-DVS protocol.
# Optimizer/regularization AND scheduler families are transferred from the
# successful DVS-Gesture runs. In particular, the two SNNs retain the
# epoch-wise cosine schedule from EVS_DVSGesture.py rather than being silently
# changed to the add-on's per-step warmup-cosine schedule.
MODEL_SPECS = {
    "coarse_frameformer": dict(
        builder=CoarseFrameFormer,
        epochs=50, batch=48,
        lr=4e-4, wd=3e-3,
        clip=2.0, label_smoothing=0.05,
        betas=(0.9, 0.95), schedule="warmup_cosine_steps", warmup_frac=0.05,
    ),
    "conv_snn": dict(
        builder=ConvSNN,
        epochs=45, batch=24,
        lr=2e-3, wd=1e-4,
        clip=5.0, label_smoothing=0.0,
        betas=(0.9, 0.999), schedule="epoch_cosine", warmup_frac=0.0,
    ),
    "sew_resnet18": dict(
        builder=SEWResNet18,
        epochs=55, batch=12,
        lr=2e-3, wd=1e-4,
        clip=5.0, label_smoothing=0.0,
        betas=(0.9, 0.999), schedule="epoch_cosine", warmup_frac=0.0,
    ),
    "event_transformer_v2": dict(
        builder=EventTemporalTransformerV2,
        epochs=50, batch=24,
        lr=3e-4, wd=5e-3,
        clip=1.0, label_smoothing=0.05,
        betas=(0.9, 0.95), schedule="warmup_cosine_steps", warmup_frac=0.08,
    ),
    "temporal_gru": dict(
        builder=TemporalGRU_DVS,
        epochs=30, batch=32,
        lr=1e-3, wd=1e-4,
        clip=5.0, label_smoothing=0.0,
        betas=(0.9, 0.999), schedule="warmup_cosine_steps", warmup_frac=0.02,
    ),
}

def _validate_model_list(kind, names):
    unknown = [m for m in names if m not in MODEL_SPECS]
    if unknown:
        raise ValueError(f"unknown {kind} model(s): {unknown}; valid={sorted(MODEL_SPECS)}")
    return names

TRAIN_MODELS = _validate_model_list("TRAIN_MODELS", TRAIN_MODELS)
ATTACK_MODELS = _validate_model_list("ATTACK_MODELS", ATTACK_MODELS)

# The paper protocol is intentionally one full three-seed launch.  This also
# prevents model-scoped jobs from accidentally overwriting canonical CSVs with
# partial tables.  Scoped runs are still available explicitly for debugging.
if not ALLOW_SCOPED_RUN:
    if RUN_SEEDS != [0, 1, 2]:
        raise ValueError(
            f"official protocol requires RUN_SEEDS=0,1,2; got {RUN_SEEDS}. "
            "Set ALLOW_SCOPED_RUN=1 only for debugging.")
    if TRAIN_MODELS != OFFICIAL_MODELS:
        raise ValueError(
            f"official protocol requires TRAIN_MODELS={OFFICIAL_MODELS}; "
            f"got {TRAIN_MODELS}. Set ALLOW_SCOPED_RUN=1 only for debugging.")
    if ATTACK_MODELS != TEMPORAL_MODELS:
        raise ValueError(
            f"official protocol requires ATTACK_MODELS={TEMPORAL_MODELS}; "
            f"got {ATTACK_MODELS}. Set ALLOW_SCOPED_RUN=1 only for debugging.")
    if DIAGNOSTIC_SEED != 0:
        raise ValueError("official protocol requires DIAGNOSTIC_SEED=0")
    if ATTACK_TOTAL != 1000:
        raise ValueError(
            f"official protocol requires ATTACK_TOTAL=1000; got {ATTACK_TOTAL}")

# ---- startup smoke test: forward every registered model once
print(f"\nsmoke test at N-Caltech101 constants (T_FINE={T_FINE}, classes={NUM_CLASSES}):")
smoke_x = torch.zeros(2, T_FINE, 2, MODEL_H, MODEL_W, device=DEVICE)
for name in TRAIN_MODELS:
    m = MODEL_SPECS[name]["builder"]().to(DEVICE)
    with torch.no_grad(), autocast_context():
        out = m(smoke_x)
    if out.shape != (2, NUM_CLASSES):
        raise RuntimeError(f"{name}: smoke-test output shape {out.shape}, expected (2, {NUM_CLASSES})")
    n_par = sum(p.numel() for p in m.parameters()) / 1e6
    print(f"  {name}: ok ({n_par:.1f}M params)")
    del m
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()


# ============================================================
# 7. Resumable trainer — full three-seed protocol, no accuracy gate
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
        X = X.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)
        with autocast_context():
            logits = model(X)
        correct += int((logits.argmax(1) == y).sum())
        total += len(y)
    return correct / total


def warmup_cosine(step, total_steps, warmup_steps):
    """The exact manual step schedule used by the DVS-Gesture add-on."""
    if step < warmup_steps:
        return max(1e-6, step / max(1, warmup_steps))
    p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, p)))


def epoch_cosine_multiplier(epoch_zero_based, total_epochs):
    """LR used at the start of each epoch by CosineAnnealingLR(T_max=E).

    The completed DVS main run called scheduler.step() once after every epoch.
    Thus epoch 0 uses the base LR and epoch e uses
      0.5 * (1 + cos(pi * e / E)).
    Computing it directly keeps resume behavior exact without scheduler state.
    """
    return 0.5 * (1.0 + math.cos(math.pi * epoch_zero_based / max(1, total_epochs)))


def train_model(name, seed, force=False):
    spec = MODEL_SPECS[name]
    set_seed(seed)
    ckpt = CKPT_ROOT / f"{name}_seed{seed}_{PROTOCOL_TAG}.pt"
    part = CKPT_ROOT / f"{name}_seed{seed}_{PROTOCOL_TAG}_partial.pt"
    hist_path = CKPT_ROOT / f"{name}_seed{seed}_{PROTOCOL_TAG}_history.csv"

    model = spec["builder"]().to(DEVICE)
    if ckpt.exists() and not force:
        print("loading final:", ckpt)
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        return model

    loader = make_loader(train_ds, spec["batch"], True)
    opt = torch.optim.AdamW(
        model.parameters(), lr=spec["lr"], weight_decay=spec["wd"],
        betas=spec["betas"])

    total_steps = spec["epochs"] * len(loader)
    warmup_steps = (0 if spec["warmup_frac"] <= 0 else
                    max(100, int(spec["warmup_frac"] * total_steps)))

    start_ep, gstep, history = 0, 0, []
    if part.exists() and not force:
        try:
            st = torch.load(part, map_location=DEVICE)
            model.load_state_dict(st["model"])
            opt.load_state_dict(st["opt"])
            start_ep = int(st["epoch"])
            gstep = int(st["gstep"])
            history = st.get("history", [])
            print(f"resuming {name} seed {seed} at epoch {start_ep}")
        except Exception as e:
            print(f"partial unreadable ({e!r}); from scratch")
            start_ep, gstep, history = 0, 0, []

    for ep in range(start_ep, spec["epochs"]):
        model.train()
        losses, correct, total = [], 0, 0
        t0 = time.time()

        # DVS main SNNs used one fixed LR within each epoch followed by one
        # CosineAnnealingLR step. Add-on models used a per-optimizer-step
        # warmup-cosine. Preserve those scheduler families here.
        if spec["schedule"] == "epoch_cosine":
            lr_epoch = spec["lr"] * epoch_cosine_multiplier(ep, spec["epochs"])
            for pg in opt.param_groups:
                pg["lr"] = lr_epoch
        elif spec["schedule"] != "warmup_cosine_steps":
            raise ValueError(f"unknown schedule {spec['schedule']} for {name}")

        lr_now = float(opt.param_groups[0]["lr"])
        for X, y, _ in loader:
            X = X.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            if spec["schedule"] == "warmup_cosine_steps":
                lr_now = spec["lr"] * warmup_cosine(gstep, total_steps, warmup_steps)
                for pg in opt.param_groups:
                    pg["lr"] = lr_now
            else:
                lr_now = lr_epoch

            opt.zero_grad(set_to_none=True)
            with autocast_context():
                logits = model(X)
                loss = F.cross_entropy(
                    logits, y, label_smoothing=spec["label_smoothing"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), spec["clip"])
            opt.step()
            gstep += 1

            losses.append(float(loss.detach().cpu()))
            correct += int((logits.detach().argmax(1) == y).sum())
            total += len(y)

        test_acc = evaluate(model, test_loader)
        row = {
            "epoch": ep + 1,
            "loss": float(np.mean(losses)),
            "train_acc": correct / total,
            "test_acc": test_acc,
            "lr": float(lr_now),
            "seconds": time.time() - t0,
        }
        history.append(row)
        print(f"{name} seed={seed} ep={ep+1:02d}/{spec['epochs']} "
              f"loss={row['loss']:.4f} train={row['train_acc']:.4f} "
              f"test={test_acc:.4f} lr={row['lr']:.2e} "
              f"t={row['seconds']:.0f}s")
        pd.DataFrame(history).to_csv(hist_path, index=False)

        tmp = part.with_suffix(".pt.tmp")
        torch.save({
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "epoch": ep + 1,
            "gstep": gstep,
            "history": history,
            "protocol_tag": PROTOCOL_TAG,
               "allow_scoped_run": ALLOW_SCOPED_RUN,
            "schedule": spec["schedule"],
        }, tmp)
        os.replace(tmp, part)

    torch.save(model.state_dict(), ckpt)
    if part.exists():
        part.unlink()
    return model


# Normal paper run: all declared models x all requested seeds. There is no
# seed-0 selection stage. A low clean score produces a warning but the model
# remains in the experiment, preventing N-Caltech101-result-contingent selection.
clean_rows = []
for name in TRAIN_MODELS:
    for seed in RUN_SEEDS:
        model = train_model(name, seed, force=FORCE_RETRAIN)
        acc = evaluate(model, test_loader)
        clean_rows.append({
            "seed": seed,
            "model": name,
            "clean_test_accuracy": float(acc),
            "asr_interpretable": bool(acc >= MIN_CLEAN_INTERPRET),
        })
        print("FINAL", name, "seed", seed, "acc", round(acc, 4))
        if acc < CLEAN_WARN:
            print(f"*** WARNING ONLY: {name} seed {seed} clean {acc:.3f} "
                  f"< {CLEAN_WARN:.3f}; retained in the predeclared run")
        if acc < MIN_CLEAN_INTERPRET:
            print(f"*** INTERPRETATION FLAG: {name} seed {seed} clean {acc:.3f} "
                  f"< {MIN_CLEAN_INTERPRET:.3f}; attack rows will be retained but "
                  "must not be used as vulnerability evidence")
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

clean_df = pd.DataFrame(clean_rows)
clean_df.to_csv(OUT / "clean_accuracy_by_seed.csv", index=False)
print(clean_df.to_string(index=False))
if len(clean_df):
    print(clean_df.groupby("model")["clean_test_accuracy"]
          .agg(["mean", "std"]).round(4).to_string())
full_clean_acc_map = {
    (str(r["model"]), int(r["seed"])): float(r["clean_test_accuracy"])
    for r in clean_rows
}

# ============================================================
# 8. Attack phase (resumable units) + ablation + cost
# ============================================================
def proportional_attack_quotas(test_df, total):
    """Largest-remainder allocation proportional to test-set class availability."""
    counts = np.bincount(test_df["label"].to_numpy(np.int64), minlength=NUM_CLASSES)
    available = int(counts.sum())
    if total <= 0:
        raise ValueError("ATTACK_TOTAL must be positive")
    if total > available:
        raise ValueError(f"ATTACK_TOTAL={total} exceeds test-set size {available}")
    ideal = counts.astype(np.float64) * (float(total) / available)
    quotas = np.floor(ideal).astype(np.int64)
    quotas = np.minimum(quotas, counts)
    remainder = int(total - quotas.sum())
    frac = ideal - np.floor(ideal)
    # Stable deterministic tie-break: larger remainder, then larger availability,
    # then lower class id.
    order = sorted(range(NUM_CLASSES), key=lambda c: (-frac[c], -counts[c], c))
    for c in order:
        if remainder <= 0:
            break
        if quotas[c] < counts[c]:
            quotas[c] += 1
            remainder -= 1
    if remainder:
        # Defensive capacity fill; normally unreachable because total<=available.
        for c in sorted(range(NUM_CLASSES), key=lambda c: (-(counts[c]-quotas[c]), c)):
            take = min(remainder, int(counts[c] - quotas[c]))
            quotas[c] += take
            remainder -= take
            if remainder == 0:
                break
    if int(quotas.sum()) != total or np.any(quotas > counts):
        raise RuntimeError("proportional attack allocation failed")
    return counts, quotas


def wilson95(k, n):
    if n <= 0:
        return float("nan"), float("nan")
    z = 1.959963984540054
    phat = k / n
    den = 1.0 + z*z/n
    ctr = (phat + z*z/(2*n)) / den
    half = z * math.sqrt(phat*(1-phat)/n + z*z/(4*n*n)) / den
    return max(0.0, ctr-half), min(1.0, ctr+half)


def attack_metrics(pred, clean_pred, y, clean_correct):
    ncc = int(clean_correct.sum())
    succ = int((pred[clean_correct] != y[clean_correct]).sum()) if ncc else 0
    asr = succ / ncc if ncc else float("nan")
    lo, hi = wilson95(succ, ncc)
    return {
        "attacked_accuracy": float((pred == y).mean()),
        "asr_clean_correct": float(asr),
        "asr_wilson95_low": float(lo),
        "asr_wilson95_high": float(hi),
        "n_attack": int(len(y)),
        "n_clean_correct": ncc,
        "n_attack_success_clean_correct": succ,
        "prediction_flip_rate": float((pred != clean_pred).mean()),
    }


if RUN_ATTACKS:
    # Deterministic, availability-proportional stratified subset. This replaces
    # the statistically thin 5/class v1 design while preserving class coverage
    # according to the natural N-Caltech test distribution.
    rng = np.random.default_rng(9090)
    test_counts, quotas = proportional_attack_quotas(test_manifest, ATTACK_TOTAL)
    parts = []
    for cls in range(NUM_CLASSES):
        sub = test_manifest[test_manifest["label"] == cls].reset_index(drop=True)
        q = int(quotas[cls])
        if q == 0:
            continue
        chosen = rng.choice(len(sub), size=q, replace=False)
        parts.append(sub.iloc[chosen])
    attack_manifest = (pd.concat(parts).sample(frac=1, random_state=9091)
                       .reset_index(drop=True))
    if len(attack_manifest) != ATTACK_TOTAL:
        raise RuntimeError(f"attack subset has {len(attack_manifest)} != {ATTACK_TOTAL}")
    attack_manifest.to_csv(OUT / "attack_subset_manifest.csv", index=False)
    pd.DataFrame({
        "label": np.arange(NUM_CLASSES),
        "class_name": CLASS_NAMES,
        "test_available": test_counts,
        "attack_selected": quotas,
    }).to_csv(OUT / "attack_subset_counts_by_class.csv", index=False)
    Xattack, yattack, dur_us = [], [], []
    for _, r in attack_manifest.iterrows():
        d = np.load(r["path"])
        Xattack.append(d["X"].astype(np.int16))
        yattack.append(int(d["label"]))
        dur_us.append(int(d["duration_us"]))
    Xattack = np.stack(Xattack)
    yattack = np.asarray(yattack, np.int64)
    dur_us = np.asarray(dur_us, np.int64)
    print("attack set:", Xattack.shape,
          "| class quota range:", int(quotas.min()), "..", int(quotas.max()))

    UNIT_DIR = OUT / f"attack_partial_{PROTOCOL_TAG}"
    UNIT_DIR.mkdir(parents=True, exist_ok=True)

    def _unit_fp(tag):
        return UNIT_DIR / f"{tag}.json"

    def _load_unit(tag):
        fp = _unit_fp(tag)
        if not fp.exists():
            return None
        try:
            return json.load(open(fp))
        except Exception:
            return None

    def _save_unit(tag, payload):
        tmp = _unit_fp(tag).with_suffix(".json.tmp")
        json.dump(payload, open(tmp, "w"))
        os.replace(tmp, _unit_fp(tag))

    def load_model(name, seed):
        m = MODEL_SPECS[name]["builder"]().to(DEVICE)
        m.load_state_dict(torch.load(
            CKPT_ROOT / f"{name}_seed{seed}_{PROTOCOL_TAG}.pt",
            map_location=DEVICE))
        m.eval()
        return m

    trained_ok = {(r["model"], r["seed"]) for r in clean_rows}

    main_rows, verify_rows = [], []
    for seed in RUN_SEEDS:
        if ("coarse_frameformer", seed) not in trained_ok:
            raise RuntimeError(
                f"coarse_frameformer seed {seed} is required for protected-consumer checks")
        coarse2 = load_model("coarse_frameformer", seed)
        clean_c2 = predict_logits(coarse2, Xattack)
        clean_c2_pred = clean_c2.argmax(1)

        for name in ATTACK_MODELS:
            if (name, seed) not in trained_ok:
                raise RuntimeError(f"requested attack model missing: {name} seed {seed}")
            tag = f"unit_seed{seed}_{name}"
            cached = _load_unit(tag)
            if cached is not None:
                print(f"resuming: {tag} done")
                main_rows += cached["main_rows"]
                verify_rows += cached.get("verify_rows", [])
                continue

            u_main, u_verify = [], []
            model = load_model(name, seed)
            clean_pred = predict_logits(model, Xattack).argmax(1)
            clean_correct = clean_pred == yattack
            clean_acc = float(clean_correct.mean())
            full_clean_acc = full_clean_acc_map[(name, seed)]
            print(f"\n[{name} seed {seed}] attack-subset clean {clean_acc:.4f} "
                  f"| full-test clean {full_clean_acc:.4f}")

            adv_by_b = {float(b): [] for b in ATTACK_BUDGETS}
            shf_by_b = {float(b): [] for b in ATTACK_BUDGETS}
            t0 = time.time()
            for i in range(len(Xattack)):
                ck = progressive_unique_attack(model, Xattack[i], yattack[i],
                                               ATTACK_BUDGETS)
                for b in ATTACK_BUDGETS:
                    adv_by_b[float(b)].append(ck[float(b)]["X"])
                    shf_by_b[float(b)].append(ck[float(b)]["shifts"])
                if (i + 1) % 100 == 0:
                    print(f"  gradient {i+1}/{len(Xattack)} "
                          f"({time.time()-t0:.0f}s)")

            bin_ms = dur_us / T_FINE / 1000.0
            for b in ATTACK_BUDGETS:
                b = float(b)
                Xadv = np.stack(adv_by_b[b])
                adv_pred = predict_logits(model, Xadv).argmax(1)
                fm = batch_coarse_metrics(Xattack, Xadv)
                mean_ms = [float(np.abs(s).mean()) * bin_ms[i] if len(s) else 0.0
                           for i, s in enumerate(shf_by_b[b])]
                realized = [len(s) / max(1, int(Xattack[i].sum()))
                            for i, s in enumerate(shf_by_b[b])]
                max_ms = [float(np.abs(s).max()) * bin_ms[i] if len(s) else 0.0
                          for i, s in enumerate(shf_by_b[b])]
                row = {"seed": seed, "model": name, "attack": "gradient_unique",
                       "budget": b, "clean_accuracy": clean_acc,
                       "clean_accuracy_full_test": full_clean_acc,
                       "asr_interpretable": bool(full_clean_acc >= MIN_CLEAN_INTERPRET),
                       **attack_metrics(adv_pred, clean_pred, yattack, clean_correct),
                       "mean_realized_unique_fraction": float(np.mean(realized)),
                       "mean_abs_shift_ms": float(np.mean(mean_ms)),
                       "max_abs_shift_ms": float(np.max(max_ms)), **fm}
                c2 = predict_logits(coarse2, Xadv)
                row["coarse_frameformer_logits_exact"] = bool(np.array_equal(c2, clean_c2))
                row["coarse_frameformer_flip_rate"] = float(
                    (c2.argmax(1) != clean_c2_pred).mean())
                u_main.append(row)
                # uniform control
                rng = np.random.default_rng(310000 + seed * 1000 + int(b * 10000))
                Xu, Xu_shifts = [], []
                for i in range(len(Xattack)):
                    n_mv = len(shf_by_b[b][i])
                    ug, us = event_units_from_original(Xattack[i])
                    n_mv = min(n_mv, len(ug))
                    chosen = rng.permutation(len(ug))[:n_mv]
                    Xg = grouped(Xattack[i]).astype(np.int32)
                    gs = ug[chosen]
                    ss = us[chosen].astype(np.int64)
                    ds = (ss + rng.integers(1, S_PER_FRAME, size=n_mv)) % S_PER_FRAME
                    np.subtract.at(Xg, (gs, ss), 1)
                    np.add.at(Xg, (gs, ds), 1)
                    Xr = ungrouped(Xg).astype(np.int16)
                    assert Xr.min() >= 0 and coarse_equal(Xattack[i], Xr)
                    Xu.append(Xr)
                    Xu_shifts.append((ds - ss).astype(np.int16))
                Xu = np.stack(Xu)
                upred = predict_logits(model, Xu).argmax(1)
                u_mean_ms = [float(np.abs(s).mean()) * bin_ms[i] if len(s) else 0.0
                             for i, s in enumerate(Xu_shifts)]
                u_max_ms = [float(np.abs(s).max()) * bin_ms[i] if len(s) else 0.0
                            for i, s in enumerate(Xu_shifts)]
                u_realized = [len(s) / max(1, int(Xattack[i].sum()))
                              for i, s in enumerate(Xu_shifts)]
                urow = {"seed": seed, "model": name, "attack": "random_uniform",
                        "budget": b, "clean_accuracy": clean_acc,
                        "clean_accuracy_full_test": full_clean_acc,
                        "asr_interpretable": bool(full_clean_acc >= MIN_CLEAN_INTERPRET),
                        **attack_metrics(upred, clean_pred, yattack, clean_correct),
                        "mean_realized_unique_fraction": float(np.mean(u_realized)),
                        "mean_abs_shift_ms": float(np.mean(u_mean_ms)),
                        "max_abs_shift_ms": float(np.max(u_max_ms)),
                        **batch_coarse_metrics(Xattack, Xu)}
                c2u = predict_logits(coarse2, Xu)
                urow["coarse_frameformer_logits_exact"] = bool(np.array_equal(c2u, clean_c2))
                urow["coarse_frameformer_flip_rate"] = float(
                    (c2u.argmax(1) != clean_c2_pred).mean())
                u_main.append(urow)

                # exact displacement-matched control
                rng = np.random.default_rng(320000 + seed * 1000 + int(b * 10000))
                Xm, Xm_shifts, n_ver = [], [], 0
                for i in range(len(Xattack)):
                    xm, sm, ok = exact_displacement_matched_random_attack(
                        Xattack[i], shf_by_b[b][i], rng)
                    if not ok:
                        raise RuntimeError("strict exact matcher returned non-exact result")
                    target_nonzero = np.asarray(shf_by_b[b][i], dtype=np.int8)
                    target_nonzero = target_nonzero[target_nonzero != 0]
                    is_exact = Counter(sm.tolist()) == Counter(target_nonzero.tolist())
                    if not is_exact:
                        raise RuntimeError(f"signed shift histogram mismatch at sample {i}, budget {b}")
                    n_ver += 1
                    Xm.append(xm)
                    Xm_shifts.append(sm.astype(np.int16))
                Xm = np.stack(Xm)
                mpred = predict_logits(model, Xm).argmax(1)
                m_mean_ms = [float(np.abs(s).mean()) * bin_ms[i] if len(s) else 0.0
                             for i, s in enumerate(Xm_shifts)]
                m_max_ms = [float(np.abs(s).max()) * bin_ms[i] if len(s) else 0.0
                            for i, s in enumerate(Xm_shifts)]
                m_realized = [len(s) / max(1, int(Xattack[i].sum()))
                              for i, s in enumerate(Xm_shifts)]
                mrow = {"seed": seed, "model": name,
                        "attack": "random_displacement_matched_exact",
                        "budget": b, "clean_accuracy": clean_acc,
                        "clean_accuracy_full_test": full_clean_acc,
                        "asr_interpretable": bool(full_clean_acc >= MIN_CLEAN_INTERPRET),
                        **attack_metrics(mpred, clean_pred, yattack, clean_correct),
                        "mean_realized_unique_fraction": float(np.mean(m_realized)),
                        "mean_abs_shift_ms": float(np.mean(m_mean_ms)),
                        "max_abs_shift_ms": float(np.max(m_max_ms)),
                        **batch_coarse_metrics(Xattack, Xm),
                        "exact_match_fraction": 1.0,
                        "matched_fallback_count": 0}
                c2m = predict_logits(coarse2, Xm)
                mrow["coarse_frameformer_logits_exact"] = bool(np.array_equal(c2m, clean_c2))
                mrow["coarse_frameformer_flip_rate"] = float(
                    (c2m.argmax(1) != clean_c2_pred).mean())
                u_main.append(mrow)
                u_verify.append({"seed": seed, "model": name, "budget": b,
                                 "samples": len(Xattack),
                                 "signed_histogram_exact": int(n_ver),
                                 "fallback": 0})
                print(f"  b={b}: grad {u_main[-3]['asr_clean_correct']:.3f} | "
                      f"unif {u_main[-2]['asr_clean_correct']:.3f} | "
                      f"exact-matched {u_main[-1]['asr_clean_correct']:.3f} "
                      f"(exact {n_ver}/{len(Xattack)})")
                del Xadv, Xu, Xm
                gc.collect()

            _save_unit(tag, {"main_rows": u_main,
                             "verify_rows": u_verify})
            main_rows += u_main
            verify_rows += u_verify
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if coarse2 is not None:
            del coarse2
            gc.collect()

    pd.DataFrame(main_rows).to_csv(OUT / "main_attack_results_by_seed.csv",
                                   index=False)
    # Per-class ASR is intentionally omitted: even with 1,000 proportional
    # attacks, rare N-Caltech classes have too few clean-correct attacked samples
    # for stable classwise inference. Quotas are saved for transparency instead.
    pd.DataFrame(verify_rows).to_csv(OUT / "exact_match_verification.csv",
                                     index=False)
    mv = pd.DataFrame(main_rows)
    if len(mv):
        agg = (mv.groupby(["model", "attack", "budget"])
               [["attacked_accuracy", "asr_clean_correct"]].agg(["mean", "std"]))
        agg.to_csv(OUT / "main_attack_aggregate.csv")
        print(agg.round(4).to_string())
        print("frame_exact all:", bool(mv["frame_exact_equal"].all()),
              "| maxdiff:", int(mv["max_integer_frame_difference"].max()))
        if "coarse_frameformer_logits_exact" in mv:
            print("coarse_frameformer logits exact all:",
                  bool(mv["coarse_frameformer_logits_exact"].dropna().all()))

    first_seed = DIAGNOSTIC_SEED

    # ---- max-shift ablation (diagnostic seed; seed 0 by default)
    if RUN_SHIFT_ABLATION and DIAGNOSTIC_SEED in RUN_SEEDS:
        abl_rows = []
        for name in ATTACK_MODELS:
            if (name, first_seed) not in trained_ok:
                continue
            tag = f"ablation_{DIAG_TAG}_seed{first_seed}_{name}"
            cached = _load_unit(tag)
            if cached is not None:
                abl_rows += cached["rows"]
                continue
            model = load_model(name, first_seed)
            clean_pred = predict_logits(model, Xattack).argmax(1)
            clean_correct = clean_pred == yattack
            full_clean_acc = full_clean_acc_map[(name, first_seed)]
            rows = []
            for ms in SHIFT_ABLATION_BINS:
                Xa, sh = [], []
                for i in range(len(Xattack)):
                    ck = progressive_unique_attack(
                        model, Xattack[i], yattack[i],
                        ABLATION_PATH, max_shift_bins=ms)
                    Xa.append(ck[SHIFT_ABLATION_BUDGET]["X"])
                    sh.append(ck[SHIFT_ABLATION_BUDGET]["shifts"])
                Xa = np.stack(Xa)
                pred = predict_logits(model, Xa).argmax(1)
                bin_ms = dur_us / T_FINE / 1000.0
                mean_ms = [float(np.abs(s).mean()) * bin_ms[i] if len(s) else 0.0
                           for i, s in enumerate(sh)]
                rows.append({"seed": first_seed, "model": name,
                             "budget": SHIFT_ABLATION_BUDGET,
                             "optimization_path": "->".join(
                                 f"{b:g}" for b in ABLATION_PATH),
                             "max_shift_label": "full" if ms is None else str(ms),
                             "mean_abs_shift_ms": float(np.mean(mean_ms)),
                             "clean_accuracy_full_test": full_clean_acc,
                             "asr_interpretable": bool(full_clean_acc >= MIN_CLEAN_INTERPRET),
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
        abl_df.to_csv(OUT / "max_shift_ablation_v3.csv", index=False)
        # The 'full' row now walks the same path as the main table's 10% row,
        # so the two should agree up to nondeterministic gradient ordering.
        main_path = OUT / "main_attack_results_by_seed.csv"
        if main_path.exists() and len(abl_df):
            main_df = pd.read_csv(main_path)
            checks = []
            for name in abl_df["model"].unique():
                old = main_df[(main_df["model"] == name)
                              & (main_df["seed"] == first_seed)
                              & (main_df["attack"] == "gradient_unique")
                              & np.isclose(main_df["budget"].astype(float),
                                           SHIFT_ABLATION_BUDGET)]
                new = abl_df[(abl_df["model"] == name)
                             & (abl_df["max_shift_label"] == "full")]
                if len(old) == 1 and len(new) == 1:
                    a = float(old.iloc[0]["asr_clean_correct"])
                    b = float(new.iloc[0]["asr_clean_correct"])
                    checks.append({
                        "model": name,
                        "main_10pct_asr": a,
                        "ablation_full_asr": b,
                        "absolute_difference_pct_points": 100.0 * abs(a - b),
                        "consistent_within_0p5pt": bool(abs(a - b) <= 0.005),
                    })
            if checks:
                cdf = pd.DataFrame(checks)
                cdf.to_csv(OUT / "max_shift_ablation_main_check_v3.csv", index=False)
                print("\nfull-shift consistency check:")
                print(cdf.round(4).to_string(index=False))

    # ---- cost of stealth (diagnostic seed; seed 0 by default)
    if RUN_COST_OF_STEALTH and DIAGNOSTIC_SEED in RUN_SEEDS:
        cost_rows = []
        sub = np.arange(min(COST_SUBSET, len(Xattack)))
        for name in ATTACK_MODELS:
            if (name, first_seed) not in trained_ok:
                continue
            tag = f"cost_{DIAG_TAG}_seed{first_seed}_{name}"
            cached = _load_unit(tag)
            if cached is not None:
                cost_rows += cached["rows"]
                continue
            model = load_model(name, first_seed)
            coarse2_l = load_model("coarse_frameformer", first_seed) \
                if ("coarse_frameformer", first_seed) in trained_ok else None
            Xs = Xattack[sub]; ys = yattack[sub]
            clean_pred = predict_logits(model, Xs).argmax(1)
            clean_correct = clean_pred == ys
            full_clean_acc = full_clean_acc_map[(name, first_seed)]
            c2_clean = predict_logits(coarse2_l, Xs) if coarse2_l is not None else None
            rows = []
            # One progressive run per arm, read out at each COST_BUDGET, so
            # stealth and free receive identical optimization effort.
            arms = {"stealth": {b: [] for b in COST_BUDGETS},
                    "free": {b: [] for b in COST_BUDGETS}}
            arm_sh = {"stealth": {b: [] for b in COST_BUDGETS},
                      "free": {b: [] for b in COST_BUDGETS}}
            for i in range(len(Xs)):
                ck_s = progressive_unique_attack(model, Xs[i], ys[i], COST_PATH)
                ck_f = progressive_free_unique_attack(model, Xs[i], ys[i], COST_PATH)
                for b in COST_BUDGETS:
                    arms["stealth"][b].append(ck_s[float(b)]["X"])
                    arm_sh["stealth"][b].append(
                        ck_s[float(b)]["shifts"].astype(np.int16))
                    arms["free"][b].append(ck_f[float(b)]["X"])
                    arm_sh["free"][b].append(
                        ck_f[float(b)]["shifts"].astype(np.int16))
                if (i + 1) % 50 == 0:
                    print(f"  cost {name}: {i+1}/{len(Xs)}")
            for b in COST_BUDGETS:
                for kind in ["stealth", "free"]:
                    sh = arm_sh[kind][b]
                    Xa = np.stack(arms[kind][b])
                    pred = predict_logits(model, Xa).argmax(1)
                    bin_ms = dur_us[sub] / T_FINE / 1000.0
                    mean_ms = [float(np.abs(s).mean()) * bin_ms[i] if len(s) else 0.0
                               for i, s in enumerate(sh)]
                    row = {"seed": first_seed, "model": name, "kind": kind,
                           "budget": float(b), "n": len(Xs),
                           "optimization_path": "->".join(
                               f"{x:g}" for x in COST_PATH),
                           "clean_accuracy_full_test": full_clean_acc,
                           "asr_interpretable": bool(full_clean_acc >= MIN_CLEAN_INTERPRET),
                           **attack_metrics(pred, clean_pred, ys, clean_correct),
                           "mean_abs_shift_ms": float(np.mean(mean_ms)),
                           **batch_coarse_metrics(Xs, Xa)}
                    if coarse2_l is not None:
                        c2a = predict_logits(coarse2_l, Xa)
                        row["coarse_frameformer_logits_exact"] = bool(
                            np.array_equal(c2a, c2_clean))
                        row["coarse_frameformer_flip_rate"] = float(
                            (c2a.argmax(1) != c2_clean.argmax(1)).mean())
                    rows.append(row)
                    print(f"  cost {name} {kind} b={b}: acc "
                          f"{row['attacked_accuracy']:.3f} maxdiff "
                          f"{row['max_integer_frame_difference']}")
                    del Xa
                    gc.collect()
            del arms, arm_sh
            gc.collect()
            _save_unit(tag, {"rows": rows})
            cost_rows += rows
            del model
            if coarse2_l is not None:
                del coarse2_l
            gc.collect()
        pd.DataFrame(cost_rows).to_csv(OUT / "cost_of_stealth_v3.csv", index=False)

# run config
with open(OUT / "run_config.json", "w") as f:
    json.dump({"dataset": "N-Caltech101", "data_source": str(DATA_ROOT),
               "t_fine": T_FINE, "s_per_frame": S_PER_FRAME,
               "n_coarse": N_COARSE, "model_hw": MODEL_H,
               "split_seed": SPLIT_SEED,
               "train_fraction": TRAIN_FRACTION,
               "test_fraction": TEST_FRACTION,
               "num_classes": NUM_CLASSES,
               "class_names": CLASS_NAMES,
               "attack_budgets": ATTACK_BUDGETS,
               "ablation_path": ABLATION_PATH,
               "cost_path": COST_PATH,
               "attack_total": ATTACK_TOTAL,
               "attack_sampling": "deterministic_proportional_stratified_without_replacement",
               "per_class_asr_reported": False,
               "spatial_mapping": "fixed_letterbox_240x180_to_64x64",
               "preprocess_version": "ncaltech_letterbox_int32accum_v3",
               "force_rebuild_cache": FORCE_REBUILD_CACHE,
               "max_bad_file_fraction": MAX_BAD_FILE_FRACTION,
               "allow_noncanonical_class_count": ALLOW_NONCANONICAL_CLASS_COUNT,
               "excluded_classes": EXCLUDE_CLASSES,
               "min_clean_interpret": MIN_CLEAN_INTERPRET,
               "seeds": RUN_SEEDS, "clean_warn": CLEAN_WARN,
               "diagnostic_seed": DIAGNOSTIC_SEED,
               "protocol_tag": PROTOCOL_TAG,
               "allow_scoped_run": ALLOW_SCOPED_RUN,
               "official_models": OFFICIAL_MODELS,
               "temporal_models": TEMPORAL_MODELS,
               "train_models": TRAIN_MODELS,
               "attack_models": ATTACK_MODELS,
               "model_training_specs": {
                   name: {
                       "epochs": int(spec["epochs"]),
                       "batch": int(spec["batch"]),
                       "lr": float(spec["lr"]),
                       "weight_decay": float(spec["wd"]),
                       "clip": float(spec["clip"]),
                       "label_smoothing": float(spec["label_smoothing"]),
                       "betas": [float(x) for x in spec["betas"]],
                       "warmup_frac": float(spec["warmup_frac"]),
                       "schedule": spec["schedule"],
                   }
                   for name, spec in MODEL_SPECS.items()
               }}, f, indent=2)

print("\nDONE. Outputs under:", OUT)

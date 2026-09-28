"""
Event-Security Study — CIFAR10-DVS (headless cluster run).

FINAL PRE-RESULT CIFAR10-DVS PROTOCOL
=====================================

This file is the single official third-dataset run following the completed
N-MNIST and DVS128-Gesture experiments and the shared planning notes.
It is deliberately frozen BEFORE any CIFAR10-DVS result is inspected.

Run design
----------
* One normal full run over seeds 0,1,2.  There is NO seed-0 accuracy gate and
  no results-contingent model exclusion.
* Five trained models are pre-declared:
    1) coarse_frameformer       — protected coarse-frame consumer
    2) conv_snn                 — temporal/spiking victim
    3) sew_resnet18             — temporal/spiking victim
    4) event_transformer_v2     — temporal ANN victim
    5) temporal_gru             — temporal recurrent victim
* The four temporal victims are attacked for all three seeds.
* The max-shift and cost-of-stealth diagnostics remain seed-0 diagnostics,
  matching the earlier DVS-Gesture study; they are executed inside this same
  full run and do not require a separate seed-0 training launch.

Why TemporalGRU is included
---------------------------
TemporalGRU was a full three-seed victim in the landed DVS-Gesture add-on and
was also present in N-MNIST.  Including it now, before CIFAR results exist,
keeps the cross-dataset recurrent-consumer row complete and avoids deciding
its inclusion after seeing CIFAR numbers.

Transferred model recipes
-------------------------
ConvSNN / SEWResNet18 (completed DVS-Gesture main run):
  AdamW lr=2e-3, weight_decay=1e-4, betas=(0.9,0.999), clip=5.0,
  no label smoothing, beta=0.90, x0.15 input scaling, epoch-wise cosine LR.

EventTemporalTransformerV2 (completed DVS-Gesture add-on):
  AdamW lr=3e-4, weight_decay=5e-3, betas=(0.9,0.95), clip=1.0,
  label_smoothing=0.05, step-wise warmup-cosine with warmup_frac=0.08.

TemporalGRU (completed DVS-Gesture add-on):
  AdamW lr=1e-3, weight_decay=1e-4, betas=(0.9,0.999), clip=5.0,
  no label smoothing, step-wise warmup-cosine with warmup_frac=0.02.

CoarseFrameFormer (completed DVS-Gesture add-on):
  AdamW lr=4e-4, weight_decay=3e-3, betas=(0.9,0.95), clip=2.0,
  label_smoothing=0.05, step-wise warmup-cosine with warmup_frac=0.05.

CIFAR10-DVS-specific schedule fixed by the shared plan
------------------------------------------------------
  coarse_frameformer:      50 epochs, batch 48
  conv_snn:                45 epochs, batch 24
  sew_resnet18:            55 epochs, batch 12
  event_transformer_v2:    50 epochs, batch 24
  temporal_gru:            30 epochs, batch 32

Core representation / split / attack protocol
---------------------------------------------
* DVS128 sensor -> deterministic 64x64 model grid.
* T_FINE=80; S_PER_FRAME=8; N_COARSE=10.
* Split seed 2027; 800 train / 200 test recordings per class.
* Balanced attack subset: 100/class by default; RNG 9090 then shuffle 9091.
* Unique-event budgets: 1%, 2%, 5%, 10%, 20%.
* Constrained gradient attack moves original events only inside their original
  coarse frame, preserving the coarse integer tensor exactly.
* Uniform random unique-event control.
* Strict signed-displacement-matched random control solved by integral
  max-flow: exact histogram only, with no approximate fallback.
* 10% maximum-shift diagnostic over [1,2,4,full].
* Instrumented cost-of-stealth diagnostic.
* CoarseFrameFormer is evaluated as the learned protected consumer for every
  constrained attack/control row.

Implementation hardening
------------------------
* AEDAT2 external/trigger events (address bit 15) are filtered before decode.
* Archive sanity checks run before caching.
* uint16 -> int16 overflow is guarded.
* Model-list typos and smoke-test failures are fatal.
* Checkpoints and partial attack units are isolated by PROTOCOL_TAG.
* Training and attacks are resumable; rerunning the same command resumes.

Normal launch
-------------
  python3 EVS_CIFAR10DVS_FINAL.py

Equivalent explicit launch:
  RUN_SEEDS=0,1,2 python3 EVS_CIFAR10DVS_FINAL.py

Environment knobs
-----------------
  CIFAR_DVS_DATA      dataset root; default
      /path/to/workspace/EVS_CIFAR10DVS/data
  CIFAR_DVS_RUN       work root; default
      /path/to/workspace/EVS_CIFAR10DVS/run
  RUN_SEEDS           default "0,1,2"
  TRAIN_MODELS        optional comma list; full five-model set by default
  ATTACK_MODELS       optional comma list; all four temporal victims by default
  ATTACK_PER_CLASS    default 100
  CLEAN_WARN          default 0.55; warning only, never excludes a seed/model
  RUN_ATTACKS         default 1
  RUN_SHIFT_ABLATION  default 1
  RUN_COST_OF_STEALTH default 1
  DIAGNOSTIC_SEED     default 0
  FORCE_RETRAIN       default 0
  ALLOW_SCOPED_RUN    default 0; set 1 only for debugging/non-paper runs
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
    "CIFAR_DVS_DATA", "/path/to/workspace/EVS_CIFAR10DVS/data"))
ROOT = Path(os.environ.get(
    "CIFAR_DVS_RUN", "/path/to/workspace/EVS_CIFAR10DVS/run"))
RAW_ROOT = ROOT / "raw"
TENSOR_ROOT = ROOT / "tensor_cache"
CKPT_ROOT = ROOT / "checkpoints"
OUT = ROOT / "results"
for p in [ROOT, RAW_ROOT, TENSOR_ROOT, CKPT_ROOT, OUT]:
    p.mkdir(parents=True, exist_ok=True)
assert DATA_ROOT.exists(), f"dataset root missing: {DATA_ROOT}"

CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]
CLASS_TO_ID = {c: i for i, c in enumerate(CLASS_NAMES)}
NUM_CLASSES = 10

SENSOR_H = SENSOR_W = 128
MODEL_H = MODEL_W = 64
SPATIAL_DIV = 2

T_FINE = 80
S_PER_FRAME = 8
N_COARSE = T_FINE // S_PER_FRAME
assert T_FINE % S_PER_FRAME == 0

SPLIT_SEED = 2027
TRAIN_PER_CLASS, TEST_PER_CLASS = 800, 200

ATTACK_BUDGETS = [0.01, 0.02, 0.05, 0.10, 0.20]
ATTACK_PER_CLASS = int(os.environ.get("ATTACK_PER_CLASS", "100"))
SHIFT_ABLATION_BUDGET = 0.10
SHIFT_ABLATION_BINS = [1, 2, 4, None]
COST_BUDGETS = [0.10, 0.20]
COST_SUBSET = 250
EVAL_BATCH = 48

RUN_SEEDS = [int(s.strip()) for s in os.environ.get("RUN_SEEDS", "0,1,2").split(",") if s.strip()]
if len(set(RUN_SEEDS)) != len(RUN_SEEDS):
    raise ValueError(f"RUN_SEEDS contains duplicates: {RUN_SEEDS}")

# Final pre-result registry. CoarseFrameFormer is the protected consumer;
# the remaining four are temporal victims. TemporalGRU is intentionally
# included now (before any CIFAR result exists) for cross-dataset completeness.
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
TRAIN_MODELS = [m.strip() for m in os.environ.get("TRAIN_MODELS", DEFAULT_TRAIN).split(",") if m.strip()]
ATTACK_MODELS = [m.strip() for m in os.environ.get("ATTACK_MODELS", DEFAULT_ATTACK).split(",") if m.strip()]

# New tag because this final protocol differs from earlier gate/four-model
# variants and also restores the exact DVS main-run scheduler for the SNNs.
PROTOCOL_TAG = "cifar3seed_dvslocked5_exact_v2"

# Warning only. The full three-seed protocol NEVER drops a model because of
# observed CIFAR accuracy; low accuracy is reported and interpreted, not used
# to tune or select the model after the fact.
CLEAN_WARN = float(os.environ.get("CLEAN_WARN", "0.55"))
DIAGNOSTIC_SEED = int(os.environ.get("DIAGNOSTIC_SEED", "0"))
FORCE_RETRAIN = os.environ.get("FORCE_RETRAIN", "0") == "1"
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
print("ATTACK_PER_CLASS:", ATTACK_PER_CLASS, "| budgets:", ATTACK_BUDGETS)


# ============================================================
# 1. Archive preparation
# ============================================================
def prepare_class(class_name):
    target = RAW_ROOT / class_name
    if target.exists() and any(target.rglob("*")):
        return target
    target.mkdir(parents=True, exist_ok=True)
    src_dir = DATA_ROOT / class_name
    if src_dir.exists() and src_dir.is_dir():
        print("linking/copying", src_dir)
        for fp in src_dir.rglob("*"):
            if not fp.is_file():
                continue
            dst = target / fp.relative_to(src_dir)
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                try:
                    os.link(fp, dst)          # hardlink: no extra disk
                except OSError:
                    shutil.copy2(fp, dst)
        return target
    for cand in [DATA_ROOT / f"{class_name}.zip",
                 DATA_ROOT / f"{class_name.capitalize()}.zip"]:
        if cand.exists():
            print("extracting", cand)
            with zipfile.ZipFile(cand, "r") as z:
                z.extractall(target)
            return target
    raise FileNotFoundError(f"no folder or zip for {class_name} under {DATA_ROOT}")


def discover_files(directory):
    files = sorted(directory.rglob("*.aedat"))
    if not files:
        files = [p for p in sorted(directory.rglob("*"))
                 if p.is_file() and not p.name.startswith(".")
                 and p.suffix.lower() not in {".txt", ".csv", ".zip", ".mat"}
                 and p.stat().st_size > 100_000]
    return files


# ============================================================
# 2. AEDAT2 parser (DVS128 masks) with trigger filtering
# ============================================================
def skip_aedat_header(fp):
    while True:
        pos = fp.tell()
        line = fp.readline()
        if not line:
            return
        if not line.startswith(b"#"):
            fp.seek(pos)
            return


def read_cifar10dvs(path):
    with open(path, "rb") as fp:
        skip_aedat_header(fp)
        raw = np.frombuffer(fp.read(), dtype=">u4")
    if len(raw) % 2:
        raw = raw[:-1]
    addr = raw[0::2].astype(np.uint32)
    t = raw[1::2].astype(np.int64)

    # FIX: drop external/trigger events (bit 15) before decoding.
    real = (addr & 0x8000) == 0
    addr, t = addr[real], t[real]

    x_raw = ((addr & 0x000000FE) >> 1).astype(np.int16)
    y_raw = ((addr & 0x00007F00) >> 8).astype(np.int16)
    p_raw = (addr & 0x00000001).astype(np.int8)

    x = (127 - y_raw).astype(np.int16)      # standard CIFAR10-DVS orientation
    y = (127 - x_raw).astype(np.int16)
    p = (1 - p_raw).astype(np.int8)

    valid = (x >= 0) & (x < 128) & (y >= 0) & (y < 128) & ((p == 0) | (p == 1))
    return {"t": t[valid], "x": x[valid], "y": y[valid], "p": p[valid]}


def sanity_check_class(cls, files, n_probe=5):
    """Fail fast if the archive variant does not match the parser."""
    probe = files[:: max(1, len(files) // n_probe)][:n_probe]
    for fp in probe:
        ev = read_cifar10dvs(fp)
        n = len(ev["t"])
        assert n > 10_000, f"{fp}: only {n} events — wrong format?"
        dur_s = (int(ev["t"].max()) - int(ev["t"].min())) / 1e6
        assert 0.3 < dur_s < 5.0, f"{fp}: duration {dur_s:.2f}s implausible"
        assert int(ev["x"].max()) <= 127 and int(ev["y"].max()) <= 127
        assert set(np.unique(ev["p"]).tolist()) <= {0, 1}
    print(f"  sanity ok: {cls} ({len(files)} files, probe {len(probe)})")


# ============================================================
# 3. Manifest, fixed split, tensor cache
# ============================================================
def build_raw_manifest():
    rows = []
    for cls in CLASS_NAMES:
        d = prepare_class(cls)
        files = discover_files(d)
        assert 800 <= len(files) <= 1200, \
            f"{cls}: {len(files)} recordings (expected ~1000)"
        sanity_check_class(cls, files)
        rows += [{"path": str(fp), "label": CLASS_TO_ID[cls], "class_name": cls}
                 for fp in files]
    m = pd.DataFrame(rows)
    print("total recordings:", len(m))
    print(m["label"].value_counts().sort_index().to_string())
    return m


def fixed_split(manifest):
    rng = np.random.default_rng(SPLIT_SEED)
    tr, te = [], []
    for cls in range(NUM_CLASSES):
        sub = manifest[manifest["label"] == cls].reset_index(drop=True)
        assert len(sub) >= TRAIN_PER_CLASS + TEST_PER_CLASS, \
            f"class {cls}: only {len(sub)}"
        order = rng.permutation(len(sub))
        tr.append(sub.iloc[order[:TRAIN_PER_CLASS]])
        te.append(sub.iloc[order[TRAIN_PER_CLASS:TRAIN_PER_CLASS + TEST_PER_CLASS]])
    tr = pd.concat(tr).sample(frac=1, random_state=SPLIT_SEED).reset_index(drop=True)
    te = pd.concat(te).sample(frac=1, random_state=SPLIT_SEED + 1).reset_index(drop=True)
    tr.to_csv(OUT / "split_train_raw.csv", index=False)
    te.to_csv(OUT / "split_test_raw.csv", index=False)
    print("split:", len(tr), "train /", len(te), "test (CSVs saved for the paper)")
    return tr, te


def raw_to_tensor(path):
    ev = read_cifar10dvs(path)
    t = ev["t"].astype(np.int64)
    x = ev["x"].astype(np.int64) // SPATIAL_DIV
    y = ev["y"].astype(np.int64) // SPATIAL_DIV
    p = ev["p"].astype(np.int64)
    if len(t) == 0:
        raise RuntimeError(f"no events in {path}")
    t = t - int(t.min())
    duration_us = max(1, int(t.max()) + 1)
    tb = np.clip(np.floor(t.astype(np.float64) * T_FINE / duration_us
                          ).astype(np.int64), 0, T_FINE - 1)
    X = np.zeros((T_FINE, 2, MODEL_H, MODEL_W), dtype=np.uint16)
    np.add.at(X, (tb, p, y, x), 1)
    assert int(X.max()) < 32000, \
        f"{path}: cell count {int(X.max())} would overflow int16"
    return X, duration_us


def build_tensor_cache(df, split):
    folder = TENSOR_ROOT / split
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    t0 = time.time()
    for i, r in df.reset_index(drop=True).iterrows():
        outp = folder / f"{int(r['label']):02d}_{i:05d}.npz"
        if not outp.exists():
            X, duration_us = raw_to_tensor(r["path"])
            np.savez_compressed(outp, X=X, label=np.int16(int(r["label"])),
                                duration_us=np.int64(duration_us),
                                source=np.array(str(r["path"])))
        rows.append({"path": str(outp), "label": int(r["label"])})
        if (i + 1) % 500 == 0:
            print(f"  cache {split}: {i+1}/{len(df)} ({time.time()-t0:.0f}s)")
    res = pd.DataFrame(rows)
    res.to_csv(OUT / f"{split}_tensor_manifest.csv", index=False)
    return res


if (OUT / "train_tensor_manifest.csv").exists() and not FORCE_RETRAIN:
    train_manifest = pd.read_csv(OUT / "train_tensor_manifest.csv")
    test_manifest = pd.read_csv(OUT / "test_tensor_manifest.csv")
    print("tensor caches found:", len(train_manifest), "/", len(test_manifest))
else:
    raw_manifest = build_raw_manifest()
    tr_raw, te_raw = fixed_split(raw_manifest)
    train_manifest = build_tensor_cache(tr_raw, "train")
    test_manifest = build_tensor_cache(te_raw, "test")

d0 = np.load(train_manifest.iloc[0]["path"])
print("tensor:", d0["X"].shape, d0["X"].dtype,
      "| events:", int(d0["X"].sum()),
      "| duration ms:", int(d0["duration_us"]) / 1000)


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


class CIFAR10DVSDataset(Dataset):
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


train_ds = CIFAR10DVSDataset(train_manifest, train=True)
test_ds = CIFAR10DVSDataset(test_manifest, train=False)


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


def free_unique_attack(model, X0, y, budget, rounds=3):
    total = int(X0.sum())
    Xcur = X0.copy().astype(np.int16)
    Ag = grouped_free(X0).astype(np.int32)
    moved, shifts_all = 0, []
    target = int(round(budget * total))
    for _ in range(rounds):
        need = target - moved
        if need <= 0:
            break
        grad = input_gradient(model, Xcur, y)
        Xg = grouped_free(Xcur).astype(np.int32)
        Gg = grouped_free(grad).astype(np.float32)
        Xg, Ag, sh = _select_moves(Xg, Gg, Ag, need, T_FINE, None)
        Xcur = ungrouped_free(Xg).astype(np.int16)
        moved += len(sh)
        shifts_all.extend(sh.tolist())
    assert Xcur.min() >= 0 and int(Xcur.sum()) == total
    return Xcur, np.asarray(shifts_all, dtype=np.int16)


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


# Epochs/batches are CIFAR10-DVS-specific and fixed by the shared plan.
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
    if ATTACK_PER_CLASS != 100:
        raise ValueError(
            f"official protocol requires ATTACK_PER_CLASS=100; got {ATTACK_PER_CLASS}")

# ---- startup smoke test: forward every registered model once
print("\nsmoke test at CIFAR constants (T_FINE=80, 10 classes):")
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
# remains in the experiment, preventing CIFAR-result-contingent selection.
clean_rows = []
for name in TRAIN_MODELS:
    for seed in RUN_SEEDS:
        model = train_model(name, seed, force=FORCE_RETRAIN)
        acc = evaluate(model, test_loader)
        clean_rows.append({
            "seed": seed,
            "model": name,
            "clean_test_accuracy": float(acc),
        })
        print("FINAL", name, "seed", seed, "acc", round(acc, 4))
        if acc < CLEAN_WARN:
            print(f"*** WARNING ONLY: {name} seed {seed} clean {acc:.3f} "
                  f"< {CLEAN_WARN:.3f}; retained in the predeclared run")
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

# ============================================================
# 8. Attack phase (resumable units) + ablation + cost
# ============================================================
if RUN_ATTACKS:
    # balanced attack subset (published)
    rng = np.random.default_rng(9090)
    parts = []
    for cls in range(NUM_CLASSES):
        sub = test_manifest[test_manifest["label"] == cls].reset_index(drop=True)
        chosen = rng.choice(len(sub), size=min(ATTACK_PER_CLASS, len(sub)),
                            replace=False)
        parts.append(sub.iloc[chosen])
    attack_manifest = (pd.concat(parts).sample(frac=1, random_state=9091)
                       .reset_index(drop=True))
    attack_manifest.to_csv(OUT / "attack_subset_manifest.csv", index=False)
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
          "| per class:", np.bincount(yattack, minlength=10))

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

    main_rows, class_rows, verify_rows = [], [], []
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
                class_rows += cached["class_rows"]
                verify_rows += cached.get("verify_rows", [])
                continue

            u_main, u_class, u_verify = [], [], []
            model = load_model(name, seed)
            clean_pred = predict_logits(model, Xattack).argmax(1)
            clean_correct = clean_pred == yattack
            clean_acc = float(clean_correct.mean())
            print(f"\n[{name} seed {seed}] clean {clean_acc:.4f}")

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
                       "attacked_accuracy": float((adv_pred == yattack).mean()),
                       "asr_clean_correct": float(
                           (adv_pred[clean_correct] != yattack[clean_correct]).mean()),
                       "prediction_flip_rate": float((adv_pred != clean_pred).mean()),
                       "mean_realized_unique_fraction": float(np.mean(realized)),
                       "mean_abs_shift_ms": float(np.mean(mean_ms)),
                       "max_abs_shift_ms": float(np.max(max_ms)), **fm}
                c2 = predict_logits(coarse2, Xadv)
                row["coarse_frameformer_logits_exact"] = bool(np.array_equal(c2, clean_c2))
                row["coarse_frameformer_flip_rate"] = float(
                    (c2.argmax(1) != clean_c2_pred).mean())
                u_main.append(row)
                for cls in range(NUM_CLASSES):
                    m_ = yattack == cls
                    cm = m_ & clean_correct
                    u_class.append({"seed": seed, "model": name,
                                    "attack": "gradient_unique", "budget": b,
                                    "class": cls, "n": int(m_.sum()),
                                    "attacked_accuracy": float(
                                        (adv_pred[m_] == yattack[m_]).mean()),
                                    "asr_clean_correct": float(
                                        (adv_pred[cm] != yattack[cm]).mean())
                                        if cm.sum() else float("nan")})

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
                        "attacked_accuracy": float((upred == yattack).mean()),
                        "asr_clean_correct": float(
                            (upred[clean_correct] != yattack[clean_correct]).mean()),
                        "prediction_flip_rate": float((upred != clean_pred).mean()),
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
                        "attacked_accuracy": float((mpred == yattack).mean()),
                        "asr_clean_correct": float(
                            (mpred[clean_correct] != yattack[clean_correct]).mean()),
                        "prediction_flip_rate": float((mpred != clean_pred).mean()),
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

            _save_unit(tag, {"main_rows": u_main, "class_rows": u_class,
                             "verify_rows": u_verify})
            main_rows += u_main
            class_rows += u_class
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
    pd.DataFrame(class_rows).to_csv(OUT / "per_class_gradient_results.csv",
                                    index=False)
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
            tag = f"ablation_seed{first_seed}_{name}"
            cached = _load_unit(tag)
            if cached is not None:
                abl_rows += cached["rows"]
                continue
            model = load_model(name, first_seed)
            clean_pred = predict_logits(model, Xattack).argmax(1)
            clean_correct = clean_pred == yattack
            rows = []
            for ms in SHIFT_ABLATION_BINS:
                Xa, sh = [], []
                for i in range(len(Xattack)):
                    ck = progressive_unique_attack(
                        model, Xattack[i], yattack[i],
                        [SHIFT_ABLATION_BUDGET], max_shift_bins=ms)
                    Xa.append(ck[SHIFT_ABLATION_BUDGET]["X"])
                    sh.append(ck[SHIFT_ABLATION_BUDGET]["shifts"])
                Xa = np.stack(Xa)
                pred = predict_logits(model, Xa).argmax(1)
                bin_ms = dur_us / T_FINE / 1000.0
                mean_ms = [float(np.abs(s).mean()) * bin_ms[i] if len(s) else 0.0
                           for i, s in enumerate(sh)]
                rows.append({"seed": first_seed, "model": name,
                             "budget": SHIFT_ABLATION_BUDGET,
                             "max_shift_label": "full" if ms is None else str(ms),
                             "mean_abs_shift_ms": float(np.mean(mean_ms)),
                             "attacked_accuracy": float((pred == yattack).mean()),
                             "asr_clean_correct": float(
                                 (pred[clean_correct] != yattack[clean_correct]).mean()),
                             **batch_coarse_metrics(Xattack, Xa)})
                print(f"  ablation {name} shift={rows[-1]['max_shift_label']}: "
                      f"ASR {rows[-1]['asr_clean_correct']:.3f}")
                del Xa
                gc.collect()
            _save_unit(tag, {"rows": rows})
            abl_rows += rows
            del model
            gc.collect()
        pd.DataFrame(abl_rows).to_csv(OUT / "max_shift_ablation.csv", index=False)

    # ---- cost of stealth (diagnostic seed; seed 0 by default)
    if RUN_COST_OF_STEALTH and DIAGNOSTIC_SEED in RUN_SEEDS:
        cost_rows = []
        sub = np.arange(min(COST_SUBSET, len(Xattack)))
        for name in ATTACK_MODELS:
            if (name, first_seed) not in trained_ok:
                continue
            tag = f"cost_seed{first_seed}_{name}"
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
            c2_clean = predict_logits(coarse2_l, Xs) if coarse2_l is not None else None
            rows = []
            for b in COST_BUDGETS:
                for kind in ["stealth", "free"]:
                    Xa, sh = [], []
                    for i in range(len(Xs)):
                        if kind == "stealth":
                            ck = progressive_unique_attack(model, Xs[i], ys[i], [b])
                            Xa.append(ck[float(b)]["X"])
                            sh.append(ck[float(b)]["shifts"].astype(np.int16))
                        else:
                            xa, sf = free_unique_attack(model, Xs[i], ys[i], b)
                            Xa.append(xa)
                            sh.append(sf)
                    Xa = np.stack(Xa)
                    pred = predict_logits(model, Xa).argmax(1)
                    bin_ms = dur_us[sub] / T_FINE / 1000.0
                    mean_ms = [float(np.abs(s).mean()) * bin_ms[i] if len(s) else 0.0
                               for i, s in enumerate(sh)]
                    row = {"seed": first_seed, "model": name, "kind": kind,
                           "budget": float(b), "n": len(Xs),
                           "attacked_accuracy": float((pred == ys).mean()),
                           "asr_clean_correct": float(
                               (pred[clean_correct] != ys[clean_correct]).mean()),
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
            _save_unit(tag, {"rows": rows})
            cost_rows += rows
            del model
            if coarse2_l is not None:
                del coarse2_l
            gc.collect()
        pd.DataFrame(cost_rows).to_csv(OUT / "cost_of_stealth.csv", index=False)

# run config
with open(OUT / "run_config.json", "w") as f:
    json.dump({"dataset": "CIFAR10-DVS", "data_source": str(DATA_ROOT),
               "t_fine": T_FINE, "s_per_frame": S_PER_FRAME,
               "n_coarse": N_COARSE, "model_hw": MODEL_H,
               "split_seed": SPLIT_SEED,
               "train_per_class": TRAIN_PER_CLASS,
               "test_per_class": TEST_PER_CLASS,
               "attack_budgets": ATTACK_BUDGETS,
               "attack_per_class": ATTACK_PER_CLASS,
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

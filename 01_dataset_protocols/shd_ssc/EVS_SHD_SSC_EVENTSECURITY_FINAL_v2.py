#!/usr/bin/env python3
"""
Event-Security Study — Heidelberg Spiking Data Sets (SHD / SSC).

This is a cross-modal replication of the locked CIFAR10-DVS / N-Caltech101 /
DailyDVS protocol. The linked IEEE paper (document 9311226) releases two
neuromorphic AUDIO benchmarks rather than event-camera datasets:

  * SHD — Spiking Heidelberg Digits, 20 classes, 700 cochlear channels.
  * SSC — Spiking Speech Commands, 35 classes, 700 cochlear channels.

The default is SHD. Set HSD_DATASET=SSC to run the same protocol on SSC.

IMPORTANT SCOPE
---------------
The event-retiming mechanism transfers exactly, because the data are still
asynchronous spikes with timestamps and channel identities. The protected
observer sums fine time bins within coarse windows. A legal attack retimes a
spike within its original coarse window and leaves its cochlear channel fixed,
so the protected representation is bit-exact.

The VISUAL 2-D backbones used on CIFAR10-DVS/N-Caltech101 cannot be applied
literally to 700 cochlear channels. Therefore this script preserves the SAME
five architecture families and optimizer recipes, replacing only 2-D spatial
operators with 1-D cochlear/spectral operators:

  1) coarse_frameformer       protected coarse-time consumer
  2) conv_snn                 temporal/spiking victim
  3) sew_resnet18             temporal/spiking residual victim
  4) event_transformer_v2     temporal Transformer victim
  5) temporal_gru             temporal recurrent victim

Frozen cross-dataset attack protocol
------------------------------------
  * T_FINE=80, S_PER_FRAME=8, N_COARSE=10.
  * Official dataset split(s); no result-dependent sample/model selection.
  * Seeds 0,1,2.
  * ATTACK_TOTAL=1000, deterministic proportional stratification.
  * Budgets 1/2/5/10/20% of original spike units.
  * Constrained attack: gradient-guided unique-spike retiming within the
    original protected coarse window and same cochlear channel.
  * Controls: uniform within-window retiming + exact signed-displacement-
    matched random assignment (integral max-flow; no approximate fallback).
  * Max-shift diagnostic at 10% over {1,2,4,full} fine bins.
  * Cost-of-stealth diagnostic at 10%/20% with constrained vs free retiming
    on the identical progressive path 1->2->5->10->20%.
  * Protected consumer logits and predictions are checked whenever the coarse
    integer representation is unchanged.
  * Per-sample D_A and D_inf are saved for the cost-of-stealth experiment so
    SC-ASR_A(tau) can be computed post hoc exactly as in the current paper.

Data format
-----------
The official files are HDF5 archives from Zenke Lab. The script downloads and
verifies them automatically unless HSD_AUTO_DOWNLOAD=0. No Tonic dependency
is required; only h5py is used.

Run examples
------------
  # Full SHD run (default)
  HSD_ROOT=/path/to/workspace/shd_eventsecurity \
      python3 EVS_SHD_SSC_EVENTSECURITY_FINAL.py

  # Train only, then resume later with attacks
  HSD_ROOT=/path/to/workspace/shd_eventsecurity RUN_ATTACKS=0 \
      python3 EVS_SHD_SSC_EVENTSECURITY_FINAL.py

  # SSC instead
  HSD_DATASET=SSC HSD_ROOT=/path/to/workspace/ssc_eventsecurity \
      python3 EVS_SHD_SSC_EVENTSECURITY_FINAL.py

This dataset is NOT an event-camera dataset. In a manuscript it should be
framed as a cross-modal neuromorphic-spike generalization test, not counted as
an additional event-camera benchmark.
"""

import os
import gc
import json
import math
import time
import random
import hashlib
import shutil
import urllib.request
import zipfile
from pathlib import Path
from collections import Counter, defaultdict, deque

import h5py
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================================================
# 0. Configuration
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("torch:", torch.__version__, "| device:", DEVICE)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
torch.backends.cudnn.benchmark = True

DATASET = os.environ.get("HSD_DATASET", "SHD").strip().upper()
if DATASET not in {"SHD", "SSC"}:
    raise ValueError("HSD_DATASET must be SHD or SSC")

ROOT = Path(os.environ.get(
    "HSD_ROOT", f"/path/to/workspace/{DATASET.lower()}_eventsecurity"))
DATA_ROOT = Path(os.environ.get("HSD_DATA", str(ROOT / "data")))
RUN_ROOT = Path(os.environ.get("HSD_RUN", str(ROOT / "run")))
CACHE_ROOT = RUN_ROOT / "tensor_cache"
CKPT_ROOT = RUN_ROOT / "checkpoints"
OUT = RUN_ROOT / "results"
for p in (DATA_ROOT, RUN_ROOT, CACHE_ROOT, CKPT_ROOT, OUT):
    p.mkdir(parents=True, exist_ok=True)

AUTO_DOWNLOAD = os.environ.get("HSD_AUTO_DOWNLOAD", "1") == "1"
FORCE_REBUILD_CACHE = os.environ.get("FORCE_REBUILD_CACHE", "0") == "1"
FORCE_RETRAIN = os.environ.get("FORCE_RETRAIN", "0") == "1"
RUN_ATTACKS = os.environ.get("RUN_ATTACKS", "1") == "1"
RUN_SHIFT_ABLATION = os.environ.get("RUN_SHIFT_ABLATION", "1") == "1"
RUN_COST_OF_STEALTH = os.environ.get("RUN_COST_OF_STEALTH", "1") == "1"
ALLOW_SCOPED_RUN = os.environ.get("ALLOW_SCOPED_RUN", "0") == "1"

N_FREQ = 700
T_FINE = 80
S_PER_FRAME = 8
N_COARSE = T_FINE // S_PER_FRAME
assert T_FINE % S_PER_FRAME == 0
NUM_CLASSES = 20 if DATASET == "SHD" else 35

ATTACK_BUDGETS = [0.01, 0.02, 0.05, 0.10, 0.20]
ATTACK_TOTAL = int(os.environ.get("ATTACK_TOTAL", "1000"))
SHIFT_ABLATION_BUDGET = 0.10
SHIFT_ABLATION_BINS = [1, 2, 4, None]
ABLATION_PATH = [0.01, 0.02, 0.05, 0.10]
COST_BUDGETS = [0.10, 0.20]
COST_PATH = [0.01, 0.02, 0.05, 0.10, 0.20]
COST_SUBSET = int(os.environ.get("COST_SUBSET", "250"))

EVAL_BATCH = int(os.environ.get("EVAL_BATCH", "64"))
MIN_CLEAN_INTERPRET = float(os.environ.get("MIN_CLEAN_INTERPRET", "0.15"))
CLEAN_WARN = float(os.environ.get("CLEAN_WARN", "0.25"))
RUN_SEEDS = [int(s.strip()) for s in os.environ.get("RUN_SEEDS", "0,1,2").split(",") if s.strip()]
DIAGNOSTIC_SEED = 0

OFFICIAL_MODELS = [
    "coarse_frameformer", "conv_snn", "sew_resnet18",
    "event_transformer_v2", "temporal_gru",
]
TEMPORAL_MODELS = [
    "conv_snn", "sew_resnet18", "event_transformer_v2", "temporal_gru",
]
TRAIN_MODELS = [x.strip() for x in os.environ.get(
    "TRAIN_MODELS", ",".join(OFFICIAL_MODELS)).split(",") if x.strip()]
ATTACK_MODELS = [x.strip() for x in os.environ.get(
    "ATTACK_MODELS", ",".join(TEMPORAL_MODELS)).split(",") if x.strip()]

if len(set(RUN_SEEDS)) != len(RUN_SEEDS):
    raise ValueError(f"RUN_SEEDS contains duplicates: {RUN_SEEDS}")
if not ALLOW_SCOPED_RUN:
    if RUN_SEEDS != [0, 1, 2] or ATTACK_TOTAL != 1000 or \
       TRAIN_MODELS != OFFICIAL_MODELS or ATTACK_MODELS != TEMPORAL_MODELS:
        raise SystemExit(
            "official protocol requires RUN_SEEDS=0,1,2, ATTACK_TOTAL=1000 "
            "and the full five-model registry; use ALLOW_SCOPED_RUN=1 only "
            "for deliberate debugging runs")
else:
    # Scoped runs are isolated so they cannot overwrite canonical artifacts.
    scope = os.environ.get("SCOPED_TAG", "debug")
    OUT = RUN_ROOT / f"results_scoped_{scope}"
    CKPT_ROOT = RUN_ROOT / f"checkpoints_scoped_{scope}"
    OUT.mkdir(parents=True, exist_ok=True)
    CKPT_ROOT.mkdir(parents=True, exist_ok=True)

PROTOCOL_TAG = f"{DATASET.lower()}_locked5_exact_v1_T{T_FINE}_S{S_PER_FRAME}_n{ATTACK_TOTAL}"
DIAG_TAG = f"{PROTOCOL_TAG}_diagv1"
PREPROCESS_VERSION = f"{DATASET.lower()}_700ch_relativetime_T80_v1"

print("protocol:", PROTOCOL_TAG)
print("dataset:", DATASET, "| classes:", NUM_CLASSES, "| channels:", N_FREQ)
print("seeds:", RUN_SEEDS, "| attack_total:", ATTACK_TOTAL)


# ============================================================
# 1. Official Heidelberg files: download / verify
# ============================================================
# compneuro.net/datasets/ (same lab) serves byte-identical files; either host
# passes the MD5 gate below. Override with HSD_BASE_URL if one is unreachable.
BASE_URL = os.environ.get("HSD_BASE_URL", "https://zenkelab.org/datasets/")
FILES = {
    "SHD": {
        "train": ("shd_train.h5.zip", "f3252aeb598ac776c1b526422d90eecb"),
        "test":  ("shd_test.h5.zip",  "1503a5064faa34311c398fb0a1ed0a6f"),
    },
    "SSC": {
        "train": ("ssc_train.h5.zip", "d102be95e7144fcc0553d1f45ba94170"),
        "valid": ("ssc_valid.h5.zip", "b4eee3516a4a90dd0c71a6ac23a8ae43"),
        "test":  ("ssc_test.h5.zip",  "a35ff1e9cffdd02a20eb850c17c37748"),
    },
}


def md5sum(path, chunk=1 << 24):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def download_file(url, out):
    tmp = out.with_suffix(out.suffix + ".part")
    print("downloading:", url)
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, out)


def ensure_h5(split):
    zip_name, want_md5 = FILES[DATASET][split]
    h5_name = zip_name[:-4]
    h5_path = DATA_ROOT / h5_name
    if h5_path.exists():
        return h5_path
    zip_path = DATA_ROOT / zip_name
    if not zip_path.exists():
        if not AUTO_DOWNLOAD:
            raise FileNotFoundError(
                f"missing {zip_path}; enable HSD_AUTO_DOWNLOAD=1 or place the official file there")
        download_file(BASE_URL + zip_name, zip_path)
    got = md5sum(zip_path)
    if got.lower() != want_md5.lower():
        raise RuntimeError(
            f"MD5 mismatch for {zip_path.name}: expected {want_md5}, got {got}. "
            "Delete the file and download again.")
    print("md5 ok:", zip_path.name)
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.namelist() if m.endswith(".h5")]
        if len(members) != 1:
            raise RuntimeError(f"expected one .h5 in {zip_path}, got {members}")
        zf.extract(members[0], DATA_ROOT)
        extracted = DATA_ROOT / members[0]
        if extracted != h5_path:
            shutil.move(str(extracted), str(h5_path))
    return h5_path


H5_PATHS = {split: ensure_h5(split) for split in FILES[DATASET]}
print("HDF5 files:", {k: str(v) for k, v in H5_PATHS.items()})


def inspect_h5(path):
    with h5py.File(path, "r") as f:
        n = len(f["labels"])
        labels = np.asarray(f["labels"][:], dtype=np.int64)
        if labels.min() < 0 or labels.max() >= NUM_CLASSES:
            raise RuntimeError(f"label range wrong in {path}: {labels.min()}..{labels.max()}")
        # Verify first nonempty sample and 700-channel range.
        checked = False
        for i in range(min(100, n)):
            units = np.asarray(f["spikes/units"][i], dtype=np.int64)
            times = np.asarray(f["spikes/times"][i], dtype=np.float64)
            if len(units):
                if len(units) != len(times):
                    raise RuntimeError(f"times/units length mismatch in {path} sample {i}")
                if units.min() < 0 or units.max() >= N_FREQ:
                    raise RuntimeError(f"unit range wrong in {path}: {units.min()}..{units.max()}")
                checked = True
                break
        if not checked:
            raise RuntimeError(f"no nonempty spike sample found in {path}")
    return n


for sp, hp in H5_PATHS.items():
    print(f"{sp}: {inspect_h5(hp)} samples")


# ============================================================
# 2. Cache: asynchronous spikes -> 80 x 1 x 700 count tensor
# ============================================================

def spikes_to_tensor(times_s, units):
    times_s = np.asarray(times_s, dtype=np.float64)
    units = np.asarray(units, dtype=np.int64)
    if len(times_s) != len(units):
        raise ValueError("times and units differ in length")
    X = np.zeros((T_FINE, 1, N_FREQ), dtype=np.int32)
    if len(times_s) == 0:
        return X.astype(np.int16), 0.0, 1e-6
    t0 = float(times_s.min())
    # + tiny epsilon keeps the final spike inside bin 79 under exact arithmetic.
    duration = max(float(times_s.max()) - t0, 1e-6)
    rel = times_s - t0
    tb = np.floor(rel * T_FINE / (duration + np.finfo(np.float64).eps)).astype(np.int64)
    tb = np.clip(tb, 0, T_FINE - 1)
    np.add.at(X, (tb, np.zeros_like(tb), units), 1)
    if X.max() >= np.iinfo(np.int16).max:
        raise RuntimeError(f"binned spike count overflow risk: peak={int(X.max())}")
    return X.astype(np.int16), t0, duration


def build_cache(split, h5_path):
    folder = CACHE_ROOT / split
    folder.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT / f"{split}_tensor_manifest.csv"
    rows = []
    with h5py.File(h5_path, "r") as f:
        n = len(f["labels"])
        labels = np.asarray(f["labels"][:], dtype=np.int64)
        for i in range(n):
            outp = folder / f"{i:06d}.npz"
            if FORCE_REBUILD_CACHE and outp.exists():
                outp.unlink()
            if not outp.exists():
                times = np.asarray(f["spikes/times"][i], dtype=np.float64)
                units = np.asarray(f["spikes/units"][i], dtype=np.int64)
                X, t0, duration = spikes_to_tensor(times, units)
                tmp = outp.with_suffix(".tmp.npz")
                np.savez_compressed(
                    tmp, X=X, label=np.int16(labels[i]), index=np.int32(i),
                    t0_s=np.float64(t0), duration_s=np.float64(duration),
                    n_spikes=np.int32(len(times)),
                    preprocess_version=np.array(PREPROCESS_VERSION),
                    source_h5=np.array(str(h5_path)))
                os.replace(tmp, outp)
            rows.append({"path": str(outp), "label": int(labels[i]), "index": int(i)})
            if (i + 1) % 1000 == 0:
                print(f"  cache {split}: {i+1}/{n}")
    df = pd.DataFrame(rows)
    df.to_csv(manifest_path, index=False)
    return df


def load_or_build_manifest(split):
    mp = OUT / f"{split}_tensor_manifest.csv"
    if mp.exists() and not FORCE_REBUILD_CACHE:
        df = pd.read_csv(mp)
        if len(df) and Path(df.iloc[0].path).exists():
            with np.load(df.iloc[0].path) as d:
                ver = str(d["preprocess_version"]) if "preprocess_version" in d.files else ""
                shape = tuple(d["X"].shape)
            if ver != PREPROCESS_VERSION or shape != (T_FINE, 1, N_FREQ):
                raise RuntimeError(
                    "legacy/incompatible HSD cache detected; set FORCE_REBUILD_CACHE=1")
            return df
    return build_cache(split, H5_PATHS[split])


train_manifest = load_or_build_manifest("train")
test_manifest = load_or_build_manifest("test")
valid_manifest = load_or_build_manifest("valid") if "valid" in H5_PATHS else None

print("cache sizes:", len(train_manifest), "/",
      (len(valid_manifest) if valid_manifest is not None else "no-valid"), "/", len(test_manifest))

# Official split sizes, hard-asserted. SHD values verified empirically against
# the current MD5-matching zenkelab files (2026-09-22); the older 8332/2088
# figures on IEEE DataPort describe a previous release. SSC values are the
# published official split (75466/9981/20382).
EXPECTED_SPLIT_SIZES = {
    "SHD": {"train": 8156, "test": 2264},
    "SSC": {"train": 75466, "valid": 9981, "test": 20382},
}
_actual_sizes = {"train": len(train_manifest), "test": len(test_manifest)}
if valid_manifest is not None:
    _actual_sizes["valid"] = len(valid_manifest)
for _split, _want in EXPECTED_SPLIT_SIZES[DATASET].items():
    _got = _actual_sizes.get(_split)
    if _got != _want:
        raise AssertionError(
            f"official split size mismatch for {DATASET} {_split}: "
            f"expected {_want}, cache has {_got}. MD5 passed, so this points at "
            "extraction or cache corruption; set FORCE_REBUILD_CACHE=1.")
print("official split sizes OK:", EXPECTED_SPLIT_SIZES[DATASET])
print("train class counts:", train_manifest.label.value_counts().sort_index().to_dict())
print("test class counts:", test_manifest.label.value_counts().sort_index().to_dict())


# ============================================================
# 3. PyTorch data
# ============================================================
class HSDTensorDataset(Dataset):
    def __init__(self, manifest, train=False):
        self.df = manifest.reset_index(drop=True)
        self.train = train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        d = np.load(self.df.iloc[idx].path)
        X = d["X"].astype(np.float32)
        # Dataset-specific analogue of small spatial jitter: a small cochlear
        # channel translation. Training-only; spikes rolled past the edge are
        # dropped, so counts are NOT exactly preserved here — acceptable as a
        # training augmentation, never applied to eval or attack tensors.
        if self.train:
            if np.random.rand() < 0.5:
                shift = int(np.random.randint(-4, 5))
                if shift:
                    X = np.roll(X, shift=shift, axis=-1)
                    if shift > 0:
                        X[..., :shift] = 0
                    else:
                        X[..., shift:] = 0
            # Tiny global time translation, analogous to visual translation.
            if np.random.rand() < 0.25:
                dt = int(np.random.randint(-2, 3))
                if dt:
                    X = np.roll(X, shift=dt, axis=0)
                    if dt > 0:
                        X[:dt] = 0
                    else:
                        X[dt:] = 0
        return (torch.from_numpy(X),
                torch.tensor(int(d["label"]), dtype=torch.long),
                torch.tensor(float(d["duration_s"]), dtype=torch.float32))


train_ds = HSDTensorDataset(train_manifest, train=True)
test_ds = HSDTensorDataset(test_manifest, train=False)
valid_ds = HSDTensorDataset(valid_manifest, train=False) if valid_manifest is not None else None


def make_loader(ds, batch, shuffle):
    if ds is None:
        return None
    workers = int(os.environ.get("NUM_WORKERS", "6"))
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=workers,
                      pin_memory=True, persistent_workers=(workers > 0))


test_loader = make_loader(test_ds, EVAL_BATCH, False)
valid_loader = make_loader(valid_ds, EVAL_BATCH, False) if valid_ds is not None else None


def autocast_context():
    if DEVICE.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return torch.autocast(device_type="cpu", enabled=False)


# ============================================================
# 4. Attack primitives — exact analogue of visual runs
# ============================================================

def grouped(X):
    # (K,S,1,F) -> (K*1*F,S). Each row is one channel in one protected window.
    return (X.reshape(N_COARSE, S_PER_FRAME, 1, N_FREQ)
             .transpose(0, 2, 3, 1).reshape(-1, S_PER_FRAME))


def ungrouped(Xg):
    return (Xg.reshape(N_COARSE, 1, N_FREQ, S_PER_FRAME)
              .transpose(0, 3, 1, 2).reshape(T_FINE, 1, N_FREQ))


def coarse_np(X):
    return X.reshape(N_COARSE, S_PER_FRAME, 1, N_FREQ).sum(axis=1)


def coarse_equal(a, b):
    return np.array_equal(coarse_np(a), coarse_np(b))


def batch_coarse_metrics(A, B):
    Ac = A.reshape(len(A), N_COARSE, S_PER_FRAME, 1, N_FREQ).sum(axis=2).astype(np.int64)
    Bc = B.reshape(len(B), N_COARSE, S_PER_FRAME, 1, N_FREQ).sum(axis=2).astype(np.int64)
    diff = np.abs(Ac - Bc)
    return {
        "frame_exact_equal": bool(np.array_equal(Ac, Bc)),
        "max_integer_frame_difference": int(diff.max()) if diff.size else 0,
    }


def per_sample_rep_distortion(A, B):
    Ac = A.reshape(len(A), N_COARSE, S_PER_FRAME, 1, N_FREQ).sum(axis=2).astype(np.int64)
    Bc = B.reshape(len(B), N_COARSE, S_PER_FRAME, 1, N_FREQ).sum(axis=2).astype(np.int64)
    diff = np.abs(Bc - Ac)
    num = diff.reshape(len(A), -1).sum(axis=1).astype(np.float64)
    den = np.abs(Ac).reshape(len(A), -1).sum(axis=1).astype(np.float64)
    DA = np.divide(num, den, out=np.full(len(A), np.inf), where=den > 0)
    DA[(den == 0) & (num == 0)] = 0.0
    Dinf = diff.reshape(len(A), -1).max(axis=1).astype(np.float64)
    return DA, Dinf


@torch.no_grad()
def predict_logits(model, X_np, batch=EVAL_BATCH):
    model.eval()
    outs = []
    for i in range(0, len(X_np), batch):
        xb = torch.from_numpy(X_np[i:i+batch].astype(np.float32)).to(DEVICE)
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
    yt = torch.tensor([int(y)], dtype=torch.long, device=DEVICE)
    if _has_rnn(model):
        with torch.backends.cudnn.flags(enabled=False):
            g = torch.autograd.grad(F.cross_entropy(model(x), yt), x)[0][0]
    else:
        g = torch.autograd.grad(F.cross_entropy(model(x), yt), x)[0][0]
    return g.detach().cpu().numpy()


def _select_moves(Xg, Gg, Ag, n_moves, S, max_shift_bins=None):
    g_idx, s_idx = np.nonzero(Ag > 0)
    if len(g_idx) == 0 or n_moves <= 0:
        return Xg, Ag, np.asarray([], dtype=np.int16)
    s_all = np.arange(S)
    if max_shift_bins is None:
        band = np.ones((S, S), dtype=bool)
    else:
        band = np.abs(s_all[None, :] - s_all[:, None]) <= int(max_shift_bins)
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
        take[cut] = max(0, n_moves - (csum[cut-1] if cut > 0 else 0))
        take[cut+1:] = 0
    Xg[g_idx, s_idx] -= take
    np.add.at(Xg, (g_idx, dst), take)
    Ag[g_idx, s_idx] -= take
    return Xg, Ag, np.repeat(dst - s_idx, take).astype(np.int16)


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
    return X.reshape(T_FINE, N_FREQ).T.copy()


def ungrouped_free(Xg):
    return Xg.T.reshape(T_FINE, 1, N_FREQ)


def progressive_free_unique_attack(model, X0, y, budgets):
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
            f = int(flow[i, n_sh+b])
            if f > 0:
                out[(b, d)] = f
    return out


def exact_displacement_matched_random_attack(X0, target_shifts, rng):
    target = np.asarray(target_shifts, dtype=np.int16)
    target = target[target != 0]
    if len(target) == 0:
        return X0.copy().astype(np.int16), np.empty(0, dtype=np.int8), True
    if int(np.abs(target).max()) >= S_PER_FRAME:
        raise ValueError("target shift exceeds protected window")
    unit_g, unit_s = event_units_from_original(X0)
    source_counts = np.bincount(unit_s.astype(np.int64), minlength=S_PER_FRAME)
    shift_counts = Counter(int(d) for d in target.tolist())
    alloc = _exact_shift_source_flow(source_counts, shift_counts, S_PER_FRAME, rng)
    if alloc is None:
        raise RuntimeError("no feasible exact signed-displacement matching")
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
            chosen = pool[pos:pos+n]
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
    den = 1 + z*z/n
    cen = (ph + z*z/(2*n)) / den
    hw = z * math.sqrt(ph*(1-ph)/n + z*z/(4*n*n)) / den
    return max(0.0, cen-hw), min(1.0, cen+hw)


def attack_metrics(pred, clean_pred, y, clean_correct):
    n_cc = int(clean_correct.sum())
    k = int((pred[clean_correct] != y[clean_correct]).sum()) if n_cc else 0
    lo, hi = wilson95(k, n_cc)
    return {
        "attacked_accuracy": float((pred == y).mean()),
        "asr_clean_correct": (k / n_cc) if n_cc else float("nan"),
        "asr_ci95_low": lo, "asr_ci95_high": hi,
        "n_clean_correct": n_cc,
        "prediction_flip_rate": float((pred != clean_pred).mean()),
    }


# ============================================================
# 5. Model families — 1-D cochlear adaptations of same registry
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


def td_apply_1d(module, x):
    B, T = x.shape[:2]
    z = module(x.reshape(B*T, *x.shape[2:]))
    return z.reshape(B, T, *z.shape[1:])


class TDConvBN1D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.op = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 5, stride=stride, padding=2, bias=False),
            nn.BatchNorm1d(out_ch))

    def forward(self, x):
        return td_apply_1d(self.op, x)


class ConvSNN(nn.Module):
    def __init__(self, beta=0.90):
        super().__init__()
        self.beta = beta
        self.c1 = TDConvBN1D(1, 32, 2)
        self.c2 = TDConvBN1D(32, 64, 2)
        self.c3 = TDConvBN1D(64, 128, 2)
        self.c4 = TDConvBN1D(128, 256, 2)
        self.head = nn.Linear(256, NUM_CLASSES)

    def forward(self, X):
        x = X * 0.15
        for c in (self.c1, self.c2, self.c3, self.c4):
            x = lif_sequence(c(x), self.beta)
        return self.head(x.mean(dim=-1).mean(dim=1))


class SEWBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, beta=0.90):
        super().__init__()
        self.beta = beta
        self.conv1 = TDConvBN1D(in_ch, out_ch, stride)
        self.conv2 = TDConvBN1D(out_ch, out_ch, 1)
        self.skip = (nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
            nn.BatchNorm1d(out_ch)) if stride != 1 or in_ch != out_ch else None)

    def forward(self, x):
        residual = x
        z = lif_sequence(self.conv1(x), self.beta)
        z = lif_sequence(self.conv2(z), self.beta)
        if self.skip is not None:
            residual = lif_sequence(td_apply_1d(self.skip, residual), self.beta)
        return z + residual


class SEWResNet18(nn.Module):
    def __init__(self, beta=0.90):
        super().__init__()
        self.beta = beta
        self.stem = TDConvBN1D(1, 32, 2)
        self.s1 = nn.Sequential(SEWBlock1D(32,32,1,beta), SEWBlock1D(32,32,1,beta))
        self.s2 = nn.Sequential(SEWBlock1D(32,64,2,beta), SEWBlock1D(64,64,1,beta))
        self.s3 = nn.Sequential(SEWBlock1D(64,128,2,beta), SEWBlock1D(128,128,1,beta))
        self.s4 = nn.Sequential(SEWBlock1D(128,256,2,beta), SEWBlock1D(256,256,1,beta))
        self.fc = nn.Linear(256, NUM_CLASSES)

    def forward(self, X):
        x = lif_sequence(self.stem(X * 0.15), self.beta)
        x = self.s1(x); x = self.s2(x); x = self.s3(x); x = self.s4(x)
        return self.fc(x.mean(dim=-1).mean(dim=1))


class GNResBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        g = min(16, out_ch)
        while out_ch % g != 0:
            g -= 1
        self.conv1 = nn.Conv1d(in_ch, out_ch, 5, stride=stride, padding=2, bias=False)
        self.gn1 = nn.GroupNorm(g, out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(g, out_ch)
        self.skip = (nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False)
                     if stride != 1 or in_ch != out_ch else nn.Identity())

    def forward(self, x):
        r = self.skip(x)
        x = F.gelu(self.gn1(self.conv1(x)))
        x = self.gn2(self.conv2(x))
        return F.gelu(x + r)


class EventTemporalTransformerV2(nn.Module):
    def __init__(self, d_model=384, nhead=8, depth=8, mlp_ratio=4, dropout=0.10):
        super().__init__()
        self.spectral = nn.Sequential(
            nn.Conv1d(1, 48, 7, stride=2, padding=3, bias=False),
            nn.GroupNorm(8, 48), nn.GELU(),
            GNResBlock1D(48, 64, 1), GNResBlock1D(64, 96, 2),
            GNResBlock1D(96, 128, 2), GNResBlock1D(128, 192, 2),
            nn.AdaptiveAvgPool1d(1))
        self.token_proj = nn.Sequential(nn.Linear(192, d_model), nn.LayerNorm(d_model))
        self.temporal_dwconv = nn.Conv1d(d_model, d_model, 5, padding=2,
                                         groups=d_model, bias=False)
        self.temporal_pwconv = nn.Conv1d(d_model, d_model, 1, bias=False)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos = nn.Parameter(torch.zeros(1, T_FINE + 1, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*mlp_ratio,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.out_norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(2*d_model, d_model), nn.GELU(), nn.Dropout(0.20),
            nn.Linear(d_model, NUM_CLASSES))
        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, X):
        B, T = X.shape[:2]
        z = torch.log1p(X).reshape(B*T, 1, N_FREQ)
        z = self.spectral(z).flatten(1)
        z = self.token_proj(z).reshape(B, T, -1)
        local = self.temporal_pwconv(F.gelu(self.temporal_dwconv(z.transpose(1,2))))
        z = z + local.transpose(1,2)
        z = torch.cat([self.cls.expand(B,-1,-1), z], dim=1)
        z = z + self.pos[:, :T+1]
        z = self.out_norm(self.encoder(z))
        return self.head(torch.cat([z[:,0], z[:,1:].mean(dim=1)], dim=1))


class TemporalGRU_DVS(nn.Module):
    def __init__(self, hidden=256):
        super().__init__()
        self.spectral = nn.Sequential(
            nn.Conv1d(1,32,7,stride=2,padding=3), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32,64,5,stride=2,padding=2), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64,128,3,stride=2,padding=1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1))
        self.gru = nn.GRU(128, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, NUM_CLASSES)

    def forward(self, X):
        z = td_apply_1d(self.spectral, X * 0.15).flatten(2)
        _, h = self.gru(z)
        return self.fc(h[-1])


class CoarseSpectralBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch,out_ch,5,stride=stride,padding=2,bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch,out_ch,3,padding=1,bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.skip = (nn.Sequential(nn.Conv1d(in_ch,out_ch,1,stride=stride,bias=False),
                                   nn.BatchNorm1d(out_ch))
                     if stride != 1 or in_ch != out_ch else nn.Identity())

    def forward(self, x):
        r = self.skip(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + r)


class CoarseFrameFormer(nn.Module):
    """Consumes only the protected ten-window cochlear count sequence."""
    def __init__(self, d_model=320, nhead=8, depth=4, dropout=0.10):
        super().__init__()
        self.spectral = nn.Sequential(
            nn.Conv1d(1,64,7,stride=2,padding=3,bias=False),
            nn.BatchNorm1d(64), nn.ReLU(),
            CoarseSpectralBlock(64,64,1), CoarseSpectralBlock(64,128,2),
            CoarseSpectralBlock(128,192,2), CoarseSpectralBlock(192,256,2),
            nn.AdaptiveAvgPool1d(1))
        self.proj = nn.Linear(256, d_model)
        self.cls = nn.Parameter(torch.zeros(1,1,d_model))
        self.pos = nn.Parameter(torch.zeros(1,N_COARSE+1,d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4*d_model,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(2*d_model,d_model), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(d_model, NUM_CLASSES))
        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, X):
        B = X.shape[0]
        coarse = X.reshape(B,N_COARSE,S_PER_FRAME,1,N_FREQ).sum(dim=2)
        z = torch.log1p(coarse).reshape(B*N_COARSE,1,N_FREQ)
        z = self.spectral(z).flatten(1)
        z = self.proj(z).reshape(B,N_COARSE,-1)
        z = torch.cat([self.cls.expand(B,-1,-1), z], dim=1) + self.pos
        z = self.norm(self.encoder(z))
        return self.head(torch.cat([z[:,0], z[:,1:].mean(dim=1)], dim=1))


# Same optimizer/scheduler recipes and epochs/batches as frozen CIFAR/N-Caltech.
MODEL_SPECS = {
    "coarse_frameformer": dict(builder=CoarseFrameFormer, epochs=50, batch=48,
        lr=4e-4, wd=3e-3, clip=2.0, label_smoothing=0.05,
        betas=(0.9,0.95), schedule="warmup_cosine_steps", warmup_frac=0.05),
    "conv_snn": dict(builder=ConvSNN, epochs=45, batch=24,
        lr=2e-3, wd=1e-4, clip=5.0, label_smoothing=0.0,
        betas=(0.9,0.999), schedule="epoch_cosine", warmup_frac=0.0),
    "sew_resnet18": dict(builder=SEWResNet18, epochs=55, batch=12,
        lr=2e-3, wd=1e-4, clip=5.0, label_smoothing=0.0,
        betas=(0.9,0.999), schedule="epoch_cosine", warmup_frac=0.0),
    "event_transformer_v2": dict(builder=EventTemporalTransformerV2, epochs=50, batch=24,
        lr=3e-4, wd=5e-3, clip=1.0, label_smoothing=0.05,
        betas=(0.9,0.95), schedule="warmup_cosine_steps", warmup_frac=0.08),
    "temporal_gru": dict(builder=TemporalGRU_DVS, epochs=30, batch=32,
        lr=1e-3, wd=1e-4, clip=5.0, label_smoothing=0.0,
        betas=(0.9,0.999), schedule="warmup_cosine_steps", warmup_frac=0.02),
}

for m in TRAIN_MODELS + ATTACK_MODELS:
    if m not in MODEL_SPECS:
        raise ValueError(f"unknown model {m}")

print("\nmodel smoke tests:")
smoke = torch.zeros(2, T_FINE, 1, N_FREQ, device=DEVICE)
for name in OFFICIAL_MODELS:
    m = MODEL_SPECS[name]["builder"]().to(DEVICE)
    with torch.no_grad(), autocast_context():
        z = m(smoke)
    assert z.shape == (2, NUM_CLASSES), (name, z.shape)
    print(f"  {name}: ok ({sum(p.numel() for p in m.parameters())/1e6:.2f}M params)")
    del m
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()


# ============================================================
# 6. Training — fixed schedule, resumable, no accuracy gate
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
    return correct / max(1, total)


def warmup_cosine_step(step, total_steps, warmup_steps):
    if step < warmup_steps:
        return max(1e-6, step / max(1, warmup_steps))
    p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, p)))


def epoch_cosine_multiplier(epoch, total_epochs):
    return 0.5 * (1.0 + math.cos(math.pi * epoch / max(1, total_epochs)))


def train_model(name, seed, force=False):
    spec = MODEL_SPECS[name]
    set_seed(seed)
    ckpt = CKPT_ROOT / f"{name}_seed{seed}_{PROTOCOL_TAG}.pt"
    part = CKPT_ROOT / f"{name}_seed{seed}_{PROTOCOL_TAG}_partial.pt"
    hist = CKPT_ROOT / f"{name}_seed{seed}_{PROTOCOL_TAG}_history.csv"
    model = spec["builder"]().to(DEVICE)
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
            start_ep, gstep = int(st["epoch"]), int(st["gstep"])
            history = st.get("history", [])
            print(f"resuming {name} seed {seed} at epoch {start_ep}")
        except Exception as e:
            print("partial unreadable; restarting:", repr(e))

    for ep in range(start_ep, spec["epochs"]):
        model.train()
        losses, correct, total = [], 0, 0
        t0 = time.time()
        if spec["schedule"] == "epoch_cosine":
            lr_now = spec["lr"] * epoch_cosine_multiplier(ep, spec["epochs"])
            for pg in opt.param_groups:
                pg["lr"] = lr_now
        for X, y, _ in loader:
            X = X.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
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
            correct += int((logits.detach().argmax(1) == y).sum())
            total += len(y)

        # SSC has an official validation split. SHD has none; do not touch the
        # test set during training merely to create a monitoring number.
        monitor_acc = evaluate(model, valid_loader) if valid_loader is not None else float("nan")
        row = {
            "epoch": ep+1, "loss": float(np.mean(losses)),
            "train_acc": correct/max(1,total),
            "monitor_split": "valid" if valid_loader is not None else "none",
            "monitor_acc": monitor_acc, "lr": float(lr_now),
            "seconds": time.time()-t0,
        }
        history.append(row)
        mon = f" valid={monitor_acc:.4f}" if valid_loader is not None else ""
        print(f"{name} seed={seed} ep={ep+1:02d}/{spec['epochs']} "
              f"loss={row['loss']:.4f} train={row['train_acc']:.4f}{mon} "
              f"t={row['seconds']:.0f}s")
        pd.DataFrame(history).to_csv(hist, index=False)
        tmp = part.with_suffix(".pt.tmp")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "epoch": ep+1, "gstep": gstep, "history": history}, tmp)
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
        clean_rows.append({
            "seed": seed, "model": name, "clean_test_accuracy": float(acc),
            "asr_interpretable": bool(acc >= MIN_CLEAN_INTERPRET),
        })
        print("FINAL", name, "seed", seed, "test_acc", round(acc, 4))
        if acc < CLEAN_WARN:
            print(f"*** WARNING ONLY: clean accuracy {acc:.3f} < {CLEAN_WARN}; "
                  "model remains in the frozen protocol but ASR interpretation is flagged.")
        del model; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

clean_df = pd.DataFrame(clean_rows)
clean_df.to_csv(OUT / "clean_accuracy_by_seed.csv", index=False)
if len(clean_df):
    print(clean_df.to_string(index=False))
    print(clean_df.groupby("model")["clean_test_accuracy"].agg(["mean","std"]).round(4))
else:
    print("no models trained in this scoped run; clean_accuracy_by_seed.csv is empty")


# ============================================================
# 7. Attack subset and full attack protocol
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
    order = np.argsort(-(raw - np.floor(raw)))
    i = 0
    while rem > 0:
        j = order[i % len(order)]
        if q[j] < avail[j]:
            q[j] += 1; rem -= 1
        i += 1
        if i > 50 * len(order):
            raise RuntimeError("quota allocation failed")
    return labels, avail, q


if RUN_ATTACKS:
    labels, avail, quotas = proportional_attack_quotas(test_manifest, ATTACK_TOTAL)
    rng = np.random.default_rng(9090)
    parts = []
    for lab, q in zip(labels, quotas):
        sub = test_manifest[test_manifest.label == lab].reset_index(drop=True)
        pick = rng.choice(len(sub), size=int(q), replace=False)
        parts.append(sub.iloc[pick])
    attack_manifest = (pd.concat(parts).sample(frac=1, random_state=9091)
                       .reset_index(drop=True))
    if len(attack_manifest) != ATTACK_TOTAL:
        raise RuntimeError("attack subset size mismatch")
    attack_manifest.to_csv(OUT / "attack_subset_manifest.csv", index=False)
    pd.DataFrame({"label": labels, "test_available": avail,
                  "attack_selected": quotas}).to_csv(
        OUT / "attack_subset_counts_by_class.csv", index=False)

    Xattack, yattack, duration_s = [], [], []
    for _, r in attack_manifest.iterrows():
        d = np.load(r.path)
        Xattack.append(d["X"].astype(np.int16))
        yattack.append(int(d["label"]))
        duration_s.append(float(d["duration_s"]))
    Xattack = np.stack(Xattack)
    yattack = np.asarray(yattack, np.int64)
    duration_s = np.asarray(duration_s, np.float64)
    bin_ms = duration_s / T_FINE * 1000.0
    print("attack set:", Xattack.shape, "| classes:", len(np.unique(yattack)))

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
        with open(tmp, "w") as f:
            json.dump(payload, f)
        os.replace(tmp, _unit_fp(tag))

    def load_model(name, seed):
        m = MODEL_SPECS[name]["builder"]().to(DEVICE)
        fp = CKPT_ROOT / f"{name}_seed{seed}_{PROTOCOL_TAG}.pt"
        if not fp.exists():
            raise FileNotFoundError(fp)
        m.load_state_dict(torch.load(fp, map_location=DEVICE))
        m.eval()
        return m

    clean_map = {(r["model"], r["seed"]): r["clean_test_accuracy"] for r in clean_rows}
    main_rows, verify_rows, main_sample_rows = [], [], []

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
                main_sample_rows += cached.get("sample_rows", [])
                continue

            model = load_model(name, seed)
            clean_pred = predict_logits(model, Xattack).argmax(1)
            clean_correct = clean_pred == yattack
            full_clean = clean_map[(name, seed)]
            print(f"\n[{name} seed {seed}] subset clean={clean_correct.mean():.4f} "
                  f"full-test={full_clean:.4f}")

            adv = {b: [] for b in ATTACK_BUDGETS}
            shf = {b: [] for b in ATTACK_BUDGETS}
            t0 = time.time()
            for i in range(len(Xattack)):
                ck = progressive_unique_attack(model, Xattack[i], yattack[i], ATTACK_BUDGETS)
                for b in ATTACK_BUDGETS:
                    adv[b].append(ck[float(b)]["X"])
                    shf[b].append(ck[float(b)]["shifts"])
                if (i+1) % 100 == 0:
                    print(f"  gradient {i+1}/{len(Xattack)} ({time.time()-t0:.0f}s)")

            u_main, u_verify, u_sample = [], [], []
            for b in ATTACK_BUDGETS:
                b = float(b)
                Xadv = np.stack(adv[b])
                pred = predict_logits(model, Xadv).argmax(1)
                mean_ms = [float(np.abs(s).mean())*bin_ms[i] if len(s) else 0.0
                           for i,s in enumerate(shf[b])]
                DA, Dinf = per_sample_rep_distortion(Xattack, Xadv)
                row = {
                    "seed": seed, "model": name, "attack": "gradient_unique",
                    "budget": b, "n_attack": len(Xattack),
                    "clean_accuracy_full_test": full_clean,
                    "asr_interpretable": bool(full_clean >= MIN_CLEAN_INTERPRET),
                    **attack_metrics(pred, clean_pred, yattack, clean_correct),
                    "mean_realized_unique_fraction": float(np.mean([
                        len(shf[b][i]) / max(1, int(Xattack[i].sum()))
                        for i in range(len(Xattack))])),
                    "mean_abs_shift_ms": float(np.mean(mean_ms)),
                    "mean_D_A": float(np.mean(DA)), "max_D_A": float(np.max(DA)),
                    "mean_D_inf": float(np.mean(Dinf)), "max_D_inf": float(np.max(Dinf)),
                    **batch_coarse_metrics(Xattack, Xadv),
                }
                c2a = predict_logits(coarse2, Xadv)
                row["coarse_frameformer_logits_exact"] = bool(np.array_equal(c2a, c2_clean))
                row["coarse_frameformer_flip_rate"] = float(
                    (c2a.argmax(1) != c2_clean_pred).mean())
                u_main.append(row)

                # Per-sample rows for the full main grid (all seeds/victims),
                # consumed by the stealth-constrained SR@tau analysis.
                c2a_pred = c2a.argmax(1)
                for i in range(len(Xattack)):
                    n_ev = max(1, int(Xattack[i].sum()))
                    u_sample.append({
                        "seed": seed, "model": name, "attack": "gradient_unique",
                        "budget": b, "sample_index": int(i),
                        "label": int(yattack[i]),
                        "clean_correct": bool(clean_correct[i]),
                        "attack_success": (bool(pred[i] != yattack[i])
                                           if clean_correct[i] else False),
                        "D_A": float(DA[i]), "D_inf": float(Dinf[i]),
                        "coarse_exact": bool(DA[i] == 0.0 and Dinf[i] == 0.0),
                        "protected_prediction_flip": bool(c2a_pred[i] != c2_clean_pred[i]),
                        "moved_events": int(len(shf[b][i])),
                        "realized_fraction": float(len(shf[b][i]) / n_ev),
                        "mean_abs_shift_ms": float(mean_ms[i]),
                    })

                # Uniform within-window control.
                r_u = np.random.default_rng(310000 + seed*1000 + int(b*10000))
                Xu = []
                for i in range(len(Xattack)):
                    n_mv = len(shf[b][i])
                    ug, us = event_units_from_original(Xattack[i])
                    n_mv = min(n_mv, len(ug))
                    chosen = r_u.permutation(len(ug))[:n_mv]
                    Xg = grouped(Xattack[i]).astype(np.int32)
                    gs, ss = ug[chosen], us[chosen].astype(np.int64)
                    ds = (ss + r_u.integers(1, S_PER_FRAME, size=n_mv)) % S_PER_FRAME
                    np.subtract.at(Xg, (gs,ss), 1)
                    np.add.at(Xg, (gs,ds), 1)
                    Xr = ungrouped(Xg).astype(np.int16)
                    assert Xr.min() >= 0 and coarse_equal(Xattack[i], Xr)
                    Xu.append(Xr)
                Xu = np.stack(Xu)
                pu = predict_logits(model, Xu).argmax(1)
                c2u = predict_logits(coarse2, Xu)
                DAu, Dinfu = per_sample_rep_distortion(Xattack, Xu)
                u_main.append({
                    "seed":seed,"model":name,"attack":"random_uniform","budget":b,
                    "n_attack":len(Xattack),"clean_accuracy_full_test":full_clean,
                    "asr_interpretable":bool(full_clean>=MIN_CLEAN_INTERPRET),
                    **attack_metrics(pu,clean_pred,yattack,clean_correct),
                    "mean_D_A":float(np.mean(DAu)),"max_D_A":float(np.max(DAu)),
                    "mean_D_inf":float(np.mean(Dinfu)),"max_D_inf":float(np.max(Dinfu)),
                    **batch_coarse_metrics(Xattack,Xu),
                    "coarse_frameformer_logits_exact":bool(np.array_equal(c2u,c2_clean)),
                    "coarse_frameformer_flip_rate":float((c2u.argmax(1)!=c2_clean_pred).mean()),
                })

                # Exact displacement-matched random control.
                r_m = np.random.default_rng(320000 + seed*1000 + int(b*10000))
                Xm, n_exact = [], 0
                for i in range(len(Xattack)):
                    xm, sm, ok = exact_displacement_matched_random_attack(
                        Xattack[i], shf[b][i], r_m)
                    n_exact += int(ok); Xm.append(xm)
                Xm = np.stack(Xm)
                pm = predict_logits(model, Xm).argmax(1)
                c2m = predict_logits(coarse2, Xm)
                DAm, Dinfm = per_sample_rep_distortion(Xattack, Xm)
                u_main.append({
                    "seed":seed,"model":name,
                    "attack":"random_displacement_matched_exact","budget":b,
                    "n_attack":len(Xattack),"clean_accuracy_full_test":full_clean,
                    "asr_interpretable":bool(full_clean>=MIN_CLEAN_INTERPRET),
                    **attack_metrics(pm,clean_pred,yattack,clean_correct),
                    "mean_abs_shift_ms":float(np.mean(mean_ms)),
                    "mean_D_A":float(np.mean(DAm)),"max_D_A":float(np.max(DAm)),
                    "mean_D_inf":float(np.mean(Dinfm)),"max_D_inf":float(np.max(Dinfm)),
                    **batch_coarse_metrics(Xattack,Xm),
                    "exact_match_fraction":float(n_exact/len(Xattack)),
                    "coarse_frameformer_logits_exact":bool(np.array_equal(c2m,c2_clean)),
                    "coarse_frameformer_flip_rate":float((c2m.argmax(1)!=c2_clean_pred).mean()),
                })
                u_verify.append({"seed":seed,"model":name,"budget":b,
                                 "samples":len(Xattack),
                                 "signed_histogram_exact":n_exact,
                                 "fallback":len(Xattack)-n_exact})
                print(f"  b={b}: grad {u_main[-3]['asr_clean_correct']:.3f} | "
                      f"unif {u_main[-2]['asr_clean_correct']:.3f} | "
                      f"exact {u_main[-1]['asr_clean_correct']:.3f}")
                del Xadv, Xu, Xm; gc.collect()

            _save_unit(tag, {"main_rows": u_main, "verify_rows": u_verify,
                             "sample_rows": u_sample})
            main_rows += u_main; verify_rows += u_verify
            main_sample_rows += u_sample
            del model; gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        del coarse2; gc.collect()

    mv = pd.DataFrame(main_rows)
    mv.to_csv(OUT / "main_attack_results_by_seed.csv", index=False)
    pd.DataFrame(main_sample_rows).to_csv(
        OUT / "main_attack_per_sample.csv", index=False)
    pd.DataFrame(verify_rows).to_csv(OUT / "exact_match_verification.csv", index=False)
    agg = (mv.groupby(["model","attack","budget"])[["attacked_accuracy","asr_clean_correct"]]
             .agg(["mean","std"]))
    agg.to_csv(OUT / "main_attack_aggregate.csv")
    print(agg.round(4).to_string())
    print("coarse exact all:", bool(mv.frame_exact_equal.all()),
          "| maxdiff:", int(mv.max_integer_frame_difference.max()),
          "| protected logits exact all:", bool(mv.coarse_frameformer_logits_exact.all()))
    # Every arm in this protocol is window-preserving, so both invariants are
    # requirements of the run, not diagnostics. CSVs are already on disk above,
    # so a violation stops the run without destroying evidence.
    if (not bool(mv.frame_exact_equal.all())) or \
            int(mv.max_integer_frame_difference.max()) != 0:
        raise AssertionError(
            "PROTOCOL INVARIANT VIOLATED: nonzero integer coarse-representation "
            "difference in the main grid; inspect main_attack_results_by_seed.csv")
    if not bool(mv.coarse_frameformer_logits_exact.all()):
        raise AssertionError(
            "PROTOCOL INVARIANT VIOLATED: protected-consumer logits not "
            "bit-identical on some row; inspect main_attack_results_by_seed.csv")

    # --------------------------------------------------------
    # Max-shift diagnostic (seed 0, same progressive path)
    # --------------------------------------------------------
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
                    ck = progressive_unique_attack(model, Xattack[i], yattack[i],
                                                   ABLATION_PATH, max_shift_bins=ms)
                    Xa.append(ck[SHIFT_ABLATION_BUDGET]["X"])
                    sh.append(ck[SHIFT_ABLATION_BUDGET]["shifts"])
                Xa = np.stack(Xa)
                pred = predict_logits(model, Xa).argmax(1)
                DA, Dinf = per_sample_rep_distortion(Xattack, Xa)
                mean_ms = [float(np.abs(s).mean())*bin_ms[i] if len(s) else 0.0
                           for i,s in enumerate(sh)]
                rows.append({
                    "seed":DIAGNOSTIC_SEED,"model":name,
                    "budget":SHIFT_ABLATION_BUDGET,"n_attack":len(Xattack),
                    "optimization_path":"->".join(f"{x:g}" for x in ABLATION_PATH),
                    "max_shift_label":"full" if ms is None else str(ms),
                    "mean_abs_shift_ms":float(np.mean(mean_ms)),
                    "mean_D_A":float(np.mean(DA)),"max_D_A":float(np.max(DA)),
                    **attack_metrics(pred,clean_pred,yattack,clean_correct),
                    **batch_coarse_metrics(Xattack,Xa),
                })
                print(f"  ablation {name} shift={rows[-1]['max_shift_label']}: "
                      f"ASR={rows[-1]['asr_clean_correct']:.3f}")
                del Xa; gc.collect()
            _save_unit(tag,{"rows":rows}); abl_rows += rows
            del model; gc.collect()
        abl_df = pd.DataFrame(abl_rows)
        abl_df.to_csv(OUT / "max_shift_ablation.csv", index=False)

        checks = []
        for name in abl_df.model.unique():
            old = mv[(mv.model==name)&(mv.seed==DIAGNOSTIC_SEED)&
                     (mv.attack=="gradient_unique")&np.isclose(mv.budget,0.10)]
            new = abl_df[(abl_df.model==name)&(abl_df.max_shift_label=="full")]
            if len(old)==1 and len(new)==1:
                a=float(old.iloc[0].asr_clean_correct); bb=float(new.iloc[0].asr_clean_correct)
                checks.append({"model":name,"main_10pct_asr":a,"ablation_full_asr":bb,
                               "absolute_difference_pct_points":100*abs(a-bb),
                               "consistent_within_0p5pt":bool(abs(a-bb)<=0.005)})
        pd.DataFrame(checks).to_csv(OUT / "max_shift_ablation_main_check.csv", index=False)

    # --------------------------------------------------------
    # Cost of stealth + per-sample D_A / D_inf for SC-ASR(tau)
    # --------------------------------------------------------
    if RUN_COST_OF_STEALTH:
        cost_rows, sample_rows = [], []
        n_sub = min(COST_SUBSET, len(Xattack))
        Xs, ys, bms = Xattack[:n_sub], yattack[:n_sub], bin_ms[:n_sub]
        for name in ATTACK_MODELS:
            tag = f"cost_{DIAG_TAG}_seed{DIAGNOSTIC_SEED}_{name}"
            cached = _load_unit(tag)
            if cached is not None:
                cost_rows += cached["rows"]
                sample_rows += cached.get("sample_rows", [])
                continue
            model = load_model(name, DIAGNOSTIC_SEED)
            coarse2 = load_model("coarse_frameformer", DIAGNOSTIC_SEED)
            clean_logits = predict_logits(model, Xs)
            clean_pred = clean_logits.argmax(1)
            clean_correct = clean_pred == ys
            c2c = predict_logits(coarse2, Xs)
            arms = {k:{b:[] for b in COST_BUDGETS} for k in ("stealth","free")}
            arm_sh = {k:{b:[] for b in COST_BUDGETS} for k in ("stealth","free")}
            for i in range(n_sub):
                ck_s = progressive_unique_attack(model, Xs[i], ys[i], COST_PATH)
                ck_f = progressive_free_unique_attack(model, Xs[i], ys[i], COST_PATH)
                for b in COST_BUDGETS:
                    arms["stealth"][b].append(ck_s[float(b)]["X"])
                    arm_sh["stealth"][b].append(ck_s[float(b)]["shifts"].astype(np.int16))
                    arms["free"][b].append(ck_f[float(b)]["X"])
                    arm_sh["free"][b].append(ck_f[float(b)]["shifts"])
                if (i+1)%50==0:
                    print(f"  cost {name}: {i+1}/{n_sub}")
            rows, srows = [], []
            for b in COST_BUDGETS:
                for kind in ("stealth","free"):
                    Xa = np.stack(arms[kind][b])
                    sh = arm_sh[kind][b]
                    logits = predict_logits(model, Xa)
                    pred = logits.argmax(1)
                    c2a = predict_logits(coarse2, Xa)
                    DA, Dinf = per_sample_rep_distortion(Xs, Xa)
                    mean_ms = [float(np.abs(s).mean())*bms[i] if len(s) else 0.0
                               for i,s in enumerate(sh)]
                    rows.append({
                        "seed":DIAGNOSTIC_SEED,"model":name,"kind":kind,
                        "budget":float(b),"n":n_sub,
                        "optimization_path":"->".join(f"{x:g}" for x in COST_PATH),
                        **attack_metrics(pred,clean_pred,ys,clean_correct),
                        "mean_abs_shift_ms":float(np.mean(mean_ms)),
                        "mean_D_A":float(np.mean(DA)),"max_D_A":float(np.max(DA)),
                        "mean_D_inf":float(np.mean(Dinf)),"max_D_inf":float(np.max(Dinf)),
                        **batch_coarse_metrics(Xs,Xa),
                        "coarse_frameformer_logits_exact":bool(np.array_equal(c2a,c2c)),
                        "coarse_frameformer_flip_rate":float((c2a.argmax(1)!=c2c.argmax(1)).mean()),
                    })
                    for i in range(n_sub):
                        srows.append({
                            "seed":DIAGNOSTIC_SEED,"model":name,"kind":kind,
                            "budget":float(b),"sample_index":int(i),
                            "label":int(ys[i]),"clean_correct":bool(clean_correct[i]),
                            "attack_success":bool(pred[i] != ys[i]) if clean_correct[i] else False,
                            "D_A":float(DA[i]),"D_inf":float(Dinf[i]),
                            "moved_events":int(len(sh[i])),
                            "realized_fraction":float(len(sh[i])/max(1,int(Xs[i].sum()))),
                            "mean_abs_shift_ms":float(mean_ms[i]),
                            "coarse_exact":bool(DA[i] == 0.0 and Dinf[i] == 0.0),
                            "protected_prediction_flip":bool(c2a.argmax(1)[i] != c2c.argmax(1)[i]),
                        })
                    del Xa; gc.collect()
            _save_unit(tag,{"rows":rows,"sample_rows":srows})
            cost_rows += rows; sample_rows += srows
            del model, coarse2, arms, arm_sh; gc.collect()
        pd.DataFrame(cost_rows).to_csv(OUT / "cost_of_stealth.csv", index=False)
        pd.DataFrame(sample_rows).to_csv(OUT / "cost_of_stealth_per_sample.csv", index=False)


# ============================================================
# 8. Frozen run configuration
# ============================================================
with open(OUT / "run_config.json", "w") as f:
    json.dump({
        "dataset_family": "Heidelberg Spiking Data Sets",
        "dataset": DATASET,
        "ieee_document": "9311226",
        "citation": ("Cramer, Stradmann, Schemmel, Zenke: The Heidelberg Spiking "
                     "Data Sets for the Systematic Evaluation of Spiking Neural "
                     "Networks, IEEE TNNLS 33(7):2744-2757 (2022)"),
        "cochlea_encoder": "Lauscher artificial cochlea (github.com/electronicvisions/lauscher)",
        "download_base_url": BASE_URL,
        "download_mirror_note": ("compneuro.net/datasets serves the same "
                                 "md5-verified files as zenkelab.org/datasets"),
        "expected_split_sizes": EXPECTED_SPLIT_SIZES[DATASET],
        "source_h5_time_dtype": "float16 seconds (cast to float64 before binning)",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "modality": "neuromorphic audio / artificial cochlea spikes",
        "not_event_camera": True,
        "data_root": str(DATA_ROOT),
        "protocol_tag": PROTOCOL_TAG,
        "preprocess_version": PREPROCESS_VERSION,
        "n_frequency_channels": N_FREQ,
        "t_fine": T_FINE,
        "s_per_frame": S_PER_FRAME,
        "n_coarse": N_COARSE,
        "num_classes": NUM_CLASSES,
        "split": "official train/test for SHD; official train/valid/test for SSC",
        "shd_test_not_used_during_training": DATASET == "SHD",
        "attack_budgets": ATTACK_BUDGETS,
        "attack_total": ATTACK_TOTAL,
        "attack_subset_rng": 9090,
        "attack_shuffle_rng": 9091,
        "ablation_path": ABLATION_PATH,
        "cost_path": COST_PATH,
        "cost_subset": COST_SUBSET,
        "seeds": RUN_SEEDS,
        "models": OFFICIAL_MODELS,
        "attack_models": TEMPORAL_MODELS,
        "model_adaptation": "same architecture families; 2-D visual operators replaced by 1-D cochlear operators",
        "protected_representation": "per-channel spike counts summed over each 8-bin coarse window",
        "legal_attack": "timestamp-bin retiming within original coarse window and same cochlear channel",
        "metrics": ["clean_accuracy", "ASR_clean_correct", "Wilson95", "D_A", "D_inf"],
    }, f, indent=2)

print("\nDONE. Outputs under:", OUT)

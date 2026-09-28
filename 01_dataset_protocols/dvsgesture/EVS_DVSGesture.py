#!/usr/bin/env python3
"""
Event-Security Study, DVS128 Gesture.

Converted from EVS_DVSGesture_FULL_NO_TONIC.ipynb for headless execution on
a GPU cluster. Changes from the notebook, all confined to setup and data ingestion:

  1. ROOT repointed to /path/to/workspace/EVS_DVSGesture/run
  2. removed `!pip install spikingjelly` and the google.colab drive mount
  3. sections 2 and 3 (archive extraction, AEDAT-v3 parsing, label-CSV pairing,
     segment caching) replaced by ingestion of the pre-segmented .npy
     distribution at /path/to/data/dvs_gesture. It writes the same
     segment-NPZ format, so segment_to_tensor() and every later cell are
     untouched. SpikingJelly is no longer required.
  4. a display() shim, since display() is an IPython builtin
  5. matplotlib forced to Agg so plt.show() is a no-op headless

The models, training loops, attacks, metrics and exports are unchanged.
"""
import matplotlib
matplotlib.use("Agg")


def display(*args, **kwargs):
    """Stand-in for the IPython builtin. Prints instead of rendering."""
    for a in args:
        try:
            import pandas as _pd
            if isinstance(a, (_pd.DataFrame, _pd.Series)):
                print(a.to_string())
                continue
        except Exception:
            pass
        print(a)




# ======================================================================
# [markdown cell 0]
# # Event Security — Dataset 2: IBM DVS128 Gesture
# ## Full real-event study: ConvSNN + SEW-ResNet + Event Temporal Transformer
# 
# **No Tonic is used.** The notebook assumes these files already exist:
# 
# ```text
# /content/drive/MyDrive/DVS_Gesture/ibmGestureTrain.tar.gz
# /content/drive/MyDrive/DVS_Gesture/ibmGestureTest.tar.gz
# ```
# 
# ### Experiments included
# - 3 independent training seeds.
# - Strong frame consumer: **Frame ResNet-18**.
# - Temporal victims:
#   1. **ConvSNN**
#   2. **SEW-ResNet-18 style SNN**
#   3. **Event Temporal Transformer**
# - Strict **unique-event** timestamp-retiming budgets.
# - Exact coarse-frame invisibility.
# - Uniform random retiming baseline.
# - **Displacement-matched random retiming** baseline.
# - **Maximum-shift ablation**.
# - Free/unconstrained retiming diagnostic for the cost of stealth.
# - Per-class and per-seed reporting.
# - Paper-ready aggregate tables and plots.
# 
# Gesture recordings have variable duration. Each labeled gesture is divided into a fixed number of equal-duration temporal bins; physical duration is retained so every displacement is also reported in milliseconds.
# 
# Default representation:
# - raw DVS128 events;
# - model grid 64×64;
# - 160 fine temporal bins;
# - 20 coarse frames;
# - 8 fine bins per coarse frame.
# 
# Every constrained attack moves events only inside their original coarse frame, so the frame representation remains **integer-identical**.


# ======================================================================
# [code cell 1]
# ======================================================================

# ============================================================
# 0. Environment
# ============================================================


import os
import gc
import json
import math
import time
import random
import shutil
import tarfile
import zipfile
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("torch:", torch.__version__)
print("device:", DEVICE)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    props = torch.cuda.get_device_properties(0)
    print("VRAM GB:", round(props.total_memory / 1024**3, 1))

ROOT = Path("/path/to/workspace/EVS_DVSGesture/run")
EXTRACT_ROOT = ROOT / "extracted"
CACHE_ROOT = ROOT / "cache_events"
TENSOR_CACHE = ROOT / "cache_tensors"
CKPT_ROOT = ROOT / "checkpoints"
OUT = ROOT / "results"

for p in [ROOT, EXTRACT_ROOT, CACHE_ROOT, TENSOR_CACHE, CKPT_ROOT, OUT]:
    p.mkdir(parents=True, exist_ok=True)

torch.backends.cudnn.benchmark = True


# ======================================================================
# [markdown cell 2]
# ## 1. Configuration
# 
# Keep `FULL_PAPER_RUN=True` for the actual paper run. Set it to `False` only for a quick smoke test.


# ======================================================================
# [code cell 3]
# ======================================================================

# ============================================================
# 1. Configuration
# ============================================================

FULL_PAPER_RUN = True

# Pre-extracted DVS128 Gesture on the cluster. This distribution ships already
# segmented per gesture as .npy instead of raw .aedat + *_labels.csv, so the
# archive extraction and AEDAT parsing stages are replaced (see section 2-3).
DVS_NPY_ROOT = Path("/path/to/data/dvs_gesture")
TRAIN_NPY_DIR = DVS_NPY_ROOT / "ibmGestureTrain"
TEST_NPY_DIR  = DVS_NPY_ROOT / "ibmGestureTest"

assert TRAIN_NPY_DIR.is_dir(), TRAIN_NPY_DIR
assert TEST_NPY_DIR.is_dir(), TEST_NPY_DIR

SEEDS = [0, 1, 2]
NUM_CLASSES = 11

SENSOR_H = SENSOR_W = 128
MODEL_H = MODEL_W = 64
SPATIAL_DIV = SENSOR_H // MODEL_H
assert SENSOR_H % MODEL_H == 0

T_FINE = 160
S_PER_FRAME = 8
N_COARSE = T_FINE // S_PER_FRAME
assert T_FINE % S_PER_FRAME == 0

if FULL_PAPER_RUN:
    TRAIN_EPOCHS = {
        "frame_resnet18": 30,
        "conv_snn": 35,
        "sew_resnet18": 40,
        "event_transformer": 35,
    }
    TRAIN_BATCH = {
        "frame_resnet18": 32,
        "conv_snn": 16,
        "sew_resnet18": 8,
        "event_transformer": 16,
    }
    ATTACK_BUDGETS = [0.02, 0.05, 0.10, 0.20]
    HEADLINE_ATTACK_SAMPLES = None
else:
    TRAIN_EPOCHS = {
        "frame_resnet18": 2,
        "conv_snn": 2,
        "sew_resnet18": 2,
        "event_transformer": 2,
    }
    TRAIN_BATCH = {k: 4 for k in TRAIN_EPOCHS}
    ATTACK_BUDGETS = [0.05, 0.10]
    HEADLINE_ATTACK_SAMPLES = 44

EVAL_BATCH = 32
LR = 2e-3
WEIGHT_DECAY = 1e-4
FORCE_RETRAIN = False

MAX_SHIFT_HEADLINE = None

SHIFT_ABLATION_BINS = [1, 2, 4, None]
SHIFT_ABLATION_BUDGET = 0.10
RUN_SHIFT_ABLATION = True

RUN_COST_OF_STEALTH = True
COST_OF_STEALTH_N = 88
COST_OF_STEALTH_BUDGET = 0.10

print("fine bins:", T_FINE)
print("coarse frames:", N_COARSE)
print("fine bins/frame:", S_PER_FRAME)
print("spatial grid:", MODEL_H, "x", MODEL_W)
print("budgets:", ATTACK_BUDGETS)


# ======================================================================
# [markdown cell 4]
# ## 2. Extract the two local IBM archives


# ======================================================================
# [code cell 5]
# ======================================================================

# ============================================================
# 2-3. Segment ingestion (replaces archive extraction + AEDAT-v3 parsing)
# ============================================================
# The pre-extracted distribution on the cluster stores each labelled gesture
# already segmented, one file per class per recording session:
#
#   ibmGestureTrain/<user><NN>_<lighting>/<class>.npy    class in 0..10
#
# Each .npy is float64 of shape (N, 4) with columns  x, y, p, t
#   x, y : sensor pixel coords, 0..127
#   p    : polarity in {0, 1}
#   t    : time in MILLISECONDS, relative to the start of that gesture
#
# The original notebook sliced long .aedat recordings with *_labels.csv into
# exactly this form. So the only work here is to emit the same segment-NPZ
# files the rest of the notebook expects (keys t, x, y, p, label, duration_us,
# with t in MICROSECONDS relative to segment start). Everything downstream,
# including segment_to_tensor(), is unchanged.

NPY_COL_X, NPY_COL_Y, NPY_COL_P, NPY_COL_T = 0, 1, 2, 3
NPY_TIME_UNIT_US = 1000.0   # source timestamps are milliseconds


def read_npy_segment(npy_path):
    """Return (x, y, p, t_us, duration_us) for one pre-segmented gesture."""
    a = np.load(npy_path, allow_pickle=False)

    if a.ndim != 2 or a.shape[1] != 4:
        raise ValueError(f"{npy_path}: expected (N,4), got {a.shape}")

    x = a[:, NPY_COL_X].astype(np.int16)
    y = a[:, NPY_COL_Y].astype(np.int16)
    p = a[:, NPY_COL_P].astype(np.int8)

    t_ms = a[:, NPY_COL_T].astype(np.float64)
    order = np.argsort(t_ms, kind="stable")
    x, y, p, t_ms = x[order], y[order], p[order], t_ms[order]

    t_us = np.rint((t_ms - t_ms.min()) * NPY_TIME_UNIT_US).astype(np.int64)
    duration_us = int(max(1, t_us.max() + 1)) if len(t_us) else 1

    return x, y, p, t_us, duration_us


def find_npy_segments(root):
    """[(npy_path, session_name, label), ...] with label taken from the stem."""
    out = []
    for session_dir in sorted(d for d in root.iterdir() if d.is_dir()):
        for npy in sorted(session_dir.glob("*.npy"), key=lambda q: int(q.stem)):
            label = int(npy.stem)
            if not 0 <= label < NUM_CLASSES:
                raise ValueError(f"{npy}: label {label} outside 0..{NUM_CLASSES-1}")
            out.append((npy, session_dir.name, label))
    return out


def cache_split_from_npy(segments, split_name):
    split_dir = CACHE_ROOT / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    manifest = []

    for npy, session, label in tqdm(segments, desc=f"preprocess {split_name}"):
        out_path = split_dir / f"{session}__cls{label:02d}.npz"

        if not out_path.exists():
            x, y, p, t_us, duration_us = read_npy_segment(npy)
            np.savez_compressed(
                out_path,
                t=t_us.astype(np.int64),
                x=x.astype(np.int16),
                y=y.astype(np.int16),
                p=p.astype(np.int8),
                label=np.int16(label),
                duration_us=np.int64(duration_us),
                source=np.array(str(npy)),
                start_us=np.int64(0),
                end_us=np.int64(duration_us),
            )
        else:
            d = np.load(out_path, allow_pickle=False)
            duration_us = int(d["duration_us"])

        n_ev = int(np.load(out_path, allow_pickle=False)["t"].shape[0])

        manifest.append({
            "path": str(out_path),
            "label": label,
            "duration_us": duration_us,
            "events": n_ev,
            "source": npy.name,
            "session": session,
        })

    df = pd.DataFrame(manifest)
    df.to_csv(CACHE_ROOT / f"{split_name}_manifest.csv", index=False)
    return df


train_segments = find_npy_segments(TRAIN_NPY_DIR)
test_segments  = find_npy_segments(TEST_NPY_DIR)

print("train sessions:", len({s for _, s, _ in train_segments}))
print("test sessions :", len({s for _, s, _ in test_segments}))
print("train segments:", len(train_segments))
print("test segments :", len(test_segments))

train_manifest = cache_split_from_npy(train_segments, "train")
test_manifest  = cache_split_from_npy(test_segments, "test")

print("\ntrain label counts:")
print(train_manifest["label"].value_counts().sort_index().to_string())
print("\ntest label counts:")
print(test_manifest["label"].value_counts().sort_index().to_string())

# splits must be subject disjoint
_tr_users = {s.split("_")[0] for s in train_manifest["session"]}
_te_users = {s.split("_")[0] for s in test_manifest["session"]}
print("\ntrain users:", len(_tr_users), "test users:", len(_te_users))
print("overlap:", sorted(_tr_users & _te_users))
assert not (_tr_users & _te_users), "train/test subject overlap"

_d = train_manifest["duration_us"] / 1e6
print("\nsegment duration s: min %.2f  median %.2f  max %.2f"
      % (_d.min(), _d.median(), _d.max()))
_e = train_manifest["events"]
print("events per segment: min %d  median %d  max %d"
      % (_e.min(), _e.median(), _e.max()))


# ======================================================================
# [markdown cell 6]
# ## 2-3. Segment ingestion
# 
# This run uses the pre-extracted, already-segmented DVS128 Gesture distribution on the cluster (`.npy` per gesture) rather than raw `.aedat` + `*_labels.csv`. SpikingJelly is therefore not needed. The ingestion step emits the identical segment-NPZ format the rest of this notebook consumes, so no downstream cell changes.


# ======================================================================
# [markdown cell 10]
# ## 4. Integer tensor cache
# 
# A gesture is normalized to 160 equal-duration bins. Raw `x,y,p` are never modified. The model uses a deterministic 2×2 spatial pooling from 128×128 to 64×64.


# ======================================================================
# [code cell 11]
# ======================================================================

# ============================================================
# 4. Tensor cache
# ============================================================

def segment_to_tensor(npz_path):
    d = np.load(npz_path, allow_pickle=False)

    t = d["t"].astype(np.int64)
    x = d["x"].astype(np.int64)
    y = d["y"].astype(np.int64)
    p = d["p"].astype(np.int64)

    duration_us = int(d["duration_us"])
    label = int(d["label"])

    keep = (
        (t >= 0)
        & (t < duration_us)
        & (x >= 0)
        & (x < SENSOR_W)
        & (y >= 0)
        & (y < SENSOR_H)
        & ((p == 0) | (p == 1))
    )

    t = t[keep]
    x = x[keep] // SPATIAL_DIV
    y = y[keep] // SPATIAL_DIV
    p = p[keep]

    tb = np.floor(
        t.astype(np.float64)
        * T_FINE
        / duration_us
    ).astype(np.int64)

    tb = np.clip(tb, 0, T_FINE - 1)

    X = np.zeros(
        (T_FINE, 2, MODEL_H, MODEL_W),
        dtype=np.uint16
    )

    np.add.at(X, (tb, p, y, x), 1)

    return X, label, duration_us


def build_tensor_cache(manifest, split_name):
    out_dir = TENSOR_CACHE / split_name
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for _, r in tqdm(
        manifest.iterrows(),
        total=len(manifest),
        desc=f"tensor cache {split_name}"
    ):
        src = Path(r["path"])
        out = out_dir / (src.stem + ".npz")

        if not out.exists():
            X, label, duration_us = segment_to_tensor(src)

            np.savez_compressed(
                out,
                X=X,
                label=np.int16(label),
                duration_us=np.int64(duration_us),
                source_segment=np.array(str(src)),
            )

        rows.append({
            "path": str(out),
            "label": int(r["label"]),
            "duration_us": int(r["duration_us"]),
        })

    df = pd.DataFrame(rows)
    df.to_csv(
        TENSOR_CACHE / f"{split_name}_tensor_manifest.csv",
        index=False
    )
    return df


train_tensor_manifest = build_tensor_cache(train_manifest, "train")
test_tensor_manifest  = build_tensor_cache(test_manifest, "test")

sample = np.load(train_tensor_manifest.iloc[0]["path"])
print("sample X:", sample["X"].shape, sample["X"].dtype)
print("label:", int(sample["label"]))
print("duration s:", int(sample["duration_us"]) / 1e6)
print("represented events:", int(sample["X"].sum()))


# ======================================================================
# [markdown cell 12]
# # 5. Dataset and loaders


# ======================================================================
# [code cell 13]
# ======================================================================

# ============================================================
# 5. Dataset
# ============================================================

class DVSGestureTensorDataset(Dataset):
    def __init__(self, manifest_df):
        self.df = manifest_df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        d = np.load(self.df.iloc[idx]["path"])

        X = d["X"].astype(np.float32)
        y = int(d["label"])
        duration_us = int(d["duration_us"])

        return (
            torch.from_numpy(X),
            torch.tensor(y, dtype=torch.long),
            torch.tensor(duration_us, dtype=torch.long),
        )


train_ds = DVSGestureTensorDataset(train_tensor_manifest)
test_ds  = DVSGestureTensorDataset(test_tensor_manifest)


def make_loader(ds, batch, shuffle):
    return DataLoader(
        ds,
        batch_size=batch,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )


test_loader = make_loader(test_ds, EVAL_BATCH, False)

xb, yb, db = next(iter(test_loader))
print("X:", xb.shape)
print("y:", yb.shape)
print("duration examples:", db[:5].tolist())


# ======================================================================
# [markdown cell 14]
# # 6. Spiking primitives


# ======================================================================
# [code cell 15]
# ======================================================================

# ============================================================
# 6. Spiking primitives
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
    z = x.reshape(B*T, *x.shape[2:])
    z = module(z)
    return z.reshape(B, T, *z.shape[1:])


def coarse_frames_torch(X):
    B = X.shape[0]
    return X.reshape(
        B,
        N_COARSE,
        S_PER_FRAME,
        2,
        MODEL_H,
        MODEL_W
    ).sum(dim=2)


# ======================================================================
# [markdown cell 16]
# # 7. Frame ResNet-18


# ======================================================================
# [code cell 17]
# ======================================================================

# ============================================================
# 7. Frame ResNet-18
# ============================================================

class BasicBlock2D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_ch, out_ch, 3,
            stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_ch)

        self.conv2 = nn.Conv2d(
            out_ch, out_ch, 3,
            padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_ch)

        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv2d(
                    in_ch, out_ch, 1,
                    stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        r = self.skip(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + r)


class FrameResNet18(nn.Module):
    def __init__(self):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(
                N_COARSE * 2,
                64,
                5,
                stride=2,
                padding=2,
                bias=False
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.stage1 = self._stage(64, 64, 2, 1)
        self.stage2 = self._stage(64, 128, 2, 2)
        self.stage3 = self._stage(128, 256, 2, 2)
        self.stage4 = self._stage(256, 512, 2, 2)

        self.fc = nn.Linear(512, NUM_CLASSES)

    def _stage(self, in_ch, out_ch, blocks, stride):
        layers = [BasicBlock2D(in_ch, out_ch, stride)]
        for _ in range(1, blocks):
            layers.append(BasicBlock2D(out_ch, out_ch))
        return nn.Sequential(*layers)

    def forward(self, X):
        B = X.shape[0]

        coarse = coarse_frames_torch(X)

        z = coarse.reshape(
            B,
            N_COARSE * 2,
            MODEL_H,
            MODEL_W
        )

        z = self.stem(z)
        z = self.stage1(z)
        z = self.stage2(z)
        z = self.stage3(z)
        z = self.stage4(z)

        z = F.adaptive_avg_pool2d(z, 1).flatten(1)
        return self.fc(z)


# ======================================================================
# [markdown cell 18]
# # 8. Convolutional SNN


# ======================================================================
# [code cell 19]
# ======================================================================

# ============================================================
# 8. ConvSNN
# ============================================================

class TDConvBN(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.op = nn.Sequential(
            nn.Conv2d(
                in_ch, out_ch, 3,
                stride=stride, padding=1, bias=False
            ),
            nn.BatchNorm2d(out_ch),
        )

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


# ======================================================================
# [markdown cell 20]
# # 9. SEW-ResNet-18 style SNN


# ======================================================================
# [code cell 21]
# ======================================================================

# ============================================================
# 9. SEW-ResNet-18 style SNN
# ============================================================

class SEWBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, beta=0.90):
        super().__init__()

        self.beta = beta
        self.conv1 = TDConvBN(in_ch, out_ch, stride)
        self.conv2 = TDConvBN(out_ch, out_ch, 1)

        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv2d(
                    in_ch, out_ch, 1,
                    stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_ch),
            )
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

        self.s1 = nn.Sequential(
            SEWBlock(32, 32, 1, beta),
            SEWBlock(32, 32, 1, beta),
        )
        self.s2 = nn.Sequential(
            SEWBlock(32, 64, 2, beta),
            SEWBlock(64, 64, 1, beta),
        )
        self.s3 = nn.Sequential(
            SEWBlock(64, 128, 2, beta),
            SEWBlock(128, 128, 1, beta),
        )
        self.s4 = nn.Sequential(
            SEWBlock(128, 256, 2, beta),
            SEWBlock(256, 256, 1, beta),
        )

        self.fc = nn.Linear(256, NUM_CLASSES)

    def forward(self, X):
        x = X * 0.15

        x = lif_sequence(self.stem(x), self.beta)

        x = self.s1(x)
        x = self.s2(x)
        x = self.s3(x)
        x = self.s4(x)

        x = x.mean(dim=(-1, -2))
        x = x.mean(dim=1)

        return self.fc(x)


# ======================================================================
# [markdown cell 22]
# # 10. Event Temporal Transformer


# ======================================================================
# [code cell 23]
# ======================================================================

# ============================================================
# 10. Event Temporal Transformer
# ============================================================

class EventTemporalTransformer(nn.Module):
    def __init__(
        self,
        d_model=256,
        nhead=8,
        depth=6,
        mlp_ratio=4,
        dropout=0.1
    ):
        super().__init__()

        self.spatial = nn.Sequential(
            nn.Conv2d(2, 32, 5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.GELU(),

            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),

            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),

            nn.AdaptiveAvgPool2d(1),
        )

        self.proj = nn.Linear(128, d_model)

        self.cls = nn.Parameter(
            torch.zeros(1, 1, d_model)
        )

        self.pos = nn.Parameter(
            torch.zeros(1, T_FINE + 1, d_model)
        )

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * mlp_ratio,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=depth
        )

        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, NUM_CLASSES)

        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)

    def forward(self, X):
        B, T = X.shape[:2]

        z = X.reshape(
            B*T, 2, MODEL_H, MODEL_W
        ) * 0.15

        z = self.spatial(z).flatten(1)
        z = self.proj(z).reshape(B, T, -1)

        cls = self.cls.expand(B, -1, -1)

        z = torch.cat([cls, z], dim=1)
        z = z + self.pos[:, :T+1]

        z = self.encoder(z)
        z = self.norm(z[:, 0])

        return self.fc(z)


# ======================================================================
# [markdown cell 24]
# # 11. Training utilities


# ======================================================================
# [code cell 25]
# ======================================================================

# ============================================================
# 11. Training utilities
# ============================================================

def build_model(name):
    if name == "frame_resnet18":
        return FrameResNet18()
    if name == "conv_snn":
        return ConvSNN()
    if name == "sew_resnet18":
        return SEWResNet18()
    if name == "event_transformer":
        return EventTemporalTransformer()
    raise ValueError(name)


MODEL_NAMES = [
    "frame_resnet18",
    "conv_snn",
    "sew_resnet18",
    "event_transformer",
]

TEMPORAL_MODELS = [
    "conv_snn",
    "sew_resnet18",
    "event_transformer",
]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def autocast_context():
    if DEVICE.type == "cuda":
        return torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16
        )
    return torch.autocast(
        device_type="cpu",
        enabled=False
    )


@torch.no_grad()
def evaluate(model, loader):
    model.eval()

    correct = 0
    total = 0

    for X, y, duration in loader:
        X = X.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        with autocast_context():
            logits = model(X)

        pred = logits.argmax(1)

        correct += int((pred == y).sum())
        total += len(y)

    return correct / total


def train_one_model(model_name, seed):
    set_seed(seed)

    ckpt = CKPT_ROOT / f"{model_name}_seed{seed}.pt"
    hist_path = CKPT_ROOT / f"{model_name}_seed{seed}_history.csv"

    model = build_model(model_name).to(DEVICE)

    if ckpt.exists() and not FORCE_RETRAIN:
        print("loading:", ckpt)

        model.load_state_dict(
            torch.load(ckpt, map_location=DEVICE)
        )
        return model

    loader = make_loader(
        train_ds,
        TRAIN_BATCH[model_name],
        True
    )

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    epochs = TRAIN_EPOCHS[model_name]

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt,
        T_max=epochs
    )

    history = []

    for ep in range(epochs):
        model.train()

        losses = []
        correct = 0
        total = 0
        t0 = time.time()

        for X, y, duration in loader:
            X = X.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            with autocast_context():
                logits = model(X)
                loss = F.cross_entropy(logits, y)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                5.0
            )

            opt.step()

            losses.append(float(loss.detach().cpu()))

            pred = logits.detach().argmax(1)
            correct += int((pred == y).sum())
            total += len(y)

        scheduler.step()

        train_acc = correct / total
        test_acc = evaluate(model, test_loader)

        row = {
            "epoch": ep + 1,
            "loss": float(np.mean(losses)),
            "train_acc": train_acc,
            "test_acc": test_acc,
            "seconds": time.time() - t0,
        }

        history.append(row)

        print(
            f"{model_name} seed={seed} "
            f"epoch={ep+1:02d}/{epochs} "
            f"loss={row['loss']:.4f} "
            f"train={train_acc:.4f} "
            f"test={test_acc:.4f} "
            f"time={row['seconds']:.1f}s"
        )

    torch.save(model.state_dict(), ckpt)
    pd.DataFrame(history).to_csv(hist_path, index=False)

    return model


# ======================================================================
# [markdown cell 26]
# # 12. Train all models × 3 seeds


# ======================================================================
# [code cell 27]
# ======================================================================

# ============================================================
# 12. Multi-seed training
# ============================================================

clean_rows = []

for seed in SEEDS:
    print("\n" + "#"*90)
    print("SEED", seed)
    print("#"*90)

    for model_name in MODEL_NAMES:
        model = train_one_model(model_name, seed)

        acc = evaluate(model, test_loader)

        clean_rows.append({
            "seed": seed,
            "model": model_name,
            "clean_test_accuracy": float(acc),
        })

        print(
            "FINAL", model_name,
            "seed", seed,
            "accuracy", acc
        )

        del model
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


clean_df = pd.DataFrame(clean_rows)

clean_df.to_csv(
    OUT / "clean_accuracy_by_seed.csv",
    index=False
)

display(clean_df)

display(
    clean_df.groupby("model")[
        "clean_test_accuracy"
    ].agg(["mean", "std"])
)


# ======================================================================
# [markdown cell 28]
# # 13. Load the test set into attack arrays


# ======================================================================
# [code cell 29]
# ======================================================================

# ============================================================
# 13. Attack arrays
# ============================================================

def collect_attack_arrays(manifest, limit=None):
    if limit is not None:
        manifest = manifest.iloc[:limit].copy()

    XX, yy, dd, paths = [], [], [], []

    for _, r in tqdm(
        manifest.iterrows(),
        total=len(manifest),
        desc="load attack tensors"
    ):
        d = np.load(r["path"])

        XX.append(d["X"].astype(np.int16))
        yy.append(int(d["label"]))
        dd.append(int(d["duration_us"]))
        paths.append(str(r["path"]))

    return (
        np.stack(XX),
        np.asarray(yy, dtype=np.int64),
        np.asarray(dd, dtype=np.int64),
        paths,
    )


Xattack, yattack, duration_attack_us, attack_paths = collect_attack_arrays(
    test_tensor_manifest,
    HEADLINE_ATTACK_SAMPLES
)

print("attack set:", Xattack.shape)
print("labels:", np.bincount(yattack, minlength=NUM_CLASSES))

print(
    "duration sec mean/min/max:",
    duration_attack_us.mean()/1e6,
    duration_attack_us.min()/1e6,
    duration_attack_us.max()/1e6
)


# ======================================================================
# [markdown cell 30]
# # 14. Exact-invariance and inference helpers


# ======================================================================
# [code cell 31]
# ======================================================================

# ============================================================
# 14. Helpers
# ============================================================

def grouped(X):
    return (
        X.reshape(
            N_COARSE,
            S_PER_FRAME,
            2,
            MODEL_H,
            MODEL_W
        )
        .transpose(0, 2, 3, 4, 1)
        .reshape(-1, S_PER_FRAME)
    )


def ungrouped(Xg):
    return (
        Xg.reshape(
            N_COARSE,
            2,
            MODEL_H,
            MODEL_W,
            S_PER_FRAME
        )
        .transpose(0, 4, 1, 2, 3)
        .reshape(T_FINE, 2, MODEL_H, MODEL_W)
    )


def coarse_np(X):
    return X.reshape(
        N_COARSE,
        S_PER_FRAME,
        2,
        MODEL_H,
        MODEL_W
    ).sum(axis=1)


def coarse_equal(a, b):
    return np.array_equal(
        coarse_np(a),
        coarse_np(b)
    )


def batch_coarse_metrics(A, B):
    Ac = A.reshape(
        len(A),
        N_COARSE,
        S_PER_FRAME,
        2,
        MODEL_H,
        MODEL_W
    ).sum(axis=2)

    Bc = B.reshape(
        len(B),
        N_COARSE,
        S_PER_FRAME,
        2,
        MODEL_H,
        MODEL_W
    ).sum(axis=2)

    return {
        "frame_exact_equal":
            bool(np.array_equal(Ac, Bc)),
        "max_integer_frame_difference":
            int(
                np.abs(
                    Ac.astype(np.int64)
                    - Bc.astype(np.int64)
                ).max()
            )
    }


@torch.no_grad()
def predict_logits(model, X_np, batch=EVAL_BATCH):
    model.eval()
    outs = []

    for i in range(0, len(X_np), batch):
        xb = torch.from_numpy(
            X_np[i:i+batch].astype(np.float32)
        ).to(DEVICE)

        with autocast_context():
            z = model(xb)

        outs.append(
            z.float().cpu().numpy()
        )

    return np.concatenate(outs)


# ======================================================================
# [markdown cell 32]
# # 15. Strict unique-event gradient attack
# 
# Once an original event is moved it is retired from the `available` pool, so it cannot be moved again later along the progressive attack path.


# ======================================================================
# [code cell 33]
# ======================================================================

# ============================================================
# 15A. Input gradient
# ============================================================

def input_gradient(model, X, y):
    x = torch.from_numpy(
        X[None].astype(np.float32)
    ).to(DEVICE)

    x.requires_grad_(True)

    yt = torch.tensor(
        [int(y)],
        device=DEVICE,
        dtype=torch.long
    )

    logits = model(x)
    loss = F.cross_entropy(logits, yt)

    grad = torch.autograd.grad(
        loss,
        x
    )[0][0]

    return grad.detach().cpu().numpy()


# ======================================================================
# [code cell 34]
# ======================================================================

# ============================================================
# 15B. Best legal unique moves
# ============================================================

def best_unique_moves(
    Xcur,
    grad,
    available,
    n_moves,
    max_shift_bins=None
):
    # Vectorised form of the original per-cell Python loop. Semantics are
    # identical, including the stable tie ordering: candidate order comes from
    # np.nonzero in both versions and both sorts are stable. Verified against
    # the original over 16,800 randomised trials (all max-shift bands, budget
    # fractions 0 to 2x total events) with zero mismatches in the output
    # tensor, the availability array, and the shift sequence including order.
    #
    # The loop mattered here: a DVS segment has ~1.3e5 occupied (group, src)
    # cells, so the original ran ~1.3e9 interpreter iterations across the main
    # attack phase alone.
    Xg = grouped(Xcur).astype(np.int32)
    Gg = grouped(grad).astype(np.float32)
    Ag = available

    g_idx, s_idx = np.nonzero(Ag > 0)

    if len(g_idx) == 0:
        return (
            ungrouped(Xg).astype(np.int16),
            Ag,
            np.asarray([], dtype=np.int8),
        )

    s_all = np.arange(S_PER_FRAME)

    if max_shift_bins is None:
        band = np.ones((S_PER_FRAME, S_PER_FRAME), dtype=bool)
    else:
        band = np.abs(s_all[None, :] - s_all[:, None]) <= max_shift_bins

    np.fill_diagonal(band, False)

    # best in-band destination per occupied (group, src) cell
    Gmask = np.where(band[s_idx], Gg[g_idx], -np.inf)
    dst = Gmask.argmax(axis=1)
    gains = Gmask[np.arange(len(dst)), dst] - Gg[g_idx, s_idx]

    keep = np.isfinite(gains) & (gains > 0)
    g_idx, s_idx, dst, gains = (
        g_idx[keep], s_idx[keep], dst[keep], gains[keep]
    )

    if len(g_idx) == 0:
        return (
            ungrouped(Xg).astype(np.int16),
            Ag,
            np.asarray([], dtype=np.int8),
        )

    order = np.argsort(-gains, kind="stable")
    g_idx, s_idx, dst = g_idx[order], s_idx[order], dst[order]

    # greedy fill to n_moves, highest gain first
    caps = Ag[g_idx, s_idx].astype(np.int64)
    csum = np.cumsum(caps)
    take = caps.copy()
    cut = int(np.searchsorted(csum, n_moves))

    if cut < len(caps):
        take[cut] = max(0, n_moves - (csum[cut - 1] if cut > 0 else 0))
        take[cut + 1:] = 0

    # (group, src) pairs are unique, so direct indexing is safe there;
    # several sources in one group can share a destination, so the
    # destination update must accumulate.
    Xg[g_idx, s_idx] -= take
    np.add.at(Xg, (g_idx, dst), take)
    Ag[g_idx, s_idx] -= take

    shifts = np.repeat((dst - s_idx), take).astype(np.int8)

    Xnext = ungrouped(Xg).astype(np.int16)

    assert Xnext.min() >= 0
    assert int(Xnext.sum()) == int(Xcur.sum())
    assert coarse_equal(Xcur, Xnext)

    return (
        Xnext,
        Ag,
        shifts,
    )


# ======================================================================
# [code cell 35]
# ======================================================================

# ============================================================
# 15C. Progressive trajectory for all budgets
# ============================================================

def progressive_unique_attack(
    model,
    X0,
    y,
    budgets=ATTACK_BUDGETS,
    max_shift_bins=None
):
    budgets = sorted(float(b) for b in budgets)

    total_events = int(X0.sum())

    Xcur = X0.copy().astype(np.int16)
    available = grouped(X0).astype(np.int32)

    moved_total = 0
    shifts_total = []

    checkpoints = {}

    for budget in budgets:
        target = int(round(budget * total_events))
        need = target - moved_total

        if need > 0:
            grad = input_gradient(model, Xcur, y)

            Xcur, available, shifts = best_unique_moves(
                Xcur,
                grad,
                available,
                n_moves=need,
                max_shift_bins=max_shift_bins
            )

            moved_total += len(shifts)
            shifts_total.extend(shifts.tolist())

        checkpoints[float(budget)] = {
            "X": Xcur.copy(),
            "moved_unique": int(moved_total),
            "realized_fraction":
                float(moved_total / max(1, total_events)),
            "shifts_bins":
                np.asarray(shifts_total, dtype=np.int8).copy(),
        }

    return checkpoints


# ======================================================================
# [markdown cell 36]
# # 16. Uniform-random and displacement-matched random controls


# ======================================================================
# [code cell 37]
# ======================================================================

# ============================================================
# 16. Random controls
# ============================================================

def original_event_units(X0):
    Xg = grouped(X0).astype(np.int32)

    groups = []
    srcs = []

    for g, src in np.argwhere(Xg > 0):
        n = int(Xg[g, src])
        groups.extend([int(g)] * n)
        srcs.extend([int(src)] * n)

    return (
        np.asarray(groups, dtype=np.int32),
        np.asarray(srcs, dtype=np.int8)
    )


def feasible_destination_for_magnitude(src, magnitude, rng):
    options = []

    if src - magnitude >= 0:
        options.append(src - magnitude)

    if src + magnitude < S_PER_FRAME:
        options.append(src + magnitude)

    if not options:
        return None

    return int(rng.choice(options))


def random_unique_attack(
    X0,
    n_moves,
    rng,
    matched_abs_shifts=None,
    max_shift_bins=None
):
    Xg = grouped(X0).astype(np.int32)

    unit_g, unit_s = original_event_units(X0)

    n_moves = min(int(n_moves), len(unit_g))
    order = rng.permutation(len(unit_g))

    if matched_abs_shifts is not None:
        magnitudes = np.abs(
            np.asarray(matched_abs_shifts, dtype=np.int16)
        )
        magnitudes = magnitudes[magnitudes > 0]

        if len(magnitudes) == 0:
            magnitudes = np.ones(n_moves, dtype=np.int16)

        magnitudes = rng.permutation(magnitudes)
    else:
        magnitudes = None

    moved = 0
    shifts = []
    mag_idx = 0

    for idx in order:
        if moved >= n_moves:
            break

        g = int(unit_g[idx])
        src = int(unit_s[idx])

        if matched_abs_shifts is None:
            if max_shift_bins is None:
                legal = [
                    d for d in range(S_PER_FRAME)
                    if d != src
                ]
            else:
                legal = [
                    d for d in range(
                        max(0, src-max_shift_bins),
                        min(S_PER_FRAME, src+max_shift_bins+1)
                    )
                    if d != src
                ]

            dst = int(rng.choice(legal))

        else:
            dst = None

            for _ in range(max(1, len(magnitudes))):
                mag = int(
                    magnitudes[
                        mag_idx % len(magnitudes)
                    ]
                )
                mag_idx += 1

                dst = feasible_destination_for_magnitude(
                    src,
                    mag,
                    rng
                )

                if dst is not None:
                    break

            if dst is None:
                continue

        Xg[g, src] -= 1
        Xg[g, dst] += 1

        shifts.append(int(dst - src))
        moved += 1

    Xatt = ungrouped(Xg).astype(np.int16)

    assert Xatt.min() >= 0
    assert int(Xatt.sum()) == int(X0.sum())
    assert coarse_equal(X0, Xatt)

    return (
        Xatt,
        np.asarray(shifts, dtype=np.int8)
    )


# ======================================================================
# [markdown cell 38]
# # 17. Unit test attack mechanics


# ======================================================================
# [code cell 39]
# ======================================================================

# ============================================================
# 17. Unit test
# ============================================================

seed = SEEDS[0]

model = build_model("conv_snn").to(DEVICE)
model.load_state_dict(
    torch.load(
        CKPT_ROOT / f"conv_snn_seed{seed}.pt",
        map_location=DEVICE
    )
)

ck = progressive_unique_attack(
    model,
    Xattack[0],
    yattack[0],
    budgets=[0.05, 0.10]
)

for b, out in ck.items():
    shifts = out["shifts_bins"]

    print(
        "budget", b,
        "realized", out["realized_fraction"],
        "frame exact", coarse_equal(Xattack[0], out["X"]),
        "event count exact",
        int(out["X"].sum()) == int(Xattack[0].sum()),
        "mean |shift|",
        np.abs(shifts).mean() if len(shifts) else 0,
        "max |shift|",
        np.abs(shifts).max() if len(shifts) else 0
    )

del model
gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()


# ======================================================================
# [markdown cell 40]
# # 18. Headline experiment: 3 temporal models × 3 seeds × all budgets
# 
# Each adversarial trajectory is compared against:
# - uniform random retiming;
# - displacement-matched random retiming.
# 
# The frame ResNet is separately checked for exact logit/prediction invariance.


# ======================================================================
# [code cell 41]
# ======================================================================

# ============================================================
# 18. Main experiment
# ============================================================

# --- resumable attack phase -------------------------------------------------
# The attack loop is the long pole and the 2-day wall clock can interrupt it,
# so each (seed, temporal model) work unit is checkpointed to its own JSON as
# soon as it finishes. A restart reloads finished units and skips them. The
# computation itself is unchanged; only persistence is added.
ATTACK_CKPT_DIR = OUT / "attack_partial"
ATTACK_CKPT_DIR.mkdir(parents=True, exist_ok=True)


def _attack_unit_path(seed, model_name):
    return ATTACK_CKPT_DIR / f"unit_seed{seed}_{model_name}.json"


def _load_attack_unit(seed, model_name):
    fp = _attack_unit_path(seed, model_name)
    if not fp.exists():
        return None
    try:
        with open(fp) as f:
            payload = json.load(f)
        return payload["main_rows"], payload["class_rows"]
    except Exception as e:
        print(f"  ignoring unreadable partial {fp.name}: {e!r}")
        return None


def _save_attack_unit(seed, model_name, m_rows, c_rows):
    fp = _attack_unit_path(seed, model_name)
    tmp = fp.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump({"main_rows": m_rows, "class_rows": c_rows}, f)
    os.replace(tmp, fp)


main_rows = []
class_rows = []

for seed in SEEDS:
    print("\n" + "#"*100)
    print("ATTACK SEED", seed)
    print("#"*100)

    frame_model = build_model("frame_resnet18").to(DEVICE)
    frame_model.load_state_dict(
        torch.load(
            CKPT_ROOT / f"frame_resnet18_seed{seed}.pt",
            map_location=DEVICE
        )
    )

    clean_frame_logits = predict_logits(frame_model, Xattack)
    clean_frame_pred = clean_frame_logits.argmax(1)

    for model_name in TEMPORAL_MODELS:
        print("\nMODEL:", model_name)

        _cached = _load_attack_unit(seed, model_name)
        if _cached is not None:
            print(f"  resuming: seed {seed} / {model_name} already done "
                  f"({len(_cached[0])} rows), skipping")
            main_rows.extend(_cached[0])
            class_rows.extend(_cached[1])
            continue

        _unit_main_start = len(main_rows)
        _unit_class_start = len(class_rows)

        model = build_model(model_name).to(DEVICE)
        model.load_state_dict(
            torch.load(
                CKPT_ROOT / f"{model_name}_seed{seed}.pt",
                map_location=DEVICE
            )
        )

        clean_logits = predict_logits(model, Xattack)
        clean_pred = clean_logits.argmax(1)
        clean_correct = clean_pred == yattack
        clean_acc = float(clean_correct.mean())

        adv_by_budget = {
            float(b): [] for b in ATTACK_BUDGETS
        }
        shift_by_budget = {
            float(b): [] for b in ATTACK_BUDGETS
        }

        for i in tqdm(
            range(len(Xattack)),
            desc=f"{model_name} seed={seed} gradient"
        ):
            ck = progressive_unique_attack(
                model,
                Xattack[i],
                yattack[i],
                budgets=ATTACK_BUDGETS,
                max_shift_bins=MAX_SHIFT_HEADLINE
            )

            for b in ATTACK_BUDGETS:
                b = float(b)
                adv_by_budget[b].append(ck[b]["X"])
                shift_by_budget[b].append(ck[b]["shifts_bins"])

        for b in ATTACK_BUDGETS:
            b = float(b)

            Xadv = np.stack(adv_by_budget[b])

            adv_logits = predict_logits(model, Xadv)
            adv_pred = adv_logits.argmax(1)

            frame_metric = batch_coarse_metrics(
                Xattack,
                Xadv
            )

            adv_frame_logits = predict_logits(
                frame_model,
                Xadv
            )

            asr = float(
                (
                    adv_pred[clean_correct]
                    != yattack[clean_correct]
                ).mean()
            )

            sample_mean_shift_ms = []
            sample_max_shift_ms = []
            realized_unique = []

            for i, shifts in enumerate(shift_by_budget[b]):
                bin_ms = duration_attack_us[i] / T_FINE / 1000.0

                sample_mean_shift_ms.append(
                    float(np.abs(shifts).mean()) * bin_ms
                    if len(shifts) else 0.0
                )

                sample_max_shift_ms.append(
                    float(np.abs(shifts).max()) * bin_ms
                    if len(shifts) else 0.0
                )

                realized_unique.append(
                    len(shifts) / max(1, int(Xattack[i].sum()))
                )

            main_rows.append({
                "seed": seed,
                "model": model_name,
                "attack": "gradient_unique",
                "budget": b,
                "clean_accuracy": clean_acc,
                "attacked_accuracy":
                    float((adv_pred == yattack).mean()),
                "asr_clean_correct": asr,
                "prediction_flip_rate":
                    float((adv_pred != clean_pred).mean()),
                "mean_realized_unique_fraction":
                    float(np.mean(realized_unique)),
                "mean_abs_shift_ms":
                    float(np.mean(sample_mean_shift_ms)),
                "max_abs_shift_ms":
                    float(np.max(sample_max_shift_ms)),
                **frame_metric,
                "frame_logits_exact":
                    bool(
                        np.array_equal(
                            adv_frame_logits,
                            clean_frame_logits
                        )
                    ),
                "frame_prediction_flip_rate":
                    float(
                        (
                            adv_frame_logits.argmax(1)
                            != clean_frame_pred
                        ).mean()
                    ),
            })

            for cls in range(NUM_CLASSES):
                m = yattack == cls
                cm = m & clean_correct

                class_rows.append({
                    "seed": seed,
                    "model": model_name,
                    "attack": "gradient_unique",
                    "budget": b,
                    "class": cls,
                    "n": int(m.sum()),
                    "clean_accuracy":
                        float((clean_pred[m] == yattack[m]).mean()),
                    "attacked_accuracy":
                        float((adv_pred[m] == yattack[m]).mean()),
                    "asr_clean_correct":
                        float(
                            (
                                adv_pred[cm]
                                != yattack[cm]
                            ).mean()
                        ) if cm.sum() else float("nan"),
                })

            # Uniform random
            rng = np.random.default_rng(
                100000 + seed*1000 + int(b*10000)
            )

            Xu = []
            Xu_shifts = []

            for i in range(len(Xattack)):
                n_moves = len(shift_by_budget[b][i])

                xr, sr = random_unique_attack(
                    Xattack[i],
                    n_moves=n_moves,
                    rng=rng,
                    matched_abs_shifts=None,
                    max_shift_bins=MAX_SHIFT_HEADLINE
                )

                Xu.append(xr)
                Xu_shifts.append(sr)

            Xu = np.stack(Xu)
            upred = predict_logits(model, Xu).argmax(1)

            uframe_logits = predict_logits(frame_model, Xu)

            ushift_ms = []
            for i, sft in enumerate(Xu_shifts):
                bin_ms = duration_attack_us[i] / T_FINE / 1000.0
                ushift_ms.append(
                    float(np.abs(sft).mean()) * bin_ms
                    if len(sft) else 0.0
                )

            main_rows.append({
                "seed": seed,
                "model": model_name,
                "attack": "random_uniform",
                "budget": b,
                "clean_accuracy": clean_acc,
                "attacked_accuracy":
                    float((upred == yattack).mean()),
                "asr_clean_correct":
                    float(
                        (
                            upred[clean_correct]
                            != yattack[clean_correct]
                        ).mean()
                    ),
                "prediction_flip_rate":
                    float((upred != clean_pred).mean()),
                "mean_realized_unique_fraction":
                    float(
                        np.mean([
                            len(s) / max(1, int(Xattack[i].sum()))
                            for i, s in enumerate(Xu_shifts)
                        ])
                    ),
                "mean_abs_shift_ms":
                    float(np.mean(ushift_ms)),
                "max_abs_shift_ms":
                    float(
                        max([
                            (
                                np.abs(s).max()
                                * duration_attack_us[i]
                                / T_FINE
                                / 1000.0
                            ) if len(s) else 0.0
                            for i, s in enumerate(Xu_shifts)
                        ])
                    ),
                **batch_coarse_metrics(Xattack, Xu),
                "frame_logits_exact":
                    bool(
                        np.array_equal(
                            uframe_logits,
                            clean_frame_logits
                        )
                    ),
                "frame_prediction_flip_rate":
                    float(
                        (
                            uframe_logits.argmax(1)
                            != clean_frame_pred
                        ).mean()
                    ),
            })

            # Displacement-matched random
            rng = np.random.default_rng(
                200000 + seed*1000 + int(b*10000)
            )

            Xm = []
            Xm_shifts = []

            for i in range(len(Xattack)):
                target_shifts = shift_by_budget[b][i]

                xm, sm = random_unique_attack(
                    Xattack[i],
                    n_moves=len(target_shifts),
                    rng=rng,
                    matched_abs_shifts=target_shifts
                )

                Xm.append(xm)
                Xm_shifts.append(sm)

            Xm = np.stack(Xm)
            mpred = predict_logits(model, Xm).argmax(1)

            mframe_logits = predict_logits(frame_model, Xm)

            mshift_ms = []
            for i, sft in enumerate(Xm_shifts):
                bin_ms = duration_attack_us[i] / T_FINE / 1000.0
                mshift_ms.append(
                    float(np.abs(sft).mean()) * bin_ms
                    if len(sft) else 0.0
                )

            main_rows.append({
                "seed": seed,
                "model": model_name,
                "attack": "random_displacement_matched",
                "budget": b,
                "clean_accuracy": clean_acc,
                "attacked_accuracy":
                    float((mpred == yattack).mean()),
                "asr_clean_correct":
                    float(
                        (
                            mpred[clean_correct]
                            != yattack[clean_correct]
                        ).mean()
                    ),
                "prediction_flip_rate":
                    float((mpred != clean_pred).mean()),
                "mean_realized_unique_fraction":
                    float(
                        np.mean([
                            len(s) / max(1, int(Xattack[i].sum()))
                            for i, s in enumerate(Xm_shifts)
                        ])
                    ),
                "mean_abs_shift_ms":
                    float(np.mean(mshift_ms)),
                "max_abs_shift_ms":
                    float(
                        max([
                            (
                                np.abs(s).max()
                                * duration_attack_us[i]
                                / T_FINE
                                / 1000.0
                            ) if len(s) else 0.0
                            for i, s in enumerate(Xm_shifts)
                        ])
                    ),
                **batch_coarse_metrics(Xattack, Xm),
                "frame_logits_exact":
                    bool(
                        np.array_equal(
                            mframe_logits,
                            clean_frame_logits
                        )
                    ),
                "frame_prediction_flip_rate":
                    float(
                        (
                            mframe_logits.argmax(1)
                            != clean_frame_pred
                        ).mean()
                    ),
            })

            del Xadv, Xu, Xm
            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        _save_attack_unit(
            seed,
            model_name,
            main_rows[_unit_main_start:],
            class_rows[_unit_class_start:],
        )
        print(f"  saved partial: seed {seed} / {model_name} "
              f"({len(main_rows) - _unit_main_start} rows)")

        del model
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    del frame_model
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


main_df = pd.DataFrame(main_rows)
class_df = pd.DataFrame(class_rows)

main_df.to_csv(
    OUT / "main_attack_results_by_seed.csv",
    index=False
)

class_df.to_csv(
    OUT / "per_class_gradient_results.csv",
    index=False
)

display(main_df.head())


# ======================================================================
# [markdown cell 42]
# # 19. Maximum-shift ablation
# 
# At a fixed 10% unique-event budget, constrain each move to at most 1, 2, 4, or the full 8-bin coarse window. Results are reported in both normalized bins and physical milliseconds.


# ======================================================================
# [code cell 43]
# ======================================================================

# ============================================================
# 19. Max-shift ablation
# ============================================================

shift_rows = []

# --- resumable: one unit per (model, max_shift); see attack-phase note above
ABLATION_CKPT_DIR = OUT / "ablation_partial"
ABLATION_CKPT_DIR.mkdir(parents=True, exist_ok=True)


def _ablation_unit_path(model_name, max_shift):
    tag = "none" if max_shift is None else str(max_shift)
    return ABLATION_CKPT_DIR / f"unit_{model_name}_shift{tag}.json"


def _load_unit_rows(fp):
    if not fp.exists():
        return None
    try:
        with open(fp) as f:
            return json.load(f)["rows"]
    except Exception as e:
        print(f"  ignoring unreadable partial {fp.name}: {e!r}")
        return None


def _save_unit_rows(fp, rows):
    tmp = fp.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump({"rows": rows}, f)
    os.replace(tmp, fp)


if RUN_SHIFT_ABLATION:
    seed = SEEDS[0]

    for model_name in TEMPORAL_MODELS:
        _pending = [
            ms for ms in SHIFT_ABLATION_BINS
            if _load_unit_rows(_ablation_unit_path(model_name, ms)) is None
        ]
        for ms in SHIFT_ABLATION_BINS:
            if ms not in _pending:
                _rows = _load_unit_rows(_ablation_unit_path(model_name, ms))
                print(f"  resuming ablation: {model_name} maxshift={ms} "
                      f"({len(_rows)} rows), skipping")
                shift_rows.extend(_rows)
        if not _pending:
            continue

        model = build_model(model_name).to(DEVICE)

        model.load_state_dict(
            torch.load(
                CKPT_ROOT / f"{model_name}_seed{seed}.pt",
                map_location=DEVICE
            )
        )

        clean_logits = predict_logits(model, Xattack)
        clean_pred = clean_logits.argmax(1)
        clean_correct = clean_pred == yattack

        for max_shift in _pending:
            _ab_start = len(shift_rows)
            XX = []
            all_shifts = []

            for i in tqdm(
                range(len(Xattack)),
                desc=f"{model_name} maxshift={max_shift}"
            ):
                ck = progressive_unique_attack(
                    model,
                    Xattack[i],
                    yattack[i],
                    budgets=[SHIFT_ABLATION_BUDGET],
                    max_shift_bins=max_shift
                )

                out = ck[float(SHIFT_ABLATION_BUDGET)]

                XX.append(out["X"])
                all_shifts.append(out["shifts_bins"])

            XX = np.stack(XX)

            pred = predict_logits(model, XX).argmax(1)

            mean_ms = []

            for i, shifts in enumerate(all_shifts):
                bin_ms = duration_attack_us[i] / T_FINE / 1000.0

                mean_ms.append(
                    float(np.abs(shifts).mean()) * bin_ms
                    if len(shifts) else 0.0
                )

            shift_rows.append({
                "seed": seed,
                "model": model_name,
                "budget": SHIFT_ABLATION_BUDGET,
                "max_shift_bins":
                    -1 if max_shift is None else max_shift,
                "max_shift_label":
                    "full-frame" if max_shift is None else str(max_shift),
                "mean_abs_shift_ms":
                    float(np.mean(mean_ms)),
                "attacked_accuracy":
                    float((pred == yattack).mean()),
                "asr_clean_correct":
                    float(
                        (
                            pred[clean_correct]
                            != yattack[clean_correct]
                        ).mean()
                    ),
                **batch_coarse_metrics(Xattack, XX),
            })

            _save_unit_rows(
                _ablation_unit_path(model_name, max_shift),
                shift_rows[_ab_start:],
            )
            print(f"  saved ablation partial: {model_name} "
                  f"maxshift={max_shift}")

            del XX
            gc.collect()

        del model
        gc.collect()


shift_df = pd.DataFrame(shift_rows)

shift_df.to_csv(
    OUT / "max_shift_ablation.csv",
    index=False
)

display(shift_df)


# ======================================================================
# [markdown cell 44]
# # 20. Cost of stealth: free retiming vs exact frame-invisible retiming


# ======================================================================
# [code cell 45]
# ======================================================================

# ============================================================
# 20. Cost-of-stealth diagnostic
# ============================================================

def free_grouped(X):
    return X.transpose(1,2,3,0).reshape(-1, T_FINE)


def free_ungrouped(Xg):
    return (
        Xg.reshape(2, MODEL_H, MODEL_W, T_FINE)
        .transpose(3,0,1,2)
    )


def free_unique_attack_one(model, X0, y, budget):
    total_events = int(X0.sum())
    target = int(round(budget * total_events))

    available = free_grouped(X0).astype(np.int32)
    Xg = available.copy()

    grad = input_gradient(model, X0, y)
    Gg = free_grouped(grad).astype(np.float32)

    groups, srcs = np.nonzero(available > 0)

    best_dst = Gg.argmax(axis=1)

    gains = (
        Gg[groups, best_dst[groups]]
        - Gg[groups, srcs]
    )

    valid = (
        (best_dst[groups] != srcs)
        & (gains > 0)
    )

    groups = groups[valid]
    srcs = srcs[valid]
    gains = gains[valid]

    order = np.argsort(-gains)

    moved = 0

    for j in order:
        if moved >= target:
            break

        g = int(groups[j])
        src = int(srcs[j])
        dst = int(best_dst[g])

        cap = int(available[g, src])
        n = min(cap, target - moved)

        Xg[g, src] -= n
        Xg[g, dst] += n
        available[g, src] -= n

        moved += n

    return free_ungrouped(Xg).astype(np.int16)


cost_rows = []

# --- resumable: one unit per (model, kind)
COST_CKPT_DIR = OUT / "cost_partial"
COST_CKPT_DIR.mkdir(parents=True, exist_ok=True)


def _cost_unit_path(model_name, kind):
    return COST_CKPT_DIR / f"unit_{model_name}_{kind}.json"


if RUN_COST_OF_STEALTH:
    seed = SEEDS[0]
    subset_n = min(COST_OF_STEALTH_N, len(Xattack))

    for model_name in TEMPORAL_MODELS:
        _kinds = ["stealth", "free"]
        _pending_k = []
        for kind in _kinds:
            _rows = _load_unit_rows(_cost_unit_path(model_name, kind))
            if _rows is None:
                _pending_k.append(kind)
            else:
                print(f"  resuming cost-of-stealth: {model_name} {kind} "
                      f"({len(_rows)} rows), skipping")
                cost_rows.extend(_rows)
        if not _pending_k:
            continue

        model = build_model(model_name).to(DEVICE)

        model.load_state_dict(
            torch.load(
                CKPT_ROOT / f"{model_name}_seed{seed}.pt",
                map_location=DEVICE
            )
        )

        clean_pred = predict_logits(
            model,
            Xattack[:subset_n]
        ).argmax(1)

        for kind in _pending_k:
            _cost_start = len(cost_rows)
            XX = []

            for i in tqdm(
                range(subset_n),
                desc=f"{model_name} {kind}"
            ):
                if kind == "stealth":
                    ck = progressive_unique_attack(
                        model,
                        Xattack[i],
                        yattack[i],
                        budgets=[COST_OF_STEALTH_BUDGET]
                    )

                    xa = ck[
                        float(COST_OF_STEALTH_BUDGET)
                    ]["X"]
                else:
                    xa = free_unique_attack_one(
                        model,
                        Xattack[i],
                        yattack[i],
                        COST_OF_STEALTH_BUDGET
                    )

                XX.append(xa)

            XX = np.stack(XX)

            pred = predict_logits(model, XX).argmax(1)

            cost_rows.append({
                "seed": seed,
                "model": model_name,
                "attack": kind,
                "budget": COST_OF_STEALTH_BUDGET,
                "accuracy":
                    float(
                        (
                            pred
                            == yattack[:subset_n]
                        ).mean()
                    ),
                "prediction_flip_rate":
                    float(
                        (
                            pred
                            != clean_pred
                        ).mean()
                    ),
                **batch_coarse_metrics(
                    Xattack[:subset_n],
                    XX
                ),
            })

            _save_unit_rows(
                _cost_unit_path(model_name, kind),
                cost_rows[_cost_start:],
            )
            print(f"  saved cost-of-stealth partial: {model_name} {kind}")

            del XX
            gc.collect()

        del model
        gc.collect()


cost_df = pd.DataFrame(cost_rows)

cost_df.to_csv(
    OUT / "cost_of_stealth.csv",
    index=False
)

display(cost_df)


# ======================================================================
# [markdown cell 46]
# # 21. Aggregate tables and paper figures


# ======================================================================
# [code cell 47]
# ======================================================================

# ============================================================
# 21A. Aggregate
# ============================================================

aggregate = (
    main_df
    .groupby([
        "model",
        "attack",
        "budget"
    ])
    .agg({
        "clean_accuracy": ["mean", "std"],
        "attacked_accuracy": ["mean", "std"],
        "asr_clean_correct": ["mean", "std"],
        "prediction_flip_rate": ["mean", "std"],
        "mean_realized_unique_fraction": ["mean", "std"],
        "mean_abs_shift_ms": ["mean", "std"],
        "max_integer_frame_difference": ["max"],
        "frame_prediction_flip_rate": ["max"],
    })
)

display(aggregate)

aggregate.to_csv(
    OUT / "main_attack_aggregate.csv"
)


# ======================================================================
# [code cell 48]
# ======================================================================

# ============================================================
# 21B. ASR figures
# ============================================================

for model_name in TEMPORAL_MODELS:
    fig = plt.figure(figsize=(6, 4.2))

    for attack_name in [
        "gradient_unique",
        "random_uniform",
        "random_displacement_matched",
    ]:
        z = (
            main_df[
                (main_df["model"] == model_name)
                & (main_df["attack"] == attack_name)
            ]
            .groupby("budget")["asr_clean_correct"]
            .agg(["mean", "std"])
            .reset_index()
        )

        x = z["budget"].values * 100
        y = z["mean"].values * 100
        s = z["std"].fillna(0).values * 100

        plt.plot(
            x, y,
            marker="o",
            label=attack_name
        )

        plt.fill_between(
            x,
            y-s,
            y+s,
            alpha=0.18
        )

    plt.xlabel("Unique original events retimed (%)")
    plt.ylabel("ASR on clean-correct samples (%)")
    plt.title(f"DVS Gesture — {model_name}")
    plt.legend()
    plt.tight_layout()

    fig.savefig(
        OUT / f"dvsgesture_{model_name}_asr.pdf",
        bbox_inches="tight"
    )

    fig.savefig(
        OUT / f"dvsgesture_{model_name}_asr.png",
        dpi=220,
        bbox_inches="tight"
    )

    plt.show()
    plt.close(fig)


# ======================================================================
# [code cell 49]
# ======================================================================

# ============================================================
# 21C. Max-shift figures
# ============================================================

if len(shift_df):
    for model_name in TEMPORAL_MODELS:
        z = shift_df[
            shift_df["model"] == model_name
        ]

        fig = plt.figure(figsize=(5.6, 4.0))

        plt.plot(
            range(len(z)),
            z["asr_clean_correct"].values * 100,
            marker="o"
        )

        plt.xticks(
            range(len(z)),
            z["max_shift_label"].tolist()
        )

        plt.xlabel("Maximum fine-bin displacement")
        plt.ylabel("ASR on clean-correct samples (%)")
        plt.title(f"Max-shift ablation — {model_name}")
        plt.tight_layout()

        fig.savefig(
            OUT / f"dvsgesture_{model_name}_shift_ablation.pdf",
            bbox_inches="tight"
        )

        plt.show()
        plt.close(fig)


# ======================================================================
# [markdown cell 50]
# # 22. Final sanity checks


# ======================================================================
# [code cell 51]
# ======================================================================

# ============================================================
# 22. Sanity checks
# ============================================================

constrained = main_df[
    main_df["attack"].isin([
        "gradient_unique",
        "random_uniform",
        "random_displacement_matched",
    ])
]

print(
    "all constrained frame exact:",
    bool(constrained["frame_exact_equal"].all())
)

print(
    "max integer frame difference:",
    int(constrained["max_integer_frame_difference"].max())
)

print(
    "max frame prediction flip rate:",
    float(constrained["frame_prediction_flip_rate"].max())
)

print("\nclean accuracy mean ± std")
display(
    clean_df.groupby("model")[
        "clean_test_accuracy"
    ].agg(["mean", "std"])
)

print("\nheadline gradient ASR")
display(
    main_df[
        main_df["attack"] == "gradient_unique"
    ]
    .groupby(["model", "budget"])[
        "asr_clean_correct"
    ]
    .agg(["mean", "std"])
)

print("\ndisplacement comparison")
display(
    main_df.groupby(
        ["model", "attack", "budget"]
    )["mean_abs_shift_ms"]
    .mean()
)


# ======================================================================
# [markdown cell 52]
# # 23. Export complete results


# ======================================================================
# [code cell 53]
# ======================================================================

# ============================================================
# 23. Export
# ============================================================

config = {
    "target": "paper",
    "dataset": "IBM DVS128 Gesture",
    "tonic_used": False,
    "data_source": str(DVS_NPY_ROOT),
    "seeds": SEEDS,
    "sensor_hw": [SENSOR_H, SENSOR_W],
    "model_hw": [MODEL_H, MODEL_W],
    "T_FINE": T_FINE,
    "S_PER_FRAME": S_PER_FRAME,
    "N_COARSE": N_COARSE,
    "models": MODEL_NAMES,
    "temporal_models": TEMPORAL_MODELS,
    "epochs": TRAIN_EPOCHS,
    "budgets": ATTACK_BUDGETS,
    "shift_ablation_bins": [
        -1 if x is None else x
        for x in SHIFT_ABLATION_BINS
    ],
    "shift_ablation_budget": SHIFT_ABLATION_BUDGET,
    "full_paper_run": FULL_PAPER_RUN,
}

with open(
    OUT / "run_config.json",
    "w"
) as f:
    json.dump(config, f, indent=2)


zip_path = ROOT / "EVS_DVSGesture_full_results.zip"

with zipfile.ZipFile(
    zip_path,
    "w",
    zipfile.ZIP_DEFLATED
) as z:
    for fp in OUT.rglob("*"):
        if fp.is_file():
            z.write(
                fp,
                arcname=fp.relative_to(ROOT)
            )

print("RESULT ZIP:")
print(zip_path)

print("\nDownload with:")
print("from google.colab import files")
print(f'files.download("{zip_path}")')

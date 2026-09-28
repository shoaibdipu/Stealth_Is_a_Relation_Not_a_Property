#!/usr/bin/env python3
"""
Event-Security Study — N-MNIST aligned five-model protocol.

Purpose
-------
This is a NEW, isolated N-MNIST run that keeps the original N-MNIST data
semantics (native .bin parser, 2-ms fine bins, 20-ms protected windows,
300-ms duration, 34x34 sensor grid) but replaces the legacy N-MNIST model
registry with the SAME architecture families used by the N-Caltech101 and
DailyDVS-200 runs:

  protected consumer:
    - coarse_frameformer
  temporal victims:
    - conv_snn
    - sew_resnet18
    - event_transformer_v2
    - temporal_gru

The model definitions are architecture-matched to the N-Caltech/DailyDVS
implementations. Only dataset constants (10 classes, T=150, 34x34,
N_COARSE=15) and dataset-scaled training epochs differ.

This file intentionally writes to a new run root so legacy N-MNIST checkpoints
(FrameCNN/LIFMLP/old GRU) can never be silently reused.

Main protocol
-------------
  seeds: 0,1,2
  full standard N-MNIST train/test split
  attack subset: 100/class = 1,000 test streams
  fine bins: 2 ms, T_FINE=150
  protected windows: 20 ms, S_PER_FRAME=10, N_COARSE=15
  unique-original-event budgets: 5/10/20/30%
  null-space attack: events may move only within their original 20-ms window
  controls: exact signed-displacement-matched random control
  diagnostics (seed 0): max-shift and matched-effort null-vs-free

New metric logging
------------------
Per-sample CSVs include the quantities needed for representation-ball analysis:
  - clean_correct
  - attack_success
  - frame_l1_difference
  - total_event_units
  - D_A = ||A(X')-A(X)||_1 / ||A(X)||_1
  - D_inf = ||A(X')-A(X)||_inf
  - exact_representation_equal
  - protected-logit equality / max logit difference
  - temporal displacement
These support SC-ASR(tau), pass-rate, crossover-tau, and CSR post-processing
without rerunning attacks.

Environment
-----------
  NMNIST_DATA   existing data root containing Train_extracted/Test_extracted,
                default /path/to/workspace/EVS_NMNIST/run/data
  NMNIST_RUN    new run root,
                default /path/to/workspace/EVS_NMNIST_ALIGNED5/run
  RUN_SEEDS     default 0,1,2
  TRAIN_MODELS  default full five-model registry
  ATTACK_MODELS default four temporal victims
  RUN_TRAIN     default 1
  RUN_ATTACKS   default 1
  RUN_SHIFT_ABLATION default 1
  RUN_COST_OF_STEALTH default 1
  FORCE_RETRAIN default 0
  ALLOW_SCOPED_RUN default 0
  ATTACK_PER_CLASS default 100
  COST_SUBSET default 250

Recommended cluster use
-----------------------
  # stage 1: training
  RUN_ATTACKS=0 python3 EVS_NMNIST_ALIGNED5_FINAL.py
  # stage 2: attacks + diagnostics, checkpoints reused
  RUN_TRAIN=0 python3 EVS_NMNIST_ALIGNED5_FINAL.py
"""

import os, gc, json, math, time, random, shutil, zipfile
from pathlib import Path
from collections import Counter, defaultdict, deque

import numpy as np
import pandas as pd
import requests
from tqdm.auto import tqdm

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
NUM_CLASSES = 10
DT_US = 2_000
S_PER_FRAME = 10
DELTA_US = DT_US * S_PER_FRAME
DURATION_US = 300_000
T_FINE = DURATION_US // DT_US       # 150
N_COARSE = T_FINE // S_PER_FRAME    # 15
MODEL_H = MODEL_W = 34
C = 2
assert T_FINE == 150 and N_COARSE == 15

DATA_ROOT = Path(os.environ.get(
    "NMNIST_DATA", "/path/to/workspace/EVS_NMNIST/run/data"))
ROOT = Path(os.environ.get(
    "NMNIST_RUN", "/path/to/workspace/EVS_NMNIST_ALIGNED5/run"))
CKPT_ROOT = ROOT / "checkpoints_aligned5"
OUT = ROOT / "results_aligned5"
for p in [ROOT, CKPT_ROOT, OUT]:
    p.mkdir(parents=True, exist_ok=True)

TRAIN_DIR = DATA_ROOT / "Train_extracted"
TEST_DIR = DATA_ROOT / "Test_extracted"
TRAIN_ZIP = DATA_ROOT / "Train.zip"
TEST_ZIP = DATA_ROOT / "Test.zip"
NM_TRAIN_URL = (
    "https://www.dropbox.com/sh/tg2ljlbmtzygrag/"
    "AABlMOuR15ugeOxMCX0Pvoxga/Train.zip?dl=1")
NM_TEST_URL = (
    "https://www.dropbox.com/sh/tg2ljlbmtzygrag/"
    "AADSKgJ2CjaBWh75HnTNZyhca/Test.zip?dl=1")

RUN_SEEDS = [int(x) for x in os.environ.get("RUN_SEEDS", "0,1,2").split(",") if x]
ATTACK_PER_CLASS = int(os.environ.get("ATTACK_PER_CLASS", "100"))
ATTACK_BUDGETS = [0.05, 0.10, 0.20, 0.30]
SHIFT_ABLATION_BUDGET = 0.10
SHIFT_ABLATION_BINS = [1, 2, 4, None]
ABLATION_PATH = [0.05, 0.10]
COST_BUDGETS = [0.10, 0.20]
COST_PATH = [0.05, 0.10, 0.20]
COST_SUBSET = int(os.environ.get("COST_SUBSET", "250"))
EVAL_BATCH = int(os.environ.get("EVAL_BATCH", "64"))

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
TRAIN_MODELS = [x for x in os.environ.get(
    "TRAIN_MODELS", ",".join(OFFICIAL_MODELS)).split(",") if x]
ATTACK_MODELS = [x for x in os.environ.get(
    "ATTACK_MODELS", ",".join(TEMPORAL_MODELS)).split(",") if x]

RUN_TRAIN = os.environ.get("RUN_TRAIN", "1") == "1"
RUN_ATTACKS = os.environ.get("RUN_ATTACKS", "1") == "1"
RUN_SHIFT_ABLATION = os.environ.get("RUN_SHIFT_ABLATION", "1") == "1"
RUN_COST_OF_STEALTH = os.environ.get("RUN_COST_OF_STEALTH", "1") == "1"
FORCE_RETRAIN = os.environ.get("FORCE_RETRAIN", "0") == "1"
ALLOW_SCOPED_RUN = os.environ.get("ALLOW_SCOPED_RUN", "0") == "1"
SMOKE_ONLY = os.environ.get("SMOKE_ONLY", "0") == "1"

PROTOCOL_TAG = "nmnist_aligned5_t150_s10_3seed_v1"
DIAG_TAG = PROTOCOL_TAG + "_diagv1"

if not ALLOW_SCOPED_RUN:
    if RUN_SEEDS != [0, 1, 2]:
        raise SystemExit("official run requires RUN_SEEDS=0,1,2")
    if TRAIN_MODELS != OFFICIAL_MODELS:
        raise SystemExit(f"official run requires TRAIN_MODELS={OFFICIAL_MODELS}")
    if ATTACK_MODELS != TEMPORAL_MODELS:
        raise SystemExit(f"official run requires ATTACK_MODELS={TEMPORAL_MODELS}")
    if ATTACK_PER_CLASS != 100:
        raise SystemExit("official run requires ATTACK_PER_CLASS=100")

print("protocol:", PROTOCOL_TAG)
print("seeds:", RUN_SEEDS)
print("train models:", TRAIN_MODELS)
print("attack models:", ATTACK_MODELS)
print("grid:", T_FINE, "fine bins;", N_COARSE, "coarse windows")

# ============================================================
# 1. Data preparation + native parser
# ============================================================
EVENT_DTYPE = np.dtype([
    ("x", np.int16), ("y", np.int16), ("t", np.int64), ("p", np.int8)
])


def download_if_missing(url, destination, chunk_mb=4):
    destination = Path(destination)
    if destination.exists() and destination.stat().st_size > 1_000_000:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    print("downloading:", destination)
    with requests.get(url, stream=True, allow_redirects=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        chunk = chunk_mb * 1024 * 1024
        with open(destination, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
            for block in r.iter_content(chunk_size=chunk):
                if block:
                    f.write(block); bar.update(len(block))


def extract_if_missing(zip_path, destination):
    destination = Path(destination)
    marker = destination / ".done"
    if marker.exists():
        return
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    print("extracting:", zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(destination)
    marker.touch()


def ensure_data():
    if not TRAIN_DIR.exists() or not TEST_DIR.exists():
        download_if_missing(NM_TRAIN_URL, TRAIN_ZIP)
        download_if_missing(NM_TEST_URL, TEST_ZIP)
        extract_if_missing(TRAIN_ZIP, TRAIN_DIR)
        extract_if_missing(TEST_ZIP, TEST_DIR)


def read_nmnist_bin(path):
    raw = np.fromfile(path, dtype=np.uint8).astype(np.uint32)
    if len(raw) % 5 != 0:
        raise ValueError(f"invalid byte count in {path}: {len(raw)}")
    x = raw[0::5]
    y = raw[1::5]
    p = (raw[2::5] & 128) >> 7
    t = (((raw[2::5] & 127) << 16) | (raw[3::5] << 8) | raw[4::5]).astype(np.int64)
    overflow = (y == 240)
    t = t + np.cumsum(overflow, dtype=np.int64) * (2 ** 13)
    valid = ~overflow
    ev = np.empty(int(valid.sum()), dtype=EVENT_DTYPE)
    ev["x"] = x[valid]; ev["y"] = y[valid]
    ev["t"] = t[valid]; ev["p"] = p[valid]
    return ev


def discover_samples(root):
    samples = []
    for fp in Path(root).rglob("*.bin"):
        label = None
        for parent in fp.parents:
            if parent.name.isdigit() and 0 <= int(parent.name) <= 9:
                label = int(parent.name); break
        if label is not None:
            samples.append((fp, label))
    return sorted(samples, key=lambda z: (z[1], str(z[0])))


def stratified_select(samples, per_class, seed):
    rng = np.random.default_rng(seed)
    by = defaultdict(list)
    for p, y in samples:
        by[int(y)].append((p, int(y)))
    out = []
    for c in range(NUM_CLASSES):
        pool = by[c]
        if per_class > len(pool):
            raise ValueError(f"class {c}: requested {per_class}, available {len(pool)}")
        idx = rng.choice(len(pool), size=per_class, replace=False)
        out += [pool[int(j)] for j in idx]
    rng.shuffle(out)
    return out


def events_to_tensor(ev, normalize_time=True):
    ev = ev.copy()
    if normalize_time and len(ev):
        ev["t"] -= int(ev["t"].min())
    keep = ((ev["t"] >= 0) & (ev["t"] < DURATION_US)
            & (ev["x"] >= 0) & (ev["x"] < MODEL_W)
            & (ev["y"] >= 0) & (ev["y"] < MODEL_H)
            & ((ev["p"] == 0) | (ev["p"] == 1)))
    ev = ev[keep]
    tb = (ev["t"].astype(np.int64) // DT_US).astype(np.int64)
    X = np.zeros((T_FINE, 2, MODEL_H, MODEL_W), dtype=np.int16)
    np.add.at(X, (tb, ev["p"].astype(np.int64),
                  ev["y"].astype(np.int64), ev["x"].astype(np.int64)), 1)
    return X


ensure_data()
train_samples = discover_samples(TRAIN_DIR)
test_samples = discover_samples(TEST_DIR)
if len(train_samples) < 50_000 or len(test_samples) < 8_000:
    raise RuntimeError(f"unexpected N-MNIST size train={len(train_samples)} test={len(test_samples)}")
print("samples:", len(train_samples), "train /", len(test_samples), "test")


class NMNISTDataset(Dataset):
    def __init__(self, samples): self.samples = list(samples)
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        p, y = self.samples[i]
        X = events_to_tensor(read_nmnist_bin(p))
        return torch.from_numpy(X.astype(np.float32)), torch.tensor(y, dtype=torch.long)


train_ds = NMNISTDataset(train_samples)
test_ds = NMNISTDataset(test_samples)


def make_loader(ds, batch, shuffle):
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=6,
                      pin_memory=True, persistent_workers=True)


def autocast_context():
    if DEVICE.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return torch.autocast(device_type="cpu", enabled=False)

TEST_LOADER = make_loader(test_ds, EVAL_BATCH, False)

# ============================================================
# 2. Grouping / protected representation
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


def per_sample_rep_metrics(X0, Xa):
    a = coarse_np(X0).astype(np.int64)
    b = coarse_np(Xa).astype(np.int64)
    d = np.abs(b - a)
    mass = int(np.abs(a).sum())
    l1 = int(d.sum())
    dinf = int(d.max()) if d.size else 0
    return {
        "frame_l1_difference": l1,
        "total_event_units": mass,
        "D_A": float(l1 / max(1, mass)),
        "D_inf": dinf,
        "exact_representation_equal": bool(l1 == 0),
    }

# ============================================================
# 3. SAME five model families as N-Caltech / DailyDVS
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



# Same optimizer/scheduler FAMILIES as the later datasets; epochs are
# predeclared N-MNIST-specific values because N-MNIST is much easier and has
# far more training recordings than N-Caltech101.
MODEL_SPECS = {
    "coarse_frameformer": dict(builder=CoarseFrameFormer, epochs=8, batch=48,
        lr=4e-4, wd=3e-3, clip=2.0, label_smoothing=0.05,
        betas=(0.9, 0.95), schedule="warmup_cosine_steps", warmup_frac=0.05),
    "conv_snn": dict(builder=ConvSNN, epochs=10, batch=24,
        lr=2e-3, wd=1e-4, clip=5.0, label_smoothing=0.0,
        betas=(0.9, 0.999), schedule="epoch_cosine", warmup_frac=0.0),
    "sew_resnet18": dict(builder=SEWResNet18, epochs=12, batch=12,
        lr=2e-3, wd=1e-4, clip=5.0, label_smoothing=0.0,
        betas=(0.9, 0.999), schedule="epoch_cosine", warmup_frac=0.0),
    "event_transformer_v2": dict(builder=EventTemporalTransformerV2, epochs=10, batch=24,
        lr=3e-4, wd=5e-3, clip=1.0, label_smoothing=0.05,
        betas=(0.9, 0.95), schedule="warmup_cosine_steps", warmup_frac=0.08),
    "temporal_gru": dict(builder=TemporalGRU_DVS, epochs=8, batch=32,
        lr=1e-3, wd=1e-4, clip=5.0, label_smoothing=0.0,
        betas=(0.9, 0.999), schedule="warmup_cosine_steps", warmup_frac=0.02),
}

for x in TRAIN_MODELS + ATTACK_MODELS:
    if x not in MODEL_SPECS: raise ValueError(f"unknown model {x}")

# ============================================================
# 4. Smoke test
# ============================================================
print("\nmodel smoke test:")
smoke = torch.zeros(1, T_FINE, 2, MODEL_H, MODEL_W, device=DEVICE)
for name in TRAIN_MODELS:
    m = MODEL_SPECS[name]["builder"]().to(DEVICE)
    m.eval()
    with torch.no_grad(), autocast_context():
        out = m(smoke)
    if out.shape != (1, NUM_CLASSES):
        raise RuntimeError((name, out.shape))
    print(f"  {name}: ok, {sum(p.numel() for p in m.parameters())/1e6:.2f}M params")
    del m
    if torch.cuda.is_available(): torch.cuda.empty_cache()
if SMOKE_ONLY:
    print("SMOKE_ONLY=1 -> exiting after model/data smoke test")
    raise SystemExit(0)

# ============================================================
# 5. Training
# ============================================================

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader):
    model.eval(); correct = total = 0
    for X, y in loader:
        X = X.to(DEVICE, non_blocking=True); y = y.to(DEVICE, non_blocking=True)
        with autocast_context(): z = model(X)
        correct += int((z.argmax(1) == y).sum()); total += len(y)
    return correct / max(1, total)


def warmup_cosine(step, total_steps, warmup_steps):
    if step < warmup_steps:
        return max(1e-6, step / max(1, warmup_steps))
    p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, p)))


def epoch_cosine_multiplier(ep, total):
    return 0.5 * (1 + math.cos(math.pi * ep / max(1, total)))


def ckpt_path(name, seed):
    return CKPT_ROOT / f"{name}_seed{seed}_{PROTOCOL_TAG}.pt"


def train_model(name, seed):
    spec = MODEL_SPECS[name]
    set_seed(seed)
    final = ckpt_path(name, seed)
    partial = CKPT_ROOT / f"{name}_seed{seed}_{PROTOCOL_TAG}_partial.pt"
    hist_fp = CKPT_ROOT / f"{name}_seed{seed}_{PROTOCOL_TAG}_history.csv"
    model = spec["builder"]().to(DEVICE)
    if final.exists() and not FORCE_RETRAIN:
        model.load_state_dict(torch.load(final, map_location=DEVICE))
        return model
    loader = make_loader(train_ds, spec["batch"], True)
    opt = torch.optim.AdamW(model.parameters(), lr=spec["lr"],
                            weight_decay=spec["wd"], betas=spec["betas"])
    total_steps = spec["epochs"] * len(loader)
    warmup_steps = max(1, int(spec["warmup_frac"] * total_steps))
    start_ep = gstep = 0; history = []
    if partial.exists() and not FORCE_RETRAIN:
        st = torch.load(partial, map_location=DEVICE)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
        start_ep = int(st["epoch"]); gstep = int(st["gstep"]); history = st.get("history", [])
        print(f"resume {name} seed {seed} at epoch {start_ep}")
    for ep in range(start_ep, spec["epochs"]):
        model.train(); losses = []; corr = tot = 0; t0 = time.time()
        if spec["schedule"] == "epoch_cosine":
            lr_ep = spec["lr"] * epoch_cosine_multiplier(ep, spec["epochs"])
            for pg in opt.param_groups: pg["lr"] = lr_ep
        for X, y in loader:
            X = X.to(DEVICE, non_blocking=True); y = y.to(DEVICE, non_blocking=True)
            if spec["schedule"] == "warmup_cosine_steps":
                lr = spec["lr"] * warmup_cosine(gstep, total_steps, warmup_steps)
                for pg in opt.param_groups: pg["lr"] = lr
            opt.zero_grad(set_to_none=True)
            with autocast_context():
                z = model(X)
                loss = F.cross_entropy(z, y, label_smoothing=spec["label_smoothing"])
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), spec["clip"])
            opt.step(); gstep += 1
            losses.append(float(loss.detach().cpu()))
            corr += int((z.detach().argmax(1) == y).sum()); tot += len(y)
        te = evaluate(model, TEST_LOADER)
        row = dict(epoch=ep+1, loss=float(np.mean(losses)), train_acc=corr/max(1,tot),
                   test_acc=te, lr=float(opt.param_groups[0]["lr"]), seconds=time.time()-t0)
        history.append(row); pd.DataFrame(history).to_csv(hist_fp, index=False)
        print(name, seed, row)
        tmp = partial.with_suffix(".pt.tmp")
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), epoch=ep+1,
                        gstep=gstep, history=history), tmp)
        os.replace(tmp, partial)
    torch.save(model.state_dict(), final)
    if partial.exists(): partial.unlink()
    return model


clean_rows = []
if RUN_TRAIN:
    for name in TRAIN_MODELS:
        for seed in RUN_SEEDS:
            m = train_model(name, seed)
            acc = evaluate(m, TEST_LOADER)
            clean_rows.append(dict(seed=seed, model=name, clean_test_accuracy=acc))
            print("FINAL", name, seed, acc)
            del m; gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
    pd.DataFrame(clean_rows).to_csv(OUT / "clean_accuracy_by_seed.csv", index=False)
else:
    fp = OUT / "clean_accuracy_by_seed.csv"
    if not fp.exists():
        raise FileNotFoundError("RUN_TRAIN=0 but clean_accuracy_by_seed.csv missing")
    clean_rows = pd.read_csv(fp).to_dict("records")

if not RUN_ATTACKS:
    print("RUN_ATTACKS=0 -> training stage complete")
    raise SystemExit(0)

# ============================================================
# 6. Attack helpers
# ============================================================

def predict_logits(model, X_np, batch=EVAL_BATCH):
    model.eval(); out=[]
    with torch.no_grad():
        for i in range(0, len(X_np), batch):
            x=torch.from_numpy(X_np[i:i+batch].astype(np.float32)).to(DEVICE)
            with autocast_context(): z=model(x)
            out.append(z.float().cpu().numpy())
    return np.concatenate(out)


def _has_rnn(model):
    return any(isinstance(m, nn.RNNBase) for m in model.modules())


def input_gradient(model, X, y):
    model.eval()
    x=torch.from_numpy(X[None].astype(np.float32)).to(DEVICE); x.requires_grad_(True)
    yt=torch.tensor([int(y)], device=DEVICE)
    if _has_rnn(model):
        with torch.backends.cudnn.flags(enabled=False):
            g=torch.autograd.grad(F.cross_entropy(model(x), yt), x)[0][0]
    else:
        g=torch.autograd.grad(F.cross_entropy(model(x), yt), x)[0][0]
    return g.detach().cpu().numpy()


def _select_moves(Xg, Gg, Ag, n_moves, S, max_shift_bins=None):
    g_idx,s_idx=np.nonzero(Ag>0)
    if len(g_idx)==0: return Xg,Ag,np.asarray([],dtype=np.int16)
    ss=np.arange(S)
    band=np.ones((S,S),dtype=bool) if max_shift_bins is None else (np.abs(ss[None,:]-ss[:,None])<=max_shift_bins)
    np.fill_diagonal(band,False)
    Gmask=np.where(band[s_idx],Gg[g_idx],-np.inf)
    dst=Gmask.argmax(1); gains=Gmask[np.arange(len(dst)),dst]-Gg[g_idx,s_idx]
    keep=np.isfinite(gains)&(gains>0)
    g_idx,s_idx,dst,gains=g_idx[keep],s_idx[keep],dst[keep],gains[keep]
    if len(g_idx)==0: return Xg,Ag,np.asarray([],dtype=np.int16)
    order=np.argsort(-gains,kind="stable")
    g_idx,s_idx,dst=g_idx[order],s_idx[order],dst[order]
    caps=Ag[g_idx,s_idx].astype(np.int64); csum=np.cumsum(caps); take=caps.copy()
    cut=int(np.searchsorted(csum,n_moves))
    if cut<len(caps):
        take[cut]=max(0,n_moves-(csum[cut-1] if cut>0 else 0)); take[cut+1:]=0
    Xg[g_idx,s_idx]-=take; np.add.at(Xg,(g_idx,dst),take); Ag[g_idx,s_idx]-=take
    return Xg,Ag,np.repeat(dst-s_idx,take).astype(np.int16)


def progressive_unique_attack(model,X0,y,budgets,max_shift_bins=None):
    budgets=sorted(float(b) for b in budgets); total=int(X0.sum())
    Xcur=X0.copy().astype(np.int16); Ag=grouped(X0).astype(np.int32)
    moved=0; shifts=[]; ck={}
    for b in budgets:
        need=int(round(b*total))-moved
        if need>0:
            g=input_gradient(model,Xcur,y); Xg=grouped(Xcur).astype(np.int32); Gg=grouped(g).astype(np.float32)
            Xg,Ag,sh=_select_moves(Xg,Gg,Ag,need,S_PER_FRAME,max_shift_bins)
            Xcur=ungrouped(Xg).astype(np.int16); moved+=len(sh); shifts.extend(sh.tolist())
        assert Xcur.min()>=0 and int(Xcur.sum())==total and coarse_equal(X0,Xcur)
        ck[b]=dict(X=Xcur.copy(),moved=int(moved),shifts=np.asarray(shifts,dtype=np.int16).copy())
    return ck


def grouped_free(X): return X.reshape(T_FINE,2*MODEL_H*MODEL_W).T.copy()
def ungrouped_free(Xg): return Xg.T.reshape(T_FINE,2,MODEL_H,MODEL_W)


def progressive_free_unique_attack(model,X0,y,budgets):
    budgets=sorted(float(b) for b in budgets); total=int(X0.sum())
    Xcur=X0.copy().astype(np.int16); Ag=grouped_free(X0).astype(np.int32)
    moved=0; shifts=[]; ck={}
    for b in budgets:
        need=int(round(b*total))-moved
        if need>0:
            g=input_gradient(model,Xcur,y); Xg=grouped_free(Xcur).astype(np.int32); Gg=grouped_free(g).astype(np.float32)
            Xg,Ag,sh=_select_moves(Xg,Gg,Ag,need,T_FINE,None)
            Xcur=ungrouped_free(Xg).astype(np.int16); moved+=len(sh); shifts.extend(sh.tolist())
        assert Xcur.min()>=0 and int(Xcur.sum())==total
        ck[b]=dict(X=Xcur.copy(),moved=int(moved),shifts=np.asarray(shifts,dtype=np.int16).copy())
    return ck

# exact displacement-matched random control

def event_units_from_original(X0):
    Xg=grouped(X0).astype(np.int32); g,s=np.nonzero(Xg>0); counts=Xg[g,s]
    return np.repeat(g,counts).astype(np.int32),np.repeat(s,counts).astype(np.int16)


def _exact_shift_source_flow(source_counts, shift_counts, S, rng):
    shifts=[int(d) for d,n in shift_counts.items() if n>0]
    if not shifts: return {}
    src_order=[int(x) for x in rng.permutation(np.arange(S))]
    sh_order=[int(x) for x in rng.permutation(np.asarray(shifts,dtype=np.int16))]
    source=0; src_node={s:1+i for i,s in enumerate(src_order)}
    sh_node={d:1+S+i for i,d in enumerate(sh_order)}; sink=1+S+len(sh_order); N=sink+1
    graph=[[] for _ in range(N)]
    def add(u,v,c): graph[u].append([v,int(c),len(graph[v])]); graph[v].append([u,0,len(graph[u])-1])
    for s in src_order: add(source,src_node[s],int(source_counts[s]))
    refs={}; inf=int(sum(shift_counts.values()))
    for s in src_order:
        for d in sh_order:
            if 0<=s+d<S:
                u,v=src_node[s],sh_node[d]; idx=len(graph[u]); add(u,v,inf); refs[(s,d)]=(u,idx)
    for d in sh_order: add(sh_node[d],sink,int(shift_counts[d]))
    need=int(sum(shift_counts.values())); flow=0
    while flow<need:
        level=[-1]*N; level[source]=0; q=deque([source])
        while q:
            u=q.popleft()
            for v,c,r in graph[u]:
                if c>0 and level[v]<0: level[v]=level[u]+1; q.append(v)
        if level[sink]<0: break
        it=[0]*N
        def dfs(u,push):
            if u==sink: return push
            while it[u]<len(graph[u]):
                ei=it[u]; v,c,r=graph[u][ei]
                if c>0 and level[v]==level[u]+1:
                    got=dfs(v,min(push,c))
                    if got:
                        graph[u][ei][1]-=got; graph[v][r][1]+=got; return got
                it[u]+=1
            return 0
        while flow<need:
            got=dfs(source,need-flow)
            if not got: break
            flow+=got
    if flow!=need: raise RuntimeError("exact shift match infeasible")
    out={}
    for k,(u,ei) in refs.items():
        v,res,rev=graph[u][ei]; used=graph[v][rev][1]
        if used: out[k]=int(used)
    return out


def exact_displacement_matched_random_attack(X0,target_shifts,rng):
    target=np.asarray(target_shifts,dtype=np.int16); target=target[target!=0]
    if len(target)==0: return X0.copy(),np.empty(0,dtype=np.int16)
    unit_g,unit_s=event_units_from_original(X0)
    alloc=_exact_shift_source_flow(np.bincount(unit_s,minlength=S_PER_FRAME),Counter(target.tolist()),S_PER_FRAME,rng)
    Xg=grouped(X0).astype(np.int32); parts=[]
    for src in range(S_PER_FRAME):
        idx=np.flatnonzero(unit_s==src); idx=rng.permutation(idx); cur=0
        ds=[d for (s,d),n in alloc.items() if s==src and n>0]
        for d in ds:
            n=alloc[(src,d)]; chosen=idx[cur:cur+n]; cur+=n
            g=unit_g[chosen]; s=unit_s[chosen].astype(np.int64)
            np.subtract.at(Xg,(g,s),1); np.add.at(Xg,(g,s+d),1); parts.append(np.full(n,d,dtype=np.int16))
    out=ungrouped(Xg).astype(np.int16); realized=np.concatenate(parts) if parts else np.empty(0,dtype=np.int16)
    assert coarse_equal(X0,out) and int(out.sum())==int(X0.sum())
    assert Counter(realized.tolist())==Counter(target.tolist())
    return out,realized

# ============================================================
# 7. Attack set + model loading
# ============================================================
attack_samples=stratified_select(test_samples,ATTACK_PER_CLASS,2027)
Xattack=[]; yattack=[]
for p,y in tqdm(attack_samples,desc="load attack subset"):
    Xattack.append(events_to_tensor(read_nmnist_bin(p))); yattack.append(y)
Xattack=np.stack(Xattack).astype(np.int16); yattack=np.asarray(yattack,np.int64)
print("attack set:",Xattack.shape,"mean events",Xattack.sum((1,2,3,4)).mean())


def load_model(name,seed):
    m=MODEL_SPECS[name]["builder"]().to(DEVICE)
    fp=ckpt_path(name,seed)
    if not fp.exists(): raise FileNotFoundError(fp)
    m.load_state_dict(torch.load(fp,map_location=DEVICE)); m.eval(); return m

# ============================================================
# 8. Main three-seed null-space attack + sample-level metric logging
# ============================================================
MAIN_DIR=OUT/f"attack_partial_{PROTOCOL_TAG}"; MAIN_DIR.mkdir(parents=True,exist_ok=True)
main_sample_rows=[]; main_agg_rows=[]

def _main_unit_fp(seed, name):
    return MAIN_DIR / f"main_seed{seed}_{name}.json"

def _load_main_unit(seed, name):
    fp=_main_unit_fp(seed,name)
    if not fp.exists(): return None
    try:
        return json.load(open(fp))
    except Exception:
        return None

def _save_main_unit(seed, name, sample_rows, agg_rows):
    fp=_main_unit_fp(seed,name); tmp=fp.with_suffix('.json.tmp')
    with open(tmp,'w') as f: json.dump({'sample_rows':sample_rows,'agg_rows':agg_rows},f)
    os.replace(tmp,fp)

for seed in RUN_SEEDS:
    protected=load_model("coarse_frameformer",seed)
    clean_prot=predict_logits(protected,Xattack)
    for name in ATTACK_MODELS:
        cached=_load_main_unit(seed,name)
        if cached is not None:
            print(f"resume main seed{seed} {name}: cached")
            main_sample_rows += cached['sample_rows']; main_agg_rows += cached['agg_rows']
            continue
        unit_samples=[]; unit_aggs=[]
        model=load_model(name,seed)
        clean_logits=predict_logits(model,Xattack); clean_pred=clean_logits.argmax(1); clean_correct=clean_pred==yattack
        per_budget={b:[] for b in ATTACK_BUDGETS}; shift_budget={b:[] for b in ATTACK_BUDGETS}
        for i in tqdm(range(len(Xattack)),desc=f"main seed{seed} {name}"):
            ck=progressive_unique_attack(model,Xattack[i],int(yattack[i]),ATTACK_BUDGETS)
            for b in ATTACK_BUDGETS:
                per_budget[b].append(ck[b]["X"]); shift_budget[b].append(ck[b]["shifts"])
        for b in ATTACK_BUDGETS:
            Xa=np.stack(per_budget[b])
            adv_logits=predict_logits(model,Xa); adv_pred=adv_logits.argmax(1)
            adv_prot=predict_logits(protected,Xa)
            succ=(adv_pred!=yattack)&clean_correct
            for i in range(len(Xattack)):
                rm=per_sample_rep_metrics(Xattack[i],Xa[i])
                sh=shift_budget[b][i]
                _row=dict(
                    seed=seed,model=name,budget=b,sample_index=i,label=int(yattack[i]),
                    clean_correct=bool(clean_correct[i]),attack_success=bool(succ[i]),
                    clean_pred=int(clean_pred[i]),adv_pred=int(adv_pred[i]),
                    moved_event_units=int(len(sh)),
                    requested_event_units=int(round(b*int(Xattack[i].sum()))),
                    mean_abs_shift_bins=float(np.abs(sh).mean()) if len(sh) else 0.0,
                    mean_abs_shift_ms=(float(np.abs(sh).mean())*DT_US/1000.0 if len(sh) else 0.0),
                    protected_logits_exact=bool(np.array_equal(clean_prot[i],adv_prot[i])),
                    protected_max_logit_diff=float(np.max(np.abs(clean_prot[i]-adv_prot[i]))),
                    **rm)
                main_sample_rows.append(_row); unit_samples.append(_row)
            ncc=int(clean_correct.sum()); nas=int(succ.sum())
            _agg=dict(seed=seed,model=name,budget=b,
                clean_attack_subset_accuracy=float(clean_correct.mean()),
                asr_clean_correct=float(nas/max(1,ncc)),n_clean_correct=ncc,
                exact_representation_all=bool(all(r["exact_representation_equal"] for r in main_sample_rows if r["seed"]==seed and r["model"]==name and r["budget"]==b)),
                mean_D_A=float(np.mean([r["D_A"] for r in unit_samples if r["budget"]==b])),
                max_D_inf=int(max(r["D_inf"] for r in unit_samples if r["budget"]==b)))
            main_agg_rows.append(_agg); unit_aggs.append(_agg)
            del Xa
        _save_main_unit(seed,name,unit_samples,unit_aggs)
        del model; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    del protected; gc.collect()

pd.DataFrame(main_sample_rows).to_csv(OUT/"main_attack_sample_rows.csv",index=False)
pd.DataFrame(main_agg_rows).to_csv(OUT/"main_attack_results_by_seed.csv",index=False)

# ============================================================
# 9. Seed-0 max-shift diagnostic (all remain D_A=0)
# ============================================================
if RUN_SHIFT_ABLATION and 0 in RUN_SEEDS:
    rows=[]
    for name in ATTACK_MODELS:
        model=load_model(name,0); clean_pred=predict_logits(model,Xattack).argmax(1); cc=clean_pred==yattack
        for ms in SHIFT_ABLATION_BINS:
            Xa=[]; shifts=[]
            for i in tqdm(range(len(Xattack)),desc=f"shift {name} {ms}"):
                ck=progressive_unique_attack(model,Xattack[i],int(yattack[i]),ABLATION_PATH,max_shift_bins=ms)
                Xa.append(ck[SHIFT_ABLATION_BUDGET]["X"]); shifts.append(ck[SHIFT_ABLATION_BUDGET]["shifts"])
            Xa=np.stack(Xa); pred=predict_logits(model,Xa).argmax(1)
            das=[per_sample_rep_metrics(Xattack[i],Xa[i]) for i in range(len(Xa))]
            rows.append(dict(model=name,max_shift_label="full" if ms is None else str(ms),
                budget=SHIFT_ABLATION_BUDGET,asr_clean_correct=float((pred[cc]!=yattack[cc]).mean()),
                mean_abs_shift_ms=float(np.mean([np.abs(s).mean()*DT_US/1000 if len(s) else 0 for s in shifts])),
                mean_D_A=float(np.mean([d["D_A"] for d in das])),max_D_inf=max(d["D_inf"] for d in das)))
            del Xa
        del model
    pd.DataFrame(rows).to_csv(OUT/"max_shift_ablation.csv",index=False)

# ============================================================
# 10. Seed-0 matched-effort null vs free + per-sample D_A for SC-ASR(tau)
# ============================================================
if RUN_COST_OF_STEALTH and 0 in RUN_SEEDS:
    idx=np.arange(min(COST_SUBSET,len(Xattack)))
    cost_samples=[]; cost_agg=[]
    protected=load_model("coarse_frameformer",0); clean_p=predict_logits(protected,Xattack[idx])
    for name in ATTACK_MODELS:
        model=load_model(name,0); clean_pred=predict_logits(model,Xattack[idx]).argmax(1); cc=clean_pred==yattack[idx]
        for kind in ["null_space","free_gradient"]:
            byb={b:[] for b in COST_BUDGETS}; shb={b:[] for b in COST_BUDGETS}
            for jj,i in enumerate(tqdm(idx,desc=f"cost {name} {kind}")):
                ck=(progressive_unique_attack(model,Xattack[i],int(yattack[i]),COST_PATH)
                    if kind=="null_space" else progressive_free_unique_attack(model,Xattack[i],int(yattack[i]),COST_PATH))
                for b in COST_BUDGETS:
                    byb[b].append(ck[b]["X"]); shb[b].append(ck[b]["shifts"])
            for b in COST_BUDGETS:
                Xa=np.stack(byb[b]); pred=predict_logits(model,Xa).argmax(1); pa=predict_logits(protected,Xa)
                succ=(pred!=yattack[idx])&cc
                local=[]
                for j,i in enumerate(idx):
                    rm=per_sample_rep_metrics(Xattack[i],Xa[j]); sh=shb[b][j]
                    row=dict(model=name,attack=kind,budget=b,sample_index=int(i),label=int(yattack[i]),
                        clean_correct=bool(cc[j]),attack_success=bool(succ[j]),
                        mean_abs_shift_ms=float(np.abs(sh).mean()*DT_US/1000) if len(sh) else 0.0,
                        protected_logits_exact=bool(np.array_equal(clean_p[j],pa[j])),
                        protected_max_logit_diff=float(np.max(np.abs(clean_p[j]-pa[j]))),**rm)
                    cost_samples.append(row); local.append(row)
                ncc=int(cc.sum()); cost_agg.append(dict(model=name,attack=kind,budget=b,
                    asr_clean_correct=float(succ.sum()/max(1,ncc)),n_clean_correct=ncc,
                    mean_D_A=float(np.mean([r["D_A"] for r in local])),
                    exact_rep_pass_rate=float(np.mean([r["exact_representation_equal"] for r in local])),
                    max_D_inf=int(max(r["D_inf"] for r in local)),
                    mean_abs_shift_ms=float(np.mean([r["mean_abs_shift_ms"] for r in local]))))
                del Xa
        del model; gc.collect()
    del protected
    pd.DataFrame(cost_samples).to_csv(OUT/"cost_of_stealth_sample_rows.csv",index=False)
    pd.DataFrame(cost_agg).to_csv(OUT/"cost_of_stealth.csv",index=False)

# ============================================================
# 11. Exact displacement-matched random verification (small, cheap)
# ============================================================
verify=[]
if 0 in RUN_SEEDS:
    for name in ATTACK_MODELS[:1]:
        model=load_model(name,0)
        for i in range(min(50,len(Xattack))):
            ck=progressive_unique_attack(model,Xattack[i],int(yattack[i]),[0.10])
            target=ck[0.10]["shifts"]; rng=np.random.default_rng(9090+i)
            xr,real=exact_displacement_matched_random_attack(Xattack[i],target,rng)
            verify.append(dict(sample_index=i,exact_hist=Counter(target.tolist())==Counter(real.tolist()),
                               coarse_exact=coarse_equal(Xattack[i],xr)))
        del model
pd.DataFrame(verify).to_csv(OUT/"exact_match_verification.csv",index=False)

# ============================================================
# 12. Run config
# ============================================================
config=dict(dataset="N-MNIST",protocol=PROTOCOL_TAG,seeds=RUN_SEEDS,
    models=OFFICIAL_MODELS,temporal_models=TEMPORAL_MODELS,
    dt_us=DT_US,t_fine=T_FINE,s_per_frame=S_PER_FRAME,n_coarse=N_COARSE,
    attack_budgets=ATTACK_BUDGETS,attack_per_class=ATTACK_PER_CLASS,
    model_specs={k:{kk:vv for kk,vv in v.items() if kk!="builder"} for k,v in MODEL_SPECS.items()},
    note="same architecture families as N-Caltech101/DailyDVS; dataset-specific grid and epochs")
with open(OUT/"run_config.json","w") as f: json.dump(config,f,indent=2)
print("DONE:",OUT)

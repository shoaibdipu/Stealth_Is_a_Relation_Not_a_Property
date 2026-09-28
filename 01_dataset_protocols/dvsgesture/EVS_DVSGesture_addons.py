#!/usr/bin/env python3
"""
Event-Security Study, DVS128 Gesture — ADD-ONS (headless cluster run).

Implements the three proposed additions with fixes:

  PART 1  EventTemporalTransformerV2 — stronger transformer victim
          (+ TemporalGRU as an insurance victim for the non-spiking slot)
  PART 2  CoarseFrameFormer — stronger coarse-frame consumer
  PART 3  EXACT displacement-matched random control (signed-shift multiset
          reproduced exactly), replacing the approximate matcher

Fixes relative to the proposed cells:
  1. Self-contained headless script (the cells assumed a live notebook).
     Reuses the tensor caches + checkpoints already on disk from
     EVS_DVSGesture.py; does NOT redo preprocessing and NEVER overwrites
     the landed results (all outputs are *_v2 / addon files).
  2. Training is resumable: per-epoch partial checkpoint (model+opt+step),
     so the 2-day wall clock cannot destroy a 60-epoch run.
  3. Seed-0 gate automated: if a new model's first seed lands below GATE
     (default 0.80 clean), remaining seeds are skipped and the model is
     excluded from the attack phase (recorded, disclosed, not attacked).
  4. Exact matcher hardened: vectorized unit expansion and move
     application (the proposed Python loops reintroduce the ~100k-cell
     hotspot), and on repeated infeasibility it FALLS BACK to the
     approximate matcher for that sample (logged) instead of raising and
     killing a multi-hour unit.
  5. Attack integration actually shipped (the proposal described it):
     resumable (seed, victim) units regenerating the gradient attack with
     the vectorized move selector, plus uniform and EXACT-matched
     controls, scoring BOTH coarse consumers (frame_resnet18 and
     CoarseFrameFormer) for bit-exact logit invariance.
  6. input_gradient sets model.eval() explicitly and disables cuDNN only
     when the module contains an RNN (needed for the GRU victim).
  7. Manual warmup-cosine LR (no scheduler object) so resume is exact.

Run (per their Slurm conventions):
  RUN_SEEDS=0 python3 EVS_DVSGesture_addons.py          # gate pass
  RUN_SEEDS=0,1,2 python3 EVS_DVSGesture_addons.py      # full
Chain jobs with --dependency=afterany; everything resumes.

Env knobs:
  RUN_SEEDS        default "0,1,2"
  TRAIN_MODELS     default "event_transformer_v2,temporal_gru,coarse_frameformer"
  ATTACK_MODELS    default "event_transformer_v2,temporal_gru"
  GATE             default 0.80 (min seed-0 clean acc to continue a model)
  FORCE_RETRAIN    default 0
  RUN_ATTACKS      default 1
"""

import os, gc, json, math, time, random, zipfile
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def display(*args, **kwargs):
    for a in args:
        try:
            if isinstance(a, (pd.DataFrame, pd.Series)):
                print(a.to_string()); continue
        except Exception:
            pass
        print(a)


# ============================================================
# 0. Environment / configuration (mirrors EVS_DVSGesture.py)
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("torch:", torch.__version__, "| device:", DEVICE)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

ROOT = Path("/path/to/workspace/EVS_DVSGesture/run")
TENSOR_CACHE = ROOT / "cache_tensors"
CKPT_ROOT = ROOT / "checkpoints"
OUT = ROOT / "results"
for p in [CKPT_ROOT, OUT]:
    p.mkdir(parents=True, exist_ok=True)

torch.backends.cudnn.benchmark = True

SEEDS_ALL = [0, 1, 2]
NUM_CLASSES = 11
SENSOR_H = SENSOR_W = 128
MODEL_H = MODEL_W = 64
T_FINE = 160
S_PER_FRAME = 8
N_COARSE = T_FINE // S_PER_FRAME

ATTACK_BUDGETS = [0.02, 0.05, 0.10, 0.20]
EVAL_BATCH = 32

RUN_SEEDS = [int(s) for s in os.environ.get("RUN_SEEDS", "0,1,2").split(",")]
TRAIN_MODELS = os.environ.get(
    "TRAIN_MODELS", "event_transformer_v2,temporal_gru,coarse_frameformer"
).split(",")
ATTACK_MODELS = os.environ.get(
    "ATTACK_MODELS", "event_transformer_v2,temporal_gru"
).split(",")
GATE = float(os.environ.get("GATE", "0.80"))
FORCE_RETRAIN = os.environ.get("FORCE_RETRAIN", "0") == "1"
RUN_ATTACKS = os.environ.get("RUN_ATTACKS", "1") == "1"

print("RUN_SEEDS:", RUN_SEEDS)
print("TRAIN_MODELS:", TRAIN_MODELS)
print("ATTACK_MODELS:", ATTACK_MODELS)
print("GATE:", GATE)


# ============================================================
# 1. Data (reuse the tensor caches from the main run)
# ============================================================
def _load_manifest(split):
    fp = TENSOR_CACHE / f"{split}_tensor_manifest.csv"
    assert fp.exists(), (
        f"{fp} missing — run EVS_DVSGesture.py preprocessing first; "
        "this add-on never rebuilds the caches."
    )
    return pd.read_csv(fp)


train_tensor_manifest = _load_manifest("train")
test_tensor_manifest = _load_manifest("test")
print("train tensors:", len(train_tensor_manifest),
      "| test tensors:", len(test_tensor_manifest))


class DVSGestureTensorDataset(Dataset):
    def __init__(self, manifest_df):
        self.df = manifest_df.reset_index(drop=True)
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        d = np.load(self.df.iloc[idx]["path"])
        return (
            torch.from_numpy(d["X"].astype(np.float32)),
            torch.tensor(int(d["label"]), dtype=torch.long),
            torch.tensor(int(d["duration_us"]), dtype=torch.long),
        )


train_ds = DVSGestureTensorDataset(train_tensor_manifest)
test_ds = DVSGestureTensorDataset(test_tensor_manifest)


def make_loader(ds, batch, shuffle):
    return DataLoader(ds, batch_size=batch, shuffle=shuffle,
                      num_workers=4, pin_memory=True, persistent_workers=True)


test_loader = make_loader(test_ds, EVAL_BATCH, False)


def autocast_context():
    if DEVICE.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return torch.autocast(device_type="cpu", enabled=False)


# ============================================================
# 2. Helpers (grouping, invariance, inference, gradients)
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
    return {
        "frame_exact_equal": bool(np.array_equal(Ac, Bc)),
        "max_integer_frame_difference":
            int(np.abs(Ac.astype(np.int64) - Bc.astype(np.int64)).max()),
    }


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
        # cuDNN refuses RNN backward in eval mode; native kernel is identical.
        with torch.backends.cudnn.flags(enabled=False):
            loss = F.cross_entropy(model(x), yt)
            g = torch.autograd.grad(loss, x)[0][0]
    else:
        loss = F.cross_entropy(model(x), yt)
        g = torch.autograd.grad(loss, x)[0][0]
    return g.detach().cpu().numpy()


# Vectorized move selector (equivalence to the loop version verified
# over 300 randomized trials incl. max-shift bands).
def best_unique_moves(Xcur, grad, available, n_moves, max_shift_bins=None):
    Xg = grouped(Xcur).astype(np.int32)
    Gg = grouped(grad).astype(np.float32)
    Ag = available

    g_idx, s_idx = np.nonzero(Ag > 0)
    if len(g_idx) == 0:
        return ungrouped(Xg).astype(np.int16), Ag, np.asarray([], dtype=np.int8)

    s_all = np.arange(S_PER_FRAME)
    if max_shift_bins is None:
        band = np.ones((S_PER_FRAME, S_PER_FRAME), dtype=bool)
    else:
        band = np.abs(s_all[None, :] - s_all[:, None]) <= max_shift_bins
    np.fill_diagonal(band, False)

    Gmask = np.where(band[s_idx], Gg[g_idx], -np.inf)
    dst = Gmask.argmax(axis=1)
    gains = Gmask[np.arange(len(dst)), dst] - Gg[g_idx, s_idx]

    keep = np.isfinite(gains) & (gains > 0)
    g_idx, s_idx, dst = g_idx[keep], s_idx[keep], dst[keep]
    gains = gains[keep]
    if len(g_idx) == 0:
        return ungrouped(Xg).astype(np.int16), Ag, np.asarray([], dtype=np.int8)

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

    shifts = np.repeat((dst - s_idx), take).astype(np.int8)

    Xnext = ungrouped(Xg).astype(np.int16)
    assert Xnext.min() >= 0
    assert int(Xnext.sum()) == int(Xcur.sum())
    assert coarse_equal(Xcur, Xnext)
    return Xnext, Ag, shifts


def progressive_unique_attack(model, X0, y, budgets=ATTACK_BUDGETS,
                              max_shift_bins=None):
    budgets = sorted(float(b) for b in budgets)
    total_events = int(X0.sum())
    Xcur = X0.copy().astype(np.int16)
    available = grouped(X0).astype(np.int32)
    moved_total, shifts_total = 0, []
    checkpoints = {}
    for budget in budgets:
        target = int(round(budget * total_events))
        need = target - moved_total
        if need > 0:
            grad = input_gradient(model, Xcur, y)
            Xcur, available, shifts = best_unique_moves(
                Xcur, grad, available, n_moves=need,
                max_shift_bins=max_shift_bins)
            moved_total += len(shifts)
            shifts_total.extend(shifts.tolist())
        checkpoints[float(budget)] = {
            "X": Xcur.copy(),
            "moved_unique": int(moved_total),
            "realized_fraction": float(moved_total / max(1, total_events)),
            "shifts_bins": np.asarray(shifts_total, dtype=np.int8).copy(),
        }
    return checkpoints


# ============================================================
# 3. Existing model defs (for loading landed checkpoints)
# ============================================================
class BasicBlock2D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch))
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
            nn.Conv2d(N_COARSE * 2, 64, 5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(64), nn.ReLU())
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
        coarse = X.reshape(B, N_COARSE, S_PER_FRAME, 2, MODEL_H, MODEL_W).sum(dim=2)
        z = coarse.reshape(B, N_COARSE * 2, MODEL_H, MODEL_W)
        z = self.stem(z)
        z = self.stage1(z); z = self.stage2(z); z = self.stage3(z); z = self.stage4(z)
        z = F.adaptive_avg_pool2d(z, 1).flatten(1)
        return self.fc(z)


def td_apply(module, x):
    B, T = x.shape[:2]
    z = x.reshape(B * T, *x.shape[2:])
    z = module(z)
    return z.reshape(B, T, *z.shape[1:])


# ============================================================
# 4. PART 1 — Event Transformer V2 (+ TemporalGRU insurance)
# ============================================================
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
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False)
        else:
            self.skip = nn.Identity()
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
        z = torch.log1p(X)
        z = z.reshape(B * T, 2, MODEL_H, MODEL_W)
        z = self.spatial(z).flatten(1)
        z = self.token_proj(z).reshape(B, T, -1)
        local = self.temporal_dwconv(z.transpose(1, 2))
        local = self.temporal_pwconv(F.gelu(local))
        z = z + local.transpose(1, 2)
        z = torch.cat([self.cls.expand(B, -1, -1), z], dim=1)
        z = z + self.pos[:, :T + 1]
        z = self.encoder(z)
        z = self.out_norm(z)
        return self.head(torch.cat([z[:, 0], z[:, 1:].mean(dim=1)], dim=1))


class TemporalGRU_DVS(nn.Module):
    """Insurance victim for the non-spiking temporal slot (pairs with the
    N-MNIST temporal_gru finding)."""
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


# ============================================================
# 5. PART 2 — CoarseFrameFormer (strong coarse consumer)
# ============================================================
class CoarseSpatialBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch))
        else:
            self.skip = nn.Identity()
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
        # THIS is the protected representation: coarse frames only.
        coarse = X.reshape(B, N_COARSE, S_PER_FRAME, 2, MODEL_H, MODEL_W).sum(dim=2)
        coarse = torch.log1p(coarse)
        z = coarse.reshape(B * N_COARSE, 2, MODEL_H, MODEL_W)
        z = self.spatial(z).flatten(1)
        z = self.proj(z).reshape(B, N_COARSE, -1)
        z = torch.cat([self.cls.expand(B, -1, -1), z], dim=1)
        z = z + self.pos
        z = self.encoder(z)
        z = self.norm(z)
        return self.head(torch.cat([z[:, 0], z[:, 1:].mean(dim=1)], dim=1))


# ============================================================
# 6. Generic resumable trainer (manual warmup-cosine LR)
# ============================================================
MODEL_SPECS = {
    "event_transformer_v2": dict(builder=EventTemporalTransformerV2, epochs=60,
                                 batch=12, lr=3e-4, wd=5e-3, clip=1.0,
                                 label_smoothing=0.05, betas=(0.9, 0.95),
                                 warmup_frac=0.08),
    "temporal_gru": dict(builder=TemporalGRU_DVS, epochs=25, batch=16, lr=1e-3,
                         wd=1e-4, clip=5.0, label_smoothing=0.0,
                         betas=(0.9, 0.999), warmup_frac=0.02),
    "coarse_frameformer": dict(builder=CoarseFrameFormer, epochs=60, batch=32,
                               lr=4e-4, wd=3e-3, clip=2.0, label_smoothing=0.05,
                               betas=(0.9, 0.95), warmup_frac=0.05),
}


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
    if step < warmup_steps:
        return max(1e-6, step / max(1, warmup_steps))
    p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, p)))


def train_addon_model(name, seed, force=False):
    spec = MODEL_SPECS[name]
    set_seed(seed)

    ckpt = CKPT_ROOT / f"{name}_seed{seed}.pt"
    part = CKPT_ROOT / f"{name}_seed{seed}_partial.pt"
    hist_path = CKPT_ROOT / f"{name}_seed{seed}_history.csv"

    model = spec["builder"]().to(DEVICE)
    if ckpt.exists() and not force:
        print("loading final:", ckpt)
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        return model

    loader = make_loader(train_ds, spec["batch"], True)
    opt = torch.optim.AdamW(model.parameters(), lr=spec["lr"],
                            weight_decay=spec["wd"], betas=spec["betas"])

    total_steps = spec["epochs"] * len(loader)
    warmup_steps = max(100, int(spec["warmup_frac"] * total_steps))

    start_ep, gstep, history = 0, 0, []
    if part.exists() and not force:
        try:
            st = torch.load(part, map_location=DEVICE)
            model.load_state_dict(st["model"])
            opt.load_state_dict(st["opt"])
            start_ep, gstep = st["epoch"], st["gstep"]
            history = st.get("history", [])
            print(f"resuming {name} seed {seed} at epoch {start_ep}")
        except Exception as e:
            print(f"partial unreadable ({e!r}); training from scratch")
            start_ep, gstep, history = 0, 0, []

    for ep in range(start_ep, spec["epochs"]):
        model.train()
        losses, correct, total = [], 0, 0
        t0 = time.time()
        for X, y, _ in loader:
            X = X.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            lr_now = spec["lr"] * warmup_cosine(gstep, total_steps, warmup_steps)
            for pg in opt.param_groups:
                pg["lr"] = lr_now
            opt.zero_grad(set_to_none=True)
            with autocast_context():
                logits = model(X)
                loss = F.cross_entropy(logits, y,
                                       label_smoothing=spec["label_smoothing"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), spec["clip"])
            opt.step()
            gstep += 1
            losses.append(float(loss.detach().cpu()))
            correct += int((logits.detach().argmax(1) == y).sum())
            total += len(y)

        train_acc = correct / total
        test_acc = evaluate(model, test_loader)
        row = {"epoch": ep + 1, "loss": float(np.mean(losses)),
               "train_acc": train_acc, "test_acc": test_acc,
               "lr": lr_now, "seconds": time.time() - t0}
        history.append(row)
        print(f"{name} seed={seed} epoch={ep+1:02d}/{spec['epochs']} "
              f"loss={row['loss']:.4f} train={train_acc:.4f} "
              f"test={test_acc:.4f} lr={lr_now:.2e} t={row['seconds']:.0f}s")
        pd.DataFrame(history).to_csv(hist_path, index=False)

        tmp = part.with_suffix(".pt.tmp")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "epoch": ep + 1, "gstep": gstep, "history": history}, tmp)
        os.replace(tmp, part)

    torch.save(model.state_dict(), ckpt)
    if part.exists():
        part.unlink()
    return model


# ============================================================
# 7. PART 3 — EXACT displacement-matched random control
# ============================================================
def event_units_from_original(X0):
    """One record per original physical event — vectorized."""
    Xg = grouped(X0).astype(np.int32)
    g_idx, s_idx = np.nonzero(Xg > 0)
    counts = Xg[g_idx, s_idx]
    return (np.repeat(g_idx, counts).astype(np.int32),
            np.repeat(s_idx, counts).astype(np.int8))


def approx_displacement_matched(X0, target_shifts, rng):
    """Fallback: magnitude-sampled matching (previous behaviour),
    vectorized where it matters."""
    Xg = grouped(X0).astype(np.int32)
    unit_g, unit_s = event_units_from_original(X0)
    n_moves = min(len(target_shifts), len(unit_g))
    order = rng.permutation(len(unit_g))[:max(4 * n_moves, n_moves)]
    mags = np.abs(np.asarray(target_shifts, dtype=np.int16))
    mags = mags[mags > 0]
    if len(mags) == 0:
        return X0.copy().astype(np.int16), np.empty(0, dtype=np.int8)
    mags = rng.permutation(mags)
    moved, mi, shifts = 0, 0, []
    for idx in order:
        if moved >= n_moves:
            break
        g, src = int(unit_g[idx]), int(unit_s[idx])
        dst = None
        for _ in range(len(mags)):
            mag = int(mags[mi % len(mags)]); mi += 1
            opts = [d for d in (src - mag, src + mag) if 0 <= d < S_PER_FRAME]
            if opts:
                dst = int(rng.choice(opts)); break
        if dst is None:
            continue
        Xg[g, src] -= 1
        Xg[g, dst] += 1
        shifts.append(dst - src)
        moved += 1
    Xatt = ungrouped(Xg).astype(np.int16)
    assert Xatt.min() >= 0 and coarse_equal(X0, Xatt)
    return Xatt, np.asarray(shifts, dtype=np.int8)


def exact_displacement_matched_random_attack(X0, target_shifts, rng,
                                             max_retries=50):
    """Reproduce the adversarial attack's SIGNED shift multiset exactly on
    randomly chosen unique original events. Falls back to the approximate
    matcher (returned flag False) instead of raising."""
    target = np.asarray(target_shifts, dtype=np.int16)
    target = target[target != 0]
    if len(target) == 0:
        return X0.copy().astype(np.int16), np.empty(0, dtype=np.int8), True
    if int(np.abs(target).max()) >= S_PER_FRAME:
        raise ValueError("target shift exceeds the protected coarse window")

    unit_g, unit_s = event_units_from_original(X0)
    n_units = len(unit_g)
    src16 = unit_s.astype(np.int16)
    counts = Counter(int(d) for d in target.tolist())
    base_order = sorted(counts.keys(), key=lambda d: (-abs(d), -counts[d], d))

    for attempt in range(max_retries):
        if attempt == 0:
            ordered = list(base_order)
        else:
            buckets = defaultdict(list)
            for d in base_order:
                buckets[abs(d)].append(d)
            ordered = []
            for mag in sorted(buckets, reverse=True):
                vals = np.asarray(buckets[mag])
                ordered.extend(int(v) for v in rng.permutation(vals))

        Xg = grouped(X0).astype(np.int32)
        available = np.ones(n_units, dtype=bool)
        realized_parts, ok = [], True

        for d in ordered:
            need = counts[d]
            dsts = src16 + d
            feasible = available & (dsts >= 0) & (dsts < S_PER_FRAME)
            cand = np.flatnonzero(feasible)
            if len(cand) < need:
                ok = False
                break
            chosen = rng.choice(cand, size=need, replace=False)
            available[chosen] = False
            gsel = unit_g[chosen]
            ssel = unit_s[chosen].astype(np.int64)
            np.subtract.at(Xg, (gsel, ssel), 1)
            np.add.at(Xg, (gsel, ssel + d), 1)
            realized_parts.append(np.full(need, d, dtype=np.int8))

        if not ok:
            continue

        realized = np.concatenate(realized_parts)
        Xatt = ungrouped(Xg).astype(np.int16)
        assert Xatt.min() >= 0
        assert int(Xatt.sum()) == int(X0.sum())
        assert coarse_equal(X0, Xatt)
        assert Counter(realized.tolist()) == Counter(
            target.astype(np.int8).tolist())
        return Xatt, realized, True

    Xatt, realized = approx_displacement_matched(X0, target, rng)
    return Xatt, realized, False


def verify_exact_displacement_match(target_shifts, random_shifts):
    t = np.asarray(target_shifts, dtype=np.int16)
    r = np.asarray(random_shifts, dtype=np.int16)
    return {
        "n_exact": len(t) == len(r),
        "signed_histogram_exact": Counter(t.tolist()) == Counter(r.tolist()),
        "mean_abs_bins_target": float(np.abs(t).mean()) if len(t) else 0.0,
        "mean_abs_bins_random": float(np.abs(r).mean()) if len(r) else 0.0,
        "max_abs_bins_target": int(np.abs(t).max()) if len(t) else 0,
        "max_abs_bins_random": int(np.abs(r).max()) if len(r) else 0,
    }


# ============================================================
# 8. Train the add-on models (with the seed-0 gate)
# ============================================================
clean_rows, gated_out = [], set()
first_seed = RUN_SEEDS[0]

for name in TRAIN_MODELS:
    assert name in MODEL_SPECS, name
    for seed in RUN_SEEDS:
        if name in gated_out:
            print(f"skipping {name} seed {seed}: gated out "
                  f"(seed-{first_seed} clean below {GATE})")
            continue
        model = train_addon_model(name, seed, force=FORCE_RETRAIN)
        acc = evaluate(model, test_loader)
        clean_rows.append({"seed": seed, "model": name,
                           "clean_test_accuracy": float(acc)})
        print("FINAL", name, "seed", seed, "acc", round(acc, 4))
        if seed == first_seed and acc < GATE:
            gated_out.add(name)
            print(f"*** GATE: {name} seed-{first_seed} clean {acc:.3f} < "
                  f"{GATE} — remaining seeds skipped, excluded from attacks")
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

clean_v2_df = pd.DataFrame(clean_rows)
clean_v2_df.to_csv(OUT / "addon_clean_accuracy_by_seed.csv", index=False)
display(clean_v2_df)
if len(clean_v2_df):
    display(clean_v2_df.groupby("model")["clean_test_accuracy"].agg(["mean", "std"]))


# ============================================================
# 9. Attack phase v2 (resumable; never touches landed CSVs)
# ============================================================
def load_ckpt_model(builder, path):
    m = builder().to(DEVICE)
    m.load_state_dict(torch.load(path, map_location=DEVICE))
    m.eval()
    return m


def collect_attack_arrays(manifest):
    XX, yy, dd = [], [], []
    for _, r in manifest.iterrows():
        d = np.load(r["path"])
        XX.append(d["X"].astype(np.int16))
        yy.append(int(d["label"]))
        dd.append(int(d["duration_us"]))
    return np.stack(XX), np.asarray(yy, np.int64), np.asarray(dd, np.int64)


if RUN_ATTACKS:
    Xattack, yattack, duration_attack_us = collect_attack_arrays(test_tensor_manifest)
    print("attack set:", Xattack.shape)

    V2_CKPT_DIR = OUT / "attack_partial_v2"
    V2_CKPT_DIR.mkdir(parents=True, exist_ok=True)

    def _unit_path(seed, name):
        return V2_CKPT_DIR / f"unit_seed{seed}_{name}.json"

    def _load_unit(seed, name):
        fp = _unit_path(seed, name)
        if not fp.exists():
            return None
        try:
            with open(fp) as f:
                return json.load(f)
        except Exception as e:
            print(f"ignoring unreadable partial {fp.name}: {e!r}")
            return None

    def _save_unit(seed, name, payload):
        fp = _unit_path(seed, name)
        tmp = fp.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(payload, f)
        os.replace(tmp, fp)

    main_rows, class_rows, verify_rows = [], [], []
    trained_ok = set(
        (r["model"], r["seed"]) for r in clean_rows
        if r["clean_test_accuracy"] >= GATE
    )

    for seed in RUN_SEEDS:
        frame_model = load_ckpt_model(
            FrameResNet18, CKPT_ROOT / f"frame_resnet18_seed{seed}.pt")
        clean_frame_logits = predict_logits(frame_model, Xattack)
        clean_frame_pred = clean_frame_logits.argmax(1)

        coarse2 = None
        c2_ckpt = CKPT_ROOT / f"coarse_frameformer_seed{seed}.pt"
        if c2_ckpt.exists():
            coarse2 = load_ckpt_model(CoarseFrameFormer, c2_ckpt)
            clean_c2_logits = predict_logits(coarse2, Xattack)
            clean_c2_pred = clean_c2_logits.argmax(1)

        for name in ATTACK_MODELS:
            if (name, seed) not in trained_ok:
                print(f"skip attack: {name} seed {seed} (gated out or untrained)")
                continue
            cached = _load_unit(seed, name)
            if cached is not None:
                print(f"resuming: seed {seed}/{name} done "
                      f"({len(cached['main_rows'])} rows)")
                main_rows.extend(cached["main_rows"])
                class_rows.extend(cached["class_rows"])
                verify_rows.extend(cached.get("verify_rows", []))
                continue

            u_main, u_class, u_verify = [], [], []
            model = load_ckpt_model(
                MODEL_SPECS[name]["builder"], CKPT_ROOT / f"{name}_seed{seed}.pt")

            clean_logits = predict_logits(model, Xattack)
            clean_pred = clean_logits.argmax(1)
            clean_correct = clean_pred == yattack
            clean_acc = float(clean_correct.mean())
            print(f"\n[{name} seed {seed}] clean on attack set: {clean_acc:.4f}")

            adv_by_b = {float(b): [] for b in ATTACK_BUDGETS}
            shf_by_b = {float(b): [] for b in ATTACK_BUDGETS}
            t0 = time.time()
            for i in range(len(Xattack)):
                ck = progressive_unique_attack(model, Xattack[i], yattack[i],
                                               budgets=ATTACK_BUDGETS)
                for b in ATTACK_BUDGETS:
                    adv_by_b[float(b)].append(ck[float(b)]["X"])
                    shf_by_b[float(b)].append(ck[float(b)]["shifts_bins"])
                if (i + 1) % 50 == 0:
                    print(f"  gradient {i+1}/{len(Xattack)} "
                          f"({time.time()-t0:.0f}s)")

            for b in ATTACK_BUDGETS:
                b = float(b)
                Xadv = np.stack(adv_by_b[b])
                adv_pred = predict_logits(model, Xadv).argmax(1)
                fm = batch_coarse_metrics(Xattack, Xadv)
                adv_frame_logits = predict_logits(frame_model, Xadv)

                bin_ms = duration_attack_us / T_FINE / 1000.0
                mean_ms = [float(np.abs(s).mean()) * bin_ms[i] if len(s) else 0.0
                           for i, s in enumerate(shf_by_b[b])]
                realized = [len(s) / max(1, int(Xattack[i].sum()))
                            for i, s in enumerate(shf_by_b[b])]

                row = {
                    "seed": seed, "model": name, "attack": "gradient_unique",
                    "budget": b, "clean_accuracy": clean_acc,
                    "attacked_accuracy": float((adv_pred == yattack).mean()),
                    "asr_clean_correct": float(
                        (adv_pred[clean_correct] != yattack[clean_correct]).mean()),
                    "prediction_flip_rate": float((adv_pred != clean_pred).mean()),
                    "mean_realized_unique_fraction": float(np.mean(realized)),
                    "mean_abs_shift_ms": float(np.mean(mean_ms)),
                    **fm,
                    "frame_logits_exact": bool(
                        np.array_equal(adv_frame_logits, clean_frame_logits)),
                    "frame_prediction_flip_rate": float(
                        (adv_frame_logits.argmax(1) != clean_frame_pred).mean()),
                }
                if coarse2 is not None:
                    c2 = predict_logits(coarse2, Xadv)
                    row["coarse2_logits_exact"] = bool(
                        np.array_equal(c2, clean_c2_logits))
                    row["coarse2_flip_rate"] = float(
                        (c2.argmax(1) != clean_c2_pred).mean())
                u_main.append(row)

                for cls in range(NUM_CLASSES):
                    m = yattack == cls
                    cm = m & clean_correct
                    u_class.append({
                        "seed": seed, "model": name,
                        "attack": "gradient_unique", "budget": b, "class": cls,
                        "n": int(m.sum()),
                        "clean_accuracy": float((clean_pred[m] == yattack[m]).mean()),
                        "attacked_accuracy": float((adv_pred[m] == yattack[m]).mean()),
                        "asr_clean_correct": float(
                            (adv_pred[cm] != yattack[cm]).mean())
                            if cm.sum() else float("nan"),
                    })

                # ---- uniform random control
                rng = np.random.default_rng(310000 + seed * 1000 + int(b * 10000))
                Xu = []
                for i in range(len(Xattack)):
                    n_mv = len(shf_by_b[b][i])
                    unit_g, unit_s = event_units_from_original(Xattack[i])
                    n_mv = min(n_mv, len(unit_g))
                    chosen = rng.permutation(len(unit_g))[:n_mv]
                    Xg = grouped(Xattack[i]).astype(np.int32)
                    gs = unit_g[chosen]
                    ss = unit_s[chosen].astype(np.int64)
                    offs = rng.integers(1, S_PER_FRAME, size=n_mv)
                    ds = (ss + offs) % S_PER_FRAME     # uniform over the 7 non-src bins
                    np.subtract.at(Xg, (gs, ss), 1)
                    np.add.at(Xg, (gs, ds), 1)
                    Xr = ungrouped(Xg).astype(np.int16)
                    assert Xr.min() >= 0 and coarse_equal(Xattack[i], Xr)
                    Xu.append(Xr)
                Xu = np.stack(Xu)
                upred = predict_logits(model, Xu).argmax(1)
                ufl = predict_logits(frame_model, Xu)
                u_main.append({
                    "seed": seed, "model": name, "attack": "random_uniform",
                    "budget": b, "clean_accuracy": clean_acc,
                    "attacked_accuracy": float((upred == yattack).mean()),
                    "asr_clean_correct": float(
                        (upred[clean_correct] != yattack[clean_correct]).mean()),
                    "prediction_flip_rate": float((upred != clean_pred).mean()),
                    **batch_coarse_metrics(Xattack, Xu),
                    "frame_logits_exact": bool(
                        np.array_equal(ufl, clean_frame_logits)),
                    "frame_prediction_flip_rate": float(
                        (ufl.argmax(1) != clean_frame_pred).mean()),
                })

                # ---- EXACT displacement-matched control
                rng = np.random.default_rng(320000 + seed * 1000 + int(b * 10000))
                Xm, n_fallback, n_verified = [], 0, 0
                for i in range(len(Xattack)):
                    xm, sm, exact_ok = exact_displacement_matched_random_attack(
                        Xattack[i], shf_by_b[b][i], rng)
                    if exact_ok:
                        v = verify_exact_displacement_match(shf_by_b[b][i], sm)
                        n_verified += int(v["signed_histogram_exact"])
                    else:
                        n_fallback += 1
                    Xm.append(xm)
                Xm = np.stack(Xm)
                mpred = predict_logits(model, Xm).argmax(1)
                mfl = predict_logits(frame_model, Xm)
                u_main.append({
                    "seed": seed, "model": name,
                    "attack": "random_displacement_matched_exact",
                    "budget": b, "clean_accuracy": clean_acc,
                    "attacked_accuracy": float((mpred == yattack).mean()),
                    "asr_clean_correct": float(
                        (mpred[clean_correct] != yattack[clean_correct]).mean()),
                    "prediction_flip_rate": float((mpred != clean_pred).mean()),
                    **batch_coarse_metrics(Xattack, Xm),
                    "frame_logits_exact": bool(
                        np.array_equal(mfl, clean_frame_logits)),
                    "frame_prediction_flip_rate": float(
                        (mfl.argmax(1) != clean_frame_pred).mean()),
                    "exact_match_fraction": float(
                        (len(Xattack) - n_fallback) / len(Xattack)),
                    "matched_fallback_count": int(n_fallback),
                })
                u_verify.append({
                    "seed": seed, "model": name, "budget": b,
                    "samples": len(Xattack),
                    "signed_histogram_exact": int(n_verified),
                    "fallback": int(n_fallback),
                })
                print(f"  budget {b}: grad ASR "
                      f"{u_main[-3]['asr_clean_correct']:.3f} | uniform "
                      f"{u_main[-2]['asr_clean_correct']:.3f} | exact-matched "
                      f"{u_main[-1]['asr_clean_correct']:.3f} "
                      f"(fallback {n_fallback})")
                del Xadv, Xu, Xm
                gc.collect()

            _save_unit(seed, name, {"main_rows": u_main, "class_rows": u_class,
                                    "verify_rows": u_verify})
            main_rows.extend(u_main)
            class_rows.extend(u_class)
            verify_rows.extend(u_verify)
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        del frame_model
        if coarse2 is not None:
            del coarse2
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    main_v2 = pd.DataFrame(main_rows)
    class_v2 = pd.DataFrame(class_rows)
    verify_v2 = pd.DataFrame(verify_rows)
    main_v2.to_csv(OUT / "main_attack_results_by_seed_v2.csv", index=False)
    class_v2.to_csv(OUT / "per_class_gradient_results_v2.csv", index=False)
    verify_v2.to_csv(OUT / "exact_match_verification.csv", index=False)

    if len(main_v2):
        print("\n=== v2 invariance ===")
        print("frame_exact_equal all:", bool(main_v2["frame_exact_equal"].all()),
              "| max frame diff:", int(main_v2["max_integer_frame_difference"].max()),
              "| frame logits exact all:", bool(main_v2["frame_logits_exact"].all()))
        if "coarse2_logits_exact" in main_v2:
            print("coarse_frameformer logits exact all:",
                  bool(main_v2["coarse2_logits_exact"].dropna().all()))
        agg = (main_v2.groupby(["model", "attack", "budget"])
               [["attacked_accuracy", "asr_clean_correct"]].agg(["mean", "std"]))
        display(agg)
        agg.to_csv(OUT / "main_attack_aggregate_v2.csv")

print("\nDONE. Outputs under:", OUT)
print("  addon_clean_accuracy_by_seed.csv")
print("  main_attack_results_by_seed_v2.csv / aggregate_v2 / per_class_v2")
print("  exact_match_verification.csv")

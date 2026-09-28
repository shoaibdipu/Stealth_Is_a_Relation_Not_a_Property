#!/usr/bin/env python3
"""
Event-Security Study — CIFAR10-DVS diagnostics correction (v2 fixed tolerance).

PURPOSE
-------
Rerun ONLY the two diagnostics whose optimization schedules were mismatched in
EVS_CIFAR10DVS_FINAL.py. This script does NOT retrain models and does NOT
rerun the main attack table.

Corrections
-----------
1) Max-shift ablation:
   Old diagnostic attacked [0.10] in one gradient step.
   Correct diagnostic follows the same progressive path as the main attack:
       0.01 -> 0.02 -> 0.05 -> 0.10
   and reports the 0.10 checkpoint. Therefore the "full" (no max-shift limit)
   row should reproduce the main seed-0 10% attack, modulo numerical
   nondeterminism.

2) Cost of stealth:
   Old diagnostic gave the constrained attack one gradient update at each
   requested budget while the free attack used multiple rounds.
   Correct diagnostic gives BOTH attacks the identical progressive schedule:
       0.01 -> 0.02 -> 0.05 -> 0.10 -> 0.20
   and reports the 0.10 and 0.20 checkpoints.

All model definitions and the constrained attack primitive below are extracted
unchanged from the frozen CIFAR10-DVS codebase.

Outputs (new names; old diagnostics are never overwritten)
-----------------------------------------------------------
  results/max_shift_ablation_v2.csv
  results/max_shift_ablation_v2_main_check.csv
  results/cost_of_stealth_v2.csv
  results/diagnostic_fix_v2_config.json
  results/attack_partial_<protocol>/diagfix_v2_*.json

Normal launch
-------------
  python3 EVS_CIFAR10DVS_DIAGNOSTICS_V2.py

Environment
-----------
  CIFAR_DVS_RUN       existing CIFAR run root
      default /path/to/workspace/EVS_CIFAR10DVS/run
  DIAGNOSTIC_SEED     default 0
  RUN_SHIFT_ABLATION  default 1
  RUN_COST_OF_STEALTH default 1
  COST_SUBSET         default 250

Requirements
------------
The original CIFAR run must already contain:
  results/test_tensor_manifest.csv
  results/attack_subset_manifest.csv    (preferred; exact published subset)
  results/clean_accuracy_by_seed.csv
  results/run_config.json
  checkpoints/*_<protocol_tag>.pt
"""

import os, gc, json, math, time, random
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("torch:", torch.__version__, "| device:", DEVICE)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
torch.backends.cudnn.benchmark = True

ROOT = Path(os.environ.get(
    "CIFAR_DVS_RUN", "/path/to/workspace/EVS_CIFAR10DVS/run"))
CKPT_ROOT = ROOT / "checkpoints"
OUT = ROOT / "results"

cfg_path = OUT / "run_config.json"
if not cfg_path.exists():
    raise FileNotFoundError(f"missing original run config: {cfg_path}")
CFG = json.load(open(cfg_path))

# Freeze against the original run rather than silently inventing new constants.
T_FINE = int(CFG["t_fine"])
S_PER_FRAME = int(CFG["s_per_frame"])
N_COARSE = int(CFG["n_coarse"])
MODEL_H = MODEL_W = int(CFG["model_hw"])
NUM_CLASSES = 10
PROTOCOL_TAG = str(CFG["protocol_tag"])

if (T_FINE, S_PER_FRAME, N_COARSE, MODEL_H) != (80, 8, 10, 64):
    raise RuntimeError(
        "This correction script targets the frozen CIFAR protocol "
        f"(80,8,10,64); run config contains "
        f"{(T_FINE,S_PER_FRAME,N_COARSE,MODEL_H)}")

ATTACK_BUDGETS = [float(x) for x in CFG["attack_budgets"]]
EXPECTED = [0.01, 0.02, 0.05, 0.10, 0.20]
if ATTACK_BUDGETS != EXPECTED:
    raise RuntimeError(f"unexpected original attack budgets: {ATTACK_BUDGETS}")

SHIFT_ABLATION_BUDGET = 0.10
SHIFT_ABLATION_BINS = [1, 2, 4, None]
ABLATION_PATH = [0.01, 0.02, 0.05, 0.10]

COST_BUDGETS = [0.10, 0.20]
COST_PATH = [0.01, 0.02, 0.05, 0.10, 0.20]
COST_SUBSET = int(os.environ.get("COST_SUBSET", "250"))

EVAL_BATCH = 48
DIAGNOSTIC_SEED = int(os.environ.get("DIAGNOSTIC_SEED", "0"))
RUN_SHIFT_ABLATION = os.environ.get("RUN_SHIFT_ABLATION", "1") == "1"
RUN_COST_OF_STEALTH = os.environ.get("RUN_COST_OF_STEALTH", "1") == "1"

TEMPORAL_MODELS = [
    "conv_snn",
    "sew_resnet18",
    "event_transformer_v2",
    "temporal_gru",
]
OFFICIAL_MODELS = ["coarse_frameformer"] + TEMPORAL_MODELS

print("ROOT:", ROOT)
print("protocol:", PROTOCOL_TAG)
print("diagnostic seed:", DIAGNOSTIC_SEED)
print("ablation path:", ABLATION_PATH)
print("cost path:", COST_PATH)


def autocast_context():
    if DEVICE.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return torch.autocast(device_type="cpu", enabled=False)


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


def grouped_free(X):
    return X.reshape(T_FINE, 2 * MODEL_H * MODEL_W).T.copy()


def ungrouped_free(Xg):
    return Xg.T.reshape(T_FINE, 2, MODEL_H, MODEL_W)


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


# Builders intentionally match the original frozen CIFAR registry.
MODEL_BUILDERS = {
    "coarse_frameformer": CoarseFrameFormer,
    "conv_snn": ConvSNN,
    "sew_resnet18": SEWResNet18,
    "event_transformer_v2": EventTemporalTransformerV2,
    "temporal_gru": TemporalGRU_DVS,
}


# ============================================================
# Correct free progressive attack
# ============================================================
def best_free_unique_moves(Xcur, grad, available, n_moves):
    """Same greedy selector as the constrained attack, but the group is the
    whole recording for each (polarity, pixel), so destinations span T_FINE.
    `available` contains only original, never-before-moved events.
    """
    Xg = grouped_free(Xcur).astype(np.int32)
    Gg = grouped_free(grad).astype(np.float32)
    Xg, Ag, shifts = _select_moves(
        Xg, Gg, available, n_moves, T_FINE, max_shift_bins=None)
    Xnext = ungrouped_free(Xg).astype(np.int16)
    assert Xnext.min() >= 0
    assert int(Xnext.sum()) == int(Xcur.sum())
    return Xnext, Ag, shifts.astype(np.int16)


def progressive_free_unique_attack(model, X0, y, budgets):
    """Free retiming with exactly one gradient recomputation at every
    progressive budget checkpoint, matching progressive_unique_attack.
    """
    budgets = sorted(float(b) for b in budgets)
    total = int(X0.sum())
    Xcur = X0.copy().astype(np.int16)
    available = grouped_free(X0).astype(np.int32)
    moved, shifts_all = 0, []
    ck = {}
    for b in budgets:
        need = int(round(b * total)) - moved
        if need > 0:
            grad = input_gradient(model, Xcur, y)
            Xcur, available, sh = best_free_unique_moves(
                Xcur, grad, available, need)
            moved += len(sh)
            shifts_all.extend(sh.tolist())
        ck[b] = {
            "X": Xcur.copy(),
            "moved": int(moved),
            "shifts": np.asarray(shifts_all, dtype=np.int16).copy(),
        }
    return ck


# ============================================================
# Load exact published attack subset and landed checkpoints
# ============================================================
def load_attack_set():
    test_manifest_path = OUT / "test_tensor_manifest.csv"
    attack_manifest_path = OUT / "attack_subset_manifest.csv"
    if not test_manifest_path.exists():
        raise FileNotFoundError(test_manifest_path)

    if attack_manifest_path.exists():
        attack_manifest = pd.read_csv(attack_manifest_path)
        print("using published attack subset:", attack_manifest_path)
    else:
        # Exact reconstruction of the frozen selection, only as a fallback.
        test_manifest = pd.read_csv(test_manifest_path)
        attack_per_class = int(CFG["attack_per_class"])
        rng = np.random.default_rng(9090)
        parts = []
        for cls in range(NUM_CLASSES):
            sub = test_manifest[test_manifest["label"] == cls].reset_index(drop=True)
            chosen = rng.choice(
                len(sub), size=min(attack_per_class, len(sub)), replace=False)
            parts.append(sub.iloc[chosen])
        attack_manifest = (
            pd.concat(parts).sample(frac=1, random_state=9091).reset_index(drop=True)
        )
        attack_manifest.to_csv(
            OUT / "attack_subset_manifest_reconstructed_diagfix_v2.csv", index=False)
        print("WARNING: original attack_subset_manifest.csv absent; reconstructed")

    XX, yy, dd = [], [], []
    for _, r in attack_manifest.iterrows():
        d = np.load(r["path"])
        XX.append(d["X"].astype(np.int16))
        yy.append(int(d["label"]))
        dd.append(int(d["duration_us"]))
    X = np.stack(XX)
    y = np.asarray(yy, np.int64)
    dur = np.asarray(dd, np.int64)
    print("attack set:", X.shape,
          "| per class:", np.bincount(y, minlength=NUM_CLASSES))
    return attack_manifest, X, y, dur


def load_model(name, seed):
    if name not in MODEL_BUILDERS:
        raise ValueError(name)
    fp = CKPT_ROOT / f"{name}_seed{seed}_{PROTOCOL_TAG}.pt"
    if not fp.exists():
        raise FileNotFoundError(f"missing landed checkpoint: {fp}")
    m = MODEL_BUILDERS[name]().to(DEVICE)
    m.load_state_dict(torch.load(fp, map_location=DEVICE))
    m.eval()
    return m


UNIT_DIR = OUT / f"attack_partial_{PROTOCOL_TAG}"
UNIT_DIR.mkdir(parents=True, exist_ok=True)


def _diag_fp(tag):
    return UNIT_DIR / f"diagfix_v2_{tag}.json"


def _load_diag(tag):
    fp = _diag_fp(tag)
    if not fp.exists():
        return None
    try:
        return json.load(open(fp))
    except Exception as e:
        print("ignoring unreadable diagnostic cache", fp, repr(e))
        return None


def _save_diag(tag, payload):
    fp = _diag_fp(tag)
    tmp = fp.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f)
    os.replace(tmp, fp)


def safe_asr(pred, y, clean_correct):
    n = int(clean_correct.sum())
    if n == 0:
        return float("nan"), 0
    return float((pred[clean_correct] != y[clean_correct]).mean()), n


def main():
    clean_path = OUT / "clean_accuracy_by_seed.csv"
    if not clean_path.exists():
        raise FileNotFoundError(clean_path)
    clean_df = pd.read_csv(clean_path)
    trained_ok = {
        (str(r["model"]), int(r["seed"]))
        for _, r in clean_df.iterrows()
    }
    for name in OFFICIAL_MODELS:
        if (name, DIAGNOSTIC_SEED) not in trained_ok:
            raise RuntimeError(
                f"{name} seed {DIAGNOSTIC_SEED} missing from clean result table")

    _, Xattack, yattack, dur_us = load_attack_set()
    bin_ms = dur_us / T_FINE / 1000.0

    # --------------------------------------------------------
    # A. Corrected max-shift ablation
    # --------------------------------------------------------
    if RUN_SHIFT_ABLATION:
        abl_rows = []
        for name in TEMPORAL_MODELS:
            tag = f"ablation_seed{DIAGNOSTIC_SEED}_{name}"
            cached = _load_diag(tag)
            if cached is not None:
                print("resume corrected ablation:", name)
                abl_rows += cached["rows"]
                continue

            model = load_model(name, DIAGNOSTIC_SEED)
            clean_pred = predict_logits(model, Xattack).argmax(1)
            clean_correct = clean_pred == yattack
            rows = []

            for ms in SHIFT_ABLATION_BINS:
                t0 = time.time()
                Xa, sh = [], []
                for i in range(len(Xattack)):
                    ck = progressive_unique_attack(
                        model, Xattack[i], yattack[i],
                        ABLATION_PATH, max_shift_bins=ms)
                    out = ck[SHIFT_ABLATION_BUDGET]
                    Xa.append(out["X"])
                    sh.append(out["shifts"].astype(np.int16))
                    if (i + 1) % 100 == 0:
                        print(
                            f"  {name} shift={'full' if ms is None else ms}: "
                            f"{i+1}/{len(Xattack)} ({time.time()-t0:.0f}s)")
                Xa = np.stack(Xa)
                pred = predict_logits(model, Xa).argmax(1)
                asr, ncc = safe_asr(pred, yattack, clean_correct)
                mean_ms = [
                    float(np.abs(s).mean()) * bin_ms[i] if len(s) else 0.0
                    for i, s in enumerate(sh)
                ]
                row = {
                    "seed": DIAGNOSTIC_SEED,
                    "model": name,
                    "budget": SHIFT_ABLATION_BUDGET,
                    "optimization_path": "0.01->0.02->0.05->0.10",
                    "gradient_updates_max": len(ABLATION_PATH),
                    "max_shift_label": "full" if ms is None else str(ms),
                    "n": len(Xattack),
                    "n_clean_correct": ncc,
                    "clean_accuracy": float(clean_correct.mean()),
                    "attacked_accuracy": float((pred == yattack).mean()),
                    "asr_clean_correct": asr,
                    "mean_abs_shift_ms": float(np.mean(mean_ms)),
                    **batch_coarse_metrics(Xattack, Xa),
                    "seconds": float(time.time() - t0),
                }
                rows.append(row)
                print(
                    f"  corrected ablation {name} shift={row['max_shift_label']}: "
                    f"ASR={asr:.4f}, acc={row['attacked_accuracy']:.4f}")
                del Xa
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            _save_diag(tag, {"rows": rows})
            abl_rows += rows
            del model
            gc.collect()

        abl_df = pd.DataFrame(abl_rows)
        abl_out = OUT / "max_shift_ablation_v2.csv"
        abl_df.to_csv(abl_out, index=False)
        print("wrote:", abl_out)

        # Internal consistency check against the landed main seed-0 10% row.
        main_path = OUT / "main_attack_results_by_seed.csv"
        checks = []
        if main_path.exists():
            main_df = pd.read_csv(main_path)
            for name in TEMPORAL_MODELS:
                old = main_df[
                    (main_df["seed"] == DIAGNOSTIC_SEED)
                    & (main_df["model"] == name)
                    & (main_df["attack"] == "gradient_unique")
                    & np.isclose(main_df["budget"].astype(float), 0.10)
                ]
                new = abl_df[
                    (abl_df["model"] == name)
                    & (abl_df["max_shift_label"] == "full")
                ]
                if len(old) == 1 and len(new) == 1:
                    a = float(old.iloc[0]["asr_clean_correct"])
                    b = float(new.iloc[0]["asr_clean_correct"])
                    diff = abs(a - b)
                    tol = 0.005  # ASR fraction = 0.5 percentage point
                    checks.append({
                        "model": name,
                        "main_seed0_10pct_asr": a,
                        "corrected_ablation_full_asr": b,
                        "absolute_difference": diff,
                        "absolute_difference_pct_points": 100.0 * diff,
                        "consistency_tolerance_pct_points": 0.5,
                        "consistent_within_0p5pt": bool(diff <= tol),
                    })
        check_df = pd.DataFrame(checks)
        check_out = OUT / "max_shift_ablation_v2_main_check.csv"
        check_df.to_csv(check_out, index=False)
        if len(check_df):
            print("\nfull-shift consistency check:")
            print(check_df.to_string(index=False))
        print("wrote:", check_out)

    # --------------------------------------------------------
    # B. Corrected cost-of-stealth
    # --------------------------------------------------------
    if RUN_COST_OF_STEALTH:
        n_sub = min(COST_SUBSET, len(Xattack))
        sub = np.arange(n_sub)
        Xs = Xattack[sub]
        ys = yattack[sub]
        bins_ms_sub = bin_ms[sub]

        cost_rows = []
        for name in TEMPORAL_MODELS:
            tag = f"cost_seed{DIAGNOSTIC_SEED}_{name}"
            cached = _load_diag(tag)
            if cached is not None:
                print("resume corrected cost:", name)
                cost_rows += cached["rows"]
                continue

            model = load_model(name, DIAGNOSTIC_SEED)
            coarse = load_model("coarse_frameformer", DIAGNOSTIC_SEED)
            clean_pred = predict_logits(model, Xs).argmax(1)
            clean_correct = clean_pred == ys
            cclean = predict_logits(coarse, Xs)

            # Run each progressive attack ONCE per sample through 20%, then
            # read both 10% and 20% checkpoints.
            stealth_by_b = {b: [] for b in COST_BUDGETS}
            stealth_sh = {b: [] for b in COST_BUDGETS}
            free_by_b = {b: [] for b in COST_BUDGETS}
            free_sh = {b: [] for b in COST_BUDGETS}

            t0 = time.time()
            for i in range(n_sub):
                cs = progressive_unique_attack(
                    model, Xs[i], ys[i], COST_PATH)
                cf = progressive_free_unique_attack(
                    model, Xs[i], ys[i], COST_PATH)
                for b in COST_BUDGETS:
                    stealth_by_b[b].append(cs[b]["X"])
                    stealth_sh[b].append(cs[b]["shifts"].astype(np.int16))
                    free_by_b[b].append(cf[b]["X"])
                    free_sh[b].append(cf[b]["shifts"].astype(np.int16))
                if (i + 1) % 50 == 0:
                    print(
                        f"  cost {name}: {i+1}/{n_sub} "
                        f"({time.time()-t0:.0f}s)")

            rows = []
            for b in COST_BUDGETS:
                for kind, arrays, shifts in [
                    ("stealth", stealth_by_b[b], stealth_sh[b]),
                    ("free", free_by_b[b], free_sh[b]),
                ]:
                    Xa = np.stack(arrays)
                    pred = predict_logits(model, Xa).argmax(1)
                    asr, ncc = safe_asr(pred, ys, clean_correct)
                    mean_ms = [
                        float(np.abs(s).mean()) * bins_ms_sub[i] if len(s) else 0.0
                        for i, s in enumerate(shifts)
                    ]
                    row = {
                        "seed": DIAGNOSTIC_SEED,
                        "model": name,
                        "kind": kind,
                        "budget": float(b),
                        "optimization_path": "0.01->0.02->0.05->0.10->0.20",
                        "gradient_updates_max": len(COST_PATH),
                        "n": n_sub,
                        "n_clean_correct": ncc,
                        "clean_accuracy": float(clean_correct.mean()),
                        "attacked_accuracy": float((pred == ys).mean()),
                        "asr_clean_correct": asr,
                        "prediction_flip_rate": float((pred != clean_pred).mean()),
                        "mean_abs_shift_ms": float(np.mean(mean_ms)),
                        **batch_coarse_metrics(Xs, Xa),
                    }
                    ca = predict_logits(coarse, Xa)
                    row["coarse_frameformer_logits_exact"] = bool(
                        np.array_equal(ca, cclean))
                    row["coarse_frameformer_flip_rate"] = float(
                        (ca.argmax(1) != cclean.argmax(1)).mean())
                    rows.append(row)
                    print(
                        f"  corrected cost {name} {kind} b={b}: "
                        f"ASR={asr:.4f}, frame-maxdiff="
                        f"{row['max_integer_frame_difference']}")
                    del Xa
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            _save_diag(tag, {"rows": rows})
            cost_rows += rows
            del model, coarse
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        cost_df = pd.DataFrame(cost_rows)
        cost_out = OUT / "cost_of_stealth_v2.csv"
        cost_df.to_csv(cost_out, index=False)
        print("wrote:", cost_out)

    with open(OUT / "diagnostic_fix_v2_config.json", "w") as f:
        json.dump({
            "source_protocol_tag": PROTOCOL_TAG,
            "diagnostic_seed": DIAGNOSTIC_SEED,
            "ablation_path": ABLATION_PATH,
            "shift_ablation_budget": SHIFT_ABLATION_BUDGET,
            "shift_ablation_bins": [
                "full" if x is None else int(x) for x in SHIFT_ABLATION_BINS
            ],
            "cost_path": COST_PATH,
            "cost_budgets": COST_BUDGETS,
            "cost_subset": COST_SUBSET,
            "notes": (
                "Diagnostics-only correction. No retraining and no main-attack "
                "rerun. Both cost-of-stealth arms receive identical progressive "
                "gradient-update schedules."
            ),
        }, f, indent=2)

    print("\nDONE. New corrected diagnostics are under:", OUT)


if __name__ == "__main__":
    main()

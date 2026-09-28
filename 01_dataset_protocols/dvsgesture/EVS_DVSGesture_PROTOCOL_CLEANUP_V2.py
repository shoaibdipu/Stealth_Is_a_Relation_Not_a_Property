#!/usr/bin/env python3
"""
Event-Security Study — DVS128 Gesture protocol cleanup v2.

This script reruns ONLY protocol-cleanup experiments. It never retrains a model
and never overwrites the landed main/add-on result tables.

What it does
------------
A) Corrected max-shift ablation, seed 0, ALL FOUR successful temporal victims:
      conv_snn, sew_resnet18, event_transformer_v2, temporal_gru
   using the same progressive optimization path as the DVS headline attack:
      0.02 -> 0.05 -> 0.10
   at max-shift limits [1, 2, 4, full].

B) Corrected cost-of-stealth, seed 0, ALL FOUR victims, ALL 264 test streams:
   stealth and free attacks both use exactly the same progressive path:
      0.02 -> 0.05 -> 0.10 -> 0.20
   and report the 10% and 20% checkpoints.

C) Strict exact signed-displacement controls for the two original main-run SNNs
   (ConvSNN and SEW-ResNet18), all three seeds and all four budgets. The
   integral max-flow matcher has NO approximate fallback. TransformerV2 and GRU
   already have landed exact controls from the add-on run.

Checkpoint provenance
---------------------
Main-run checkpoints:
  frame_resnet18_seed{seed}.pt
  conv_snn_seed{seed}.pt
  sew_resnet18_seed{seed}.pt

Add-on checkpoints:
  event_transformer_v2_seed{seed}.pt
  temporal_gru_seed{seed}.pt
  coarse_frameformer_seed{seed}.pt

Normal launch
-------------
  python3 EVS_DVSGesture_PROTOCOL_CLEANUP_V2.py

Environment
-----------
  DVS_GESTURE_RUN          existing run root
  RUN_ABLATION             default 1
  RUN_COST                 default 1
  RUN_EXACT_CONTROLS       default 1
  DIAGNOSTIC_SEED          default 0
  CONSISTENCY_TOL_PP       default 0.5

Outputs are new *_protocol_v2 files and cleanup_v2 partials.
"""

import os, gc, json, math, time, random
from pathlib import Path
from collections import Counter, deque

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("torch:", torch.__version__, "| device:", DEVICE)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
torch.backends.cudnn.benchmark = True

ROOT = Path(os.environ.get(
    "DVS_GESTURE_RUN", "/path/to/workspace/EVS_DVSGesture/run"))
TENSOR_CACHE = ROOT / "cache_tensors"
CKPT_ROOT = ROOT / "checkpoints"
OUT = ROOT / "results"
for p in [CKPT_ROOT, OUT]:
    p.mkdir(parents=True, exist_ok=True)

NUM_CLASSES = 11
MODEL_H = MODEL_W = 64
T_FINE = 160
S_PER_FRAME = 8
N_COARSE = T_FINE // S_PER_FRAME
EVAL_BATCH = 32

ATTACK_BUDGETS = [0.02, 0.05, 0.10, 0.20]
ABLATION_PATH = [0.02, 0.05, 0.10]
SHIFT_ABLATION_BUDGET = 0.10
SHIFT_ABLATION_BINS = [1, 2, 4, None]
COST_PATH = [0.02, 0.05, 0.10, 0.20]
COST_BUDGETS = [0.10, 0.20]

RUN_ABLATION = os.environ.get("RUN_ABLATION", "1") == "1"
RUN_COST = os.environ.get("RUN_COST", "1") == "1"
RUN_EXACT_CONTROLS = os.environ.get("RUN_EXACT_CONTROLS", "1") == "1"
DIAGNOSTIC_SEED = int(os.environ.get("DIAGNOSTIC_SEED", "0"))
CONSISTENCY_TOL_PP = float(os.environ.get("CONSISTENCY_TOL_PP", "0.5"))

TEMPORAL_MODELS = [
    "conv_snn",
    "sew_resnet18",
    "event_transformer_v2",
    "temporal_gru",
]
EXACT_CONTROL_MODELS = ["conv_snn", "sew_resnet18"]
SEEDS = [0, 1, 2]


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
    return {
        "frame_exact_equal": bool(np.array_equal(Ac, Bc)),
        "max_integer_frame_difference":
            int(np.abs(Ac.astype(np.int64) - Bc.astype(np.int64)).max()),
    }


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


def event_units_from_original(X0):
    Xg = grouped(X0).astype(np.int32)
    g_idx, s_idx = np.nonzero(Xg > 0)
    counts = Xg[g_idx, s_idx]
    return (np.repeat(g_idx, counts).astype(np.int32),
            np.repeat(s_idx, counts).astype(np.int8))


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


MODEL_BUILDERS = {
    "frame_resnet18": FrameResNet18,
    "conv_snn": ConvSNN,
    "sew_resnet18": SEWResNet18,
    "event_transformer_v2": EventTemporalTransformerV2,
    "temporal_gru": TemporalGRU_DVS,
    "coarse_frameformer": CoarseFrameFormer,
}


def _manifest(split):
    fp = TENSOR_CACHE / f"{split}_tensor_manifest.csv"
    if not fp.exists():
        raise FileNotFoundError(
            f"{fp} missing. Run the landed DVS main preprocessing first.")
    return pd.read_csv(fp)


def collect_attack_arrays(manifest):
    XX, yy, dd = [], [], []
    for _, r in manifest.iterrows():
        d = np.load(r["path"])
        XX.append(d["X"].astype(np.int16))
        yy.append(int(d["label"]))
        dd.append(int(d["duration_us"]))
    return np.stack(XX), np.asarray(yy, np.int64), np.asarray(dd, np.int64)


def checkpoint_path(name, seed):
    fp = CKPT_ROOT / f"{name}_seed{seed}.pt"
    if not fp.exists():
        raise FileNotFoundError(f"missing checkpoint: {fp}")
    return fp


def load_model(name, seed):
    m = MODEL_BUILDERS[name]().to(DEVICE)
    m.load_state_dict(torch.load(checkpoint_path(name, seed), map_location=DEVICE))
    m.eval()
    return m


def grouped_free(X):
    return X.reshape(T_FINE, 2 * MODEL_H * MODEL_W).T.copy()


def ungrouped_free(Xg):
    return Xg.T.reshape(T_FINE, 2, MODEL_H, MODEL_W)


def best_free_unique_moves(Xcur, grad, available, n_moves):
    """Unrestricted temporal move selector with persistent unique-event pool."""
    Xg = grouped_free(Xcur).astype(np.int32)
    Gg = grouped_free(grad).astype(np.float32)
    Ag = available

    groups, srcs = np.nonzero(Ag > 0)
    if len(groups) == 0:
        return Xcur.copy(), Ag, np.asarray([], dtype=np.int16)

    best_dst_all = Gg.argmax(axis=1)
    dst = best_dst_all[groups]
    gains = Gg[groups, dst] - Gg[groups, srcs]
    keep = (dst != srcs) & np.isfinite(gains) & (gains > 0)
    groups, srcs, dst, gains = (
        groups[keep], srcs[keep], dst[keep], gains[keep])

    if len(groups) == 0:
        return Xcur.copy(), Ag, np.asarray([], dtype=np.int16)

    order = np.argsort(-gains, kind="stable")
    groups, srcs, dst = groups[order], srcs[order], dst[order]

    caps = Ag[groups, srcs].astype(np.int64)
    csum = np.cumsum(caps)
    take = caps.copy()
    cut = int(np.searchsorted(csum, n_moves))
    if cut < len(caps):
        take[cut] = max(0, n_moves - (csum[cut - 1] if cut > 0 else 0))
        take[cut + 1:] = 0

    Xg[groups, srcs] -= take
    np.add.at(Xg, (groups, dst), take)
    Ag[groups, srcs] -= take
    shifts = np.repeat(dst - srcs, take).astype(np.int16)

    Xnext = ungrouped_free(Xg).astype(np.int16)
    assert Xnext.min() >= 0
    assert int(Xnext.sum()) == int(Xcur.sum())
    return Xnext, Ag, shifts


def progressive_free_unique_attack(model, X0, y, budgets):
    """One gradient recomputation at each budget checkpoint, like stealth."""
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
            "moved_unique": int(moved),
            "realized_fraction": float(moved / max(1, total)),
            "shifts_bins": np.asarray(shifts_all, dtype=np.int16).copy(),
        }
    return ck


PARTIAL = OUT / "protocol_cleanup_v2_partial"
PARTIAL.mkdir(parents=True, exist_ok=True)


def _part(tag):
    return PARTIAL / f"{tag}.json"


def _load_part(tag):
    fp = _part(tag)
    if not fp.exists():
        return None
    try:
        with open(fp) as f:
            return json.load(f)
    except Exception as e:
        print("ignoring unreadable partial", fp, repr(e))
        return None


def _save_part(tag, obj):
    fp = _part(tag)
    tmp = fp.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, fp)


def safe_asr(pred, y, clean_correct):
    n = int(clean_correct.sum())
    if n == 0:
        return float("nan"), 0
    return float((pred[clean_correct] != y[clean_correct]).mean()), n


def _main_reference_asr(name, seed=0, budget=0.10):
    """Read the correct landed table for consistency checks."""
    candidates = []
    if name in {"conv_snn", "sew_resnet18"}:
        candidates.append(OUT / "main_attack_results_by_seed.csv")
    else:
        candidates.append(OUT / "main_attack_results_by_seed_v2.csv")
    for fp in candidates:
        if not fp.exists():
            continue
        df = pd.read_csv(fp)
        model_col = "model" if "model" in df.columns else "temporal_model"
        budget_col = "budget" if "budget" in df.columns else "budget_requested"
        attack_col = "attack"
        asr_col = (
            "asr_clean_correct" if "asr_clean_correct" in df.columns
            else "asr_on_clean_correct")
        z = df[
            (df[model_col].astype(str) == name)
            & (df["seed"].astype(int) == int(seed))
            & (df[attack_col].astype(str).isin(
                ["gradient_unique", "unique_iterative_gradient"]))
            & np.isclose(df[budget_col].astype(float), float(budget))
        ]
        if len(z) == 1:
            return float(z.iloc[0][asr_col]), str(fp)
    return None, None


def run_ablation(Xattack, yattack, dur_us):
    rows = []
    for name in TEMPORAL_MODELS:
        tag = f"ablation_seed{DIAGNOSTIC_SEED}_{name}"
        cached = _load_part(tag)
        if cached is not None:
            rows += cached["rows"]
            print("resume ablation:", name)
            continue

        model = load_model(name, DIAGNOSTIC_SEED)
        clean_pred = predict_logits(model, Xattack).argmax(1)
        clean_correct = clean_pred == yattack
        unit = []

        for max_shift in SHIFT_ABLATION_BINS:
            XX, all_shifts = [], []
            t0 = time.time()
            for i in range(len(Xattack)):
                ck = progressive_unique_attack(
                    model, Xattack[i], yattack[i],
                    budgets=ABLATION_PATH, max_shift_bins=max_shift)
                out = ck[SHIFT_ABLATION_BUDGET]
                XX.append(out["X"])
                all_shifts.append(out["shifts_bins"])
                if (i + 1) % 50 == 0:
                    print(
                        f"  {name} shift={'full' if max_shift is None else max_shift} "
                        f"{i+1}/{len(Xattack)} ({time.time()-t0:.0f}s)")
            XX = np.stack(XX)
            pred = predict_logits(model, XX).argmax(1)
            asr, ncc = safe_asr(pred, yattack, clean_correct)
            mean_ms = []
            for i, sh in enumerate(all_shifts):
                bin_ms = dur_us[i] / T_FINE / 1000.0
                mean_ms.append(
                    float(np.abs(sh).mean()) * bin_ms if len(sh) else 0.0)
            unit.append({
                "seed": DIAGNOSTIC_SEED,
                "model": name,
                "budget": SHIFT_ABLATION_BUDGET,
                "optimization_path": "0.02->0.05->0.10",
                "gradient_updates_max": len(ABLATION_PATH),
                "max_shift_bins": -1 if max_shift is None else int(max_shift),
                "max_shift_label": "full-frame" if max_shift is None else str(max_shift),
                "n": len(Xattack),
                "n_clean_correct": ncc,
                "clean_accuracy": float(clean_correct.mean()),
                "attacked_accuracy": float((pred == yattack).mean()),
                "asr_clean_correct": asr,
                "mean_abs_shift_ms": float(np.mean(mean_ms)),
                **batch_coarse_metrics(Xattack, XX),
                "seconds": float(time.time() - t0),
            })
            print("  corrected ablation:", unit[-1])
            del XX
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        _save_part(tag, {"rows": unit})
        rows += unit
        del model
        gc.collect()

    df = pd.DataFrame(rows)
    fp = OUT / "max_shift_ablation_protocol_v2.csv"
    df.to_csv(fp, index=False)

    checks = []
    for name in TEMPORAL_MODELS:
        ref, ref_fp = _main_reference_asr(
            name, DIAGNOSTIC_SEED, SHIFT_ABLATION_BUDGET)
        z = df[(df["model"] == name) & (df["max_shift_label"] == "full-frame")]
        if ref is None or len(z) != 1:
            continue
        got = float(z.iloc[0]["asr_clean_correct"])
        diff = abs(ref - got)
        checks.append({
            "model": name,
            "reference_file": ref_fp,
            "main_seed0_10pct_asr": ref,
            "corrected_full_ablation_asr": got,
            "absolute_difference_pct_points": 100.0 * diff,
            "consistency_tolerance_pct_points": CONSISTENCY_TOL_PP,
            "consistent": bool(100.0 * diff <= CONSISTENCY_TOL_PP),
        })
    cdf = pd.DataFrame(checks)
    cdf.to_csv(OUT / "max_shift_ablation_protocol_v2_consistency.csv",
               index=False)

    # Paper-ready regenerated figure.
    if len(df):
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        labels = ["1", "2", "4", "full-frame"]
        for name in TEMPORAL_MODELS:
            z = df[df["model"] == name].copy()
            order = {v: i for i, v in enumerate(labels)}
            z["_o"] = z["max_shift_label"].map(order)
            z = z.sort_values("_o")
            ax.plot(
                range(len(z)), 100.0 * z["asr_clean_correct"].values,
                marker="o", label=name)
        ax.set_xticks(range(len(labels)), labels)
        ax.set_xlabel("Maximum fine-bin displacement")
        ax.set_ylabel("ASR on clean-correct samples (%)")
        ax.set_title("DVS128 Gesture — corrected max-shift ablation")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(OUT / "fig_dvs_shift_ablation_protocol_v2.pdf",
                    bbox_inches="tight")
        fig.savefig(OUT / "fig_dvs_shift_ablation_protocol_v2.png",
                    dpi=220, bbox_inches="tight")
        plt.close(fig)
    return df


def run_cost(Xattack, yattack, dur_us):
    rows = []
    # Requirement from audit: all 264 published test streams.
    Xs, ys, ds = Xattack, yattack, dur_us

    for name in TEMPORAL_MODELS:
        tag = f"cost_seed{DIAGNOSTIC_SEED}_{name}"
        cached = _load_part(tag)
        if cached is not None:
            rows += cached["rows"]
            print("resume cost:", name)
            continue

        model = load_model(name, DIAGNOSTIC_SEED)
        frame = load_model("frame_resnet18", DIAGNOSTIC_SEED)
        coarse = load_model("coarse_frameformer", DIAGNOSTIC_SEED)

        clean_pred = predict_logits(model, Xs).argmax(1)
        clean_correct = clean_pred == ys
        frame_clean = predict_logits(frame, Xs)
        coarse_clean = predict_logits(coarse, Xs)

        stealth = {b: [] for b in COST_BUDGETS}
        stealth_sh = {b: [] for b in COST_BUDGETS}
        free = {b: [] for b in COST_BUDGETS}
        free_sh = {b: [] for b in COST_BUDGETS}

        t0 = time.time()
        for i in range(len(Xs)):
            cs = progressive_unique_attack(model, Xs[i], ys[i], COST_PATH)
            cf = progressive_free_unique_attack(model, Xs[i], ys[i], COST_PATH)
            for b in COST_BUDGETS:
                stealth[b].append(cs[b]["X"])
                stealth_sh[b].append(cs[b]["shifts_bins"])
                free[b].append(cf[b]["X"])
                free_sh[b].append(cf[b]["shifts_bins"])
            if (i + 1) % 50 == 0:
                print(f"  cost {name}: {i+1}/{len(Xs)} ({time.time()-t0:.0f}s)")

        unit = []
        for b in COST_BUDGETS:
            for kind, arrs, shifts in [
                ("stealth", stealth[b], stealth_sh[b]),
                ("free", free[b], free_sh[b]),
            ]:
                XA = np.stack(arrs)
                pred = predict_logits(model, XA).argmax(1)
                asr, ncc = safe_asr(pred, ys, clean_correct)
                mean_ms = []
                for i, sh in enumerate(shifts):
                    bin_ms = ds[i] / T_FINE / 1000.0
                    mean_ms.append(
                        float(np.abs(sh).mean()) * bin_ms if len(sh) else 0.0)
                fl = predict_logits(frame, XA)
                cl = predict_logits(coarse, XA)
                unit.append({
                    "seed": DIAGNOSTIC_SEED,
                    "model": name,
                    "kind": kind,
                    "budget": float(b),
                    "optimization_path": "0.02->0.05->0.10->0.20",
                    "gradient_updates_max": len(COST_PATH),
                    "n": len(Xs),
                    "n_clean_correct": ncc,
                    "clean_accuracy": float(clean_correct.mean()),
                    "attacked_accuracy": float((pred == ys).mean()),
                    "asr_clean_correct": asr,
                    "prediction_flip_rate": float((pred != clean_pred).mean()),
                    "mean_abs_shift_ms": float(np.mean(mean_ms)),
                    **batch_coarse_metrics(Xs, XA),
                    "frame_resnet_logits_exact": bool(
                        np.array_equal(fl, frame_clean)),
                    "frame_resnet_flip_rate": float(
                        (fl.argmax(1) != frame_clean.argmax(1)).mean()),
                    "coarse_frameformer_logits_exact": bool(
                        np.array_equal(cl, coarse_clean)),
                    "coarse_frameformer_flip_rate": float(
                        (cl.argmax(1) != coarse_clean.argmax(1)).mean()),
                })
                del XA
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        _save_part(tag, {"rows": unit})
        rows += unit
        del model, frame, coarse
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "cost_of_stealth_protocol_v2.csv", index=False)
    return df


def run_exact_controls_conv_sew(Xattack, yattack, dur_us):
    """Fill the only DVS exact-control gap: ConvSNN and SEW, all 3 seeds."""
    rows, verify = [], []

    for seed in SEEDS:
        frame = load_model("frame_resnet18", seed)
        coarse = load_model("coarse_frameformer", seed)
        frame_clean = predict_logits(frame, Xattack)
        coarse_clean = predict_logits(coarse, Xattack)

        for name in EXACT_CONTROL_MODELS:
            tag = f"exact_seed{seed}_{name}"
            cached = _load_part(tag)
            if cached is not None:
                rows += cached["rows"]
                verify += cached["verify"]
                print("resume exact control:", seed, name)
                continue

            model = load_model(name, seed)
            clean_pred = predict_logits(model, Xattack).argmax(1)
            clean_correct = clean_pred == yattack
            clean_acc = float(clean_correct.mean())

            target_shifts = {b: [] for b in ATTACK_BUDGETS}
            t0 = time.time()
            for i in range(len(Xattack)):
                ck = progressive_unique_attack(
                    model, Xattack[i], yattack[i], ATTACK_BUDGETS)
                for b in ATTACK_BUDGETS:
                    target_shifts[b].append(ck[b]["shifts_bins"].copy())
                if (i + 1) % 50 == 0:
                    print(
                        f"  exact-source {name} seed={seed}: "
                        f"{i+1}/{len(Xattack)} ({time.time()-t0:.0f}s)")

            unit_rows, unit_verify = [], []
            for b in ATTACK_BUDGETS:
                rng = np.random.default_rng(
                    920000 + seed * 1000 + int(b * 10000)
                    + (0 if name == "conv_snn" else 100000))
                Xm, realized_sh = [], []
                n_exact = 0
                for i in range(len(Xattack)):
                    xm, sm, ok = exact_displacement_matched_random_attack(
                        Xattack[i], target_shifts[b][i], rng)
                    if not ok:
                        raise RuntimeError("strict exact matcher returned false")
                    t = np.asarray(target_shifts[b][i], dtype=np.int16)
                    t = t[t != 0]
                    if Counter(sm.tolist()) != Counter(t.astype(np.int8).tolist()):
                        raise RuntimeError(
                            f"signed shift mismatch: {name} seed={seed} b={b} i={i}")
                    n_exact += 1
                    Xm.append(xm)
                    realized_sh.append(sm)
                Xm = np.stack(Xm)
                pred = predict_logits(model, Xm).argmax(1)
                asr, ncc = safe_asr(pred, yattack, clean_correct)
                mean_ms = []
                for i, sh in enumerate(realized_sh):
                    bin_ms = dur_us[i] / T_FINE / 1000.0
                    mean_ms.append(
                        float(np.abs(sh).mean()) * bin_ms if len(sh) else 0.0)
                fl = predict_logits(frame, Xm)
                cl = predict_logits(coarse, Xm)
                unit_rows.append({
                    "seed": seed,
                    "model": name,
                    "attack": "random_displacement_matched_exact",
                    "budget": float(b),
                    "clean_accuracy": clean_acc,
                    "n": len(Xattack),
                    "n_clean_correct": ncc,
                    "attacked_accuracy": float((pred == yattack).mean()),
                    "asr_clean_correct": asr,
                    "prediction_flip_rate": float((pred != clean_pred).mean()),
                    "mean_abs_shift_ms": float(np.mean(mean_ms)),
                    **batch_coarse_metrics(Xattack, Xm),
                    "frame_resnet_logits_exact": bool(
                        np.array_equal(fl, frame_clean)),
                    "frame_resnet_flip_rate": float(
                        (fl.argmax(1) != frame_clean.argmax(1)).mean()),
                    "coarse_frameformer_logits_exact": bool(
                        np.array_equal(cl, coarse_clean)),
                    "coarse_frameformer_flip_rate": float(
                        (cl.argmax(1) != coarse_clean.argmax(1)).mean()),
                    "exact_match_fraction": 1.0,
                    "matched_fallback_count": 0,
                })
                unit_verify.append({
                    "seed": seed,
                    "model": name,
                    "budget": float(b),
                    "samples": len(Xattack),
                    "signed_histogram_exact": int(n_exact),
                    "fallback": 0,
                })
                del Xm
                gc.collect()

            _save_part(tag, {"rows": unit_rows, "verify": unit_verify})
            rows += unit_rows
            verify += unit_verify
            del model
            gc.collect()

        del frame, coarse
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    rdf = pd.DataFrame(rows)
    vdf = pd.DataFrame(verify)
    rdf.to_csv(OUT / "exact_matched_conv_sew_protocol_v2.csv", index=False)
    vdf.to_csv(
        OUT / "exact_matched_conv_sew_verification_protocol_v2.csv", index=False)
    return rdf, vdf


def main():
    test_manifest = _manifest("test")
    Xattack, yattack, dur_us = collect_attack_arrays(test_manifest)
    print("DVS attack set:", Xattack.shape)
    if len(Xattack) != 264:
        print(
            f"WARNING: expected the landed 264 test streams, found {len(Xattack)}. "
            "The script will use all available streams.")

    # Fail fast on all required checkpoints.
    for seed in SEEDS:
        for name in [
            "frame_resnet18", "conv_snn", "sew_resnet18",
            "event_transformer_v2", "temporal_gru", "coarse_frameformer",
        ]:
            checkpoint_path(name, seed)

    if RUN_ABLATION:
        run_ablation(Xattack, yattack, dur_us)
    if RUN_COST:
        run_cost(Xattack, yattack, dur_us)
    if RUN_EXACT_CONTROLS:
        run_exact_controls_conv_sew(Xattack, yattack, dur_us)

    with open(OUT / "dvs_protocol_cleanup_v2_config.json", "w") as f:
        json.dump({
            "diagnostic_seed": DIAGNOSTIC_SEED,
            "temporal_models": TEMPORAL_MODELS,
            "ablation_path": ABLATION_PATH,
            "cost_path": COST_PATH,
            "cost_budgets": COST_BUDGETS,
            "exact_control_models": EXACT_CONTROL_MODELS,
            "exact_control_seeds": SEEDS,
            "consistency_tolerance_pct_points": CONSISTENCY_TOL_PP,
            "notes": (
                "No retraining. Corrected optimization effort for ablation/cost; "
                "filled strict exact-control gap for ConvSNN/SEW."
            ),
        }, f, indent=2)

    print("\nDONE. Protocol-cleanup outputs are under:", OUT)


if __name__ == "__main__":
    main()

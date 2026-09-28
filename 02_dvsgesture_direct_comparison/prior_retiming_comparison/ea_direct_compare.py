#!/usr/bin/env python3
"""
Direct retiming-method comparison for the event-security paper.

Primary paper experiment (DVS128 Gesture):
  1) Prior retiming: Yu et al., ICLR 2026, official PGDTimeShiftAfterEncoder_L0
  2) Free gradient retiming: same EA unique-event greedy optimizer, unconstrained in time
  3) Null-space retiming: same EA optimizer, constrained inside coarse accumulation windows

The script DOES NOT retrain anything. It reuses the frozen DVS-Gesture tensor cache
and checkpoints produced by EVS_DVSGesture.py.

Default paper run:
  - victim: ConvSNN
  - protected consumer: CoarseFrameFormer (strong coarse-only consumer)
  - seeds: 0,1,2
  - all DVS-Gesture test clips (normally 264)
  - budget: 10% of event units
  - free/null optimization path: 2% -> 5% -> 10%
  - prior method: official method-native 40 PIL steps, with a native B0 point budget chosen to
    approximately match the same event-unit budget. The ACTUAL moved event-unit
    percentage is always reported, so no budget mismatch is hidden.

Important representation note:
The released official L0 implementation defines B0 over moved occupied source
time-space points/cells. On an integer count tensor, one such source cell can carry
multiple event units. Our EA protocol instead budgets individual event units. For a
fair table, this script reports BOTH the prior method's native moved-point count and
the actual number/fraction of event units whose timestamps changed. An optional one-step
budget recalibration is used to bring the latter close to the requested EA budget.

No code from the prior repository is vendored here. The official attack class is
loaded at runtime from a local clone specified by SPIKE_RETIMING_REPO.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import inspect
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Frozen DVS-Gesture representation from the existing EA run
# -----------------------------------------------------------------------------
NUM_CLASSES = 11
MODEL_H = MODEL_W = 64
T_FINE = 160
S_PER_FRAME = 8
N_COARSE = T_FINE // S_PER_FRAME
EVAL_BATCH = 32

DEFAULT_ROOT = Path("/path/to/workspace/EVS_DVSGesture/run")
DEFAULT_PRIOR_REPO = Path("/path/to/workspace/Spike-Retiming-Attacks")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Match the landed/frozen DVS-Gesture experiment runtime.  The official Yu et al.
# reproduction script uses deterministic=True/benchmark=False for its own models,
# but this add-on evaluates all three attacks on OUR frozen checkpoints and should
# therefore keep the frozen EA backend setting for a like-for-like comparison.
torch.backends.cudnn.benchmark = True


# -----------------------------------------------------------------------------
# Models copied from the frozen DVS-Gesture experiment implementation.
# These definitions are needed only to load the already-trained checkpoints.
# -----------------------------------------------------------------------------
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


def coarse_frames_torch(X):
    B = X.shape[0]
    return X.reshape(B, N_COARSE, S_PER_FRAME, 2, MODEL_H, MODEL_W).sum(dim=2)


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
            nn.Conv2d(N_COARSE * 2, 64, 5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.stage1 = self._stage(64, 64, 2, 1)
        self.stage2 = self._stage(64, 128, 2, 2)
        self.stage3 = self._stage(128, 256, 2, 2)
        self.stage4 = self._stage(256, 512, 2, 2)
        self.fc = nn.Linear(512, NUM_CLASSES)

    @staticmethod
    def _stage(in_ch, out_ch, blocks, stride):
        layers = [BasicBlock2D(in_ch, out_ch, stride)]
        for _ in range(1, blocks):
            layers.append(BasicBlock2D(out_ch, out_ch))
        return nn.Sequential(*layers)

    def forward(self, X):
        B = X.shape[0]
        coarse = coarse_frames_torch(X)
        z = coarse.reshape(B, N_COARSE * 2, MODEL_H, MODEL_W)
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
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
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
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
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
        self.s1 = nn.Sequential(SEWBlock(32, 32, 1, beta), SEWBlock(32, 32, 1, beta))
        self.s2 = nn.Sequential(SEWBlock(32, 64, 2, beta), SEWBlock(64, 64, 1, beta))
        self.s3 = nn.Sequential(SEWBlock(64, 128, 2, beta), SEWBlock(128, 128, 1, beta))
        self.s4 = nn.Sequential(SEWBlock(128, 256, 2, beta), SEWBlock(256, 256, 1, beta))
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
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        r = self.skip(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + r)


class CoarseFrameFormer(nn.Module):
    """Strong protected consumer copied from EVS_DVSGesture_addons.py.

    It consumes only the 20 protected coarse accumulation frames. Therefore an
    integer-identical coarse tensor must produce identical logits/predictions.
    """
    def __init__(self, d_model=320, nhead=8, depth=4, dropout=0.10):
        super().__init__()
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 64, 5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(),
            CoarseSpatialBlock(64, 64, 1),
            CoarseSpatialBlock(64, 128, 2),
            CoarseSpatialBlock(128, 192, 2),
            CoarseSpatialBlock(192, 256, 2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(256, d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos = nn.Parameter(torch.zeros(1, N_COARSE + 1, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4 * d_model,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(d_model, NUM_CLASSES),
        )
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
        z = self.encoder(z)
        z = self.norm(z)
        return self.head(torch.cat([z[:, 0], z[:, 1:].mean(dim=1)], dim=1))


BUILDERS = {
    "frame_resnet18": FrameResNet18,
    "coarse_frameformer": CoarseFrameFormer,
    "conv_snn": ConvSNN,
    "sew_resnet18": SEWResNet18,
}


# -----------------------------------------------------------------------------
# EA attack helpers (same representation and move semantics as landed code)
# -----------------------------------------------------------------------------
def grouped_null(X: np.ndarray) -> np.ndarray:
    return (
        X.reshape(N_COARSE, S_PER_FRAME, 2, MODEL_H, MODEL_W)
        .transpose(0, 2, 3, 4, 1)
        .reshape(-1, S_PER_FRAME)
    )


def ungrouped_null(Xg: np.ndarray) -> np.ndarray:
    return (
        Xg.reshape(N_COARSE, 2, MODEL_H, MODEL_W, S_PER_FRAME)
        .transpose(0, 4, 1, 2, 3)
        .reshape(T_FINE, 2, MODEL_H, MODEL_W)
    )


def grouped_free(X: np.ndarray) -> np.ndarray:
    return X.reshape(T_FINE, 2 * MODEL_H * MODEL_W).T.copy()


def ungrouped_free(Xg: np.ndarray) -> np.ndarray:
    return Xg.T.reshape(T_FINE, 2, MODEL_H, MODEL_W)


def coarse_np(X: np.ndarray) -> np.ndarray:
    return X.reshape(N_COARSE, S_PER_FRAME, 2, MODEL_H, MODEL_W).sum(axis=1)


def _select_moves(
    Xg: np.ndarray,
    Gg: np.ndarray,
    Ag: np.ndarray,
    n_moves: int,
    S: int,
    max_shift_bins: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Greedy unique-event move selector used by the EA attacks."""
    g_idx, s_idx = np.nonzero(Ag > 0)
    if len(g_idx) == 0 or n_moves <= 0:
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
        take[cut + 1 :] = 0

    Xg[g_idx, s_idx] -= take
    np.add.at(Xg, (g_idx, dst), take)
    Ag[g_idx, s_idx] -= take
    shifts = np.repeat((dst - s_idx), take).astype(np.int16)
    return Xg, Ag, shifts


def paired_protected_logits(model: nn.Module, X0: np.ndarray, Xa: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate clean/adversarial protected views in one batch.

    Using one batch removes batch-shape-dependent kernel selection from the
    exact-invariance check.  For a null-space retiming the protected consumer
    receives exactly the same coarse tensor for both elements, so its returned
    logits must be bit-identical in eval mode.
    """
    pair = np.stack([X0, Xa], axis=0)
    z = predict_logits(model, pair, batch=2)
    if z.shape[0] != 2:
        raise AssertionError(f"paired protected-consumer evaluation returned {z.shape}")
    return z[0], z[1]


def input_gradient(model: nn.Module, X: np.ndarray, y: int) -> np.ndarray:
    model.eval()
    x = torch.from_numpy(X[None].astype(np.float32)).to(DEVICE)
    x.requires_grad_(True)
    yt = torch.tensor([int(y)], device=DEVICE, dtype=torch.long)
    loss = F.cross_entropy(model(x), yt)
    g = torch.autograd.grad(loss, x, retain_graph=False, create_graph=False)[0][0]
    return g.detach().cpu().numpy()


def optimization_path(target_budget: float) -> List[float]:
    """DVS main-run path up to the requested endpoint."""
    base = [0.02, 0.05, 0.10, 0.20]
    path = [b for b in base if b <= target_budget + 1e-12]
    if not path or abs(path[-1] - target_budget) > 1e-12:
        path.append(float(target_budget))
    return sorted(set(path))


def progressive_ea_attack(
    model: nn.Module,
    X0: np.ndarray,
    y: int,
    target_budget: float,
    null_space: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Progressive unique-event attack with matched gradient refreshes.

    For 10%, both free and null-space arms use 2% -> 5% -> 10%, fixing the
    earlier diagnostic mismatch where one arm received fewer gradient refreshes.
    """
    path = optimization_path(target_budget)
    total = int(X0.sum())
    Xcur = X0.copy().astype(np.int32)
    if null_space:
        available = grouped_null(X0).astype(np.int32)
        group_fn, ungroup_fn, S = grouped_null, ungrouped_null, S_PER_FRAME
    else:
        available = grouped_free(X0).astype(np.int32)
        group_fn, ungroup_fn, S = grouped_free, ungrouped_free, T_FINE

    moved = 0
    all_shifts: List[int] = []
    for b in path:
        target = int(round(float(b) * total))
        need = target - moved
        if need <= 0:
            continue
        grad = input_gradient(model, Xcur.astype(np.int16), y)
        Xg = group_fn(Xcur).astype(np.int32)
        Gg = group_fn(grad).astype(np.float32)
        Xg, available, sh = _select_moves(Xg, Gg, available, need, S, None)
        Xcur = ungroup_fn(Xg).astype(np.int32)
        moved += len(sh)
        all_shifts.extend(sh.tolist())

    Xout = Xcur.astype(np.int16)
    if int(Xout.sum()) != total or Xout.min() < 0:
        raise AssertionError("EA retiming violated event-count conservation")
    if null_space and not np.array_equal(coarse_np(X0), coarse_np(Xout)):
        raise AssertionError("null-space attack violated exact coarse-frame invariance")
    return Xout, np.asarray(all_shifts, dtype=np.int16)


# -----------------------------------------------------------------------------
# Prior attack adapter: load the official implementation, do not vendor it.
# -----------------------------------------------------------------------------
class TimeMajorToBatchMajor(nn.Module):
    """Adapter so the official [T,B,C,H,W] attack can call our [B,T,C,H,W] model."""

    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base

    def forward(self, x_tbchw: torch.Tensor) -> torch.Tensor:
        return self.base(x_tbchw.permute(1, 0, 2, 3, 4).contiguous())


def verify_official_prior_interface(prior_cls) -> Dict[str, object]:
    """Fail fast if the official class interface differs from the adapter contract."""
    init_sig = inspect.signature(prior_cls.__init__)
    forward_sig = inspect.signature(prior_cls.forward)
    required_init = {
        "device", "model_without_encoder", "reduction", "steps", "alpha_phi",
        "lambda_cap", "temperature", "random_start", "cap_limit",
        "l0_moves_budget", "lambda_B", "dual_lr",
    }
    missing_init = sorted(required_init - set(init_sig.parameters))
    if missing_init:
        raise RuntimeError(f"official prior constructor changed; missing {missing_init}: {init_sig}")
    if "return_disp" not in forward_sig.parameters:
        raise RuntimeError(f"official prior forward() lacks return_disp: {forward_sig}")
    if not issubclass(prior_cls, nn.Module):
        raise RuntimeError("official prior class is no longer an nn.Module; .to(device) contract changed")
    return {
        "constructor_signature": str(init_sig),
        "forward_signature": str(forward_sig),
        "required_constructor_fields_verified": True,
        "return_disp_verified": True,
        "nn_module_verified": True,
    }


def load_official_prior_class(repo: Path):
    attack_py = repo / "utils" / "attack.py"
    if not attack_py.exists():
        raise FileNotFoundError(
            f"Official prior attack not found at {attack_py}. Clone "
            "https://github.com/yuyi-sd/Spike-Retiming-Attacks and set SPIKE_RETIMING_REPO."
        )
    spec = importlib.util.spec_from_file_location("yu_spike_retiming_attack", attack_py)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {attack_py}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "PGDTimeShiftAfterEncoder_L0"):
        raise AttributeError("official attack.py lacks PGDTimeShiftAfterEncoder_L0")
    prior_cls = mod.PGDTimeShiftAfterEncoder_L0
    interface = verify_official_prior_interface(prior_cls)
    sha = hashlib.sha256(attack_py.read_bytes()).hexdigest()
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        commit = "unknown"
    return prior_cls, sha, commit, interface


@dataclass
class PriorResult:
    Xadv: np.ndarray
    delta: np.ndarray
    b0_packets: int
    moved_event_units: int
    moved_event_fraction: float
    attempts: int


def verify_prior_delta_reconstruction(
    X0: np.ndarray, Xadv: np.ndarray, delta: np.ndarray
) -> bool:
    """Verify the official returned displacement is source-indexed and exactly rebuilds Xadv.

    The official implementation documents delta[s,...] = t-s for a moved source
    source point/cell. This check makes that semantic assumption executable for every sample.
    """
    if X0.shape != Xadv.shape or delta.shape != X0.shape:
        raise AssertionError(
            f"prior delta shape mismatch: X0={X0.shape}, Xadv={Xadv.shape}, delta={delta.shape}"
        )
    if np.any(delta[X0 == 0] != 0):
        raise AssertionError("prior delta is nonzero at an empty source location")
    t, c, h, w = np.nonzero(X0)
    if len(t) == 0:
        if not np.array_equal(X0.astype(np.int64), Xadv.astype(np.int64)):
            raise AssertionError("empty input changed under prior attack")
        return True
    shift = delta[t, c, h, w].astype(np.int64)
    dst_t = t.astype(np.int64) + shift
    if np.any(dst_t < 0) or np.any(dst_t >= X0.shape[0]):
        raise AssertionError("prior delta points outside the valid time axis")
    rebuilt = np.zeros_like(X0, dtype=np.int64)
    np.add.at(
        rebuilt,
        (dst_t, c.astype(np.int64), h.astype(np.int64), w.astype(np.int64)),
        X0[t, c, h, w].astype(np.int64),
    )
    if not np.array_equal(rebuilt, Xadv.astype(np.int64)):
        diff = int(np.abs(rebuilt - Xadv.astype(np.int64)).sum())
        raise AssertionError(f"prior delta does not reconstruct Xadv exactly (L1 diff={diff})")
    return True


def prior_phi_preflight() -> Dict[str, float]:
    """Static memory diagnostic for the official all-time L0 parameterization."""
    elements = int(T_FINE * 1 * 2 * MODEL_H * MODEL_W * T_FINE)
    gib_fp32 = elements * 4 / (1024 ** 3)
    return {
        "phi_elements": elements,
        "phi_single_tensor_fp32_gib": float(gib_fp32),
        "our_T": int(T_FINE),
        "official_example_T": 10,
        "temporal_quadratic_ratio_vs_T10": float((T_FINE / 10) ** 2),
    }


def _run_prior_once(
    prior_cls,
    model: nn.Module,
    X0: np.ndarray,
    y: int,
    b0_packets: int,
    steps: int,
) -> PriorResult:
    wrapped = TimeMajorToBatchMajor(model)
    atk = prior_cls(
        device=DEVICE,
        model_without_encoder=wrapped,
        reduction="mean",
        steps=int(steps),
        alpha_phi=1.0,
        lambda_cap=20.0,
        temperature=1.0,
        random_start=False,
        cap_limit=1.0,
        l0_moves_budget=int(b0_packets),
        lambda_B=0.0,
        dual_lr=0.1,
    ).to(DEVICE)

    x_tm = torch.from_numpy(X0.astype(np.float32)).unsqueeze(1).to(DEVICE)  # [T,1,C,H,W]
    yt = torch.tensor([int(y)], device=DEVICE, dtype=torch.long)
    adv_tm, delta_tm = atk(x_tm, yt, return_disp=True)
    adv_f = adv_tm[:, 0].detach().float().cpu().numpy()
    delta = delta_tm[:, 0].detach().cpu().numpy().astype(np.int16)

    # Final strict projection should retain integer source-cell values exactly.
    if not np.allclose(adv_f, np.rint(adv_f), atol=1e-5):
        raise AssertionError("official prior attack returned non-integer final projected tensor")
    Xadv = np.rint(adv_f).astype(np.int16)
    if Xadv.min() < 0 or int(Xadv.sum()) != int(X0.sum()):
        raise AssertionError("official prior attack violated event-count conservation")
    verify_prior_delta_reconstruction(X0, Xadv, delta)
    moved_point_count = int(np.count_nonzero(delta))
    if moved_point_count > int(b0_packets):
        raise AssertionError(
            f"official prior L0 projection moved {moved_point_count} native points > requested {b0_packets}"
        )

    moved_mask = delta != 0
    moved_units = int(X0[moved_mask].astype(np.int64).sum())
    frac = moved_units / max(1, int(X0.sum()))
    return PriorResult(Xadv, delta, int(b0_packets), moved_units, float(frac), 1)


def prior_attack_budget_matched(
    prior_cls,
    model: nn.Module,
    X0: np.ndarray,
    y: int,
    target_budget: float,
    steps: int,
    recalibrate: bool,
    tolerance: float,
) -> PriorResult:
    """Run the official native-point B0 attack while matching EA event-unit budget as closely as possible.

    The first native-point budget uses mean event-unit multiplicity per occupied source cell. If requested, one deterministic
    rescaling retry is allowed. The closer result is kept. Actual event-unit movement is
    reported regardless of whether tolerance is reached.
    """
    total_units = int(X0.sum())
    active_packets = int((X0 > 0).sum())
    if total_units <= 0 or active_packets <= 0:
        return PriorResult(X0.copy(), np.zeros_like(X0, dtype=np.int16), 0, 0, 0.0, 0)
    target_units = int(round(float(target_budget) * total_units))
    mean_packet = total_units / active_packets
    b0 = int(np.clip(round(target_units / max(mean_packet, 1e-12)), 1, active_packets))

    r1 = _run_prior_once(prior_cls, model, X0, y, b0, steps)
    r1.attempts = 1
    if (not recalibrate) or abs(r1.moved_event_fraction - target_budget) <= tolerance:
        return r1

    if r1.moved_event_units <= 0:
        b02 = int(np.clip(b0 * 2, 1, active_packets))
    else:
        b02 = int(np.clip(round(b0 * target_units / r1.moved_event_units), 1, active_packets))
    if b02 == b0:
        return r1

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    r2 = _run_prior_once(prior_cls, model, X0, y, b02, steps)
    r2.attempts = 2
    if abs(r2.moved_event_fraction - target_budget) < abs(r1.moved_event_fraction - target_budget):
        return r2
    r1.attempts = 2
    return r1


# -----------------------------------------------------------------------------
# Metrics and data loading
# -----------------------------------------------------------------------------
def autocast_context():
    # Match the frozen DVS-Gesture evaluation path exactly on CUDA.
    if DEVICE.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return torch.autocast(device_type="cpu", enabled=False)


@torch.no_grad()
def predict_logits(model: nn.Module, X: np.ndarray, batch: int = EVAL_BATCH) -> np.ndarray:
    model.eval()
    outs = []
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(X[i : i + batch].astype(np.float32)).to(DEVICE)
        with autocast_context():
            z = model(xb)
        outs.append(z.detach().float().cpu().numpy())
    return np.concatenate(outs, axis=0)


def load_model(root: Path, name: str, seed: int) -> nn.Module:
    if name not in BUILDERS:
        raise ValueError(f"unsupported model {name}; choose {sorted(BUILDERS)}")
    ckpt = root / "checkpoints" / f"{name}_seed{seed}.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"missing checkpoint: {ckpt}")
    model = BUILDERS[name]().to(DEVICE)
    state = torch.load(ckpt, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model


def load_attack_set(root: Path, limit: Optional[int] = None):
    manifest_fp = root / "cache_tensors" / "test_tensor_manifest.csv"
    if not manifest_fp.exists():
        raise FileNotFoundError(f"missing frozen test tensor manifest: {manifest_fp}")
    m = pd.read_csv(manifest_fp)
    if limit is not None and limit > 0:
        m = m.iloc[:limit].copy()
    Xs, ys, ds, paths = [], [], [], []
    for _, r in m.iterrows():
        d = np.load(r["path"])
        Xs.append(d["X"].astype(np.int16))
        ys.append(int(d["label"]))
        ds.append(int(d["duration_us"]))
        paths.append(str(r["path"]))
    return np.stack(Xs), np.asarray(ys, np.int64), np.asarray(ds, np.int64), paths


def frozen_clean_accuracy(root: Path, model_name: str, seed: int) -> Optional[float]:
    """Read the landed clean-accuracy record for a checkpoint when available."""
    candidates = [
        root / "results" / "clean_accuracy_by_seed.csv",
        root / "results" / "addon_clean_accuracy_by_seed.csv",
    ]
    for fp in candidates:
        if not fp.exists():
            continue
        df = pd.read_csv(fp)
        if not {"model", "seed", "clean_test_accuracy"}.issubset(df.columns):
            continue
        z = df[(df["model"].astype(str) == str(model_name)) &
               (df["seed"].astype(int) == int(seed))]
        if len(z) == 1:
            return float(z.iloc[0]["clean_test_accuracy"])
    return None


def verify_clean_accuracy_against_frozen(
    root: Path, model_name: str, seed: int, observed: float, *, tol: float = 1e-12
) -> Optional[float]:
    expected = frozen_clean_accuracy(root, model_name, seed)
    if expected is None:
        print(f"WARNING: no landed clean-accuracy row found for {model_name} seed={seed}; semantic guard skipped")
        return None
    if abs(float(observed) - float(expected)) > tol:
        raise AssertionError(
            f"checkpoint semantic guard failed for {model_name} seed={seed}: "
            f"recomputed={observed:.12f}, landed={expected:.12f}"
        )
    print(f"clean-accuracy guard PASS: {model_name} seed={seed} = {observed:.6f}")
    return expected


def coarse_metrics(X0: np.ndarray, Xa: np.ndarray) -> Dict[str, float]:
    a = coarse_np(X0).astype(np.int64)
    b = coarse_np(Xa).astype(np.int64)
    d = np.abs(a - b)
    return {
        "frame_exact_equal": bool(np.array_equal(a, b)),
        "max_integer_frame_difference": int(d.max(initial=0)),
        "frame_l1_difference": int(d.sum()),
    }


def shifts_from_prior(X0: np.ndarray, delta: np.ndarray) -> Tuple[int, float]:
    mask = delta != 0
    weights = X0[mask].astype(np.float64)
    if weights.size == 0 or weights.sum() <= 0:
        return 0, 0.0
    vals = np.abs(delta[mask].astype(np.float64))
    moved = int(weights.sum())
    return moved, float(np.sum(weights * vals) / weights.sum())


def sample_row(
    *,
    seed: int,
    victim: str,
    sample_index: int,
    sample_path: str,
    label: int,
    clean_pred: int,
    clean_consumer_pred: int,
    protected_consumer: str,
    method: str,
    budget: float,
    X0: np.ndarray,
    Xa: np.ndarray,
    pred_adv: int,
    consumer_pred_adv: int,
    clean_consumer_logits_pair: Optional[np.ndarray] = None,
    consumer_logits_adv_pair: Optional[np.ndarray] = None,
    duration_us: int = 0,
    shifts: Optional[np.ndarray] = None,
    prior_delta: Optional[np.ndarray] = None,
    prior_b0_packets: Optional[int] = None,
    prior_attempts: Optional[int] = None,
    seconds: float = 0.0,
) -> Dict[str, object]:
    total_units = int(X0.sum())
    if prior_delta is not None:
        moved_units, mean_abs_bins = shifts_from_prior(X0, prior_delta)
        moved_packets = int((prior_delta != 0).sum())
    else:
        sh = np.asarray([] if shifts is None else shifts, dtype=np.int64)
        moved_units = int(len(sh))
        mean_abs_bins = float(np.abs(sh).mean()) if len(sh) else 0.0
        moved_packets = np.nan
    bin_ms = float(duration_us) / T_FINE / 1000.0
    cm = coarse_metrics(X0, Xa)
    if clean_consumer_logits_pair is not None and consumer_logits_adv_pair is not None:
        z0 = np.asarray(clean_consumer_logits_pair)
        za = np.asarray(consumer_logits_adv_pair)
        logits_bit_identical = bool(np.array_equal(z0, za))
        max_logit_diff = float(np.max(np.abs(z0.astype(np.float64) - za.astype(np.float64)), initial=0.0))
    else:
        logits_bit_identical = False
        max_logit_diff = float("nan")
    return {
        "seed": int(seed),
        "victim": victim,
        "sample_index": int(sample_index),
        "sample_path": sample_path,
        "label": int(label),
        "clean_pred": int(clean_pred),
        "clean_correct": bool(clean_pred == label),
        "method": method,
        "requested_event_budget": float(budget),
        "total_event_units": total_units,
        "unique_event_units_changed": moved_units,
        "unique_event_units_changed_pct": moved_units / max(1, total_units),
        "budget_match_abs_error": abs((moved_units / max(1, total_units)) - float(budget)),
        "prior_b0_points_changed": moved_packets,
        "prior_b0_points_requested": prior_b0_packets if prior_b0_packets is not None else np.nan,
        # Backward-compatible aliases used by v2/v3 downstream notebooks.
        "prior_b0_packets_changed": moved_packets,
        "prior_b0_packets_requested": prior_b0_packets if prior_b0_packets is not None else np.nan,
        "prior_budget_attempts": prior_attempts if prior_attempts is not None else np.nan,
        "avg_abs_shift_bins": mean_abs_bins,
        "avg_abs_shift_ms": mean_abs_bins * bin_ms,
        "victim_pred_adv": int(pred_adv),
        "attack_success_on_clean_correct": bool((clean_pred == label) and (pred_adv != label)),
        "protected_consumer": protected_consumer,
        "clean_protected_consumer_pred": int(clean_consumer_pred),
        "protected_consumer_pred_adv": int(consumer_pred_adv),
        "protected_consumer_flip": bool(consumer_pred_adv != clean_consumer_pred),
        "protected_consumer_logits_bit_identical": logits_bit_identical,
        "protected_consumer_max_abs_logit_difference": max_logit_diff,
        **cm,
        "seconds": float(seconds),
    }


def aggregate_outputs(rows: pd.DataFrame, out: Path) -> None:
    rows.to_csv(out / "direct_retiming_rows.csv", index=False)

    method_order = ["prior_pil_l0", "free_gradient", "null_space"]
    display_names = {
        "prior_pil_l0": "Yu et al. PIL-L0 retiming",
        "free_gradient": "Free gradient retiming",
        "null_space": "Null-space retiming",
    }

    # Denominator for ASR is clean-correct samples, matching the main EA paper protocol.
    summaries = []
    keys = ["seed", "victim", "method", "requested_event_budget"]
    for key, g in rows.groupby(keys, dropna=False, sort=False):
        clean = g[g["clean_correct"]]
        summaries.append({
            "seed": key[0],
            "victim": key[1],
            "method": key[2],
            "budget": key[3],
            "protected_consumer": str(g["protected_consumer"].iloc[0]),
            "n": len(g),
            "n_clean_correct": len(clean),
            "asr_clean_correct": float(clean["attack_success_on_clean_correct"].mean()) if len(clean) else np.nan,
            "unique_event_units_changed_pct": float(g["unique_event_units_changed_pct"].mean()),
            "unique_event_units_changed": float(g["unique_event_units_changed"].mean()),
            "budget_match_abs_error_mean": float(g["budget_match_abs_error"].mean()),
            "budget_match_abs_error_max": float(g["budget_match_abs_error"].max()),
            "avg_abs_shift_ms": float(g["avg_abs_shift_ms"].mean()),
            "max_integer_frame_difference": int(g["max_integer_frame_difference"].max()),
            "mean_frame_l1_difference": float(g["frame_l1_difference"].mean()),
            "protected_consumer_flip_rate": float(g["protected_consumer_flip"].mean()),
            "protected_consumer_logits_bit_identical_rate": float(g["protected_consumer_logits_bit_identical"].mean()),
            "protected_consumer_max_abs_logit_difference": float(g["protected_consumer_max_abs_logit_difference"].max()),
            "runtime_seconds_per_sample": float(g["seconds"].mean()),
            "prior_b0_points_requested_mean": float(g["prior_b0_points_requested"].dropna().mean()) if g["prior_b0_points_requested"].notna().any() else np.nan,
            "prior_b0_points_changed_mean": float(g["prior_b0_points_changed"].dropna().mean()) if g["prior_b0_points_changed"].notna().any() else np.nan,
            "prior_b0_packets_requested_mean": float(g["prior_b0_packets_requested"].dropna().mean()) if g["prior_b0_packets_requested"].notna().any() else np.nan,
            "prior_b0_packets_changed_mean": float(g["prior_b0_packets_changed"].dropna().mean()) if g["prior_b0_packets_changed"].notna().any() else np.nan,
        })
    by_seed = pd.DataFrame(summaries)
    by_seed.to_csv(out / "direct_retiming_by_seed.csv", index=False)

    agg_rows = []
    for key, g in by_seed.groupby(["victim", "method", "budget", "protected_consumer"], dropna=False, sort=False):
        def ms(col, pct=False):
            v = g[col].astype(float)
            mean = float(v.mean())
            std = float(v.std(ddof=1)) if len(v) > 1 else 0.0
            if pct:
                mean *= 100.0; std *= 100.0
            return mean, std

        asr_m, asr_s = ms("asr_clean_correct", pct=True)
        ev_m, ev_s = ms("unique_event_units_changed_pct", pct=True)
        sh_m, sh_s = ms("avg_abs_shift_ms")
        pf_m, pf_s = ms("protected_consumer_flip_rate", pct=True)
        maxdiff = int(g["max_integer_frame_difference"].max())
        agg_rows.append({
            "victim": key[0],
            "method": key[1],
            "budget": float(key[2]),
            "protected_consumer": key[3],
            "ASR_mean_pct": asr_m,
            "ASR_std_pct": asr_s,
            "Unique_events_changed_mean_pct": ev_m,
            "Unique_events_changed_std_pct": ev_s,
            "Avg_shift_mean_ms": sh_m,
            "Avg_shift_std_ms": sh_s,
            "Frame_difference_max_integer": maxdiff,
            "Protected_consumer_flips_mean_pct": pf_m,
            "Protected_consumer_flips_std_pct": pf_s,
            "ASR": f"{asr_m:.2f} ± {asr_s:.2f}",
            "Event units retimed": f"{ev_m:.2f} ± {ev_s:.2f}%",
            "Avg. shift": f"{sh_m:.3f} ± {sh_s:.3f} ms",
            "Max coarse-frame diff": str(maxdiff),
            "Protected frame-consumer flips": f"{pf_m:.2f} ± {pf_s:.2f}%",
        })
    agg = pd.DataFrame(agg_rows)
    if len(agg):
        order_map = {m: i for i, m in enumerate(method_order)}
        agg["_method_order"] = agg["method"].map(order_map).fillna(99).astype(int)
        agg = agg.sort_values(["victim", "budget", "_method_order"], kind="stable").drop(columns="_method_order")
    agg.to_csv(out / "direct_retiming_table_numeric.csv", index=False)

    paper = pd.DataFrame({
        "victim": agg["victim"] if len(agg) else pd.Series(dtype=str),
        "budget": agg["budget"].map(lambda x: f"{100*x:g}%") if len(agg) else pd.Series(dtype=str),
        "Method": agg["method"].map(display_names) if len(agg) else pd.Series(dtype=str),
        "ASR": agg["ASR"] if len(agg) else pd.Series(dtype=str),
        "Event units retimed": agg["Event units retimed"] if len(agg) else pd.Series(dtype=str),
        "Avg. shift": agg["Avg. shift"] if len(agg) else pd.Series(dtype=str),
        "Max coarse-frame diff": agg["Max coarse-frame diff"] if len(agg) else pd.Series(dtype=str),
        "Protected frame-consumer flips": agg["Protected frame-consumer flips"] if len(agg) else pd.Series(dtype=str),
    })
    paper.to_csv(out / "direct_retiming_table.csv", index=False)
    try:
        tex_paper = paper.copy()
        # escape=False preserves ±, so explicitly protect LaTeX comment characters.
        for col in tex_paper.columns:
            tex_paper[col] = tex_paper[col].astype(str).str.replace("%", r"\%", regex=False)
        tex = tex_paper.to_latex(index=False, escape=False)
        (out / "direct_retiming_table.tex").write_text(tex)
    except Exception as e:
        (out / "direct_retiming_table.tex.error.txt").write_text(repr(e))

    print("\nPAPER TABLE")
    print(paper.to_string(index=False))


# -----------------------------------------------------------------------------
# Main experiment
# -----------------------------------------------------------------------------
def parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_str_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def atomic_json(path: Path, obj: Dict[str, object]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=True))
    os.replace(tmp, path)


def protocol_fingerprint(payload: Dict[str, object], n: int = 16) -> str:
    """Stable cache namespace for one exact attack protocol.

    This prevents the mandatory 2-sample/2-step smoke from contaminating the
    later 40-step paper run, and also separates caches when the official source,
    consumer, budget-matching rule, or backend settings change.
    """
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:n]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path(os.environ.get("DVS_GESTURE_RUN", DEFAULT_ROOT)))
    p.add_argument("--prior-repo", type=Path, default=Path(os.environ.get("SPIKE_RETIMING_REPO", DEFAULT_PRIOR_REPO)))
    p.add_argument("--seeds", default=os.environ.get("RUN_SEEDS", "0,1,2"))
    p.add_argument("--victims", default=os.environ.get("VICTIMS", "conv_snn"))
    p.add_argument("--budgets", default=os.environ.get("DIRECT_BUDGETS", "0.10"))
    p.add_argument("--limit", type=int, default=int(os.environ.get("DIRECT_N", "0")), help="0 = all test clips")
    p.add_argument("--expect-n", type=int, default=int(os.environ.get("EXPECT_N", "264")))
    p.add_argument("--prior-steps", type=int, default=int(os.environ.get("PRIOR_STEPS", "40")))
    p.add_argument("--prior-recalibrate", type=int, choices=[0,1], default=int(os.environ.get("PRIOR_RECALIBRATE", "1")))
    p.add_argument("--prior-budget-tol", type=float, default=float(os.environ.get("PRIOR_BUDGET_TOL", "0.005")))
    p.add_argument("--methods", default=os.environ.get("METHODS", "prior_pil_l0,free_gradient,null_space"))
    p.add_argument(
        "--protected-consumer",
        default=os.environ.get("PROTECTED_CONSUMER", "coarse_frameformer"),
        choices=["coarse_frameformer", "frame_resnet18"],
    )
    p.add_argument(
        "--verify-clean-accuracy", type=int, choices=[0, 1],
        default=int(os.environ.get("VERIFY_CLEAN_ACCURACY", "1")),
        help="For a full-set run, compare recomputed clean accuracy to landed CSV records.",
    )
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    root = args.root
    out = root / "results" / "direct_prior_retiming_comparison"
    out.mkdir(parents=True, exist_ok=True)

    seeds = parse_int_list(args.seeds)
    victims = parse_str_list(args.victims)
    budgets = parse_float_list(args.budgets)
    methods = parse_str_list(args.methods)
    valid_methods = {"prior_pil_l0", "free_gradient", "null_space"}
    bad = set(methods) - valid_methods
    if bad:
        raise ValueError(f"unknown methods: {sorted(bad)}")
    for v in victims:
        if v not in {"conv_snn", "sew_resnet18"}:
            raise ValueError("direct prior comparison is intentionally restricted to SNN victims: conv_snn,sew_resnet18")

    prior_cls = None
    prior_sha = prior_commit = None
    prior_interface = None
    if "prior_pil_l0" in methods:
        prior_cls, prior_sha, prior_commit, prior_interface = load_official_prior_class(args.prior_repo)

    X, y, duration_us, paths = load_attack_set(root, None if args.limit <= 0 else args.limit)
    if args.limit <= 0 and args.expect_n > 0 and len(X) != args.expect_n:
        raise RuntimeError(
            f"Expected {args.expect_n} frozen DVS-Gesture test clips, found {len(X)}. "
            "Set EXPECT_N to the verified count if your frozen run intentionally differs."
        )
    print(f"device={DEVICE} samples={len(X)} seeds={seeds} victims={victims} budgets={budgets}")
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    preflight = prior_phi_preflight()
    if "prior_pil_l0" in methods:
        print(
            "PRIOR MEMORY PREFLIGHT: official all-time phi at T=160 has "
            f"{preflight['phi_elements']:,} elements = "
            f"{preflight['phi_single_tensor_fp32_gib']:.3f} GiB for phi alone. "
            "Official DVS-Gesture example scripts use T=10; run the 2-sample smoke before any paper queue."
        )

    manifest_fp = root / "cache_tensors" / "test_tensor_manifest.csv"
    manifest_sha = hashlib.sha256(manifest_fp.read_bytes()).hexdigest()
    cache_protocol = {
        "protocol_version": "v5",
        "root": str(root.resolve()),
        "manifest_sha256": manifest_sha,
        "limit": int(args.limit),
        "expect_n": int(args.expect_n),
        "seeds": seeds,
        "victims": victims,
        "budgets": budgets,
        "methods": methods,
        "protected_consumer": args.protected_consumer,
        "prior_repo_commit": prior_commit,
        "prior_attack_py_sha256": prior_sha,
        "prior_steps": int(args.prior_steps),
        "prior_recalibrate": bool(args.prior_recalibrate),
        "prior_budget_tol": float(args.prior_budget_tol),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
    }
    cache_id = protocol_fingerprint(cache_protocol)
    partial = out / "partial" / cache_id
    partial.mkdir(parents=True, exist_ok=True)
    print(f"partial-cache protocol id={cache_id} path={partial}")

    run_cfg = {
        "dataset": "IBM DVS128 Gesture",
        "root": str(root),
        "samples": len(X),
        "manifest_sha256": manifest_sha,
        "partial_cache_protocol_id": cache_id,
        "partial_cache_protocol": cache_protocol,
        "seeds": seeds,
        "victims": victims,
        "protected_consumer": args.protected_consumer,
        "budgets_event_units": budgets,
        "free_null_optimization_path_rule": "[0.02,0.05,0.10,0.20] prefixes ending at target",
        "methods": methods,
        "prior_method": "Yu et al. ICLR 2026 PGDTimeShiftAfterEncoder_L0 official implementation",
        "prior_repo": str(args.prior_repo),
        "prior_repo_commit": prior_commit,
        "prior_attack_py_sha256": prior_sha,
        "prior_interface_runtime_verification": prior_interface,
        "prior_memory_preflight": preflight,
        "prior_steps": args.prior_steps,
        "prior_recalibrate_to_event_units": bool(args.prior_recalibrate),
        "prior_budget_tolerance_abs_fraction": args.prior_budget_tol,
        "torch_cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "torch_cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "protected_invariant_evaluation": (
            "Clean/adversarial protected-consumer logits are evaluated as a paired batch of size 2; "
            "null-space rows require bit-identical logits, zero coarse-tensor difference, and zero prediction flip."
        ),
        "representation_note": (
            "The released official L0 code counts moved occupied source time-space points/cells; "
            "on our integer tensor a cell may contain multiple event units. EA budgets count event units. "
            "Actual event units moved are reported for all methods; native prior B0 point budget is retained separately."
        ),
        "threat_model_note": (
            "Yu et al. additionally enforce their native capacity/non-overlap feasibility on occupied source points. "
            "The free/null EA controls use the frozen integer event-unit retiming protocol. The direct comparison "
            "therefore matches realized event-unit mass and victim/data, but does not claim identical feasible sets."
        ),
    }
    atomic_json(out / "direct_retiming_run_config.json", run_cfg)

    all_rows = []
    for seed in seeds:
        protected_model = load_model(root, args.protected_consumer, seed)
        clean_consumer_logits = predict_logits(protected_model, X)
        clean_consumer_pred = clean_consumer_logits.argmax(1)
        if args.limit <= 0 and bool(args.verify_clean_accuracy):
            consumer_acc = float((clean_consumer_pred == y).mean())
            verify_clean_accuracy_against_frozen(
                root, args.protected_consumer, seed, consumer_acc
            )

        for victim_name in victims:
            victim = load_model(root, victim_name, seed)
            clean_pred = predict_logits(victim, X).argmax(1)
            clean_acc = float((clean_pred == y).mean())
            print(f"\nseed={seed} victim={victim_name} clean_acc={clean_acc:.4f}")
            if args.limit <= 0 and bool(args.verify_clean_accuracy):
                verify_clean_accuracy_against_frozen(root, victim_name, seed, clean_acc)

            for budget in budgets:
                for i in range(len(X)):
                    X0 = X[i]
                    yi = int(y[i])
                    for method in methods:
                        tag = (
                            f"v5_pc-{args.protected_consumer}_s{seed}_{victim_name}_"
                            f"b{budget:.6f}_i{i}_{method}"
                        )
                        fp = partial / f"{tag}.json"
                        if fp.exists() and not args.force:
                            row = json.loads(fp.read_text())
                            all_rows.append(row)
                            continue

                        t0 = time.time()
                        if method == "null_space":
                            Xa, shifts = progressive_ea_attack(victim, X0, yi, budget, null_space=True)
                            pdelta = None; pb0 = None; patt = None
                        elif method == "free_gradient":
                            Xa, shifts = progressive_ea_attack(victim, X0, yi, budget, null_space=False)
                            pdelta = None; pb0 = None; patt = None
                        else:
                            assert prior_cls is not None
                            pr = prior_attack_budget_matched(
                                prior_cls, victim, X0, yi, budget,
                                steps=args.prior_steps,
                                recalibrate=bool(args.prior_recalibrate),
                                tolerance=args.prior_budget_tol,
                            )
                            Xa = pr.Xadv
                            shifts = None
                            pdelta = pr.delta
                            pb0 = pr.b0_packets
                            patt = pr.attempts
                            err = abs(pr.moved_event_fraction - float(budget))
                            if err > args.prior_budget_tol:
                                print(
                                    f"WARNING prior budget match outside tolerance: seed={seed} "
                                    f"victim={victim_name} sample={i} target={budget:.4f} "
                                    f"realized={pr.moved_event_fraction:.4f} abs_err={err:.4f}; "
                                    "the realized fraction will be reported explicitly."
                                )

                        pred_adv = int(predict_logits(victim, Xa[None], batch=1).argmax(1)[0])

                        # Evaluate the protected clean/adversarial pair in one call.
                        # This avoids comparing batch-32 clean logits with batch-1 adversarial
                        # logits under potentially different cuDNN kernels.
                        consumer_z0, consumer_za = paired_protected_logits(protected_model, X0, Xa)
                        consumer_pred_clean_pair = int(np.argmax(consumer_z0))
                        consumer_pred_adv = int(np.argmax(consumer_za))

                        row = sample_row(
                            seed=seed,
                            victim=victim_name,
                            sample_index=i,
                            sample_path=paths[i],
                            label=yi,
                            clean_pred=int(clean_pred[i]),
                            clean_consumer_pred=consumer_pred_clean_pair,
                            protected_consumer=args.protected_consumer,
                            method=method,
                            budget=budget,
                            X0=X0,
                            Xa=Xa,
                            pred_adv=pred_adv,
                            consumer_pred_adv=consumer_pred_adv,
                            clean_consumer_logits_pair=consumer_z0,
                            consumer_logits_adv_pair=consumer_za,
                            duration_us=int(duration_us[i]),
                            shifts=shifts,
                            prior_delta=pdelta,
                            prior_b0_packets=pb0,
                            prior_attempts=patt,
                            seconds=time.time() - t0,
                        )

                        # Strong invariant: null-space row must be exactly invisible to the protected consumer.
                        if method == "null_space":
                            if (
                                row["max_integer_frame_difference"] != 0
                                or row["protected_consumer_flip"]
                                or not row["protected_consumer_logits_bit_identical"]
                                or row["protected_consumer_max_abs_logit_difference"] != 0.0
                            ):
                                raise AssertionError(
                                    f"null-space invariant failed seed={seed} victim={victim_name} sample={i}: "
                                    f"frame_delta={row['max_integer_frame_difference']} "
                                    f"flip={row['protected_consumer_flip']} "
                                    f"logits_identical={row['protected_consumer_logits_bit_identical']} "
                                    f"max_logit_delta={row['protected_consumer_max_abs_logit_difference']}"
                                )

                        atomic_json(fp, row)
                        all_rows.append(row)
                        if (i + 1) % 10 == 0 or i == 0:
                            print(
                                f" seed={seed} {victim_name} b={budget:.2f} i={i+1}/{len(X)} "
                                f"{method}: success={int(row['attack_success_on_clean_correct'])} "
                                f"moved={100*row['unique_event_units_changed_pct']:.2f}% "
                                f"shift={row['avg_abs_shift_ms']:.3f}ms "
                                f"frameΔ={row['max_integer_frame_difference']} "
                                f"consumerflip={int(row['protected_consumer_flip'])} "
                                f"logitEq={int(row['protected_consumer_logits_bit_identical'])}"
                            )

                        del Xa
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

            del victim
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        del protected_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    rows = pd.DataFrame(all_rows)
    aggregate_outputs(rows, out)
    print("\nDONE:", out)


if __name__ == "__main__":
    main()

"""Mechanism-focused ablations for the observer-relative event-retiming paper.

This module deliberately leaves the frozen dataset/model code untouched.  It only
changes the temporal observation operator / legal move set used by an ablation.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Iterable, Sequence
import numpy as np
import torch
import torch.nn.functional as F

@dataclass(frozen=True)
class Observer:
    width: int
    phase: int = 0
    def ids(self, T: int) -> np.ndarray:
        # Integer partition with boundaries shifted by phase. Negative ids for the
        # leading partial window are intentional and harmless.
        t = np.arange(T, dtype=np.int64)
        return np.floor_divide(t - int(self.phase), int(self.width))


def operator_matrix(T: int, observer: Observer) -> np.ndarray:
    ids = observer.ids(T)
    vals = np.unique(ids)
    A = np.zeros((len(vals), T), dtype=np.float64)
    for r, wid in enumerate(vals):
        A[r, ids == wid] = 1.0
    return A


def stacked_nullity_time(T: int, observers: Sequence[Observer]) -> int:
    A = np.concatenate([operator_matrix(T, o) for o in observers], axis=0)
    rank = int(np.linalg.matrix_rank(A))
    return int(T - rank)


def full_nullity(T: int, C: int, H: int, W: int, observers: Sequence[Observer]) -> int:
    return int(stacked_nullity_time(T, observers) * C * H * W)


def aggregate_operator(X: np.ndarray, observer: Observer) -> np.ndarray:
    """Aggregate [T,C,H,W] under an arbitrary shifted fixed-width observer."""
    T = X.shape[0]
    ids = observer.ids(T)
    vals = np.unique(ids)
    out = np.zeros((len(vals),) + tuple(X.shape[1:]), dtype=np.int64)
    Xi = X.astype(np.int64, copy=False)
    for r, wid in enumerate(vals):
        out[r] = Xi[ids == wid].sum(axis=0)
    return out


def observer_metrics(X0: np.ndarray, Xa: np.ndarray, observer: Observer) -> dict:
    a = aggregate_operator(X0, observer)
    b = aggregate_operator(Xa, observer)
    d = np.abs(b - a)
    total = max(1, int(X0.sum()))
    return {
        "operator_width": int(observer.width),
        "operator_phase": int(observer.phase),
        "operator_l1": int(d.sum()),
        "D_A_operator": float(d.sum() / total),
        "D_inf_operator": int(d.max(initial=0)),
        "operator_exact_equal": bool(np.array_equal(a, b)),
    }


def equivalence_mask(T: int, observers: Sequence[Observer], max_shift_bins=None) -> np.ndarray:
    """allowed[s,d] iff source and destination are in the same window for every observer."""
    allowed = np.ones((T, T), dtype=bool)
    for o in observers:
        ids = o.ids(T)
        allowed &= ids[:, None] == ids[None, :]
    if max_shift_bins is not None:
        t = np.arange(T)
        allowed &= np.abs(t[:, None] - t[None, :]) <= int(max_shift_bins)
    np.fill_diagonal(allowed, False)
    return allowed


def _has_rnn(model):
    return any(isinstance(m, torch.nn.RNNBase) for m in model.modules())


def input_gradient(model, X, y, device):
    x = torch.from_numpy(X[None].astype(np.float32)).to(device).requires_grad_(True)
    yt = torch.tensor([int(y)], device=device, dtype=torch.long)
    if _has_rnn(model):
        with torch.backends.cudnn.flags(enabled=False):
            g = torch.autograd.grad(F.cross_entropy(model(x), yt), x)[0][0]
    else:
        g = torch.autograd.grad(F.cross_entropy(model(x), yt), x)[0][0]
    return g.detach().cpu().numpy()


def _select_general(Xcur, grad, available, n_moves, allowed):
    """Move original event units only; moved units cannot move again."""
    T, C, H, W = Xcur.shape
    Xg = Xcur.reshape(T, -1).T.astype(np.int32)
    Gg = grad.reshape(T, -1).T.astype(np.float32)
    Ag = available
    cell, src = np.nonzero(Ag > 0)
    if len(cell) == 0 or n_moves <= 0:
        return Xcur.copy(), Ag, np.empty(0, dtype=np.int16)
    gm = np.where(allowed[src], Gg[cell], -np.inf)
    dst = gm.argmax(axis=1)
    gains = gm[np.arange(len(dst)), dst] - Gg[cell, src]
    keep = np.isfinite(gains) & (gains > 0)
    cell, src, dst, gains = cell[keep], src[keep], dst[keep], gains[keep]
    if len(cell) == 0:
        return Xcur.copy(), Ag, np.empty(0, dtype=np.int16)
    order = np.argsort(-gains, kind="stable")
    cell, src, dst = cell[order], src[order], dst[order]
    caps = Ag[cell, src].astype(np.int64)
    csum = np.cumsum(caps)
    take = caps.copy()
    cut = int(np.searchsorted(csum, int(n_moves)))
    if cut < len(take):
        prev = csum[cut - 1] if cut > 0 else 0
        take[cut] = max(0, int(n_moves) - int(prev))
        take[cut + 1:] = 0
    Xg[cell, src] -= take
    np.add.at(Xg, (cell, dst), take)
    Ag[cell, src] -= take
    Xa = Xg.T.reshape(T, C, H, W).astype(np.int16)
    shifts = np.repeat(dst - src, take).astype(np.int16)
    assert Xa.min() >= 0 and int(Xa.sum()) == int(Xcur.sum())
    return Xa, Ag, shifts


# Gradient-refresh schedule used by the frozen main experiment. Every ablation
# except `refresh` must use it, otherwise its ASR is not comparable with the
# main table (single-refresh vs progressive path differ substantially).
MAIN_REFRESH_PATH = [0.02, 0.05, 0.10]


def main_protocol_refresh(final_budget):
    pts = [b for b in MAIN_REFRESH_PATH if b < float(final_budget)]
    return pts + [float(final_budget)]


def constrained_attack(model, X0, y, budgets, observers, device, max_shift_bins=None,
                       refresh_points=None):
    """Gradient retiming constrained to the joint null space of `observers`.

    budgets are cumulative fractions of original event units. refresh_points can
    override the gradient-refresh checkpoints while preserving the same final budget.
    """
    budgets = sorted(float(b) for b in budgets)
    if not budgets:
        return {}
    final_b = budgets[-1]
    schedule = sorted(set(float(x) for x in (refresh_points or budgets) if float(x) <= final_b))
    if not schedule or schedule[-1] != final_b:
        schedule.append(final_b)
    T = X0.shape[0]
    allowed = equivalence_mask(T, observers, max_shift_bins=max_shift_bins)
    total = int(X0.sum())
    x = X0.astype(np.int16).copy()
    available = X0.reshape(T, -1).T.astype(np.int32)
    moved = 0
    all_shifts = []
    snapshots = {}
    for b in schedule:
        need = int(round(b * total)) - moved
        if need > 0:
            g = input_gradient(model, x, y, device)
            x, available, sh = _select_general(x, g, available, need, allowed)
            moved += len(sh)
            all_shifts.extend(sh.tolist())
        snapshots[b] = {"X": x.copy(), "shifts": np.asarray(all_shifts, np.int16), "moved": int(moved)}
        for obs in observers:
            assert observer_metrics(X0, x, obs)["operator_exact_equal"]
    # Return requested budgets. If a requested budget was not a refresh point,
    # run it as its own progressive checkpoint so no interpolation is invented.
    if set(budgets).issubset(snapshots):
        return {b: snapshots[b] for b in budgets}
    # conservative fallback: rerun with requested budgets as refresh points
    return constrained_attack(model, X0, y, budgets, observers, device,
                              max_shift_bins=max_shift_bins, refresh_points=budgets)


def one_shot_attack(model, X0, y, final_budget, observers, device, max_shift_bins=None):
    """Exactly one gradient computation at the clean input."""
    T = X0.shape[0]
    allowed = equivalence_mask(T, observers, max_shift_bins=max_shift_bins)
    total = int(X0.sum())
    g = input_gradient(model, X0, y, device)
    available = X0.reshape(T, -1).T.astype(np.int32)
    Xa, _, sh = _select_general(X0.astype(np.int16), g, available,
                                int(round(float(final_budget) * total)), allowed)
    for obs in observers:
        assert observer_metrics(X0, Xa, obs)["operator_exact_equal"]
    return {float(final_budget): {"X": Xa, "shifts": sh, "moved": int(len(sh))}}

# ---------------------------------------------------------------------------
# Post-hoc observer-family visibility diagnostics.
# These do NOT claim a trained classifier for every observer family.  They
# measure representation visibility of the exact same perturbation, which is
# the quantity needed by the paper's consumer-relative stealth claim.
# ---------------------------------------------------------------------------
def _repr_delta_metrics(X0: np.ndarray, Xa: np.ndarray, R0: np.ndarray, Ra: np.ndarray,
                        family: str, detail: str = "") -> dict:
    d = np.abs(Ra.astype(np.float64) - R0.astype(np.float64))
    total = max(1.0, float(np.asarray(X0, dtype=np.float64).sum()))
    return {
        "observer_family": family,
        "observer_detail": detail,
        "family_l1": float(d.sum()),
        "D_A_family": float(d.sum() / total),
        "D_inf_family": float(d.max(initial=0.0)),
        "family_exact_equal": bool(np.array_equal(R0, Ra)),
    }


def aggregate_overlapping(X: np.ndarray, width: int, stride: int) -> np.ndarray:
    T = int(X.shape[0]); width = int(width); stride = max(1, int(stride))
    starts = list(range(0, T, stride))
    out = []
    Xi = X.astype(np.int64, copy=False)
    for s in starts:
        e = min(T, s + width)
        if e <= s:
            continue
        out.append(Xi[s:e].sum(axis=0))
        if e == T:
            break
    return np.stack(out, axis=0) if out else np.zeros((0,) + X.shape[1:], dtype=np.int64)


def ema_final(X: np.ndarray, alpha: float) -> np.ndarray:
    alpha = float(alpha)
    y = np.zeros(X.shape[1:], dtype=np.float64)
    for t in range(X.shape[0]):
        y = (1.0 - alpha) * y + alpha * X[t].astype(np.float64)
    return y


def observer_family_metrics(X0: np.ndarray, Xa: np.ndarray, canonical_width: int) -> list[dict]:
    """Visibility of one perturbation to several temporal representations.

    This is intentionally representation-level.  Only the canonical boxcar has
    the trained protected consumer used in the main experiment.  The other
    families answer whether the SAME perturbation remains invisible when the
    temporal observer changes.
    """
    S = int(canonical_width)
    T = int(X0.shape[0])
    out = []

    # Canonical fixed boxcar (the observer for which the attack was constructed).
    o0 = Observer(S, 0)
    out.append(_repr_delta_metrics(X0, Xa, aggregate_operator(X0, o0),
                                   aggregate_operator(Xa, o0), "boxcar", f"width={S},phase=0"))

    # Same width, shifted boundaries.
    ph = max(1, S // 2)
    os = Observer(S, ph)
    out.append(_repr_delta_metrics(X0, Xa, aggregate_operator(X0, os),
                                   aggregate_operator(Xa, os), "shifted_boxcar",
                                   f"width={S},phase={ph}"))

    # Overlapping windows: same nominal width, 50% stride.
    stride = max(1, S // 2)
    out.append(_repr_delta_metrics(X0, Xa, aggregate_overlapping(X0, S, stride),
                                   aggregate_overlapping(Xa, S, stride), "overlap_boxcar",
                                   f"width={S},stride={stride}"))

    # Multi-scale concatenation.  Widths are clipped and de-duplicated.
    widths = sorted({max(2, S // 2), S, min(T, 2 * S)})
    r0 = np.concatenate([aggregate_operator(X0, Observer(w, 0)).reshape(-1)
                         for w in widths])
    ra = np.concatenate([aggregate_operator(Xa, Observer(w, 0)).reshape(-1)
                         for w in widths])
    out.append(_repr_delta_metrics(X0, Xa, r0, ra, "multiscale_boxcar",
                                   "widths=" + "+".join(map(str, widths))))

    # Leaky/EMA summaries approximate a consumer with non-boxcar temporal response.
    for a in (0.2, 0.5):
        out.append(_repr_delta_metrics(X0, Xa, ema_final(X0, a), ema_final(Xa, a),
                                       "ema_final", f"alpha={a:g}"))
    return out

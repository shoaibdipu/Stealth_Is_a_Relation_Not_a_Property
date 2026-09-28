#!/usr/bin/env python3
"""
Event-Security Study — N-MNIST cost-of-stealth diagnostic cleanup FINAL.

Purpose
-------
Correct ONLY the landed N-MNIST cost-of-stealth optimization mismatch.
The original stealth arm used 4 gradient rounds while free retiming used 2.
This rerun gives BOTH arms exactly 4 rounds at the same 20% unique-event budget.

No training and no headline attack table are rerun. The original 1,000-sample
balanced attack subset is reconstructed deterministically; the cost diagnostic
uses the same first 100 samples as the landed implementation by default.

Normal launch
-------------
  python3 EVS_NMNIST_COST_CLEANUP_FINAL_PATCHED.py

Environment
-----------
  NMNIST_RUN   existing completed N-MNIST run root
  COST_N       default 100

Optional progressive experiments remain in the file but are OFF by default.
"""

import os, gc, json, math, time, random, shutil
from pathlib import Path
from collections import Counter, defaultdict, deque

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
    "NMNIST_RUN", "/path/to/workspace/EVS_NMNIST/run"))
DATA_ROOT = ROOT / "data"
CKPT_ROOT = ROOT / "checkpoints"
OUT = ROOT / "results"
for p in [CKPT_ROOT, OUT]:
    p.mkdir(parents=True, exist_ok=True)

TRAIN_DIR = DATA_ROOT / "Train_extracted"
TEST_DIR = DATA_ROOT / "Test_extracted"
if not TEST_DIR.exists():
    raise FileNotFoundError(
        f"{TEST_DIR} missing; this cleanup script expects the landed N-MNIST run.")

SEEDS = [0, 1, 2]
DT_US = 2_000
S = 10
DELTA_US = DT_US * S
DURATION_US = 300_000
T = DURATION_US // DT_US
NW = T // S
H = W = 34
C = 2

# Aliases so the later progressive attack uses the same naming as DVS/CIFAR.
T_FINE = T
S_PER_FRAME = S
N_COARSE = NW
MODEL_H = H
MODEL_W = W

TAU_SEC = 0.004
BETA = math.exp(-(DT_US / 1e6) / TAU_SEC)
LIF_HIDDEN = 256
GRU_INPUT = 128
GRU_HIDDEN = 128

ATTACK_BUDGETS = [0.05, 0.10, 0.20, 0.30]
ITER_ROUNDS = 4
MAX_SHIFT_BINS = None
ATTACK_PER_CLASS = int(os.environ.get("ATTACK_PER_CLASS", "100"))
if ATTACK_PER_CLASS != 100:
    raise ValueError(
        "paper protocol requires ATTACK_PER_CLASS=100; use a separate debug copy "
        "rather than changing the canonical cleanup run")
COST_N = int(os.environ.get("COST_N", "100"))
LEGACY_COST_BUDGET = 0.20
LEGACY_ROUNDS = 4
PROGRESSIVE_COST_PATH = [0.05, 0.10, 0.20]
PROGRESSIVE_COST_BUDGETS = [0.10, 0.20]

RUN_EQUAL_ROUND_COST = os.environ.get("RUN_EQUAL_ROUND_COST", "1") == "1"
RUN_PROGRESSIVE_MAIN = os.environ.get("RUN_PROGRESSIVE_MAIN", "0") == "1"
RUN_PROGRESSIVE_COST = os.environ.get("RUN_PROGRESSIVE_COST", "0") == "1"

EVAL_BATCH = 128
TEMPORAL_MODELS = ["lif_mlp", "temporal_gru"]

EVENT_DTYPE = np.dtype([
    ("x", np.int16),
    ("y", np.int16),
    ("t", np.int64),
    ("p", np.int8),
])


def read_nmnist_bin(path):
    raw = np.fromfile(path, dtype=np.uint8).astype(np.uint32)
    if len(raw) % 5 != 0:
        raise ValueError(f"Invalid byte count in {path}: {len(raw)}")

    x = raw[0::5]
    y = raw[1::5]
    p = (raw[2::5] & 128) >> 7
    t = (((raw[2::5] & 127) << 16) | (raw[3::5] << 8) | raw[4::5]).astype(np.int64)

    overflow = (y == 240)
    t = t + np.cumsum(overflow, dtype=np.int64) * (2**13)
    valid = ~overflow

    ev = np.empty(int(valid.sum()), dtype=EVENT_DTYPE)
    ev["x"] = x[valid]
    ev["y"] = y[valid]
    ev["t"] = t[valid]
    ev["p"] = p[valid]
    return ev


def discover_samples(root):
    samples = []
    for fp in Path(root).rglob("*.bin"):
        label = None
        for parent in fp.parents:
            if parent.name.isdigit():
                v = int(parent.name)
                if 0 <= v <= 9:
                    label = v
                    break
        if label is not None:
            samples.append((fp, label))
    samples.sort(key=lambda z: (z[1], str(z[0])))
    return samples


def stratified_select(samples, per_class, seed):
    rng = np.random.default_rng(seed)
    by_class = defaultdict(list)
    for path, label in samples:
        by_class[int(label)].append((path, int(label)))
    selected = []
    for cls in range(10):
        pool = by_class[cls]
        if per_class > len(pool):
            raise ValueError(f"class {cls}: requested {per_class}, available {len(pool)}")
        idx = rng.choice(len(pool), size=per_class, replace=False)
        selected.extend([pool[int(j)] for j in idx])
    rng.shuffle(selected)
    return selected


def events_to_tensor(ev, dt_us=DT_US, duration_us=DURATION_US, normalize_time=True):
    ev = ev.copy()
    if normalize_time and len(ev):
        ev["t"] -= int(ev["t"].min())

    keep = (
        (ev["t"] >= 0) & (ev["t"] < duration_us)
        & (ev["x"] >= 0) & (ev["x"] < W)
        & (ev["y"] >= 0) & (ev["y"] < H)
        & ((ev["p"] == 0) | (ev["p"] == 1))
    )
    ev = ev[keep]

    tb = (ev["t"].astype(np.int64) // dt_us).astype(np.int64)
    x = ev["x"].astype(np.int64)
    y = ev["y"].astype(np.int64)
    p = ev["p"].astype(np.int64)

    X = np.zeros((T, C, H, W), dtype=np.int16)
    np.add.at(X, (tb, p, y, x), 1)
    return X


class SurrGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u):
        ctx.save_for_backward(u)
        return (u > 0).float()
    @staticmethod
    def backward(ctx, grad_output):
        (u,) = ctx.saved_tensors
        return grad_output / (10.0 * u.abs() + 1.0)**2

spike = SurrGrad.apply


class FrameCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(NW*C, 32, 3, padding=1)
        self.c2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc = nn.Linear(64*8*8, 10)
    def forward(self, X):
        B = X.shape[0]
        coarse = X.reshape(B, NW, S, C, H, W).sum(dim=2)
        z = coarse.reshape(B, NW*C, H, W)
        z = F.max_pool2d(F.relu(self.c1(z)), 2)
        z = F.max_pool2d(F.relu(self.c2(z)), 2)
        return self.fc(z.flatten(1))


class LIFMLP(nn.Module):
    def __init__(self, hidden=LIF_HIDDEN, beta=BETA, threshold=1.0):
        super().__init__()
        self.fc1 = nn.Linear(C*H*W, hidden)
        self.fc2 = nn.Linear(hidden, 10)
        self.beta = beta
        self.threshold = threshold
    def forward(self, X):
        B, TT = X.shape[:2]
        X = X.flatten(2) * 0.25
        u1 = torch.zeros(B, self.fc1.out_features, device=X.device)
        u2 = torch.zeros(B, 10, device=X.device)
        readout = torch.zeros(B, 10, device=X.device)
        for t in range(TT):
            u1 = self.beta*u1 + self.fc1(X[:,t])
            s1 = spike(u1 - self.threshold)
            u1 = u1 - s1*self.threshold
            u2 = self.beta*u2 + self.fc2(s1)
            readout = readout + u2
        return readout / TT


class TemporalGRU(nn.Module):
    def __init__(self, input_dim=GRU_INPUT, hidden=GRU_HIDDEN):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(C*H*W, input_dim),
            nn.ReLU(),
        )
        self.gru = nn.GRU(input_dim, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 10)
    def forward(self, X):
        z = self.proj(X.flatten(2) * 0.25)
        _, h = self.gru(z)
        return self.fc(h[-1])


def predict_logits(model, X_np, batch=EVAL_BATCH):
    model.eval()
    outs = []
    for i in range(0, len(X_np), batch):
        xb = torch.from_numpy(X_np[i:i+batch].astype(np.float32)).to(DEVICE)
        outs.append(model(xb).detach().cpu().numpy())
    return np.concatenate(outs, axis=0)


def batch_frame_metrics(Xclean, Xatt):
    Fc = Xclean.reshape(len(Xclean),NW,S,C,H,W).sum(axis=2)
    Fa = Xatt.reshape(len(Xatt),NW,S,C,H,W).sum(axis=2)
    return {
        "frame_exact_equal": bool(np.array_equal(Fc, Fa)),
        "max_integer_frame_difference": int(
            np.abs(Fa.astype(np.int64)-Fc.astype(np.int64)).max()
        )
    }


def to_grouped(X):
    return X.reshape(NW,S,C,H,W).transpose(0,2,3,4,1).reshape(-1,S)


def from_grouped(Xg):
    return Xg.reshape(NW,C,H,W,S).transpose(0,4,1,2,3).reshape(T,C,H,W)


def apply_best_unique_moves(Xcur, grad, available, n_moves, max_shift_bins=MAX_SHIFT_BINS):
    Xg = to_grouped(Xcur).astype(np.int32)
    Gg = to_grouped(grad).astype(np.float32)
    Ag = available

    groups, srcs = np.nonzero(Ag > 0)
    if len(groups) == 0:
        return Xcur.copy(), Ag, 0, 0.0, 0

    dsts = np.empty(len(groups), dtype=np.int16)
    gains = np.full(len(groups), -np.inf, dtype=np.float32)

    for i, (g, src) in enumerate(zip(groups, srcs)):
        if max_shift_bins is None:
            lo, hi = 0, S
        else:
            lo = max(0, int(src)-max_shift_bins)
            hi = min(S, int(src)+max_shift_bins+1)

        local = Gg[g, lo:hi].copy()
        local[int(src)-lo] = -np.inf
        d = lo + int(np.argmax(local))
        dsts[i] = d
        gains[i] = Gg[g,d] - Gg[g,src]

    valid = np.isfinite(gains) & (gains > 0)
    groups, srcs, dsts, gains = groups[valid], srcs[valid], dsts[valid], gains[valid]

    if len(groups) == 0:
        return Xcur.copy(), Ag, 0, 0.0, 0

    order = np.argsort(-gains)
    moved = 0
    disp_sum = 0
    max_disp = 0

    for j in order:
        if moved >= n_moves:
            break
        g, src, dst = int(groups[j]), int(srcs[j]), int(dsts[j])
        cap = int(Ag[g,src])
        if cap <= 0:
            continue
        n = min(cap, n_moves-moved)
        Xg[g,src] -= n
        Xg[g,dst] += n
        Ag[g,src] -= n  # moved originals are permanently retired
        disp = abs(dst-src)
        disp_sum += n*disp
        max_disp = max(max_disp, disp)
        moved += n

    Xnext = from_grouped(Xg).astype(np.int16)
    assert Xnext.min() >= 0
    assert coarse_equal(Xcur, Xnext)

    return Xnext, Ag, moved, (disp_sum/moved if moved else 0.0), max_disp


def input_gradient(model, X_np, y):
    x = torch.from_numpy(X_np[None].astype(np.float32)).to(DEVICE)
    x.requires_grad_(True)
    yt = torch.tensor([int(y)], dtype=torch.long, device=DEVICE)
    # cuDNN refuses RNN backward on a module in eval mode, so the cuDNN path is
    # disabled for this call only. Numerically identical; the native kernel is
    # used instead. Everything else, including model.eval(), is unchanged.
    with torch.backends.cudnn.flags(enabled=False):
        loss = F.cross_entropy(model(x), yt)
        g = torch.autograd.grad(loss, x)[0][0]
    return g.detach().cpu().numpy()


def unique_iterative_attack_one(model, X0, y, budget_frac, rounds=ITER_ROUNDS):
    Xcur = X0.copy().astype(np.int16)
    total_events = int(X0.sum())
    target_budget = int(round(budget_frac * total_events))

    available = to_grouped(X0).astype(np.int32)
    moved_total = 0
    disp_weighted = 0.0
    max_disp_total = 0

    for rr in range(rounds):
        remaining = target_budget - moved_total
        if remaining <= 0:
            break
        rounds_left = rounds - rr
        this_target = int(math.ceil(remaining / rounds_left))

        g = input_gradient(model, Xcur, y)
        Xnext, available, moved, mean_disp, max_disp = apply_best_unique_moves(
            Xcur, g, available, this_target, MAX_SHIFT_BINS
        )
        if moved == 0:
            break

        moved_total += moved
        disp_weighted += moved * mean_disp
        max_disp_total = max(max_disp_total, max_disp)
        Xcur = Xnext

    assert int(Xcur.sum()) == total_events
    assert coarse_equal(X0, Xcur)

    return {
        "X": Xcur,
        "moved_unique": int(moved_total),
        "requested_unique": int(target_budget),
        "realized_unique_fraction": float(moved_total/max(1,total_events)),
        "mean_abs_shift_bins": float(disp_weighted/moved_total if moved_total else 0.0),
        "max_abs_shift_bins": int(max_disp_total),
    }


def free_grouped(X):
    return X.transpose(1,2,3,0).reshape(-1,T)


def free_from_grouped(Xg):
    return Xg.reshape(C,H,W,T).transpose(3,0,1,2)


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


def build_model(name):
    if name == "frame_cnn":
        return FrameCNN()
    if name == "lif_mlp":
        return LIFMLP()
    if name == "temporal_gru":
        return TemporalGRU()
    raise ValueError(name)


def checkpoint_path(name, seed):
    fp = CKPT_ROOT / f"{name}_seed{seed}.pt"
    if not fp.exists():
        raise FileNotFoundError(fp)
    return fp


def load_model(name, seed):
    m = build_model(name).to(DEVICE)
    m.load_state_dict(torch.load(checkpoint_path(name, seed), map_location=DEVICE))
    m.eval()
    return m


def load_attack_set():
    test_samples_all = discover_samples(TEST_DIR)
    attack_samples = stratified_select(
        test_samples_all, per_class=ATTACK_PER_CLASS, seed=2027)
    XX, yy = [], []
    for i, (path, label) in enumerate(attack_samples):
        XX.append(events_to_tensor(read_nmnist_bin(path)))
        yy.append(label)
        if (i + 1) % 200 == 0:
            print("  loaded attack streams:", i + 1, "/", len(attack_samples))
    X = np.stack(XX).astype(np.int16)
    y = np.asarray(yy, np.int64)
    print("attack set:", X.shape,
          "| per class:", np.bincount(y, minlength=10))
    return attack_samples, X, y


def safe_asr(pred, y, clean_correct):
    n = int(clean_correct.sum())
    if n == 0:
        return float("nan"), 0
    return float((pred[clean_correct] != y[clean_correct]).mean()), n


def uniform_control_from_count(X0, n_moves, rng):
    unit_g, unit_s = event_units_from_original(X0)
    n_moves = min(int(n_moves), len(unit_g))
    if n_moves <= 0:
        return X0.copy().astype(np.int16), np.empty(0, dtype=np.int16)
    chosen = rng.permutation(len(unit_g))[:n_moves]
    Xg = grouped(X0).astype(np.int32)
    gs = unit_g[chosen]
    ss = unit_s[chosen].astype(np.int64)
    ds = (ss + rng.integers(1, S_PER_FRAME, size=n_moves)) % S_PER_FRAME
    np.subtract.at(Xg, (gs, ss), 1)
    np.add.at(Xg, (gs, ds), 1)
    Xr = ungrouped(Xg).astype(np.int16)
    assert Xr.min() >= 0
    assert int(Xr.sum()) == int(X0.sum())
    assert coarse_equal(X0, Xr)
    return Xr, (ds - ss).astype(np.int16)


def best_free_unique_moves_progressive(Xcur, grad, available, n_moves):
    Xg = free_grouped(Xcur).astype(np.int32)
    Gg = free_grouped(grad).astype(np.float32)
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
    Xnext = free_from_grouped(Xg).astype(np.int16)
    assert Xnext.min() >= 0 and int(Xnext.sum()) == int(Xcur.sum())
    return Xnext, Ag, shifts


def progressive_free_unique_attack(model, X0, y, budgets):
    budgets = sorted(float(b) for b in budgets)
    total = int(X0.sum())
    Xcur = X0.copy().astype(np.int16)
    available = free_grouped(X0).astype(np.int32)
    moved, shifts_all = 0, []
    ck = {}
    for b in budgets:
        need = int(round(b * total)) - moved
        if need > 0:
            grad = input_gradient(model, Xcur, y)
            Xcur, available, sh = best_free_unique_moves_progressive(
                Xcur, grad, available, need)
            moved += len(sh)
            shifts_all.extend(sh.tolist())
        ck[b] = {
            "X": Xcur.copy(),
            "moved": int(moved),
            "shifts": np.asarray(shifts_all, dtype=np.int16).copy(),
        }
    return ck


def free_unique_attack_rounds(model, X0, y, budget_frac, rounds=4):
    """Original N-MNIST free attack generalized from hard-coded 2 to `rounds`."""
    Xcur = X0.copy().astype(np.int16)
    total = int(X0.sum())
    target = int(round(budget_frac * total))
    available = free_grouped(X0).astype(np.int32)
    moved = 0

    for rr in range(rounds):
        remaining = target - moved
        if remaining <= 0:
            break
        grad = input_gradient(model, Xcur, y)
        Xg = free_grouped(Xcur).astype(np.int32)
        Gg = free_grouped(grad).astype(np.float32)
        groups, srcs = np.nonzero(available > 0)
        best_dst = Gg.argmax(axis=1)
        gains = Gg[groups, best_dst[groups]] - Gg[groups, srcs]
        valid = (best_dst[groups] != srcs) & (gains > 0)
        groups, srcs, gains = groups[valid], srcs[valid], gains[valid]
        order = np.argsort(-gains)

        this_target = int(math.ceil(remaining / (rounds - rr)))
        this_moved = 0
        for j in order:
            if this_moved >= this_target:
                break
            gg, src = int(groups[j]), int(srcs[j])
            dst = int(best_dst[gg])
            cap = int(available[gg, src])
            if cap <= 0:
                continue
            n = min(cap, this_target - this_moved)
            Xg[gg, src] -= n
            Xg[gg, dst] += n
            available[gg, src] -= n
            this_moved += n

        Xcur = free_from_grouped(Xg).astype(np.int16)
        moved += this_moved
        if this_moved == 0:
            break
    assert int(Xcur.sum()) == total
    return Xcur, moved


PARTIAL = OUT / "nmnist_protocol_cleanup_v2_partial"
PARTIAL.mkdir(parents=True, exist_ok=True)
TMP = OUT / "nmnist_protocol_cleanup_v2_tmp"
TMP.mkdir(parents=True, exist_ok=True)


def _part(tag):
    return PARTIAL / f"{tag}.json"


def _load_part(tag):
    fp = _part(tag)
    if not fp.exists():
        return None
    try:
        with open(fp) as f:
            return json.load(f)
    except Exception:
        return None


def _save_part(tag, obj):
    fp = _part(tag)
    tmp = fp.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, fp)


def run_equal_round_cost(Xattack, yattack):
    rows = []
    n = min(COST_N, len(Xattack))
    Xs, ys = Xattack[:n], yattack[:n]

    for name in TEMPORAL_MODELS:
        tag = f"equal_round_cost_{name}"
        cached = _load_part(tag)
        if cached is not None:
            rows += cached["rows"]
            continue

        model = load_model(name, 0)
        clean_pred = predict_logits(model, Xs).argmax(1)
        clean_correct = clean_pred == ys
        unit = []

        for kind in ["stealth", "free"]:
            AA = []
            t0 = time.time()
            for i in range(n):
                if kind == "stealth":
                    r = unique_iterative_attack_one(
                        model, Xs[i], ys[i],
                        LEGACY_COST_BUDGET, rounds=LEGACY_ROUNDS)
                    xa = r["X"]
                else:
                    xa, _ = free_unique_attack_rounds(
                        model, Xs[i], ys[i],
                        LEGACY_COST_BUDGET, rounds=LEGACY_ROUNDS)
                AA.append(xa)
            AA = np.stack(AA)
            pred = predict_logits(model, AA).argmax(1)
            asr, ncc = safe_asr(pred, ys, clean_correct)
            unit.append({
                "seed": 0,
                "model": name,
                "kind": kind,
                "budget": LEGACY_COST_BUDGET,
                "rounds": LEGACY_ROUNDS,
                "optimization": "equal_rounds_4_vs_4",
                "n": n,
                "n_clean_correct": ncc,
                "clean_accuracy": float(clean_correct.mean()),
                "attacked_accuracy": float((pred == ys).mean()),
                "asr_clean_correct": asr,
                "prediction_flip_rate": float((pred != clean_pred).mean()),
                **batch_frame_metrics(Xs, AA),
                "seconds": float(time.time() - t0),
            })
            del AA
            gc.collect()

        _save_part(tag, {"rows": unit})
        rows += unit
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "cost_of_stealth_4v4_cleanup_final.csv", index=False)
    return df


def run_progressive_main(Xattack, yattack):
    """Uniform progressive N-MNIST rerun with strict exact controls."""
    main_rows, class_rows, verify_rows = [], [], []

    for seed in SEEDS:
        frame = load_model("frame_cnn", seed)
        clean_frame_logits = predict_logits(frame, Xattack)
        clean_frame_pred = clean_frame_logits.argmax(1)

        for name in TEMPORAL_MODELS:
            tag = f"progressive_seed{seed}_{name}"
            cached = _load_part(tag)
            if cached is not None:
                main_rows += cached["main_rows"]
                class_rows += cached["class_rows"]
                verify_rows += cached["verify_rows"]
                continue

            model = load_model(name, seed)
            clean_pred = predict_logits(model, Xattack).argmax(1)
            clean_correct = clean_pred == yattack
            clean_acc = float(clean_correct.mean())

            # Store progressive adversarial tensors on disk to avoid >2.5 GB RAM.
            adv_maps = {}
            adv_paths = {}
            shifts_by_b = {b: [] for b in ATTACK_BUDGETS}
            for b in ATTACK_BUDGETS:
                fp = TMP / f"adv_seed{seed}_{name}_b{int(100*b):02d}.npy"
                adv_paths[b] = fp
                adv_maps[b] = np.lib.format.open_memmap(
                    fp, mode="w+", dtype=np.int16, shape=Xattack.shape)

            t0 = time.time()
            for i in range(len(Xattack)):
                ck = progressive_unique_attack(
                    model, Xattack[i], yattack[i], ATTACK_BUDGETS)
                for b in ATTACK_BUDGETS:
                    adv_maps[b][i] = ck[b]["X"]
                    shifts_by_b[b].append(ck[b]["shifts"].astype(np.int16))
                if (i + 1) % 100 == 0:
                    print(
                        f"  progressive {name} seed={seed}: "
                        f"{i+1}/{len(Xattack)} ({time.time()-t0:.0f}s)")
            for m in adv_maps.values():
                m.flush()
            del adv_maps

            unit_main, unit_class, unit_verify = [], [], []
            for b in ATTACK_BUDGETS:
                Xadv = np.load(adv_paths[b], mmap_mode="r")
                adv_pred = predict_logits(model, Xadv).argmax(1)
                asr, ncc = safe_asr(adv_pred, yattack, clean_correct)
                fl = predict_logits(frame, Xadv)
                shifts = shifts_by_b[b]
                unit_main.append({
                    "seed": seed,
                    "model": name,
                    "attack": "gradient_unique_progressive",
                    "budget": float(b),
                    "optimization_path": "0.05->0.10->0.20->0.30",
                    "clean_accuracy": clean_acc,
                    "n": len(Xattack),
                    "n_clean_correct": ncc,
                    "attacked_accuracy": float((adv_pred == yattack).mean()),
                    "asr_clean_correct": asr,
                    "prediction_flip_rate": float((adv_pred != clean_pred).mean()),
                    "mean_realized_unique_fraction": float(np.mean([
                        len(s) / max(1, int(Xattack[i].sum()))
                        for i, s in enumerate(shifts)])),
                    "mean_abs_shift_ms": float(np.mean([
                        float(np.abs(s).mean()) * DT_US / 1000.0 if len(s) else 0.0
                        for s in shifts])),
                    **batch_frame_metrics(Xattack, Xadv),
                    "frame_cnn_logits_exact": bool(
                        np.array_equal(fl, clean_frame_logits)),
                    "frame_cnn_flip_rate": float(
                        (fl.argmax(1) != clean_frame_pred).mean()),
                })
                for cls in range(10):
                    m = yattack == cls
                    cm = m & clean_correct
                    unit_class.append({
                        "seed": seed,
                        "model": name,
                        "budget": float(b),
                        "class": cls,
                        "n": int(m.sum()),
                        "n_clean_correct": int(cm.sum()),
                        "clean_accuracy": float(
                            (clean_pred[m] == yattack[m]).mean()),
                        "attacked_accuracy": float(
                            (adv_pred[m] == yattack[m]).mean()),
                        "asr_clean_correct": float(
                            (adv_pred[cm] != yattack[cm]).mean())
                            if cm.sum() else float("nan"),
                    })

                # Uniform random control: same number of moved original events.
                ufp = TMP / f"uniform_seed{seed}_{name}_b{int(100*b):02d}.npy"
                Um = np.lib.format.open_memmap(
                    ufp, mode="w+", dtype=np.int16, shape=Xattack.shape)
                rng = np.random.default_rng(
                    510000 + seed * 1000 + int(b * 10000)
                    + (0 if name == "lif_mlp" else 100000))
                ushifts = []
                for i in range(len(Xattack)):
                    xr, sh = uniform_control_from_count(
                        Xattack[i], len(shifts[i]), rng)
                    Um[i] = xr
                    ushifts.append(sh)
                Um.flush()
                Xu = np.load(ufp, mmap_mode="r")
                upred = predict_logits(model, Xu).argmax(1)
                uasr, _ = safe_asr(upred, yattack, clean_correct)
                ufl = predict_logits(frame, Xu)
                unit_main.append({
                    "seed": seed,
                    "model": name,
                    "attack": "random_uniform",
                    "budget": float(b),
                    "optimization_path": "matched_moved_event_count",
                    "clean_accuracy": clean_acc,
                    "n": len(Xattack),
                    "n_clean_correct": ncc,
                    "attacked_accuracy": float((upred == yattack).mean()),
                    "asr_clean_correct": uasr,
                    "prediction_flip_rate": float((upred != clean_pred).mean()),
                    "mean_realized_unique_fraction": float(np.mean([
                        len(s) / max(1, int(Xattack[i].sum()))
                        for i, s in enumerate(ushifts)])),
                    "mean_abs_shift_ms": float(np.mean([
                        float(np.abs(s).mean()) * DT_US / 1000.0 if len(s) else 0.0
                        for s in ushifts])),
                    **batch_frame_metrics(Xattack, Xu),
                    "frame_cnn_logits_exact": bool(
                        np.array_equal(ufl, clean_frame_logits)),
                    "frame_cnn_flip_rate": float(
                        (ufl.argmax(1) != clean_frame_pred).mean()),
                })
                del Um, Xu
                try:
                    ufp.unlink()
                except OSError:
                    pass

                # Strict exact signed-displacement control.
                efp = TMP / f"exact_seed{seed}_{name}_b{int(100*b):02d}.npy"
                Em = np.lib.format.open_memmap(
                    efp, mode="w+", dtype=np.int16, shape=Xattack.shape)
                rng = np.random.default_rng(
                    520000 + seed * 1000 + int(b * 10000)
                    + (0 if name == "lif_mlp" else 100000))
                eshifts = []
                n_exact = 0
                for i in range(len(Xattack)):
                    xe, sh, ok = exact_displacement_matched_random_attack(
                        Xattack[i], shifts[i], rng)
                    if not ok:
                        raise RuntimeError("strict exact matcher returned false")
                    target = np.asarray(shifts[i], dtype=np.int16)
                    target = target[target != 0]
                    if Counter(sh.tolist()) != Counter(target.astype(np.int8).tolist()):
                        raise RuntimeError(
                            f"exact histogram mismatch seed={seed} {name} b={b} i={i}")
                    n_exact += 1
                    Em[i] = xe
                    eshifts.append(sh)
                Em.flush()
                Xe = np.load(efp, mmap_mode="r")
                epred = predict_logits(model, Xe).argmax(1)
                easr, _ = safe_asr(epred, yattack, clean_correct)
                efl = predict_logits(frame, Xe)
                unit_main.append({
                    "seed": seed,
                    "model": name,
                    "attack": "random_displacement_matched_exact",
                    "budget": float(b),
                    "optimization_path": "exact_signed_shift_histogram",
                    "clean_accuracy": clean_acc,
                    "n": len(Xattack),
                    "n_clean_correct": ncc,
                    "attacked_accuracy": float((epred == yattack).mean()),
                    "asr_clean_correct": easr,
                    "prediction_flip_rate": float((epred != clean_pred).mean()),
                    "mean_realized_unique_fraction": float(np.mean([
                        len(s) / max(1, int(Xattack[i].sum()))
                        for i, s in enumerate(eshifts)])),
                    "mean_abs_shift_ms": float(np.mean([
                        float(np.abs(s).mean()) * DT_US / 1000.0 if len(s) else 0.0
                        for s in eshifts])),
                    **batch_frame_metrics(Xattack, Xe),
                    "frame_cnn_logits_exact": bool(
                        np.array_equal(efl, clean_frame_logits)),
                    "frame_cnn_flip_rate": float(
                        (efl.argmax(1) != clean_frame_pred).mean()),
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
                del Em, Xe
                try:
                    efp.unlink()
                except OSError:
                    pass

            _save_part(tag, {
                "main_rows": unit_main,
                "class_rows": unit_class,
                "verify_rows": unit_verify,
            })
            main_rows += unit_main
            class_rows += unit_class
            verify_rows += unit_verify

            for fp in adv_paths.values():
                try:
                    fp.unlink()
                except OSError:
                    pass
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        del frame
        gc.collect()

    mdf = pd.DataFrame(main_rows)
    cdf = pd.DataFrame(class_rows)
    vdf = pd.DataFrame(verify_rows)
    mdf.to_csv(OUT / "main_attack_progressive_v2.csv", index=False)
    cdf.to_csv(OUT / "per_class_progressive_v2.csv", index=False)
    vdf.to_csv(OUT / "exact_match_verification_progressive_v2.csv", index=False)
    if len(mdf):
        agg = (
            mdf.groupby(["model", "attack", "budget"])
            [["attacked_accuracy", "asr_clean_correct"]]
            .agg(["mean", "std"])
        )
        agg.to_csv(OUT / "main_attack_progressive_v2_aggregate.csv")
    return mdf


def run_progressive_cost(Xattack, yattack):
    rows = []
    n = min(COST_N, len(Xattack))
    Xs, ys = Xattack[:n], yattack[:n]

    for name in TEMPORAL_MODELS:
        tag = f"progressive_cost_{name}"
        cached = _load_part(tag)
        if cached is not None:
            rows += cached["rows"]
            continue

        model = load_model(name, 0)
        frame = load_model("frame_cnn", 0)
        clean_pred = predict_logits(model, Xs).argmax(1)
        clean_correct = clean_pred == ys
        fclean = predict_logits(frame, Xs)

        stealth = {b: [] for b in PROGRESSIVE_COST_BUDGETS}
        free = {b: [] for b in PROGRESSIVE_COST_BUDGETS}

        for i in range(n):
            cs = progressive_unique_attack(
                model, Xs[i], ys[i], PROGRESSIVE_COST_PATH)
            cf = progressive_free_unique_attack(
                model, Xs[i], ys[i], PROGRESSIVE_COST_PATH)
            for b in PROGRESSIVE_COST_BUDGETS:
                stealth[b].append(cs[b]["X"])
                free[b].append(cf[b]["X"])

        unit = []
        for b in PROGRESSIVE_COST_BUDGETS:
            for kind, arrs in [("stealth", stealth[b]), ("free", free[b])]:
                A = np.stack(arrs)
                pred = predict_logits(model, A).argmax(1)
                asr, ncc = safe_asr(pred, ys, clean_correct)
                fl = predict_logits(frame, A)
                unit.append({
                    "seed": 0,
                    "model": name,
                    "kind": kind,
                    "budget": float(b),
                    "optimization_path": "0.05->0.10->0.20",
                    "gradient_updates_max": len(PROGRESSIVE_COST_PATH),
                    "n": n,
                    "n_clean_correct": ncc,
                    "clean_accuracy": float(clean_correct.mean()),
                    "attacked_accuracy": float((pred == ys).mean()),
                    "asr_clean_correct": asr,
                    "prediction_flip_rate": float((pred != clean_pred).mean()),
                    **batch_frame_metrics(Xs, A),
                    "frame_cnn_logits_exact": bool(
                        np.array_equal(fl, fclean)),
                    "frame_cnn_flip_rate": float(
                        (fl.argmax(1) != fclean.argmax(1)).mean()),
                })
                del A
        _save_part(tag, {"rows": unit})
        rows += unit
        del model, frame
        gc.collect()

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "cost_of_stealth_progressive_v2.csv", index=False)
    return df


def main():
    # Fail fast only on checkpoints required by the selected cleanup task(s).
    if RUN_EQUAL_ROUND_COST:
        for name in TEMPORAL_MODELS:
            checkpoint_path(name, 0)
    if RUN_PROGRESSIVE_MAIN:
        for seed in SEEDS:
            for name in ["frame_cnn", "lif_mlp", "temporal_gru"]:
                checkpoint_path(name, seed)
    if RUN_PROGRESSIVE_COST:
        for name in ["frame_cnn"] + TEMPORAL_MODELS:
            checkpoint_path(name, 0)

    _, Xattack, yattack = load_attack_set()

    if RUN_EQUAL_ROUND_COST:
        run_equal_round_cost(Xattack, yattack)

    if RUN_PROGRESSIVE_MAIN:
        run_progressive_main(Xattack, yattack)

    if RUN_PROGRESSIVE_COST:
        run_progressive_cost(Xattack, yattack)

    with open(OUT / "nmnist_cost_cleanup_final_config.json", "w") as f:
        json.dump({
            "seeds": SEEDS,
            "attack_samples": int(len(Xattack)),
            "attack_per_class": ATTACK_PER_CLASS,
            "progressive_budgets": ATTACK_BUDGETS,
            "exact_matcher": "strict_integral_max_flow_no_fallback",
            "legacy_cost_rounds_each_arm": LEGACY_ROUNDS,
            "legacy_cost_budget": LEGACY_COST_BUDGET,
            "progressive_cost_path": PROGRESSIVE_COST_PATH,
            "progressive_cost_budgets": PROGRESSIVE_COST_BUDGETS,
            "notes": (
                "Original landed files preserved. Equal-round legacy cost fixes "
                "4-vs-2 mismatch; progressive rerun supplies a cross-dataset-"
                "uniform headline/control protocol."
            ),
        }, f, indent=2)

    print("\nDONE. N-MNIST cleanup outputs are under:", OUT)


if __name__ == "__main__":
    main()

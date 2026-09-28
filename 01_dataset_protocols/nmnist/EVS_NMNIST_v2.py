#!/usr/bin/env python3
"""
Event-Security Study, Dataset 1: N-MNIST.

Auto-converted from EVS_NMNIST.ipynb for headless execution on a GPU cluster.
The experiment code is byte-identical to the notebook's code cells. The only
changes are:
  1. ROOT repointed from /content/event_security_nmnist
     to /path/to/workspace/EVS_NMNIST/run
  2. a `display()` shim, since display() is an IPython builtin and this runs
     as a plain script; it prints the object instead
  3. matplotlib forced to the Agg backend so plt.show() is a no-op headless
Markdown cells are carried through as comments.
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
# # Event-Security Study — Dataset 1: N-MNIST
# ## Full paper-grade real-event validation, multi-seed, unique-event budgets, NO TONIC
# 
# This is the first dataset notebook for the expanded study.
# 
# ### Main goals
# - train on the standard N-MNIST split with **3 independent seeds**;
# - evaluate a frame consumer and two temporally sensitive consumers;
# - enforce a **strict unique-event retiming budget**;
# - preserve every coarse 20-ms frame **exactly as an integer tensor**;
# - compare against matched random retiming;
# - report conditional attack success, class-wise results, and displacement statistics;
# - export paper-ready CSVs, figures, and raw timestamp-edited examples.
# 
# ### Models
# - `FrameCNN`: 20-ms accumulated frames.
# - `LIFMLP`: spiking temporal consumer at 2-ms resolution.
# - `TemporalGRU`: conventional recurrent temporal consumer at 2-ms resolution.
# 
# The notebook uses the original N-MNIST `.bin` files directly. **No Tonic is used.**


# ======================================================================
# [code cell 1]
# ======================================================================

# ============================================================
# 0. Imports and runtime
# ============================================================
import os, gc, json, math, time, random, shutil, zipfile
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import requests
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt

print("PyTorch:", torch.__version__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

ROOT = Path("/path/to/workspace/EVS_NMNIST/run")
DATA_ROOT = ROOT / "data"
CKPT_ROOT = ROOT / "checkpoints"
OUT = ROOT / "results"
for p in [DATA_ROOT, CKPT_ROOT, OUT]:
    p.mkdir(parents=True, exist_ok=True)

torch.backends.cudnn.benchmark = True


# ======================================================================
# [markdown cell 2]
# ## 1. Configuration
# 
# The temporal discretization matches the manuscript theory: \(dt=2\) ms, \(\Delta=20\) ms, and 10 fine bins per coarse window. The first boxcar null is therefore 50 Hz.
# 
# `PAPER_MODE=True` is the intended paper run. Set it to `False` only for a short smoke test.


# ======================================================================
# [code cell 3]
# ======================================================================

# ============================================================
# 1. Configuration
# ============================================================
PAPER_MODE = True

SEEDS = [0, 1, 2]

DT_US = 2_000
S = 10
DELTA_US = DT_US * S
DURATION_US = 300_000

T = DURATION_US // DT_US
NW = T // S
H = W = 34
C = 2

TAU_SEC = 0.004
BETA = math.exp(-(DT_US / 1e6) / TAU_SEC)

if PAPER_MODE:
    N_TRAIN = None              # full standard training split
    N_TEST = None               # full standard test split
    ATTACK_PER_CLASS = 100      # 1,000 balanced attack samples
    EPOCHS_FRAME = 6
    EPOCHS_LIF = 10
    EPOCHS_GRU = 8
    TRAIN_BATCH = 64
    EVAL_BATCH = 128
    ATTACK_BUDGETS = [0.05, 0.10, 0.20, 0.30]
    ITER_ROUNDS = 4
else:
    N_TRAIN = 10_000
    N_TEST = 2_000
    ATTACK_PER_CLASS = 10
    EPOCHS_FRAME = 2
    EPOCHS_LIF = 2
    EPOCHS_GRU = 2
    TRAIN_BATCH = 64
    EVAL_BATCH = 128
    ATTACK_BUDGETS = [0.10, 0.30]
    ITER_ROUNDS = 2

LR = 1e-3
WEIGHT_DECAY = 1e-5
LIF_HIDDEN = 256
GRU_INPUT = 128
GRU_HIDDEN = 128

# None means an event may move anywhere inside its ORIGINAL 20-ms window.
MAX_SHIFT_BINS = None

FORCE_RETRAIN = False

print("Paper mode:", PAPER_MODE)
print("Seeds:", SEEDS)
print("T:", T, "NW:", NW)
print("dt:", DT_US/1000, "ms")
print("Delta:", DELTA_US/1000, "ms")
print("first boxcar null:", 1e6/DELTA_US, "Hz")
print("attack samples:", ATTACK_PER_CLASS*10)


# ======================================================================
# [markdown cell 4]
# ## 2. Download and extract original N-MNIST
# 
# These are the direct archive links used in the earlier successful no-Tonic experiments.


# ======================================================================
# [code cell 5]
# ======================================================================

# ============================================================
# 2. Download / extract
# ============================================================
NM_TRAIN_URL = (
    "https://www.dropbox.com/sh/"
    "tg2ljlbmtzygrag/"
    "AABlMOuR15ugeOxMCX0Pvoxga/"
    "Train.zip?dl=1"
)
NM_TEST_URL = (
    "https://www.dropbox.com/sh/"
    "tg2ljlbmtzygrag/"
    "AADSKgJ2CjaBWh75HnTNZyhca/"
    "Test.zip?dl=1"
)

TRAIN_ZIP = DATA_ROOT / "Train.zip"
TEST_ZIP = DATA_ROOT / "Test.zip"
TRAIN_DIR = DATA_ROOT / "Train_extracted"
TEST_DIR = DATA_ROOT / "Test_extracted"

def download_if_missing(url, destination, chunk_mb=4):
    destination = Path(destination)
    if destination.exists() and destination.stat().st_size > 1_000_000:
        print("Already downloaded:", destination)
        return
    print("Downloading:", destination.name)
    with requests.get(url, stream=True, allow_redirects=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        chunk = chunk_mb * 1024 * 1024
        with open(destination, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=destination.name
        ) as bar:
            for block in r.iter_content(chunk_size=chunk):
                if block:
                    f.write(block)
                    bar.update(len(block))

def extract_if_missing(zip_path, destination):
    destination = Path(destination)
    marker = destination / ".done"
    if marker.exists():
        print("Already extracted:", destination)
        return
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    print("Extracting:", zip_path.name)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(destination)
    marker.touch()

download_if_missing(NM_TRAIN_URL, TRAIN_ZIP)
download_if_missing(NM_TEST_URL, TEST_ZIP)
extract_if_missing(TRAIN_ZIP, TRAIN_DIR)
extract_if_missing(TEST_ZIP, TEST_DIR)


# ======================================================================
# [markdown cell 6]
# ## 3. Native N-MNIST parser and sample discovery


# ======================================================================
# [code cell 7]
# ======================================================================

# ============================================================
# 3. Native parser
# ============================================================
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

def label_hist(samples):
    return dict(sorted(Counter(int(y) for _, y in samples).items()))

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

train_samples_all = discover_samples(TRAIN_DIR)
test_samples_all = discover_samples(TEST_DIR)

print("Full train:", len(train_samples_all), label_hist(train_samples_all))
print("Full test :", len(test_samples_all), label_hist(test_samples_all))

assert len(train_samples_all) > 0 and len(test_samples_all) > 0

example_path, example_label = train_samples_all[0]
example_events = read_nmnist_bin(example_path)
print("Example label:", example_label, "events:", len(example_events))
print(example_events[:5])


# ======================================================================
# [markdown cell 8]
# ## 4. Integer event binning


# ======================================================================
# [code cell 9]
# ======================================================================

# ============================================================
# 4. Event list -> [T,C,H,W] integer count tensor
# ============================================================
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

def coarse_frames_np(X):
    return X.reshape(NW, S, C, H, W).sum(axis=1)

X0 = events_to_tensor(example_events)
print("Tensor:", X0.shape, X0.dtype)
print("Represented events:", int(X0.sum()))
print("Coarse:", coarse_frames_np(X0).shape)


# ======================================================================
# [markdown cell 10]
# ## 5. Standard benchmark loaders


# ======================================================================
# [code cell 11]
# ======================================================================

# ============================================================
# 5. Dataset / loaders
# ============================================================
class NMNISTBinaryDataset(Dataset):
    def __init__(self, samples):
        self.samples = list(samples)
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        ev = read_nmnist_bin(path)
        X = events_to_tensor(ev, normalize_time=True)
        return X.astype(np.float32), int(label)

def balanced_subset_total(samples, n_total, seed):
    if n_total is None:
        return list(samples)
    return stratified_select(samples, per_class=n_total//10, seed=seed)

train_samples = balanced_subset_total(train_samples_all, N_TRAIN, seed=123)
test_samples = balanced_subset_total(test_samples_all, N_TEST, seed=456)

print("Train:", len(train_samples), label_hist(train_samples))
print("Test :", len(test_samples), label_hist(test_samples))

train_ds = NMNISTBinaryDataset(train_samples)
test_ds = NMNISTBinaryDataset(test_samples)

def make_loader(dataset, batch_size, shuffle):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

train_loader = make_loader(train_ds, TRAIN_BATCH, True)
test_loader = make_loader(test_ds, EVAL_BATCH, False)

xb, yb = next(iter(train_loader))
print("Batch:", xb.shape, yb.shape)


# ======================================================================
# [markdown cell 12]
# # 6. Heterogeneous consumers


# ======================================================================
# [code cell 13]
# ======================================================================

# ============================================================
# 6. Models
# ============================================================
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

def build_model(name):
    if name == "frame_cnn":
        return FrameCNN()
    if name == "lif_mlp":
        return LIFMLP()
    if name == "temporal_gru":
        return TemporalGRU()
    raise ValueError(name)

MODEL_NAMES = ["frame_cnn", "lif_mlp", "temporal_gru"]
TEMPORAL_MODELS = ["lif_mlp", "temporal_gru"]
EPOCH_MAP = {
    "frame_cnn": EPOCHS_FRAME,
    "lif_mlp": EPOCHS_LIF,
    "temporal_gru": EPOCHS_GRU,
}


# ======================================================================
# [markdown cell 14]
# ## 7. Train 3 seeds and save checkpoints


# ======================================================================
# [code cell 15]
# ======================================================================

# ============================================================
# 7. Training
# ============================================================
def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

@torch.no_grad()
def evaluate_accuracy(model, loader):
    model.eval()
    correct = total = 0
    for X, y in loader:
        X = X.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)
        pred = model(X).argmax(1)
        correct += int((pred == y).sum())
        total += len(y)
    return correct / total

def train_model(model, loader, epochs):
    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    history = []
    for ep in range(epochs):
        model.train()
        losses = []
        t0 = time.time()
        for X, y in loader:
            X = X.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            loss = F.cross_entropy(model(X), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        sched.step()
        mean_loss = float(np.mean(losses))
        history.append(mean_loss)
        print(f"epoch {ep+1:02d}/{epochs} loss={mean_loss:.4f} time={time.time()-t0:.1f}s")
    return model, history

clean_rows = []
trained_paths = {}

for seed in SEEDS:
    print("\n" + "#"*80)
    print("SEED", seed)
    print("#"*80)
    set_all_seeds(seed)

    for model_name in MODEL_NAMES:
        print("\nMODEL:", model_name)
        ckpt = CKPT_ROOT / f"{model_name}_seed{seed}.pt"
        hist_path = CKPT_ROOT / f"{model_name}_seed{seed}_history.json"

        model = build_model(model_name).to(DEVICE)

        if ckpt.exists() and not FORCE_RETRAIN:
            print("Loading:", ckpt)
            model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
            history = []
        else:
            model, history = train_model(model, train_loader, EPOCH_MAP[model_name])
            torch.save(model.state_dict(), ckpt)
            with open(hist_path, "w") as f:
                json.dump(history, f, indent=2)

        acc = evaluate_accuracy(model, test_loader)
        print("Clean test accuracy:", acc)

        clean_rows.append({
            "seed": seed,
            "model": model_name,
            "clean_test_accuracy": float(acc),
        })
        trained_paths[(seed, model_name)] = str(ckpt)

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

clean_df = pd.DataFrame(clean_rows)
display(clean_df)
clean_df.to_csv(OUT / "clean_accuracy_by_seed.csv", index=False)

print("\nMean ± std")
display(clean_df.groupby("model")["clean_test_accuracy"].agg(["mean","std"]))


# ======================================================================
# [markdown cell 16]
# # 8. Balanced attack subset


# ======================================================================
# [code cell 17]
# ======================================================================

# ============================================================
# 8. Attack subset
# ============================================================
attack_samples = stratified_select(
    test_samples_all,
    per_class=ATTACK_PER_CLASS,
    seed=2027
)
print("Attack labels:", label_hist(attack_samples))

def collect_dense_attack_set(samples):
    XX, yy = [], []
    for path, label in tqdm(samples, desc="Loading attack subset"):
        XX.append(events_to_tensor(read_nmnist_bin(path)))
        yy.append(label)
    return np.stack(XX).astype(np.int16), np.asarray(yy, dtype=np.int64)

Xattack, yattack = collect_dense_attack_set(attack_samples)
print("Xattack:", Xattack.shape)
print("labels:", np.bincount(yattack, minlength=10))
print("mean events/sample:", Xattack.sum(axis=(1,2,3,4)).mean())


# ======================================================================
# [markdown cell 18]
# # 9. Attack helpers: strict unique-event budget
# 
# The `available` tensor tracks original events that have **never** been moved. Moved events are not added back to `available`, so iterative optimization cannot move the same physical event twice.


# ======================================================================
# [code cell 19]
# ======================================================================

# ============================================================
# 9A. General helpers
# ============================================================
@torch.no_grad()
def predict_logits(model, X_np, batch=EVAL_BATCH):
    model.eval()
    outs = []
    for i in range(0, len(X_np), batch):
        xb = torch.from_numpy(X_np[i:i+batch].astype(np.float32)).to(DEVICE)
        outs.append(model(xb).detach().cpu().numpy())
    return np.concatenate(outs, axis=0)

def coarse_exact_equal(Xa, Xb):
    Fa = Xa.reshape(NW,S,C,H,W).sum(axis=1)
    Fb = Xb.reshape(NW,S,C,H,W).sum(axis=1)
    return np.array_equal(Fa, Fb)

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


# ======================================================================
# [code cell 20]
# ======================================================================

# ============================================================
# 9B. Unique-event move primitive
# ============================================================
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
    assert coarse_exact_equal(Xcur, Xnext)

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
    assert coarse_exact_equal(X0, Xcur)

    return {
        "X": Xcur,
        "moved_unique": int(moved_total),
        "requested_unique": int(target_budget),
        "realized_unique_fraction": float(moved_total/max(1,total_events)),
        "mean_abs_shift_bins": float(disp_weighted/moved_total if moved_total else 0.0),
        "max_abs_shift_bins": int(max_disp_total),
    }


# ======================================================================
# [code cell 21]
# ======================================================================

# ============================================================
# 9C. Matched UNIQUE random-retiming control
# ============================================================
def unique_random_attack_one(X0, budget_frac, rng):
    Xg = to_grouped(X0).astype(np.int32)
    original = Xg.copy()

    groups, srcs = np.nonzero(original > 0)
    unit_groups, unit_srcs = [], []

    for g, src in zip(groups, srcs):
        count = int(original[g,src])
        unit_groups.extend([int(g)] * count)
        unit_srcs.extend([int(src)] * count)

    unit_groups = np.asarray(unit_groups, dtype=np.int32)
    unit_srcs = np.asarray(unit_srcs, dtype=np.int16)

    total = len(unit_groups)
    budget = min(int(round(budget_frac*total)), total)

    if budget == 0:
        return {
            "X": X0.copy(),
            "moved_unique": 0,
            "requested_unique": 0,
            "realized_unique_fraction": 0.0,
            "mean_abs_shift_bins": 0.0,
            "max_abs_shift_bins": 0,
        }

    chosen = rng.choice(total, size=budget, replace=False)
    disp_sum = 0
    max_disp = 0

    for idx in chosen:
        g = int(unit_groups[int(idx)])
        src = int(unit_srcs[int(idx)])

        if MAX_SHIFT_BINS is None:
            choices = [d for d in range(S) if d != src]
        else:
            choices = [
                d for d in range(max(0,src-MAX_SHIFT_BINS), min(S,src+MAX_SHIFT_BINS+1))
                if d != src
            ]

        dst = int(rng.choice(choices))
        Xg[g,src] -= 1
        Xg[g,dst] += 1

        disp = abs(dst-src)
        disp_sum += disp
        max_disp = max(max_disp, disp)

    Xatt = from_grouped(Xg).astype(np.int16)
    assert Xatt.min() >= 0
    assert int(Xatt.sum()) == int(X0.sum())
    assert coarse_exact_equal(X0, Xatt)

    return {
        "X": Xatt,
        "moved_unique": int(budget),
        "requested_unique": int(budget),
        "realized_unique_fraction": float(budget/max(1,total)),
        "mean_abs_shift_bins": float(disp_sum/max(1,budget)),
        "max_abs_shift_bins": int(max_disp),
    }


# ======================================================================
# [markdown cell 22]
# ## 10. Attack mechanics sanity check


# ======================================================================
# [code cell 23]
# ======================================================================

# ============================================================
# 10. Unit test
# ============================================================
test_model = build_model("lif_mlp").to(DEVICE)
test_model.load_state_dict(
    torch.load(trained_paths[(SEEDS[0],"lif_mlp")], map_location=DEVICE)
)

out = unique_iterative_attack_one(
    test_model, Xattack[0], yattack[0], budget_frac=0.10, rounds=2
)
Xa = out["X"]

print(out)
print("same total events:", int(Xa.sum()) == int(Xattack[0].sum()))
print("coarse exact:", coarse_exact_equal(Xattack[0], Xa))
print("min count:", Xa.min())

del test_model
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()


# ======================================================================
# [markdown cell 24]
# # 11. Main multi-seed attack experiment
# 
# This is the central experiment. It compares the strict unique-event iterative attack against a matched unique random-retiming control on the exact same balanced recordings.


# ======================================================================
# [code cell 25]
# ======================================================================

# ============================================================
# 11. Main experiment
# ============================================================
attack_rows = []
class_rows = []

for seed in SEEDS:
    print("\n" + "#"*90)
    print("ATTACK EVALUATION — SEED", seed)
    print("#"*90)

    frame_model = build_model("frame_cnn").to(DEVICE)
    frame_model.load_state_dict(
        torch.load(trained_paths[(seed,"frame_cnn")], map_location=DEVICE)
    )
    clean_frame_logits = predict_logits(frame_model, Xattack)
    clean_frame_pred = clean_frame_logits.argmax(1)

    for model_name in TEMPORAL_MODELS:
        print("\nTEMPORAL MODEL:", model_name)

        temporal_model = build_model(model_name).to(DEVICE)
        temporal_model.load_state_dict(
            torch.load(trained_paths[(seed,model_name)], map_location=DEVICE)
        )

        clean_logits = predict_logits(temporal_model, Xattack)
        clean_pred = clean_logits.argmax(1)
        clean_acc = float((clean_pred == yattack).mean())
        clean_correct = (clean_pred == yattack)
        print("Clean attack-subset accuracy:", clean_acc)

        for budget in ATTACK_BUDGETS:
            print(f"\nGradient unique-event attack: {budget:.0%}")

            attacked_list = []
            meta = []

            for i in tqdm(range(len(Xattack)), desc=f"{model_name} grad {budget:.0%}"):
                r = unique_iterative_attack_one(
                    temporal_model, Xattack[i], yattack[i], budget, rounds=ITER_ROUNDS
                )
                attacked_list.append(r["X"])
                meta.append(r)

            Xadv = np.stack(attacked_list)
            frame_inv = batch_frame_metrics(Xattack, Xadv)
            adv_logits = predict_logits(temporal_model, Xadv)
            adv_pred = adv_logits.argmax(1)

            attacked_acc = float((adv_pred == yattack).mean())
            flip_rate = float((adv_pred != clean_pred).mean())
            asr = float((adv_pred[clean_correct] != yattack[clean_correct]).mean()) if clean_correct.sum() else float("nan")

            attacked_frame_logits = predict_logits(frame_model, Xadv)
            frame_logits_exact = bool(np.array_equal(attacked_frame_logits, clean_frame_logits))
            frame_flip_rate = float((attacked_frame_logits.argmax(1) != clean_frame_pred).mean())

            row = {
                "seed": seed,
                "temporal_model": model_name,
                "attack": "unique_iterative_gradient",
                "budget_requested": budget,
                "rounds": ITER_ROUNDS,
                "clean_accuracy": clean_acc,
                "attacked_accuracy": attacked_acc,
                "prediction_flip_rate": flip_rate,
                "asr_on_clean_correct": asr,
                "mean_unique_events_moved": float(np.mean([m["moved_unique"] for m in meta])),
                "mean_unique_events_requested": float(np.mean([m["requested_unique"] for m in meta])),
                "mean_realized_unique_fraction": float(np.mean([m["realized_unique_fraction"] for m in meta])),
                "mean_abs_shift_bins": float(np.mean([m["mean_abs_shift_bins"] for m in meta])),
                "mean_abs_shift_ms": float(np.mean([m["mean_abs_shift_bins"] for m in meta])) * DT_US/1000.0,
                "max_abs_shift_bins": int(np.max([m["max_abs_shift_bins"] for m in meta])),
                "max_abs_shift_ms": int(np.max([m["max_abs_shift_bins"] for m in meta])) * DT_US/1000.0,
                "frame_exact_equal": frame_inv["frame_exact_equal"],
                "max_integer_frame_difference": frame_inv["max_integer_frame_difference"],
                "frame_cnn_logits_exact_equal": frame_logits_exact,
                "frame_cnn_prediction_flip_rate": frame_flip_rate,
            }
            print(row)
            attack_rows.append(row)

            for cls in range(10):
                m = (yattack == cls)
                cm = m & clean_correct
                class_rows.append({
                    "seed": seed,
                    "temporal_model": model_name,
                    "attack": "unique_iterative_gradient",
                    "budget_requested": budget,
                    "class": cls,
                    "n": int(m.sum()),
                    "clean_accuracy": float((clean_pred[m] == yattack[m]).mean()),
                    "attacked_accuracy": float((adv_pred[m] == yattack[m]).mean()),
                    "asr_on_clean_correct": float((adv_pred[cm] != yattack[cm]).mean()) if cm.sum() else float("nan"),
                })

            del Xadv, attacked_list, meta
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print(f"Random unique-event control: {budget:.0%}")
            rng = np.random.default_rng(100000 + 1000*seed + int(budget*100))
            random_list = []
            random_meta = []

            for i in tqdm(range(len(Xattack)), desc=f"{model_name} random {budget:.0%}"):
                r = unique_random_attack_one(Xattack[i], budget, rng)
                random_list.append(r["X"])
                random_meta.append(r)

            Xrand = np.stack(random_list)
            rand_inv = batch_frame_metrics(Xattack, Xrand)
            rand_logits = predict_logits(temporal_model, Xrand)
            rand_pred = rand_logits.argmax(1)

            rand_acc = float((rand_pred == yattack).mean())
            rand_flip = float((rand_pred != clean_pred).mean())
            rand_asr = float((rand_pred[clean_correct] != yattack[clean_correct]).mean()) if clean_correct.sum() else float("nan")

            rand_frame_logits = predict_logits(frame_model, Xrand)
            rand_frame_exact = bool(np.array_equal(rand_frame_logits, clean_frame_logits))
            rand_frame_flip = float((rand_frame_logits.argmax(1) != clean_frame_pred).mean())

            row_r = {
                "seed": seed,
                "temporal_model": model_name,
                "attack": "unique_random",
                "budget_requested": budget,
                "rounds": 1,
                "clean_accuracy": clean_acc,
                "attacked_accuracy": rand_acc,
                "prediction_flip_rate": rand_flip,
                "asr_on_clean_correct": rand_asr,
                "mean_unique_events_moved": float(np.mean([m["moved_unique"] for m in random_meta])),
                "mean_unique_events_requested": float(np.mean([m["requested_unique"] for m in random_meta])),
                "mean_realized_unique_fraction": float(np.mean([m["realized_unique_fraction"] for m in random_meta])),
                "mean_abs_shift_bins": float(np.mean([m["mean_abs_shift_bins"] for m in random_meta])),
                "mean_abs_shift_ms": float(np.mean([m["mean_abs_shift_bins"] for m in random_meta])) * DT_US/1000.0,
                "max_abs_shift_bins": int(np.max([m["max_abs_shift_bins"] for m in random_meta])),
                "max_abs_shift_ms": int(np.max([m["max_abs_shift_bins"] for m in random_meta])) * DT_US/1000.0,
                "frame_exact_equal": rand_inv["frame_exact_equal"],
                "max_integer_frame_difference": rand_inv["max_integer_frame_difference"],
                "frame_cnn_logits_exact_equal": rand_frame_exact,
                "frame_cnn_prediction_flip_rate": rand_frame_flip,
            }
            print(row_r)
            attack_rows.append(row_r)

            for cls in range(10):
                m = (yattack == cls)
                cm = m & clean_correct
                class_rows.append({
                    "seed": seed,
                    "temporal_model": model_name,
                    "attack": "unique_random",
                    "budget_requested": budget,
                    "class": cls,
                    "n": int(m.sum()),
                    "clean_accuracy": float((clean_pred[m] == yattack[m]).mean()),
                    "attacked_accuracy": float((rand_pred[m] == yattack[m]).mean()),
                    "asr_on_clean_correct": float((rand_pred[cm] != yattack[cm]).mean()) if cm.sum() else float("nan"),
                })

            del Xrand, random_list, random_meta
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        del temporal_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    del frame_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

attack_df = pd.DataFrame(attack_rows)
class_df = pd.DataFrame(class_rows)

attack_df.to_csv(OUT / "main_attack_results_by_seed.csv", index=False)
class_df.to_csv(OUT / "per_class_attack_results.csv", index=False)

display(attack_df.head())


# ======================================================================
# [markdown cell 26]
# # 12. Aggregate across seeds and make paper figures


# ======================================================================
# [code cell 27]
# ======================================================================

# ============================================================
# 12. Aggregate
# ============================================================
agg = (
    attack_df
    .groupby(["temporal_model","attack","budget_requested"])
    .agg({
        "clean_accuracy":["mean","std"],
        "attacked_accuracy":["mean","std"],
        "asr_on_clean_correct":["mean","std"],
        "prediction_flip_rate":["mean","std"],
        "mean_realized_unique_fraction":["mean","std"],
        "mean_abs_shift_ms":["mean","std"],
        "frame_cnn_prediction_flip_rate":["mean","max"],
        "max_integer_frame_difference":["max"],
    })
)

display(agg)
agg.to_csv(OUT / "main_attack_results_aggregate.csv")


# ======================================================================
# [code cell 28]
# ======================================================================

# ============================================================
# 13. Figures
# ============================================================
for model_name in TEMPORAL_MODELS:
    fig = plt.figure(figsize=(5.5,4.0))

    for attack_name in ["unique_iterative_gradient","unique_random"]:
        z = (
            attack_df[
                (attack_df["temporal_model"] == model_name)
                & (attack_df["attack"] == attack_name)
            ]
            .groupby("budget_requested")["asr_on_clean_correct"]
            .agg(["mean","std"])
            .reset_index()
        )

        x = z["budget_requested"].values * 100
        y = z["mean"].values * 100
        s = z["std"].fillna(0).values * 100

        plt.plot(x, y, marker="o", label=attack_name)
        plt.fill_between(x, y-s, y+s, alpha=0.2)

    plt.xlabel("Unique original events retimed (%)")
    plt.ylabel("Attack success on clean-correct samples (%)")
    plt.title(f"N-MNIST — {model_name}")
    plt.legend()
    plt.tight_layout()

    fig.savefig(OUT / f"nmnist_{model_name}_asr_vs_budget.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"nmnist_{model_name}_asr_vs_budget.png", dpi=200, bbox_inches="tight")
    plt.show()
    plt.close(fig)


# ======================================================================
# [markdown cell 29]
# # 13. Optional cost-of-stealth diagnostic
# 
# This compares the frame-invisible attack to a same-budget **free temporal retiming** attack that may move an event anywhere in the 300-ms recording at fixed pixel/polarity. The free attack is intentionally not frame-stealthy.
# 
# It is run only on 100 samples because its search space is much larger.


# ======================================================================
# [code cell 30]
# ======================================================================

# ============================================================
# 14. Optional cost-of-stealth baseline
# ============================================================
RUN_COST_OF_STEALTH = True
COST_N = min(100, len(Xattack))
COST_BUDGET = 0.20

def free_grouped(X):
    return X.transpose(1,2,3,0).reshape(-1,T)

def free_from_grouped(Xg):
    return Xg.reshape(C,H,W,T).transpose(3,0,1,2)

def free_unique_attack_one(model, X0, y, budget_frac):
    Xcur = X0.copy().astype(np.int16)
    total = int(X0.sum())
    target = int(round(budget_frac*total))
    available = free_grouped(X0).astype(np.int32)
    moved = 0

    for rr in range(2):
        remaining = target - moved
        if remaining <= 0:
            break

        g = input_gradient(model, Xcur, y)
        Xg = free_grouped(Xcur).astype(np.int32)
        Gg = free_grouped(g).astype(np.float32)

        groups, srcs = np.nonzero(available > 0)
        best_dst = Gg.argmax(axis=1)
        gains = Gg[groups, best_dst[groups]] - Gg[groups, srcs]
        valid = (best_dst[groups] != srcs) & (gains > 0)

        groups, srcs, gains = groups[valid], srcs[valid], gains[valid]
        order = np.argsort(-gains)

        this_target = int(math.ceil(remaining/(2-rr)))
        this_moved = 0

        for j in order:
            if this_moved >= this_target:
                break
            gg, src = int(groups[j]), int(srcs[j])
            dst = int(best_dst[gg])
            cap = int(available[gg,src])
            if cap <= 0:
                continue
            n = min(cap, this_target-this_moved)
            Xg[gg,src] -= n
            Xg[gg,dst] += n
            available[gg,src] -= n
            this_moved += n

        Xcur = free_from_grouped(Xg).astype(np.int16)
        moved += this_moved
        if this_moved == 0:
            break

    return Xcur, moved

cost_rows = []

if RUN_COST_OF_STEALTH:
    seed = SEEDS[0]
    model = build_model("lif_mlp").to(DEVICE)
    model.load_state_dict(torch.load(trained_paths[(seed,"lif_mlp")], map_location=DEVICE))
    clean_pred = predict_logits(model, Xattack[:COST_N]).argmax(1)

    for attack_kind in ["stealth","free"]:
        AA = []
        for i in tqdm(range(COST_N), desc=attack_kind):
            if attack_kind == "stealth":
                r = unique_iterative_attack_one(
                    model, Xattack[i], yattack[i], COST_BUDGET, rounds=ITER_ROUNDS
                )
                xa = r["X"]
            else:
                xa, _ = free_unique_attack_one(model, Xattack[i], yattack[i], COST_BUDGET)
            AA.append(xa)

        AA = np.stack(AA)
        pred = predict_logits(model, AA).argmax(1)
        frame_metric = batch_frame_metrics(Xattack[:COST_N], AA)

        cost_rows.append({
            "attack": attack_kind,
            "budget": COST_BUDGET,
            "accuracy": float((pred == yattack[:COST_N]).mean()),
            "prediction_flip_rate": float((pred != clean_pred).mean()),
            **frame_metric,
        })

        del AA
        gc.collect()

    del model
    cost_df = pd.DataFrame(cost_rows)
    display(cost_df)
    cost_df.to_csv(OUT / "cost_of_stealth.csv", index=False)


# ======================================================================
# [markdown cell 31]
# # 14. Raw-stream materialization
# 
# This converts a count-space attack back into a timestamp-edited raw event list while preserving the full event count and every event's `x`, `y`, and polarity.


# ======================================================================
# [code cell 32]
# ======================================================================

# ============================================================
# 15. Raw event materialization
# ============================================================
def materialize_retimed_events(ev_raw, Xclean, Xatt, seed=0):
    rng = np.random.default_rng(seed)

    full = ev_raw.copy()
    clean_t0 = int(full["t"].min()) if len(full) else 0
    full["t"] -= clean_t0

    in_eval = (
        (full["t"] >= 0) & (full["t"] < DURATION_US)
        & (full["x"] >= 0) & (full["x"] < W)
        & (full["y"] >= 0) & (full["y"] < H)
        & ((full["p"] == 0) | (full["p"] == 1))
    )

    eval_indices = np.where(in_eval)[0]
    eval_events = full[in_eval]
    old_tbin = (eval_events["t"].astype(np.int64)//DT_US).astype(np.int64)

    buckets = {}
    for local_i, global_i in enumerate(eval_indices):
        tb = int(old_tbin[local_i])
        w, s = tb//S, tb%S
        key = (w, int(full["p"][global_i]), int(full["y"][global_i]), int(full["x"][global_i]), s)
        buckets.setdefault(key, []).append(int(global_i))

    Xc = Xclean.reshape(NW,S,C,H,W)
    Xa = Xatt.reshape(NW,S,C,H,W)

    for w in range(NW):
        for p in range(C):
            for y in range(H):
                for x in range(W):
                    diff = Xa[w,:,p,y,x].astype(int) - Xc[w,:,p,y,x].astype(int)
                    src_bins, dst_bins = [], []
                    for s in range(S):
                        if diff[s] < 0:
                            src_bins += [s] * (-diff[s])
                        elif diff[s] > 0:
                            dst_bins += [s] * diff[s]

                    if not src_bins:
                        continue
                    assert len(src_bins) == len(dst_bins)

                    for src, dst in zip(src_bins, dst_bins):
                        key = (w,p,y,x,src)
                        if key not in buckets or not buckets[key]:
                            raise RuntimeError(f"Materialization mismatch: {key}->{dst}")
                        gi = buckets[key].pop()
                        lo = (w*S + dst) * DT_US
                        hi = lo + DT_US
                        full["t"][gi] = int(rng.integers(lo, hi))

    # Verify evaluated portion at the fixed clean-stream origin.
    eval_after = full[in_eval].copy()
    Xmat = events_to_tensor(eval_after, normalize_time=False)
    assert np.array_equal(Xmat, Xatt)

    # Verify event identity fields before timestamp sorting.
    assert len(full) == len(ev_raw)
    assert np.array_equal(full["x"], ev_raw["x"])
    assert np.array_equal(full["y"], ev_raw["y"])
    assert np.array_equal(full["p"], ev_raw["p"])

    return np.sort(full, order="t")

seed = SEEDS[0]
model = build_model("lif_mlp").to(DEVICE)
model.load_state_dict(torch.load(trained_paths[(seed,"lif_mlp")], map_location=DEVICE))

for i in range(3):
    r = unique_iterative_attack_one(
        model, Xattack[i], yattack[i], budget_frac=0.20, rounds=ITER_ROUNDS
    )

    raw_path, label = attack_samples[i]
    raw_clean = read_nmnist_bin(raw_path)
    raw_attacked = materialize_retimed_events(raw_clean, Xattack[i], r["X"], seed=9000+i)

    np.savez_compressed(
        OUT / f"raw_retimed_example_{i}.npz",
        clean=raw_clean,
        attacked=raw_attacked,
        label=int(label),
    )

    print(
        i,
        "label=", label,
        "full event count equal=", len(raw_clean)==len(raw_attacked),
        "unique moved=", r["moved_unique"],
        "coarse exact=", coarse_exact_equal(Xattack[i], r["X"]),
    )

del model
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()


# ======================================================================
# [markdown cell 33]
# # 15. Final validation and export


# ======================================================================
# [code cell 34]
# ======================================================================

# ============================================================
# 16. Final validation
# ============================================================
print("All gradient frame invariance:",
      bool(attack_df.loc[
          attack_df["attack"]=="unique_iterative_gradient",
          "frame_exact_equal"
      ].all()))

print("Maximum integer frame difference:",
      int(attack_df["max_integer_frame_difference"].max()))

print("Maximum frame-CNN prediction flip rate:",
      float(attack_df["frame_cnn_prediction_flip_rate"].max()))

print("\nClean model accuracy:")
display(clean_df.groupby("model")["clean_test_accuracy"].agg(["mean","std"]))

print("\nGradient attack summary:")
display(
    attack_df[
        attack_df["attack"]=="unique_iterative_gradient"
    ]
    .groupby(["temporal_model","budget_requested"])
    [["attacked_accuracy","asr_on_clean_correct","mean_realized_unique_fraction","mean_abs_shift_ms"]]
    .agg(["mean","std"])
)


# ======================================================================
# [code cell 35]
# ======================================================================

# ============================================================
# 17. Export ZIP
# ============================================================
config = {
    "target": "paper",
    "dataset": "N-MNIST",
    "loader": "custom native 5-byte parser; NO TONIC",
    "paper_mode": PAPER_MODE,
    "seeds": SEEDS,
    "dt_us": DT_US,
    "S": S,
    "delta_us": DELTA_US,
    "duration_us": DURATION_US,
    "T": T,
    "NW": NW,
    "tau_sec": TAU_SEC,
    "beta": BETA,
    "n_train": len(train_samples),
    "n_test": len(test_samples),
    "attack_samples": len(attack_samples),
    "epochs": EPOCH_MAP,
    "attack_budgets": ATTACK_BUDGETS,
    "iterative_rounds": ITER_ROUNDS,
    "max_shift_bins": MAX_SHIFT_BINS,
    "models": MODEL_NAMES,
}

with open(OUT/"run_config.json","w") as f:
    json.dump(config, f, indent=2)

zip_path = ROOT / "EVS_NMNIST_full_results.zip"

with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for fp in OUT.rglob("*"):
        if fp.is_file():
            zf.write(fp, arcname=fp.relative_to(ROOT))

print("RESULT ZIP:")
print(zip_path)
print("\nDownload with:")
print("from google.colab import files")
print(f'files.download("{zip_path}")')

"""Build the pinned 264-sample DVS manifest used by the landed direct comparison.

The old comparison evaluated cache_tensors/test_tensor_manifest.csv (264 rows,
sha256 fd9014ac...), not results*/attack_subset_manifest.csv. Point the suite at
the same population before claiming reconciliation:

    python pin_dvs_manifest.py
    export ATTACK_MANIFEST=/path/to/workspace/EVS_DVSGesture/run/results/pinned_dvs264_manifest.csv
"""
import hashlib, os, sys
from pathlib import Path
import pandas as pd

EXPECT_SHA = "fd9014acf6261f8930eb3363f3ae68483c0bd077f086a01a7b8052e29f8f1715"
root = Path(os.getenv("RUN_ROOT", "/path/to/workspace/EVS_DVSGesture/run"))
src  = root / "cache_tensors" / "test_tensor_manifest.csv"
dst  = root / "results" / "pinned_dvs264_manifest.csv"

if not src.exists():
    sys.exit(f"missing {src}")
sha = hashlib.sha256(src.read_bytes()).hexdigest()
print(f"source      : {src}")
print(f"source sha  : {sha}")
print(f"landed sha  : {EXPECT_SHA}")
print("  MATCH" if sha == EXPECT_SHA else "  WARNING: differs from the landed comparison manifest")

df = pd.read_csv(src)
print(f"rows        : {len(df)} (expect 264)")
missing = [p for p in df["path"].astype(str) if not Path(p).exists()]
if missing:
    sys.exit(f"{len(missing)} tensor paths do not resolve, first: {missing[0]}")
dst.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(dst, index=False)
print(f"wrote       : {dst}")
print(f"pinned sha  : {hashlib.sha256(dst.read_bytes()).hexdigest()}")
print(f"\nexport ATTACK_MANIFEST={dst}")

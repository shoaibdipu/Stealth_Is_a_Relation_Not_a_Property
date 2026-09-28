#!/usr/bin/env python3
import argparse, glob, hashlib
from pathlib import Path
import numpy as np, pandas as pd
p=argparse.ArgumentParser(description='Build a frozen disjoint clean calibration manifest from cached .npz tensors.')
p.add_argument('--source-glob',required=True,help='Glob for held-out clean/train cached tensors, e.g. /run/cache_tensors/train/*.npz')
p.add_argument('--out',required=True)
p.add_argument('--exclude-manifest',default='',help='Attack manifest whose paths must be excluded')
p.add_argument('--limit',type=int,default=0,help='0 = keep all. If >0, select approximately class-balanced examples.')
p.add_argument('--seed',type=int,default=2027)
a=p.parse_args()
paths=sorted(glob.glob(a.source_glob,recursive=True))
if not paths: raise SystemExit(f'no files match {a.source_glob}')
exclude=set()
if a.exclude_manifest:
    d=pd.read_csv(a.exclude_manifest)
    if 'path' not in d.columns: raise SystemExit('exclude manifest requires path column')
    exclude=set(d.path.astype(str))
rows=[]
for fp in paths:
    if fp in exclude: continue
    try:
        z=np.load(fp)
        if 'label' not in z.files: continue
        rows.append({'path':fp,'label':int(z['label'])})
    except Exception as e:
        print('skip',fp,repr(e))
if len(rows)<20: raise SystemExit(f'only {len(rows)} usable clean tensors')
df=pd.DataFrame(rows)
if a.limit and len(df)>a.limit:
    rng=np.random.default_rng(a.seed); chosen=[]
    groups={int(c):g.index.to_numpy().copy() for c,g in df.groupby('label')}
    for idx in groups.values(): rng.shuffle(idx)
    # round-robin classes to avoid collapsing 200-class DailyDVS calibration.
    classes=sorted(groups); ptr={c:0 for c in classes}
    while len(chosen)<a.limit:
        progress=False
        for c in classes:
            arr=groups[c]; j=ptr[c]
            if j<len(arr) and len(chosen)<a.limit:
                chosen.append(int(arr[j])); ptr[c]+=1; progress=True
        if not progress: break
    df=df.loc[chosen].reset_index(drop=True)
out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);df.to_csv(out,index=False)
h=hashlib.sha256(out.read_bytes()).hexdigest()
print(f'wrote {out} n={len(df)} classes={df.label.nunique()} sha256={h}')

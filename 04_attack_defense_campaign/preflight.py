#!/usr/bin/env python3
import argparse, hashlib, os
from pathlib import Path
import pandas as pd, numpy as np

p=argparse.ArgumentParser(description='Fail-fast campaign preflight checks')
p.add_argument('--dataset',required=True,choices=['dvsgesture','dailydvs200','cifar10dvs'])
p.add_argument('--run-root',required=True)
p.add_argument('--clean-pool',required=True)
p.add_argument('--direct-n',type=int,default=0)
a=p.parse_args()
root=Path(a.run_root); cp=Path(a.clean_pool)
if not root.exists(): raise SystemExit(f'RUN_ROOT missing: {root}')
if not cp.exists(): raise SystemExit(f'clean pool manifest missing: {cp}')
mans=sorted(root.glob('results*/attack_subset_manifest.csv'))
if not mans: raise SystemExit('no attack_subset_manifest.csv found under RUN_ROOT/results*')
if len(mans)>1 and not os.environ.get('ATTACK_MANIFEST'):
    raise SystemExit('multiple attack manifests found; export ATTACK_MANIFEST explicitly:\n  '+'\n  '.join(map(str,mans)))
man=Path(os.environ.get('ATTACK_MANIFEST',mans[0]))
attack=pd.read_csv(man); clean=pd.read_csv(cp)
if 'path' not in attack or 'path' not in clean: raise SystemExit('both manifests must contain a path column')
# Conservative disjointness: check the entire attack manifest, which is stronger than checking DIRECT_N.
a_paths=set(attack.path.astype(str)); c_paths=set(clean.path.astype(str)); overlap=a_paths & c_paths
if overlap: raise SystemExit(f'clean pool overlaps attack manifest on {len(overlap)} clips; examples={list(sorted(overlap))[:3]}')
missing=[x for x in clean.path.astype(str) if not Path(x).exists()]
if missing: raise SystemExit(f'clean pool contains {len(missing)} missing tensors; examples={missing[:3]}')
missing_attack=[x for x in attack.path.astype(str).head(a.direct_n or len(attack)) if not Path(x).exists()]
if missing_attack: raise SystemExit(f'attack manifest contains missing tensors; examples={missing_attack[:3]}')
if len(clean)<20: raise SystemExit(f'clean pool too small: {len(clean)}')
print(f'PREFLIGHT OK dataset={a.dataset} attack_manifest={man} n_attack_manifest={len(attack)} clean_pool={cp} n_clean={len(clean)} disjoint=yes')

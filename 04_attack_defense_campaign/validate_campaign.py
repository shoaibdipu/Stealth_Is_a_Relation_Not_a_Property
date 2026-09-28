#!/usr/bin/env python3
import argparse, os, re
from pathlib import Path
import numpy as np, pandas as pd
p=argparse.ArgumentParser();p.add_argument('--dataset',required=True,choices=['dvsgesture','dailydvs200','cifar10dvs']);p.add_argument('--run-root',default='');p.add_argument('--campaign-id',default='')
a=p.parse_args()
root=Path(a.run_root or os.environ.get('RUN_ROOT',''))
if not str(root):raise SystemExit('RUN_ROOT or --run-root required')
cid=a.campaign_id or os.environ.get('CAMPAIGN_ID','stealth_asr_defense_v1')
out=root/'results'/f"combined_attack_defense_{a.dataset}_{re.sub(r'[^A-Za-z0-9_.-]+','_',cid)}"
rows=out/'per_sample_rows.csv'
if not rows.exists():raise SystemExit(f'missing {rows}')
df=pd.read_csv(rows); main=['yu_pil_l0','pdsg_sda','yao_gumbel','free_retiming','null_space']
s0=df[(df.seed==0)&df.method.isin(main)&df.clean_correct.astype(bool)]
missing=[m for m in main if m not in set(s0.method)]
if missing:raise SystemExit(f'missing seed0 main methods: {missing}')
# All five main methods must use the same clean-correct population.
sets={m:set(s0.loc[s0.method==m,'sample_uid'].astype(str)) for m in main}
ref=sets[main[0]]
for m in main[1:]:
    if sets[m]!=ref:raise SystemExit(f'population mismatch {main[0]} vs {m}: {len(ref)} vs {len(sets[m])}')
null=s0[s0.method=='null_space']
if len(null)==0 or not ((null.D_A==0)&(null.D_inf==0)&(~null.protected_consumer_flip.astype(bool))).all():
    raise SystemExit('null-space exact-invariance assertion failed in merged output')
# Defense must exist for every main row.
required=['def_phase1_score','def_all_score','def_phase1_det_05','def_all_det_05']
for c in required:
    if c not in s0.columns or s0[c].isna().any():raise SystemExit(f'missing defense field {c}')
# Controlled free sweep coverage.
sw=df[(df.seed==0)&(df.method=='free_da_sweep')&df.clean_correct.astype(bool)]
expected={2,4,8,16,32}; got=set(sw.sweep_max_shift.dropna().astype(int)) if 'sweep_max_shift' in sw else set()
if got!=expected:raise SystemExit(f'controlled sweep max_shift mismatch: got {sorted(got)}, expected {sorted(expected)}')
cal=out/'defender_calibration.csv'
if not cal.exists():raise SystemExit('missing defender_calibration.csv')
print(f'VALIDATED {a.dataset}: n_clean_correct={len(ref)} seed0 five attacks, exact null invariance, defense fields, sweep={sorted(got)}')

# ---------------------------------------------------------------------------
# Calibration identity across every partial in this campaign.
#
# Under sharding each job rebuilds the defense bundle. If any shard calibrated
# on a different clean pool, its TPR/UASR are not comparable with the others and
# the frontier plot silently mixes calibrations. Every partial must agree on the
# pool, the fitted thresholds, the defense version and the eval population.
#
# Floats are compared through a canonical JSON encoding (sorted keys, fixed
# precision) rather than raw bytes, so harmless serialization differences across
# machines do not trip the check while real calibration drift still does.
# ---------------------------------------------------------------------------
import json as _json, glob as _glob, math as _math

def _canon(obj, ndigits=12):
    """Stable representation for diagnostics only; cryptographic identity uses saved hash."""
    if isinstance(obj, float):
        if _math.isnan(obj): return 'nan'
        if _math.isinf(obj): return 'inf' if obj > 0 else '-inf'
        return format(round(obj, ndigits), f'.{ndigits}f')
    if isinstance(obj, dict): return {k: _canon(obj[k], ndigits) for k in sorted(obj)}
    if isinstance(obj, (list, tuple)): return [_canon(v, ndigits) for v in obj]
    return obj

_parts = sorted(_glob.glob(str(out / 'partials' / '*' / '*.json')))
_parts = [p for p in _parts if not p.endswith('.progress.json')]
if not _parts:
    raise SystemExit('no completed partials found for calibration-identity check')

# Calibration is intentionally seed-dependent. Identity is therefore required
# within (dataset, seed, victim, consumer checkpoint, eval population), across all
# methods and shards. Cross-seed equality is neither required nor expected.
_refs = {}
_checked = 0
for _fp in _parts:
    try:
        _j = _json.loads(Path(_fp).read_text())
    except Exception as e:
        raise SystemExit(f'unreadable partial {_fp}: {e}')
    _cal = _j.get('defense_calibration') or []
    if not _cal:
        continue
    _pool = _j.get('defense_clean_pool') or {}
    _key = (
        str(_j.get('dataset')),
        int(_j.get('seed', -1)),
        str(_j.get('victim')),
        str(_j.get('consumer_sha')),
        str((_j.get('manifest') or {}).get('eval_population_sha256')),
    )
    _sig = {
        'pool_manifest_sha256': _pool.get('sha256'),
        'loaded_pool_sha256': _pool.get('loaded_pool_sha256'),
        'pool_n': _pool.get('n'),
        'defense_version': _j.get('defense_version'),
        'calibration_sha256': _j.get('defense_calibration_sha256'),
        'calibration': _canon(_cal),
    }
    if not _sig['calibration_sha256']:
        raise SystemExit(f'missing defense_calibration_sha256 in {_fp}')
    if _key not in _refs:
        _refs[_key] = (_sig, _fp)
    else:
        _ref, _ref_fp = _refs[_key]
        if _sig != _ref:
            _diff = [k for k in _sig if _sig[k] != _ref[k]]
            raise SystemExit(
                f'defense calibration differs within comparison group {_key} on {_diff}\n'
                f'  reference: {_ref_fp}\n  offender:  {_fp}\n'
                'All attack methods and shards at a fixed dataset/seed/victim must '
                'use the identical frozen clean pool and fitted defender.')
    _checked += 1

if not _checked:
    raise SystemExit('completed partials contain no defense calibration metadata')
print(f'calibration identity OK across {_checked} completed partials in {len(_refs)} seed/victim groups')

# Zero-fallback on the exact displacement-matched control.
if 'prov_exact_match' in df.columns:
    _bad = int((~df['prov_exact_match'].fillna(True).astype(bool)).sum())
    if _bad:
        raise SystemExit(f'exact displacement-matched control fell back on {_bad} samples; protocol requires zero fallback')
    print('exact-matched control: zero fallback')

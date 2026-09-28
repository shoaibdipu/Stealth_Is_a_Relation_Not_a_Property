#!/usr/bin/env python3
import json, tempfile
from pathlib import Path
import numpy as np
from common.representation import Grid
from common.defense import DefenseBundle

rng=np.random.default_rng(123)
grid=Grid(T=16,S=4,C=2,H=4,W=4)
Xcal=rng.poisson(0.15,size=(30,16,2,4,4)).astype(np.int16)
ycal=np.asarray([i%3 for i in range(30)],int)
Xeval=rng.poisson(0.15,size=(8,16,2,4,4)).astype(np.int16)
yeval=np.asarray([i%3 for i in range(8)],int)

a=DefenseBundle(Xeval,yeval,grid,0,calibration_X=Xcal,calibration_classes=ycal)
b=DefenseBundle(Xeval,yeval,grid,0,calibration_X=Xcal,calibration_classes=ycal)
c=DefenseBundle(Xeval,yeval,grid,1,calibration_X=Xcal,calibration_classes=ycal)
assert a.calibration_sha256()==b.calibration_sha256(), 'same seed/pool must calibrate identically'
assert a.calibration_sha256()!=c.calibration_sha256(), 'different seeds should have distinct calibration identity'
s=a.calibration_summary()
assert len(s)==8 and all('fold_thresholds_json' in r for r in s)
for r in s:
    assert len(json.loads(r['fold_thresholds_json']))==5
score=a.score(Xeval[0],0,int(yeval[0]),Xclean=Xeval[0])
assert 'def_phase1_score' in score and 'def_all_det_05' in score
print('SELF-TEST OK: deterministic calibration hash, seed separation, saved thresholds, defense scoring')

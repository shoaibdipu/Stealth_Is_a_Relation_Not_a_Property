#!/usr/bin/env python3
import os, sys
from pathlib import Path
import pandas as pd
root=Path(os.environ.get('NMNIST_RUN','/path/to/workspace/EVS_NMNIST_ALIGNED5/run'))/'results_aligned5'
req=['clean_accuracy_by_seed.csv','main_attack_results_by_seed.csv','main_attack_sample_rows.csv','run_config.json']
missing=[x for x in req if not (root/x).exists()]
if missing:
    print('MISSING',missing); sys.exit(2)
clean=pd.read_csv(root/'clean_accuracy_by_seed.csv')
main=pd.read_csv(root/'main_attack_results_by_seed.csv')
samp=pd.read_csv(root/'main_attack_sample_rows.csv')
print('clean rows',len(clean),'models',sorted(clean.model.unique()),'seeds',sorted(clean.seed.unique()))
print('main rows',len(main),'sample rows',len(samp))
assert set(clean.seed)=={0,1,2}
assert set(main.seed)=={0,1,2}
assert set(main.model)=={'conv_snn','sew_resnet18','event_transformer_v2','temporal_gru'}
assert set(round(x,2) for x in main.budget)=={0.05,0.10,0.20,0.30}
assert samp['exact_representation_equal'].all()
assert (samp['D_A']==0).all()
assert (samp['D_inf']==0).all()
print('PASS: aligned5 N-MNIST outputs structurally complete; all main null-space rows have D_A=D_inf=0')

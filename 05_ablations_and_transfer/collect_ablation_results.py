#!/usr/bin/env python3
from pathlib import Path
import os
import pandas as pd
from common.datasets import SUITE

def main():
    out=Path(os.getenv('ABLATION_COLLECT_OUT','ablation_combined_2day')); out.mkdir(parents=True,exist_ok=True)
    sums=[]; rows=[]; transfers=[]
    for spec in SUITE.values():
        root=Path(os.getenv(f'{spec.key.upper()}_RUN_ROOT', os.getenv('RUN_ROOT', spec.default_run_root)))
        rdir=root/'results'
        if not rdir.exists(): continue
        for d in rdir.glob(f'ablations_{spec.key}_*'):
            s=d/'ablation_summary.csv'; p=d/'per_sample_ablation_rows.csv'; t=d/'transfer_matrix.csv'
            if s.exists():
                x=pd.read_csv(s); x['source_dir']=str(d); sums.append(x)
            if p.exists():
                x=pd.read_csv(p); x['source_dir']=str(d); rows.append(x)
            if t.exists():
                x=pd.read_csv(t); x['dataset']=spec.key; x['source_dir']=str(d); transfers.append(x)
    if sums: pd.concat(sums,ignore_index=True,sort=False).to_csv(out/'all_ablation_summaries.csv',index=False)
    if rows: pd.concat(rows,ignore_index=True,sort=False).to_csv(out/'all_ablation_rows.csv',index=False)
    if transfers: pd.concat(transfers,ignore_index=True,sort=False).to_csv(out/'all_transfer_matrices.csv',index=False)
    print('wrote',out,'summaries',len(sums),'rowsets',len(rows),'transfers',len(transfers))
if __name__=='__main__': main()

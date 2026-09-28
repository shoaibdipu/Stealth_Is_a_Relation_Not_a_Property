#!/usr/bin/env python3
"""Portable multi-GPU launcher for one dataset. One worker per listed GPU.

Example:
  python launch_local_parallel.py --dataset dvsgesture --run-root /run \
    --clean-pool /clean.csv --direct-n 264 --gpus 0,1,2,3
"""
import argparse, os, subprocess, time, sys
from collections import deque
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--dataset',required=True,choices=['dvsgesture','dailydvs200','cifar10dvs'])
p.add_argument('--run-root',required=True)
p.add_argument('--clean-pool',required=True)
p.add_argument('--direct-n',type=int,default=0)
p.add_argument('--gpus',default=os.environ.get('GPU_LIST','0'))
p.add_argument('--campaign-id',default=os.environ.get('CAMPAIGN_ID',''))
p.add_argument('--yu-shards',type=int,default=0); p.add_argument('--sda-shards',type=int,default=0); p.add_argument('--yao-shards',type=int,default=0)
p.add_argument('--cheap-shards',type=int,default=2); p.add_argument('--sweep-shards',type=int,default=2)
a=p.parse_args()
if not a.campaign_id:
    a.campaign_id='stealth_asr_defense_'+time.strftime('%Y%m%d_%H%M%S')
defs={'dvsgesture':(4,6,8,264),'dailydvs200':(3,6,8,250),'cifar10dvs':(2,5,6,150)}
dyu,dsda,dyao,dn=defs[a.dataset]
N=a.direct_n or dn; ys=a.yu_shards or dyu; ss=a.sda_shards or dsda; gs=a.yao_shards or dyao
gpus=[x.strip() for x in a.gpus.split(',') if x.strip()]
if not gpus: raise SystemExit('no GPUs specified')
root=Path(__file__).resolve().parent
subprocess.run([sys.executable,str(root/'preflight.py'),'--dataset',a.dataset,'--run-root',a.run_root,'--clean-pool',a.clean_pool,'--direct-n',str(N)],check=True)

jobs=[]
def add(method,seed,nsh):
    for i in range(nsh): jobs.append((method,str(seed),i,nsh))
add('yu_pil_l0',0,ys); add('pdsg_sda',0,ss); add('yao_gumbel',0,gs)
add('free_retiming',0,a.cheap_shards); add('null_space',0,a.cheap_shards)
for seed in (1,2):
    add('free_retiming',seed,a.cheap_shards); add('null_space',seed,a.cheap_shards)
add('free_da_sweep',0,a.sweep_shards)
q=deque(jobs); running={}; failed=[]
base=os.environ.copy(); base.update(DATASET=a.dataset,RUN_ROOT=a.run_root,DEFENSE_CLEAN_POOL_MANIFEST=a.clean_pool,DIRECT_N=str(N),CAMPAIGN_ID=a.campaign_id)
while q or running:
    for gpu in gpus:
        if gpu in running or not q: continue
        method,seed,idx,cnt=q.popleft(); env=base.copy(); env.update(METHOD=method,RUN_SEEDS=seed,SHARD_INDEX=str(idx),SHARD_COUNT=str(cnt),CUDA_VISIBLE_DEVICES=gpu)
        logdir=root/'logs'; logdir.mkdir(exist_ok=True); log=logdir/f'local_{a.dataset}_{method}_s{seed}_sh{idx}of{cnt}_gpu{gpu}.log'
        fh=open(log,'w'); proc=subprocess.Popen([str(root/'run_one_shard.sh')],env=env,stdout=fh,stderr=subprocess.STDOUT,cwd=root)
        running[gpu]=(proc,fh,(method,seed,idx,cnt),log); print(f'LAUNCH gpu={gpu} {method} seed={seed} shard={idx}/{cnt} pid={proc.pid}')
    time.sleep(2)
    for gpu,(proc,fh,meta,log) in list(running.items()):
        rc=proc.poll()
        if rc is None: continue
        fh.close(); del running[gpu]
        if rc: failed.append((meta,rc,str(log))); print(f'FAIL gpu={gpu} {meta} rc={rc} log={log}',file=sys.stderr)
        else: print(f'DONE gpu={gpu} {meta}')
    if failed:
        for proc,fh,meta,log in running.values(): proc.terminate(); fh.close()
        raise SystemExit(f'{len(failed)} shard(s) failed; first={failed[0]}')
subprocess.run([sys.executable,str(root/'aggregate_campaign.py'),'--dataset',a.dataset],env=base,cwd=root,check=True)
subprocess.run([sys.executable,str(root/'validate_campaign.py'),'--dataset',a.dataset,'--run-root',a.run_root,'--campaign-id',a.campaign_id],env=base,cwd=root,check=True)
print('CAMPAIGN COMPLETE AND VALIDATED')

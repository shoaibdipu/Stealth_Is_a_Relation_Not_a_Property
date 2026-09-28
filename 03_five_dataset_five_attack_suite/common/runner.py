import os,sys,glob,json,hashlib,time,math
from pathlib import Path
import numpy as np,pandas as pd,torch
from .datasets import SUITE
from .representation import Grid
from .frozen_loader import merge_definitions
from .attacks.retiming import null_attack,free_attack
from .attacks.controls import uniform_control,exact_matched
from .attacks import yu,sda,yao
from . import metrics
METHODS=['yu_pil_l0','pdsg_sda','yao_gumbel','free_retiming','null_space']

def _device():
    # FIX 10: the landed DVS direct comparison ran with these flags; keep them so a
    # reconciliation run differs from it by as little as possible.
    torch.backends.cudnn.benchmark=os.getenv('CUDNN_BENCHMARK','1')=='1'
    torch.backends.cudnn.deterministic=os.getenv('CUDNN_DETERMINISTIC','0')=='1'
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def _git_commit(p):
    import subprocess
    try:
        return subprocess.run(['git','-C',str(p),'rev-parse','HEAD'],capture_output=True,text=True,check=False).stdout.strip() or 'unknown'
    except Exception:return 'unknown'
def _sha(p):
    h=hashlib.sha256();
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

_MANIFEST_USED={}
def _landed_clean_accuracy(root,role,seed):
    """Best-effort lookup of the landed full-test clean accuracy for one role/seed."""
    for f in sorted(root.glob('results*/clean_accuracy_by_seed.csv'))+sorted(root.glob('results*/addon_clean_accuracy_by_seed.csv')):
        try:d=pd.read_csv(f)
        except Exception:continue
        mc=next((c for c in d.columns if c.lower() in ('model','victim','name')),None)
        ac=next((c for c in d.columns if 'acc' in c.lower()),None)
        sc=next((c for c in d.columns if c.lower()=='seed'),None)
        if not(mc and ac and sc):continue
        m=d[(d[mc].astype(str)==role)&(d[sc].astype(int)==int(seed))]
        if len(m):return float(m.iloc[0][ac]),str(f)
    return None,None

def _check_clean_parity(root,role,seed,measured,n,tol_pp):
    """FIX 4: the subset clean accuracy must reproduce the landed full-test value.

    The attack subset is a sample of the test split, so the allowance is the
    protocol tolerance plus a 95% sampling band; a wrong checkpoint misses by
    far more than that. Set CLEAN_PARITY=0 to record without enforcing."""
    landed,src=_landed_clean_accuracy(root,role,seed)
    if landed is None:
        print(f'clean-parity: no landed value for {role} seed{seed} (skipped)');return None
    band=1.959963984540054*math.sqrt(max(measured*(1-measured),1e-9)/max(n,1))
    tol=tol_pp/100.0+band;d=abs(measured-landed)
    msg=f'clean-parity {role} seed{seed}: subset={measured:.4f} landed={landed:.4f} diff={100*d:.3f}pp tol={100*tol:.3f}pp src={Path(src).name}'
    if d>tol:
        if os.getenv('CLEAN_PARITY','1')=='1':raise SystemExit('FAIL '+msg)
        print('WARN '+msg)
    else:print('OK '+msg)
    return {'role':role,'seed':int(seed),'subset_accuracy':float(measured),'landed_accuracy':float(landed),
            'abs_diff_pct_points':float(100*d),'tolerance_pct_points':float(100*tol),'source':Path(src).name}

def _find_ckpt(root,role,seed):
    c=[]
    for p in root.glob('checkpoints*/**/*.pt'):
        n=p.name.lower()
        if role.lower() in n and (f'seed{seed}' in n or f'_{seed}_' in n):c.append(p)
    if not c:
        for p in root.glob('**/*.pt'):
            n=p.name.lower()
            if role.lower() in n and f'seed{seed}' in n:c.append(p)
    if not c:raise FileNotFoundError(f'checkpoint not found: {role} seed{seed} under {root}')
    c=sorted(c,key=lambda p:(len(p.name),str(p)))
    # FIX 3: never silently pick among several generations of checkpoints.
    # CKPT_<ROLE>_SEED<N> pins one explicitly; CKPT_ALLOW_AMBIGUOUS=1 restores the old behaviour.
    pin=os.getenv(f'CKPT_{role.upper()}_SEED{seed}','')
    if pin:
        q=Path(pin)
        if not q.exists():raise FileNotFoundError(f'pinned checkpoint missing: {q}')
        return q
    if len(c)>1 and os.getenv('CKPT_ALLOW_AMBIGUOUS','0')!='1':
        raise SystemExit('ambiguous checkpoint for %s seed%d (%d candidates):\n  %s\nPin one with CKPT_%s_SEED%d=/path/to.pt'
                         %(role,seed,len(c),'\n  '.join(str(x) for x in c),role.upper(),seed))
    return c[0]
def _model(ns,cls,ckpt,dev):
    m=ns[cls]().to(dev).eval();st=torch.load(ckpt,map_location=dev);st=st.get('model',st) if isinstance(st,dict) else st;m.load_state_dict(st,strict=True);return m
def _logits(m,dev,batch):
    def f(X):
        out=[]
        with torch.no_grad():
            for i in range(0,len(X),batch):
                xb=torch.from_numpy(X[i:i+batch].astype(np.float32)).to(dev)
                with torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=(dev.type=='cuda')):z=m(xb)
                out.append(z.float().cpu().numpy())
        return np.concatenate(out)
    return f
def _subset(spec,root,ns):
    if spec.nmnist_special:
        # Prefer a landed manifest if one exists; otherwise rebuild the frozen balanced 100/class subset.
        mans=sorted(root.glob('results*/attack_subset_manifest.csv'))
        if not mans:
            test_dirs=list(root.glob('**/Test_extracted'))
            if not test_dirs:raise FileNotFoundError('N-MNIST: no attack manifest and no Test_extracted')
            samples=ns['discover_samples'](test_dirs[0]);sub=ns['stratified_select'](samples,per_class=100,seed=2027);X=[];y=[];paths=[]
            for p,l in sub:X.append(ns['events_to_tensor'](ns['read_nmnist_bin'](p)).astype(np.int16));y.append(int(l));paths.append(str(p))
            globals()['_MANIFEST_USED']={'path':'rebuilt:stratified_select(per_class=100,seed=2027)',
                                          'sha256':hashlib.sha256('\n'.join(paths).encode()).hexdigest(),
                                          'n':len(paths),'source_dir':test_dirs[0].name}
            print('attack subset: rebuilt',len(paths),'samples',_MANIFEST_USED['sha256'][:16])
            return np.stack(X),np.asarray(y),np.full(len(y),300000,np.int64),pd.DataFrame({'path':paths,'label':y})
    mp=os.getenv('ATTACK_MANIFEST','')
    if mp:
        man=Path(mp)
        if not man.exists():raise FileNotFoundError(f'ATTACK_MANIFEST missing: {man}')
    else:
        mans=sorted(root.glob('results*/attack_subset_manifest.csv'))
        if not mans:raise FileNotFoundError(f'attack_subset_manifest.csv not found under {root}/results*')
        if len(mans)>1 and os.getenv('MANIFEST_ALLOW_AMBIGUOUS','0')!='1':
            raise SystemExit('ambiguous attack manifest (%d found):\n  %s\nPin one with ATTACK_MANIFEST=/path/to.csv'
                             %(len(mans),'\n  '.join(str(x) for x in mans)))
        man=mans[0]
    globals()['_MANIFEST_USED']={'path':man.name,'sha256':_sha(man)}
    print('attack manifest:',man,_MANIFEST_USED['sha256'][:16])
    df=pd.read_csv(man);X=[];y=[];dur=[]
    for _,r in df.iterrows():
        p=Path(str(r['path']));d=np.load(p);X.append(d['X'].astype(np.int16));y.append(int(d['label']) if 'label' in d.files else int(r['label']));dur.append(int(d['duration_us']) if 'duration_us' in d.files else 0)
    return np.stack(X),np.asarray(y),np.asarray(dur),df
def _protocol(spec,root,method,seeds,victims,repos):
    all_method_cfg={"yu":{"steps":os.getenv('PRIOR_STEPS','40'),"recal":os.getenv('PRIOR_RECALIBRATE','1')},"sda":{"k":os.getenv('SDA_K_INIT','10'),"N":os.getenv('SDA_N','500')},"yao":{"steps":os.getenv('YAO_STEPS','300'),"lr":os.getenv('YAO_LR','0.1'),"kappa":os.getenv('YAO_KAPPA','1.0')}}
    obj={"dataset":spec.key,"grid":spec.grid,"budgets":spec.budgets,"seeds":seeds,"victims":victims,"repos":repos,"all_method_cfg":all_method_cfg,"tau":metrics.TAU_GRID,"eps":metrics.EPS_GRID_MS}
    return hashlib.sha256(json.dumps(obj,sort_keys=True).encode()).hexdigest()[:16],obj
def run(dataset_key):
    spec=SUITE[dataset_key];root=Path(os.getenv('RUN_ROOT',spec.default_run_root));method=os.getenv('METHOD','null_space');
    if method not in METHODS:raise SystemExit(f'METHOD must be one of {METHODS}')
    seeds=[int(x) for x in os.getenv('RUN_SEEDS','0,1,2').split(',') if x];victims=[x for x in os.getenv('VICTIMS',','.join(spec.victim_classes)).split(',') if x];dev=_device();ns=merge_definitions(spec.frozen_scripts,spec.overrides);grid=Grid(*spec.grid);repos={"yu":os.getenv('SPIKE_RETIMING_REPO',''),"sda":os.getenv('PDSG_SDA_DIR',''),"yao":os.getenv('YAO_DIR','')};
    repo_prov={}
    for rk,rp in repos.items():
      q=Path(rp) if rp else None; repo_prov[rk]={"configured":bool(rp),"commit":_git_commit(q) if q and q.exists() else 'missing'}
    repos_for_hash={k:v for k,v in repo_prov.items()}
    pid,pobj=_protocol(spec,root,method,seeds,victims,repos_for_hash);out=root/'results'/f'five_attack_{spec.key}_{pid}';part=out/'partials'/method;part.mkdir(parents=True,exist_ok=True);(out/'protocol.json').write_text(json.dumps(pobj,indent=2))
    X,y,dur,man=_subset(spec,root,ns);limit=int(os.getenv('DIRECT_N','0')); 
    if limit>0:X,y,dur=X[:limit],y[:limit],dur[:limit]
    ext=None
    if method=='yu_pil_l0':ext=yu.load(repos['yu'])
    elif method=='pdsg_sda':ext=sda.load(repos['sda'])
    elif method=='yao_gumbel':ext=yao.load(repos['yao'])
    for seed in seeds:
      for victim_role in victims:
        fp=part/f'seed{seed}_{victim_role}.json'
        if fp.exists():print('reuse',fp);continue
        vck=_find_ckpt(root,victim_role,seed);cck=_find_ckpt(root,spec.consumer_role,seed);victim=_model(ns,spec.victim_classes[victim_role],vck,dev);consumer=_model(ns,spec.consumer_class,cck,dev);batch=int(spec.overrides.get('EVAL_BATCH',32));vfn=_logits(victim,dev,batch);cfn=_logits(consumer,dev,batch);clean=vfn(X).argmax(1);c0=cfn(X);rows=[];rng=np.random.default_rng(9090+seed)
        parity=[p for p in (_check_clean_parity(root,victim_role,seed,float((clean==y).mean()),len(y),spec.clean_tol_pp),
                            _check_clean_parity(root,spec.consumer_role,seed,float((c0.argmax(1)==y).mean()),len(y),spec.clean_tol_pp)) if p]
        for i in range(len(X)):
          if clean[i]!=y[i]:continue
          bin_ms=(dur[i]/grid.T/1000.) if dur[i]>0 else spec.fallback_bin_ms;results={}
          if method=='null_space':results=null_attack(victim,grid,X[i],int(y[i]),spec.budgets,dev)
          elif method=='free_retiming':results=free_attack(victim,grid,X[i],int(y[i]),spec.budgets,dev)
          elif method=='yu_pil_l0':
            b=float(os.getenv('NATIVE_COMPARE_BUDGET','0.10'));Xa,sh,p=yu.attack(victim,X[i],int(y[i]),b,dev,ext);results={b:{'X':Xa,'shifts':sh,'prov':p}}
          elif method=='pdsg_sda':Xa,p=sda.attack(ext,victim,X[i],int(y[i]),dev);results={'native':{'X':Xa,'shifts':None,'prov':p}}
          elif method=='yao_gumbel':Xa,p=yao.attack(ext,victim,X[i],int(y[i]),dev,spec.n_classes);results={'native':{'X':Xa,'shifts':None,'prov':p}}
          for b,res in results.items():
            r=metrics.row(grid,X[i],res['X'],int(y[i]),int(clean[i]),vfn,c0[i:i+1],cfn,res.get('shifts'),bin_ms,res.get('prov'));r.update(dataset=spec.key,seed=seed,victim=victim_role,method=method,sample_index=i,requested_budget=(float(b) if b!='native' else np.nan),budget_label=(f'{100*float(b):g}%' if b!='native' else 'native'));rows.append(r)
            if method=='null_space':
              n=len(res.get('shifts',[]));Xu,su=uniform_control(grid,X[i],n,rng);ru=metrics.row(grid,X[i],Xu,int(y[i]),int(clean[i]),vfn,c0[i:i+1],cfn,su,bin_ms);ru.update(dataset=spec.key,seed=seed,victim=victim_role,method='random_uniform',sample_index=i,requested_budget=float(b),budget_label=f'{100*float(b):g}%');rows.append(ru)
              Xm,sm,ok=exact_matched(grid,X[i],res.get('shifts',[]),rng);rm=metrics.row(grid,X[i],Xm,int(y[i]),int(clean[i]),vfn,c0[i:i+1],cfn,sm,bin_ms,{'exact_match':ok});rm.update(dataset=spec.key,seed=seed,victim=victim_role,method='matched_exact',sample_index=i,requested_budget=float(b),budget_label=f'{100*float(b):g}%');rows.append(rm)
          if method=='null_space':
            for r in rows[-3*len(results):]:
              if r['method']=='null_space' and (r['D_A']!=0 or r['D_inf']!=0 or r['protected_consumer_flip']):raise AssertionError('null-space invariant violated')
        tmp=fp.with_suffix('.tmp');tmp.write_text(json.dumps({'dataset':spec.key,'method':method,'seed':seed,'victim':victim_role,'victim_ckpt':vck.name,'victim_sha':_sha(vck),'consumer_ckpt':cck.name,'consumer_sha':_sha(cck),'manifest':_MANIFEST_USED,'clean_parity':parity,'rows':rows},allow_nan=True));os.replace(tmp,fp);print('wrote',fp,len(rows))
    if os.getenv('PARTIAL_ONLY','0')!='1':aggregate(dataset_key,out,pid)
def aggregate(dataset_key,out=None,pid=None):
    spec=SUITE[dataset_key];root=Path(os.getenv('RUN_ROOT',spec.default_run_root));
    if out is None:
      candidates=sorted((root/'results').glob(f'five_attack_{spec.key}_*'),key=lambda p:p.stat().st_mtime)
      if not candidates:raise SystemExit('no five-attack result directory')
      out=candidates[-1]
    rows=[]
    for p in (out/'partials').glob('*/*.json'):rows+=json.loads(p.read_text())['rows']
    df=pd.DataFrame(rows);df.to_csv(out/'per_sample_rows.csv',index=False);metrics.extended(df,out)
    cc=df[df.clean_correct.astype(bool)].copy();summary=[]
    for (v,m,s,b),g in cc.groupby(['victim','method','seed','budget_label'],dropna=False):
      succ=g.attack_success_on_clean_correct.astype(bool);summary.append(dict(victim=v,method=m,seed=s,budget=b,n_clean_correct=len(g),ASR=float(succ.mean()),mean_D_A=float(g.D_A.mean()),mean_D_inf=float(g.D_inf.mean()),mean_footprint_pct=float(g.realized_footprint_pct.mean()),count_preserving=float(g.count_preserving.mean()),protected_flip_rate=float(g.protected_consumer_flip.mean()),mean_shift_ms=float(g.avg_abs_shift_ms.mean()) if g.avg_abs_shift_ms.notna().any() else np.nan))
    pd.DataFrame(summary).to_csv(out/'headline_by_seed.csv',index=False);print('aggregated',out)

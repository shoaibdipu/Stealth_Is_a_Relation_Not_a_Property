import os, sys, json, hashlib, math, random, re
from pathlib import Path
import numpy as np, pandas as pd, torch
from .datasets import SUITE
from .representation import Grid
from .frozen_loader import merge_definitions
from .attacks.retiming import null_attack, free_attack
from .attacks.controls import uniform_control, exact_matched
from .attacks import yu, sda, yao
from . import metrics, defense

METHODS=['yu_pil_l0','pdsg_sda','yao_gumbel','free_retiming','null_space','free_da_sweep']
MAIN_METHODS=['yu_pil_l0','pdsg_sda','yao_gumbel','free_retiming','null_space']


def _device():
    torch.backends.cudnn.benchmark=os.getenv('CUDNN_BENCHMARK','1')=='1'
    torch.backends.cudnn.deterministic=os.getenv('CUDNN_DETERMINISTIC','0')=='1'
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _git_commit(p):
    import subprocess
    try:
        return subprocess.run(['git','-C',str(p),'rev-parse','HEAD'],capture_output=True,text=True,check=False).stdout.strip() or 'unknown'
    except Exception:
        return 'unknown'


def _sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):
            h.update(b)
    return h.hexdigest()


def _safe_name(x):
    return re.sub(r'[^A-Za-z0-9_.-]+','_',str(x))


def _sample_seed(dataset,method,train_seed,sample_uid,extra=''):
    z=f'{dataset}|{method}|{train_seed}|{sample_uid}|{extra}'.encode()
    return int(hashlib.sha256(z).hexdigest()[:8],16) & 0x7fffffff


def _seed_everything(seed):
    random.seed(int(seed)); np.random.seed(int(seed)%(2**32-1)); torch.manual_seed(int(seed))
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(int(seed))


_MANIFEST_USED={}


def _landed_clean_accuracy(root,role,seed):
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
            'abs_diff_pct_points':float(100*d),'tolerance_pct_points':float(100*tol),'source':src}


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


def _population_manifest_indices(full):
    direct_n=int(os.getenv('DIRECT_N','0') or 0)
    if direct_n<=0 or direct_n>=len(full):
        return np.arange(len(full),dtype=int)
    mode=os.getenv('EVAL_SELECTION','stratified').strip().lower()
    if mode=='stratified' and 'label' in full.columns:
        rng=np.random.default_rng(int(os.getenv('POPULATION_SEED','2027')))
        groups={}
        for c,g in full.groupby('label'):
            arr=g.index.to_numpy(dtype=int).copy(); rng.shuffle(arr); groups[str(c)]=arr
        classes=sorted(groups); ptr={c:0 for c in classes}; chosen=[]
        while len(chosen)<direct_n:
            progressed=False
            for c in classes:
                arr=groups[c]; j=ptr[c]
                if j<len(arr) and len(chosen)<direct_n:
                    chosen.append(int(arr[j])); ptr[c]+=1; progressed=True
            if not progressed:break
        if len(chosen)!=direct_n:raise RuntimeError(f'could select only {len(chosen)} of requested {direct_n}')
        return np.asarray(chosen,dtype=int)
    if mode not in ('first','stratified'):
        raise ValueError(f'EVAL_SELECTION must be stratified or first, got {mode}')
    return np.arange(direct_n,dtype=int)


def _shard_positions(n_eval):
    base=np.arange(n_eval,dtype=int)
    ss=os.getenv('SAMPLE_START','').strip(); se=os.getenv('SAMPLE_END','').strip()
    if ss or se:
        a=int(ss or 0); b=int(se or n_eval)
        if not (0<=a<=b<=n_eval):raise ValueError(f'invalid SAMPLE_START/END {a}:{b} for n_eval={n_eval}')
        chosen=base[a:b]; tag=f'range{a:05d}-{b:05d}'
    else:
        cnt=int(os.getenv('SHARD_COUNT','1') or 1); idx=int(os.getenv('SHARD_INDEX','0') or 0)
        if cnt<1 or idx<0 or idx>=cnt:raise ValueError(f'invalid shard {idx}/{cnt}')
        chunks=np.array_split(base,cnt); chosen=np.asarray(chunks[idx],dtype=int); tag=f'sh{idx:03d}of{cnt:03d}' if cnt>1 else 'full'
    return chosen,tag

def _subset(spec,root,ns):
    if spec.nmnist_special:
        raise RuntimeError('This three-dataset package is intended for DVS Gesture, DailyDVS-200 and CIFAR10-DVS only.')
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
    full=pd.read_csv(man).reset_index(drop=True)
    pop_idx=_population_manifest_indices(full); n_eval=len(pop_idx); chosen,tag=_shard_positions(n_eval)
    eval_df=full.iloc[pop_idx].copy().reset_index().rename(columns={'index':'_manifest_index'})
    eval_df['_eval_index']=np.arange(n_eval,dtype=int)
    pop_paths='\n'.join(eval_df['path'].astype(str).tolist()) if 'path' in eval_df.columns else '\n'.join(map(str,pop_idx.tolist()))
    pop_sha=hashlib.sha256(pop_paths.encode()).hexdigest()
    df=eval_df.iloc[chosen].copy()
    msha=_sha(man)
    uids=[]
    for _,r in df.iterrows():
        p=str(r['path']) if 'path' in r else ''
        raw=f'{msha}|{int(r._manifest_index)}|{p}'.encode(); uids.append(hashlib.sha256(raw).hexdigest()[:24])
    df['_sample_uid']=uids
    globals()['_EVAL_POPULATION_PATHS']=set(eval_df['path'].astype(str).tolist()) if 'path' in eval_df.columns else set()
    globals()['_MANIFEST_USED']={'path':man.name,'sha256':msha,'n_manifest':len(full),'n_eval_population':n_eval,
                                  'eval_population_sha256':pop_sha,'shard_tag':tag,'n_in_shard':len(df),
                                  'eval_indices':[int(x) for x in chosen.tolist()]}
    print('attack manifest:',man,msha[:16],'population',n_eval,'shard',tag,'n',len(df))
    X=[];y=[];dur=[]
    for _,r in df.iterrows():
        p=Path(str(r['path']));d=np.load(p);X.append(d['X'].astype(np.int16));y.append(int(d['label']) if 'label' in d.files else int(r['label']));dur.append(int(d['duration_us']) if 'duration_us' in d.files else 0)
    if not X: raise RuntimeError(f'empty shard {tag}')
    return np.stack(X),np.asarray(y),np.asarray(dur),df


def _load_clean_pool_manifest(path,eval_population_paths=None,limit=0):
    """Load the frozen clean calibration pool.

    The pool MUST be disjoint from the full evaluation population. Disjointness
    is asserted against the whole population (not the current shard) and any
    overlap is fatal: silently filtering overlaps would make the surviving pool
    depend on which shard is running, so every shard would fit a different
    detector and per-shard TPR/UASR would not be comparable. With the assertion
    in place no exclusion filtering happens at all, so the pool -- and therefore
    the fitted detector and its thresholds -- is identical in every shard.
    """
    p=Path(path)
    if not p.exists(): raise FileNotFoundError(f'DEFENSE_CLEAN_POOL_MANIFEST missing: {p}')
    df=pd.read_csv(p)
    if 'path' not in df.columns: raise SystemExit(f'clean pool manifest has no "path" column: {p}')
    pool_paths=df['path'].astype(str).tolist()
    pop=set(eval_population_paths or ())
    overlap=sorted(set(pool_paths)&pop)
    if overlap:
        raise SystemExit(
            f'clean calibration pool overlaps the evaluation population on {len(overlap)} clips '
            f'(e.g. {overlap[:3]}). The defense protocol requires a disjoint pool; '
            f'rebuild it with build_clean_pool_manifest.py excluding the attack manifest.')
    X=[]; y=[]; paths=[]
    for _,r in df.iterrows():
        rp=str(r['path'])
        d=np.load(rp); X.append(d['X'].astype(np.int16)); y.append(int(d['label']) if 'label' in d.files else int(r['label'])); paths.append(rp)
        if limit and len(X)>=limit: break
    if len(X)<20: raise RuntimeError(f'external defense clean pool too small: {len(X)}')
    # Fingerprint the exact pool actually loaded, so the merger can assert that
    # every shard calibrated on byte-identical data.
    pool_sha=hashlib.sha256('\n'.join(paths).encode()).hexdigest()
    return np.stack(X),np.asarray(y),{'path':str(p),'sha256':_sha(p),'n':len(X),
                                      'loaded_pool_sha256':pool_sha,'limit':int(limit)}


def _protocol(spec,seeds,victims,repos):
    cp=os.getenv('DEFENSE_CLEAN_POOL_MANIFEST','').strip(); cpsha=_sha(Path(cp)) if cp and Path(cp).exists() else ('missing' if cp else '')
    return {
        'dataset':spec.key,'grid':spec.grid,'budgets':spec.budgets,'seeds':seeds,'victims':victims,'repos':repos,
        'yu':{'steps':os.getenv('PRIOR_STEPS','40'),'recal':os.getenv('PRIOR_RECALIBRATE','1'),'compare_budget':os.getenv('NATIVE_COMPARE_BUDGET','0.10')},
        'sda':{'k':os.getenv('SDA_K_INIT','10'),'N':os.getenv('SDA_N','500')},
        'yao':{'steps':os.getenv('YAO_STEPS','300'),'lr':os.getenv('YAO_LR','0.1'),'kappa':os.getenv('YAO_KAPPA','1.0')},
        'tau':metrics.TAU_GRID,'eps':metrics.EPS_GRID_MS,'direct_n':int(os.getenv('DIRECT_N','0') or 0),
        'defense_enabled':os.getenv('DEFENSE_ANALYSIS','1')=='1','defense_version':defense.DEFENSE_VERSION,
        'defense_clean_pool_manifest':Path(cp).name if cp else '','defense_clean_pool_sha256':cpsha,
        'free_da_sweep_max_shifts':os.getenv('FREE_SWEEP_MAX_SHIFTS','2,4,8,16,32'),
        'free_da_sweep_budget':os.getenv('FREE_SWEEP_BUDGET','0.10'),
        'manifest_sha256':_MANIFEST_USED.get('sha256','unknown'),'eval_population_sha256':_MANIFEST_USED.get('eval_population_sha256','unknown'),
        'n_eval_population':_MANIFEST_USED.get('n_eval_population',None),'eval_selection':os.getenv('EVAL_SELECTION','stratified'),'population_seed':int(os.getenv('POPULATION_SEED','2027'))
    }


def _campaign_out(root,spec,pobj):
    cid=os.getenv('CAMPAIGN_ID','').strip()
    if cid:
        return root/'results'/f'combined_attack_defense_{spec.key}_{_safe_name(cid)}'
    pid=hashlib.sha256(json.dumps(pobj,sort_keys=True).encode()).hexdigest()[:16]
    return root/'results'/f'combined_attack_defense_{spec.key}_{pid}'


def _save_sparse_perturbation(out,method,seed,victim,budget_label,sample_uid,X0,Xa,shifts=None,extra=None):
    if os.getenv('SAVE_PERTURBATIONS','1')!='1':return None,None
    d=(Xa.astype(np.int32)-X0.astype(np.int32)).ravel(); idx=np.flatnonzero(d); vals=d[idx].astype(np.int32)
    p=out/'perturbations'/method/f'seed{seed}'/victim/f'{_safe_name(budget_label)}_{sample_uid}.npz';p.parent.mkdir(parents=True,exist_ok=True)
    payload={'flat_idx':idx.astype(np.int64),'delta':vals,'shape':np.asarray(X0.shape,dtype=np.int32)}
    if shifts is not None:payload['shifts']=np.asarray(shifts,dtype=np.int16)
    if extra:payload['meta_json']=np.asarray(json.dumps(extra,sort_keys=True))
    tmp=p.with_suffix('.tmp.npz');np.savez_compressed(tmp,**payload);os.replace(tmp,p)
    return str(p),_sha(p)


def _score_control(defense_bundle,cfn,Xclean,Xa,sample_index):
    if defense_bundle is None:return {}
    observed=int(cfn(Xa[None]).argmax(1)[0]);return defense_bundle.score(Xa,sample_index,observed,Xclean=Xclean)


def run(dataset_key):
    if dataset_key not in ('dvsgesture','dailydvs200','cifar10dvs'):
        raise SystemExit('This package is frozen for dvsgesture, dailydvs200 and cifar10dvs.')
    spec=SUITE[dataset_key]
    rr=os.getenv('RUN_ROOT','').strip()
    if not rr:
        raise SystemExit('RUN_ROOT is required; this portable package never falls back to cluster-specific paths')
    root=Path(rr)
    method=os.getenv('METHOD','null_space')
    if method not in METHODS:raise SystemExit(f'METHOD must be one of {METHODS}')
    seeds=[int(x) for x in os.getenv('RUN_SEEDS','0').split(',') if x.strip()]
    victims=[x for x in os.getenv('VICTIMS','conv_snn').split(',') if x.strip()]
    dev=_device();ns=merge_definitions(spec.frozen_scripts,spec.overrides);grid=Grid(*spec.grid)
    repos={'yu':os.getenv('SPIKE_RETIMING_REPO',''),'sda':os.getenv('PDSG_SDA_DIR',''),'yao':os.getenv('YAO_DIR','')}
    repo_prov={}
    for rk,rp in repos.items():
        q=Path(rp) if rp else None; repo_prov[rk]={'configured':bool(rp),'commit':_git_commit(q) if q and q.exists() else 'missing'}
    X,y,dur,man=_subset(spec,root,ns)
    pobj=_protocol(spec,seeds,victims,repo_prov)
    unit_protocol_sha=hashlib.sha256(
        json.dumps(pobj|{'method':method},sort_keys=True,separators=(',',':')).encode()
    ).hexdigest()
    out=_campaign_out(root,spec,pobj);out.mkdir(parents=True,exist_ok=True)
    shard_tag=_MANIFEST_USED['shard_tag']; part=out/'partials'/method;part.mkdir(parents=True,exist_ok=True)
    prot_dir=out/'protocols';prot_dir.mkdir(exist_ok=True)
    prot_path=prot_dir/f'{method}_seeds-{"-".join(map(str,seeds))}_{shard_tag}.json'; prot_path.write_text(json.dumps(pobj|{'shard':_MANIFEST_USED},indent=2))

    if os.getenv('DEFENSE_ANALYSIS','1')=='1' and os.getenv('REQUIRE_DISJOINT_CLEAN_POOL','1')=='1' and not os.getenv('DEFENSE_CLEAN_POOL_MANIFEST','').strip():
        raise SystemExit('DEFENSE_CLEAN_POOL_MANIFEST is required for production defense runs. Use REQUIRE_DISJOINT_CLEAN_POOL=0 only for smoke/debug.')

    ext=None
    if method=='yu_pil_l0':ext=yu.load(repos['yu'])
    elif method=='pdsg_sda':ext=sda.load(repos['sda'])
    elif method=='yao_gumbel':ext=yao.load(repos['yao'])

    for seed in seeds:
      for victim_role in victims:
        fp=part/f'seed{seed}_{victim_role}_{shard_tag}.json'
        vck=_find_ckpt(root,victim_role,seed);cck=_find_ckpt(root,spec.consumer_role,seed);vsha=_sha(vck);csha=_sha(cck)
        sample_uids=man['_sample_uid'].astype(str).tolist()
        if fp.exists():
          try:
            oldj=json.loads(fp.read_text())
            if ((oldj.get('manifest') or {}).get('eval_population_sha256')==_MANIFEST_USED.get('eval_population_sha256')
                and oldj.get('victim_sha')==vsha and oldj.get('consumer_sha')==csha
                and oldj.get('sample_uids')==sample_uids
                and oldj.get('protocol_sha256')==unit_protocol_sha
                and oldj.get('defense_version')==(defense.DEFENSE_VERSION if os.getenv('DEFENSE_ANALYSIS','1')=='1' else None)):
              print('reuse verified',fp);continue
            print('stale partial ignored',fp)
          except Exception:print('unreadable partial ignored',fp)
        victim=_model(ns,spec.victim_classes[victim_role],vck,dev);consumer=_model(ns,spec.consumer_class,cck,dev)
        batch=int(spec.overrides.get('EVAL_BATCH',32));vfn=_logits(victim,dev,batch);cfn=_logits(consumer,dev,batch);clean=vfn(X).argmax(1);c0=cfn(X);rows=[]
        defense_bundle=None;clean_pool_prov=None
        if os.getenv('DEFENSE_ANALYSIS','1')=='1':
          cp=os.getenv('DEFENSE_CLEAN_POOL_MANIFEST','').strip()
          if cp:
            Xcal,ycal,clean_pool_prov=_load_clean_pool_manifest(
                cp,globals().get('_EVAL_POPULATION_PATHS',set()),int(os.getenv('DEFENSE_CLEAN_POOL_LIMIT','0') or 0))
            ccal=cfn(Xcal).argmax(1); defense_bundle=defense.DefenseBundle(X,c0.argmax(1),grid,seed,calibration_X=Xcal,calibration_classes=ccal)
          else:
            defense_bundle=defense.DefenseBundle(X,c0.argmax(1),grid,seed)
        defense_cal_sha=defense_bundle.calibration_sha256() if defense_bundle is not None else None
        parity=[p for p in (_check_clean_parity(root,victim_role,seed,float((clean==y).mean()),len(y),spec.clean_tol_pp),
                            _check_clean_parity(root,spec.consumer_role,seed,float((c0.argmax(1)==y).mean()),len(y),spec.clean_tol_pp)) if p]

        # Periodic flush so a preempted shard keeps finished samples, and can
        # resume mid-shard from the uids already recorded.
        flush_every=max(1,int(os.getenv('FLUSH_EVERY','1')))
        done_uids=set()
        prog=fp.with_suffix('.progress.json')
        if prog.exists() and os.getenv('FORCE_RECOMPUTE','0')!='1':
          try:
            pj=json.loads(prog.read_text())
            if ((pj.get('manifest') or {}).get('eval_population_sha256')==_MANIFEST_USED.get('eval_population_sha256')
                and pj.get('victim_sha')==vsha and pj.get('consumer_sha')==csha
                and pj.get('protocol_sha256')==unit_protocol_sha
                and pj.get('defense_calibration_sha256')==defense_cal_sha
                and (pj.get('defense_clean_pool') or {}).get('loaded_pool_sha256')==((clean_pool_prov or {}).get('loaded_pool_sha256'))):
              rows=pj.get('rows',[]); done_uids={str(r.get('sample_uid')) for r in rows}
              print(f'resuming shard from progress file: {len(done_uids)} samples already done')
            else:
              print('stale progress file ignored',prog)
          except Exception:print('unreadable progress file ignored',prog)

        def _flush_progress():
            tmpp=prog.with_suffix('.tmp')
            tmpp.write_text(json.dumps({'dataset':spec.key,'method':method,'seed':seed,'victim':victim_role,
                                        'victim_sha':vsha,'consumer_sha':csha,'manifest':_MANIFEST_USED,
                                        'protocol_sha256':unit_protocol_sha,
                                        'defense_version':defense.DEFENSE_VERSION if defense_bundle is not None else None,
                                        'defense_calibration_sha256':defense_cal_sha,
                                        'defense_clean_pool':clean_pool_prov,'rows':rows},allow_nan=True))
            os.replace(tmpp,prog)

        for i in range(len(X)):
          if clean[i]!=y[i]:continue
          eval_index=int(man.iloc[i]['_eval_index']); manifest_index=int(man.iloc[i]['_manifest_index']); sample_uid=str(man.iloc[i]['_sample_uid'])
          if sample_uid in done_uids: continue
          bin_ms=(dur[i]/grid.T/1000.) if dur[i]>0 else spec.fallback_bin_ms
          base_seed=_sample_seed(spec.key,method,seed,sample_uid);_seed_everything(base_seed)
          results={}
          if method=='null_space':results=null_attack(victim,grid,X[i],int(y[i]),spec.budgets,dev)
          elif method=='free_retiming':results=free_attack(victim,grid,X[i],int(y[i]),spec.budgets,dev)
          elif method=='yu_pil_l0':
            b=float(os.getenv('NATIVE_COMPARE_BUDGET','0.10'));Xa,sh,p=yu.attack(victim,X[i],int(y[i]),b,dev,ext);results={b:{'X':Xa,'shifts':sh,'prov':p}}
          elif method=='pdsg_sda':
            Xa,p=sda.attack(ext,victim,X[i],int(y[i]),dev);results={'native':{'X':Xa,'shifts':None,'prov':p}}
          elif method=='yao_gumbel':
            Xa,p=yao.attack(ext,victim,X[i],int(y[i]),dev,spec.n_classes);results={'native':{'X':Xa,'shifts':None,'prov':p}}
          elif method=='free_da_sweep':
            b=float(os.getenv('FREE_SWEEP_BUDGET','0.10'))
            for ms in [int(x) for x in os.getenv('FREE_SWEEP_MAX_SHIFTS','2,4,8,16,32').split(',') if x.strip()]:
                _seed_everything(_sample_seed(spec.key,method,seed,sample_uid,str(ms)))
                rr=free_attack(victim,grid,X[i],int(y[i]),[b],dev,max_shift=ms)[b]
                results[f'maxshift_{ms}']=rr|{'sweep_max_shift':ms,'requested_budget':b}

          for b,res in results.items():
            is_sweep=(method=='free_da_sweep')
            budget_label=str(b) if is_sweep else (f'{100*float(b):g}%' if b!='native' else 'native')
            requested=float(res.get('requested_budget',np.nan if b=='native' else b))
            r=metrics.row(grid,X[i],res['X'],int(y[i]),int(clean[i]),vfn,c0[i:i+1],cfn,res.get('shifts'),bin_ms,res.get('prov'))
            r.update(dataset=spec.key,seed=seed,victim=victim_role,method=method,sample_index=eval_index,manifest_index=manifest_index,sample_uid=sample_uid,
                     requested_budget=requested,budget_label=budget_label,grid_T=grid.T,grid_S=grid.S)
            if is_sweep:r['sweep_max_shift']=int(res['sweep_max_shift'])
            if defense_bundle is not None:
                observed_class=int(cfn(res['X'][None]).argmax(1)[0]);r.update(defense_bundle.score(res['X'],eval_index,observed_class,Xclean=X[i]))
            ap,ash=_save_sparse_perturbation(out,method,seed,victim_role,budget_label,sample_uid,X[i],res['X'],res.get('shifts'),{'sample_seed':base_seed,'eval_index':eval_index})
            if ap:r.update(perturbation_path=ap,perturbation_sha256=ash)
            rows.append(r)
            if method=='null_space' and (r['D_A']!=0 or r['D_inf']!=0 or r['protected_consumer_flip']):raise AssertionError('null-space invariant violated')

            if method=='null_space':
              n=len(res.get('shifts',[]));rng=np.random.default_rng(_sample_seed(spec.key,'controls',seed,sample_uid,budget_label))
              Xu,su=uniform_control(grid,X[i],n,rng);ru=metrics.row(grid,X[i],Xu,int(y[i]),int(clean[i]),vfn,c0[i:i+1],cfn,su,bin_ms)
              ru.update(dataset=spec.key,seed=seed,victim=victim_role,method='random_uniform',sample_index=eval_index,manifest_index=manifest_index,sample_uid=sample_uid,requested_budget=float(b),budget_label=f'{100*float(b):g}%',grid_T=grid.T,grid_S=grid.S)
              ru.update(_score_control(defense_bundle,cfn,X[i],Xu,eval_index));rows.append(ru)
              Xm,sm,ok=exact_matched(grid,X[i],res.get('shifts',[]),rng)
              if not ok and os.getenv('ALLOW_MATCH_FALLBACK','0')!='1':
                  raise AssertionError(f'exact displacement-matched control infeasible (uid={sample_uid}, budget={budget_label}); '
                                       'the protocol requires zero fallback')
              rm=metrics.row(grid,X[i],Xm,int(y[i]),int(clean[i]),vfn,c0[i:i+1],cfn,sm,bin_ms,{'exact_match':ok})
              rm.update(dataset=spec.key,seed=seed,victim=victim_role,method='matched_exact',sample_index=eval_index,manifest_index=manifest_index,sample_uid=sample_uid,requested_budget=float(b),budget_label=f'{100*float(b):g}%',grid_T=grid.T,grid_S=grid.S)
              rm.update(_score_control(defense_bundle,cfn,X[i],Xm,eval_index));rows.append(rm)

          if (i+1)%flush_every==0: _flush_progress()

        payload={'dataset':spec.key,'method':method,'seed':seed,'victim':victim_role,'victim_ckpt':vck.name,'victim_sha':vsha,'consumer_ckpt':cck.name,'consumer_sha':csha,
                 'manifest':_MANIFEST_USED,'sample_uids':sample_uids,'clean_parity':parity,
                 'protocol_sha256':unit_protocol_sha,
                 'defense_version':defense.DEFENSE_VERSION if defense_bundle is not None else None,
                 'defense_calibration_sha256':defense_cal_sha,
                 'defense_calibration':defense_bundle.calibration_summary() if defense_bundle is not None else [],
                 'defense_clean_pool':clean_pool_prov,'rows':rows}
        tmp=fp.with_suffix('.tmp');tmp.write_text(json.dumps(payload,allow_nan=True));os.replace(tmp,fp)
        try: prog.unlink()
        except FileNotFoundError: pass
        print('wrote',fp,len(rows))
    if os.getenv('PARTIAL_ONLY','1')!='1':aggregate(dataset_key,out)


def _dedupe_or_fail(df):
    keys=[c for c in ['dataset','victim','method','seed','budget_label','sample_uid'] if c in df.columns]
    dup=df.duplicated(keys,keep=False)
    if dup.any():
        # Exact duplicate rows can arise after a harmless resubmission; conflicting duplicates are fatal.
        bad=[]
        for _,g in df[dup].groupby(keys,dropna=False):
            cols=[c for c in df.columns if c not in ['perturbation_path']]
            if len(g[cols].drop_duplicates())>1:bad.append(g[keys].iloc[0].to_dict())
        if bad:raise RuntimeError(f'conflicting duplicate rows after shard merge: {bad[:5]}')
        df=df.drop_duplicates(keys,keep='last')
    return df


def aggregate(dataset_key,out=None):
    spec=SUITE[dataset_key]
    rr=os.getenv('RUN_ROOT','').strip()
    if not rr:
        raise SystemExit('RUN_ROOT is required for aggregation')
    root=Path(rr)
    if out is None:
        cid=os.getenv('CAMPAIGN_ID','').strip()
        if cid: out=root/'results'/f'combined_attack_defense_{spec.key}_{_safe_name(cid)}'
        else:
            candidates=sorted((root/'results').glob(f'combined_attack_defense_{spec.key}_*'),key=lambda p:p.stat().st_mtime)
            if not candidates:raise SystemExit('no combined attack-defense result directory')
            out=candidates[-1]
    rows=[];cal=[]
    for p in (out/'partials').glob('*/*.json'):
        if p.name.endswith('.progress.json'):
            continue
        j=json.loads(p.read_text())
        if 'rows' not in j:
            raise RuntimeError(f'completed partial missing rows: {p}')
        rows+=j['rows']
        for r in j.get('defense_calibration',[]):cal.append(dict(dataset=j.get('dataset'),victim=j.get('victim'),seed=j.get('seed'),**r))
    if not rows:raise RuntimeError(f'no partial rows under {out}')
    df=_dedupe_or_fail(pd.DataFrame(rows));df.to_csv(out/'per_sample_rows.csv',index=False);metrics.extended(df,out)
    if any(c.startswith('def_') and c.endswith('_score') for c in df.columns):
        defense.extend_results(df,out,cal); defense.extend_sweep_results(df,out)
    cc=df[df.clean_correct.astype(bool)].copy();summary=[]
    for (v,m,s,b),g in cc.groupby(['victim','method','seed','budget_label'],dropna=False):
        succ=g.attack_success_on_clean_correct.astype(bool);summary.append(dict(victim=v,method=m,seed=s,budget=b,n_clean_correct=len(g),ASR=float(succ.mean()),mean_D_A=float(g.D_A.mean()),median_D_A=float(g.D_A.median()),mean_D_inf=float(g.D_inf.mean()),mean_edit_budget_pct=float(g.edit_budget_pct.mean()),mean_footprint_pct=float(g.realized_footprint_pct.mean()),count_preserving=float(g.count_preserving.mean()),protected_flip_rate=float(g.protected_consumer_flip.mean()),mean_shift_ms=float(g.avg_abs_shift_ms.mean()) if g.avg_abs_shift_ms.notna().any() else np.nan))
    pd.DataFrame(summary).to_csv(out/'headline_by_seed.csv',index=False)
    print('aggregated',out,'rows',len(df))

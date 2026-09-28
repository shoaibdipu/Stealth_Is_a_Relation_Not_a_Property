from __future__ import annotations
import math, hashlib, json
from dataclasses import dataclass
from typing import Dict, List
import numpy as np
import pandas as pd

DEFENSE_VERSION = "dvs-defense-v2"
FPRS = (0.01, 0.05)
DA_BINS = np.array([0.0,0.01,0.02,0.05,0.10,0.15,0.18,0.19,0.20,0.21,0.25,0.30,0.50,np.inf],dtype=float)
TAU_GRID=[0,.001,.002,.005,.01,.02,.05,.10,.15,.17,.18,.19,.195,.20,.205,.21,.22,.23,.24,.5,1.0]


def _safe_quantile(x, q):
    x=np.asarray(x,float)
    if len(x)==0:return float('nan')
    try:return float(np.quantile(x,q,method='higher'))
    except TypeError:return float(np.quantile(x,q,interpolation='higher'))


def _rank_average(x):
    return pd.Series(np.asarray(x,float)).rank(method='average').to_numpy(float)


def auroc(clean_scores, attack_scores):
    a=np.asarray(clean_scores,float); b=np.asarray(attack_scores,float)
    a=a[np.isfinite(a)]; b=b[np.isfinite(b)]
    if len(a)==0 or len(b)==0:return float('nan')
    vals=np.concatenate([a,b]); ranks=_rank_average(vals)
    n0=len(a); n1=len(b); rank_pos=ranks[n0:].sum()
    return float((rank_pos-n1*(n1+1)/2)/(n0*n1))


def spearman(x,y):
    x=np.asarray(x,float); y=np.asarray(y,float); m=np.isfinite(x)&np.isfinite(y)
    if m.sum()<3:return float('nan')
    rx=_rank_average(x[m]); ry=_rank_average(y[m])
    if np.std(rx)==0 or np.std(ry)==0:return float('nan')
    return float(np.corrcoef(rx,ry)[0,1])


def _stratified_folds(labels,n_splits=5,seed=2027):
    labels=np.asarray(labels,int); n=len(labels); folds=np.empty(n,dtype=int); rng=np.random.default_rng(seed)
    for c in np.unique(labels):
        idx=np.where(labels==c)[0].copy(); rng.shuffle(idx)
        for j,i in enumerate(idx):folds[i]=j % n_splits
    return folds


@dataclass
class _DiagModel:
    mean: np.ndarray
    std: np.ndarray
    class_means: Dict[int,np.ndarray]
    global_mean: np.ndarray
    def score(self,x,cls):
        z=(np.asarray(x,float)-self.mean)/self.std
        mu=self.class_means.get(int(cls),self.global_mean)
        return float(np.mean((z-mu)**2))


def _fit_diag(X,classes):
    X=np.asarray(X,float); classes=np.asarray(classes,int)
    mean=X.mean(0); std=X.std(0); std=np.where(std<1e-6,1.0,std)
    Z=(X-mean)/std; gm=Z.mean(0); cms={}
    for c in np.unique(classes):
        z=Z[classes==c]
        cms[int(c)]=z.mean(0) if len(z)>=2 else gm
    return _DiagModel(mean,std,cms,gm)


class CrossFitDetector:
    """Clean-only class-conditioned diagonal Gaussian anomaly detector.

    Every attacked sample is scored by a model fitted without the corresponding
    clean sample. Fold-specific thresholds are calibrated from out-of-fold clean
    scores belonging to other folds, so the sample under test does not set its
    own threshold.
    """
    def __init__(self, Xclean, classes, seed=2027, n_splits=5):
        Xclean=np.asarray(Xclean,float); classes=np.asarray(classes,int); n=len(Xclean)
        if n<10:raise ValueError('defense calibration needs >=10 clean examples')
        self.folds=_stratified_folds(classes,n_splits=n_splits,seed=seed)
        self.models={}; self.clean_scores=np.full(n,np.nan,float)
        for f in range(n_splits):
            tr=self.folds!=f; va=self.folds==f
            if tr.sum()<5 or va.sum()==0:continue
            mdl=_fit_diag(Xclean[tr],classes[tr]); self.models[f]=mdl
            for i in np.where(va)[0]:self.clean_scores[i]=mdl.score(Xclean[i],classes[i])
        if np.isnan(self.clean_scores).any():
            fallback=_fit_diag(Xclean,classes)
            for i in np.where(np.isnan(self.clean_scores))[0]:self.clean_scores[i]=fallback.score(Xclean[i],classes[i])
        self.thresholds={}
        for f in range(n_splits):
            ref=self.clean_scores[self.folds!=f]
            self.thresholds[f]={a:_safe_quantile(ref,1-a) for a in FPRS}
        self.actual_fpr={}
        for a in FPRS:
            hit=[]
            for i,s in enumerate(self.clean_scores):hit.append(s>self.thresholds[int(self.folds[i])][a])
            self.actual_fpr[a]=float(np.mean(hit))
    def score(self,x,cls,index):
        f=int(self.folds[int(index)]); mdl=self.models.get(f)
        if mdl is None:raise RuntimeError(f'missing fold model {f}')
        s=mdl.score(x,cls)
        return s,{a:bool(s>self.thresholds[f][a]) for a in FPRS}
    def clean_score(self,index):return float(self.clean_scores[int(index)])


def _aggregate_fixed(X,width,phase=0):
    T=X.shape[0]; ids=np.floor_divide(np.arange(T,dtype=np.int64)-int(phase),int(width)); vals=np.unique(ids); out=[]
    Xi=X.astype(np.float64,copy=False)
    for v in vals:out.append(Xi[ids==v].sum(axis=0))
    return np.stack(out,axis=0)


def _aggregate_overlap(X,width,stride):
    T=X.shape[0]; out=[]; Xi=X.astype(np.float64,copy=False)
    for s in range(0,T,max(1,int(stride))):
        e=min(T,s+int(width));
        if e<=s:continue
        out.append(Xi[s:e].sum(axis=0))
        if e==T:break
    return np.stack(out,axis=0)


def _resample(v,n):
    v=np.asarray(v,float)
    if len(v)==n:return v
    if len(v)==1:return np.full(n,v[0],float)
    xp=np.linspace(0,1,len(v)); xq=np.linspace(0,1,n)
    return np.interp(xq,xp,v)


def _repr_features(R,target_k,total):
    """Compact but spatially aware statistics from one temporal representation."""
    R=np.asarray(R,float); K,C,H,W=R.shape; den=max(float(total),1.0); feats=[]
    # Temporal polarity profile.
    q=R.sum(axis=(2,3))/den
    for c in range(C):feats.extend(_resample(q[:,c],target_k).tolist())
    # Four spatial quadrants per temporal window, pooled over polarity.
    h2=max(1,H//2); w2=max(1,W//2)
    quads=[R[:,:,:h2,:w2],R[:,:,:h2,w2:],R[:,:,h2:,:w2],R[:,:,h2:,w2:]]
    for z in quads:
        u=z.sum(axis=(1,2,3))/den; feats.extend(_resample(u,target_k).tolist())
    # Per-pixel temporal statistics.  These retain the local temporal signal that
    # quadrant pooling can erase under timestamp retiming.  We summarize the
    # distribution rather than flattening CxHxW, keeping the detector cheap.
    if K > 1:
        dR=np.diff(R,axis=0)
        tv_pix=np.abs(dR).sum(axis=0)/den             # C x H x W
        tv_step=np.abs(dR).sum(axis=(1,2,3))/den      # K-1 temporal profile
        feats.extend(_resample(tv_step,max(1,target_k-1)).tolist())
    else:
        tv_pix=np.zeros((C,H,W),dtype=float)
        feats.extend([0.0]*max(1,target_k-1))
    occ=(R>0).astype(np.float64)
    occ_var=occ.var(axis=0)                            # C x H x W
    count_var=R.var(axis=0)/(den*den)                  # C x H x W
    def _dist_stats(v):
        v=np.asarray(v,float).ravel()
        q=np.quantile(v,[0.50,0.75,0.90,0.95,0.99]) if len(v) else np.zeros(5)
        return [float(v.mean()) if len(v) else 0.0,float(v.std()) if len(v) else 0.0,*map(float,q),float(v.max(initial=0.0))]
    feats.extend(_dist_stats(tv_pix))
    feats.extend(_dist_stats(occ_var))
    feats.extend(_dist_stats(count_var))
    for c in range(C):
        feats.extend(_dist_stats(tv_pix[c]))
        feats.extend(_dist_stats(occ_var[c]))

    # Global summaries retained because add/remove attacks can change total mass.
    pol=R.sum(axis=(0,2,3))/den; feats.extend(pol.tolist()); feats.append(math.log1p(float(total)))
    qt=R.sum(axis=(1,2,3)); p=qt/max(qt.sum(),1.0); nz=p[p>0]
    feats.extend([float(-(nz*np.log(nz)).sum()),float(p.max(initial=0.0)),float(np.abs(np.diff(p)).sum())])
    return np.asarray(feats,dtype=np.float64)


def _view_feature(X,grid,width,phase=0,overlap_stride=None):
    R=_aggregate_overlap(X,width,overlap_stride) if overlap_stride is not None else _aggregate_fixed(X,width,phase)
    return _repr_features(R,grid.K,float(X.sum()))


def observer_feature_sets(X,grid,sample_index,seed):
    S=int(grid.S); T=int(grid.T); phases=[0,min(1,S-1),min(2,S-1),min(4,S-1)]
    phase_views=[_view_feature(X,grid,S,p) for p in phases]
    overlap=_view_feature(X,grid,S,0,max(1,S//2))
    widths=sorted({max(2,S//2),S,min(T,2*S)})
    multiscale=np.concatenate([_view_feature(X,grid,w,0) for w in widths])
    rng=np.random.default_rng(int(seed)*1000003+int(sample_index)*9176+73)
    choices=sorted({max(2,S-2),max(2,S-1),S,S+1,S+2})
    rv=[]
    for _ in range(3):
        w=int(rng.choice(choices)); p=int(rng.integers(0,max(1,w))); rv.append(_view_feature(X,grid,w,p))
    randomized=np.concatenate(rv)
    return {
        'phase1':phase_views[0],
        'phase2':np.concatenate(phase_views[:3:2]),   # phase 0 + phase 2
        'phase3':np.concatenate([phase_views[0],phase_views[2],phase_views[3]]), # 0+2+4
        'phase4':np.concatenate(phase_views),        # 0+1+2+4
        'overlap':np.concatenate([phase_views[0],overlap]),
        'multiscale':multiscale,
        'randomized':np.concatenate([phase_views[0],randomized]),
        'all':np.concatenate(phase_views+[overlap,multiscale,randomized]),
    }


def _op_matrix(T,S,p):
    ids=np.floor_divide(np.arange(T)-int(p),int(S)); vals=np.unique(ids); A=np.zeros((len(vals),T),float)
    for r,v in enumerate(vals):A[r,ids==v]=1.0
    return A


def phase_blind_fraction(T,S,detector):
    phases={'phase1':[0],'phase2':[0,2],'phase3':[0,2,4],'phase4':[0,1,2,4]}.get(detector)
    if phases is None:return float('nan')
    A=np.concatenate([_op_matrix(T,S,p) for p in phases],axis=0); rank=int(np.linalg.matrix_rank(A))
    return float((T-rank)/T)


class DefenseBundle:
    def __init__(self,X,classes,grid,seed,calibration_X=None,calibration_classes=None):
        self.grid=grid; self.seed=int(seed); self.names=['phase1','phase2','phase3','phase4','overlap','multiscale','randomized','all']
        # Default: cross-fit on the attacked evaluation population.  Preferred
        # publication mode: provide a larger disjoint clean calibration pool.
        source_X=X if calibration_X is None else calibration_X
        source_cls=classes if calibration_classes is None else calibration_classes
        clean={k:[] for k in self.names}
        for i,x in enumerate(source_X):
            fs=observer_feature_sets(x,grid,i,seed)
            for k in self.names:clean[k].append(fs[k])
        self.external_calibration=calibration_X is not None
        self.det={k:CrossFitDetector(np.stack(clean[k]),source_cls,seed=2027+seed) for k in self.names}
        # When calibration is external, attack sample indices do not correspond to
        # detector folds. Use a deterministic fold mapping at scoring time.
        self.eval_n=len(X)
    def score(self,Xa,sample_index,observed_class,Xclean=None):
        fs=observer_feature_sets(Xa,self.grid,sample_index,self.seed); out={}
        fsc=observer_feature_sets(Xclean,self.grid,sample_index,self.seed) if Xclean is not None else None
        for k in self.names:
            if self.external_calibration:
                # Deterministically pick one calibration fold; thresholds/models
                # remain independent of the attacked evaluation sample.
                idx=int(sample_index)%len(self.det[k].folds); f=int(self.det[k].folds[idx]); mdl=self.det[k].models[f]
                s=mdl.score(fs[k],observed_class); h={a:bool(s>self.det[k].thresholds[f][a]) for a in FPRS}
                cs=mdl.score(fsc[k],observed_class) if fsc is not None else float('nan')
            else:
                s,h=self.det[k].score(fs[k],observed_class,sample_index); cs=self.det[k].clean_score(sample_index)
            out[f'def_{k}_score']=float(s); out[f'def_{k}_clean_score']=float(cs)
            for a in FPRS:
                out[f'def_{k}_det_{int(a*100):02d}']=bool(h[a])
                # Evaluation-set clean flag under the same threshold.  This is
                # especially important when thresholds came from an external
                # clean pool: nominal and realized test FPR need not match.
                if self.external_calibration:
                    out[f'def_{k}_clean_det_{int(a*100):02d}']=bool(cs>self.det[k].thresholds[f][a])
                else:
                    fi=int(self.det[k].folds[int(sample_index)])
                    out[f'def_{k}_clean_det_{int(a*100):02d}']=bool(cs>self.det[k].thresholds[fi][a])
        return out
    def calibration_summary(self):
        """Human-readable calibration record, including the actual fold thresholds."""
        rows=[]
        for k in self.names:
            d=self.det[k]
            thresholds={
                str(int(f)):{f'fpr_{int(a*100):02d}':float(d.thresholds[f][a]) for a in FPRS}
                for f in sorted(d.thresholds)
            }
            r={'detector':k,'n_clean':len(d.clean_scores),
               'blind_fraction':phase_blind_fraction(self.grid.T,self.grid.S,k),
               # Store as canonical JSON so CSV aggregation/drop_duplicates stays hashable.
               'fold_thresholds_json':json.dumps(thresholds,sort_keys=True,separators=(',',':'))}
            for a in FPRS:r[f'actual_fpr_{int(a*100):02d}']=d.actual_fpr[a]
            rows.append(r)
        return rows

    def calibration_sha256(self):
        """Cryptographic fingerprint of the fitted clean-only defender.

        This hashes the fitted model parameters, fold assignments, clean scores,
        thresholds and defense configuration.  It is used only for identity
        validation across shards/methods at the same dataset/seed/victim.
        """
        h=hashlib.sha256()
        def put_text(x):
            h.update(str(x).encode('utf-8')); h.update(b'\\0')
        def put_arr(x):
            a=np.ascontiguousarray(np.asarray(x))
            put_text(a.dtype.str); put_text(a.shape); h.update(a.tobytes(order='C')); h.update(b'\\0')
        put_text(DEFENSE_VERSION); put_text(self.seed); put_text(self.grid.T); put_text(self.grid.S)
        put_text(bool(self.external_calibration))
        for k in self.names:
            put_text(k); d=self.det[k]
            put_arr(d.folds); put_arr(d.clean_scores)
            for f in sorted(d.models):
                put_text(f); m=d.models[f]
                put_arr(m.mean); put_arr(m.std); put_arr(m.global_mean)
                for c in sorted(m.class_means):
                    put_text(c); put_arr(m.class_means[c])
            for f in sorted(d.thresholds):
                put_text(f)
                for a in FPRS:
                    put_text(a); put_text(format(float(d.thresholds[f][a]), '.17g'))
            for a in FPRS:
                put_text(a); put_text(format(float(d.actual_fpr[a]), '.17g'))
        return h.hexdigest()


def _supervised_attack_aware(df,out,dets):
    """Cross-fitted attack-aware detector using the clean-only detector scores as features.

    This is intentionally a secondary, stronger defender.  Folds are grouped by
    sample_index so a sample and its attacked counterpart never cross train/test.
    Thresholds are calibrated only on held-out CLEAN scores at 1%/5% FPR.  For
    phase1, exact null-space invariance means the clean and attacked score feature
    are identical, so supervised training cannot manufacture separability.
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
    except Exception:
        return
    rows=[]
    # Two variants: canonical-only and full multi-observer score vector.
    variants={'phase1':['phase1']}
    if dets: variants['all_scores']=list(dets)
    keys=['dataset','victim','method','seed','budget_label']
    for gkeys,g in df.groupby(keys,dropna=False):
        if len(g)<20: continue
        base=dict(zip(keys,gkeys))
        sid=g.sample_index.to_numpy(int)
        # deterministic grouped folds by sample id
        uniq=np.unique(sid); rng=np.random.default_rng(7717+int(base['seed']))
        uu=uniq.copy(); rng.shuffle(uu); foldmap={int(x):i%5 for i,x in enumerate(uu)}
        folds=np.array([foldmap[int(x)] for x in sid],int)
        succ=g.attack_success_on_clean_correct.astype(bool).to_numpy()
        for vname,vdet in variants.items():
            acols=[f'def_{d}_score' for d in vdet]
            ccols=[f'def_{d}_clean_score' for d in vdet]
            if not set(acols+ccols).issubset(g.columns): continue
            Xa=g[acols].to_numpy(float); Xc=g[ccols].to_numpy(float)
            ps=np.full(len(g),np.nan,float); pc=np.full(len(g),np.nan,float)
            for f in range(5):
                tr=folds!=f; te=folds==f
                if tr.sum()<10 or te.sum()==0: continue
                Xtr=np.vstack([Xc[tr],Xa[tr]]); ytr=np.r_[np.zeros(tr.sum(),int),np.ones(tr.sum(),int)]
                sc=StandardScaler().fit(Xtr); Z=sc.transform(Xtr)
                clf=LogisticRegression(max_iter=2000,class_weight='balanced',solver='lbfgs').fit(Z,ytr)
                pc[te]=clf.predict_proba(sc.transform(Xc[te]))[:,1]
                ps[te]=clf.predict_proba(sc.transform(Xa[te]))[:,1]
            good=np.isfinite(ps)&np.isfinite(pc)
            if good.sum()<10: continue
            for a in FPRS:
                thr=_safe_quantile(pc[good],1-a)
                hit=ps[good]>thr; cleanhit=pc[good]>thr; sg=succ[good]
                rows.append(base|{'supervised_detector':vname,'nominal_fpr':a,'realized_clean_fpr':float(cleanhit.mean()),
                                  'n':int(good.sum()),'auroc':auroc(pc[good],ps[good]),'TPR':float(hit.mean()),
                                  'ASR':float(sg.mean()),'UASR':float(np.mean(sg&(~hit)))})
    if rows:
        pd.DataFrame(rows).to_csv(out/'supervised_attack_aware_defense.csv',index=False)

def extend_results(df,out,calibration_rows=None):
    out.mkdir(parents=True,exist_ok=True)
    methods=['yu_pil_l0','pdsg_sda','yao_gumbel','free_retiming','null_space']
    d=df[df.method.isin(methods)&df.clean_correct.astype(bool)].copy()
    dets=[c[len('def_'):-len('_score')] for c in d.columns if c.startswith('def_') and c.endswith('_score') and not c.endswith('_clean_score')]
    dets=sorted(set(dets))
    if calibration_rows:pd.DataFrame(calibration_rows).drop_duplicates().to_csv(out/'defender_calibration.csv',index=False)
    # Operational table.
    rows=[]
    groupcols=['dataset','victim','method','seed','budget_label']
    for keys,g in d.groupby(groupcols,dropna=False):
        base=dict(zip(groupcols,keys)); succ=g.attack_success_on_clean_correct.astype(bool).to_numpy(); DA=g.D_A.to_numpy(float)
        for det in dets:
            adv=g[f'def_{det}_score'].to_numpy(float); cln=g[f'def_{det}_clean_score'].to_numpy(float)
            r=base|{'detector':det,'n':len(g),'ASR':float(succ.mean()),'mean_D_A':float(np.mean(DA)),'median_D_A':float(np.median(DA)),'AUROC':auroc(cln,adv)}
            for pct in (1,5):
                hit=g[f'def_{det}_det_{pct:02d}'].astype(bool).to_numpy();
                r[f'detection_rate_fpr{pct}']=float(hit.mean());r[f'detection_rate_success_fpr{pct}']=float(hit[succ].mean()) if succ.any() else float('nan');r[f'UASR_fpr{pct}']=float(np.mean(succ&(~hit)))
                cflag=f'def_{det}_clean_det_{pct:02d}'
                r[f'realized_eval_clean_fpr{pct}']=float(g[cflag].astype(bool).mean()) if cflag in g.columns else float('nan')
            rows.append(r)
    pd.DataFrame(rows).to_csv(out/'defense_operational_by_seed.csv',index=False)
    # D_A -> detectability bins, pooled and method-specific.
    br=[]
    for det in dets:
      for pct in (1,5):
        hitcol=f'def_{det}_det_{pct:02d}'
        for method_scope,g0 in [('ALL',d)]+[(m,d[d.method==m]) for m in methods]:
          if len(g0)==0:continue
          idx=np.digitize(g0.D_A.to_numpy(float),DA_BINS,right=False)-1
          for j in range(len(DA_BINS)-1):
            g=g0[idx==j]
            if len(g)==0:continue
            succ=g.attack_success_on_clean_correct.astype(bool)
            br.append({'detector':det,'fpr_pct':pct,'method':method_scope,'bin_lo':DA_BINS[j],'bin_hi':DA_BINS[j+1],'bin_mid':float(DA_BINS[j]+(DA_BINS[j+1]-DA_BINS[j])/2) if np.isfinite(DA_BINS[j+1]) else float(DA_BINS[j]),'n':len(g),'mean_D_A':float(g.D_A.mean()),'detection_rate':float(g[hitcol].astype(bool).mean()),'detection_rate_success':float(g.loc[succ,hitcol].astype(bool).mean()) if succ.any() else float('nan'),'ASR':float(succ.mean())})
    pd.DataFrame(br).to_csv(out/'detectability_vs_DA_bins.csv',index=False)
    # Direct tau_shift -> detectability analysis.  Detection is measured among all
    # attacks within the representation budget and, separately, among successful
    # attacks within the same budget.
    tr=[]
    for keys,g in d.groupby(groupcols,dropna=False):
        base=dict(zip(groupcols,keys)); succ=g.attack_success_on_clean_correct.astype(bool).to_numpy(); DA=g.D_A.to_numpy(float)
        for det in dets:
          for pct in (1,5):
            hit=g[f'def_{det}_det_{pct:02d}'].astype(bool).to_numpy()
            for tau in TAU_GRID:
                inside=DA<=tau+1e-12; sis=inside&succ
                tr.append(base|{'detector':det,'fpr_pct':pct,'tau':tau,'n_total':len(g),'n_within_tau':int(inside.sum()),'n_success_within_tau':int(sis.sum()),
                                'fraction_within_tau':float(inside.mean()),'SC_ASR':float(sis.mean()),
                                'detection_rate_within_tau':float(hit[inside].mean()) if inside.any() else float('nan'),
                                'detection_rate_success_within_tau':float(hit[sis].mean()) if sis.any() else float('nan'),
                                'undetected_SC_ASR':float(np.mean(sis&(~hit)))})
    pd.DataFrame(tr).to_csv(out/'tau_shift_vs_detectability.csv',index=False)
    # Method-wise shift distribution.
    qs=[]
    for keys,g in d.groupby(groupcols,dropna=False):
        r=dict(zip(groupcols,keys)); q=np.quantile(g.D_A.to_numpy(float),[0,.1,.25,.5,.75,.9,1]); r.update(zip(['DA_min','DA_q10','DA_q25','DA_median','DA_q75','DA_q90','DA_max'],map(float,q)));qs.append(r)
    pd.DataFrame(qs).to_csv(out/'DA_distribution.csv',index=False)
    # Correlation between realized shift and defender anomaly score.
    cr=[]
    for keys,g in d.groupby(groupcols,dropna=False):
        base=dict(zip(groupcols,keys)); succ=g.attack_success_on_clean_correct.astype(bool)
        for det in dets:
            cr.append(base|{'detector':det,'n':len(g),'spearman_DA_score_all':spearman(g.D_A,g[f'def_{det}_score']),'spearman_DA_score_success':spearman(g.loc[succ,'D_A'],g.loc[succ,f'def_{det}_score']) if succ.sum()>=3 else float('nan')})
    pd.DataFrame(cr).to_csv(out/'DA_detectability_correlation.csv',index=False)
    # Multi-observer defense progression.
    op=pd.DataFrame(rows)
    mo=op[op.detector.isin(['phase1','phase2','phase3','phase4'])].copy()
    if len(mo):
        _grid={'dvsgesture':(160,8),'dailydvs200':(80,8),'cifar10dvs':(80,8)}
        _T,_S=_grid.get(str(d.dataset.iloc[0]),(None,None))
        mo['blind_fraction']=mo.detector.map(lambda k:phase_blind_fraction(_T,_S,k) if _T is not None else float('nan'))
        mo.to_csv(out/'multiobserver_defense.csv',index=False)
    _supervised_attack_aware(d,out,dets)
    _plots(d,out,dets)



def extend_sweep_results(df,out):
    """Controlled within-family D_A sweep for the mechanism claim.

    This keeps the attack family fixed (free retiming) while changing only the
    allowed temporal max shift.  It is deliberately separate from the five-arm
    comparison so correlations are not driven by attack-family clusters.
    """
    if 'method' not in df.columns or not (df.method=='free_da_sweep').any():
        return
    d=df[(df.method=='free_da_sweep') & df.clean_correct.astype(bool)].copy()
    if 'sweep_max_shift' not in d.columns or len(d)==0:return
    dets=sorted({c[len('def_'):-len('_score')] for c in d.columns if c.startswith('def_') and c.endswith('_score') and not c.endswith('_clean_score')})
    rows=[]
    for (dataset,victim,seed,ms),g in d.groupby(['dataset','victim','seed','sweep_max_shift'],dropna=False):
        succ=g.attack_success_on_clean_correct.astype(bool).to_numpy(); DA=g.D_A.to_numpy(float)
        base={'dataset':dataset,'victim':victim,'seed':seed,'max_shift_bins':int(ms),'n':len(g),
              'mean_D_A':float(np.mean(DA)),'median_D_A':float(np.median(DA)),'ASR':float(np.mean(succ)),
              'mean_shift_ms':float(g.avg_abs_shift_ms.mean()) if g.avg_abs_shift_ms.notna().any() else float('nan')}
        for det in dets:
            r=base|{'detector':det,'spearman_DA_score':spearman(g.D_A,g[f'def_{det}_score'])}
            for pct in (1,5):
                hit=g[f'def_{det}_det_{pct:02d}'].astype(bool).to_numpy()
                r[f'detection_rate_fpr{pct}']=float(hit.mean())
                r[f'UASR_fpr{pct}']=float(np.mean(succ & (~hit)))
            rows.append(r)
    z=pd.DataFrame(rows).sort_values(['dataset','victim','seed','detector','max_shift_bins'])
    z.to_csv(out/'controlled_DA_sweep.csv',index=False)
    # Per-sample values are retained for reviewer-auditable monotonicity checks.
    keep=[c for c in ['dataset','victim','seed','sample_uid','sample_index','sweep_max_shift','D_A','attack_success_on_clean_correct','avg_abs_shift_ms'] if c in d.columns]
    keep += [c for c in d.columns if c.startswith('def_') and (c.endswith('_score') or c.endswith('_det_05') or c.endswith('_det_01'))]
    d[keep].to_csv(out/'controlled_DA_sweep_per_sample.csv',index=False)
    try:
        import matplotlib.pyplot as plt
        for det in [x for x in ['phase1','all'] if x in dets]:
            q=z[z.detector==det]
            if len(q)==0:continue
            q=q.groupby('max_shift_bins',as_index=False).agg(mean_D_A=('mean_D_A','mean'),detection_rate_fpr5=('detection_rate_fpr5','mean'),ASR=('ASR','mean'),UASR_fpr5=('UASR_fpr5','mean'))
            plt.figure(figsize=(6.4,4.6));plt.plot(q.mean_D_A,100*q.detection_rate_fpr5,marker='o')
            plt.xlabel(r'Realized representation shift $D_A$');plt.ylabel('Detection rate @ 5% clean FPR (%)');plt.grid(alpha=.25);plt.tight_layout();plt.savefig(out/f'fig_controlled_DA_detectability_{det}.png',dpi=250);plt.close()
    except Exception as e:
        print('controlled sweep plot skipped',repr(e))

def _plots(d,out,dets):
    import matplotlib.pyplot as plt
    # SC-ASR(tau) from the unified metric output, one figure per victim.
    scp=out/'sc_asr_tau_grid.csv'
    if scp.exists():
        sc=pd.read_csv(scp);sc=sc[sc.method.isin(['yu_pil_l0','pdsg_sda','yao_gumbel','free_retiming','null_space'])]
        # Main five-arm figure uses one explicit operating point per method to avoid
        # pooling progressive Free/Null budgets. Yu/Free/Null use 10%; SDA/Yao use native.
        headline_budget={'yu_pil_l0':'10%','free_retiming':'10%','null_space':'10%','pdsg_sda':'native','yao_gumbel':'native'}
        for victim,g0 in sc.groupby('victim'):
            plt.figure(figsize=(6.6,4.7))
            for method,g in g0.groupby('method'):
                want=headline_budget.get(method)
                if want is not None and 'budget_label' in g.columns:
                    g=g[g.budget_label.astype(str)==want]
                if len(g)==0: continue
                z=g.groupby('tau',as_index=False).SC_ASR.mean();plt.plot(z.tau,100*z.SC_ASR,label=method)
            plt.xlabel(r'Representation tolerance $\tau$');plt.ylabel(r'$\mathrm{SC\!-\!ASR}_A(\tau)$ (%)');plt.xlim(0,.30);plt.ylim(0,100);plt.grid(alpha=.25);plt.legend(fontsize=7);plt.tight_layout();plt.savefig(out/f'fig_SC_ASR_tau_{victim}.png',dpi=250);plt.close()
    # D_A distribution by method.
    methods=['yu_pil_l0','pdsg_sda','yao_gumbel','free_retiming','null_space']
    vals=[d.loc[d.method==m,'D_A'].to_numpy(float) for m in methods if (d.method==m).any()]; labs=[m for m in methods if (d.method==m).any()]
    if vals:
        plt.figure(figsize=(8,4.8));plt.boxplot(vals,labels=labs,showfliers=False);plt.ylabel(r'Representation shift $D_A$');plt.xticks(rotation=20,ha='right');plt.tight_layout();plt.savefig(out/'fig_DA_distribution.png',dpi=250);plt.close()
    # Detectability vs D_A for strongest multi-observer detector and canonical detector.
    for det in [x for x in ['phase1','all'] if x in dets]:
        plt.figure(figsize=(6.4,4.6))
        g=d.copy();idx=np.digitize(g.D_A.to_numpy(float),DA_BINS,right=False)-1;xs=[];ys=[];ns=[]
        for j in range(len(DA_BINS)-1):
            z=g[idx==j]
            if len(z)<5:continue
            xs.append(float(z.D_A.mean()));ys.append(float(z[f'def_{det}_det_05'].astype(bool).mean())*100);ns.append(len(z))
        if xs:
            plt.plot(xs,ys,marker='o');plt.xlabel(r'Realized representation shift $D_A$');plt.ylabel('Detection rate at 5% clean FPR (%)');plt.ylim(0,100);plt.grid(alpha=.25);plt.tight_layout();plt.savefig(out/f'fig_detectability_vs_DA_{det}.png',dpi=250);plt.close()
    # Efficacy-detectability frontier at 5% FPR using all-observer detector.
    if 'all' in dets:
        plt.figure(figsize=(6.4,5.0))
        for (m,b),g in d.groupby(['method','budget_label'],dropna=False):
            x=float(g.def_all_det_05.astype(bool).mean())*100;y=float(g.attack_success_on_clean_correct.astype(bool).mean())*100
            plt.scatter([x],[y]);plt.annotate(f'{m}:{b}',(x,y),fontsize=7,xytext=(3,3),textcoords='offset points')
        plt.xlabel('Defender detection rate @ 5% clean FPR (%)');plt.ylabel('ASR (%)');plt.xlim(0,100);plt.ylim(0,100);plt.grid(alpha=.25);plt.tight_layout();plt.savefig(out/'fig_efficacy_detectability_frontier.png',dpi=250);plt.close()
    # UASR @5% FPR with all-observer detector.
    if 'all' in dets:
        bars=[]
        for (m,b),g in d.groupby(['method','budget_label'],dropna=False):
            succ=g.attack_success_on_clean_correct.astype(bool).to_numpy();hit=g.def_all_det_05.astype(bool).to_numpy();bars.append((f'{m}:{b}',100*float(np.mean(succ&(~hit)))))
        if bars:
            plt.figure(figsize=(8,4.8));plt.bar([x for x,_ in bars],[y for _,y in bars]);plt.ylabel('UASR @ 5% clean FPR (%)');plt.xticks(rotation=30,ha='right');plt.ylim(0,100);plt.tight_layout();plt.savefig(out/'fig_UASR_fpr5.png',dpi=250);plt.close()

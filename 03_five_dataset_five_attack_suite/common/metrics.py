import math,numpy as np,pandas as pd
TAU_GRID=[0,.001,.002,.005,.01,.02,.05,.10,.15,.17,.18,.19,.195,.20,.205,.21,.22,.23,.24,.5,1.0]
EPS_GRID_MS=[0,1,2,5,10,20,50,100,200,300,404,500,600,945,1000,1200,1500,2000,3000,5000,7000]
def wilson(k,n,z=1.959963984540054):
    if not n:return (float('nan'),float('nan'))
    p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d;h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d;return max(0,c-h),min(1,c+h)
def row(grid,X0,Xa,y,clean_pred,vfn,c0,cfn,shifts,bin_ms,prov=None):
    m=grid.metrics(X0,Xa);total=max(1,int(X0.sum()));pred=int(vfn(Xa[None]).argmax(1)[0]);_cp=cfn(np.stack([X0,Xa]));c0=_cp[0:1];ca=_cp[1:2];d=Xa.astype(np.int64)-X0.astype(np.int64);fine_l1=int(np.abs(d).sum());success=bool(clean_pred==y and pred!=y);r={"clean_correct":bool(clean_pred==y),"attack_success_on_clean_correct":success,"victim_pred_adv":pred,"total_event_units":int(X0.sum()),"frame_l1_difference":m['frame_l1_difference'],"D_A":m['frame_l1_difference']/total,"D_inf":m['D_inf'],"frame_exact_equal":m['frame_exact_equal'],"fine_tensor_L1":fine_l1,"count_preserving":bool(int(Xa.sum())==int(X0.sum())),"units_added":int(np.clip(d,0,None).sum()),"units_removed":int(np.clip(-d,0,None).sum()),"changed_cells":int(np.count_nonzero(d)),"realized_footprint_pct":100.0*fine_l1/(2*total),"edit_budget_units":int(max(int(np.clip(d,0,None).sum()),int(np.clip(-d,0,None).sum()))),"edit_budget_pct":100.0*max(int(np.clip(d,0,None).sum()),int(np.clip(-d,0,None).sum()))/total,"l1_perturbation_pct":100.0*fine_l1/total,"protected_consumer_flip":bool(int(ca.argmax(1)[0])!=int(c0.argmax(1)[0])),"protected_consumer_logits_bit_identical":bool(np.array_equal(ca,c0)),"protected_consumer_max_abs_logit_difference":float(np.max(np.abs(ca-c0)))}
    if shifts is None:r.update(avg_abs_shift_bins=np.nan,avg_abs_shift_ms=np.nan,max_abs_shift_ms=np.nan,moved_event_units=np.nan)
    else:
        s=np.abs(np.asarray(shifts,float));r.update(avg_abs_shift_bins=float(s.mean()) if len(s) else 0.,avg_abs_shift_ms=float(s.mean()*bin_ms) if len(s) else 0.,max_abs_shift_ms=float(s.max()*bin_ms) if len(s) else 0.,moved_event_units=int(len(s)))
    if prov:r.update({f"prov_{k}":v for k,v in prov.items() if np.isscalar(v) or isinstance(v,str)})
    return r
def extended(df,out):
    cc=df[df.clean_correct.astype(bool)].copy();ret=[]
    for (ds,v,m,s),g in cc.groupby(['dataset','victim','method','seed']):
        succ=g.attack_success_on_clean_correct.astype(bool).to_numpy();DA=g.D_A.to_numpy(float);n=len(g)
        for t in TAU_GRID:
            ok=DA<=t+1e-12;k=int((succ&ok).sum());kp=int(ok.sum())
            lo,hi=wilson(k,n);plo,phi=wilson(kp,n)
            ret.append(dict(dataset=ds,victim=v,method=m,seed=s,tau=t,n_clean_correct=n,
                            SC_ASR=float(k/n) if n else float('nan'),SC_ASR_ci_low=lo,SC_ASR_ci_high=hi,
                            SPR=float(kp/n) if n else float('nan'),SPR_ci_low=plo,SPR_ci_high=phi))
    pd.DataFrame(ret).to_csv(out/'sc_asr_tau_grid.csv',index=False)
    ret=[]
    rt=cc[cc.method.isin(['yu_pil_l0','free_retiming','null_space'])]
    for (ds,v,m,s),g in rt.groupby(['dataset','victim','method','seed']):
        succ=g.attack_success_on_clean_correct.astype(bool).to_numpy();dt=g.avg_abs_shift_ms.to_numpy(float)
        for e in EPS_GRID_MS:ret.append(dict(dataset=ds,victim=v,method=m,seed=s,epsilon_ms=e,A_eps=float(np.mean(succ&(dt<=e)))))
    pd.DataFrame(ret).to_csv(out/'temporal_a_eps.csv',index=False)
    cols=['dataset','victim','method','seed','sample_index','realized_footprint_pct','edit_budget_units','edit_budget_pct','l1_perturbation_pct','fine_tensor_L1','units_added','units_removed','count_preserving','D_A','D_inf']
    cc[[c for c in cols if c in cc.columns]].to_csv(out/'footprint_table.csv',index=False)

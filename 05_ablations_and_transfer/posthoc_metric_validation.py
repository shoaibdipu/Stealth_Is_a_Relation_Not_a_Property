#!/usr/bin/env python3
"""Cheap CPU analysis that validates the paper's novel stealth metrics.

Run this as soon as ANY five-arm result directories exist. It never launches an
attack; it only consumes per_sample_rows.csv or partial JSONs already produced by
EA_EVS_5x5_FINAL_20260923.
"""
from __future__ import annotations
import os, json
from pathlib import Path
import numpy as np
import pandas as pd
from common.datasets import SUITE
from common.metrics import TAU_GRID, EPS_GRID_MS


def _load_one(spec):
    root=Path(os.getenv(f"{spec.key.upper()}_RUN_ROOT", os.getenv("RUN_ROOT", spec.default_run_root)))
    dirs=sorted((root/"results").glob(f"five_attack_{spec.key}_*"), key=lambda p:p.stat().st_mtime if p.exists() else 0)
    frames=[]
    for d in dirs:
        csv=d/"per_sample_rows.csv"
        if csv.exists():
            x=pd.read_csv(csv); x["source_dir"]=str(d); frames.append(x); continue
        rows=[]
        for p in (d/"partials").glob("*/*.json"):
            try: rows += json.loads(p.read_text()).get("rows",[])
            except Exception: pass
        if rows:
            x=pd.DataFrame(rows); x["source_dir"]=str(d); frames.append(x)
    if not frames: return None
    # Use newest protocol directory only to avoid duplicate partial campaigns.
    return frames[-1]


def _rankcorr(x,y):
    a=pd.Series(x).rank(method="average").to_numpy(float); b=pd.Series(y).rank(method="average").to_numpy(float)
    if len(a)<3 or np.std(a)==0 or np.std(b)==0: return np.nan
    return float(np.corrcoef(a,b)[0,1])


def main():
    out=Path(os.getenv("METRIC_OUT","metric_validation_2day")); out.mkdir(parents=True,exist_ok=True)
    allf=[]
    for spec in SUITE.values():
        d=_load_one(spec)
        if d is not None and len(d): allf.append(d)
    if not allf:
        raise SystemExit("No completed five_attack_<dataset>_* results found. Run after at least one 5x5 arm has landed.")
    df=pd.concat(allf,ignore_index=True,sort=False)
    cc=df[df["clean_correct"].astype(bool)].copy()
    if "budget_label" not in cc: cc["budget_label"]="unknown"

    # Raw ASR vs constrained success and retention.
    rows=[]
    grp=["dataset","victim","method","seed","budget_label"]
    for key,g in cc.groupby(grp,dropna=False):
        succ=g["attack_success_on_clean_correct"].astype(bool).to_numpy(); DA=g["D_A"].to_numpy(float)
        raw=float(succ.mean()) if len(g) else np.nan
        rec=dict(zip(grp,key)); rec.update(n=len(g),raw_ASR=raw,mean_D_A=float(np.nanmean(DA)),
                                           mean_D_inf=float(np.nanmean(g["D_inf"])),
                                           mean_footprint_pct=float(np.nanmean(g["realized_footprint_pct"])))
        for t in TAU_GRID:
            sc=float(np.mean(succ & (DA <= t+1e-12)))
            rec[f"SC_ASR_tau_{t:g}"]=sc
            rec[f"retention_tau_{t:g}"]=sc/raw if raw>0 else np.nan
        rec["visibility_gap_at_0"]=raw-rec["SC_ASR_tau_0"]
        rows.append(rec)
    pd.DataFrame(rows).to_csv(out/"raw_vs_sc_asr.csv",index=False)

    # Full tau curves in tidy form.
    tr=[]
    for key,g in cc.groupby(grp,dropna=False):
        succ=g["attack_success_on_clean_correct"].astype(bool).to_numpy(); DA=g["D_A"].to_numpy(float); raw=float(succ.mean())
        for t in TAU_GRID:
            sc=float(np.mean(succ & (DA<=t+1e-12)))
            tr.append({**dict(zip(grp,key)),"tau":t,"raw_ASR":raw,"SC_ASR":sc,"retention":sc/raw if raw>0 else np.nan,
                       "SPR":float(np.mean(DA<=t+1e-12))})
    pd.DataFrame(tr).to_csv(out/"tau_curves_all.csv",index=False)

    # Temporal efficiency for retiming methods only.
    er=[]; rt=cc[cc.method.isin(["yu_pil_l0","free_retiming","null_space"]) & cc.avg_abs_shift_ms.notna()]
    for key,g in rt.groupby(grp,dropna=False):
        succ=g["attack_success_on_clean_correct"].astype(bool).to_numpy(); dt=g["avg_abs_shift_ms"].to_numpy(float)
        for e in EPS_GRID_MS:
            er.append({**dict(zip(grp,key)),"epsilon_ms":e,"A_eps":float(np.mean(succ & (dt<=e))),
                       "raw_ASR":float(succ.mean())})
    pd.DataFrame(er).to_csv(out/"temporal_efficiency_all.csv",index=False)

    # Does representation distortion correspond to protected-consumer visibility?
    sens=[]
    for key,g in cc.groupby(grp,dropna=False):
        da=g["D_A"].to_numpy(float); ld=g["protected_consumer_max_abs_logit_difference"].to_numpy(float)
        flips=g["protected_consumer_flip"].astype(int).to_numpy()
        sens.append({**dict(zip(grp,key)),"n":len(g),"spearman_DA_logitdiff":_rankcorr(da,ld),
                     "mean_DA":float(np.nanmean(da)),"protected_flip_rate":float(np.mean(flips)),
                     "mean_logit_diff":float(np.nanmean(ld))})
    pd.DataFrame(sens).to_csv(out/"metric_protected_sensitivity.csv",index=False)

    # Operating-point table for Pareto plotting. Native arms remain explicitly native.
    pareto=(cc.groupby(grp,dropna=False)
              .agg(n=("sample_index","count"),raw_ASR=("attack_success_on_clean_correct","mean"),
                   mean_D_A=("D_A","mean"),mean_footprint_pct=("realized_footprint_pct","mean"),
                   protected_flip_rate=("protected_consumer_flip","mean"),mean_shift_ms=("avg_abs_shift_ms","mean"))
              .reset_index())
    pareto.to_csv(out/"pareto_operating_points.csv",index=False)

    # Exact-stealth headline: useful for the metric contribution table.
    exact=(pd.DataFrame(rows)[["dataset","victim","method","seed","budget_label","n","raw_ASR","SC_ASR_tau_0","visibility_gap_at_0","mean_D_A","mean_footprint_pct"]])
    exact.to_csv(out/"exact_stealth_headline.csv",index=False)
    print(f"wrote metric validation tables to {out} from {len(cc)} clean-correct sample rows")

if __name__=="__main__": main()

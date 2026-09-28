#!/usr/bin/env python3
"""Fast file/protocol preflight before spending GPU time on ablations."""
from __future__ import annotations
import os, sys, json
from pathlib import Path
import numpy as np
import pandas as pd
from common.ablation_datasets import ABLATION_SUITE as SUITE
from common.runner import _find_ckpt

PRIMARY = {
    "dvsgesture": ["conv_snn","sew_resnet18","event_transformer_v2","temporal_gru"],
    "cifar10dvs": ["event_transformer_v2","temporal_gru"],
    "dailydvs200": ["swin_t","timesformer","mvfnet","actionnet"],
}

def latest_five(root,spec):
    ds=sorted((root/"results").glob(f"five_attack_{spec.key}_*"),key=lambda p:p.stat().st_mtime if p.exists() else 0)
    return ds[-1] if ds else None

def manifest(root,spec):
    mans=sorted(root.glob("results*/attack_subset_manifest.csv"))
    if mans: return mans[0]
    return None

def check_dataset(key):
    spec=SUITE[key]; root=Path(os.getenv(f"{key.upper()}_RUN_ROOT", os.getenv("RUN_ROOT",spec.default_run_root)))
    rec={"dataset":key,"root":str(root),"root_exists":root.exists()}
    man=manifest(root,spec); rec["manifest"] = str(man) if man else "MISSING"; rec["manifest_ok"]=bool(man and man.exists())
    rec["duration_us_ok"] = None
    if man and man.exists():
        try:
            df=pd.read_csv(man)
            rec["manifest_rows"]=len(df)
            if len(df) and "path" in df:
                p=Path(str(df.iloc[0]["path"])); rec["first_tensor_exists"]=p.exists()
                if p.exists():
                    with np.load(p) as d: rec["duration_us_ok"] = "duration_us" in d.files
        except Exception as e: rec["manifest_error"]=repr(e)
    roles=[spec.consumer_role]+PRIMARY.get(key,list(spec.victim_classes))
    ck={}
    for role in roles:
        try: ck[role]=str(_find_ckpt(root,role,0))
        except Exception as e: ck[role]=f"MISSING: {e}"
    rec["checkpoints_seed0"]=ck
    rec["checkpoints_ok"]=all(not str(v).startswith("MISSING") for v in ck.values())
    f=latest_five(root,spec); rec["latest_5x5_dir"]=str(f) if f else "MISSING"
    if f:
        rec["five_per_sample_rows"]=(f/"per_sample_rows.csv").exists()
        rec["five_partials_n"]=len(list((f/"partials").glob("*/*.json"))) if (f/"partials").exists() else 0
    return rec

def main():
    keys=[x for x in os.getenv("PREFLIGHT_DATASETS","dvsgesture,cifar10dvs,dailydvs200,nmnist,ncaltech101").split(",") if x]
    rows=[check_dataset(k) for k in keys]
    print("\n=== ABLATION PREFLIGHT ===")
    hard_fail=False
    for r in rows:
        primary=r["dataset"] in PRIMARY
        ok=r["root_exists"] and r["checkpoints_ok"] and (r["manifest_ok"] or r["dataset"]=="nmnist")
        if primary and not ok: hard_fail=True
        print(f"\n[{r['dataset']}] {'READY' if ok else 'NOT READY'}")
        print(" root:",r["root"])
        print(" manifest:",r.get("manifest"),"rows=",r.get("manifest_rows","?"),"duration_us=",r.get("duration_us_ok"))
        for role,p in r["checkpoints_seed0"].items(): print(f" ckpt {role}: {p}")
        print(" latest 5x5:",r.get("latest_5x5_dir"),"per_sample=",r.get("five_per_sample_rows",False),"partials=",r.get("five_partials_n",0))
    Path("preflight_report.json").write_text(json.dumps(rows,indent=2))
    print("\nWrote preflight_report.json")
    if hard_fail:
        print("\nCRITICAL: at least one priority ablation dataset is missing a manifest or seed-0 checkpoint.")
        if os.getenv("PREFLIGHT_STRICT","1")=="1": sys.exit(2)
    else:
        print("\nPriority datasets have the files needed to start seed-0 ablations.")

if __name__=="__main__": main()

from __future__ import annotations
import os, json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd

from .ablation_datasets import ABLATION_SUITE as SUITE
from .representation import Grid
from .frozen_loader import merge_definitions
from .runner import _device, _find_ckpt, _model, _logits, _subset, _sha
from .ablation_ops import Observer, constrained_attack, observer_metrics


def run(dataset_key="dvsgesture"):
    spec = SUITE[dataset_key]
    if dataset_key not in {"dvsgesture", "cifar10dvs"}:
        raise SystemExit("cross-victim transfer is scoped to DVS-Gesture/CIFAR10-DVS")
    root = Path(os.getenv("RUN_ROOT", spec.default_run_root))
    dev = _device(); ns = merge_definitions(spec.frozen_scripts, spec.overrides); grid = Grid(*spec.grid)
    seed = int(os.getenv("ABLATION_SEED", "0")); budget = float(os.getenv("ABLATION_BUDGET", "0.10"))
    source = os.getenv("TRANSFER_SOURCE", "temporal_gru")
    targets = [x for x in os.getenv("TRANSFER_TARGETS", ",".join(spec.victim_classes)).split(",") if x]
    if source not in spec.victim_classes:
        raise SystemExit(f"unknown TRANSFER_SOURCE={source}; choices={list(spec.victim_classes)}")
    for t in targets:
        if t not in spec.victim_classes:
            raise SystemExit(f"unknown transfer target {t}")
    cfg = {"dataset":dataset_key,"source":source,"targets":targets,"seed":seed,"budget":budget,"version":"transfer_v1"}
    rid = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:14]
    out = root / "results" / f"ablations_{dataset_key}_transfer_{rid}"; out.mkdir(parents=True, exist_ok=True)
    (out/"protocol.json").write_text(json.dumps(cfg, indent=2))

    X,y,dur,man = _subset(spec, root, ns)
    limit = int(os.getenv("ABLATION_N", "0"))
    if limit > 0: X,y,dur = X[:limit],y[:limit],dur[:limit]

    src_ck = _find_ckpt(root, source, seed)
    src_model = _model(ns, spec.victim_classes[source], src_ck, dev)
    src_fn = _logits(src_model, dev, int(spec.overrides.get("EVAL_BATCH",32)))
    src_clean = src_fn(X).argmax(1)

    target_models = {}
    target_clean = {}
    target_fns = {}
    for t in targets:
        ck = _find_ckpt(root, t, seed)
        m = _model(ns, spec.victim_classes[t], ck, dev)
        f = _logits(m, dev, int(spec.overrides.get("EVAL_BATCH",32)))
        target_models[t] = (m,ck); target_fns[t] = f; target_clean[t] = f(X).argmax(1)

    canonical = Observer(grid.S,0)
    rows=[]
    for i in range(len(X)):
        if src_clean[i] != y[i]:
            continue
        res = constrained_attack(src_model, X[i], int(y[i]), [budget], [canonical], dev)[budget]
        om = observer_metrics(X[i], res["X"], canonical)
        if not om["operator_exact_equal"]:
            raise AssertionError("canonical observer equality violated")
        bin_ms = (dur[i]/grid.T/1000.) if dur[i] > 0 else spec.fallback_bin_ms
        shifts = np.asarray(res["shifts"])
        for t in targets:
            clean_pred = int(target_clean[t][i]); adv_pred = int(target_fns[t](res["X"][None]).argmax(1)[0])
            rows.append({
                "dataset":dataset_key,"seed":seed,"source_victim":source,"target_victim":t,"sample_index":i,
                "target_clean_correct":bool(clean_pred==y[i]),
                "source_clean_correct":True,
                "transfer_attack_success":bool(clean_pred==y[i] and adv_pred!=y[i]),
                "target_clean_pred":clean_pred,"target_adv_pred":adv_pred,
                "D_A_operator":om["D_A_operator"],"D_inf_operator":om["D_inf_operator"],
                "operator_exact_equal":om["operator_exact_equal"],
                "moved_event_units":int(len(shifts)),
                "avg_abs_shift_ms":float(np.abs(shifts).mean()*bin_ms) if len(shifts) else 0.0,
            })
    df=pd.DataFrame(rows); df.to_csv(out/"transfer_rows.csv",index=False)
    if len(df):
        sm=(df[df.target_clean_correct].groupby(["source_victim","target_victim"],dropna=False)
            .agg(n=("sample_index","count"),transfer_ASR=("transfer_attack_success","mean"),
                 exact_observer=("operator_exact_equal","mean"),mean_shift_ms=("avg_abs_shift_ms","mean"))
            .reset_index())
        sm.to_csv(out/"transfer_matrix.csv",index=False)
    print("wrote",out)
    return out

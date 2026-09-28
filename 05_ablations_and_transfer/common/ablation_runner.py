from __future__ import annotations
import os, json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from .ablation_datasets import ABLATION_SUITE as SUITE
from .representation import Grid
from .frozen_loader import merge_definitions
from .runner import _device, _find_ckpt, _model, _logits, _subset, _sha
from . import metrics
from .attacks.controls import uniform_control, exact_matched
from .ablation_ops import (Observer, constrained_attack, one_shot_attack, observer_metrics,
                           full_nullity, observer_family_metrics, main_protocol_refresh)

ABLATIONS = ["window", "phase", "intersection", "observer_family", "locality", "controls", "refresh", "randomized"]

def _victims_for(spec, ablation):
    env = os.getenv("ABLATION_VICTIMS", "").strip()
    if env:
        return [x for x in env.split(",") if x]
    if spec.key == "dvsgesture":
        return list(spec.victim_classes)
    if spec.key == "cifar10dvs":
        return [x for x in ["temporal_gru", "event_transformer_v2"] if x in spec.victim_classes]
    if spec.key == "dailydvs200":
        return list(spec.victim_classes)
    return list(spec.victim_classes)[:1]

def _parse_ints(name, default):
    return [int(x) for x in os.getenv(name, default).split(",") if str(x).strip()]

def _run_id(spec, ablation, victims, seed, cfg):
    obj = {"dataset": spec.key, "ablation": ablation, "victims": victims, "seed": seed,
           "grid": spec.grid, "cfg": cfg, "version": "ablation_v1"}
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:14], obj

def _verify_clean_accuracy(root, spec, model_role, seed, measured, n_samples):
    """Compare measured subset clean accuracy against the landed frozen run.

    This is the executable proof that AST-extracted model definitions plus the
    globbed checkpoint reproduce the frozen semantics. The attack subset is a
    subset of the test set, so the tolerance is the protocol tolerance plus
    sampling slack; a gross mismatch (wrong checkpoint, wrong class) is what
    this catches.
    """
    import pandas as _pd
    cands = list((root / "results").glob("**/clean_accuracy_by_seed.csv")) + \
            list((root / "results").glob("**/addon_clean_accuracy_by_seed.csv"))
    if not cands:
        print(f"WARNING clean-accuracy guard: no landed CSV under {root}/results; skipping")
        return
    for fp in cands:
        try:
            df = _pd.read_csv(fp)
        except Exception:
            continue
        if not {"model", "seed", "clean_test_accuracy"} <= set(df.columns):
            continue
        hit = df[(df.model == model_role) & (df.seed == seed)]
        if len(hit) != 1:
            continue
        ref = float(hit.clean_test_accuracy.iloc[0])
        tol = spec.clean_tol_pp / 100.0 + 2.0 * (max(ref * (1 - ref), 1e-6) / max(n_samples, 1)) ** 0.5
        if abs(measured - ref) > tol:
            raise AssertionError(
                f"clean-accuracy guard FAILED for {model_role} seed{seed}: "
                f"subset={measured:.4f} landed={ref:.4f} tol={tol:.4f} ({fp}). "
                "Wrong checkpoint or drifted model definition.")
        print(f"clean-accuracy guard OK: {model_role} seed{seed} "
              f"subset={measured:.4f} vs landed={ref:.4f} (tol {tol:.4f})")
        return
    print(f"WARNING clean-accuracy guard: no row for {model_role} seed{seed}; skipping")


def _attack_success(vfn, Xa, y, clean_pred):
    adv = int(vfn(Xa[None]).argmax(1)[0])
    return bool(clean_pred == y and adv != y), adv

def _base_row(spec, victim, seed, idx, y, clean_pred, name, condition, X0, Xa, shifts, bin_ms, vfn):
    success, adv = _attack_success(vfn, Xa, y, clean_pred)
    d = Xa.astype(np.int64) - X0.astype(np.int64)
    s = None if shifts is None else np.asarray(shifts)
    return {
        "dataset": spec.key, "victim": victim, "seed": seed, "sample_index": idx,
        "ablation": name, "condition": condition, "clean_correct": bool(clean_pred == y),
        "attack_success": success, "victim_pred_adv": adv,
        "total_event_units": int(X0.sum()), "fine_tensor_L1": int(np.abs(d).sum()),
        "realized_footprint_pct": 100.0 * int(np.abs(d).sum()) / max(1, 2 * int(X0.sum())),
        "moved_event_units": int(len(s)) if s is not None else np.nan,
        "avg_abs_shift_bins": float(np.abs(s).mean()) if s is not None and len(s) else (0.0 if s is not None else np.nan),
        "avg_abs_shift_ms": float(np.abs(s).mean() * bin_ms) if s is not None and len(s) else (0.0 if s is not None else np.nan),
        "max_abs_shift_ms": float(np.abs(s).max() * bin_ms) if s is not None and len(s) else (0.0 if s is not None else np.nan),
    }

def run(dataset_key, ablation=None):
    spec = SUITE[dataset_key]
    ablation = ablation or os.getenv("ABLATION", "phase")
    if ablation not in ABLATIONS:
        raise SystemExit(f"ABLATION must be one of {ABLATIONS}")
    if dataset_key not in {"dvsgesture", "cifar10dvs", "dailydvs200"}:
        raise SystemExit("mechanism ablations are intentionally scoped to DVS-Gesture/CIFAR10-DVS/DailyDVS-200")
    if dataset_key == "dailydvs200" and ablation not in {"phase", "observer_family", "randomized"}:
        raise SystemExit("DailyDVS-200 is scoped to cheap phase/observer-family/randomized confirmations")

    root = Path(os.getenv("RUN_ROOT", spec.default_run_root))
    dev = _device()
    ns = merge_definitions(spec.frozen_scripts, spec.overrides)
    grid = Grid(*spec.grid)
    seed = int(os.getenv("ABLATION_SEED", "0"))
    victims = _victims_for(spec, ablation)
    budget = float(os.getenv("ABLATION_BUDGET", "0.10"))
    cfg = {
        "budget": budget,
        "window_values": _parse_ints("WINDOW_VALUES", "2,4,8,16"),
        "phase_offsets": _parse_ints("PHASE_OFFSETS", f"0,1,2,{max(1,grid.S//2)}"),
        "intersection_phases": _parse_ints("INTERSECTION_PHASES", "0,2,4"),
        "locality_bins": [x for x in os.getenv("LOCALITY_BINS", "1,2,4,none").split(",") if x],
        "random_widths": _parse_ints("RANDOM_WIDTHS", f"{max(2,grid.S-2)},{max(2,grid.S-1)},{grid.S},{grid.S+1},{grid.S+2}"),
        "random_trials": int(os.getenv("RANDOM_TRIALS", "32")),
    }
    limit = int(os.getenv("ABLATION_N", "0"))
    cfg["sample_limit"] = limit          # smoke runs get their own result dir
    rid, proto = _run_id(spec, ablation, victims, seed, cfg)
    out = root / "results" / f"ablations_{spec.key}_{ablation}_{rid}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "protocol.json").write_text(json.dumps(proto, indent=2))

    X, y, dur, man = _subset(spec, root, ns)
    if limit > 0:
        X, y, dur = X[:limit], y[:limit], dur[:limit]

    rows = []
    part = out / "partials"
    part.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(os.getenv("ABLATION_RNG", "2027")))
    for victim_role in victims:
        # Resume: a completed victim unit is reused rather than recomputed.
        pfp = part / f"{ablation}_seed{seed}_{victim_role}.json"
        if pfp.exists() and os.getenv("FORCE_ABLATION", "0") != "1":
            cached = json.loads(pfp.read_text())
            rows += cached["rows"]
            print("reuse", pfp, len(cached["rows"]))
            continue
        victim_rows = []
        vck = _find_ckpt(root, victim_role, seed)
        victim = _model(ns, spec.victim_classes[victim_role], vck, dev)
        vfn = _logits(victim, dev, int(spec.overrides.get("EVAL_BATCH", 32)))
        clean_preds = vfn(X).argmax(1)
        measured_acc = float((clean_preds == y).mean())
        if limit <= 0 and os.getenv("SKIP_CLEAN_GUARD", "0") != "1":
            _verify_clean_accuracy(root, spec, victim_role, seed, measured_acc, len(X))
        else:
            print(f"clean-accuracy guard skipped (smoke or opt-out): "
                  f"{victim_role} seed{seed} subset={measured_acc:.4f}")
        for i in range(len(X)):
            if clean_preds[i] != y[i]:
                continue
            X0 = X[i]
            bin_ms = (dur[i] / grid.T / 1000.0) if dur[i] > 0 else spec.fallback_bin_ms
            canonical = Observer(grid.S, 0)

            if ablation == "window":
                for S in cfg["window_values"]:
                    if S <= 1 or S > grid.T or grid.T % S:
                        continue
                    obs = [Observer(S, 0)]
                    res = constrained_attack(victim, X0, int(y[i]), [budget], obs, dev, refresh_points=main_protocol_refresh(budget))[budget]
                    r = _base_row(spec, victim_role, seed, i, int(y[i]), int(clean_preds[i]), ablation,
                                  f"S={S}", X0, res["X"], res["shifts"], bin_ms, vfn)
                    r.update(observer_metrics(X0, res["X"], obs[0]))
                    r["blind_dimension"] = full_nullity(grid.T, grid.C, grid.H, grid.W, obs)
                    r["blind_dimension_fraction"] = r["blind_dimension"] / float(grid.T * grid.C * grid.H * grid.W)
                    victim_rows.append(r)

            elif ablation == "phase":
                res = constrained_attack(victim, X0, int(y[i]), [budget], [canonical], dev, refresh_points=main_protocol_refresh(budget))[budget]
                for ph in cfg["phase_offsets"]:
                    obs = Observer(grid.S, ph)
                    r = _base_row(spec, victim_role, seed, i, int(y[i]), int(clean_preds[i]), ablation,
                                  f"phase={ph}", X0, res["X"], res["shifts"], bin_ms, vfn)
                    r.update(observer_metrics(X0, res["X"], obs))
                    r["attack_constructed_for_phase"] = 0
                    victim_rows.append(r)

            elif ablation == "intersection":
                phases = cfg["intersection_phases"]
                for n in range(1, len(phases) + 1):
                    obs = [Observer(grid.S, p) for p in phases[:n]]
                    res = constrained_attack(victim, X0, int(y[i]), [budget], obs, dev, refresh_points=main_protocol_refresh(budget))[budget]
                    r = _base_row(spec, victim_role, seed, i, int(y[i]), int(clean_preds[i]), ablation,
                                  "phases=" + "+".join(map(str, phases[:n])), X0, res["X"], res["shifts"], bin_ms, vfn)
                    r["blind_dimension"] = full_nullity(grid.T, grid.C, grid.H, grid.W, obs)
                    r["blind_dimension_fraction"] = r["blind_dimension"] / float(grid.T * grid.C * grid.H * grid.W)
                    r["all_observers_exact"] = all(observer_metrics(X0, res["X"], o)["operator_exact_equal"] for o in obs)
                    victim_rows.append(r)

            elif ablation == "observer_family":
                # Construct exactly once for the canonical boxcar, then hold the
                # perturbation fixed while only the observer representation changes.
                res = constrained_attack(victim, X0, int(y[i]), [budget], [canonical], dev, refresh_points=main_protocol_refresh(budget))[budget]
                for fm in observer_family_metrics(X0, res["X"], grid.S):
                    r = _base_row(spec, victim_role, seed, i, int(y[i]), int(clean_preds[i]), ablation,
                                  fm["observer_family"] + ":" + fm["observer_detail"],
                                  X0, res["X"], res["shifts"], bin_ms, vfn)
                    r.update(fm)
                    victim_rows.append(r)

            elif ablation == "locality":
                for txt in cfg["locality_bins"]:
                    mx = None if txt.lower() == "none" else int(txt)
                    res = constrained_attack(victim, X0, int(y[i]), [budget], [canonical], dev, max_shift_bins=mx, refresh_points=main_protocol_refresh(budget))[budget]
                    r = _base_row(spec, victim_role, seed, i, int(y[i]), int(clean_preds[i]), ablation,
                                  f"max_shift={txt}", X0, res["X"], res["shifts"], bin_ms, vfn)
                    r.update(observer_metrics(X0, res["X"], canonical))
                    victim_rows.append(r)

            elif ablation == "controls":
                res = constrained_attack(victim, X0, int(y[i]), [budget], [canonical], dev, refresh_points=main_protocol_refresh(budget))[budget]
                base = _base_row(spec, victim_role, seed, i, int(y[i]), int(clean_preds[i]), ablation,
                                 "optimized", X0, res["X"], res["shifts"], bin_ms, vfn)
                base.update(observer_metrics(X0, res["X"], canonical)); victim_rows.append(base)
                Xu, su = uniform_control(grid, X0, len(res["shifts"]), rng)
                ru = _base_row(spec, victim_role, seed, i, int(y[i]), int(clean_preds[i]), ablation,
                               "uniform_random", X0, Xu, su, bin_ms, vfn)
                ru.update(observer_metrics(X0, Xu, canonical)); victim_rows.append(ru)
                Xm, sm, ok = exact_matched(grid, X0, res["shifts"], rng)
                rm = _base_row(spec, victim_role, seed, i, int(y[i]), int(clean_preds[i]), ablation,
                               "exact_displacement_matched", X0, Xm, sm, bin_ms, vfn)
                if not ok:
                    raise AssertionError(
                        f"exact displacement-matched control infeasible "
                        f"(dataset={spec.key} victim={victim_role} seed={seed} sample={i}). "
                        "The protocol requires zero fallback; investigate rather than record a miss.")
                rm.update(observer_metrics(X0, Xm, canonical)); rm["exact_match"] = bool(ok); victim_rows.append(rm)

            elif ablation == "refresh":
                schedules = {
                    "one_shot": None,
                    "5_10": [0.05, budget],
                    "1_2_5_10": [x for x in [0.01, 0.02, 0.05, budget] if x <= budget],
                }
                for name, sch in schedules.items():
                    if name == "one_shot":
                        res = one_shot_attack(victim, X0, int(y[i]), budget, [canonical], dev)[budget]
                    else:
                        res = constrained_attack(victim, X0, int(y[i]), [budget], [canonical], dev, refresh_points=sch)[budget]
                    r = _base_row(spec, victim_role, seed, i, int(y[i]), int(clean_preds[i]), ablation,
                                  name, X0, res["X"], res["shifts"], bin_ms, vfn)
                    r.update(observer_metrics(X0, res["X"], canonical)); victim_rows.append(r)

            elif ablation == "randomized":
                # Construct once for the canonical observer, then test exactly the same
                # perturbation under randomized observation schedules.
                res = constrained_attack(victim, X0, int(y[i]), [budget], [canonical], dev, refresh_points=main_protocol_refresh(budget))[budget]
                # schedule-blind reference: canonical observer (guaranteed invisible)
                mc = observer_metrics(X0, res["X"], canonical)
                r0 = _base_row(spec, victim_role, seed, i, int(y[i]), int(clean_preds[i]), ablation,
                               "fixed_canonical", X0, res["X"], res["shifts"], bin_ms, vfn)
                r0.update(mc); r0["random_trial"] = -1; victim_rows.append(r0)
                for tr in range(cfg["random_trials"]):
                    S = int(rng.choice(cfg["random_widths"]))
                    S = max(2, min(S, grid.T))
                    ph = int(rng.integers(0, S))
                    obs = Observer(S, ph)
                    rr = _base_row(spec, victim_role, seed, i, int(y[i]), int(clean_preds[i]), ablation,
                                   "random_schedule", X0, res["X"], res["shifts"], bin_ms, vfn)
                    rr.update(observer_metrics(X0, res["X"], obs)); rr["random_trial"] = tr
                    rr["schedule_aware_visible"] = bool(rr["D_A_operator"] > 0)
                    rr["schedule_blind_visible"] = bool(mc["D_A_operator"] > 0)
                    victim_rows.append(rr)

        tmp = pfp.with_suffix(".tmp")
        tmp.write_text(json.dumps({"dataset": spec.key, "ablation": ablation, "seed": seed,
                                   "victim": victim_role, "victim_ckpt": vck.name,
                                   "victim_sha": _sha(vck), "rows": victim_rows},
                                  allow_nan=True))
        os.replace(tmp, pfp)
        rows += victim_rows
        print("wrote", pfp, len(victim_rows))

    df = pd.DataFrame(rows)
    df.to_csv(out / "per_sample_ablation_rows.csv", index=False)
    if len(df):
        group_cols = ["dataset", "victim", "ablation", "condition"]
        agg = df.groupby(group_cols, dropna=False).agg(
            n=("sample_index", "count"),
            ASR=("attack_success", "mean"),
            mean_shift_ms=("avg_abs_shift_ms", "mean"),
            mean_footprint_pct=("realized_footprint_pct", "mean"),
        ).reset_index()
        for c in ["D_A_operator", "D_inf_operator", "D_A_family", "D_inf_family", "family_exact_equal", "blind_dimension", "blind_dimension_fraction", "schedule_aware_visible", "schedule_blind_visible", "exact_match"]:
            if c in df.columns:
                x = df.groupby(group_cols, dropna=False)[c].mean().reset_index(name=f"mean_{c}")
                agg = agg.merge(x, on=group_cols, how="left")
        agg.to_csv(out / "ablation_summary.csv", index=False)
    print("wrote", out)
    return out

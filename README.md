# Code release: event retiming and consumer-relative stealth

This repository contains the code used for the experiments reported in the accompanying paper. It is organized by experiment family rather than by development version. Exploratory or superseded duplicate snapshots are omitted. Scientific logic, model definitions, attacks, metrics, random seeds, and experiment protocols are preserved; packaging-only edits remove site-specific paths, scheduler identity output, and duplicated frozen copies.

## 1. Repository map

| Folder | Paper experiment / purpose | Main entry points |
|---|---|---|
| `01_dataset_protocols/` | Dataset preprocessing, model training, headline exact-null retiming, controls, and dataset-specific diagnostics | dataset scripts listed below |
| `02_dvsgesture_direct_comparison/prior_retiming_comparison/` | DVS128 Gesture direct comparison of prior PIL-L0 retiming, unconstrained retiming, and null-space retiming | `ea_direct_compare.py`, `run_paper.sh` |
| `03_five_dataset_five_attack_suite/` | Unified five-dataset × five-attack comparison: Yu/PIL-L0, PDSG-SDA, Yao/Gumbel, free retiming, null-space retiming | five `EVS_*_5ATTACK.py` launchers, `run_25_array.slurm`, `aggregate_all.py` |
| `04_attack_defense_campaign/` | Combined attack/observer/defense evaluation on DVS128 Gesture, DailyDVS-200, and CIFAR10-DVS | `run_combined.py`, `submit_dataset_parallel.sh`, `launch_local_parallel.py` |
| `05_ablations_and_transfer/` | Observer-family ablations, intersection/window/locality/refresh studies, cross-victim transfer, and post-hoc metric validation | `EVS_*_ABLATIONS.py`, `EVS_*_TRANSFER.py`, `posthoc_metric_validation.py` |

The multi-dataset suites load the canonical model definitions directly from `01_dataset_protocols/` and the landed DVS direct-comparison implementation in `02_dvsgesture_direct_comparison/`. Redundant `frozen_scripts/` copies are intentionally removed.

## 2. Environment

Recommended environment: Python 3.10+ with a CUDA-capable PyTorch installation.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The official PDSG-SDA implementation used here is compatible with the indexing behavior of `torch<2.9`; the requirements file therefore pins the supported range. Set `WANDB_MODE=disabled` when running the Yao/Gumbel code if `wandb` is installed.

Three external attack repositories are not vendored. For the comparison suites, either run the provided `setup_official_repos.sh` in the relevant folder or clone the repositories yourself and export:

```bash
export SPIKE_RETIMING_REPO=/path/to/Spike-Retiming-Attacks
export PDSG_SDA_DIR=/path/to/PDSG-SDA
export YAO_DIR=/path/to/GumbelSoftmaxAttack
export WANDB_MODE=disabled
```

The Yu/PIL-L0 helper pins the revision used by the experiments. Do not substitute a different revision without revalidating the comparison.

## 3. Dataset protocol runs

All run roots should be placed outside this code repository. Later experiments reuse the completed run roots, cached tensors, checkpoints, and attack-subset manifests produced here.

### N-MNIST: original protocol

Entry point: `01_dataset_protocols/nmnist/EVS_NMNIST_v2.py`.

This is the executed headless version of the original N-MNIST experiment. It uses the native 5-byte parser and downloads the public N-MNIST archives automatically.

```bash
cd 01_dataset_protocols/nmnist
python3 EVS_NMNIST_v2.py
```

The script contains a portable placeholder for its output root near the top; edit that placeholder before running.

### N-MNIST: aligned five-model protocol

Entry point: `01_dataset_protocols/nmnist/aligned5/EVS_NMNIST_ALIGNED5_FINAL.py`.

This is the aligned protocol used by the unified five-model/five-attack analysis. Set the dataset and run roots:

```bash
export NMNIST_DATA=/path/to/nmnist
export NMNIST_RUN=/path/to/runs/nmnist_aligned5
cd 01_dataset_protocols/nmnist/aligned5
```

Train first, then run attacks:

```bash
RUN_ATTACKS=0 python3 EVS_NMNIST_ALIGNED5_FINAL.py
python3 EVS_NMNIST_ALIGNED5_FINAL.py
python3 validate_outputs.py
```

Equivalent Slurm helpers are `run_train.slurm`, `run_attacks.slurm`, `submit_2stage.sh`, and `submit_chain.sh`.

### DVS128 Gesture

Base run: `01_dataset_protocols/dvsgesture/EVS_DVSGesture.py`.
Additional temporal models / protected consumer: `01_dataset_protocols/dvsgesture/EVS_DVSGesture_addons.py`.
Corrected diagnostic-only rerun: `01_dataset_protocols/dvsgesture/EVS_DVSGesture_PROTOCOL_CLEANUP_V2.py`.

The base script expects the pre-segmented gesture `.npy` layout described in its header. Set the portable `ROOT` and `DVS_NPY_ROOT` placeholders at the top of the script, then run the base protocol followed by the add-ons. The cleanup script reads the completed run root and recomputes only the diagnostic quantities; it does not retrain models or replace headline attack tables.

### CIFAR10-DVS

Main protocol: `01_dataset_protocols/cifar10dvs/EVS_CIFAR10DVS_FINAL.py`.
Diagnostic-only correction: `01_dataset_protocols/cifar10dvs/EVS_CIFAR10DVS_DIAGNOSTICS_V2_FIXED.py`.

```bash
export CIFAR_DVS_DATA=/path/to/cifar10dvs
export CIFAR_DVS_RUN=/path/to/runs/cifar10dvs
cd 01_dataset_protocols/cifar10dvs
RUN_ATTACKS=0 python3 EVS_CIFAR10DVS_FINAL.py
python3 EVS_CIFAR10DVS_FINAL.py
python3 EVS_CIFAR10DVS_DIAGNOSTICS_V2_FIXED.py
```

The final protocol uses three seeds, the five model families used in the paper, `T_FINE=80`, exact within-window matching, and the progressive attack budgets encoded in the script.

### N-Caltech101

Entry point: `01_dataset_protocols/ncaltech101/EVS_NCALTECH101_FINAL_v3_READY.py`.

```bash
export NCALTECH101_DATA=/path/to/ncaltech101
export NCALTECH101_RUN=/path/to/runs/ncaltech101
cd 01_dataset_protocols/ncaltech101
RUN_ATTACKS=0 python3 EVS_NCALTECH101_FINAL_v3_READY.py
python3 EVS_NCALTECH101_FINAL_v3_READY.py
```

The loader accepts the extracted class folders used by the experiment and applies the fixed letterbox preprocessing encoded in the script.

### DailyDVS-200

Baseline protocol: `01_dataset_protocols/dailydvs200/EVS_DAILYDVS200_HF_FULL_v2.py`.
Stronger-model extension used by the ablation campaign: `01_dataset_protocols/dailydvs200/EVS_DAILYDVS200_SOTA4_REUSE_FINAL.py`.

```bash
export DAILYDVS_DATA=/path/to/dailydvs200
export DAILYDVS_SPLIT_DIR=/path/to/dailydvs200_splits
export DAILYDVS_RUN=/path/to/runs/dailydvs200
cd 01_dataset_protocols/dailydvs200
RUN_ATTACKS=0 python3 EVS_DAILYDVS200_HF_FULL_v2.py
python3 EVS_DAILYDVS200_HF_FULL_v2.py
python3 EVS_DAILYDVS200_SOTA4_REUSE_FINAL.py
```

The baseline script can obtain the public dataset through the configured Hugging Face repository and caches decoded event tensors. The stronger-model script reuses the completed baseline run rather than rebuilding the full protocol.

### SHD / SSC cross-modal replication

Entry point: `01_dataset_protocols/shd_ssc/EVS_SHD_SSC_EVENTSECURITY_FINAL_v2.py`.

```bash
export HSD_ROOT=/path/to/runs/hsd
export HSD_DATASET=SHD   # or SSC
cd 01_dataset_protocols/shd_ssc
python3 EVS_SHD_SSC_EVENTSECURITY_FINAL_v2.py
```

## 4. DVS128 Gesture direct comparison

The canonical direct-comparison implementation is:

`02_dvsgesture_direct_comparison/prior_retiming_comparison/ea_direct_compare.py`

It compares the official Yu/PIL-L0 implementation with free gradient retiming and the exact null-space retiming attack on the same frozen DVS128 Gesture population/checkpoints. The exact-null arm asserts zero protected coarse-frame difference and protected-consumer equality.

Setup:

```bash
cd 02_dvsgesture_direct_comparison/prior_retiming_comparison
export DVS_GESTURE_RUN=/path/to/runs/dvsgesture
export SPIKE_RETIMING_REPO=/path/to/Spike-Retiming-Attacks
```

Mandatory two-sample smoke:

```bash
RUN_SEEDS=0 \
VICTIMS=conv_snn \
PROTECTED_CONSUMER=coarse_frameformer \
DIRECT_BUDGETS=0.10 \
DIRECT_N=2 \
EXPECT_N=0 \
PRIOR_STEPS=2 \
PRIOR_RECALIBRATE=0 \
python3 ea_direct_compare.py
```

Full landed comparison:

```bash
RUN_SEEDS=0,1,2 \
VICTIMS=conv_snn \
PROTECTED_CONSUMER=coarse_frameformer \
DIRECT_BUDGETS=0.10 \
DIRECT_N=0 \
EXPECT_N=264 \
PRIOR_STEPS=40 \
PRIOR_RECALIBRATE=1 \
python3 ea_direct_compare.py
```

`run_paper.sh` and the accompanying Slurm files invoke the same driver.

## 5. Unified five-dataset × five-attack suite

Folder: `03_five_dataset_five_attack_suite/`.

Methods are `yu_pil_l0`, `pdsg_sda`, `yao_gumbel`, `free_retiming`, and `null_space`; the null-space jobs also emit the random-uniform and exact displacement-matched controls used in the analysis.

The five launchers are:

```text
EVS_NMNIST_5ATTACK.py
EVS_DVSGesture_5ATTACK.py
EVS_CIFAR10DVS_5ATTACK.py
EVS_NCaltech101_5ATTACK.py
EVS_DailyDVS200_5ATTACK.py
```

Before running, set the appropriate completed base-run root using `RUN_ROOT` and source the external repositories:

```bash
cd 03_five_dataset_five_attack_suite
source setup_official_repos.sh
export RUN_ROOT=/path/to/completed/dataset/run
export RUN_SEEDS=0,1,2
```

A direct single-method launch is:

```bash
METHOD=null_space python3 EVS_DVSGesture_5ATTACK.py
```

For a two-sample smoke:

```bash
DIRECT_N=2 RUN_SEEDS=0 METHOD=null_space python3 EVS_DVSGesture_5ATTACK.py
```

For the full scheduler campaign:

```bash
bash submit_25.sh
```

The DVS reconciliation stage can be launched first with `run_dvs_first.sh`; after all partials finish, aggregate with:

```bash
python3 aggregate_all.py
```

The suite reuses the landed attack-subset manifest when present and records hashes of checkpoints/manifests rather than silently choosing among ambiguous candidates.

## 6. Combined attack + observer/defense campaign

Folder: `04_attack_defense_campaign/`.

The reported campaign covers DVS128 Gesture, DailyDVS-200, and CIFAR10-DVS. It evaluates the five attacks with the ConvSNN victim and coarse protected consumer, observer-family exposure, clean-calibrated detection, UASR/SC-ASR quantities, and the controlled within-family representation-deviation sweep.

First edit/source the portable environment template:

```bash
cd 04_attack_defense_campaign
source campaign.env.example
```

Run built-in checks:

```bash
python3 self_test.py
```

Build a disjoint clean calibration pool using the completed run root and the dataset-specific arguments shown by:

```bash
python3 build_clean_pool_manifest.py --help
```

Then preflight:

```bash
python3 preflight.py \
  --dataset dvsgesture \
  --run-root /path/to/runs/dvsgesture \
  --clean-pool /path/to/dvs_clean_pool.csv \
  --direct-n 264
```

Timing smoke:

```bash
bash smoke_timing.sh dvsgesture /path/to/runs/dvsgesture /path/to/dvs_clean_pool.csv
```

Full Slurm launch examples:

```bash
./submit_dataset_parallel.sh dvsgesture  /path/to/runs/dvsgesture  /path/to/dvs_clean_pool.csv 264
./submit_dataset_parallel.sh dailydvs200 /path/to/runs/dailydvs200 /path/to/daily_clean_pool.csv 250
./submit_dataset_parallel.sh cifar10dvs  /path/to/runs/cifar10dvs  /path/to/cifar_clean_pool.csv 150
```

On a workstation without Slurm, use `launch_local_parallel.py` with the same dataset/run-root/clean-pool settings. `validate_campaign.py` checks completed partials before using the aggregated results.

## 7. Ablations, observer studies, and transfer

Folder: `05_ablations_and_transfer/`.

Main experiment families:

- `phase`: shifted observer boundaries.
- `intersection`: multiple observer constraints and blind-space intersection.
- `window`: protected-window width sensitivity.
- `observer_family`: canonical, shifted/overlap/multiscale observer visibility.
- `randomized`: randomized observer schedules.
- `locality`: restricted temporal displacement.
- `controls`: matched-effort / exact controls.
- `refresh`: optimizer refresh schedule.
- cross-victim transfer: craft on one victim and evaluate the unchanged perturbation on the others.
- `posthoc_metric_validation.py`: metric analysis over completed unified five-attack results.

Preflight and smoke:

```bash
cd 05_ablations_and_transfer
export EA_VENV=/path/to/venv
python3 preflight_ablations.py
export ABLATION_N=2
export ABLATION_SEED=0
```

Critical scheduler campaign:

```bash
./submit_critical_2day.sh
```

Optional supporting experiments:

```bash
./submit_optional_2day.sh
```

After completion:

```bash
python3 collect_ablation_results.py
python3 posthoc_metric_validation.py
```

Individual launchers can also be called directly by setting `RUN_ROOT` and the ablation environment variables used by `common/ablation_runner.py`.

## 8. Reproduction order

For a clean reproduction, use this order:

1. Run the dataset protocols in `01_dataset_protocols/` to obtain caches, checkpoints, manifests, and headline exact-null results.
2. Run the DVS direct comparison in `02_dvsgesture_direct_comparison/`.
3. Run the unified five-dataset × five-attack suite in `03_five_dataset_five_attack_suite/`.
4. Run the three-dataset observer/defense campaign in `04_attack_defense_campaign/`.
5. Run the mechanism ablations, transfer studies, and metric validation in `05_ablations_and_transfer/`.

The scripts are resumable where the original runs were resumable. Do not mix run roots from different protocol generations: the suites validate manifest/checkpoint identity and should be pointed to the run root that produced the corresponding base results.

## 9. Privacy and portability

This code package contains no data, checkpoints, result logs, cluster host names, scheduler allocation names, personal user names, institutional paths, or personal email addresses. Scheduler files use generic resources/placeholders only. Job helpers do not emit host identity. All site-specific dataset, run-root, virtual-environment, and external-repository locations are supplied through placeholders or environment variables.

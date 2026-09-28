#!/bin/bash
# FIX 7: stage A - all five attacks on DVS128 Gesture only (array ids 5..9),
# three seeds, before any other dataset. Reconcile against the landed
# direct-comparison numbers, then launch the rest.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs
source setup_official_repos.sh

# Reconciliation against the landed direct comparison needs the same 264-sample
# population it used, not results*/attack_subset_manifest.csv.
python3 pin_dvs_manifest.py
export ATTACK_MANIFEST=${ATTACK_MANIFEST:-/path/to/workspace/EVS_DVSGesture/run/results/pinned_dvs264_manifest.csv}
echo "ATTACK_MANIFEST=$ATTACK_MANIFEST"
export WANDB_MODE=${WANDB_MODE:-disabled}
jid=$(sbatch --parsable --array=5-9%3 run_25_array.slurm)
echo "DVS-only array: $jid  (5=yu 6=sda 7=yao 8=free 9=null)"
echo "aggregate after it finishes:  RUN_SEEDS=0,1,2 python aggregate_all.py"

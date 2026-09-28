#!/bin/bash
#SBATCH --job-name=ea-ablate
# The partition name below is a placeholder; set it for your cluster.
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-12%3
#SBATCH -o logs/ablate_%A_%a.out
#SBATCH -e logs/ablate_%A_%a.err
set -euo pipefail
mkdir -p logs
cd "$SLURM_SUBMIT_DIR"
# shellcheck disable=SC1091
source "$SLURM_SUBMIT_DIR/cluster_env.sh"

# 13 mechanism jobs: 7 DVS-Gesture + 4 CIFAR confirmations + 2 DailyDVS confirmations.
JOBS=(
  "dvsgesture window"
  "dvsgesture phase"
  "dvsgesture intersection"
  "dvsgesture locality"
  "dvsgesture controls"
  "dvsgesture refresh"
  "dvsgesture randomized"
  "cifar10dvs window"
  "cifar10dvs phase"
  "cifar10dvs locality"
  "cifar10dvs controls"
  "dailydvs200 phase"
  "dailydvs200 randomized"
)
read -r DATASET ABLATION <<< "${JOBS[$SLURM_ARRAY_TASK_ID]}"
export ABLATION
case "$DATASET" in
  dvsgesture) python3 EVS_DVSGesture_ABLATIONS.py; EXIT_CODE=$?; exit $EXIT_CODE ;;
  cifar10dvs) python3 EVS_CIFAR10DVS_ABLATIONS.py; EXIT_CODE=$?; exit $EXIT_CODE ;;
  dailydvs200) python3 EVS_DailyDVS200_ABLATIONS.py; EXIT_CODE=$?; exit $EXIT_CODE ;;
  *) echo "unknown dataset $DATASET" >&2; exit 2 ;;
esac

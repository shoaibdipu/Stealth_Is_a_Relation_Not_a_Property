#!/bin/bash
#SBATCH --job-name=ea-ablate-extra
# The partition name below is a placeholder; set it for your cluster.
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-13%3
#SBATCH -o logs/optional_%A_%a.out
#SBATCH -e logs/optional_%A_%a.err
set -euo pipefail
mkdir -p logs
cd "$SLURM_SUBMIT_DIR"
# shellcheck disable=SC1091
source "$SLURM_SUBMIT_DIR/cluster_env.sh"
export ABLATION_SEED=${ABLATION_SEED:-0}
export ABLATION_BUDGET=${ABLATION_BUDGET:-0.10}
JOBS=(
  "transfer dvsgesture conv_snn"
  "transfer dvsgesture sew_resnet18"
  "transfer dvsgesture event_transformer_v2"
  "transfer dvsgesture temporal_gru"
  "ablation dvsgesture refresh event_transformer_v2"
  "ablation dvsgesture refresh temporal_gru"
  "ablation dvsgesture controls conv_snn"
  "ablation dvsgesture controls sew_resnet18"
  "ablation dvsgesture controls event_transformer_v2"
  "ablation dvsgesture controls temporal_gru"
  "ablation cifar10dvs locality event_transformer_v2"
  "ablation cifar10dvs locality temporal_gru"
  "ablation cifar10dvs controls event_transformer_v2"
  "ablation cifar10dvs controls temporal_gru"
)
set -- ${JOBS[$SLURM_ARRAY_TASK_ID]}
KIND=$1; DATASET=$2
if [[ "$KIND" == "transfer" ]]; then
  SOURCE=$3; export TRANSFER_SOURCE="$SOURCE"
  case "$DATASET" in
    dvsgesture) python3 EVS_DVSGesture_TRANSFER.py; EXIT_CODE=$?; exit $EXIT_CODE ;;
    cifar10dvs) python3 EVS_CIFAR10DVS_TRANSFER.py; EXIT_CODE=$?; exit $EXIT_CODE ;;
  esac
else
  ABLATION=$3; VICTIM=$4; export ABLATION ABLATION_VICTIMS="$VICTIM"
  case "$DATASET" in
    dvsgesture) python3 EVS_DVSGesture_ABLATIONS.py; EXIT_CODE=$?; exit $EXIT_CODE ;;
    cifar10dvs) python3 EVS_CIFAR10DVS_ABLATIONS.py; EXIT_CODE=$?; exit $EXIT_CODE ;;
  esac
fi

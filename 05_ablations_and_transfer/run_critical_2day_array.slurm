#!/bin/bash
#SBATCH --job-name=ea-ablate-critical
# The partition name below is a placeholder; set it for your cluster.
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-37%3
#SBATCH -o logs/critical_%A_%a.out
#SBATCH -e logs/critical_%A_%a.err
set -euo pipefail
mkdir -p logs
cd "$SLURM_SUBMIT_DIR"
# shellcheck disable=SC1091
source "$SLURM_SUBMIT_DIR/cluster_env.sh"
export ABLATION_SEED=${ABLATION_SEED:-0}
export ABLATION_BUDGET=${ABLATION_BUDGET:-0.10}

# Priority order is deliberate: mechanism -> metric story -> real-data defense.
JOBS=(
  "dvsgesture phase conv_snn"
  "dvsgesture phase sew_resnet18"
  "dvsgesture phase event_transformer_v2"
  "dvsgesture phase temporal_gru"
  "dvsgesture intersection conv_snn"
  "dvsgesture intersection sew_resnet18"
  "dvsgesture intersection event_transformer_v2"
  "dvsgesture intersection temporal_gru"
  "dvsgesture window conv_snn"
  "dvsgesture window sew_resnet18"
  "dvsgesture window event_transformer_v2"
  "dvsgesture window temporal_gru"
  "dvsgesture observer_family conv_snn"
  "dvsgesture observer_family sew_resnet18"
  "dvsgesture observer_family event_transformer_v2"
  "dvsgesture observer_family temporal_gru"
  "dvsgesture randomized conv_snn"
  "dvsgesture randomized sew_resnet18"
  "dvsgesture randomized event_transformer_v2"
  "dvsgesture randomized temporal_gru"
  "cifar10dvs phase event_transformer_v2"
  "cifar10dvs phase temporal_gru"
  "cifar10dvs window event_transformer_v2"
  "cifar10dvs window temporal_gru"
  "cifar10dvs observer_family event_transformer_v2"
  "cifar10dvs observer_family temporal_gru"
  "dailydvs200 phase swin_t"
  "dailydvs200 phase timesformer"
  "dailydvs200 phase mvfnet"
  "dailydvs200 phase actionnet"
  "dailydvs200 observer_family swin_t"
  "dailydvs200 observer_family timesformer"
  "dailydvs200 observer_family mvfnet"
  "dailydvs200 observer_family actionnet"
  "dailydvs200 randomized swin_t"
  "dailydvs200 randomized timesformer"
  "dailydvs200 randomized mvfnet"
  "dailydvs200 randomized actionnet"
)
read -r DATASET ABLATION VICTIM <<< "${JOBS[$SLURM_ARRAY_TASK_ID]}"
export ABLATION ABLATION_VICTIMS="$VICTIM"
case "$DATASET" in
  dvsgesture) python3 EVS_DVSGesture_ABLATIONS.py; EXIT_CODE=$?; exit $EXIT_CODE ;;
  cifar10dvs) python3 EVS_CIFAR10DVS_ABLATIONS.py; EXIT_CODE=$?; exit $EXIT_CODE ;;
  dailydvs200) python3 EVS_DailyDVS200_ABLATIONS.py; EXIT_CODE=$?; exit $EXIT_CODE ;;
  *) echo "unknown dataset $DATASET" >&2; exit 2 ;;
esac

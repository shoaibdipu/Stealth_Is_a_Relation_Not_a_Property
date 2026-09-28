#!/bin/bash
#SBATCH --job-name=nm5_smoke
#SBATCH --output=logs/nm5-smoke-%j.out
#SBATCH --error=logs/nm5-smoke-%j.err
# The partition name below is a placeholder; set it for your cluster.
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=10G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source "${VENV_ACTIVATE:-/path/to/venv/bin/activate}"
export PYTHONUNBUFFERED=1
export NMNIST_DATA=${NMNIST_DATA:-/path/to/workspace/EVS_NMNIST/run/data}
export NMNIST_RUN=${NMNIST_RUN:-/path/to/workspace/EVS_NMNIST_ALIGNED5/run}
echo "job $SLURM_JOB_ID started $(date)"
nvidia-smi -L
ALLOW_SCOPED_RUN=1 RUN_SEEDS=0 \
TRAIN_MODELS=coarse_frameformer,conv_snn,sew_resnet18,event_transformer_v2,temporal_gru \
ATTACK_MODELS=conv_snn SMOKE_ONLY=1 \
python3 EVS_NMNIST_ALIGNED5_FINAL.py
echo "finished $(date)"

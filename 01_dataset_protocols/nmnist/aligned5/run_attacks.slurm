#!/bin/bash
#SBATCH --job-name=nm5_attack
#SBATCH --output=logs/nm5-attack-%j.out
#SBATCH --error=logs/nm5-attack-%j.err
# The partition name below is a placeholder; set it for your cluster.
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=10G
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1

set -euo pipefail
mkdir -p logs
cd "$SLURM_SUBMIT_DIR"
source "${VENV_ACTIVATE:-/path/to/venv/bin/activate}"
export PYTHONUNBUFFERED=1
export NMNIST_DATA=${NMNIST_DATA:-/path/to/workspace/EVS_NMNIST/run/data}
export NMNIST_RUN=${NMNIST_RUN:-/path/to/workspace/EVS_NMNIST_ALIGNED5/run}
RUN_TRAIN=0 RUN_ATTACKS=1 python3 EVS_NMNIST_ALIGNED5_FINAL.py

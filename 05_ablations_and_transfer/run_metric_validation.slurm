#!/bin/bash
#SBATCH --job-name=ea-metric-posthoc
# The partition name below is a placeholder; set it for your cluster.
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH -o logs/metric_%j.out
#SBATCH -e logs/metric_%j.err
set -euo pipefail
mkdir -p logs
cd "$SLURM_SUBMIT_DIR"
# shellcheck disable=SC1091
source "$SLURM_SUBMIT_DIR/cluster_env.sh"
python3 posthoc_metric_validation.py
EXIT_CODE=$?
exit $EXIT_CODE

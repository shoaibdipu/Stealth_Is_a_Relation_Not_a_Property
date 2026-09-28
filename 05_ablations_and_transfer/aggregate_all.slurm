#!/bin/bash
#SBATCH --job-name=ea5x5agg
# The partition name below is a placeholder; set it for your cluster.
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%j_aggregate.out
set -euo pipefail
mkdir -p logs
cd "$SLURM_SUBMIT_DIR"
# shellcheck disable=SC1091
source "$SLURM_SUBMIT_DIR/cluster_env.sh"
python3 aggregate_all.py
EXIT_CODE=$?
exit $EXIT_CODE

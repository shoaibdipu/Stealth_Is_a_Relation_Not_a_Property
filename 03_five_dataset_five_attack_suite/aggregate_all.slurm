#!/bin/bash
# The partition name below is a placeholder; set it for your cluster.
#SBATCH --partition=gpu
#SBATCH --job-name=ea5x5agg
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%j_aggregate.out
set -euo pipefail
source "${VENV_ACTIVATE:-/path/to/venv/bin/activate}"
python aggregate_all.py

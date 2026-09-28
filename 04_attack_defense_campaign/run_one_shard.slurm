#!/bin/bash
#SBATCH --job-name=ea_ad
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
# module load <python-module>   # only if your cluster uses environment modules
source "${VENV_ACTIVATE:-/path/to/venv/bin/activate}"
cd "${CODE_ROOT:-${SLURM_SUBMIT_DIR:-$(dirname "$0")}}"
exec ./run_one_shard.sh

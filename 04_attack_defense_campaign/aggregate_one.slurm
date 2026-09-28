#!/bin/bash
#SBATCH --job-name=ea_agg
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
# module load <python-module>   # only if your cluster uses environment modules
source "${VENV_ACTIVATE:-/path/to/venv/bin/activate}"
cd "${CODE_ROOT:-${SLURM_SUBMIT_DIR:-$(dirname "$0")}}"
mkdir -p logs
if [[ -n "${VENV_ACTIVATE:-}" ]]; then
  if [[ ! -f "$VENV_ACTIVATE" ]]; then echo "VENV_ACTIVATE missing: $VENV_ACTIVATE" >&2; exit 2; fi
  # shellcheck disable=SC1090
  source "$VENV_ACTIVATE"
fi
: "${DATASET:?DATASET required}"
: "${RUN_ROOT:?RUN_ROOT required}"
export CAMPAIGN_ID="${CAMPAIGN_ID:-stealth_asr_defense_v1}"
python aggregate_campaign.py --dataset "$DATASET"
python validate_campaign.py --dataset "$DATASET" --run-root "$RUN_ROOT" --campaign-id "$CAMPAIGN_ID"

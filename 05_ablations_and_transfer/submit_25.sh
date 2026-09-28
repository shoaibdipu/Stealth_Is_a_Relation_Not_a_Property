#!/bin/bash
set -euo pipefail
: "${EA_VENV:=/path/to/venv}"
if [ ! -f "${EA_VENV}/bin/activate" ]; then
  echo "ERROR: venv not found at ${EA_VENV}; export EA_VENV=<path> first" >&2; exit 78
fi
export EA_VENV
# shellcheck disable=SC1091
source "${EA_VENV}/bin/activate"
source ./setup_official_repos.sh
mkdir -p logs
jid=$(sbatch --parsable run_25_array.slurm)
echo "attack array: $jid"
sbatch --dependency=afterok:$jid aggregate_all.slurm

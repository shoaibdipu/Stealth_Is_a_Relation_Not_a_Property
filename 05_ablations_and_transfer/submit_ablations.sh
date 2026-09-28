#!/bin/bash
set -euo pipefail
: "${EA_VENV:=/path/to/venv}"
if [ ! -f "${EA_VENV}/bin/activate" ]; then
  echo "ERROR: venv not found at ${EA_VENV}; export EA_VENV=<path> first" >&2; exit 78
fi
export EA_VENV
# shellcheck disable=SC1091
source "${EA_VENV}/bin/activate"
mkdir -p logs
sbatch run_ablation_array.slurm

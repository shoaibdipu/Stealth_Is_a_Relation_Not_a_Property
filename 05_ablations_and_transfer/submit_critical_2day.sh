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
python3 preflight_ablations.py
# Metric analysis is cheap; run immediately against whatever 5x5 results have landed.
METRIC_JOB=$(sbatch --parsable run_metric_validation.slurm || true)
GPU_JOB=$(sbatch --parsable run_critical_2day_array.slurm)
echo "critical GPU array: $GPU_JOB"
echo "metric CPU job: ${METRIC_JOB:-not submitted}"
echo "Submit optional extras with ./submit_optional_2day.sh once the critical jobs are healthy."

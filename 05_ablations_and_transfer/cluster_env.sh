#!/bin/bash
# Shared cluster environment for every Slurm job in this package.
# Sourced from $SLURM_SUBMIT_DIR (Slurm copies batch scripts to a spool dir,
# so $(dirname "$0") does NOT work inside a job).
#
# Override the venv path per site:  export EA_VENV=/path/to/venv
: "${EA_VENV:=/path/to/venv}"

if [ ! -f "${EA_VENV}/bin/activate" ]; then
  echo "ERROR: venv not found at ${EA_VENV}" >&2
  echo "If the cluster has no module system, the venv is the whole environment." >&2
  echo "Set EA_VENV to your project venv before submitting, e.g.:" >&2
  echo "  export EA_VENV=/path/to/venvs/<name>" >&2
  exit 78
fi
# shellcheck disable=SC1091
source "${EA_VENV}/bin/activate"

python3 - <<'PY' || exit 78
import sys
try:
    import torch, numpy, pandas  # noqa
except Exception as e:
    sys.exit(f"venv is missing a required package: {e}")
print(f"env ok: python {sys.version.split()[0]}, torch {torch.__version__}, "
      f"cuda_available={torch.cuda.is_available()}")
PY

# Official baseline clones (needed by the 5x5 attack arms; harmless for ablations).
export SPIKE_RETIMING_REPO="${SPIKE_RETIMING_REPO:-$SLURM_SUBMIT_DIR/Spike-Retiming-Attacks}"
export PDSG_SDA_DIR="${PDSG_SDA_DIR:-$SLURM_SUBMIT_DIR/PDSG-SDA}"
export YAO_DIR="${YAO_DIR:-$SLURM_SUBMIT_DIR/GumbelSoftmaxAttack}"

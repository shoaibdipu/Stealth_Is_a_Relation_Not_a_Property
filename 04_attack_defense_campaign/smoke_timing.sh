#!/bin/bash
# Run this on each target cluster before production. It is deliberately serial so timings are readable.
set -euo pipefail
cd "$(dirname "$0")"
DS=${1:?dataset}; ROOT=${2:?run_root}; CLEAN=${3:?clean_pool_manifest}
export DATASET="$DS" RUN_ROOT="$ROOT" DEFENSE_CLEAN_POOL_MANIFEST="$CLEAN" DIRECT_N=4 RUN_SEEDS=0 VICTIMS=conv_snn
export SHARD_INDEX=0 SHARD_COUNT=1 CAMPAIGN_ID="smoke_${DS}_$(date +%Y%m%d_%H%M%S)" PARTIAL_ONLY=1 DEFENSE_ANALYSIS=1 REQUIRE_DISJOINT_CLEAN_POOL=1 SAVE_PERTURBATIONS=1
if [[ -n "${VENV_ACTIVATE:-}" ]]; then
  if [[ ! -f "$VENV_ACTIVATE" ]]; then echo "VENV_ACTIVATE missing: $VENV_ACTIVATE" >&2; exit 2; fi
  # shellcheck disable=SC1090
  source "$VENV_ACTIVATE"
fi
for METHOD in yu_pil_l0 pdsg_sda yao_gumbel free_retiming null_space free_da_sweep; do
  export METHOD
  echo "==== $DS / $METHOD ===="
  /usr/bin/time -p python run_combined.py --dataset "$DS"
done

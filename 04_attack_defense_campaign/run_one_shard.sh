#!/bin/bash
set -euo pipefail
cd "${CODE_ROOT:-$(dirname "$0")}" || exit 2
export SPIKE_RETIMING_REPO="${SPIKE_RETIMING_REPO:-$PWD/Spike-Retiming-Attacks}"
export PDSG_SDA_DIR="${PDSG_SDA_DIR:-$PWD/PDSG-SDA}"
export YAO_DIR="${YAO_DIR:-$PWD/GumbelSoftmaxAttack}"
mkdir -p logs runtime_logs
if [[ -n "${VENV_ACTIVATE:-}" ]]; then
  if [[ ! -f "$VENV_ACTIVATE" ]]; then echo "VENV_ACTIVATE missing: $VENV_ACTIVATE" >&2; exit 2; fi
  # shellcheck disable=SC1090
  source "$VENV_ACTIVATE"
fi
: "${DATASET:?DATASET required}"
: "${METHOD:?METHOD required}"
: "${RUN_ROOT:?RUN_ROOT required}"
: "${DEFENSE_CLEAN_POOL_MANIFEST:?DEFENSE_CLEAN_POOL_MANIFEST required}"
export PARTIAL_ONLY=1 DEFENSE_ANALYSIS=1 REQUIRE_DISJOINT_CLEAN_POOL=1 SAVE_PERTURBATIONS=1
export RUN_SEEDS="${RUN_SEEDS:-0}" VICTIMS="${VICTIMS:-conv_snn}" DIRECT_N="${DIRECT_N:-0}"
export SHARD_INDEX="${SHARD_INDEX:-0}" SHARD_COUNT="${SHARD_COUNT:-1}"
export CAMPAIGN_ID="${CAMPAIGN_ID:-stealth_asr_defense_v1}"
export FLUSH_EVERY="${FLUSH_EVERY:-1}"
start=$(date +%s)
echo "START $(date -Is) dataset=$DATASET method=$METHOD seeds=$RUN_SEEDS shard=$SHARD_INDEX/$SHARD_COUNT N=$DIRECT_N campaign=$CAMPAIGN_ID gpu=${CUDA_VISIBLE_DEVICES:-scheduler}"
set +e
python run_combined.py --dataset "$DATASET"
rc=$?
set -e
end=$(date +%s)
tag="${DATASET}_${METHOD}_s${RUN_SEEDS}_sh${SHARD_INDEX}of${SHARD_COUNT}_${SLURM_JOB_ID:-local_$$}"
echo "END $(date -Is) rc=$rc elapsed_s=$((end-start)) dataset=$DATASET method=$METHOD seeds=$RUN_SEEDS shard=$SHARD_INDEX/$SHARD_COUNT" | tee "runtime_logs/${tag}.txt"
exit "$rc"

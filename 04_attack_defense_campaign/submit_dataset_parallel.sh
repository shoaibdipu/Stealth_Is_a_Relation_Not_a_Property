#!/bin/bash
# Usage: ./submit_dataset_parallel.sh DATASET RUN_ROOT CLEAN_POOL_MANIFEST [DIRECT_N]
set -euo pipefail
cd "$(dirname "$0")"
DS=${1:?dataset}; ROOT=${2:?run_root}; CLEAN=${3:?clean_pool_manifest}; N=${4:-}
case "$DS" in
  dvsgesture)    N=${N:-264}; DEF_YU=4; DEF_SDA=6; DEF_YAO=8; DEF_CHEAP=2;;
  dailydvs200)   N=${N:-250}; DEF_YU=3; DEF_SDA=6; DEF_YAO=8; DEF_CHEAP=2;;
  cifar10dvs)    N=${N:-150}; DEF_YU=2; DEF_SDA=5; DEF_YAO=6; DEF_CHEAP=2;;
  *) echo "dataset must be dvsgesture|dailydvs200|cifar10dvs" >&2; exit 2;;
esac
CAMPAIGN_ID=${CAMPAIGN_ID:-stealth_asr_defense_$(date +%Y%m%d_%H%M%S)}
python preflight.py --dataset "$DS" --run-root "$ROOT" --clean-pool "$CLEAN" --direct-n "$N"
echo "CAMPAIGN_ID=$CAMPAIGN_ID"
YU_SHARDS=${YU_SHARDS:-$DEF_YU}; SDA_SHARDS=${SDA_SHARDS:-$DEF_SDA}; YAO_SHARDS=${YAO_SHARDS:-$DEF_YAO}; CHEAP_SHARDS=${CHEAP_SHARDS:-$DEF_CHEAP}; SWEEP_SHARDS=${SWEEP_SHARDS:-$DEF_CHEAP}
SBATCH_EXTRA_ARGS=${SBATCH_EXTRA_ARGS:-}
SBATCH_GPU_ARGS=${SBATCH_GPU_ARGS:---gres=gpu:1}
JIDS=()
submit_method () {
  local method=$1 seeds=$2 nshard=$3
  for ((i=0;i<nshard;i++)); do
    jid=$(sbatch --parsable $SBATCH_GPU_ARGS $SBATCH_EXTRA_ARGS --export=ALL,DATASET="$DS",METHOD="$method",RUN_ROOT="$ROOT",DEFENSE_CLEAN_POOL_MANIFEST="$CLEAN",DIRECT_N="$N",RUN_SEEDS="$seeds",SHARD_INDEX="$i",SHARD_COUNT="$nshard",CAMPAIGN_ID="$CAMPAIGN_ID" run_one_shard.slurm)
    echo "$DS $method seeds=$seeds shard=$i/$nshard -> $jid"
    JIDS+=("$jid")
  done
}
# Symmetric five-attack seed-0 table.
submit_method yu_pil_l0 0 "$YU_SHARDS"
submit_method pdsg_sda 0 "$SDA_SHARDS"
submit_method yao_gumbel 0 "$YAO_SHARDS"
submit_method free_retiming 0 "$CHEAP_SHARDS"
submit_method null_space 0 "$CHEAP_SHARDS"
# Cheap variance estimates: submit seeds 1 and 2 independently to maximize parallelism.
submit_method free_retiming 1 "$CHEAP_SHARDS"
submit_method free_retiming 2 "$CHEAP_SHARDS"
submit_method null_space 1 "$CHEAP_SHARDS"
submit_method null_space 2 "$CHEAP_SHARDS"
# Controlled within-family D_A mechanism sweep, seed 0.
submit_method free_da_sweep 0 "$SWEEP_SHARDS"

dep=$(IFS=:; echo "${JIDS[*]}")
agg=$(sbatch --parsable $SBATCH_EXTRA_ARGS --dependency=afterok:$dep --export=ALL,DATASET="$DS",RUN_ROOT="$ROOT",CAMPAIGN_ID="$CAMPAIGN_ID" aggregate_one.slurm)
echo "aggregate -> $agg"
echo "submitted ${#JIDS[@]} GPU jobs + aggregation for $DS"

#!/bin/bash
# Adds seeds 1 and 2 for yu/sda/yao into an existing campaign, then re-aggregates.
set -euo pipefail
cd "$(dirname "$0")"
DS=${1:?dataset}; ROOT=${2:?run_root}; CLEAN=${3:?clean_pool}; N=${4:?direct_n}; CAMPAIGN_ID=${5:?campaign_id}
case "$DS" in
  dvsgesture)  YU=4; SDA=6; YAO=8;;
  dailydvs200) YU=3; SDA=6; YAO=8;;
  *) echo "unsupported: $DS" >&2; exit 2;;
esac
SBATCH_EXTRA_ARGS=${SBATCH_EXTRA_ARGS:-}
SBATCH_GPU_ARGS=${SBATCH_GPU_ARGS:---gres=gpu:1}
JIDS=()
sub () {
  local method=$1 seed=$2 nshard=$3
  for ((i=0;i<nshard;i++)); do
    jid=$(sbatch --parsable $SBATCH_GPU_ARGS $SBATCH_EXTRA_ARGS \
      --export=ALL,DATASET="$DS",METHOD="$method",RUN_ROOT="$ROOT",DEFENSE_CLEAN_POOL_MANIFEST="$CLEAN",DIRECT_N="$N",RUN_SEEDS="$seed",SHARD_INDEX="$i",SHARD_COUNT="$nshard",CAMPAIGN_ID="$CAMPAIGN_ID" run_one_shard.slurm)
    echo "$DS $method seed=$seed shard=$i/$nshard -> $jid"
    JIDS+=("$jid")
  done
}
for s in 1 2; do
  sub yu_pil_l0  "$s" "$YU"
  sub pdsg_sda   "$s" "$SDA"
  sub yao_gumbel "$s" "$YAO"
done
dep=$(IFS=:; echo "${JIDS[*]}")
agg=$(sbatch --parsable $SBATCH_EXTRA_ARGS --dependency=afterany:$dep \
  --export=ALL,DATASET="$DS",RUN_ROOT="$ROOT",CAMPAIGN_ID="$CAMPAIGN_ID" aggregate_one.slurm)
echo "re-aggregate -> $agg"
echo "submitted ${#JIDS[@]} jobs for $DS"

#!/bin/bash
#SBATCH --job-name=ea5x5
# The partition name below is a placeholder; set it for your cluster.
#SBATCH --partition=gpu
#SBATCH --array=0-24%3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%A_%a.out
set -euo pipefail
mkdir -p logs
cd "$SLURM_SUBMIT_DIR"
# shellcheck disable=SC1091
source "$SLURM_SUBMIT_DIR/cluster_env.sh"
DATASETS=(nmnist dvsgesture cifar10dvs ncaltech101 dailydvs200)
METHODS=(yu_pil_l0 pdsg_sda yao_gumbel free_retiming null_space)
D=$((SLURM_ARRAY_TASK_ID/5)); M=$((SLURM_ARRAY_TASK_ID%5)); DS=${DATASETS[$D]}; METHOD=${METHODS[$M]}
case "$DS" in
 nmnist) SCRIPT=EVS_NMNIST_5ATTACK.py;;
 dvsgesture) SCRIPT=EVS_DVSGesture_5ATTACK.py;;
 cifar10dvs) SCRIPT=EVS_CIFAR10DVS_5ATTACK.py;;
 ncaltech101) SCRIPT=EVS_NCaltech101_5ATTACK.py;;
 dailydvs200) SCRIPT=EVS_DailyDVS200_5ATTACK.py;;
esac
export METHOD PARTIAL_ONLY=1
python3 "$SCRIPT"
EXIT_CODE=$?
exit $EXIT_CODE

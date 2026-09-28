#!/bin/bash
#SBATCH --job-name=ea_smoke
# #SBATCH --account=<allocation>   (uncomment and set if your scheduler requires an allocation)
# The partition name below is a placeholder; set it for your cluster.
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time="01:00:00"
#SBATCH --output=logs/ea-smoke-%j.out
#SBATCH --error=logs/ea-smoke-%j.err

cd "$SLURM_SUBMIT_DIR"
# module load <python-module>   # only if your cluster uses environment modules
source "${VENV_ACTIVATE:-/path/to/venv/bin/activate}"
export PYTHONUNBUFFERED=1
export DVS_GESTURE_RUN=/path/to/workspace/EVS_DVSGesture/run
export SPIKE_RETIMING_REPO=/path/to/workspace/Spike-Retiming-Attacks
echo "job $SLURM_JOB_ID started $(date)"
nvidia-smi -L
RUN_SEEDS=0 VICTIMS=conv_snn PROTECTED_CONSUMER=coarse_frameformer \
DIRECT_BUDGETS=0.10 DIRECT_N=2 EXPECT_N=0 PRIOR_STEPS=2 PRIOR_RECALIBRATE=0 \
python3 ea_direct_compare.py
EXIT_CODE=$?
echo "finished $(date), exit $EXIT_CODE"
exit $EXIT_CODE

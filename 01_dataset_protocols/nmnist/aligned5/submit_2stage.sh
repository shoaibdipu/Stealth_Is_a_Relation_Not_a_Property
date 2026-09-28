#!/bin/bash
set -euo pipefail
mkdir -p logs
J1=$(sbatch --parsable run_train.slurm)
echo "training job: $J1"
J2=$(sbatch --parsable --dependency=afterok:$J1 run_attacks.slurm)
echo "attack job: $J2"

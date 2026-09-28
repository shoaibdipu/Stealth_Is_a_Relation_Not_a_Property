#!/bin/bash
# smoke -> train x2 (resumable) -> attack x2 (resumable)
# Training gets a second stage so a 48 h overrun cannot strand the attacks.
# The smoke gates everything via afterok; later links use afterany because
# each stage resumes from per-epoch / per-unit caches.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs
for f in smoke.slurm run_train.slurm run_attacks.slurm; do bash -n "$f"; done
S=$(sbatch --parsable smoke.slurm)
T1=$(sbatch --parsable --dependency=afterok:$S  run_train.slurm)
T2=$(sbatch --parsable --dependency=afterany:$T1 run_train.slurm)
A1=$(sbatch --parsable --dependency=afterany:$T2 run_attacks.slurm)
A2=$(sbatch --parsable --dependency=afterany:$A1 run_attacks.slurm)
echo "smoke $S | train $T1 $T2 | attack $A1 $A2"

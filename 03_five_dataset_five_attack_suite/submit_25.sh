#!/bin/bash
set -euo pipefail
jid=$(sbatch --parsable run_25_array.slurm)
echo "attack array: $jid"
sbatch --dependency=afterok:$jid aggregate_all.slurm

#!/bin/bash
set -euo pipefail
source "${VENV_ACTIVATE:-/path/to/venv/bin/activate}"
export NMNIST_DATA=${NMNIST_DATA:-/path/to/workspace/EVS_NMNIST/run/data}
export NMNIST_RUN=${NMNIST_RUN:-/path/to/workspace/EVS_NMNIST_ALIGNED5/run}
ALLOW_SCOPED_RUN=1 RUN_SEEDS=0 TRAIN_MODELS=coarse_frameformer,conv_snn,sew_resnet18,event_transformer_v2,temporal_gru ATTACK_MODELS=conv_snn SMOKE_ONLY=1 python3 EVS_NMNIST_ALIGNED5_FINAL.py

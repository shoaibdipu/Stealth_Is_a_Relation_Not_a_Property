#!/usr/bin/env bash
set -euo pipefail

export DVS_GESTURE_RUN="${DVS_GESTURE_RUN:-/path/to/workspace/EVS_DVSGesture/run}"
export SPIKE_RETIMING_REPO="${SPIKE_RETIMING_REPO:-/path/to/workspace/Spike-Retiming-Attacks}"

# HARD GATE: run the documented DIRECT_N=2 / PRIOR_STEPS=2 smoke first.
# v5 uses a protocol-fingerprinted partial cache, so smoke rows cannot contaminate this full run.
# The official all-time L0 method is O(T^2) in its shift logits; their released
# DVS commands use T=10 while this frozen EA representation uses T=160.

RUN_SEEDS="0,1,2" \
VICTIMS="conv_snn" \
PROTECTED_CONSUMER="coarse_frameformer" \
VERIFY_CLEAN_ACCURACY="1" \
DIRECT_BUDGETS="0.10" \
DIRECT_N="0" \
EXPECT_N="264" \
PRIOR_STEPS="40" \
PRIOR_RECALIBRATE="1" \
python3 ea_direct_compare.py

# Optional appendix after the primary run is stable:
# RUN_SEEDS="0,1,2" VICTIMS="conv_snn,sew_resnet18" \
# PROTECTED_CONSUMER="coarse_frameformer" DIRECT_BUDGETS="0.02,0.05,0.10" \
# python3 ea_direct_compare.py

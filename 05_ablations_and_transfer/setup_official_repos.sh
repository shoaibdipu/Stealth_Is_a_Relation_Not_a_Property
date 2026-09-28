#!/bin/bash
set -euo pipefail
[ -d Spike-Retiming-Attacks/.git ] || git clone https://github.com/yuyi-sd/Spike-Retiming-Attacks.git
[ -d PDSG-SDA/.git ] || git clone https://github.com/ryime/PDSG-SDA.git
[ -d GumbelSoftmaxAttack/.git ] || git clone https://github.com/JY-ura/GumbelSoftmaxAttack.git
export SPIKE_RETIMING_REPO=${SPIKE_RETIMING_REPO:-$PWD/Spike-Retiming-Attacks}
export PDSG_SDA_DIR=${PDSG_SDA_DIR:-$PWD/PDSG-SDA}
export YAO_DIR=${YAO_DIR:-$PWD/GumbelSoftmaxAttack}
echo "SPIKE_RETIMING_REPO=$SPIKE_RETIMING_REPO"
echo "PDSG_SDA_DIR=$PDSG_SDA_DIR"
echo "YAO_DIR=$YAO_DIR"

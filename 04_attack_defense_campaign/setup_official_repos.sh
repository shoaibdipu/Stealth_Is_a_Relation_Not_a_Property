#!/bin/bash
set -euo pipefail
[ -d Spike-Retiming-Attacks/.git ] || git clone https://github.com/yuyi-sd/Spike-Retiming-Attacks.git
# FIX 11: pin the Yu repository to the commit the landed comparison used, so a
# newer clone with changed constructor defaults cannot silently move results.
YU_COMMIT=${YU_COMMIT:-19f42e63d31cbdc4ef2d5ef7b6e40716f92531c3}
git -C Spike-Retiming-Attacks -c advice.detachedHead=false checkout -q "$YU_COMMIT" 2>/dev/null \
  || echo "WARNING: could not check out $YU_COMMIT; using $(git -C Spike-Retiming-Attacks rev-parse HEAD)"
YU_ATTACK_SHA=$(sha256sum Spike-Retiming-Attacks/utils/attack.py | cut -d' ' -f1)
echo "yu commit=$(git -C Spike-Retiming-Attacks rev-parse HEAD)"
echo "yu utils/attack.py sha256=$YU_ATTACK_SHA"
[ "$YU_ATTACK_SHA" = "e2f76fc6b4796d9cece0dbf498ced5f62c4f9581c17476da34ab41a1087e11ea" ] \
  && echo "  matches the landed comparison" || echo "  WARNING: differs from the landed comparison" 
[ -d PDSG-SDA/.git ] || git clone https://github.com/ryime/PDSG-SDA.git
[ -d GumbelSoftmaxAttack/.git ] || git clone https://github.com/JY-ura/GumbelSoftmaxAttack.git
export SPIKE_RETIMING_REPO=${SPIKE_RETIMING_REPO:-$PWD/Spike-Retiming-Attacks}
export PDSG_SDA_DIR=${PDSG_SDA_DIR:-$PWD/PDSG-SDA}
export YAO_DIR=${YAO_DIR:-$PWD/GumbelSoftmaxAttack}
echo "SPIKE_RETIMING_REPO=$SPIKE_RETIMING_REPO"
echo "PDSG_SDA_DIR=$PDSG_SDA_DIR"
echo "YAO_DIR=$YAO_DIR"

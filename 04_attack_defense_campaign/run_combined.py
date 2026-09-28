#!/usr/bin/env python3
import argparse
from common.runner import run
p=argparse.ArgumentParser()
p.add_argument('--dataset',required=True,choices=['dvsgesture','dailydvs200','cifar10dvs'])
a=p.parse_args()
run(a.dataset)

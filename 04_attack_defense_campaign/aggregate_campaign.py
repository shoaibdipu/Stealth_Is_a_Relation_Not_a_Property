#!/usr/bin/env python3
import argparse
from common.runner import aggregate
p=argparse.ArgumentParser()
p.add_argument('--dataset',required=True,choices=['dvsgesture','dailydvs200','cifar10dvs'])
a=p.parse_args();aggregate(a.dataset)

#!/usr/bin/env python3
from pathlib import Path
import argparse
import pandas as pd
from ea_direct_compare import aggregate_outputs

p = argparse.ArgumentParser()
p.add_argument("--results", type=Path, required=True, help="direct_prior_retiming_comparison directory")
a = p.parse_args()
rows = pd.read_csv(a.results / "direct_retiming_rows.csv")
aggregate_outputs(rows, a.results)

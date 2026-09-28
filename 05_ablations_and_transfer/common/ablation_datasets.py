"""Ablation-only dataset specs.

The frozen 5x5 campaign keeps common.datasets.SUITE untouched.  The only change
here is DailyDVS-200, whose mechanism ablations use the newer completed SOTA4
extension (Swin-T/TimeSformer/MVFNet/ACTION-Net protocol-scale victims).
"""
from .datasets import DatasetSpec, SUITE as MAIN_SUITE, PROTOCOLS

ABLATION_SUITE = dict(MAIN_SUITE)
ABLATION_SUITE["dailydvs200"] = DatasetSpec(
    "dailydvs200", "DailyDVS-200",
    [PROTOCOLS / "dailydvs200" / "EVS_DAILYDVS200_SOTA4_REUSE_FINAL.py"],
    {"NUM_CLASSES": 200, "EVAL_BATCH": 48},
    (80, 8, 2, 64, 64),
    {"swin_t": "VideoSwinDVS", "timesformer": "TimeSformerDVS",
     "mvfnet": "MVFNetDVS", "actionnet": "ACTIONNetDVS"},
    "CoarseFrameFormer", "coarse_frameformer",
    [.01, .02, .05, .10, .20], 50.0,
    "/path/to/workspace/EVS_DailyDVS200/run", 200, clean_tol_pp=.5,
)

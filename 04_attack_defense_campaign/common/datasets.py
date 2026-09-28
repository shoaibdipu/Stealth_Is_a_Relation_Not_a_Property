from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOLS = REPO_ROOT / "01_dataset_protocols"
DIRECT = REPO_ROOT / "02_dvsgesture_direct_comparison" / "prior_retiming_comparison"

@dataclass
class DatasetSpec:
    key: str
    name: str
    frozen_scripts: list
    overrides: dict
    grid: tuple
    victim_classes: dict
    consumer_class: str
    consumer_role: str
    budgets: list
    fallback_bin_ms: float
    default_run_root: str
    n_classes: int
    clean_tol_pp: float = 0.5
    nmnist_special: bool = False

DVS4={"conv_snn":"ConvSNN","sew_resnet18":"SEWResNet18","event_transformer_v2":"EventTemporalTransformerV2","temporal_gru":"TemporalGRU_DVS"}
SUITE={
 "nmnist":DatasetSpec("nmnist","N-MNIST",[PROTOCOLS/"nmnist"/"aligned5"/"EVS_NMNIST_ALIGNED5_FINAL.py"],{"EVAL_BATCH":256,"NUM_CLASSES":10},(150,10,2,34,34),DVS4,"CoarseFrameFormer","coarse_frameformer",[.05,.10,.20,.30],2.0,"/path/to/workspace/EVS_NMNIST_ALIGNED5/run",10,nmnist_special=True),
 "dvsgesture":DatasetSpec("dvsgesture","DVS128 Gesture",[DIRECT/"ea_direct_compare.py",PROTOCOLS/"dvsgesture"/"EVS_DVSGesture_addons.py"],{"EVAL_BATCH":32,"NUM_CLASSES":11},(160,8,2,64,64),DVS4,"CoarseFrameFormer","coarse_frameformer",[.02,.05,.10,.20],40.0,"/path/to/workspace/EVS_DVSGesture/run",11),
 "cifar10dvs":DatasetSpec("cifar10dvs","CIFAR10-DVS",[PROTOCOLS/"cifar10dvs"/"EVS_CIFAR10DVS_FINAL.py"],{"NUM_CLASSES":10,"EVAL_BATCH":32},(80,8,2,64,64),DVS4,"CoarseFrameFormer","coarse_frameformer",[.01,.02,.05,.10,.20],16.0,"/path/to/workspace/EVS_CIFAR10DVS/run",10),
 "ncaltech101":DatasetSpec("ncaltech101","N-Caltech101",[PROTOCOLS/"ncaltech101"/"EVS_NCALTECH101_FINAL_v3_READY.py"],{"NUM_CLASSES":101,"EVAL_BATCH":32},(80,8,2,64,64),DVS4,"CoarseFrameFormer","coarse_frameformer",[.01,.02,.05,.10,.20],3.75,"/path/to/workspace/EVS_NCALTECH101/run",101),
 "dailydvs200":DatasetSpec("dailydvs200","DailyDVS-200",[PROTOCOLS/"dailydvs200"/"EVS_DAILYDVS200_HF_FULL_v2.py"],{"NUM_CLASSES":200,"EVAL_BATCH":48},(80,8,2,64,64),DVS4,"CoarseFrameFormer","coarse_frameformer",[.01,.02,.05,.10,.20],50.0,"/path/to/workspace/EVS_DailyDVS200/run",200,clean_tol_pp=.5),
}

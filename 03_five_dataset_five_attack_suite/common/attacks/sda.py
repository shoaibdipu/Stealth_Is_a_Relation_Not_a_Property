import os,sys,inspect,time
from pathlib import Path
import numpy as np,torch,torch.nn as nn

def load(repo):
    repo=Path(repo).resolve(); fp=repo/"utils"/"attack.py"
    if not fp.exists():raise FileNotFoundError(f"PDSG-SDA clone not found: {repo}")
    major,minor=[int(x) for x in torch.__version__.split('+')[0].split('.')[:2]]
    if (major,minor)>=(2,9) and os.environ.get("SDA_TORCH_OVERRIDE")!="1":raise RuntimeError("Official SDA indexing requires torch<2.9; set SDA_TORCH_OVERRIDE=1 only after re-verification")
    saved={k:v for k,v in sys.modules.items() if k.split('.')[0] in {'utils','models','configs'}}
    for k in list(saved):sys.modules.pop(k,None)
    sys.path.insert(0,str(repo))
    try:
        from utils.attack import SDA
    finally:
        sys.path.pop(0)
        for k in [k for k in sys.modules if k.split('.')[0] in {'utils','models','configs'}]:sys.modules.pop(k,None)
        sys.modules.update(saved)
    return SDA
class Wrapper(nn.Module):
    def __init__(self,victim,counts0):super().__init__();self.victim=victim;self.register_buffer('unit_counts',torch.clamp(counts0.float(),min=1.0))
    def forward(self,x_presence):
        counts=x_presence*self.unit_counts.unsqueeze(1);logits=self.victim(counts.permute(1,0,2,3,4));return logits.unsqueeze(0)
def _run_attack(atk,pres,y,device):
    # cuDNN refuses RNN backward on an eval-mode module, which the SDA inner loop
    # triggers for temporal_gru victims. Disabling cuDNN for this call selects the
    # native kernel; numerically identical, eval mode unchanged.
    with torch.backends.cudnn.flags(enabled=False):
        return atk(pres.unsqueeze(1),torch.tensor([y],device=device)).detach()[:,0]

def attack(cls,model,X0,y,device):
    counts=torch.from_numpy(X0.astype(np.float32)).to(device);pres=(counts>0).float();wrapped=Wrapper(model,counts).to(device).eval();atk=cls(wrapped,k_init=int(os.environ.get('SDA_K_INIT','10')),N=int(os.environ.get('SDA_N','500')),batch_limit=int(os.environ.get('SDA_BATCH_LIMIT','64'))).to(device);adv=_run_attack(atk,pres,y,device);flip=adv!=pres;Xa=(adv*torch.clamp(counts,min=1.0)).cpu().numpy().round().astype(np.int16);return Xa,{"impl":"official SDA algorithm; adapted victim contract; white-box gradients","units_added":int(flip[pres==0].sum()),"units_removed":int(counts[torch.logical_and(flip,pres==1)].sum()),"cells_changed":int(flip.sum())}

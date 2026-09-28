import os,sys
from pathlib import Path
import numpy as np,torch,torch.nn as nn

def load(repo):
    repo=Path(repo).resolve(); core=repo/"attacks"/"ours"/"probability_space"/"probability_attack.py"
    if not core.exists(): raise FileNotFoundError(f"GumbelSoftmaxAttack clone not found: {repo}")
    saved={k:v for k,v in sys.modules.items() if k.split('.')[0] in {'utils','attacks','models','configs'}}
    for k in list(saved): sys.modules.pop(k,None)
    sys.path.insert(0,str(repo))
    try:
        try: import torchvision  # noqa
        except Exception:
            import types; sys.modules['utils.metrics']=types.ModuleType('utils.metrics')
        from attacks.ours.probability_space.probability_attack import ProbabilityAttacker
        from attacks.ours.probability_space.event_generator.gumbel_torch import GumbelSoftmaxTorch
        from utils.init_alpha import init_alpha_from_events
        from utils.loss_function import CrossEntropyLoss,L1Loss
    finally:
        sys.path.pop(0)
        for k in [k for k in sys.modules if k.split('.')[0] in {'utils','attacks','models','configs'}]: sys.modules.pop(k,None)
        sys.modules.update(saved)
    return ProbabilityAttacker,GumbelSoftmaxTorch,init_alpha_from_events,CrossEntropyLoss,L1Loss

def unit_events(X):
    t,p,y,x=np.nonzero(X); r=X[t,p,y,x].astype(np.int64)
    return {"t":np.repeat(t,r).astype(np.int32),"x":np.repeat(x,r).astype(np.int32),"y":np.repeat(y,r).astype(np.int32),"p":np.repeat(p,r).astype(np.int32)}, np.repeat(p,r).astype(np.int32)

class Binner(nn.Module):
    def __init__(self,e,T,H,W):
        super().__init__(); self.T,self.H,self.W=T,H,W
        flat=torch.as_tensor(e['t']*(H*W)+e['y']*W+e['x'],dtype=torch.long); self.register_buffer('flat',flat)
    def forward(self,values,event_indices=None,use_soft=False):
        if values.dim()==3:
            S=values.shape[0]; neg,pos=values[...,0].float(),values[...,1].float()
        else:
            S=values.shape[0]; v=values.float(); neg=torch.clamp(1-v,0,1); pos=torch.clamp(v-1,0,1)
        f=neg.new_zeros(S,2,self.T*self.H*self.W); idx=self.flat.unsqueeze(0).expand(S,-1)
        f[:,0].scatter_add_(1,idx,neg); f[:,1].scatter_add_(1,idx,pos)
        return f.reshape(S,2,self.T,self.H,self.W).permute(2,0,1,3,4)

def attack(bits,model,X0,y,device,n_classes):
    ProbabilityAttacker,GumbelSoftmaxTorch,init,CE,L1=bits
    events,orig_pol=unit_events(X0); T,C,H,W=X0.shape; b=Binner(events,T,H,W).to(device)
    sample_num=int(os.environ.get('YAO_SAMPLE_NUM','1')); max_tau=float(os.environ.get('YAO_MAX_TAU','2.0')); min_tau=float(os.environ.get('YAO_MIN_TAU','0.1')); decay=max(1,int(os.environ.get('YAO_TAU_DECAY_STEP','100'))); steps=int(os.environ.get('YAO_STEPS','300')); lr=float(os.environ.get('YAO_LR','0.1')); kappa=float(os.environ.get('YAO_KAPPA','1.0'))
    alpha0,_=init({k:v.copy() for k,v in events.items()},device=device)
    attacker=ProbabilityAttacker.__new__(ProbabilityAttacker); nn.Module.__init__(attacker)
    attacker.sample_num=sample_num; attacker.lamda=kappa; attacker.tau=max_tau; attacker.use_soft_event=True
    attacker.alpha=nn.Parameter(alpha0.clone(),requires_grad=True); attacker.event_indices=None
    attacker.event_generator=GumbelSoftmaxTorch(tau=max_tau,sample_num=sample_num,use_soft_event=True); attacker.frame_processor=b; attacker=attacker.to(device)
    st0=alpha0.argmax(-1); ref=torch.stack([(st0==0).float(),(st0==2).float()],dim=-1).unsqueeze(0)
    opt=torch.optim.Adam([attacker.alpha],lr=lr); ce=CE(istargeted=False,target=torch.tensor([y],device=device),sample_num=sample_num,num_class=int(n_classes)); l1=L1(ref.to(device),sample_num=sample_num,reduction='mean'); model.eval(); best=None
    def victim_logits(frames): return model(frames.permute(1,0,2,3,4))
    for step in range(steps):
        attacker.event_generator.tau=max(min_tau,max_tau*(0.5**(step//decay)))
        hard_f,soft_f,hard_v,soft_v=attacker(); logits=victim_logits(soft_f); loss=ce(logits)+kappa*l1(soft_v); opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            if bool((victim_logits(hard_f).argmax(1)!=y).all()): best=hard_v.detach()[0].cpu().numpy(); break
    with torch.no_grad():
        if best is None:
            hard_f,_,hard_v,_=attacker(); best=hard_v.detach()[0].cpu().numpy()
    states=best.astype(np.int64)-1; kept=states!=0; new_pol=((states+1)//2).astype(np.int64); Xa=np.zeros_like(X0,dtype=np.int64); np.add.at(Xa,(events['t'][kept],new_pol[kept],events['y'][kept],events['x'][kept]),1); Xa=Xa.astype(X0.dtype); d=Xa.astype(np.int64)-X0.astype(np.int64)
    return Xa,{"impl":"official Yao components/objective; adapted common-tensor driver","units_added":int(np.clip(d,0,None).sum()),"units_removed":int(np.clip(-d,0,None).sum()),"cells_changed":int(np.count_nonzero(d)),"polarity_flipped":int((kept&(new_pol!=orig_pol)).sum())}

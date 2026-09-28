import numpy as np
import torch
import torch.nn.functional as F

def _has_rnn(model): return any(isinstance(m,torch.nn.RNNBase) for m in model.modules())
def input_gradient(model,X,y,device):
    x=torch.from_numpy(X[None].astype(np.float32)).to(device).requires_grad_(True); yt=torch.tensor([int(y)],device=device)
    if _has_rnn(model):
        with torch.backends.cudnn.flags(enabled=False): g=torch.autograd.grad(F.cross_entropy(model(x),yt),x)[0][0]
    else: g=torch.autograd.grad(F.cross_entropy(model(x),yt),x)[0][0]
    return g.detach().cpu().numpy()
def _select(Xg,Gg,avail,n,S,max_shift=None):
    gi,si=np.nonzero(avail>0)
    if len(gi)==0:return Xg,avail,np.empty(0,np.int16)
    a=np.arange(S); band=np.ones((S,S),bool) if max_shift is None else np.abs(a[None]-a[:,None])<=max_shift; np.fill_diagonal(band,False)
    gm=np.where(band[si],Gg[gi],-np.inf); dst=gm.argmax(1); gain=gm[np.arange(len(dst)),dst]-Gg[gi,si]
    k=np.isfinite(gain)&(gain>0); gi,si,dst,gain=gi[k],si[k],dst[k],gain[k]
    if len(gi)==0:return Xg,avail,np.empty(0,np.int16)
    o=np.argsort(-gain,kind="stable");gi,si,dst=gi[o],si[o],dst[o];caps=avail[gi,si].astype(np.int64); cs=np.cumsum(caps); take=caps.copy(); cut=int(np.searchsorted(cs,n))
    if cut<len(caps): take[cut]=max(0,n-(cs[cut-1] if cut else 0));take[cut+1:]=0
    Xg[gi,si]-=take;np.add.at(Xg,(gi,dst),take);avail[gi,si]-=take
    return Xg,avail,np.repeat(dst-si,take).astype(np.int16)
def null_attack(model,grid,X0,y,budgets,device):
    total=int(X0.sum());x=X0.astype(np.int16).copy();avail=grid.grouped(X0).astype(np.int32);moved=0;sh=[];out={}
    for b in sorted(map(float,budgets)):
        need=int(round(b*total))-moved
        if need>0:
            g=input_gradient(model,x,y,device);xg=grid.grouped(x).astype(np.int32);gg=grid.grouped(g).astype(np.float32);xg,avail,s=_select(xg,gg,avail,need,grid.S);x=grid.ungrouped(xg).astype(np.int16);moved+=len(s);sh+=s.tolist();assert grid.coarse_equal(X0,x) and int(x.sum())==total
        out[b]={"X":x.copy(),"shifts":np.asarray(sh,np.int16),"moved":moved}
    return out
def free_attack(model,grid,X0,y,budgets,device):
    total=int(X0.sum());x=X0.astype(np.int16).copy();avail=X0.reshape(grid.T,-1).T.astype(np.int32);moved=0;sh=[];out={}
    for b in sorted(map(float,budgets)):
        need=int(round(b*total))-moved
        if need>0:
            g=input_gradient(model,x,y,device);xg=x.reshape(grid.T,-1).T.astype(np.int32);gg=g.reshape(grid.T,-1).T.astype(np.float32);xg,avail,s=_select(xg,gg,avail,need,grid.T);x=xg.T.reshape(X0.shape).astype(np.int16);moved+=len(s);sh+=s.tolist();assert int(x.sum())==total and x.min()>=0
        out[b]={"X":x.copy(),"shifts":np.asarray(sh,np.int16),"moved":moved}
    return out

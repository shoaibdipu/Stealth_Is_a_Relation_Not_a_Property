import os,sys,inspect
import numpy as np, torch, torch.nn as nn
class BatchMajor(nn.Module):
    def __init__(self,m):super().__init__();self.m=m
    def forward(self,x):return self.m(x.permute(1,0,2,3,4))
def load(repo):
    repo=str(repo);sys.path.insert(0,repo)
    try:
        from utils.attack import PGDTimeShiftAfterEncoder_L0 as C
    finally: sys.path.pop(0)
    return C
def rebuild(X0,d):
    Xa=np.zeros_like(X0)
    for t,c,h,w in np.argwhere(X0!=0):
        tt=int(t+d[t,c,h,w]);
        if not 0<=tt<X0.shape[0]:raise AssertionError("Yu displacement outside range")
        Xa[tt,c,h,w]+=X0[t,c,h,w]
    return Xa
def attack(model,X0,y,budget,device,cls):
    steps=int(os.environ.get("PRIOR_STEPS","40"));recal=int(os.environ.get("PRIOR_RECALIBRATE","1"));tol=float(os.environ.get("PRIOR_BUDGET_TOL","0.005"));total=int(X0.sum());occ=int((X0>0).sum());b0=max(1,int(round(budget*total/(total/max(1,occ)))));xt=torch.from_numpy(X0.astype(np.float32))[:,None].to(device);yt=torch.tensor([y],device=device);wrapped=BatchMajor(model).to(device).eval();best=None
    for k in range(recal+1):
        atk=cls(device,wrapped,steps=steps,l0_moves_budget=b0).to(device)
        with torch.enable_grad(): xa,d=atk(xt,yt,return_disp=True)
        Xa=xa[:,0].detach().cpu().numpy().round().astype(np.int16);D=d[:,0].detach().cpu().numpy().astype(np.int64);assert np.array_equal(rebuild(X0,D),Xa);moved=int(X0[D!=0].sum());frac=moved/max(1,total);best=(Xa,D,moved,b0)
        if abs(frac-budget)<=tol or k==recal:break
        b0=max(1,int(round(b0*budget/max(frac,1e-9))))
    Xa,D,moved,b0=best;sh=np.repeat(D[X0!=0],X0[X0!=0]).astype(np.int16);return Xa,sh,{"impl":"official","moved_units":moved,"b0_points":b0}

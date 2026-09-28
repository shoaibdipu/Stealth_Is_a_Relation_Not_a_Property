import numpy as np, math
from collections import Counter,deque,defaultdict

def uniform_control(grid,X0,n,rng):
    G=grid.grouped(X0).astype(np.int32); gi,si=np.nonzero(G>0); counts=G[gi,si]; ug=np.repeat(gi,counts);us=np.repeat(si,counts);n=min(n,len(ug));sel=rng.choice(len(ug),n,replace=False);X=G.copy();sh=[]
    for j in sel:
        g,s=int(ug[j]),int(us[j]);ds=np.delete(np.arange(grid.S),s);d=int(rng.choice(ds));X[g,s]-=1;X[g,d]+=1;sh.append(d-s)
    return grid.ungrouped(X).astype(np.int16),np.asarray(sh,np.int16)
def _flow(src_counts,shift_counts,S,rng):
    shifts=sorted(shift_counts);nsh=len(shifts);N=nsh+S+2;src=N-2;snk=N-1;cap=np.zeros((N,N),np.int64)
    for i,d in enumerate(shifts):
        cap[src,i]=shift_counts[d]
        for b in range(S):
            if 0<=b+d<S and src_counts[b]>0: cap[i,nsh+b]=shift_counts[d]
    for b in range(S):cap[nsh+b,snk]=src_counts[b]
    flow=np.zeros_like(cap)
    while True:
        par=[-1]*N;par[src]=src;q=deque([src])
        while q and par[snk]<0:
            u=q.popleft();order=list(range(N));rng.shuffle(order)
            for v in order:
                if par[v]<0 and cap[u,v]-flow[u,v]>0:par[v]=u;q.append(v)
        if par[snk]<0:break
        v=snk;aug=10**18
        while v!=src:u=par[v];aug=min(aug,cap[u,v]-flow[u,v]);v=u
        v=snk
        while v!=src:u=par[v];flow[u,v]+=aug;flow[v,u]-=aug;v=u
    if int(flow[src,:nsh].sum())!=sum(shift_counts.values()):return None
    return {(b,d):int(flow[i,nsh+b]) for i,d in enumerate(shifts) for b in range(S) if flow[i,nsh+b]>0}
def exact_matched(grid,X0,target,rng):
    target=np.asarray(target,np.int16);target=target[target!=0]
    if len(target)==0:return X0.copy(),target,True
    G=grid.grouped(X0).astype(np.int32);gi,si=np.nonzero(G>0);cnt=G[gi,si];ug=np.repeat(gi,cnt);us=np.repeat(si,cnt);src_counts=np.bincount(us,minlength=grid.S);alloc=_flow(src_counts,Counter(map(int,target.tolist())),grid.S,rng)
    if alloc is None:return X0.copy(),np.empty(0,np.int16),False
    avail=np.ones(len(ug),bool);real=[]
    for (b,d),n in alloc.items():
        pool=np.flatnonzero((us==b)&avail);rng.shuffle(pool);ch=pool[:n];avail[ch]=False;gsel=ug[ch];ssel=us[ch].astype(np.int64);np.subtract.at(G,(gsel,ssel),1);np.add.at(G,(gsel,ssel+d),1);real.extend([d]*len(ch))
    X=grid.ungrouped(G).astype(np.int16);real=np.asarray(real,np.int16)
    ok=Counter(real.tolist())==Counter(target.tolist()) and grid.coarse_equal(X0,X)
    return X,real,ok

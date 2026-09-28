from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class Grid:
    T: int
    S: int
    C: int
    H: int
    W: int
    @property
    def K(self): return self.T // self.S
    def grouped(self, X):
        return X.reshape(self.K, self.S, self.C, self.H, self.W).transpose(0,2,3,4,1).reshape(-1,self.S)
    def ungrouped(self, G):
        return G.reshape(self.K,self.C,self.H,self.W,self.S).transpose(0,4,1,2,3).reshape(self.T,self.C,self.H,self.W)
    def coarse(self, X):
        return X.reshape(self.K,self.S,self.C,self.H,self.W).sum(axis=1)
    def coarse_equal(self,a,b): return np.array_equal(self.coarse(a), self.coarse(b))
    def metrics(self,a,b):
        da=self.coarse(a).astype(np.int64); db=self.coarse(b).astype(np.int64); d=np.abs(db-da)
        return {"frame_l1_difference":int(d.sum()),"D_inf":int(d.max(initial=0)),"frame_exact_equal":bool(np.array_equal(da,db))}

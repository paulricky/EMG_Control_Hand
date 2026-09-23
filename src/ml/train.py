from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from .features import emg_features


class RidgeEmgModel:
    def __init__(self,weights): self.weights=np.asarray(weights,float)
    def predict(self,windows):
        f=np.stack([emg_features(w) for w in windows]); return np.clip(np.c_[f,np.ones(len(f))]@self.weights,0,1)
    def save(self,path): np.savez(path,weights=self.weights,model_type="ridge_features_v1")
    @classmethod
    def load(cls,path): return cls(np.load(path)["weights"])


def train_ridge(windows,targets,l2=1e-3):
    x=np.stack([emg_features(w) for w in windows]); x=np.c_[x,np.ones(len(x))]; y=np.asarray(targets,float)
    return RidgeEmgModel(np.linalg.solve(x.T@x+l2*np.eye(x.shape[1]),x.T@y))

#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import numpy as np
from src.ml.evaluation import estimate_delay_s

p=argparse.ArgumentParser(); p.add_argument("log"); p.add_argument("--rate",type=float,default=20); a=p.parse_args(); d=np.load(a.log,allow_pickle=False)
sim=np.asarray(d["sim_compact01"]); real=np.asarray(d["real_compact01"]); err=real-sim
print("MAE",np.mean(np.abs(err),axis=0)); print("RMSE",np.sqrt(np.mean(err*err,axis=0)))
print("delay_s",[estimate_delay_s(sim[:,i],real[:,i],a.rate) for i in range(7)])
try:
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(7,1,sharex=True,figsize=(10,12))
 for i,ax in enumerate(axes): ax.plot(sim[:,i],label="sim"); ax.plot(real[:,i],label="real"); ax.legend()
 plt.tight_layout(); plt.show()
except ImportError: pass

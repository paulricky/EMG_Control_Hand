#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse, json
import numpy as np
from src.ml.dataset import split_by_session
from src.ml.temporal import train_temporal, evaluate_temporal, _torch

p=argparse.ArgumentParser(); p.add_argument("sessions",nargs="+"); p.add_argument("--output",default="models/emg_temporal.pt"); p.add_argument("--epochs",type=int,default=20)
p.add_argument("--lambda-vel",type=float,default=0); p.add_argument("--window",type=int,default=400); p.add_argument("--stride",type=int,default=100); a=p.parse_args()
train,val,test=split_by_session(a.sessions)
if not train or not val or not test: raise SystemExit("at least three recording sessions are required for leakage-free train/validation/test splits")
model,_=train_temporal(train,val,a.epochs,lambda_vel=a.lambda_vel,window=a.window,stride=a.stride); metrics=evaluate_temporal(model,test,a.window,a.stride)
path=Path(a.output); path.parent.mkdir(parents=True,exist_ok=True); _torch().save({"state_dict":model.state_dict(),"channels":4,"window":a.window,"metrics":{k:np.asarray(v).tolist() for k,v in metrics.items()}},path)
print(json.dumps({k:np.asarray(v).tolist() for k,v in metrics.items()},indent=2))

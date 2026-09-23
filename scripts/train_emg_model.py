#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse, json, time
import numpy as np
from src.ml.train import train_ridge
from src.ml.temporal import windows_from_session
from src.ml.dataset import split_by_session
from src.ml.evaluation import regression_metrics

p=argparse.ArgumentParser(); p.add_argument("sessions",nargs="+"); p.add_argument("--output",default="models/emg_baseline.npz"); p.add_argument("--window",type=int,default=400); p.add_argument("--stride",type=int,default=100); a=p.parse_args()
train,val,test=split_by_session(a.sessions)
if not train or not val or not test: raise SystemExit("at least three sessions are required for session-disjoint train/validation/test splits")
def load(paths):
 pairs=[windows_from_session(path,a.window,a.stride) for path in paths]; return np.concatenate([x for x,_ in pairs]),np.concatenate([y for _,y in pairs])
x,y=load(train); model=train_ridge(x,y); tx,ty=load(test); start=time.perf_counter(); pred=model.predict(tx); latency=(time.perf_counter()-start)/len(tx)
model.save(a.output); metrics=regression_metrics(ty,pred); metrics["inference_latency_s"]=latency; metrics["overall_mae"]=float(np.mean(metrics["mae"])); metrics["overall_rmse"]=float(np.mean(metrics["rmse"]))
print(json.dumps({k:np.asarray(v).tolist() for k,v in metrics.items()},indent=2))

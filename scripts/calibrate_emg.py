#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse, json, time, uuid
from datetime import datetime, timezone
import numpy as np
from src.emg.serial_reader import SerialEmgReader

p=argparse.ArgumentParser(); p.add_argument("--port",required=True); p.add_argument("--channels",type=int,default=4); p.add_argument("--seconds",type=float,default=3)
p.add_argument("--output",default="calibration_data/emg_right_calibration.json"); a=p.parse_args(); reader=SerialEmgReader(a.port,channels=a.channels).connect()
poses=("rest","open_palm","fist","index_flexion","middle_flexion","ring_flexion","pinky_flexion","thumb_opposition","pinch"); captured={}
try:
 for pose in poses:
  input(f"Prepare {pose.replace('_',' ')}; press Enter to record {a.seconds:g} s..."); rows=[]; end=time.monotonic()+a.seconds
  while time.monotonic()<end:
   rows.extend(p.samples for p in reader.read())
  if not rows: raise RuntimeError(f"no EMG received for {pose}")
  captured[pose]=np.asarray(rows,float)
finally: reader.close()
all_active=np.concatenate([captured[k] for k in poses if k!="rest"]); baseline=np.median(captured["rest"],axis=0)
scale=np.maximum(np.percentile(np.abs(all_active-baseline),95,axis=0),1.0)
payload={"version":1,"hand":"Right","calibration_id":str(uuid.uuid4()),"created_at":datetime.now(timezone.utc).isoformat(),
 "channels":a.channels,"baseline":baseline.tolist(),"scale":scale.tolist(),"useful_min":np.percentile(all_active,1,axis=0).tolist(),
 "useful_max":np.percentile(all_active,99,axis=0).tolist(),"saturation_fraction":np.mean((all_active<=0)|(all_active>=4095),axis=0).tolist(),
 "poses":{k:{"mean":v.mean(axis=0).tolist(),"std":v.std(axis=0).tolist(),"samples":len(v)} for k,v in captured.items()}}
path=Path(a.output); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(payload,indent=2)); print(f"saved {path} ({payload['calibration_id']})")

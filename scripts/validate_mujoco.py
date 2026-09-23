#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import mujoco_menagerie as mm, numpy as np
from src.backends.mujoco_aero import MujocoAeroBackend
from src.models.aero_command import AeroCommand

b=MujocoAeroBackend(mm.get("tetheria_aero_hand_open").path()/"scene_right.xml").connect()
def run(name,joints):
    b.reset(); b.send(AeroCommand(0,np.asarray(joints,float),np.zeros(7),True,"validation"))
    for _ in range(round(4/b.model.opt.timestep)): b.step()
    state=b.read_state(); assert all(np.isfinite(state[k]).all() for k in ("joint_rad","tendon_length_m","actuator_ctrl"))
    print(f"{name:16s} q={np.round(state['joint_rad'],3)} tendons={np.round(state['tendon_length_m'],5)}"); return state["joint_rad"]
try:
    open_q=run("open",[0]*16)
    for level in (.25,.5,.75,1): run(f"curl_{int(level*100)}pct",[100*level,55*level]+[90*level]*14)
    for name,start,index in (("index",4,0),("middle",7,3),("ring",10,6),("pinky",13,9)):
        q=np.zeros(16); q[start:start+3]=90; assert run(name,q)[index]>open_q[index]+.5
    assert run("thumb_abduction",[100]+[0]*15)[12]>open_q[12]+.5
    assert run("thumb_opposition",[0,55]+[0]*14)[13]>open_q[13]+.5
    assert run("thumb_curl",[0,0,90,90]+[0]*12)[14]>open_q[14]+.5
    print("PASS: official right-hand model remained finite and all requested directions responded")
finally: b.shutdown()

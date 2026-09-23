#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
from src.backends.mujoco_aero import MujocoAeroBackend

p=argparse.ArgumentParser(); p.add_argument("roots",nargs="*",default=["mujoco_menagerie",str(Path.home()/"mujoco_menagerie")]); a=p.parse_args()
path=MujocoAeroBackend.discover(a.roots)
if path is None:
    try:
        import mujoco_menagerie as mm
        candidate=Path(mm.get("tetheria_aero_hand_open").path())/"scene_right.xml"
        if candidate.is_file(): path=candidate
    except Exception: pass
if path is None: raise SystemExit("scene_right.xml not found; install mujoco-menagerie or clone google-deepmind/mujoco_menagerie")
print(path.resolve())

#!/usr/bin/env python3
import sys, argparse, time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.app.runtime import load_config
from src.app.sources import VisionSource
from src.arm.wrist_mapper import WristMapper
from src.arm.existing_arm_adapter import ExistingArmAdapter

p=argparse.ArgumentParser(); p.add_argument("--config"); p.add_argument("--enable-hardware",action="store_true"); p.add_argument("--seconds",type=float,default=5); a=p.parse_args(); cfg=load_config(a.config)
source=VisionSource(cfg.get("camera_id",0),cfg.get("right_hand_confidence",.7),True,cfg["calibration_file"]); arm=None
try:
 input("Hold the calibrated neutral wrist pose and press Enter...")
 state=None
 while state is None: state=source.update()
 mapper=WristMapper(state.palm_rotation,cfg["wrist_axis_order"],cfg["wrist_axis_signs"],cfg["wrist_zero_offsets_rad"],cfg["wrist_limits_rad"])
 if a.enable_hardware:
  from robot_controller import SOArmHardwareController
  arm=ExistingArmAdapter(SOArmHardwareController(),cfg["arm_neutral_rad"],cfg["wrist_target_joints"]).connect()
 for motion in ("pronation / supination","flexion / extension","radial / ulnar deviation"):
  input(f"Press Enter, then slowly demonstrate {motion}..."); end=time.monotonic()+a.seconds
  while time.monotonic()<end:
   state=source.update()
   if state is None: continue
   target=mapper.map(state.palm_rotation); print(motion,dict(zip(cfg["wrist_target_joints"],target.round(3))))
   if arm: arm.send_wrist(target)
finally:
 source.close()
 if arm: arm.close()

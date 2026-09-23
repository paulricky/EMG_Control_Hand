from __future__ import annotations
from pathlib import Path
import time
import numpy as np
from .base import AeroBackend
from .aero_coupling import MUJOCO_ACTUATOR_NAMES, joints_to_mujoco_ctrl


class MujocoAeroBackend(AeroBackend):
    def __init__(self, model_path, show=False): self.model_path=Path(model_path); self.show=show; self.model=self.data=self.viewer=None
    @staticmethod
    def discover(search_roots):
        for root in map(Path,search_roots):
            candidates=(root/"scene_right.xml",root/"tetheria_aero_hand_open"/"scene_right.xml")
            for p in candidates:
                if p.is_file(): return p
            for p in root.glob("tetheria_aero_hand_open-*/scene_right.xml") if root.is_dir() else ():
                return p
        return None
    def connect(self):
        if not self.model_path.is_file():
            try:
                import mujoco_menagerie as mm
                self.model_path=Path(mm.get("tetheria_aero_hand_open").path())/"scene_right.xml"
            except Exception: pass
        if self.model_path.name != "scene_right.xml": raise ValueError("official right-hand scene_right.xml is required")
        if not self.model_path.is_file(): raise FileNotFoundError(f"official scene_right.xml not found: {self.model_path}")
        try: import mujoco
        except ImportError as exc: raise RuntimeError("install mujoco and the official Menagerie model") from exc
        self.mj=mujoco; self.model=mujoco.MjModel.from_xml_path(str(self.model_path)); self.data=mujoco.MjData(self.model)
        actual=tuple(mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_ACTUATOR,i) for i in range(self.model.nu))
        if actual != MUJOCO_ACTUATOR_NAMES: raise RuntimeError(f"unsupported official actuator layout: {actual}")
        if self.show:
            import mujoco.viewer
            self.viewer=mujoco.viewer.launch_passive(self.model,self.data)
        return self
    def send(self, command): self.data.ctrl[:]=joints_to_mujoco_ctrl(command.joint_angles_deg)
    def step(self): self.mj.mj_step(self.model,self.data)
    def reset(self): self.mj.mj_resetData(self.model,self.data)
    def render(self):
        if self.viewer is not None:
            if not self.viewer.is_running(): return False
            self.viewer.sync()
        return True
    def read_state(self):
        return {"timestamp_s":time.monotonic(),"joint_rad":self.data.qpos.copy(),"actuator_ctrl":self.data.ctrl.copy(),
                "tendon_length_m":self.data.ten_length.copy()}
    def shutdown(self):
        if self.viewer is not None: self.viewer.close()
    def handle_fault(self, action, last_safe_command=None):
        if action == "controlled_open": self.data.ctrl[:]=joints_to_mujoco_ctrl(np.zeros(16))
        elif action == "disable": self.data.ctrl[:]=0

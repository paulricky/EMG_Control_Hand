from __future__ import annotations
import time
import numpy as np
from src.models.aero_command import AeroCommand, JOINT_LIMITS_DEG


class SafetyError(RuntimeError): pass


class SafetySupervisor:
    def __init__(self, timeout_s=0.5, fault_action="hold", max_current_ma=1200, max_temperature_c=65, max_velocity_rpm=100, communication_timeout_s=0.5):
        if fault_action not in {"hold", "controlled_open", "disable"}: raise ValueError("invalid fault_action")
        self.timeout_s=timeout_s; self.fault_action=fault_action; self.max_current_ma=max_current_ma
        self.max_temperature_c=max_temperature_c; self.max_velocity_rpm=max_velocity_rpm
        self.communication_timeout_s=communication_timeout_s

    def validate(self, command, now=None):
        now=time.monotonic() if now is None else now
        if not command.valid or now-command.timestamp_s > self.timeout_s: raise SafetyError("stale or invalid command")
        j=np.asarray(command.joint_angles_deg); q=np.asarray(command.compact01)
        if not np.isfinite(j).all() or not np.isfinite(q).all(): raise SafetyError("non-finite command")
        return AeroCommand(command.timestamp_s, np.clip(j,0,JOINT_LIMITS_DEG), np.clip(q,0,1), True, command.source)

    def check_telemetry(self, state, now=None):
        now=time.monotonic() if now is None else now
        if not state.get("dry_run",False) and now-float(state.get("timestamp_s",0))>self.communication_timeout_s:
            raise SafetyError("hardware communication timeout")
        checks=(("current_ma",self.max_current_ma),("temperature_c",self.max_temperature_c),("velocity_rpm",self.max_velocity_rpm))
        for key, limit in checks:
            values=np.abs(np.asarray(state.get(key, []),float))
            if values.size and (not np.isfinite(values).all() or np.max(values)>limit): raise SafetyError(f"unsafe {key}")

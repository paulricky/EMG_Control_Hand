import math
import numpy as np
from src.models.aero_command import AeroCommand, JOINT_LIMITS_DEG
from src.control.command_mapper import full_to_compact01


class ExponentialFilter:
    def __init__(self, tau_s: float): self.tau_s, self.value, self.timestamp = float(tau_s), None, None
    def reset(self): self.value = self.timestamp = None
    def update(self, value, timestamp_s):
        x = np.asarray(value, float)
        if not np.isfinite(x).all(): raise ValueError("non-finite filter input")
        if self.value is None: self.value = x.copy()
        else:
            dt = max(0.0, float(timestamp_s) - self.timestamp)
            alpha = 1.0 if self.tau_s <= 0 else 1.0 - math.exp(-dt / self.tau_s)
            self.value += alpha * (x - self.value)
        self.timestamp = float(timestamp_s)
        return self.value.copy()


class SlewRateLimiter:
    def __init__(self, max_rate_per_s): self.rate=np.asarray(max_rate_per_s,float); self.value=None; self.timestamp=None
    def update(self, target, timestamp_s):
        x=np.asarray(target,float)
        if self.value is None: self.value=x.copy()
        else:
            dt=max(0,float(timestamp_s)-self.timestamp)
            self.value += np.clip(x-self.value, -self.rate*dt, self.rate*dt)
        self.timestamp=float(timestamp_s); return self.value.copy()


class CommandConditioner:
    """Low-pass, then slew-limit normalized 16-joint targets."""
    def __init__(self,tau_s=.08,slew_rate_01_per_s=(1,)*7):
        rates=np.asarray(slew_rate_01_per_s,float).reshape(7)
        rates16=np.array([rates[0],rates[1],rates[2],rates[2],*([rates[3]]*3),*([rates[4]]*3),*([rates[5]]*3),*([rates[6]]*3)])
        self.lowpass=ExponentialFilter(tau_s); self.slew=SlewRateLimiter(rates16)
    def update(self,command):
        q=np.asarray(command.joint_angles_deg)/JOINT_LIMITS_DEG
        q=self.lowpass.update(q,command.timestamp_s); q=np.clip(self.slew.update(q,command.timestamp_s),0,1)
        return AeroCommand(command.timestamp_s,q*JOINT_LIMITS_DEG,full_to_compact01(q*JOINT_LIMITS_DEG),command.valid,command.source)

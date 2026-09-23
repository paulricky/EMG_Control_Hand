from __future__ import annotations
import time
import numpy as np
from .base import AeroBackend


class RealAeroBackend(AeroBackend):
    """Official SDK backend. Motion is impossible unless enable_hardware=True."""
    def __init__(self, port=None, baudrate=921600, enable_hardware=False, sdk_factory=None, speed=5000, torque=300):
        self.port=port; self.baudrate=baudrate; self.enabled=bool(enable_hardware); self.factory=sdk_factory
        self.hand=None; self.last_command=None; self.speed=speed; self.torque=torque

    def connect(self):
        if not self.enabled: return self
        if self.factory is None:
            try: from aero_open_sdk.aero_hand import AeroHand
            except ImportError as exc: raise RuntimeError("install aero-open-sdk to enable hardware") from exc
            self.factory=AeroHand
        self.hand=self.factory(port=self.port, baudrate=self.baudrate)
        if len(tuple(self.hand.joint_names)) != 16: raise RuntimeError("unexpected Aero SDK joint layout")
        for i in range(7):
            if callable(getattr(self.hand,"set_speed",None)): self.hand.set_speed(i,self.speed)
            if callable(getattr(self.hand,"set_torque",None)): self.hand.set_torque(i,self.torque)
        # A telemetry read verifies communication; firmware performs automatic homing on current releases.
        self.hand.get_actuations()
        return self

    def send(self, command):
        self.last_command=command
        if self.enabled:
            if self.hand is None: raise RuntimeError("hardware backend is not connected")
            self.hand.set_joint_positions(np.asarray(command.joint_angles_deg).tolist())

    def read_state(self):
        if not self.enabled or self.hand is None: return {"dry_run": True, "timestamp_s": time.monotonic()}
        return {"dry_run": False, "timestamp_s": time.monotonic(),
                "actuation_deg": np.asarray(self.hand.get_actuations()),
                "current_ma": np.asarray(self.hand.get_actuator_currents()),
                "velocity_rpm": np.asarray(self.hand.get_actuator_speeds()),
                "temperature_c": np.asarray(self.hand.get_actuator_temperatures())}

    def shutdown(self):
        hand,self.hand=self.hand,None
        if hand is not None:
            for name in ("close", "disconnect"):
                method=getattr(hand,name,None)
                if callable(method): method(); break

    def handle_fault(self, action, last_safe_command=None):
        if not self.enabled or self.hand is None: return
        if action == "hold" and last_safe_command is not None:
            self.hand.set_joint_positions(np.asarray(last_safe_command.joint_angles_deg).tolist())
        elif action == "controlled_open":
            self.hand.set_joint_positions([0.0] * 16)
        elif action == "disable":
            for i in range(7): self.hand.set_torque(i, 0)

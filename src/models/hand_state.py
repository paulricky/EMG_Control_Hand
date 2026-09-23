from __future__ import annotations

from dataclasses import dataclass
import numpy as np


def _immutable_array(value, shape):
    array = np.asarray(value, dtype=np.float64).reshape(shape).copy()
    if not np.all(np.isfinite(array)):
        raise ValueError("state arrays must contain only finite values")
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class HumanHandState:
    """Canonical right-hand state. Quaternion convention is (x, y, z, w)."""

    timestamp_s: float
    joint_angles_rad: np.ndarray
    compact01: np.ndarray
    palm_rotation: np.ndarray
    palm_quaternion: np.ndarray
    tracking_confidence: float
    handedness: str = "Right"
    joint_normalized01: np.ndarray | None = None

    def __post_init__(self):
        object.__setattr__(self, "joint_angles_rad", _immutable_array(self.joint_angles_rad, (16,)))
        object.__setattr__(self, "compact01", _immutable_array(self.compact01, (7,)))
        object.__setattr__(self, "palm_rotation", _immutable_array(self.palm_rotation, (3, 3)))
        object.__setattr__(self, "palm_quaternion", _immutable_array(self.palm_quaternion, (4,)))
        normalized = self.compact01 if self.joint_normalized01 is None else self.joint_normalized01
        if np.asarray(normalized).size == 7:
            q = np.asarray(normalized, dtype=float).reshape(7)
            normalized = np.array([q[0], q[1], q[2], q[2], *([q[3]]*3), *([q[4]]*3), *([q[5]]*3), *([q[6]]*3)])
        object.__setattr__(self, "joint_normalized01", _immutable_array(np.clip(normalized, 0, 1), (16,)))
        if self.handedness != "Right":
            raise ValueError("this implementation accepts right hands only")
        if not 0.0 <= float(self.tracking_confidence) <= 1.0:
            raise ValueError("tracking_confidence must be in [0, 1]")

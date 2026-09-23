from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .hand_state import _immutable_array


JOINT_NAMES = (
    "thumb_cmc_abd", "thumb_cmc_flex", "thumb_mcp", "thumb_ip",
    "index_mcp", "index_pip", "index_dip",
    "middle_mcp", "middle_pip", "middle_dip",
    "ring_mcp", "ring_pip", "ring_dip",
    "pinky_mcp", "pinky_pip", "pinky_dip",
)
COMPACT_NAMES = (
    "thumb_cmc_abd", "thumb_cmc_flex", "thumb_curl", "index_curl",
    "middle_curl", "ring_curl", "pinky_curl",
)
JOINT_LIMITS_DEG = np.array([100, 55, 90, 90] + [90] * 12, dtype=np.float64)


@dataclass(frozen=True)
class AeroCommand:
    timestamp_s: float
    joint_angles_deg: np.ndarray
    compact01: np.ndarray
    valid: bool
    source: str

    def __post_init__(self):
        object.__setattr__(self, "joint_angles_deg", _immutable_array(self.joint_angles_deg, (16,)))
        object.__setattr__(self, "compact01", _immutable_array(self.compact01, (7,)))

    @property
    def compact_deg(self) -> np.ndarray:
        return self.compact01 * np.array([100, 55, 90, 90, 90, 90, 90], dtype=float)

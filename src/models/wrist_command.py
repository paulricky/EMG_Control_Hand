from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class WristCommand:
    timestamp_s: float
    rotation_vector_rad: np.ndarray
    joints_rad: np.ndarray
    valid: bool = True

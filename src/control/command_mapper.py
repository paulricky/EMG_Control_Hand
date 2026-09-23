from __future__ import annotations

import numpy as np
from src.models.aero_command import AeroCommand, JOINT_LIMITS_DEG


class CommandMapper:
    def __init__(self, finger_weights=(1/3, 1/3, 1/3), thumb_weights=(0.5, 0.5)):
        self.finger_weights = self._weights(finger_weights, 3)
        self.thumb_weights = self._weights(thumb_weights, 2)

    @staticmethod
    def _weights(values, size):
        w = np.asarray(values, float).reshape(size)
        if np.any(w < 0) or not np.isfinite(w).all() or w.sum() <= 0: raise ValueError("invalid weights")
        return w / w.sum()

    def map(self, state, source="vision") -> AeroCommand:
        # Vision supplies per-user calibrated anatomical motion. EMG supplies an
        # expanded seven-channel prediction through the same field.
        q16 = np.clip(np.asarray(state.joint_normalized01), 0, 1)
        compact = np.asarray(state.compact01, float).copy()
        compact[2] = self.thumb_weights @ q16[2:4]
        for i, start in enumerate((4, 7, 10, 13), 3): compact[i] = self.finger_weights @ q16[start:start+3]
        compact = np.clip(compact, 0, 1)
        full = q16 * JOINT_LIMITS_DEG
        full[0] = compact[0] * 100; full[1] = compact[1] * 55
        return AeroCommand(state.timestamp_s, full, compact, True, source)


def compact_to_full_deg(compact01):
    q = np.clip(np.asarray(compact01, float).reshape(7), 0, 1)
    deg = q * np.array([100, 55, 90, 90, 90, 90, 90])
    return np.array([deg[0], deg[1], deg[2], deg[2], *([deg[3]]*3), *([deg[4]]*3), *([deg[5]]*3), *([deg[6]]*3)])


def full_to_compact01(full_deg, finger_weights=(1/3, 1/3, 1/3)):
    x = np.clip(np.asarray(full_deg, float).reshape(16), 0, JOINT_LIMITS_DEG) / JOINT_LIMITS_DEG
    w = np.asarray(finger_weights, float); w /= w.sum()
    return np.array([x[0], x[1], np.mean(x[2:4]), *(w @ x[s:s+3] for s in (4,7,10,13))])

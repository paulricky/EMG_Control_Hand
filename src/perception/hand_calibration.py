from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from datetime import datetime, timezone
import numpy as np


@dataclass(frozen=True)
class HandCalibration:
    open_rad: np.ndarray
    closed_rad: np.ndarray
    hand: str = "Right"
    version: int = 1
    calibration_id: str = "uncalibrated"
    thumb_poses_rad: dict | None = None
    created_at: str = ""

    def normalize(self, angles) -> np.ndarray:
        x, lo, hi = map(lambda a: np.asarray(a, float), (angles, self.open_rad, self.closed_rad))
        delta = hi - lo
        safe = np.where(np.abs(delta) < 1e-5, 1.0, delta)
        return np.clip((x - lo) / safe, 0.0, 1.0)

    def save(self, path):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"version": self.version, "hand": self.hand,
            "calibration_id": self.calibration_id, "open_rad": np.asarray(self.open_rad).tolist(),
            "closed_rad": np.asarray(self.closed_rad).tolist(), "thumb_poses_rad": self.thumb_poses_rad or {},
            "created_at": self.created_at or datetime.now(timezone.utc).isoformat()}, indent=2))

    @classmethod
    def load(cls, path):
        d = json.loads(Path(path).read_text())
        if d.get("hand") != "Right": raise ValueError("right-hand calibration required")
        return cls(np.array(d["open_rad"]), np.array(d["closed_rad"]), d["hand"], d["version"],
                   d["calibration_id"], d.get("thumb_poses_rad", {}), d.get("created_at", ""))

    @classmethod
    def from_pose_samples(cls, poses, calibration_id):
        required={"open_palm","closed_fist","thumb_abduction","thumb_opposition","pinch"}
        if set(poses) < required: raise ValueError(f"missing calibration poses: {sorted(required-set(poses))}")
        median={k:np.median(np.asarray(v,float),axis=0) for k,v in poses.items()}
        opened=median["open_palm"].copy(); closed=median["closed_fist"].copy()
        closed[0]=median["thumb_abduction"][0]
        closed[1]=median["thumb_opposition"][1]
        closed[2:4]=np.maximum.reduce([closed[2:4],median["thumb_opposition"][2:4],median["pinch"][2:4]])
        return cls(opened,closed,"Right",2,calibration_id,
                   {k:median[k][:4].tolist() for k in ("thumb_abduction","thumb_opposition","pinch")},
                   datetime.now(timezone.utc).isoformat())

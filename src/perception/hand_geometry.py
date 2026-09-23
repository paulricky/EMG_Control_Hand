from __future__ import annotations

import numpy as np


def normalize(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    if n < eps:
        raise ValueError("degenerate landmark geometry")
    return v / n


def internal_angle(a, b, c) -> float:
    ba, bc = np.asarray(a, float) - b, np.asarray(c, float) - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom < 1e-12:
        raise ValueError("angle requires non-coincident points")
    return float(np.arccos(np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)))


def flexion_angle(a, b, c) -> float:
    return float(np.pi - internal_angle(a, b, c))


def palm_frame(landmarks) -> np.ndarray:
    """Columns are right-hand radial, distal, and normal axes in camera space."""
    p = np.asarray(landmarks, dtype=float).reshape(21, 3)
    x = normalize(p[5] - p[17])
    y_temp = normalize(p[9] - p[0])
    z = normalize(np.cross(x, y_temp))
    y = normalize(np.cross(z, x))
    result = np.column_stack((x, y, z))
    if np.linalg.det(result) < 0.0:
        raise AssertionError("palm frame is not right-handed")
    return result


def matrix_to_quaternion_xyzw(r):
    r = np.asarray(r, float).reshape(3, 3)
    # Stable eigen decomposition avoids an optional scipy dependency.
    k = np.array([
        [r[0,0]-r[1,1]-r[2,2], r[1,0]+r[0,1], r[2,0]+r[0,2], r[2,1]-r[1,2]],
        [r[1,0]+r[0,1], r[1,1]-r[0,0]-r[2,2], r[2,1]+r[1,2], r[0,2]-r[2,0]],
        [r[2,0]+r[0,2], r[2,1]+r[1,2], r[2,2]-r[0,0]-r[1,1], r[1,0]-r[0,1]],
        [r[2,1]-r[1,2], r[0,2]-r[2,0], r[1,0]-r[0,1], r[0,0]+r[1,1]+r[2,2]],
    ]) / 3.0
    values, vectors = np.linalg.eigh(k)
    q = vectors[:, np.argmax(values)]
    if q[3] < 0: q = -q
    return q


def anatomical_angles(landmarks) -> np.ndarray:
    """Estimate the 16 anatomical flexion channels from MediaPipe world landmarks."""
    p = np.asarray(landmarks, float).reshape(21, 3)
    r = palm_frame(p)
    local = (p - p[0]) @ r
    out = np.zeros(16, float)
    # Thumb quantities are measured only in palm coordinates.
    thumb = local[1:5]
    v = thumb[1] - thumb[0]
    out[0] = abs(float(np.arctan2(v[0], v[1])))
    out[1] = abs(float(np.arctan2(v[2], np.linalg.norm(v[:2]))))
    out[2] = flexion_angle(thumb[0], thumb[1], thumb[2])
    out[3] = flexion_angle(thumb[1], thumb[2], thumb[3])
    for offset, base in zip((4, 7, 10, 13), (5, 9, 13, 17)):
        # MCP uses wrist as the proximal palm reference; PIP/DIP use joint triplets.
        out[offset] = flexion_angle(local[0], local[base], local[base + 1])
        out[offset + 1] = flexion_angle(local[base], local[base + 1], local[base + 2])
        out[offset + 2] = flexion_angle(local[base + 1], local[base + 2], local[base + 3])
    return np.clip(out, 0.0, np.pi)

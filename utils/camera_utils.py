"""Pure NumPy camera and coordinate utilities."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    """Convert a wxyz quaternion to a 3x3 rotation matrix."""
    q = np.asarray(qvec, dtype=np.float64)
    q = q / max(np.linalg.norm(q), 1e-12)
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def rotmat_to_qvec(rotmat: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a wxyz quaternion."""
    m = np.asarray(rotmat, dtype=np.float64)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        q = np.array(
            [
                0.25 * s,
                (m[2, 1] - m[1, 2]) / s,
                (m[0, 2] - m[2, 0]) / s,
                (m[1, 0] - m[0, 1]) / s,
            ],
            dtype=np.float64,
        )
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        q = np.array(
            [
                (m[2, 1] - m[1, 2]) / s,
                0.25 * s,
                (m[0, 1] + m[1, 0]) / s,
                (m[0, 2] + m[2, 0]) / s,
            ],
            dtype=np.float64,
        )
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        q = np.array(
            [
                (m[0, 2] - m[2, 0]) / s,
                (m[0, 1] + m[1, 0]) / s,
                0.25 * s,
                (m[1, 2] + m[2, 1]) / s,
            ],
            dtype=np.float64,
        )
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        q = np.array(
            [
                (m[1, 0] - m[0, 1]) / s,
                (m[0, 2] + m[2, 0]) / s,
                (m[1, 2] + m[2, 1]) / s,
                0.25 * s,
            ],
            dtype=np.float64,
        )
    q = q / max(np.linalg.norm(q), 1e-12)
    return q if q[0] >= 0.0 else -q


def intrinsic_matrix(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def focal_to_fov(focal: float, pixels: int) -> float:
    return 2.0 * math.atan(float(pixels) / (2.0 * float(focal)))


def fov_to_focal(fov: float, pixels: int) -> float:
    return float(pixels) / (2.0 * math.tan(float(fov) / 2.0))


def colmap_image_to_c2w(qvec: np.ndarray, tvec: np.ndarray, opengl: bool = True) -> np.ndarray:
    """Convert COLMAP world-to-camera q/t into camera-to-world.

    COLMAP camera axes are x right, y down, z forward.  With ``opengl=True``
    the returned camera axes become x right, y up, z backward.
    """
    rot = qvec_to_rotmat(qvec)
    w2c = np.eye(4, dtype=np.float64)
    w2c[:3, :3] = rot
    w2c[:3, 3] = np.asarray(tvec, dtype=np.float64)
    c2w = np.linalg.inv(w2c)
    if opengl:
        c2w[:3, 1:3] *= -1.0
    return c2w


def c2w_to_viewmat(c2w: np.ndarray) -> np.ndarray:
    return np.linalg.inv(np.asarray(c2w, dtype=np.float64))


def estimate_scene_extent(c2w_matrices: np.ndarray, percentile: float = 90.0) -> float:
    centers = np.asarray(c2w_matrices, dtype=np.float64)[:, :3, 3]
    distances = np.linalg.norm(centers - centers.mean(axis=0, keepdims=True), axis=-1)
    return max(float(np.percentile(distances, percentile)), 1e-6)


def average_pose_center(c2w_matrices: Iterable[np.ndarray]) -> np.ndarray:
    poses = np.asarray(list(c2w_matrices), dtype=np.float64)
    return poses[:, :3, 3].mean(axis=0)

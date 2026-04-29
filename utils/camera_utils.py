"""相机数学工具：四元数、旋转矩阵、投影矩阵、坐标系转换。

无 PyTorch/JAX 依赖，仅用 NumPy，可被任何框架复用。
"""

import math
from typing import List

import numpy as np


def qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    """COLMAP wxyz 四元数 → (3,3) 旋转矩阵。"""
    w, x, y, z = qvec
    return np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - w*z),       2*(x*z + w*y)],
        [2*(x*y + w*z),        1 - 2*(x*x + z*z),   2*(y*z - w*x)],
        [2*(x*z - w*y),        2*(y*z + w*x),        1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


def rotmat_to_qvec(R: np.ndarray) -> np.ndarray:
    """(3,3) 旋转矩阵 → COLMAP wxyz 四元数。"""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z], dtype=np.float64)


def build_intrinsic_matrix(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """构建 (3,3) 相机内参矩阵（OpenCV 坐标系）。"""
    return np.array([
        [fx,  0, cx],
        [ 0, fy, cy],
        [ 0,  0,  1],
    ], dtype=np.float64)


def build_projection_matrix(
    fx: float, fy: float, cx: float, cy: float,
    width: int, height: int,
    znear: float = 0.01, zfar: float = 100.0,
) -> np.ndarray:
    """构建 (4,4) OpenGL 投影矩阵，供 diff-gaussian-rasterization 使用。

    NDC z 轴映射到 [-1, 1]。
    """
    top    =  znear * cy / fy
    bottom = -znear * (height - cy) / fy
    left   = -znear * cx / fx
    right  =  znear * (width - cx) / fx

    P = np.zeros((4, 4), dtype=np.float64)
    P[0, 0] =  2 * znear / (right - left)
    P[1, 1] =  2 * znear / (top - bottom)
    P[0, 2] =  (right + left) / (right - left)
    P[1, 2] =  (top + bottom) / (top - bottom)
    P[2, 2] = -(zfar + znear) / (zfar - znear)
    P[2, 3] = -2 * zfar * znear / (zfar - znear)
    P[3, 2] = -1.0
    return P


def focal_to_fov(focal: float, dimension: int) -> float:
    """焦距 → 视场角（弧度）。"""
    return 2 * math.atan(dimension / (2 * focal))


def fov_to_focal(fov_rad: float, dimension: int) -> float:
    """视场角（弧度）→ 焦距。"""
    return dimension / (2 * math.tan(fov_rad / 2))


def colmap_to_opengl_c2w(qvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """COLMAP w2c（qvec/tvec）→ OpenGL c2w (4,4)。

    COLMAP 坐标系：[right, down, forward]
    OpenGL/3DGS 坐标系：[right, up, backward]
    翻转：乘以 diag([1, -1, -1, 1])（在位姿矩阵右乘）。
    """
    R = qvec_to_rotmat(qvec)
    t = tvec.reshape(3, 1)
    bottom = np.array([[0, 0, 0, 1]], dtype=np.float64)
    w2c = np.concatenate([np.concatenate([R, t], axis=1), bottom], axis=0)
    c2w = np.linalg.inv(w2c)
    # 坐标系翻转：COLMAP → OpenGL/NeRF
    c2w[:3, 1:3] *= -1
    return c2w


def w2c_to_c2w(w2c: np.ndarray) -> np.ndarray:
    """(4,4) world-to-camera → camera-to-world。"""
    return np.linalg.inv(w2c)


def c2w_to_w2c(c2w: np.ndarray) -> np.ndarray:
    """(4,4) camera-to-world → world-to-camera。"""
    return np.linalg.inv(c2w)


def interpolate_camera_path(
    c2w_list: List[np.ndarray],
    num_frames: int,
    method: str = "slerp",
) -> List[np.ndarray]:
    """生成平滑相机路径。

    Args:
        c2w_list: 关键帧列表，每个元素为 (4,4) c2w 矩阵。
        num_frames: 输出帧数。
        method: "slerp"（球面线性插值旋转 + 线性位移）或 "linear"。

    Returns:
        num_frames 个 (4,4) c2w 矩阵列表。
    """
    n = len(c2w_list)
    if n == 1:
        return [c2w_list[0]] * num_frames

    quats = np.stack([rotmat_to_qvec(c2w[:3, :3]) for c2w in c2w_list])
    positions = np.stack([c2w[:3, 3] for c2w in c2w_list])

    # 在关键帧间均匀采样
    t_keys = np.linspace(0, 1, n)
    t_query = np.linspace(0, 1, num_frames)

    result = []
    for t in t_query:
        # 找到相邻关键帧
        idx = min(int(t * (n - 1)), n - 2)
        alpha = (t - t_keys[idx]) / (t_keys[idx + 1] - t_keys[idx] + 1e-8)
        alpha = float(np.clip(alpha, 0, 1))

        # 位移线性插值
        pos = (1 - alpha) * positions[idx] + alpha * positions[idx + 1]

        if method == "slerp":
            R = _slerp_rotmat(quats[idx], quats[idx + 1], alpha)
        else:
            # Nlerp（归一化线性插值）作为线性模式的回退
            q = (1 - alpha) * quats[idx] + alpha * quats[idx + 1]
            q = q / np.linalg.norm(q)
            R = qvec_to_rotmat(q)

        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, :3] = R
        c2w[:3, 3] = pos
        result.append(c2w)

    return result


def _slerp_rotmat(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """两个 wxyz 四元数之间的球面线性插值，返回旋转矩阵。"""
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = np.dot(q0, q1)
    if dot < 0:
        q1 = -q1
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        q = q0 + t * (q1 - q0)
    else:
        theta_0 = math.acos(dot)
        theta = theta_0 * t
        sin_theta = math.sin(theta)
        sin_theta_0 = math.sin(theta_0)
        s0 = math.cos(theta) - dot * sin_theta / sin_theta_0
        s1 = sin_theta / sin_theta_0
        q = s0 * q0 + s1 * q1
    q = q / np.linalg.norm(q)
    return qvec_to_rotmat(q)

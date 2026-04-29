"""球谐函数（Spherical Harmonics）工具。

提供 0~3 阶实球谐的解析式求值，以及 RGB ↔ SH DC 系数转换。
纯 PyTorch 数学运算，无 I/O 或模型状态。

SH 多项式系数与官方 3DGS 仓库（gaussian-splatting）保持一致。
"""

import math
from typing import Optional

import torch
from torch import Tensor


# ---- 0 阶 ----
SH_C0 = 0.28209479177387814

# ---- 1 阶 ----
SH_C1 = 0.4886025119029199

# ---- 2 阶 ----
SH_C2 = [
     1.0925484305920792,
    -1.0925484305920792,
     0.31539156525252005,
    -1.0925484305920792,
     0.5462742152960396,
]

# ---- 3 阶 ----
SH_C3 = [
    -0.5900435899266435,
     2.890611442640554,
    -0.4570457994644658,
     0.3731763325901154,
    -0.4570457994644658,
     1.445305721320277,
    -0.5900435899266435,
]


def num_sh_coefficients(degree: int) -> int:
    """返回 degree 阶（含）以下的 SH 系数总数：(degree+1)^2。"""
    return (degree + 1) ** 2


def eval_sh(
    degree: int,
    sh_coeffs: Tensor,      # (N, (degree+1)^2, 3)
    directions: Tensor,     # (N, 3) 单位向量
) -> Tensor:                # (N, 3) 输出 RGB（调用方应 clamp 到合法范围）
    """解析式求值球谐函数（0~3 阶），计算视角相关颜色。

    Args:
        degree: SH 阶数，0~3。
        sh_coeffs: SH 系数张量，形状 (N, (degree+1)^2, 3)。
        directions: 单位方向向量，形状 (N, 3)，从高斯指向相机。

    Returns:
        视角相关颜色，形状 (N, 3)，需调用方自行 clamp 或加 0.5。
    """
    assert 0 <= degree <= 3, "仅支持 0~3 阶"
    assert sh_coeffs.shape[1] == num_sh_coefficients(degree)

    x = directions[:, 0:1]
    y = directions[:, 1:2]
    z = directions[:, 2:3]

    result = SH_C0 * sh_coeffs[:, 0, :]   # (N, 3)

    if degree < 1:
        return result

    result = (
        result
        - SH_C1 * y * sh_coeffs[:, 1, :]
        + SH_C1 * z * sh_coeffs[:, 2, :]
        - SH_C1 * x * sh_coeffs[:, 3, :]
    )

    if degree < 2:
        return result

    xx, yy, zz = x * x, y * y, z * z
    xy, yz, xz = x * y, y * z, x * z

    result = (
        result
        + SH_C2[0] * xy * sh_coeffs[:, 4, :]
        + SH_C2[1] * yz * sh_coeffs[:, 5, :]
        + SH_C2[2] * (2.0 * zz - xx - yy) * sh_coeffs[:, 6, :]
        + SH_C2[3] * xz * sh_coeffs[:, 7, :]
        + SH_C2[4] * (xx - yy) * sh_coeffs[:, 8, :]
    )

    if degree < 3:
        return result

    result = (
        result
        + SH_C3[0] * y * (3.0 * xx - yy) * sh_coeffs[:, 9, :]
        + SH_C3[1] * xy * z * sh_coeffs[:, 10, :]
        + SH_C3[2] * y * (4.0 * zz - xx - yy) * sh_coeffs[:, 11, :]
        + SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * sh_coeffs[:, 12, :]
        + SH_C3[4] * x * (4.0 * zz - xx - yy) * sh_coeffs[:, 13, :]
        + SH_C3[5] * z * (xx - yy) * sh_coeffs[:, 14, :]
        + SH_C3[6] * x * (xx - 3.0 * yy) * sh_coeffs[:, 15, :]
    )

    return result


def rgb_to_sh_dc(rgb: Tensor) -> Tensor:
    """线性 RGB [0,1] → 0 阶 SH 系数（DC 分量）。

    公式：sh_dc = (rgb - 0.5) / C0

    输入输出形状相同，如 (..., 3)。
    """
    return (rgb - 0.5) / SH_C0


def sh_dc_to_rgb(sh_dc: Tensor) -> Tensor:
    """0 阶 SH 系数 → 线性 RGB。

    公式：rgb = C0 * sh_dc + 0.5
    """
    return SH_C0 * sh_dc + 0.5


def get_view_directions(
    means: Tensor,          # (N, 3) 高斯位置（世界坐标）
    camera_center: Tensor,  # (3,)  相机中心（世界坐标）
) -> Tensor:                # (N, 3) 单位向量，从高斯指向相机
    """计算每个高斯到相机的归一化方向向量。"""
    dirs = camera_center[None, :] - means   # (N, 3)
    norms = dirs.norm(dim=1, keepdim=True).clamp(min=1e-8)
    return dirs / norms

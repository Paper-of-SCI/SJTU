"""3DGS 训练损失函数。

所有函数均为可微分 PyTorch 运算，无模型状态。
提供标准光度损失（L1 + SSIM）和适合水下场景的可选正则化项。
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


def l1_loss(pred: Tensor, gt: Tensor) -> Tensor:
    """L1 光度损失。

    Args:
        pred: (3, H, W) 渲染图像。
        gt:   (3, H, W) GT 图像。

    Returns:
        标量损失值。
    """
    return F.l1_loss(pred, gt)


def ssim_loss(
    pred: Tensor,
    gt: Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
) -> Tensor:
    """可微分 SSIM，返回 SSIM 值（越高越好，范围 [0,1]）。

    使用可分离高斯核，匹配 3DGS 论文的评估协议。
    调用方若用作损失，请取 1 - ssim 或 (1 - ssim) / 2。

    Args:
        pred: (3, H, W) 或 (B, 3, H, W)。
        gt:   同上。
        window_size: 高斯窗大小。
        sigma: 高斯核标准差。

    Returns:
        SSIM 标量。
    """
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
        gt   = gt.unsqueeze(0)

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    channel = pred.shape[1]

    # 构建可分离高斯核
    kernel_1d = _gaussian_kernel_1d(window_size, sigma, device=pred.device, dtype=pred.dtype)
    kernel_2d = kernel_1d[:, None] @ kernel_1d[None, :]          # (ws, ws)
    kernel = kernel_2d.expand(channel, 1, window_size, window_size)  # (C, 1, ws, ws)

    pad = window_size // 2
    mu1    = F.conv2d(pred, kernel, padding=pad, groups=channel)
    mu2    = F.conv2d(gt,   kernel, padding=pad, groups=channel)

    mu1_sq  = mu1 * mu1
    mu2_sq  = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, kernel, padding=pad, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(gt   * gt,   kernel, padding=pad, groups=channel) - mu2_sq
    sigma12   = F.conv2d(pred * gt,   kernel, padding=pad, groups=channel) - mu1_mu2

    num   = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denom = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim_map = num / (denom + 1e-12)

    return ssim_map.mean()


def photometric_loss(
    pred: Tensor,
    gt: Tensor,
    lambda_dssim: float = 0.2,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    """标准 3DGS 光度损失：(1-λ)*L1 + λ*(1-SSIM)/2。

    Args:
        pred: (3, H, W) 渲染图像。
        gt:   (3, H, W) GT 图像。
        lambda_dssim: DSSIM 权重（论文默认 0.2）。

    Returns:
        (total_loss, components)，components 含键：'l1'、'ssim'、'dssim'、'total'。
    """
    l1  = l1_loss(pred, gt)
    ssim_val = ssim_loss(pred, gt)
    dssim = (1.0 - ssim_val) / 2.0
    total = (1.0 - lambda_dssim) * l1 + lambda_dssim * dssim
    return total, {
        "l1":    l1,
        "ssim":  ssim_val,
        "dssim": dssim,
        "total": total,
    }


def depth_regularization_loss(
    depth: Tensor,                      # (1, H, W)
    mask: Optional[Tensor] = None,      # (1, H, W) bool，True 为有效区域
) -> Tensor:
    """深度总变差（TV）平滑正则化。

    水下场景几何条件差，该正则化有助于减少浮动伪影。

    Returns:
        标量损失。
    """
    if mask is not None:
        depth = depth * mask.float()
    dy = (depth[:, :, 1:, :] - depth[:, :, :-1, :]).abs()
    dx = (depth[:, :, :, 1:] - depth[:, :, :, :-1]).abs()
    return dy.mean() + dx.mean()


def opacity_entropy_loss(opacities: Tensor) -> Tensor:
    """不透明度熵正则化，促使 α 趋向 0 或 1。

    公式：H = -α*log(α) - (1-α)*log(1-α)

    Args:
        opacities: (N, 1) post-sigmoid 不透明度，值域 (0, 1)。

    Returns:
        标量损失（均值）。
    """
    alpha = opacities.clamp(1e-6, 1 - 1e-6)
    entropy = -(alpha * torch.log(alpha) + (1 - alpha) * torch.log(1 - alpha))
    return entropy.mean()


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------

def _gaussian_kernel_1d(
    window_size: int,
    sigma: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """生成 1D 归一化高斯核，形状 (window_size,)。"""
    coords = torch.arange(window_size, device=device, dtype=dtype)
    coords -= window_size // 2
    kernel = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    return kernel / kernel.sum()

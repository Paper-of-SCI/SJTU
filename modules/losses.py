"""Stateless differentiable losses for external 3DGS training loops."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


def l1_loss(pred: Tensor, target: Tensor) -> Tensor:
    return F.l1_loss(pred, target)


def ssim(pred: Tensor, target: Tensor, window_size: int = 11, sigma: float = 1.5) -> Tensor:
    """Differentiable SSIM for CHW or BCHW tensors."""
    if pred.ndim == 3:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)
    channels = pred.shape[1]
    kernel_1d = _gaussian_1d(window_size, sigma, pred.device, pred.dtype)
    kernel_2d = (kernel_1d[:, None] @ kernel_1d[None, :]).expand(channels, 1, window_size, window_size)
    pad = window_size // 2

    mu_x = F.conv2d(pred, kernel_2d, padding=pad, groups=channels)
    mu_y = F.conv2d(target, kernel_2d, padding=pad, groups=channels)
    mu_x2 = mu_x.square()
    mu_y2 = mu_y.square()
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(pred * pred, kernel_2d, padding=pad, groups=channels) - mu_x2
    sigma_y2 = F.conv2d(target * target, kernel_2d, padding=pad, groups=channels) - mu_y2
    sigma_xy = F.conv2d(pred * target, kernel_2d, padding=pad, groups=channels) - mu_xy

    c1 = 0.01**2
    c2 = 0.03**2
    score = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / ((mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2) + 1e-12)
    return score.mean()


def photometric_loss(pred: Tensor, gt: Tensor, lambda_dssim: float = 0.2) -> Tuple[Tensor, Dict[str, Tensor]]:
    """3DGS photometric loss: ``(1-lambda)*L1 + lambda*(1-SSIM)/2``."""
    l1 = l1_loss(pred, gt)
    ssim_value = ssim(pred, gt)
    dssim = (1.0 - ssim_value) * 0.5
    total = (1.0 - lambda_dssim) * l1 + lambda_dssim * dssim
    return total, {"l1": l1, "ssim": ssim_value, "dssim": dssim, "total": total}


def _gaussian_1d(size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> Tensor:
    coords = torch.arange(size, device=device, dtype=dtype) - size // 2
    kernel = torch.exp(-(coords.square()) / (2 * sigma * sigma))
    return kernel / kernel.sum()

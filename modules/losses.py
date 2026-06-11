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


def medium_decorrelation_loss(rgb_medium: Tensor, rgb_object: Tensor, eps: float = 1.0e-6) -> Tensor:
    """Penalize medium edges that align with object texture edges."""
    medium_grad = image_gradient_magnitude(rgb_medium)
    object_grad = image_gradient_magnitude(rgb_object.detach())
    return torch.abs(_correlation(medium_grad.reshape(-1), object_grad.reshape(-1), eps=eps))


def underwater_loss(
    pred: Tensor,
    gt: Tensor,
    rgb_object: Tensor,
    rgb_medium: Tensor,
    medium_density_l1: Tensor,
    medium_density_smooth: Tensor,
    lambda_dssim: float = 0.2,
    lambda_medium: float = 1.0e-3,
    lambda_beta: float = 1.0e-2,
    lambda_decor: float = 1.0e-2,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    """RGB reconstruction plus medium regularization and decorrelation."""
    rgb_loss, parts = photometric_loss(pred, gt, lambda_dssim=lambda_dssim)
    medium_reg = medium_density_smooth + float(lambda_beta) * medium_density_l1
    decor = medium_decorrelation_loss(rgb_medium, rgb_object) if float(lambda_decor) > 0.0 else pred.new_zeros(())
    total = rgb_loss + float(lambda_medium) * medium_reg + float(lambda_decor) * decor
    parts.update(
        {
            "rgb": rgb_loss,
            "medium_density_l1": medium_density_l1,
            "medium_density_smooth": medium_density_smooth,
            "medium": medium_reg,
            "decor": decor,
            "total": total,
        }
    )
    return total, parts


def image_gradient_magnitude(image: Tensor) -> Tensor:
    """Return an HW gradient magnitude map from a CHW or BCHW image."""
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if image.ndim != 4:
        raise ValueError("image must be CHW or BCHW")
    dx = torch.abs(image[..., :, 1:] - image[..., :, :-1])
    dy = torch.abs(image[..., 1:, :] - image[..., :-1, :])
    dx = F.pad(dx, (0, 1, 0, 0))
    dy = F.pad(dy, (0, 0, 0, 1))
    return (dx + dy).mean(dim=1)


def _gaussian_1d(size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> Tensor:
    coords = torch.arange(size, device=device, dtype=dtype) - size // 2
    kernel = torch.exp(-(coords.square()) / (2 * sigma * sigma))
    return kernel / kernel.sum()


def _correlation(a: Tensor, b: Tensor, eps: float = 1.0e-6) -> Tensor:
    if a.numel() != b.numel():
        raise ValueError("correlation inputs must have the same number of elements")
    a_centered = a - a.mean()
    b_centered = b - b.mean()
    denom = torch.sqrt(a_centered.square().sum()).clamp_min(eps) * torch.sqrt(b_centered.square().sum()).clamp_min(eps)
    return (a_centered * b_centered).sum() / denom

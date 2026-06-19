from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from methods.leo_3DGS.functions.metrics import ssim


@dataclass(frozen=True)
class Raw3DGSLossResult:
    loss: torch.Tensor
    l1: torch.Tensor
    ssim: torch.Tensor
    dssim: torch.Tensor


def raw_3dgs_loss(
    rendered: torch.Tensor,
    target: torch.Tensor,
    lambda_dssim: float = 0.2,
) -> Raw3DGSLossResult:
    """
    原版 3DGS 风格的图像重建 loss。

    原版主项为 L1 和 DSSIM 的加权和：

    loss = (1 - lambda_dssim) * L1 + lambda_dssim * (1 - SSIM)

    Args:
        rendered: 渲染结果，shape 通常为 [3, H, W] 或 [B, 3, H, W]，范围 [0, 1]。
        target: GT 图像，shape 与 rendered 一致，范围 [0, 1]。
        lambda_dssim: DSSIM 权重，原版默认常用 0.2。

    Returns:
        Raw3DGSLossResult，包含总 loss 和 l1 / ssim / dssim 分项。
    """
    if not 0.0 <= lambda_dssim <= 1.0:
        raise ValueError(f"lambda_dssim must be in [0, 1], got {lambda_dssim}")

    if rendered.shape != target.shape:
        raise ValueError(f"rendered shape {tuple(rendered.shape)} != target shape {tuple(target.shape)}")

    l1_value = F.l1_loss(rendered, target)
    ssim_value = ssim(rendered, target, clamp=True, size_average=True)
    dssim_value = 1.0 - ssim_value
    loss = (1.0 - lambda_dssim) * l1_value + lambda_dssim * dssim_value

    return Raw3DGSLossResult(
        loss=loss,
        l1=l1_value,
        ssim=ssim_value,
        dssim=dssim_value,
    )

"""Differentiable LPIPS loss for the local 3DGS training entrypoint."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class LPIPSLoss(nn.Module):
    """Train-time LPIPS loss with gradients flowing back to the rendered image."""

    def __init__(self, net: str = "vgg") -> None:
        super().__init__()
        if net not in {"alex", "vgg", "squeeze"}:
            raise ValueError("LPIPS net must be one of: alex, vgg, squeeze")
        try:
            import lpips
        except ImportError as exc:
            raise ImportError("训练 LPIPS loss 需要安装 lpips：pip install lpips") from exc

        self.net = str(net)
        self.model = lpips.LPIPS(net=self.net).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def forward(self, rendered: Tensor, target: Tensor) -> Tensor:
        rendered_bchw = _to_bchw(rendered).float().clamp(0.0, 1.0)
        target_bchw = _to_bchw(target).to(device=rendered_bchw.device, dtype=rendered_bchw.dtype).clamp(0.0, 1.0)
        if rendered_bchw.shape != target_bchw.shape:
            raise ValueError(f"rendered shape {tuple(rendered_bchw.shape)} != target shape {tuple(target_bchw.shape)}")

        rendered_lpips = rendered_bchw * 2.0 - 1.0
        target_lpips = target_bchw * 2.0 - 1.0
        value = self.model(rendered_lpips, target_lpips)
        return value.reshape(value.shape[0], -1).mean()


def _to_bchw(image: Tensor) -> Tensor:
    if image.ndim == 3 and image.shape[0] == 3:
        return image.unsqueeze(0).contiguous()
    if image.ndim == 4 and image.shape[1] == 3:
        return image.contiguous()
    raise ValueError(f"image must have shape [3,H,W] or [B,3,H,W], got {tuple(image.shape)}")

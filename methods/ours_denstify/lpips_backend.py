"""LPIPS evaluator factory for the ours_denstify entrypoints."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

from modules import build_lpips_evaluator as build_shared_lpips_evaluator


LPIPS_BACKEND_CHOICES = ("lpips", "official_3dgs", "seasplat")
TRAIN_LPIPS_BACKEND_CHOICES = ("seasplat",)


def build_ours_lpips_evaluator(device: torch.device, net: str, backend: str):
    """Build the LPIPS evaluator selected by the ours_denstify CLI contract."""
    if backend == "seasplat":
        return SeasplatLPIPSEvaluator(device, net)
    return build_shared_lpips_evaluator(device, net, backend)


def build_ours_train_lpips_loss(device: torch.device, net: str, backend: str, max_size: int):
    """Build a differentiable LPIPS loss for the ours_denstify training path."""
    if backend != "seasplat":
        raise ValueError("train LPIPS backend currently supports only seasplat")
    return SeasplatLPIPSLoss(device, net, max_size)


class SeasplatLPIPSEvaluator:
    """Adapter around the SeaSplat vendored lpipsPyTorch implementation."""

    def __init__(self, device: torch.device, net: str) -> None:
        lpips_cls = _load_seasplat_lpips_class(net)
        self.model = lpips_cls(net_type=net).to(device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False

    @torch.no_grad()
    def __call__(self, image: Tensor, gt: Tensor) -> float:
        image_bchw = image.unsqueeze(0).clamp(0.0, 1.0)
        gt_bchw = gt.unsqueeze(0).clamp(0.0, 1.0)
        return float(self.model(image_bchw, gt_bchw).detach().reshape(-1)[0])


class SeasplatLPIPSLoss(torch.nn.Module):
    """Differentiable SeaSplat LPIPS loss used during training."""

    def __init__(self, device: torch.device, net: str, max_size: int) -> None:
        super().__init__()
        lpips_cls = _load_seasplat_lpips_class(net)
        self.model = lpips_cls(net_type=net).to(device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        self.max_size = int(max_size)

    def forward(self, image: Tensor, gt: Tensor) -> Tensor:
        if image.shape != gt.shape:
            raise ValueError("image and gt must have the same CHW shape")
        if image.ndim != 3 or int(image.shape[0]) != 3:
            raise ValueError("image and gt must be RGB CHW tensors")
        image_bchw = image.unsqueeze(0).clamp(0.0, 1.0)
        gt_bchw = gt.detach().unsqueeze(0).clamp(0.0, 1.0)
        image_bchw, gt_bchw = _resize_pair_to_max_size(image_bchw, gt_bchw, self.max_size)
        return self.model(image_bchw, gt_bchw).mean()


def _load_seasplat_lpips_class(net: str):
    if net not in {"alex", "vgg", "squeeze"}:
        raise ValueError("seasplat LPIPS net must be alex, vgg, or squeeze")
    seasplat_dir = Path(__file__).resolve().parents[1] / "seasplat"
    if str(seasplat_dir) not in sys.path:
        sys.path.insert(0, str(seasplat_dir))
    try:
        from lpipsPyTorch.modules.lpips import LPIPS
    except ImportError as exc:
        raise ImportError(f"无法导入 SeaSplat LPIPS: {seasplat_dir / 'lpipsPyTorch'}") from exc
    return LPIPS


def _resize_pair_to_max_size(image_bchw: Tensor, gt_bchw: Tensor, max_size: int) -> tuple[Tensor, Tensor]:
    if max_size <= 0:
        return image_bchw, gt_bchw
    height, width = int(image_bchw.shape[-2]), int(image_bchw.shape[-1])
    longest = max(height, width)
    if longest <= max_size:
        return image_bchw, gt_bchw
    scale = float(max_size) / float(longest)
    size = (max(int(round(height * scale)), 1), max(int(round(width * scale)), 1))
    return (
        F.interpolate(image_bchw, size=size, mode="bilinear", align_corners=False),
        F.interpolate(gt_bchw, size=size, mode="bilinear", align_corners=False),
    )

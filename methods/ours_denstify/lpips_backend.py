"""LPIPS evaluator factory for the ours_denstify entrypoints."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import Tensor

from modules import build_lpips_evaluator as build_shared_lpips_evaluator


LPIPS_BACKEND_CHOICES = ("lpips", "official_3dgs", "seasplat")


def build_ours_lpips_evaluator(device: torch.device, net: str, backend: str):
    """Build the LPIPS evaluator selected by the ours_denstify CLI contract."""
    if backend == "seasplat":
        return SeasplatLPIPSEvaluator(device, net)
    return build_shared_lpips_evaluator(device, net, backend)


class SeasplatLPIPSEvaluator:
    """Adapter around the SeaSplat vendored lpipsPyTorch implementation."""

    def __init__(self, device: torch.device, net: str) -> None:
        if net not in {"alex", "vgg", "squeeze"}:
            raise ValueError("seasplat LPIPS net must be alex, vgg, or squeeze")
        seasplat_dir = Path(__file__).resolve().parents[1] / "seasplat"
        if str(seasplat_dir) not in sys.path:
            sys.path.insert(0, str(seasplat_dir))
        try:
            from lpipsPyTorch.modules.lpips import LPIPS
        except ImportError as exc:
            raise ImportError(f"无法导入 SeaSplat LPIPS: {seasplat_dir / 'lpipsPyTorch'}") from exc
        self.model = LPIPS(net_type=net).to(device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False

    @torch.no_grad()
    def __call__(self, image: Tensor, gt: Tensor) -> float:
        image_bchw = image.unsqueeze(0).clamp(0.0, 1.0)
        gt_bchw = gt.unsqueeze(0).clamp(0.0, 1.0)
        return float(self.model(image_bchw, gt_bchw).detach().reshape(-1)[0])

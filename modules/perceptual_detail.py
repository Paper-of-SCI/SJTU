"""Perceptual patch-detail scoring for patch-guided densification."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor
from torchvision.models import VGG16_Weights, vgg16


class VGG16PerceptualPatchScorer(torch.nn.Module):
    """Frozen VGG16 relu3_3 residual mapped to the densification patch grid."""

    def __init__(self, eps: float = 1.0e-6) -> None:
        super().__init__()
        model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        self.features = model.features[:16].eval()
        for parameter in self.features.parameters():
            parameter.requires_grad_(False)
        self.eps = float(eps)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

    @torch.no_grad()
    def forward(self, render_image: Tensor, gt_image: Tensor, *, patch_size: int, max_size: int = 768) -> Tensor:
        return self.score(render_image, gt_image, patch_size=patch_size, max_size=max_size)

    @torch.no_grad()
    def score(self, render_image: Tensor, gt_image: Tensor, *, patch_size: int, max_size: int = 768) -> Tensor:
        """Return a normalized ``(patch_h, patch_w)`` VGG residual map."""
        if render_image.shape != gt_image.shape:
            raise ValueError("render_image and gt_image must have the same CHW shape")
        if render_image.ndim != 3:
            raise ValueError("render_image and gt_image must be CHW tensors")
        if int(render_image.shape[0]) != 3:
            raise ValueError("render_image and gt_image must have 3 RGB channels")

        height, width = int(gt_image.shape[-2]), int(gt_image.shape[-1])
        patch = max(int(patch_size), 1)
        patch_h = max(int(math.ceil(height / patch)), 1)
        patch_w = max(int(math.ceil(width / patch)), 1)

        image_pair = torch.stack(
            [
                render_image.detach().clamp(0.0, 1.0),
                gt_image.detach().clamp(0.0, 1.0),
            ],
            dim=0,
        ).float()
        image_pair = _resize_pair_to_max_size(image_pair, int(max_size))
        image_pair = (image_pair - self.mean.to(device=image_pair.device, dtype=image_pair.dtype)) / self.std.to(
            device=image_pair.device,
            dtype=image_pair.dtype,
        )

        features = self.features(image_pair)
        render_features = F.normalize(features[0:1], p=2, dim=1, eps=self.eps)
        gt_features = F.normalize(features[1:2], p=2, dim=1, eps=self.eps)
        residual = (render_features - gt_features).square().mean(dim=1, keepdim=True)
        patch_residual = F.interpolate(residual, size=(patch_h, patch_w), mode="bilinear", align_corners=False)
        return _normalize_minmax(patch_residual.squeeze(0).squeeze(0), self.eps).detach()


def _resize_pair_to_max_size(image_pair: Tensor, max_size: int) -> Tensor:
    if max_size <= 0:
        return image_pair
    height, width = int(image_pair.shape[-2]), int(image_pair.shape[-1])
    longest = max(height, width)
    if longest <= max_size:
        return image_pair
    scale = float(max_size) / float(longest)
    new_height = max(int(round(height * scale)), 1)
    new_width = max(int(round(width * scale)), 1)
    return F.interpolate(image_pair, size=(new_height, new_width), mode="bilinear", align_corners=False)


def _normalize_minmax(values: Tensor, eps: float = 1.0e-6) -> Tensor:
    if values.numel() == 0:
        return values
    min_value = values.amin()
    max_value = values.amax()
    denom = max_value - min_value
    normalized = (values - min_value) / denom.clamp_min(eps)
    return torch.where(denom > eps, normalized, torch.zeros_like(values))

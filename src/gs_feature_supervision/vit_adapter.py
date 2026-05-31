"""Torchvision ViT patch-feature adapter.

The adapter is optional and isolated because it owns model construction.
Training code can also pass precomputed patch features directly to the loss.
"""

from __future__ import annotations

from typing import Literal, Optional

import torch
from torch import nn
from torchvision import models


VitName = Literal["vit_b_16", "vit_b_32", "vit_l_16", "vit_l_32"]


class TorchvisionVitPatchFeatureExtractor(nn.Module):
    """Extract patch tokens from a torchvision VisionTransformer."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @classmethod
    def create(cls, name: VitName = "vit_b_16", *, pretrained: bool = False) -> "TorchvisionVitPatchFeatureExtractor":
        weights = None
        if pretrained:
            if name == "vit_b_16":
                weights = models.ViT_B_16_Weights.IMAGENET1K_V1
            elif name == "vit_b_32":
                weights = models.ViT_B_32_Weights.IMAGENET1K_V1
            elif name == "vit_l_16":
                weights = models.ViT_L_16_Weights.IMAGENET1K_V1
            elif name == "vit_l_32":
                weights = models.ViT_L_32_Weights.IMAGENET1K_V1

        factory = getattr(models, name)
        return cls(factory(weights=weights))

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return patch tokens as [B, H_p, W_p, D]."""

        tokens = self.model._process_input(images)
        batch_size, patch_count, _ = tokens.shape
        class_token = self.model.class_token.expand(batch_size, -1, -1)
        encoded = self.model.encoder(torch.cat((class_token, tokens), dim=1))
        patch_tokens = encoded[:, 1:, :]
        patch_side = int(patch_count**0.5)
        if patch_side * patch_side != patch_count:
            raise ValueError("Only square ViT patch grids are supported")
        return patch_tokens.reshape(batch_size, patch_side, patch_side, -1)

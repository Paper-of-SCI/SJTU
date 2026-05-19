"""Frozen ViT patch memory for scene-conditioned MLP-GS."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from modules import Camera


@dataclass(frozen=True)
class ViTPatchMemory:
    """Detached patch-token memory extracted from source views."""

    tokens: Tensor
    grid_height: int
    grid_width: int
    image_size: int
    backbone_name: str
    weights_name: str

    @property
    def num_views(self) -> int:
        return int(self.tokens.shape[0])

    @property
    def num_patches(self) -> int:
        return int(self.tokens.shape[1])

    @property
    def token_dim(self) -> int:
        return int(self.tokens.shape[2])

    def metadata(self) -> dict[str, object]:
        return {
            "num_views": self.num_views,
            "num_patches": self.num_patches,
            "token_dim": self.token_dim,
            "grid_height": self.grid_height,
            "grid_width": self.grid_width,
            "image_size": self.image_size,
            "backbone_name": self.backbone_name,
            "weights_name": self.weights_name,
        }


class FrozenViTPatchExtractor(nn.Module):
    """Extract ViT-B/16 encoded patch tokens with a frozen torchvision model."""

    def __init__(self, weights: str = "DEFAULT", image_size: int = 224, device: torch.device | str = "cuda") -> None:
        super().__init__()
        try:
            from torchvision.models import ViT_B_16_Weights, vit_b_16
        except ImportError as exc:
            raise ImportError("ViT patch memory 需要 torchvision；请先安装 torchvision。") from exc

        weight_arg = None
        weights_name = "none"
        if weights.lower() != "none":
            weight_enum = ViT_B_16_Weights.DEFAULT if weights.upper() == "DEFAULT" else ViT_B_16_Weights[weights.upper()]
            weight_arg = weight_enum
            weights_name = weight_enum.name

        self.backbone = vit_b_16(weights=weight_arg).to(device).eval()
        self.backbone.requires_grad_(False)
        self.patch_size = int(self.backbone.patch_size)
        if image_size <= 0 or image_size % self.patch_size != 0:
            raise ValueError(f"vit_image_size must be a positive multiple of patch size {self.patch_size}")
        self.image_size = int(image_size)
        self.grid_height = self.image_size // self.patch_size
        self.grid_width = self.image_size // self.patch_size
        self.weights_name = weights_name
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1))

    @property
    def token_dim(self) -> int:
        return int(self.backbone.hidden_dim)

    @torch.no_grad()
    def build_memory(self, cameras: Sequence[Camera], batch_size: int = 4) -> ViTPatchMemory:
        if len(cameras) == 0:
            raise ValueError("ViT patch memory needs at least one source camera")
        tokens = []
        device = next(self.backbone.parameters()).device
        for start in range(0, len(cameras), batch_size):
            batch = cameras[start : start + batch_size]
            images = []
            for camera in batch:
                if camera.image is None:
                    raise RuntimeError(f"ViT patch memory 需要相机图像: {camera.image_path}")
                images.append(camera.image.detach().to(device=device, dtype=torch.float32))
            image_batch = torch.stack(images, dim=0)
            image_batch = self._preprocess(image_batch)
            tokens.append(self._forward_patch_tokens(image_batch).detach().cpu())
        token_tensor = torch.cat(tokens, dim=0).to(device=device)
        return ViTPatchMemory(
            tokens=token_tensor,
            grid_height=self.grid_height,
            grid_width=self.grid_width,
            image_size=self.image_size,
            backbone_name="torchvision.vit_b_16",
            weights_name=self.weights_name,
        )

    def _preprocess(self, images: Tensor) -> Tensor:
        images = F.interpolate(images, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        return (images - self.mean.to(images.device)) / self.std.to(images.device)

    def _forward_patch_tokens(self, images: Tensor) -> Tensor:
        patches = self._patchify(images)
        batch_class_token = self.backbone.class_token.expand(images.shape[0], -1, -1)
        encoded = self._forward_encoder(torch.cat([batch_class_token, patches], dim=1), self.grid_height, self.grid_width)
        return encoded[:, 1:, :]

    def _patchify(self, images: Tensor) -> Tensor:
        patches = self.backbone.conv_proj(images)
        batch, hidden_dim, grid_height, grid_width = patches.shape
        patches = patches.reshape(batch, hidden_dim, grid_height * grid_width)
        return patches.permute(0, 2, 1)

    def _forward_encoder(self, tokens: Tensor, grid_height: int, grid_width: int) -> Tensor:
        encoder = self.backbone.encoder
        tokens = tokens + self._interpolated_position_embedding(grid_height, grid_width, tokens.device, tokens.dtype)
        return encoder.ln(encoder.layers(encoder.dropout(tokens)))

    def _interpolated_position_embedding(
        self,
        grid_height: int,
        grid_width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        pos = self.backbone.encoder.pos_embedding.to(device=device, dtype=dtype)
        cls_pos = pos[:, :1]
        patch_pos = pos[:, 1:]
        old_grid = int(patch_pos.shape[1] ** 0.5)
        if old_grid * old_grid != patch_pos.shape[1]:
            raise ValueError("ViT position embedding patch count is not square")
        if old_grid == grid_height and old_grid == grid_width:
            return pos
        patch_pos = patch_pos.reshape(1, old_grid, old_grid, -1).permute(0, 3, 1, 2)
        patch_pos = F.interpolate(patch_pos, size=(grid_height, grid_width), mode="bicubic", align_corners=False)
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, grid_height * grid_width, -1)
        return torch.cat([cls_pos, patch_pos], dim=1)

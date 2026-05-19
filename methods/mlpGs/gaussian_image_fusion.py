"""Multi-view image-token fusion for MLP-GS anchors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
from torch import Tensor

from modules import Camera
from vit_patch_memory import ViTPatchMemory


@dataclass(frozen=True)
class FusionCameraBatch:
    """Tensorized camera fields used for Gaussian-to-patch lookup."""

    viewmats: Tensor
    fx: Tensor
    fy: Tensor
    cx: Tensor
    cy: Tensor
    widths: Tensor
    heights: Tensor
    near: Tensor
    far: Tensor

    @classmethod
    def from_cameras(cls, cameras: Sequence[Camera], device: torch.device | str) -> "FusionCameraBatch":
        if len(cameras) == 0:
            raise ValueError("FusionCameraBatch needs at least one camera")
        return cls(
            viewmats=torch.stack([camera.viewmat.detach().to(device=device, dtype=torch.float32) for camera in cameras], dim=0),
            fx=torch.tensor([camera.fx for camera in cameras], dtype=torch.float32, device=device),
            fy=torch.tensor([camera.fy for camera in cameras], dtype=torch.float32, device=device),
            cx=torch.tensor([camera.cx for camera in cameras], dtype=torch.float32, device=device),
            cy=torch.tensor([camera.cy for camera in cameras], dtype=torch.float32, device=device),
            widths=torch.tensor([camera.width for camera in cameras], dtype=torch.float32, device=device),
            heights=torch.tensor([camera.height for camera in cameras], dtype=torch.float32, device=device),
            near=torch.tensor([camera.near for camera in cameras], dtype=torch.float32, device=device),
            far=torch.tensor([camera.far for camera in cameras], dtype=torch.float32, device=device),
        )

    @property
    def num_views(self) -> int:
        return int(self.viewmats.shape[0])


class GaussianImageFusion(nn.Module):
    """Fuse learnable Gaussian features with local multi-view ViT patch tokens."""

    def __init__(
        self,
        anchor_feature_dim: int,
        vit_token_dim: int,
        fusion_dim: int = 64,
        num_heads: int = 4,
        topk_views: int = 4,
        patch_window: int = 1,
    ) -> None:
        super().__init__()
        if anchor_feature_dim <= 0:
            raise ValueError("ViT fusion requires feature_dim > 0")
        if fusion_dim <= 0:
            raise ValueError("fusion_dim must be positive")
        if num_heads <= 0:
            raise ValueError("fusion_heads must be positive")
        if fusion_dim % num_heads != 0:
            raise ValueError("fusion_dim must be divisible by fusion_heads")
        if topk_views <= 0:
            raise ValueError("vit_topk_views must be positive")
        if patch_window < 0:
            raise ValueError("vit_patch_window must be >= 0")

        self.anchor_feature_dim = int(anchor_feature_dim)
        self.vit_token_dim = int(vit_token_dim)
        self.fusion_dim = int(fusion_dim)
        self.num_heads = int(num_heads)
        self.topk_views = int(topk_views)
        self.patch_window = int(patch_window)

        self.query_proj = nn.Sequential(
            nn.Linear(3 + self.anchor_feature_dim, self.fusion_dim),
            nn.SiLU(),
            nn.Linear(self.fusion_dim, self.fusion_dim),
        )
        self.token_adapter = nn.Sequential(
            nn.Linear(self.vit_token_dim, self.fusion_dim),
            nn.SiLU(),
            nn.Linear(self.fusion_dim, self.fusion_dim),
        )
        self.cross_attention = nn.MultiheadAttention(self.fusion_dim, self.num_heads, batch_first=True)
        self.output = nn.Linear(self.fusion_dim, self.anchor_feature_dim)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

        offsets = [(dy, dx) for dy in range(-self.patch_window, self.patch_window + 1) for dx in range(-self.patch_window, self.patch_window + 1)]
        self.register_buffer("patch_offsets", torch.tensor(offsets, dtype=torch.long))

    def forward(
        self,
        anchor_xyz: Tensor,
        normalized_anchor_xyz: Tensor,
        anchor_features: Tensor,
        cameras: FusionCameraBatch,
        memory: ViTPatchMemory,
        chunk_size: int = 4096,
    ) -> Tensor:
        if anchor_xyz.shape[0] != anchor_features.shape[0]:
            raise ValueError("anchor_xyz and anchor_features must have the same first dimension")
        if memory.num_views != cameras.num_views:
            raise ValueError("ViT patch memory view count must match FusionCameraBatch")
        if chunk_size <= 0:
            raise ValueError("fusion_chunk_size must be positive")

        fused_chunks = []
        for start in range(0, anchor_xyz.shape[0], chunk_size):
            end = min(start + chunk_size, anchor_xyz.shape[0])
            fused_chunks.append(
                self._forward_chunk(
                    anchor_xyz[start:end],
                    normalized_anchor_xyz[start:end],
                    anchor_features[start:end],
                    cameras,
                    memory,
                )
            )
        return torch.cat(fused_chunks, dim=0)

    def _forward_chunk(
        self,
        anchor_xyz: Tensor,
        normalized_anchor_xyz: Tensor,
        anchor_features: Tensor,
        cameras: FusionCameraBatch,
        memory: ViTPatchMemory,
    ) -> Tensor:
        topk_indices, patch_x, patch_y, visible = project_gaussians_to_patch_grid(anchor_xyz.detach(), cameras, memory, self.topk_views)
        token_context, token_valid = gather_local_patch_tokens(memory, topk_indices, patch_x, patch_y, visible, self.patch_offsets)

        batch, token_count, token_dim = token_context.shape
        has_context = token_valid.any(dim=1)
        safe_valid = token_valid.clone()
        if not bool(has_context.all()):
            missing = ~has_context
            safe_valid[missing, 0] = True
            token_context[missing, 0] = 0.0

        query_input = torch.cat([normalized_anchor_xyz, anchor_features], dim=-1)
        query = self.query_proj(query_input).unsqueeze(1)
        key_value = self.token_adapter(token_context.reshape(batch * token_count, token_dim)).reshape(batch, token_count, self.fusion_dim)
        attended, _ = self.cross_attention(query, key_value, key_value, key_padding_mask=~safe_valid, need_weights=False)
        delta = self.output(attended.squeeze(1)) * has_context.to(dtype=anchor_features.dtype).unsqueeze(-1)
        return anchor_features + delta


def project_gaussians_to_patch_grid(
    anchor_xyz: Tensor,
    cameras: FusionCameraBatch,
    memory: ViTPatchMemory,
    topk_views: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Project anchors into source cameras and choose nearest visible views."""
    device = anchor_xyz.device
    dtype = anchor_xyz.dtype
    ones = torch.ones(anchor_xyz.shape[0], 1, device=device, dtype=dtype)
    points_h = torch.cat([anchor_xyz, ones], dim=-1)
    cam = torch.einsum("vij,nj->vni", cameras.viewmats.to(device=device, dtype=dtype), points_h)
    x = cam[..., 0]
    y = cam[..., 1]
    z = cam[..., 2]

    fx = cameras.fx.to(device=device, dtype=dtype)[:, None]
    fy = cameras.fy.to(device=device, dtype=dtype)[:, None]
    cx = cameras.cx.to(device=device, dtype=dtype)[:, None]
    cy = cameras.cy.to(device=device, dtype=dtype)[:, None]
    widths = cameras.widths.to(device=device, dtype=dtype)[:, None]
    heights = cameras.heights.to(device=device, dtype=dtype)[:, None]
    near = cameras.near.to(device=device, dtype=dtype)[:, None]
    far = cameras.far.to(device=device, dtype=dtype)[:, None]

    z_safe = z.clamp_min(1.0e-6)
    u = fx * (x / z_safe) + cx
    v = fy * (y / z_safe) + cy
    visible = (z > near) & (z < far) & (u >= 0.0) & (u < widths) & (v >= 0.0) & (v < heights)

    depth = z.transpose(0, 1).masked_fill(~visible.transpose(0, 1), float("inf"))
    k = min(int(topk_views), cameras.num_views)
    topk_depth, topk_indices = torch.topk(depth, k=k, dim=1, largest=False)
    topk_visible = torch.isfinite(topk_depth)

    u_selected = u.transpose(0, 1).gather(1, topk_indices)
    v_selected = v.transpose(0, 1).gather(1, topk_indices)
    widths_selected = cameras.widths.to(device=device, dtype=dtype)[topk_indices]
    heights_selected = cameras.heights.to(device=device, dtype=dtype)[topk_indices]

    patch_x = torch.floor(u_selected / widths_selected.clamp_min(1.0) * memory.grid_width).to(torch.long)
    patch_y = torch.floor(v_selected / heights_selected.clamp_min(1.0) * memory.grid_height).to(torch.long)
    return topk_indices, patch_x, patch_y, topk_visible


def gather_local_patch_tokens(
    memory: ViTPatchMemory,
    view_indices: Tensor,
    patch_x: Tensor,
    patch_y: Tensor,
    view_visible: Tensor,
    offsets: Tensor,
) -> tuple[Tensor, Tensor]:
    """Gather local patch windows for each Gaussian/top-k view pair."""
    tokens = memory.tokens.to(device=view_indices.device)
    offsets = offsets.to(device=view_indices.device)
    offset_y = offsets[:, 0]
    offset_x = offsets[:, 1]

    x = patch_x[..., None] + offset_x
    y = patch_y[..., None] + offset_y
    patch_valid = view_visible[..., None] & (x >= 0) & (x < memory.grid_width) & (y >= 0) & (y < memory.grid_height)
    x_safe = x.clamp(0, memory.grid_width - 1)
    y_safe = y.clamp(0, memory.grid_height - 1)
    patch_indices = y_safe * memory.grid_width + x_safe

    gathered = tokens[view_indices[..., None].expand_as(patch_indices), patch_indices]
    batch = view_indices.shape[0]
    return gathered.reshape(batch, -1, tokens.shape[-1]), patch_valid.reshape(batch, -1)

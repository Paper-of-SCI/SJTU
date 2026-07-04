"""Local Gaussian Context Attention.

We introduce a Local Gaussian Context Attention (LGCA) module to aggregate
local Gaussian features with self-attention.
"""

from __future__ import annotations

import math

import torch.nn as nn
from torch import Tensor


class LocalGaussianContextAttention(nn.Module):
    """Local self-attention for Gaussian features.

    Args:
        feature_dim: Dimension of each Gaussian feature.
        num_heads: Number of attention heads.
        ffn_mult: Hidden-width multiplier for the feed-forward block.

    Input:
        features: [num_gaussians, feature_dim]
        local_features: [num_gaussians, num_neighbors, feature_dim]

    Output:
        context_features: [num_gaussians, feature_dim]
    """

    def __init__(
        self,
        feature_dim: int = 32,
        num_heads: int = 4,
        ffn_mult: int = 2,
    ) -> None:
        super().__init__()

        if feature_dim <= 0:
            raise ValueError("feature_dim must be > 0")
        if num_heads <= 0:
            raise ValueError("num_heads must be > 0")
        if feature_dim % num_heads != 0:
            raise ValueError("feature_dim must be divisible by num_heads")
        if ffn_mult <= 0:
            raise ValueError("ffn_mult must be > 0")

        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(feature_dim, feature_dim)
        self.k_proj = nn.Linear(feature_dim, feature_dim)
        self.v_proj = nn.Linear(feature_dim, feature_dim)
        self.out_proj = nn.Linear(feature_dim, feature_dim)

        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)

        hidden_dim = feature_dim * ffn_mult
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, feature_dim),
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        modules = [self.q_proj, self.k_proj, self.v_proj, self.out_proj]
        for module in modules:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

        for module in self.ffn:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, features: Tensor, local_features: Tensor) -> Tensor:
        if features.ndim != 2:
            raise ValueError(
                f"features must have shape [N, D], got {tuple(features.shape)}"
            )
        if local_features.ndim != 3:
            raise ValueError(
                f"local_features must have shape [N, K, D], "
                f"got {tuple(local_features.shape)}"
            )

        n, d = features.shape
        if d != self.feature_dim:
            raise ValueError(f"Expected feature_dim={self.feature_dim}, got {d}")
        if local_features.shape[0] != n:
            raise ValueError(
                f"local_features.shape[0] must match features.shape[0], "
                f"got {local_features.shape[0]} and {n}"
            )
        if local_features.shape[-1] != d:
            raise ValueError(
                f"local_features.shape[-1] must match feature_dim={d}, "
                f"got {local_features.shape[-1]}"
            )
        if local_features.device != features.device:
            local_features = local_features.to(features.device)

        k_neighbors = local_features.shape[1]

        q = self.q_proj(features).view(n, self.num_heads, self.head_dim)
        k = self.k_proj(local_features).view(
            n, k_neighbors, self.num_heads, self.head_dim
        )
        v = self.v_proj(local_features).view(
            n, k_neighbors, self.num_heads, self.head_dim
        )

        k = k.permute(0, 2, 1, 3)  # [N, H, K, C]
        v = v.permute(0, 2, 1, 3)  # [N, H, K, C]

        attn_logits = (q[:, :, None, :] * k).sum(dim=-1) * self.scale  # [N, H, K]
        attn = attn_logits.softmax(dim=-1)

        context = (attn[..., None] * v).sum(dim=2)  # [N, H, C]
        context = context.reshape(n, d)  # [N, D]

        features = self.norm1(features + self.out_proj(context))
        features = self.norm2(features + self.ffn(features))
        return features

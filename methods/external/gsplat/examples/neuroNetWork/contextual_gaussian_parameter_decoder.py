"""Contextual Gaussian Parameter Decoder.

We introduce a Contextual Gaussian Parameter Decoder (CGPD) to predict
Gaussian parameter offsets from local context-aware Gaussian features.
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor


class ContextualGaussianParameterDecoder(nn.Module):
    """Decode context-aware Gaussian features into parameter offsets.

    Args:
        feature_dim: Dimension of each input context feature.
        hidden_dim: Width of hidden MLP layers.
        num_layers: Number of hidden Linear + activation blocks.
        sh_degree: Maximum spherical harmonics degree.
        color_channels: Number of color channels per SH coefficient.
        delta_scale: Global multiplier applied to all predicted deltas.
        zero_init_output: Whether to initialize the output layer as zero.

    Input:
        context_features: [num_gaussians, feature_dim]

    Output:
        A delta dictionary with keys matching Gaussian parameters:
            means: [num_gaussians, 3]
            quats: [num_gaussians, 4]
            scales: [num_gaussians, 3]
            opacities: [num_gaussians]
            sh0: [num_gaussians, 1, color_channels]
            shN: [num_gaussians, (sh_degree + 1)^2 - 1, color_channels]
    """

    def __init__(
        self,
        feature_dim: int = 32,
        hidden_dim: int = 64,
        num_layers: int = 3,
        sh_degree: int = 3,
        color_channels: int = 3,
        delta_scale: float = 1.0,
        zero_init_output: bool = True,
    ) -> None:
        super().__init__()

        if feature_dim <= 0:
            raise ValueError("feature_dim must be > 0")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be > 0")
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if sh_degree < 0:
            raise ValueError("sh_degree must be >= 0")
        if color_channels <= 0:
            raise ValueError("color_channels must be > 0")

        self.feature_dim = feature_dim
        self.sh_degree = sh_degree
        self.color_channels = color_channels
        self.delta_scale = delta_scale
        self.zero_init_output = zero_init_output

        self.num_sh_coeffs = (sh_degree + 1) ** 2
        self.num_shN_coeffs = self.num_sh_coeffs - 1
        self.color_delta_dim = self.num_sh_coeffs * color_channels

        self.means_dim = 3
        self.quats_dim = 4
        self.scales_dim = 3
        self.opacities_dim = 1
        self.output_dim = (
            self.means_dim
            + self.quats_dim
            + self.scales_dim
            + self.opacities_dim
            + self.color_delta_dim
        )

        layers = []
        layers.append(nn.Linear(feature_dim, hidden_dim))
        layers.append(nn.SiLU(inplace=True))

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU(inplace=True))

        layers.append(nn.Linear(hidden_dim, self.output_dim))

        self.net = nn.Sequential(*layers)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.net[:-1]:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

        final = self.net[-1]
        if isinstance(final, nn.Linear):
            if self.zero_init_output:
                nn.init.zeros_(final.weight)
            else:
                nn.init.xavier_uniform_(final.weight, gain=0.01)
            nn.init.zeros_(final.bias)

    def forward(self, context_features: Tensor) -> dict[str, Tensor]:
        if context_features.ndim != 2:
            raise ValueError(
                f"context_features must have shape [N, D], "
                f"got {tuple(context_features.shape)}"
            )
        if context_features.shape[-1] != self.feature_dim:
            raise ValueError(
                f"Expected context_features.shape[-1] == {self.feature_dim}, "
                f"but got {context_features.shape[-1]}"
            )

        n = context_features.shape[0]
        delta = self.net(context_features) * self.delta_scale

        start = 0
        end = start + self.means_dim
        delta_means = delta[:, start:end]

        start = end
        end = start + self.quats_dim
        delta_quats = delta[:, start:end]

        start = end
        end = start + self.scales_dim
        delta_scales = delta[:, start:end]

        start = end
        end = start + self.opacities_dim
        delta_opacities = delta[:, start:end].squeeze(-1)

        start = end
        end = start + self.color_channels
        delta_sh0 = delta[:, start:end].view(n, 1, self.color_channels)

        start = end
        delta_shN = delta[:, start:].view(n, self.num_shN_coeffs, self.color_channels)

        return {
            "means": delta_means,
            "quats": delta_quats,
            "scales": delta_scales,
            "opacities": delta_opacities,
            "sh0": delta_sh0,
            "shN": delta_shN,
        }

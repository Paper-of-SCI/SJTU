"""Gaussian Attribute Encoder.

We introduce a Gaussian Attribute Encoder (GAE) to project the raw attributes
of each Gaussian into a latent feature space.
"""

import torch.nn as nn
from torch import Tensor


class GaussianAttributeEncoder(nn.Module):
    """Encode pre-built Gaussian attributes into latent features.

    Args:
        input_dim: Dimension of the input Gaussian attribute vector.
        feature_dim: Dimension of the output latent feature.
        hidden_dim: Width of hidden MLP layers.
        num_layers: Number of hidden Linear + activation blocks.
        normalize_output: Whether to apply LayerNorm to output features.

    Input:
        attributes: [num_gaussians, input_dim]

    Output:
        features: [num_gaussians, feature_dim]
    """

    def __init__(
        self,
        input_dim: int = 83,
        feature_dim: int = 32,
        hidden_dim: int = 64,
        num_layers: int = 3,
        normalize_output: bool = True,
    ) -> None:
        super().__init__()

        if input_dim <= 0:
            raise ValueError("input_dim must be > 0")
        if feature_dim <= 0:
            raise ValueError("feature_dim must be > 0")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be > 0")
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        self.input_dim = input_dim
        self.feature_dim = feature_dim

        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.SiLU(inplace=True))

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU(inplace=True))

        layers.append(nn.Linear(hidden_dim, feature_dim))

        self.net = nn.Sequential(*layers)
        self.output_norm = (
            nn.LayerNorm(feature_dim) if normalize_output else nn.Identity()
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

        final = self.net[-1]
        if isinstance(final, nn.Linear):
            nn.init.xavier_uniform_(final.weight, gain=0.1)
            nn.init.zeros_(final.bias)

    def forward(self, attributes: Tensor) -> Tensor:
        if attributes.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected attributes.shape[-1] == {self.input_dim}, "
                f"but got {attributes.shape[-1]}"
            )

        features = self.net(attributes)
        features = self.output_norm(features)
        return features

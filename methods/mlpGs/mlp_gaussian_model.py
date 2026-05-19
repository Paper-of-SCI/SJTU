"""MLP-parameterized Gaussian model for the first MLP-GS experiment.

This module keeps the experiment local to ``methods/mlpGs``.  The model uses
fixed Gaussian anchor positions as MLP inputs, then predicts the raw Gaussian
parameters consumed by the existing renderer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
from torch import Tensor

from modules.gaussian_model import GaussianModel, GaussianTensors
from modules.spherical_harmonics import num_sh_bases


@dataclass(frozen=True)
class RawGaussianTensors:
    """Raw tensors before renderer activations."""

    means: Tensor
    log_scales: Tensor
    quats: Tensor
    logit_opacities: Tensor
    features_dc: Tensor
    features_rest: Tensor


class MLPGaussianModel(nn.Module):
    """Predict all Gaussian parameters from fixed Gaussian positions.

    The initial GaussianModel is kept as a residual base.  The final MLP layer
    is initialized to zero, so step 0 matches the standard 3DGS initialization.
    """

    def __init__(
        self,
        anchor_xyz: Tensor,
        base_tensors: RawGaussianTensors,
        sh_degree: int = 3,
        hidden_dim: int = 128,
        hidden_layers: int = 3,
    ) -> None:
        super().__init__()
        if anchor_xyz.ndim != 2 or anchor_xyz.shape[-1] != 3:
            raise ValueError("anchor_xyz must have shape (N, 3)")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if hidden_layers < 1:
            raise ValueError("hidden_layers must be >= 1")

        self.sh_degree = int(sh_degree)
        self.hidden_dim = int(hidden_dim)
        self.hidden_layers = int(hidden_layers)
        self.rest_bases = max(num_sh_bases(self.sh_degree) - 1, 0)

        self.register_buffer("anchor_xyz", anchor_xyz.detach().clone())
        self.register_buffer("input_center", anchor_xyz.detach().mean(dim=0, keepdim=True))
        input_scale = (anchor_xyz.detach() - self.input_center).norm(dim=-1).max().clamp_min(1.0e-6)
        self.register_buffer("input_scale", input_scale)

        self.register_buffer("base_means", base_tensors.means.detach().clone())
        self.register_buffer("base_log_scales", base_tensors.log_scales.detach().clone())
        self.register_buffer("base_quats", base_tensors.quats.detach().clone())
        self.register_buffer("base_logit_opacities", base_tensors.logit_opacities.detach().clone())
        self.register_buffer("base_features_dc", base_tensors.features_dc.detach().clone())
        self.register_buffer("base_features_rest", base_tensors.features_rest.detach().clone())

        self.mlp = self._build_mlp(
            input_dim=3,
            hidden_dim=self.hidden_dim,
            hidden_layers=self.hidden_layers,
            output_dim=self.raw_output_dim,
        )

    @classmethod
    def from_gaussian_model(
        cls,
        base_model: GaussianModel,
        hidden_dim: int = 128,
        hidden_layers: int = 3,
    ) -> "MLPGaussianModel":
        """Create an MLP-GS model from the standard point-cloud initialization."""
        base_tensors = RawGaussianTensors(
            means=base_model.means.detach(),
            log_scales=base_model.log_scales.detach(),
            quats=base_model.quats.detach(),
            logit_opacities=base_model.logit_opacities.detach(),
            features_dc=base_model.features_dc.detach(),
            features_rest=base_model.features_rest.detach(),
        )
        return cls(
            anchor_xyz=base_model.means.detach(),
            base_tensors=base_tensors,
            sh_degree=base_model.sh_degree,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
        )

    @property
    def num_gaussians(self) -> int:
        return int(self.anchor_xyz.shape[0])

    @property
    def raw_output_dim(self) -> int:
        return 3 + 3 + 4 + 1 + 3 + self.rest_bases * 3

    @property
    def means(self) -> Tensor:
        return self.raw_tensors().means

    @property
    def log_scales(self) -> Tensor:
        return self.raw_tensors().log_scales

    @property
    def quats(self) -> Tensor:
        return self.raw_tensors().quats

    @property
    def logit_opacities(self) -> Tensor:
        return self.raw_tensors().logit_opacities

    @property
    def features_dc(self) -> Tensor:
        return self.raw_tensors().features_dc

    @property
    def features_rest(self) -> Tensor:
        return self.raw_tensors().features_rest

    @property
    def scales(self) -> Tensor:
        return self.log_scales.exp()

    @property
    def normalized_quats(self) -> Tensor:
        return torch.nn.functional.normalize(self.quats, dim=-1)

    @property
    def opacities(self) -> Tensor:
        return self.logit_opacities.sigmoid().squeeze(-1)

    @property
    def colors(self) -> Tensor:
        raw = self.raw_tensors()
        return torch.cat([raw.features_dc, raw.features_rest], dim=1)

    def raw_tensors(self) -> RawGaussianTensors:
        """Return the current raw Gaussian tensors predicted by the MLP."""
        deltas = self.mlp(self.normalized_inputs())
        n = self.num_gaussians

        offset = 0
        delta_means = deltas[:, offset : offset + 3]
        offset += 3
        delta_log_scales = deltas[:, offset : offset + 3]
        offset += 3
        delta_quats = deltas[:, offset : offset + 4]
        offset += 4
        delta_logit_opacities = deltas[:, offset : offset + 1]
        offset += 1
        delta_features_dc = deltas[:, offset : offset + 3].reshape(n, 1, 3)
        offset += 3

        if self.rest_bases > 0:
            delta_features_rest = deltas[:, offset:].reshape(n, self.rest_bases, 3)
        else:
            delta_features_rest = self.base_features_rest.new_zeros(n, 0, 3)

        return RawGaussianTensors(
            means=self.base_means + delta_means,
            log_scales=self.base_log_scales + delta_log_scales,
            quats=self.base_quats + delta_quats,
            logit_opacities=self.base_logit_opacities + delta_logit_opacities,
            features_dc=self.base_features_dc + delta_features_dc,
            features_rest=self.base_features_rest + delta_features_rest,
        )

    def activated_tensors(self) -> GaussianTensors:
        """Return activated tensors expected by ``gsplat.rasterization``."""
        raw = self.raw_tensors()
        return GaussianTensors(
            means=raw.means,
            scales=raw.log_scales.exp(),
            quats=torch.nn.functional.normalize(raw.quats, dim=-1),
            opacities=raw.logit_opacities.sigmoid().squeeze(-1),
            colors=torch.cat([raw.features_dc, raw.features_rest], dim=1),
        )

    def export_tensors(self) -> RawGaussianTensors:
        """Return detached raw tensors suitable for PLY checkpoint writing."""
        raw = self.raw_tensors()
        return RawGaussianTensors(
            means=raw.means.detach(),
            log_scales=raw.log_scales.detach(),
            quats=torch.nn.functional.normalize(raw.quats.detach(), dim=-1),
            logit_opacities=raw.logit_opacities.detach(),
            features_dc=raw.features_dc.detach(),
            features_rest=raw.features_rest.detach(),
        )

    def checkpoint_payload(self, extra: Dict[str, object] | None = None) -> Dict[str, object]:
        """Build a resumable torch checkpoint payload for this experiment."""
        payload: Dict[str, object] = {
            "model_state_dict": self.state_dict(),
            "sh_degree": self.sh_degree,
            "hidden_dim": self.hidden_dim,
            "hidden_layers": self.hidden_layers,
            "num_gaussians": self.num_gaussians,
            "raw_output_dim": self.raw_output_dim,
        }
        if extra:
            payload.update(extra)
        return payload

    def normalized_inputs(self) -> Tensor:
        return (self.anchor_xyz - self.input_center) / self.input_scale

    @staticmethod
    def _build_mlp(input_dim: int, hidden_dim: int, hidden_layers: int, output_dim: int) -> nn.Sequential:
        layers: list[nn.Module] = []
        dim = input_dim
        for _ in range(hidden_layers):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.SiLU())
            dim = hidden_dim
        output = nn.Linear(dim, output_dim)
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)
        layers.append(output)
        return nn.Sequential(*layers)

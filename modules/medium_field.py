"""Learnable low-capacity water medium field."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class MediumFieldConfig:
    """Narrow configuration needed to construct a 3D medium field."""

    scene_center: tuple[float, float, float]
    scene_scale: float
    num_samples: int = 16
    hidden_dim: int = 32
    density_bias: float = -4.0
    medium_rgb_init: tuple[float, float, float] = (0.45, 0.65, 0.75)


class MediumField(nn.Module):
    """Small MLP that predicts extinction and backscatter densities in 3D."""

    def __init__(self, config: MediumFieldConfig) -> None:
        super().__init__()
        if config.scene_scale <= 0.0:
            raise ValueError("scene_scale must be positive")
        hidden = max(int(config.hidden_dim), 4)
        self.config = config
        self.register_buffer("scene_center", torch.tensor(config.scene_center, dtype=torch.float32))
        self.register_buffer("scene_scale", torch.tensor(float(config.scene_scale), dtype=torch.float32))
        self.net = nn.Sequential(
            nn.Linear(3, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 6),
        )
        rgb_init = torch.tensor(config.medium_rgb_init, dtype=torch.float32).clamp(1e-4, 1.0 - 1e-4)
        self.medium_rgb_logit = nn.Parameter(torch.logit(rgb_init))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        with torch.no_grad():
            final = self.net[-1]
            if isinstance(final, nn.Linear):
                final.weight.mul_(0.1)
                final.bias.zero_()

    @property
    def num_samples(self) -> int:
        return max(int(self.config.num_samples), 1)

    @property
    def medium_rgb(self) -> Tensor:
        return torch.sigmoid(self.medium_rgb_logit)

    def forward(self, points: Tensor) -> tuple[Tensor, Tensor]:
        """Return ``(beta_ext, sigma_b)`` for points with shape ``[..., 3]``."""
        if points.shape[-1] != 3:
            raise ValueError("points must have shape [..., 3]")
        normalized = (points - self.scene_center.to(points)) / self.scene_scale.to(points).clamp_min(1e-6)
        raw = self.net(normalized.reshape(-1, 3)).reshape(*points.shape[:-1], 6)
        beta = F.softplus(raw[..., :3] + float(self.config.density_bias))
        scatter_ratio = torch.sigmoid(raw[..., 3:6])
        sigma_b = scatter_ratio * beta
        return beta, sigma_b

    def checkpoint_payload(self) -> dict[str, Any]:
        """Return a serializable checkpoint payload without doing file I/O."""
        return {
            "config": asdict(self.config),
            "state_dict": self.state_dict(),
        }

    @classmethod
    def from_checkpoint_payload(cls, payload: dict[str, Any], device: torch.device | str) -> "MediumField":
        config_data = dict(payload["config"])
        config_data["scene_center"] = tuple(float(v) for v in config_data["scene_center"])
        config_data["medium_rgb_init"] = tuple(float(v) for v in config_data.get("medium_rgb_init", (0.45, 0.65, 0.75)))
        config = MediumFieldConfig(**config_data)
        medium = cls(config).to(device)
        medium.load_state_dict(payload["state_dict"])
        return medium


def scene_medium_config(
    point_cloud_xyz,
    camera_centers,
    num_samples: int,
    hidden_dim: int,
    density_bias: float,
    medium_rgb_init: tuple[float, float, float] = (0.45, 0.65, 0.75),
) -> MediumFieldConfig:
    """Build a stable medium config from scene points and camera centers."""
    points = torch.as_tensor(point_cloud_xyz, dtype=torch.float32) if point_cloud_xyz is not None else torch.empty(0, 3)
    centers = torch.as_tensor(camera_centers, dtype=torch.float32) if camera_centers is not None else torch.empty(0, 3)
    if points.numel() > 0:
        center = points.mean(dim=0)
        distances = torch.linalg.norm(points - center[None, :], dim=-1)
        scale = torch.quantile(distances, 0.90).clamp_min(1e-3)
    elif centers.numel() > 0:
        center = centers.mean(dim=0)
        distances = torch.linalg.norm(centers - center[None, :], dim=-1)
        scale = distances.mean().clamp_min(1e-3)
    else:
        center = torch.zeros(3)
        scale = torch.tensor(1.0)
    return MediumFieldConfig(
        scene_center=tuple(float(v) for v in center.tolist()),
        scene_scale=float(scale.item() * 1.5),
        num_samples=max(int(num_samples), 1),
        hidden_dim=max(int(hidden_dim), 4),
        density_bias=float(density_bias),
        medium_rgb_init=tuple(float(v) for v in medium_rgb_init),
    )


def medium_checkpoint_path_for_ply(ply_path: Path) -> Path:
    """Return the sidecar medium checkpoint path for a PLY checkpoint path."""
    path = ply_path
    if path.name == "final.ply":
        return path.with_name("medium.pt")
    return path.with_name(f"{path.stem}_medium.pt")

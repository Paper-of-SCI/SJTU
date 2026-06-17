"""Optional optimizer helpers for external 3DGS loops."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch

from modules.gaussian_model import GaussianModel


@dataclass
class OptimConfig:
    position_lr: float = 1.6e-4
    feature_lr: float = 2.5e-3
    opacity_lr: float = 5.0e-2
    scaling_lr: float = 5.0e-3
    rotation_lr: float = 1.0e-3
    eps: float = 1.0e-15


def build_3dgs_optimizer(model: GaussianModel, config: OptimConfig | None = None) -> torch.optim.Adam:
    """Build Adam with standard 3DGS parameter groups."""
    print(model.means)
    config = config or OptimConfig()
    groups = [
        {"params": [model.means], "lr": config.position_lr, "name": "means"},
        {"params": [model.features_dc], "lr": config.feature_lr, "name": "features_dc"},
        {"params": [model.features_rest], "lr": config.feature_lr / 20.0, "name": "features_rest"},
        {"params": [model.logit_opacities], "lr": config.opacity_lr, "name": "logit_opacities"},
        {"params": [model.log_scales], "lr": config.scaling_lr, "name": "log_scales"},
        {"params": [model.quats], "lr": config.rotation_lr, "name": "quats"},
    ]
    return torch.optim.Adam(groups, eps=config.eps)


def exponential_lr(
    lr_init: float,
    lr_final: float,
    max_steps: int,
    delay_steps: int = 0,
    delay_mult: float = 1.0,
) -> Callable[[int], float]:
    """Return a 3DGS-style exponential learning-rate schedule."""

    def schedule(step: int) -> float:
        if step < 0:
            return 0.0
        if lr_init == 0.0 and lr_final == 0.0:
            return 0.0
        if delay_steps > 0:
            delay = delay_mult + (1.0 - delay_mult) * math.sin(0.5 * math.pi * min(step / delay_steps, 1.0))
        else:
            delay = 1.0
        t = min(step / max(max_steps, 1), 1.0)
        return delay * math.exp(math.log(lr_init) * (1.0 - t) + math.log(max(lr_final, 1e-15)) * t)

    return schedule


def set_group_lr(optimizer: torch.optim.Optimizer, group_name: str, lr: float) -> None:
    for group in optimizer.param_groups:
        if group.get("name") == group_name:
            group["lr"] = lr

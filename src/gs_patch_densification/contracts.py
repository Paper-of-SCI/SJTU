"""Narrow contracts for patch-guided Gaussian densification."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PatchDetailConfig:
    """Configuration for patch-level detail scoring."""

    patch_size: int = 8
    edge_weight: float = 0.75
    semantic_base: float = 0.2
    use_semantic: bool = False
    eps: float = 1e-6


@dataclass(frozen=True)
class PatchDetailResult:
    """Patch detail score and the normalized factors that produced it."""

    detail: torch.Tensor
    patch_error: torch.Tensor
    patch_edge: torch.Tensor
    semantic_importance: torch.Tensor


@dataclass(frozen=True)
class DensificationConfig:
    """Configuration for selecting and applying densification decisions."""

    gradient_threshold: float = 0.0
    detail_lambda: float = 2.0
    max_new_gaussians: int = 16
    large_scale_threshold: float = 4.0
    clone_jitter_fraction: float = 0.08
    split_scale_shrink: float = 0.65
    prune_fraction: float = 0.0
    min_opacity: float = 0.02
    eps: float = 1e-6


@dataclass(frozen=True)
class DensificationDecision:
    """Selected Gaussian indices and diagnostic scores."""

    clone_indices: torch.Tensor
    split_indices: torch.Tensor
    prune_indices: torch.Tensor
    gradient: torch.Tensor
    adjusted_gradient: torch.Tensor
    per_gaussian_detail: torch.Tensor

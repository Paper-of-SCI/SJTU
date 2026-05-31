"""Pure tensor decisions for patch-guided Gaussian densification."""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from .contracts import DensificationConfig, DensificationDecision
from .patch_detail import normalize_minmax


def _empty_indices(device: torch.device) -> torch.Tensor:
    return torch.empty(0, device=device, dtype=torch.long)


def _gradient_norm(position_gradients: torch.Tensor) -> torch.Tensor:
    if position_gradients.ndim == 1:
        return position_gradients.abs()
    if position_gradients.ndim == 2:
        return torch.linalg.norm(position_gradients, dim=-1)
    raise ValueError("position_gradients must have shape [num_gaussians] or [num_gaussians, dims]")


def select_densification_decision(
    *,
    position_gradients: torch.Tensor,
    log_scales_xy: torch.Tensor,
    opacity_logits: torch.Tensor,
    per_gaussian_detail: torch.Tensor,
    config: DensificationConfig,
    contribution_scores: Optional[torch.Tensor] = None,
) -> DensificationDecision:
    """Select clone, split and prune indices from narrow per-Gaussian signals."""

    gradient = _gradient_norm(position_gradients)
    gaussian_count = gradient.shape[0]
    if log_scales_xy.shape != (gaussian_count, 2):
        raise ValueError("log_scales_xy must have shape [num_gaussians, 2]")
    if opacity_logits.shape != (gaussian_count,):
        raise ValueError("opacity_logits must have shape [num_gaussians]")
    if per_gaussian_detail.shape != (gaussian_count,):
        raise ValueError("per_gaussian_detail must have shape [num_gaussians]")
    if contribution_scores is not None and contribution_scores.shape != (gaussian_count,):
        raise ValueError("contribution_scores must have shape [num_gaussians]")

    adjusted_gradient = gradient * (1.0 + config.detail_lambda * per_gaussian_detail)
    candidate_indices = torch.nonzero(adjusted_gradient > config.gradient_threshold, as_tuple=False).flatten()
    if candidate_indices.numel() > 0 and config.max_new_gaussians > 0:
        order = torch.argsort(adjusted_gradient[candidate_indices], descending=True)
        selected = candidate_indices[order[: config.max_new_gaussians]]
        max_scales = torch.exp(log_scales_xy[selected]).amax(dim=-1)
        split_mask = max_scales >= config.large_scale_threshold
        split_indices = selected[split_mask]
        clone_indices = selected[~split_mask]
    else:
        clone_indices = _empty_indices(gradient.device)
        split_indices = _empty_indices(gradient.device)
        selected = _empty_indices(gradient.device)

    prune_count = int(gaussian_count * config.prune_fraction)
    if prune_count <= 0:
        prune_indices = _empty_indices(gradient.device)
    else:
        opacity = torch.sigmoid(opacity_logits)
        contribution = opacity if contribution_scores is None else contribution_scores
        low_opacity = 1.0 - normalize_minmax(opacity, eps=config.eps)
        low_contribution = 1.0 - normalize_minmax(contribution, eps=config.eps)
        low_detail = 1.0 - normalize_minmax(per_gaussian_detail, eps=config.eps)
        below_opacity = (opacity < config.min_opacity).to(dtype=opacity.dtype)
        prune_score = low_opacity * low_contribution * low_detail + below_opacity
        if selected.numel() > 0:
            prune_score = prune_score.clone()
            prune_score[selected] = -1.0
        prune_count = min(prune_count, max(gaussian_count - selected.numel(), 0))
        if prune_count > 0:
            prune_indices = torch.argsort(prune_score, descending=True)[:prune_count]
        else:
            prune_indices = _empty_indices(gradient.device)

    return DensificationDecision(
        clone_indices=clone_indices,
        split_indices=split_indices,
        prune_indices=prune_indices,
        gradient=gradient,
        adjusted_gradient=adjusted_gradient,
        per_gaussian_detail=per_gaussian_detail,
    )


def build_child_gaussians(
    *,
    means_xy: torch.Tensor,
    log_scales_xy: torch.Tensor,
    opacity_logits: torch.Tensor,
    color_logits: torch.Tensor,
    decision: DensificationDecision,
    config: DensificationConfig,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create child Gaussian tensors for selected clone and split sources."""

    if color_logits.ndim != 2 or color_logits.shape[0] != means_xy.shape[0]:
        raise ValueError("color_logits must have shape [num_gaussians, channels]")

    child_means = []
    child_log_scales = []
    child_opacities = []
    child_colors = []

    if decision.clone_indices.numel() > 0:
        clone_idx = decision.clone_indices
        clone_scales = torch.exp(log_scales_xy[clone_idx])
        clone_noise = torch.randn(
            clone_scales.shape,
            device=means_xy.device,
            dtype=means_xy.dtype,
            generator=generator,
        )
        child_means.append(means_xy[clone_idx] + clone_noise * clone_scales * config.clone_jitter_fraction)
        child_log_scales.append(log_scales_xy[clone_idx])
        child_opacities.append(opacity_logits[clone_idx])
        child_colors.append(color_logits[clone_idx])

    if decision.split_indices.numel() > 0:
        split_idx = decision.split_indices
        split_scales = torch.exp(log_scales_xy[split_idx])
        split_noise = torch.randn(
            split_scales.shape,
            device=means_xy.device,
            dtype=means_xy.dtype,
            generator=generator,
        )
        child_means.append(means_xy[split_idx] + split_noise * split_scales)
        shrink = torch.log(torch.as_tensor(config.split_scale_shrink, device=means_xy.device, dtype=means_xy.dtype))
        child_log_scales.append(log_scales_xy[split_idx] + shrink)
        child_opacities.append(opacity_logits[split_idx])
        child_colors.append(color_logits[split_idx])

    if not child_means:
        return (
            means_xy.new_empty((0, 2)),
            log_scales_xy.new_empty((0, 2)),
            opacity_logits.new_empty((0,)),
            color_logits.new_empty((0, color_logits.shape[-1])),
        )

    return (
        torch.cat(child_means, dim=0),
        torch.cat(child_log_scales, dim=0),
        torch.cat(child_opacities, dim=0),
        torch.cat(child_colors, dim=0),
    )


def apply_gaussian_update(
    *,
    means_xy: torch.Tensor,
    log_scales_xy: torch.Tensor,
    opacity_logits: torch.Tensor,
    color_logits: torch.Tensor,
    decision: DensificationDecision,
    child_means_xy: torch.Tensor,
    child_log_scales_xy: torch.Tensor,
    child_opacity_logits: torch.Tensor,
    child_color_logits: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return tensors with pruned parents removed and children appended."""

    keep_mask = torch.ones(means_xy.shape[0], device=means_xy.device, dtype=torch.bool)
    if decision.prune_indices.numel() > 0:
        keep_mask[decision.prune_indices] = False

    return (
        torch.cat((means_xy[keep_mask], child_means_xy), dim=0),
        torch.cat((log_scales_xy[keep_mask], child_log_scales_xy), dim=0),
        torch.cat((opacity_logits[keep_mask], child_opacity_logits), dim=0),
        torch.cat((color_logits[keep_mask], child_color_logits), dim=0),
    )

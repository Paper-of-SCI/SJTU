"""Patch-detail scoring and Gaussian-to-patch accumulation."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from .contracts import PatchDetailConfig, PatchDetailResult


ImageSize = Tuple[int, int]


def normalize_minmax(values: torch.Tensor, *, eps: float = 1e-6) -> torch.Tensor:
    """Normalize a tensor to [0, 1] with a stable all-equal fallback."""

    value_min = values.amin()
    value_max = values.amax()
    denom = value_max - value_min
    if float(denom.detach().cpu()) <= eps:
        return torch.zeros_like(values)
    return (values - value_min) / denom.clamp_min(eps)


def _validate_image_pair(rendered_rgb: torch.Tensor, target_rgb: torch.Tensor, patch_size: int) -> None:
    if rendered_rgb.shape != target_rgb.shape:
        raise ValueError("rendered_rgb and target_rgb must have the same shape")
    if rendered_rgb.ndim != 3 or rendered_rgb.shape[-1] != 3:
        raise ValueError("rendered_rgb and target_rgb must have shape [height, width, 3]")
    height, width, _ = rendered_rgb.shape
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError("image height and width must be divisible by patch_size")


def _avg_pool_map(pixel_map: torch.Tensor, patch_size: int) -> torch.Tensor:
    return F.avg_pool2d(
        pixel_map[None, None],
        kernel_size=patch_size,
        stride=patch_size,
    )[0, 0]


def _sobel_edge(gray: torch.Tensor) -> torch.Tensor:
    sobel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=gray.device,
        dtype=gray.dtype,
    )
    sobel_y = sobel_x.t()
    image = gray[None, None]
    padded = F.pad(image, (1, 1, 1, 1), mode="replicate")
    grad_x = F.conv2d(padded, sobel_x[None, None])
    grad_y = F.conv2d(padded, sobel_y[None, None])
    return torch.sqrt(grad_x[0, 0].pow(2) + grad_y[0, 0].pow(2))


def compute_patch_detail(
    rendered_rgb: torch.Tensor,
    target_rgb: torch.Tensor,
    config: PatchDetailConfig,
    *,
    semantic_importance: Optional[torch.Tensor] = None,
) -> PatchDetailResult:
    """Compute patch_detail = error * (1 + edge_weight * edge) * semantic_factor."""

    _validate_image_pair(rendered_rgb, target_rgb, config.patch_size)

    pixel_error = (rendered_rgb.detach() - target_rgb.detach()).abs().mean(dim=-1)
    patch_error = normalize_minmax(_avg_pool_map(pixel_error, config.patch_size), eps=config.eps)

    gray = target_rgb.detach().mean(dim=-1)
    patch_edge = normalize_minmax(_avg_pool_map(_sobel_edge(gray), config.patch_size), eps=config.eps)

    if config.use_semantic:
        if semantic_importance is None:
            raise ValueError("semantic_importance is required when use_semantic is true")
        if semantic_importance.shape != target_rgb.shape[:-1]:
            raise ValueError("semantic_importance must have shape [height, width]")
        patch_semantic = _avg_pool_map(semantic_importance.detach().clamp(0.0, 1.0), config.patch_size)
        patch_semantic = patch_semantic.clamp(0.0, 1.0)
        semantic_factor = config.semantic_base + (1.0 - config.semantic_base) * patch_semantic
    else:
        patch_semantic = torch.zeros_like(patch_error)
        semantic_factor = torch.ones_like(patch_error)

    detail = patch_error * (1.0 + config.edge_weight * patch_edge) * semantic_factor
    return PatchDetailResult(
        detail=detail,
        patch_error=patch_error,
        patch_edge=patch_edge,
        semantic_importance=patch_semantic,
    )


def gaussian_patch_indices(
    means_xy: torch.Tensor,
    image_size: ImageSize,
    patch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Map Gaussian centers in pixel coordinates to flat patch indices."""

    if means_xy.ndim != 2 or means_xy.shape[-1] != 2:
        raise ValueError("means_xy must have shape [num_gaussians, 2]")
    height, width = image_size
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError("image height and width must be divisible by patch_size")

    patch_h = height // patch_size
    patch_w = width // patch_size
    x = means_xy[:, 0]
    y = means_xy[:, 1]
    valid = (x >= 0.0) & (x < float(width)) & (y >= 0.0) & (y < float(height))
    patch_x = torch.div(x.clamp(0.0, float(width - 1)), patch_size, rounding_mode="floor").long()
    patch_y = torch.div(y.clamp(0.0, float(height - 1)), patch_size, rounding_mode="floor").long()
    flat_indices = (patch_y * patch_w + patch_x).clamp(0, patch_h * patch_w - 1)
    return flat_indices, valid


def accumulate_patch_detail_to_gaussians(
    means_xy: torch.Tensor,
    patch_detail: torch.Tensor,
    image_size: ImageSize,
    patch_size: int,
) -> torch.Tensor:
    """Assign each visible Gaussian the detail value of its projected patch."""

    flat_indices, valid = gaussian_patch_indices(means_xy, image_size, patch_size)
    flat_detail = patch_detail.reshape(-1)
    per_gaussian = torch.zeros(means_xy.shape[0], device=means_xy.device, dtype=patch_detail.dtype)
    if flat_detail.numel() == 0:
        return per_gaussian
    per_gaussian[valid] = flat_detail[flat_indices[valid]]
    return per_gaussian


def patch_density_from_gaussians(
    means_xy: torch.Tensor,
    image_size: ImageSize,
    patch_size: int,
    *,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Accumulate Gaussian counts or weights into the patch grid."""

    flat_indices, valid = gaussian_patch_indices(means_xy, image_size, patch_size)
    height, width = image_size
    patch_h = height // patch_size
    patch_w = width // patch_size
    density = torch.zeros(patch_h * patch_w, device=means_xy.device, dtype=means_xy.dtype)
    if weights is None:
        valid_weights = torch.ones(valid.sum(), device=means_xy.device, dtype=means_xy.dtype)
    else:
        if weights.ndim != 1 or weights.shape[0] != means_xy.shape[0]:
            raise ValueError("weights must have shape [num_gaussians]")
        valid_weights = weights[valid].to(dtype=means_xy.dtype)
    density.scatter_add_(0, flat_indices[valid], valid_weights)
    return density.reshape(patch_h, patch_w)

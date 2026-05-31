"""Differentiable feature splatting utilities.

This module is intentionally narrow: it only renders per-Gaussian values
onto a 2D grid with front-to-back alpha blending. A full 3DGS implementation
should replace this adapter with its CUDA rasterizer while keeping the same
loss contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch


ImageSize = Tuple[int, int]


@dataclass(frozen=True)
class SplatRenderOutput:
    """Rendered value map and accumulated alpha."""

    value: torch.Tensor
    alpha: torch.Tensor


def _validate_inputs(
    means_xy: torch.Tensor,
    log_scales_xy: torch.Tensor,
    opacity_logits: torch.Tensor,
    values: torch.Tensor,
) -> None:
    if means_xy.ndim != 2 or means_xy.shape[-1] != 2:
        raise ValueError("means_xy must have shape [num_gaussians, 2]")
    if log_scales_xy.shape != means_xy.shape:
        raise ValueError("log_scales_xy must match means_xy shape")
    if opacity_logits.ndim != 1 or opacity_logits.shape[0] != means_xy.shape[0]:
        raise ValueError("opacity_logits must have shape [num_gaussians]")
    if values.ndim != 2 or values.shape[0] != means_xy.shape[0]:
        raise ValueError("values must have shape [num_gaussians, channels]")


def _patch_grid(
    image_size: ImageSize,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    height, width = image_size
    ys = torch.arange(height, device=device, dtype=dtype) + 0.5
    xs = torch.arange(width, device=device, dtype=dtype) + 0.5
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack((xx, yy), dim=-1)


def _exclusive_cumprod(values: torch.Tensor, dim: int) -> torch.Tensor:
    inclusive = torch.cumprod(values, dim=dim)
    first_shape = list(values.shape)
    first_shape[dim] = 1
    first = torch.ones(first_shape, device=values.device, dtype=values.dtype)
    return torch.cat((first, inclusive.narrow(dim, 0, values.shape[dim] - 1)), dim=dim)


def alpha_blend_splat_2d(
    *,
    means_xy: torch.Tensor,
    log_scales_xy: torch.Tensor,
    opacity_logits: torch.Tensor,
    values: torch.Tensor,
    image_size: ImageSize,
    depths: Optional[torch.Tensor] = None,
    background: Optional[torch.Tensor] = None,
    min_scale: float = 1e-3,
    eps: float = 1e-6,
) -> SplatRenderOutput:
    """Render Gaussian-carried values to a 2D map.

    Args:
        means_xy: Gaussian centers in output pixel coordinates, [N, 2].
        log_scales_xy: Log standard deviations in output pixels, [N, 2].
        opacity_logits: Per-Gaussian opacity logits, [N].
        values: Per-Gaussian values to blend, [N, C].
        image_size: Output size as (height, width).
        depths: Optional per-Gaussian depths. Lower values are rendered first.
        background: Optional background value, [C].
        min_scale: Lower bound for Gaussian standard deviations.
        eps: Numerical stability constant.

    Returns:
        SplatRenderOutput with value [H, W, C] and alpha [H, W].
    """

    _validate_inputs(means_xy, log_scales_xy, opacity_logits, values)

    if depths is not None:
        if depths.ndim != 1 or depths.shape[0] != means_xy.shape[0]:
            raise ValueError("depths must have shape [num_gaussians]")
        order = torch.argsort(depths)
        means_xy = means_xy[order]
        log_scales_xy = log_scales_xy[order]
        opacity_logits = opacity_logits[order]
        values = values[order]

    grid = _patch_grid(image_size, device=means_xy.device, dtype=means_xy.dtype)
    scales = torch.exp(log_scales_xy).clamp_min(min_scale)
    delta = grid.unsqueeze(0) - means_xy[:, None, None, :]
    normalized_delta = delta / scales[:, None, None, :]
    gaussian = torch.exp(-0.5 * torch.sum(normalized_delta * normalized_delta, dim=-1))
    alpha = torch.sigmoid(opacity_logits)[:, None, None] * gaussian
    alpha = alpha.clamp(min=0.0, max=1.0 - eps)

    transmittance = _exclusive_cumprod(1.0 - alpha, dim=0)
    weights = transmittance * alpha
    rendered = torch.einsum("nhw,nc->hwc", weights, values)
    accumulated_alpha = weights.sum(dim=0).clamp(max=1.0)

    if background is not None:
        if background.ndim != 1 or background.shape[0] != values.shape[1]:
            raise ValueError("background must have shape [channels]")
        rendered = rendered + (1.0 - accumulated_alpha).unsqueeze(-1) * background

    return SplatRenderOutput(value=rendered, alpha=accumulated_alpha)

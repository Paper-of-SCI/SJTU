"""Toy experiment for patch-guided Gaussian densification.

This script does not use feature loss. It tests whether patch-level error,
edge strength and semantic importance can guide Gaussian clone/split/prune
decisions toward high-detail regions in a small differentiable 2D splatting
setup.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.gs_patch_densification import (
    DensificationConfig,
    PatchDetailConfig,
    accumulate_patch_detail_to_gaussians,
    apply_gaussian_update,
    build_child_gaussians,
    compute_patch_detail,
    patch_density_from_gaussians,
    select_densification_decision,
)


ImageSize = Tuple[int, int]


@dataclass(frozen=True)
class ViewSpec:
    name: str
    offset_xy: Tuple[float, float]


@dataclass(frozen=True)
class SceneTarget:
    rgb: torch.Tensor
    semantic_mask: torch.Tensor


@dataclass(frozen=True)
class VariantSpec:
    name: str
    use_patch_detail: bool
    use_semantic: bool
    reallocate: bool
    detail_lambda: float


@dataclass(frozen=True)
class RenderOutput:
    rgb: torch.Tensor
    alpha: torch.Tensor
    contribution: torch.Tensor


VARIANTS = (
    VariantSpec(
        name="baseline_grad",
        use_patch_detail=False,
        use_semantic=False,
        reallocate=False,
        detail_lambda=0.0,
    ),
    VariantSpec(
        name="error_edge",
        use_patch_detail=True,
        use_semantic=False,
        reallocate=False,
        detail_lambda=2.0,
    ),
    VariantSpec(
        name="error_edge_semantic",
        use_patch_detail=True,
        use_semantic=True,
        reallocate=False,
        detail_lambda=2.5,
    ),
    VariantSpec(
        name="reallocation",
        use_patch_detail=True,
        use_semantic=True,
        reallocate=True,
        detail_lambda=2.5,
    ),
)


TRAIN_VIEWS = (
    ViewSpec("train_left", (-3.5, 0.0)),
    ViewSpec("train_right", (3.0, 1.2)),
    ViewSpec("train_up", (0.5, -2.8)),
)
HELD_OUT_VIEW = ViewSpec("held_out", (1.6, -1.4))


def _pixel_grid(image_size: ImageSize, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    height, width = image_size
    ys = torch.arange(height, device=device, dtype=torch.float32) + 0.5
    xs = torch.arange(width, device=device, dtype=torch.float32) + 0.5
    return torch.meshgrid(ys, xs, indexing="ij")


def build_underwater_target(
    image_size: ImageSize,
    view: ViewSpec,
    device: torch.device,
) -> SceneTarget:
    """Create a deterministic underwater-like target and semantic mask."""

    height, width = image_size
    yy, xx = _pixel_grid(image_size, device)
    world_x = xx - view.offset_xy[0]
    world_y = yy - view.offset_xy[1]
    nx = world_x / float(width - 1) * 2.0 - 1.0
    ny = world_y / float(height - 1) * 2.0 - 1.0

    depth_like = ((ny + 1.0) * 0.5).clamp(0.0, 1.0)
    water = torch.stack(
        (
            0.035 + 0.035 * (1.0 - depth_like),
            0.18 + 0.08 * (1.0 - nx.abs()),
            0.34 + 0.18 * (1.0 - depth_like),
        ),
        dim=-1,
    )
    seabed = (ny > 0.45).float()
    seabed_color = torch.stack(
        (
            0.22 + 0.05 * torch.sin(13.0 * nx),
            0.24 + 0.04 * torch.cos(11.0 * ny),
            0.20 + 0.03 * torch.sin(9.0 * (nx + ny)),
        ),
        dim=-1,
    )
    rgb = water * (1.0 - 0.35 * seabed[..., None]) + seabed_color * (0.35 * seabed[..., None])

    coral_core = (((nx + 0.28) / 0.42) ** 2 + ((ny + 0.04) / 0.30) ** 2) < 1.0
    branch_a = (torch.abs(ny + 0.50 * nx - 0.10) < 0.035) & (nx > -0.60) & (nx < 0.48)
    branch_b = (torch.abs(ny - 0.82 * nx - 0.12) < 0.030) & (nx > -0.25) & (nx < 0.58)
    branch_c = (torch.abs(ny + 0.12) < 0.026) & (nx > -0.48) & (nx < 0.18)
    coral_mask = coral_core | branch_a | branch_b | branch_c

    object_mask = (nx > 0.38) & (nx < 0.72) & (ny > 0.04) & (ny < 0.32)
    object_edge = object_mask & (
        (torch.abs(nx - 0.38) < 0.025)
        | (torch.abs(nx - 0.72) < 0.025)
        | (torch.abs(ny - 0.04) < 0.025)
        | (torch.abs(ny - 0.32) < 0.025)
    )

    texture = 0.5 + 0.5 * torch.sin(42.0 * nx + 0.5) * torch.cos(37.0 * ny - 0.2)
    coral_color = torch.stack(
        (
            0.62 + 0.20 * texture,
            0.28 + 0.10 * torch.sin(17.0 * nx).abs(),
            0.18 + 0.08 * torch.cos(21.0 * ny).abs(),
        ),
        dim=-1,
    )
    object_color = torch.stack(
        (
            0.64 + 0.08 * object_edge.float(),
            0.56 + 0.07 * torch.sin(23.0 * nx).abs(),
            0.40 + 0.08 * torch.cos(19.0 * ny).abs(),
        ),
        dim=-1,
    )

    rgb = torch.where(coral_mask[..., None], coral_color, rgb)
    rgb = torch.where(object_mask[..., None], object_color, rgb)

    snow_signal = torch.sin(77.0 * nx + 0.3) * torch.sin(53.0 * ny - 0.7)
    snow = (snow_signal > 0.965) & (ny < 0.25) & (~coral_mask) & (~object_mask)
    rgb = torch.where(snow[..., None], torch.full_like(rgb, 0.86), rgb)

    haze = (0.18 + 0.16 * depth_like).clamp(0.0, 0.45)
    rgb = rgb * (1.0 - haze[..., None]) + water * haze[..., None]
    semantic_mask = (coral_mask | object_mask).float()
    return SceneTarget(rgb=rgb.clamp(0.0, 1.0), semantic_mask=semantic_mask)


class ToyGaussianModel(torch.nn.Module):
    def __init__(self, gaussian_count: int, image_size: ImageSize, seed: int, device: torch.device) -> None:
        super().__init__()
        height, width = image_size
        generator = torch.Generator(device=device).manual_seed(seed)
        means = torch.rand((gaussian_count, 2), device=device, generator=generator)
        means[:, 0] *= float(width)
        means[:, 1] *= float(height)
        scales = 3.0 + 3.0 * torch.rand((gaussian_count, 2), device=device, generator=generator)
        self.means_xy = torch.nn.Parameter(means)
        self.log_scales_xy = torch.nn.Parameter(torch.log(scales))
        self.opacity_logits = torch.nn.Parameter(torch.full((gaussian_count,), -0.75, device=device))
        self.color_logits = torch.nn.Parameter(torch.randn((gaussian_count, 3), device=device, generator=generator) * 0.35)
        self.image_size = image_size

    def render(self, view: ViewSpec) -> RenderOutput:
        return render_gaussians(
            means_xy=self.means_xy,
            log_scales_xy=self.log_scales_xy,
            opacity_logits=self.opacity_logits,
            color_logits=self.color_logits,
            image_size=self.image_size,
            view_offset_xy=view.offset_xy,
        )

    @torch.no_grad()
    def clamp_parameters(self) -> None:
        height, width = self.image_size
        self.means_xy[:, 0].clamp_(-8.0, float(width + 8))
        self.means_xy[:, 1].clamp_(-8.0, float(height + 8))
        self.log_scales_xy.clamp_(math.log(0.8), math.log(10.0))
        self.opacity_logits.clamp_(-6.0, 6.0)
        self.color_logits.clamp_(-7.0, 7.0)

    def replace_state(
        self,
        means_xy: torch.Tensor,
        log_scales_xy: torch.Tensor,
        opacity_logits: torch.Tensor,
        color_logits: torch.Tensor,
    ) -> None:
        self.means_xy = torch.nn.Parameter(means_xy.detach())
        self.log_scales_xy = torch.nn.Parameter(log_scales_xy.detach())
        self.opacity_logits = torch.nn.Parameter(opacity_logits.detach())
        self.color_logits = torch.nn.Parameter(color_logits.detach())


def render_gaussians(
    *,
    means_xy: torch.Tensor,
    log_scales_xy: torch.Tensor,
    opacity_logits: torch.Tensor,
    color_logits: torch.Tensor,
    image_size: ImageSize,
    view_offset_xy: Tuple[float, float],
    eps: float = 1e-6,
) -> RenderOutput:
    """Render colored 2D Gaussians with front-to-back alpha blending."""

    yy, xx = _pixel_grid(image_size, means_xy.device)
    grid = torch.stack((xx, yy), dim=-1)
    view_offset = torch.as_tensor(view_offset_xy, device=means_xy.device, dtype=means_xy.dtype)
    projected_means = means_xy + view_offset
    scales = torch.exp(log_scales_xy).clamp_min(1e-3)
    delta = grid.unsqueeze(0) - projected_means[:, None, None, :]
    normalized_delta = delta / scales[:, None, None, :]
    gaussian = torch.exp(-0.5 * torch.sum(normalized_delta * normalized_delta, dim=-1))
    alpha = torch.sigmoid(opacity_logits)[:, None, None] * gaussian
    alpha = alpha.clamp(0.0, 1.0 - eps)
    inclusive = torch.cumprod(1.0 - alpha, dim=0)
    first = torch.ones_like(inclusive[:1])
    transmittance = torch.cat((first, inclusive[:-1]), dim=0)
    weights = transmittance * alpha
    colors = torch.sigmoid(color_logits)
    rgb = torch.einsum("nhw,nc->hwc", weights, colors)
    accumulated_alpha = weights.sum(dim=0).clamp(max=1.0)
    background = torch.tensor((0.03, 0.18, 0.32), device=means_xy.device, dtype=means_xy.dtype)
    rgb = rgb + (1.0 - accumulated_alpha).unsqueeze(-1) * background
    contribution = weights.sum(dim=(1, 2))
    return RenderOutput(rgb=rgb.clamp(0.0, 1.0), alpha=accumulated_alpha, contribution=contribution)


def psnr_from_mse(mse: torch.Tensor) -> torch.Tensor:
    return -10.0 * torch.log10(mse.clamp_min(1e-12))


def masked_mse(prediction: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    sq_error = (prediction - target).pow(2)
    if mask is None:
        return sq_error.mean()
    weighted = sq_error * mask.unsqueeze(-1)
    denom = mask.sum().clamp_min(1.0) * prediction.shape[-1]
    return weighted.sum() / denom


def ssim_index(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Small Torch SSIM implementation for RGB images in [0, 1]."""

    x = prediction.permute(2, 0, 1)[None]
    y = target.permute(2, 0, 1)[None]
    kernel_size = 7
    channels = x.shape[1]
    kernel = torch.ones((channels, 1, kernel_size, kernel_size), device=x.device, dtype=x.dtype)
    kernel = kernel / float(kernel_size * kernel_size)
    padding = kernel_size // 2
    mu_x = F.conv2d(x, kernel, padding=padding, groups=channels)
    mu_y = F.conv2d(y, kernel, padding=padding, groups=channels)
    sigma_x = F.conv2d(x * x, kernel, padding=padding, groups=channels) - mu_x.pow(2)
    sigma_y = F.conv2d(y * y, kernel, padding=padding, groups=channels) - mu_y.pow(2)
    sigma_xy = F.conv2d(x * y, kernel, padding=padding, groups=channels) - mu_x * mu_y
    c1 = 0.01**2
    c2 = 0.03**2
    numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x.pow(2) + mu_y.pow(2) + c1) * (sigma_x + sigma_y + c2)
    return (numerator / denominator.clamp_min(1e-12)).mean()


def crop_to_mask(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    pad: int = 8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    coords = torch.nonzero(mask > 0.5, as_tuple=False)
    if coords.numel() == 0:
        return prediction, target
    height, width = mask.shape
    y0 = max(int(coords[:, 0].min().item()) - pad, 0)
    y1 = min(int(coords[:, 0].max().item()) + pad + 1, height)
    x0 = max(int(coords[:, 1].min().item()) - pad, 0)
    x1 = min(int(coords[:, 1].max().item()) + pad + 1, width)
    return prediction[y0:y1, x0:x1], target[y0:y1, x0:x1]


def lpips_distance(
    prediction: torch.Tensor,
    target: torch.Tensor,
    lpips_model: Optional[torch.nn.Module],
) -> Optional[float]:
    if lpips_model is None:
        return None
    pred_batch = prediction.permute(2, 0, 1)[None] * 2.0 - 1.0
    target_batch = target.permute(2, 0, 1)[None] * 2.0 - 1.0
    with torch.no_grad():
        return float(lpips_model(pred_batch, target_batch).detach().cpu())


def image_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    lpips_model: Optional[torch.nn.Module],
    mask: Optional[torch.Tensor] = None,
) -> Dict[str, Optional[float]]:
    mse = masked_mse(prediction, target, mask)
    if mask is None:
        ssim_pred, ssim_target = prediction, target
    else:
        ssim_pred, ssim_target = crop_to_mask(prediction, target, mask)
    return {
        "mse": float(mse.detach().cpu()),
        "psnr": float(psnr_from_mse(mse).detach().cpu()),
        "ssim": float(ssim_index(ssim_pred, ssim_target).detach().cpu()),
        "lpips": lpips_distance(ssim_pred, ssim_target, lpips_model),
    }


def _quantile_or_zero(values: torch.Tensor, quantile: float) -> float:
    if values.numel() == 0:
        return 0.0
    return float(torch.quantile(values.detach(), quantile).cpu())


def maybe_densify(
    *,
    model: ToyGaussianModel,
    variant: VariantSpec,
    view: ViewSpec,
    target: SceneTarget,
    rendered: RenderOutput,
    position_gradients: torch.Tensor,
    step: int,
    seed: int,
    max_gaussians: int,
    max_new_per_event: int,
    patch_size: int,
    reallocation_start_step: int,
) -> Tuple[bool, Dict[str, float]]:
    current_count = model.means_xy.shape[0]
    projected_means = model.means_xy.detach() + torch.as_tensor(view.offset_xy, device=model.means_xy.device)

    if variant.use_patch_detail:
        detail_result = compute_patch_detail(
            rendered.rgb.detach(),
            target.rgb,
            PatchDetailConfig(
                patch_size=patch_size,
                edge_weight=0.75,
                semantic_base=0.2,
                use_semantic=variant.use_semantic,
            ),
            semantic_importance=target.semantic_mask if variant.use_semantic else None,
        )
        per_detail = accumulate_patch_detail_to_gaussians(
            projected_means,
            detail_result.detail,
            model.image_size,
            patch_size,
        )
    else:
        detail_result = compute_patch_detail(
            rendered.rgb.detach(),
            target.rgb,
            PatchDetailConfig(patch_size=patch_size, edge_weight=0.75, use_semantic=False),
        )
        per_detail = torch.zeros(current_count, device=model.means_xy.device, dtype=model.means_xy.dtype)

    should_reallocate = variant.reallocate and (current_count >= max_gaussians or step >= reallocation_start_step)
    prune_fraction = 0.10 if should_reallocate else 0.0
    expected_prune = int(current_count * prune_fraction)
    free_slots = max(max_gaussians - current_count, 0)
    max_new = min(max_new_per_event, free_slots + expected_prune)
    if max_new <= 0 and expected_prune <= 0:
        return False, {}

    gradient_norm = torch.linalg.norm(position_gradients.detach(), dim=-1)
    threshold = _quantile_or_zero(gradient_norm, 0.65)
    decision = select_densification_decision(
        position_gradients=position_gradients.detach(),
        log_scales_xy=model.log_scales_xy.detach(),
        opacity_logits=model.opacity_logits.detach(),
        per_gaussian_detail=per_detail.detach(),
        contribution_scores=rendered.contribution.detach(),
        config=DensificationConfig(
            gradient_threshold=threshold,
            detail_lambda=variant.detail_lambda,
            max_new_gaussians=max_new,
            large_scale_threshold=4.6,
            clone_jitter_fraction=0.10,
            split_scale_shrink=0.65,
            prune_fraction=prune_fraction,
            min_opacity=0.02,
        ),
    )
    source_indices = torch.cat((decision.clone_indices, decision.split_indices), dim=0)
    if source_indices.numel() == 0 and decision.prune_indices.numel() == 0:
        return False, {}

    generator = torch.Generator(device=model.means_xy.device).manual_seed(seed * 100000 + step)
    child_tensors = build_child_gaussians(
        means_xy=model.means_xy.detach(),
        log_scales_xy=model.log_scales_xy.detach(),
        opacity_logits=model.opacity_logits.detach(),
        color_logits=model.color_logits.detach(),
        decision=decision,
        config=DensificationConfig(clone_jitter_fraction=0.10, split_scale_shrink=0.65),
        generator=generator,
    )
    new_state = apply_gaussian_update(
        means_xy=model.means_xy.detach(),
        log_scales_xy=model.log_scales_xy.detach(),
        opacity_logits=model.opacity_logits.detach(),
        color_logits=model.color_logits.detach(),
        decision=decision,
        child_means_xy=child_tensors[0],
        child_log_scales_xy=child_tensors[1],
        child_opacity_logits=child_tensors[2],
        child_color_logits=child_tensors[3],
    )
    if new_state[0].shape[0] > max_gaussians:
        keep = torch.arange(max_gaussians, device=new_state[0].device)
        new_state = tuple(tensor[keep] for tensor in new_state)  # type: ignore[assignment]
    model.replace_state(*new_state)
    model.clamp_parameters()

    detail_top = torch.quantile(per_detail, 0.80) if per_detail.numel() > 0 else torch.tensor(0.0, device=per_detail.device)
    if source_indices.numel() > 0:
        child_top_fraction = (per_detail[source_indices] >= detail_top).float().mean()
        child_parent_detail = per_detail[source_indices].mean()
    else:
        child_top_fraction = torch.tensor(0.0, device=per_detail.device)
        child_parent_detail = torch.tensor(0.0, device=per_detail.device)
    if decision.prune_indices.numel() > 0:
        pruned_detail = per_detail[decision.prune_indices].mean()
    else:
        pruned_detail = torch.tensor(0.0, device=per_detail.device)

    event = {
        "step": float(step),
        "before_count": float(current_count),
        "after_count": float(model.means_xy.shape[0]),
        "clone_count": float(decision.clone_indices.numel()),
        "split_count": float(decision.split_indices.numel()),
        "prune_count": float(decision.prune_indices.numel()),
        "child_parent_detail_mean": float(child_parent_detail.detach().cpu()),
        "pruned_detail_mean": float(pruned_detail.detach().cpu()),
        "child_top_detail_fraction": float(child_top_fraction.detach().cpu()),
        "patch_detail_mean": float(detail_result.detail.mean().detach().cpu()),
    }
    return True, event


def train_variant(
    *,
    variant: VariantSpec,
    seed: int,
    steps: int,
    image_size: ImageSize,
    patch_size: int,
    initial_gaussians: int,
    max_gaussians: int,
    max_new_per_event: int,
    densify_every: int,
    output_dir: Path,
    device: torch.device,
    lpips_model: Optional[torch.nn.Module],
) -> Dict[str, object]:
    torch.manual_seed(seed)
    targets = {view.name: build_underwater_target(image_size, view, device) for view in (*TRAIN_VIEWS, HELD_OUT_VIEW)}
    model = ToyGaussianModel(initial_gaussians, image_size, seed=seed + 17, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.035)
    event_log: List[Dict[str, float]] = []
    reallocation_start = int(steps * 0.55)

    for step in range(1, steps + 1):
        view = TRAIN_VIEWS[(step + seed) % len(TRAIN_VIEWS)]
        target = targets[view.name]
        optimizer.zero_grad(set_to_none=True)
        rendered = model.render(view)
        loss = F.mse_loss(rendered.rgb, target.rgb) + 0.12 * (rendered.rgb - target.rgb).abs().mean()
        loss.backward()
        position_gradients = model.means_xy.grad.detach().clone()
        optimizer.step()
        model.clamp_parameters()

        if step % densify_every == 0 and step <= int(steps * 0.78):
            changed, event = maybe_densify(
                model=model,
                variant=variant,
                view=view,
                target=target,
                rendered=rendered,
                position_gradients=position_gradients,
                step=step,
                seed=seed,
                max_gaussians=max_gaussians,
                max_new_per_event=max_new_per_event,
                patch_size=patch_size,
                reallocation_start_step=reallocation_start,
            )
            if changed:
                event_log.append(event)
                optimizer = torch.optim.Adam(model.parameters(), lr=0.030)

    held_target = targets[HELD_OUT_VIEW.name]
    with torch.no_grad():
        held_render = model.render(HELD_OUT_VIEW)
        analysis_detail = compute_patch_detail(
            held_render.rgb,
            held_target.rgb,
            PatchDetailConfig(patch_size=patch_size, edge_weight=0.75, semantic_base=0.2, use_semantic=True),
            semantic_importance=held_target.semantic_mask,
        )
        projected_means = model.means_xy.detach() + torch.as_tensor(HELD_OUT_VIEW.offset_xy, device=device)
        count_density = patch_density_from_gaussians(projected_means, image_size, patch_size)
        opacity_density = patch_density_from_gaussians(
            projected_means,
            image_size,
            patch_size,
            weights=torch.sigmoid(model.opacity_logits.detach()),
        )

    full_metrics = image_metrics(held_render.rgb, held_target.rgb, lpips_model=lpips_model)
    roi_metrics = image_metrics(
        held_render.rgb,
        held_target.rgb,
        lpips_model=lpips_model,
        mask=held_target.semantic_mask,
    )
    density_metrics = gaussian_density_metrics(
        count_density=count_density,
        opacity_density=opacity_density,
        patch_detail=analysis_detail.detail,
    )
    variant_dir = output_dir / f"seed_{seed:03d}" / variant.name
    variant_dir.mkdir(parents=True, exist_ok=True)
    save_rgb(variant_dir / "held_out_render.png", held_render.rgb)
    save_rgb(variant_dir / "held_out_target.png", held_target.rgb)
    save_heatmap(variant_dir / "held_out_error.png", (held_render.rgb - held_target.rgb).abs().mean(dim=-1))
    save_patch_heatmap(variant_dir / "patch_detail.png", analysis_detail.detail, image_size)
    save_patch_heatmap(variant_dir / "gaussian_density.png", count_density, image_size)
    save_patch_heatmap(variant_dir / "opacity_density.png", opacity_density, image_size)

    return {
        "seed": seed,
        "variant": variant.name,
        "gaussian_count": int(model.means_xy.shape[0]),
        "full": full_metrics,
        "roi": roi_metrics,
        "density": density_metrics,
        "events": event_log,
    }


def gaussian_density_metrics(
    *,
    count_density: torch.Tensor,
    opacity_density: torch.Tensor,
    patch_detail: torch.Tensor,
) -> Dict[str, float]:
    flat_detail = patch_detail.reshape(-1)
    flat_count = count_density.reshape(-1)
    flat_opacity = opacity_density.reshape(-1)
    if flat_detail.numel() == 0:
        return {
            "top_detail_count_ratio": 0.0,
            "top_detail_opacity_ratio": 0.0,
            "low_detail_count_ratio": 0.0,
        }
    top_threshold = torch.quantile(flat_detail, 0.80)
    low_threshold = torch.quantile(flat_detail, 0.30)
    top_mask = flat_detail >= top_threshold
    low_mask = flat_detail <= low_threshold
    total_count = flat_count.sum().clamp_min(1.0)
    total_opacity = flat_opacity.sum().clamp_min(1e-6)
    return {
        "top_detail_count_ratio": float((flat_count[top_mask].sum() / total_count).detach().cpu()),
        "top_detail_opacity_ratio": float((flat_opacity[top_mask].sum() / total_opacity).detach().cpu()),
        "low_detail_count_ratio": float((flat_count[low_mask].sum() / total_count).detach().cpu()),
    }


def save_rgb(path: Path, image: torch.Tensor) -> None:
    array = (image.detach().cpu().clamp(0.0, 1.0).numpy() * 255.0).astype("uint8")
    Image.fromarray(array).save(path)


def save_heatmap(path: Path, values: torch.Tensor) -> None:
    normalized = values.detach().cpu()
    value_min = normalized.min()
    value_max = normalized.max()
    normalized = (normalized - value_min) / (value_max - value_min).clamp_min(1e-6)
    red = normalized
    green = 1.0 - (2.0 * normalized - 1.0).abs()
    blue = 1.0 - normalized
    image = torch.stack((red, green.clamp(0.0, 1.0), blue), dim=-1)
    array = (image.clamp(0.0, 1.0).numpy() * 255.0).astype("uint8")
    Image.fromarray(array).save(path)


def save_patch_heatmap(path: Path, patch_values: torch.Tensor, image_size: ImageSize) -> None:
    upsampled = F.interpolate(
        patch_values.detach()[None, None],
        size=image_size,
        mode="nearest",
    )[0, 0]
    save_heatmap(path, upsampled)


def load_lpips_model(device: torch.device, skip_lpips: bool) -> Optional[torch.nn.Module]:
    if skip_lpips:
        return None
    try:
        import lpips  # type: ignore[import-not-found]
    except Exception as exc:
        print(f"LPIPS unavailable: {exc}", file=sys.stderr)
        return None
    model = lpips.LPIPS(net="alex").to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _mean_optional(values: Iterable[Optional[float]]) -> Optional[float]:
    present = [value for value in values if value is not None]
    if not present:
        return None
    return float(sum(present) / len(present))


def aggregate_results(runs: Sequence[Dict[str, object]]) -> Dict[str, object]:
    aggregate: Dict[str, object] = {}
    for variant in VARIANTS:
        rows = [row for row in runs if row["variant"] == variant.name]
        aggregate[variant.name] = {
            "full_psnr": _mean_optional(row["full"]["psnr"] for row in rows),  # type: ignore[index]
            "full_ssim": _mean_optional(row["full"]["ssim"] for row in rows),  # type: ignore[index]
            "full_lpips": _mean_optional(row["full"]["lpips"] for row in rows),  # type: ignore[index]
            "roi_psnr": _mean_optional(row["roi"]["psnr"] for row in rows),  # type: ignore[index]
            "roi_ssim": _mean_optional(row["roi"]["ssim"] for row in rows),  # type: ignore[index]
            "roi_lpips": _mean_optional(row["roi"]["lpips"] for row in rows),  # type: ignore[index]
            "gaussian_count": _mean_optional(float(row["gaussian_count"]) for row in rows),
            "top_detail_count_ratio": _mean_optional(
                row["density"]["top_detail_count_ratio"] for row in rows  # type: ignore[index]
            ),
            "low_detail_count_ratio": _mean_optional(
                row["density"]["low_detail_count_ratio"] for row in rows  # type: ignore[index]
            ),
        }
    aggregate["interpretation"] = interpret_results(aggregate)
    return aggregate


def interpret_results(aggregate: Dict[str, object]) -> str:
    baseline = aggregate["baseline_grad"]  # type: ignore[index]
    candidates = [aggregate["error_edge_semantic"], aggregate["reallocation"]]  # type: ignore[index]
    baseline_psnr = baseline["full_psnr"]  # type: ignore[index]
    baseline_roi = baseline["roi_psnr"]  # type: ignore[index]
    baseline_lpips = baseline["full_lpips"]  # type: ignore[index]
    for candidate in candidates:
        psnr_ok = candidate["full_psnr"] is not None and baseline_psnr is not None and candidate["full_psnr"] > baseline_psnr
        roi_ok = candidate["roi_psnr"] is not None and baseline_roi is not None and candidate["roi_psnr"] > baseline_roi
        lpips_ok = (
            baseline_lpips is None
            or candidate["full_lpips"] is None
            or candidate["full_lpips"] <= baseline_lpips
        )
        if (psnr_ok or roi_ok) and lpips_ok:
            return "patch-guided densification is promising in this toy run"
    return "patch-guided densification is not stable enough in this toy run; tune scoring or test inside full 3DGS"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/patch_guided_densification_toy"))
    parser.add_argument("--initial-gaussians", type=int, default=48)
    parser.add_argument("--max-gaussians", type=int, default=128)
    parser.add_argument("--max-new-per-event", type=int, default=20)
    parser.add_argument("--densify-every", type=int, default=120)
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--skip-lpips", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    image_size = (96, 96)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lpips_model = load_lpips_model(device, args.skip_lpips)
    runs: List[Dict[str, object]] = []
    for seed in args.seeds:
        for variant in VARIANTS:
            result = train_variant(
                variant=variant,
                seed=seed,
                steps=args.steps,
                image_size=image_size,
                patch_size=args.patch_size,
                initial_gaussians=args.initial_gaussians,
                max_gaussians=args.max_gaussians,
                max_new_per_event=args.max_new_per_event,
                densify_every=args.densify_every,
                output_dir=args.output_dir,
                device=device,
                lpips_model=lpips_model,
            )
            runs.append(result)
            print(
                json.dumps(
                    {
                        "seed": seed,
                        "variant": variant.name,
                        "full": result["full"],
                        "roi": result["roi"],
                        "density": result["density"],
                    },
                    indent=2,
                )
            )

    metrics = {
        "config": {
            "steps": args.steps,
            "seeds": args.seeds,
            "device": str(device),
            "initial_gaussians": args.initial_gaussians,
            "max_gaussians": args.max_gaussians,
            "patch_size": args.patch_size,
            "lpips_enabled": lpips_model is not None,
        },
        "runs": runs,
        "aggregate": aggregate_results(runs),
    }
    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics["aggregate"], indent=2))


if __name__ == "__main__":
    main()

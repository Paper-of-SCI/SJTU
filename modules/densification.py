"""Composable adaptive densification helpers.

The controller mutates a GaussianModel only when the caller invokes ``update``.
It does not own the training loop.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from modules.gaussian_model import GaussianModel, PARAMETER_NAMES, quats_to_rotmats
from modules.renderer import RenderOutput


@dataclass
class DensificationStats:
    cloned: int = 0
    split: int = 0
    pruned: int = 0
    opacity_reset: bool = False
    total: int = 0
    densified: bool = False
    high_grad: int = 0
    grad_mean: float = 0.0
    grad_max: float = 0.0
    reallocated: int = 0
    patch_detail_mean: float = 0.0
    patch_detail_max: float = 0.0


@dataclass
class DensificationConfig:
    start_step: int = 500
    stop_step: int = 15_000
    interval: int = 100
    grad_threshold: float = 2.0e-4
    scene_extent: float = 1.0
    percent_dense: float = 0.01
    min_opacity: float = 0.005
    max_screen_radius: Optional[float] = None
    num_splits: int = 2
    opacity_reset_interval: int = 3000
    reset_opacity: float = 0.01
    use_absgrad: bool = True


@dataclass
class PatchGuidedDensificationConfig:
    """Patch scoring settings layered on top of the standard 3DGS contract."""

    densification: DensificationConfig = field(default_factory=DensificationConfig)
    patch_size: int = 16
    edge_weight: float = 0.75
    detail_lambda: float = 2.0
    reallocate_fraction: float = 0.0
    clone_jitter_scale: float = 0.05
    eps: float = 1.0e-6


class DensificationController:
    """Stateless-enough densification controller for caller-owned loops."""

    def __init__(self, config: DensificationConfig | None = None) -> None:
        self.config = config or DensificationConfig()

    def update(
        self,
        model: GaussianModel,
        render_output: RenderOutput,
        optimizer: torch.optim.Optimizer,
        step: int,
    ) -> DensificationStats:
        """Accumulate gradients and optionally clone/split/prune."""
        visibility = _visibility_from_output(render_output, model.num_gaussians)
        if render_output.means2d is not None:
            model.accumulate_gradient_stats(
                render_output.means2d,
                visibility=visibility,
                use_absgrad=self.config.use_absgrad,
                indices=render_output.metadata.get("gaussian_ids"),
            )

        stats = DensificationStats(total=model.num_gaussians)
        in_range = self.config.start_step <= step < self.config.stop_step
        if in_range and self.config.interval > 0 and step % self.config.interval == 0:
            stats = self._densify(model, optimizer, render_output)

        if self.config.opacity_reset_interval > 0 and step > 0 and step % self.config.opacity_reset_interval == 0:
            model.reset_opacities(self.config.reset_opacity)
            _zero_optimizer_state_for(model, optimizer, "logit_opacities")
            stats.opacity_reset = True
            stats.total = model.num_gaussians
        return stats

    def _densify(self, model: GaussianModel, optimizer: torch.optim.Optimizer, render_output: RenderOutput) -> DensificationStats:
        if model.num_gaussians == 0:
            return DensificationStats(total=0)

        avg_grad = model.gradient_accum / model.gradient_count.clamp_min(1.0)
        high_grad = avg_grad >= self.config.grad_threshold
        seen = model.gradient_count > 0
        seen_grad = avg_grad[seen]
        grad_mean = float(seen_grad.mean().item()) if seen_grad.numel() > 0 else 0.0
        grad_max = float(seen_grad.max().item()) if seen_grad.numel() > 0 else 0.0
        max_scale = model.scales.detach().max(dim=-1).values
        dense_threshold = self.config.scene_extent * self.config.percent_dense
        clone_mask = high_grad & (max_scale <= dense_threshold)
        split_mask = high_grad & (max_scale > dense_threshold)

        cloned = int(clone_mask.sum().item())
        split = int(split_mask.sum().item()) * self.config.num_splits

        if cloned > 0:
            old = _snapshot_optimizer(model, optimizer)
            old_n = model.num_gaussians
            model.clone(clone_mask)
            _patch_optimizer_append(model, optimizer, old, old_n)

        if split > 0:
            if split_mask.numel() < model.num_gaussians:
                split_mask = torch.cat(
                    [
                        split_mask,
                        torch.zeros(model.num_gaussians - split_mask.numel(), dtype=torch.bool, device=split_mask.device),
                    ],
                    dim=0,
                )
            old = _snapshot_optimizer(model, optimizer)
            added, keep_after_append = model.split(split_mask, num_splits=self.config.num_splits)
            _patch_optimizer_append_and_prune(model, optimizer, old, keep_after_append, added)

        prune_mask = model.opacities.detach() < self.config.min_opacity
        if self.config.max_screen_radius is not None and render_output.radii is not None:
            radii = render_output.radii.reshape(-1)
            if radii.numel() == model.num_gaussians:
                prune_mask = prune_mask | (radii.to(model.means.device) > self.config.max_screen_radius)

        pruned = int(prune_mask.sum().item())
        if pruned > 0:
            old = _snapshot_optimizer(model, optimizer)
            keep = model.prune(prune_mask)
            _patch_optimizer_prune(model, optimizer, old, keep)

        model.clear_gradient_stats()
        return DensificationStats(
            cloned=cloned,
            split=split,
            pruned=pruned,
            total=model.num_gaussians,
            densified=True,
            high_grad=int(high_grad.sum().item()),
            grad_mean=grad_mean,
            grad_max=grad_max,
        )


class PatchGuidedDensificationController:
    """Patch error/edge guided densification inside the existing 3DGS boundary."""

    def __init__(self, config: PatchGuidedDensificationConfig | None = None) -> None:
        self.config = config or PatchGuidedDensificationConfig()
        self._detail_accum: Optional[Tensor] = None
        self._detail_count: Optional[Tensor] = None

    def update(
        self,
        model: GaussianModel,
        render_output: RenderOutput,
        optimizer: torch.optim.Optimizer,
        step: int,
        gt_image: Tensor,
    ) -> DensificationStats:
        """Accumulate standard gradients plus patch detail, then optionally mutate Gaussians."""
        base = self.config.densification
        visibility = _visibility_from_output(render_output, model.num_gaussians)
        if render_output.means2d is not None:
            model.accumulate_gradient_stats(
                render_output.means2d,
                visibility=visibility,
                use_absgrad=base.use_absgrad,
                indices=render_output.metadata.get("gaussian_ids"),
            )
            with torch.no_grad():
                self._ensure_buffers(model)
                detail_map = _compute_patch_detail(
                    render_output.image.detach(),
                    gt_image.detach(),
                    patch_size=self.config.patch_size,
                    edge_weight=self.config.edge_weight,
                    eps=self.config.eps,
                )
                detail_values, detail_counts = _project_patch_detail_to_gaussians(
                    detail_map,
                    render_output.means2d.detach(),
                    render_output.metadata.get("gaussian_ids"),
                    render_output.radii,
                    gaussian_count=model.num_gaussians,
                    image_height=int(gt_image.shape[-2]),
                    image_width=int(gt_image.shape[-1]),
                    patch_size=self.config.patch_size,
                )
                self._detail_accum.add_(detail_values.to(self._detail_accum.device))
                self._detail_count.add_(detail_counts.to(self._detail_count.device))

        stats = DensificationStats(total=model.num_gaussians)
        in_range = base.start_step <= step < base.stop_step
        if in_range and base.interval > 0 and step % base.interval == 0:
            with torch.no_grad():
                stats = self._densify(model, optimizer, render_output)

        if base.opacity_reset_interval > 0 and step > 0 and step % base.opacity_reset_interval == 0:
            model.reset_opacities(base.reset_opacity)
            _zero_optimizer_state_for(model, optimizer, "logit_opacities")
            stats.opacity_reset = True
            stats.total = model.num_gaussians
        return stats

    def _ensure_buffers(self, model: GaussianModel) -> None:
        if self._detail_accum is not None and self._detail_accum.numel() == model.num_gaussians:
            return
        device = model.means.device
        self._detail_accum = torch.zeros(model.num_gaussians, device=device)
        self._detail_count = torch.zeros(model.num_gaussians, device=device)

    def _clear_buffers(self, model: GaussianModel) -> None:
        device = model.means.device
        self._detail_accum = torch.zeros(model.num_gaussians, device=device)
        self._detail_count = torch.zeros(model.num_gaussians, device=device)

    def _average_detail(self, model: GaussianModel) -> Tensor:
        self._ensure_buffers(model)
        assert self._detail_accum is not None and self._detail_count is not None
        return self._detail_accum / self._detail_count.clamp_min(1.0)

    def _densify(self, model: GaussianModel, optimizer: torch.optim.Optimizer, render_output: RenderOutput) -> DensificationStats:
        base = self.config.densification
        if model.num_gaussians == 0:
            return DensificationStats(total=0)

        avg_grad = model.gradient_accum / model.gradient_count.clamp_min(1.0)
        avg_detail = self._average_detail(model)
        adjusted_grad = avg_grad * (1.0 + self.config.detail_lambda * avg_detail)
        high_grad = adjusted_grad >= base.grad_threshold
        seen = model.gradient_count > 0
        seen_adjusted = adjusted_grad[seen]
        grad_mean = float(seen_adjusted.mean().item()) if seen_adjusted.numel() > 0 else 0.0
        grad_max = float(seen_adjusted.max().item()) if seen_adjusted.numel() > 0 else 0.0
        detail_seen = self._detail_count is not None and bool((self._detail_count > 0).any().item())
        detail_values = avg_detail[self._detail_count > 0] if detail_seen and self._detail_count is not None else avg_detail[:0]
        detail_mean = float(detail_values.mean().item()) if detail_values.numel() > 0 else 0.0
        detail_max = float(detail_values.max().item()) if detail_values.numel() > 0 else 0.0

        max_scale = model.scales.detach().max(dim=-1).values
        dense_threshold = base.scene_extent * base.percent_dense
        prune_mask = _standard_prune_mask(model, render_output, base)
        clone_mask = high_grad & (max_scale <= dense_threshold) & ~prune_mask
        split_mask = high_grad & (max_scale > dense_threshold) & ~prune_mask

        reallocated = 0
        if self.config.reallocate_fraction > 0.0:
            extra_prune, extra_clone, extra_split = _reallocation_masks(
                model=model,
                adjusted_grad=adjusted_grad.detach(),
                avg_detail=avg_detail.detach(),
                detail_count=self._detail_count.detach() if self._detail_count is not None else torch.zeros_like(avg_detail),
                clone_mask=clone_mask,
                split_mask=split_mask,
                prune_mask=prune_mask,
                max_scale=max_scale,
                dense_threshold=dense_threshold,
                reallocate_fraction=self.config.reallocate_fraction,
                num_splits=base.num_splits,
                eps=self.config.eps,
            )
            reallocated = int(extra_prune.sum().item())
            prune_mask = prune_mask | extra_prune
            clone_mask = (clone_mask | extra_clone) & ~prune_mask
            split_mask = (split_mask | extra_split) & ~prune_mask

        cloned = int(clone_mask.sum().item())
        split = int(split_mask.sum().item()) * base.num_splits
        pruned = int(prune_mask.sum().item())

        if cloned > 0 or split > 0 or pruned > 0:
            old = _snapshot_optimizer(model, optimizer)
            clone_tensors = _build_clone_tensors(model, clone_mask, self.config.clone_jitter_scale)
            split_tensors = _build_split_tensors(model, split_mask, base.num_splits)
            child_tensors = _concat_child_tensors(model, [clone_tensors, split_tensors])
            keep_old = ~(split_mask | prune_mask)
            _apply_append_and_prune_tensors(model, optimizer, old, keep_old, child_tensors)

        model.clear_gradient_stats()
        self._clear_buffers(model)
        return DensificationStats(
            cloned=cloned,
            split=split,
            pruned=pruned,
            total=model.num_gaussians,
            densified=True,
            high_grad=int(high_grad.sum().item()),
            grad_mean=grad_mean,
            grad_max=grad_max,
            reallocated=reallocated,
            patch_detail_mean=detail_mean,
            patch_detail_max=detail_max,
        )


def _visibility_from_output(output: RenderOutput, count: int) -> Optional[Tensor]:
    if output.radii is None:
        return None
    radii = output.radii.reshape(-1)
    if radii.numel() != count:
        return None
    return radii > 0


def _standard_prune_mask(model: GaussianModel, render_output: RenderOutput, config: DensificationConfig) -> Tensor:
    prune_mask = model.opacities.detach() < config.min_opacity
    if config.max_screen_radius is not None and render_output.radii is not None:
        radii = render_output.radii.reshape(-1)
        if radii.numel() == model.num_gaussians:
            prune_mask = prune_mask | (radii.to(model.means.device) > config.max_screen_radius)
    return prune_mask


def _compute_patch_detail(render_image: Tensor, gt_image: Tensor, patch_size: int, edge_weight: float, eps: float = 1.0e-6) -> Tensor:
    """Return normalized patch detail map with shape ``(patch_h, patch_w)``."""
    if render_image.shape != gt_image.shape:
        raise ValueError("render_image and gt_image must have the same CHW shape")
    if render_image.ndim != 3:
        raise ValueError("render_image and gt_image must be CHW tensors")
    patch = max(int(patch_size), 1)
    render = render_image.detach().clamp(0.0, 1.0)
    gt = gt_image.detach().clamp(0.0, 1.0)
    pixel_error = (render - gt).abs().mean(dim=0, keepdim=True).unsqueeze(0)
    patch_error = _patch_pool(pixel_error, patch)
    patch_edge = _patch_pool(_sobel_edge(gt), patch)
    patch_error = _normalize_minmax(patch_error, eps)
    patch_edge = _normalize_minmax(patch_edge, eps)
    detail = patch_error * (1.0 + float(edge_weight) * patch_edge)
    return _normalize_minmax(detail.squeeze(0).squeeze(0), eps)


def _patch_pool(image_bchw: Tensor, patch_size: int) -> Tensor:
    return F.avg_pool2d(
        image_bchw,
        kernel_size=patch_size,
        stride=patch_size,
        ceil_mode=True,
        count_include_pad=False,
    )


def _sobel_edge(image_chw: Tensor) -> Tensor:
    gray = image_chw.mean(dim=0, keepdim=True).unsqueeze(0)
    kernel_x = gray.new_tensor([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]]).view(1, 1, 3, 3)
    kernel_y = gray.new_tensor([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]]).view(1, 1, 3, 3)
    padded = F.pad(gray, (1, 1, 1, 1), mode="replicate")
    grad_x = F.conv2d(padded, kernel_x)
    grad_y = F.conv2d(padded, kernel_y)
    return torch.sqrt(grad_x.square() + grad_y.square()).clamp_min(0.0)


def _project_patch_detail_to_gaussians(
    detail_map: Tensor,
    means2d: Tensor,
    gaussian_ids: Optional[Tensor],
    radii: Optional[Tensor],
    gaussian_count: int,
    image_height: int,
    image_width: int,
    patch_size: int,
) -> tuple[Tensor, Tensor]:
    device = means2d.device
    values = torch.zeros(gaussian_count, device=device)
    counts = torch.zeros(gaussian_count, device=device)
    if gaussian_count == 0:
        return values, counts

    xy, indices = _flatten_projection(means2d, gaussian_ids, gaussian_count)
    if xy is None or indices is None:
        return values, counts

    indices = indices.to(device=device, dtype=torch.long)
    xy = xy.to(device=device, dtype=torch.float32)
    valid = (indices >= 0) & (indices < gaussian_count)
    valid = valid & torch.isfinite(xy).all(dim=-1)
    valid = valid & (xy[:, 0] >= 0.0) & (xy[:, 0] < float(image_width)) & (xy[:, 1] >= 0.0) & (xy[:, 1] < float(image_height))
    radius_visible = _projection_visibility_from_radii(radii, indices, xy.shape[0], gaussian_count, device)
    if radius_visible is not None:
        valid = valid & radius_visible
    if not bool(valid.any().item()):
        return values, counts

    patch = max(int(patch_size), 1)
    patch_h, patch_w = int(detail_map.shape[-2]), int(detail_map.shape[-1])
    px = torch.div(xy[:, 0].floor().to(torch.long), patch, rounding_mode="floor").clamp(0, patch_w - 1)
    py = torch.div(xy[:, 1].floor().to(torch.long), patch, rounding_mode="floor").clamp(0, patch_h - 1)
    patch_values = detail_map.to(device=device).reshape(-1)[py * patch_w + px]
    valid_indices = indices[valid]
    values.index_add_(0, valid_indices, patch_values[valid].to(values.dtype))
    counts.index_add_(0, valid_indices, torch.ones_like(patch_values[valid], dtype=counts.dtype))
    return values, counts


def _flatten_projection(means2d: Tensor, gaussian_ids: Optional[Tensor], gaussian_count: int) -> tuple[Optional[Tensor], Optional[Tensor]]:
    xy = means2d.reshape(-1, means2d.shape[-1])[:, :2]
    if gaussian_ids is not None:
        indices = gaussian_ids.reshape(-1)
        if indices.numel() != xy.shape[0]:
            return None, None
        return xy, indices
    if xy.shape[0] == gaussian_count:
        return xy, torch.arange(gaussian_count, device=xy.device)
    if gaussian_count > 0 and xy.shape[0] % gaussian_count == 0:
        repeats = xy.shape[0] // gaussian_count
        return xy, torch.arange(gaussian_count, device=xy.device).repeat(repeats)
    return None, None


def _projection_visibility_from_radii(
    radii: Optional[Tensor],
    indices: Tensor,
    projection_count: int,
    gaussian_count: int,
    device: torch.device,
) -> Optional[Tensor]:
    if radii is None:
        return None
    flat = radii.reshape(-1).to(device=device)
    if flat.numel() == projection_count:
        return flat > 0
    if flat.numel() == gaussian_count:
        safe_indices = indices.clamp(0, gaussian_count - 1)
        return flat[safe_indices.to(flat.device)].to(device=device) > 0
    return None


def _normalize_minmax(values: Tensor, eps: float = 1.0e-6) -> Tensor:
    if values.numel() == 0:
        return values
    min_value = values.amin()
    max_value = values.amax()
    denom = max_value - min_value
    normalized = (values - min_value) / denom.clamp_min(eps)
    return torch.where(denom > eps, normalized, torch.zeros_like(values))


def _topk_mask(score: Tensor, candidate: Tensor, count: int) -> Tensor:
    mask = torch.zeros_like(candidate, dtype=torch.bool)
    k = min(max(int(count), 0), int(candidate.sum().item()))
    if k <= 0:
        return mask
    masked_score = score.clone()
    masked_score[~candidate] = -float("inf")
    selected = torch.topk(masked_score, k=k, largest=True).indices
    mask[selected] = True
    return mask


def _reallocation_masks(
    model: GaussianModel,
    adjusted_grad: Tensor,
    avg_detail: Tensor,
    detail_count: Tensor,
    clone_mask: Tensor,
    split_mask: Tensor,
    prune_mask: Tensor,
    max_scale: Tensor,
    dense_threshold: float,
    reallocate_fraction: float,
    num_splits: int,
    eps: float,
) -> tuple[Tensor, Tensor, Tensor]:
    count = model.num_gaussians
    empty = torch.zeros(count, dtype=torch.bool, device=model.means.device)
    budget = min(int(round(count * max(float(reallocate_fraction), 0.0))), count)
    candidate = ~(clone_mask | split_mask | prune_mask)
    budget = min(budget, int(candidate.sum().item()))
    if budget <= 0:
        return empty, empty, empty

    detail_norm = _normalize_minmax(avg_detail, eps)
    seen_norm = _normalize_minmax(detail_count, eps)
    opacity = model.opacities.detach().clamp(0.0, 1.0)
    low_score = (1.0 - detail_norm) * 0.55 + (1.0 - seen_norm) * 0.20 + (1.0 - opacity) * 0.25
    extra_prune = _topk_mask(low_score, candidate, budget)
    reallocate_budget = int(extra_prune.sum().item())
    if reallocate_budget <= 0:
        return extra_prune, empty, empty

    source_candidate = ~(extra_prune | clone_mask | split_mask | prune_mask)
    source_score = adjusted_grad * (1.0 + avg_detail)
    extra_clone = _topk_mask(source_score, source_candidate & (max_scale <= dense_threshold), reallocate_budget)
    remaining = reallocate_budget - int(extra_clone.sum().item())

    extra_split = empty.clone()
    split_parent_budget = remaining // max(int(num_splits), 1)
    if split_parent_budget > 0:
        extra_split = _topk_mask(source_score, source_candidate & (max_scale > dense_threshold) & ~extra_clone, split_parent_budget)
        remaining -= int(extra_split.sum().item()) * max(int(num_splits), 1)

    if remaining > 0:
        fallback_clone = _topk_mask(source_score, source_candidate & ~extra_clone & ~extra_split, remaining)
        extra_clone = extra_clone | fallback_clone
    return extra_prune, extra_clone, extra_split


def _build_clone_tensors(model: GaussianModel, mask: Tensor, jitter_scale: float) -> Dict[str, Tensor]:
    selected = mask.nonzero(as_tuple=True)[0]
    if selected.numel() == 0:
        return _empty_child_tensors(model)
    tensors = {name: getattr(model, name).detach()[selected].clone() for name in PARAMETER_NAMES}
    if jitter_scale > 0.0:
        rotations = quats_to_rotmats(model.normalized_quats.detach()[selected])
        local_offsets = torch.randn_like(tensors["means"]) * model.scales.detach()[selected] * float(jitter_scale)
        tensors["means"] = tensors["means"] + torch.matmul(rotations, local_offsets[..., None]).squeeze(-1)
    return tensors


def _build_split_tensors(model: GaussianModel, mask: Tensor, num_splits: int, scale_shrink: float = 1.6) -> Dict[str, Tensor]:
    selected = mask.nonzero(as_tuple=True)[0]
    splits = max(int(num_splits), 1)
    if selected.numel() == 0:
        return _empty_child_tensors(model)

    means = model.means.detach()[selected]
    log_scales = model.log_scales.detach()[selected]
    rotations = quats_to_rotmats(model.normalized_quats.detach()[selected])
    scales = log_scales.exp()
    samples = torch.randn(selected.numel(), splits, 3, device=model.means.device, dtype=model.means.dtype)
    local_offsets = samples * scales[:, None, :]
    offsets = torch.matmul(rotations[:, None, :, :], local_offsets[..., None]).squeeze(-1)
    return {
        "means": (means[:, None, :] + offsets).reshape(-1, 3),
        "log_scales": (log_scales - math.log(scale_shrink))[:, None, :].expand(-1, splits, -1).reshape(-1, 3),
        "quats": model.quats.detach()[selected][:, None, :].expand(-1, splits, -1).reshape(-1, 4),
        "logit_opacities": model.logit_opacities.detach()[selected][:, None, :].expand(-1, splits, -1).reshape(-1, 1),
        "features_dc": model.features_dc.detach()[selected][:, None, :, :].expand(-1, splits, -1, -1).reshape(-1, 1, 3),
        "features_rest": model.features_rest.detach()[selected][:, None, :, :].expand(-1, splits, -1, -1).reshape(
            -1, model.features_rest.shape[1], 3
        ),
    }


def _empty_child_tensors(model: GaussianModel) -> Dict[str, Tensor]:
    return {name: getattr(model, name).detach()[:0].clone() for name in PARAMETER_NAMES}


def _concat_child_tensors(model: GaussianModel, groups: list[Dict[str, Tensor]]) -> Dict[str, Tensor]:
    tensors: Dict[str, Tensor] = {}
    for name in PARAMETER_NAMES:
        parts = [group[name] for group in groups if group[name].shape[0] > 0]
        tensors[name] = torch.cat(parts, dim=0) if parts else getattr(model, name).detach()[:0].clone()
    return tensors


def _apply_append_and_prune_tensors(
    model: GaussianModel,
    optimizer: torch.optim.Optimizer,
    old: Dict[str, dict],
    keep_old: Tensor,
    child_tensors: Dict[str, Tensor],
) -> None:
    keep_old = keep_old.to(model.means.device, dtype=torch.bool)
    added = int(child_tensors["means"].shape[0])
    updated = {}
    for name in PARAMETER_NAMES:
        base = getattr(model, name).detach()
        extra = child_tensors[name].to(device=base.device, dtype=base.dtype).detach()
        updated[name] = torch.cat([base[keep_old], extra], dim=0)
    keep_after_append = torch.cat(
        [keep_old, torch.ones(added, dtype=torch.bool, device=keep_old.device)],
        dim=0,
    )
    model.replace_tensors(updated)
    for name in PARAMETER_NAMES:
        state = _expanded_state(old[name]["state"], added)
        state = _masked_state(state, keep_after_append)
        _replace_group_param(model, optimizer, name, state)


def _snapshot_optimizer(model: GaussianModel, optimizer: torch.optim.Optimizer) -> Dict[str, dict]:
    params = model.parameter_map()
    snapshot: Dict[str, dict] = {}
    for name, param in params.items():
        state = optimizer.state.get(param, {})
        snapshot[name] = {
            "param": param,
            "state": {key: value.detach().clone() if torch.is_tensor(value) else value for key, value in state.items()},
        }
    return snapshot


def _replace_group_param(model: GaussianModel, optimizer: torch.optim.Optimizer, name: str, state: dict) -> None:
    param = getattr(model, name)
    for group in optimizer.param_groups:
        if group.get("name") == name:
            for old_param in group["params"]:
                if old_param is not param:
                    optimizer.state.pop(old_param, None)
            group["params"] = [param]
            break
    optimizer.state[param] = state


def _patch_optimizer_append(model: GaussianModel, optimizer: torch.optim.Optimizer, old: Dict[str, dict], old_n: int) -> None:
    added = model.num_gaussians - old_n
    for name in PARAMETER_NAMES:
        state = _expanded_state(old[name]["state"], added)
        _replace_group_param(model, optimizer, name, state)


def _patch_optimizer_append_and_prune(
    model: GaussianModel,
    optimizer: torch.optim.Optimizer,
    old: Dict[str, dict],
    keep_after_append: Tensor,
    added: int,
) -> None:
    keep_after_append = keep_after_append.to(model.means.device)
    for name in PARAMETER_NAMES:
        state = _expanded_state(old[name]["state"], added)
        state = _masked_state(state, keep_after_append)
        _replace_group_param(model, optimizer, name, state)


def _patch_optimizer_prune(model: GaussianModel, optimizer: torch.optim.Optimizer, old: Dict[str, dict], keep: Tensor) -> None:
    keep = keep.to(model.means.device)
    for name in PARAMETER_NAMES:
        state = _masked_state(old[name]["state"], keep)
        _replace_group_param(model, optimizer, name, state)


def _expanded_state(state: dict, added: int) -> dict:
    out = {}
    for key, value in state.items():
        if torch.is_tensor(value) and value.ndim > 0 and added > 0:
            pad = torch.zeros((added, *value.shape[1:]), dtype=value.dtype, device=value.device)
            out[key] = torch.cat([value, pad], dim=0)
        else:
            out[key] = value
    return out


def _masked_state(state: dict, keep: Tensor) -> dict:
    out = {}
    for key, value in state.items():
        if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == keep.numel():
            out[key] = value[keep.to(value.device)]
        else:
            out[key] = value
    return out


def _zero_optimizer_state_for(model: GaussianModel, optimizer: torch.optim.Optimizer, name: str) -> None:
    param = getattr(model, name)
    state = optimizer.state.get(param, {})
    for value in state.values():
        if torch.is_tensor(value):
            value.zero_()

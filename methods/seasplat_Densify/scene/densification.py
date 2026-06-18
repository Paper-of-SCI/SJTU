"""Patch-guided densification for the SeaSplat training path."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


DENSIFICATION_MODES = ("standard_3dgs", "patch_guided")


@dataclass
class PatchGuidedDensificationConfig:
    patch_size: int = 32
    edge_weight: float = 0.75
    detail_lambda: float = 2.0
    clone_jitter_scale: float = 0.05
    eps: float = 1.0e-6


@dataclass
class PatchGuidedDensificationStats:
    cloned: int = 0
    split: int = 0
    pruned: int = 0
    total: int = 0
    densified: bool = False
    high_grad: int = 0
    grad_mean: float = 0.0
    grad_max: float = 0.0
    patch_detail_mean: float = 0.0
    patch_detail_max: float = 0.0


def normalize_densification_mode(mode: str) -> str:
    normalized = str(mode).strip().lower()
    if normalized == "standard":
        normalized = "standard_3dgs"
    if normalized not in DENSIFICATION_MODES:
        raise ValueError(f"Unsupported densification_mode '{mode}'. Expected one of {DENSIFICATION_MODES}.")
    return normalized


class PatchGuidedDensificationController:
    """Accumulate patch detail and call GaussianModel's densify boundary."""

    def __init__(self, config: PatchGuidedDensificationConfig | None = None):
        self.config = config or PatchGuidedDensificationConfig()
        self._detail_accum = None
        self._detail_count = None

    def update(
        self,
        *,
        gaussians,
        viewpoint_camera,
        rendered_image,
        gt_image,
        viewspace_point_tensor,
        visibility_filter,
        radii,
        iteration: int,
        densification_interval: int,
        densify_from_iter: int,
        grad_threshold: float,
        min_opacity: float,
        scene_extent: float,
        max_screen_size,
    ) -> PatchGuidedDensificationStats:
        gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
        gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
        self._accumulate_patch_detail(gaussians, viewpoint_camera, rendered_image, gt_image, radii)

        if densification_interval <= 0 or iteration % densification_interval != 0 or iteration <= densify_from_iter:
            return PatchGuidedDensificationStats(total=gaussians.get_xyz.shape[0])

        score, detail = self._adjusted_gradient_score(gaussians)
        stats_dict = gaussians.densify_and_prune_with_scores(
            score,
            grad_threshold,
            min_opacity,
            scene_extent,
            max_screen_size,
            clone_jitter_scale=self.config.clone_jitter_scale,
        )
        detail_values = detail[self._detail_count > 0] if self._detail_count is not None else detail[:0]
        detail_mean = float(detail_values.mean().item()) if detail_values.numel() > 0 else 0.0
        detail_max = float(detail_values.max().item()) if detail_values.numel() > 0 else 0.0
        self._reset_buffers(gaussians)
        return PatchGuidedDensificationStats(
            cloned=stats_dict["cloned"],
            split=stats_dict["split"],
            pruned=stats_dict["pruned"],
            total=stats_dict["total"],
            densified=True,
            high_grad=stats_dict["high_grad"],
            grad_mean=stats_dict["grad_mean"],
            grad_max=stats_dict["grad_max"],
            patch_detail_mean=detail_mean,
            patch_detail_max=detail_max,
        )

    def _ensure_buffers(self, gaussians) -> None:
        count = gaussians.get_xyz.shape[0]
        device = gaussians.get_xyz.device
        if self._detail_accum is not None and self._detail_accum.numel() == count:
            return
        self._detail_accum = torch.zeros(count, device=device)
        self._detail_count = torch.zeros(count, device=device)

    def _reset_buffers(self, gaussians) -> None:
        self._detail_accum = None
        self._detail_count = None
        self._ensure_buffers(gaussians)

    def _accumulate_patch_detail(self, gaussians, viewpoint_camera, rendered_image, gt_image, radii) -> None:
        self._ensure_buffers(gaussians)
        detail_map = compute_patch_detail(
            rendered_image.detach(),
            gt_image.detach(),
            patch_size=self.config.patch_size,
            edge_weight=self.config.edge_weight,
            eps=self.config.eps,
        )
        values, counts = project_patch_detail_to_gaussians(
            detail_map,
            gaussians.get_xyz.detach(),
            viewpoint_camera,
            radii,
            patch_size=self.config.patch_size,
        )
        self._detail_accum.add_(values.to(self._detail_accum.device))
        self._detail_count.add_(counts.to(self._detail_count.device))

    def _adjusted_gradient_score(self, gaussians):
        self._ensure_buffers(gaussians)
        avg_grad = gaussians.xyz_gradient_accum / gaussians.denom.clamp_min(1.0)
        avg_grad = avg_grad.squeeze(-1)
        avg_detail = self._detail_accum / self._detail_count.clamp_min(1.0)
        score = avg_grad * (1.0 + float(self.config.detail_lambda) * avg_detail)
        return score, avg_detail


def compute_patch_detail(rendered_image, gt_image, patch_size: int, edge_weight: float, eps: float) -> torch.Tensor:
    rendered = _as_chw(rendered_image).clamp(0.0, 1.0)
    gt = _as_chw(gt_image).clamp(0.0, 1.0)
    if rendered.shape != gt.shape:
        raise ValueError("rendered_image and gt_image must have matching CHW shape for patch densification.")

    patch = max(int(patch_size), 1)
    pixel_error = (rendered - gt).abs().mean(dim=0, keepdim=True).unsqueeze(0)
    patch_error = _normalize_minmax(_patch_pool(pixel_error, patch), eps)
    patch_edge = _normalize_minmax(_patch_pool(_sobel_edge(gt), patch), eps)
    detail = patch_error * (1.0 + float(edge_weight) * patch_edge)
    return _normalize_minmax(detail, eps).squeeze(0).squeeze(0)


def project_patch_detail_to_gaussians(detail_map, xyz, viewpoint_camera, radii, patch_size: int):
    count = xyz.shape[0]
    values = torch.zeros(count, dtype=xyz.dtype, device=xyz.device)
    counts = torch.zeros(count, dtype=xyz.dtype, device=xyz.device)
    if count == 0:
        return values, counts

    xy, valid = project_gaussian_centers_to_pixels(xyz, viewpoint_camera)
    if radii is not None and radii.numel() == count:
        valid = valid & (radii.reshape(-1).to(device=xyz.device) > 0)
    if not bool(valid.any().item()):
        return values, counts

    patch = max(int(patch_size), 1)
    patch_h, patch_w = int(detail_map.shape[-2]), int(detail_map.shape[-1])
    px = torch.div(xy[:, 0].floor().to(torch.long), patch, rounding_mode="floor").clamp(0, patch_w - 1)
    py = torch.div(xy[:, 1].floor().to(torch.long), patch, rounding_mode="floor").clamp(0, patch_h - 1)
    patch_values = detail_map.to(device=xyz.device, dtype=xyz.dtype).reshape(-1)[py * patch_w + px]
    values[valid] = patch_values[valid]
    counts[valid] = 1.0
    return values, counts


def project_gaussian_centers_to_pixels(xyz, viewpoint_camera):
    ones = torch.ones((xyz.shape[0], 1), dtype=xyz.dtype, device=xyz.device)
    xyz_h = torch.cat([xyz, ones], dim=-1)
    clip = xyz_h @ viewpoint_camera.full_proj_transform.to(device=xyz.device, dtype=xyz.dtype)
    w = clip[:, 3]
    eps = xyz.new_tensor(1.0e-8)
    safe_w = torch.where(w.abs() > eps, w, torch.where(w >= 0.0, eps, -eps))
    ndc = clip[:, :3] / safe_w[:, None]
    width = int(viewpoint_camera.image_width)
    height = int(viewpoint_camera.image_height)
    x = (ndc[:, 0] + 1.0) * 0.5 * float(width)
    y = (1.0 - ndc[:, 1]) * 0.5 * float(height)
    xy = torch.stack([x, y], dim=-1)
    valid = torch.isfinite(xy).all(dim=-1)
    valid = valid & torch.isfinite(w) & (w > 0.0)
    valid = valid & (x >= 0.0) & (x < float(width)) & (y >= 0.0) & (y < float(height))
    return xy, valid


def _as_chw(image):
    if image.ndim == 4 and image.shape[0] == 1:
        image = image.squeeze(0)
    if image.ndim != 3:
        raise ValueError("patch densification expects CHW or 1CHW image tensors.")
    return image.detach()


def _patch_pool(image_bchw, patch_size: int):
    return F.avg_pool2d(
        image_bchw,
        kernel_size=patch_size,
        stride=patch_size,
        ceil_mode=True,
        count_include_pad=False,
    )


def _sobel_edge(image_chw):
    gray = image_chw.mean(dim=0, keepdim=True).unsqueeze(0)
    kernel_x = gray.new_tensor([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]]).view(1, 1, 3, 3)
    kernel_y = gray.new_tensor([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]]).view(1, 1, 3, 3)
    padded = F.pad(gray, (1, 1, 1, 1), mode="replicate")
    grad_x = F.conv2d(padded, kernel_x)
    grad_y = F.conv2d(padded, kernel_y)
    return torch.sqrt(grad_x.square() + grad_y.square()).clamp_min(0.0)


def _normalize_minmax(values, eps: float):
    if values.numel() == 0:
        return values
    min_value = values.amin()
    max_value = values.amax()
    denom = max_value - min_value
    normalized = (values - min_value) / denom.clamp_min(eps)
    return torch.where(denom > eps, normalized, torch.zeros_like(values))

"""Underwater rendering wrapper built on top of the plain 3DGS renderer."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
from torch import Tensor

from modules.camera import Camera
from modules.gaussian_model import GaussianModel
from modules.medium_field import MediumField
from modules.renderer import GaussianRenderer


@dataclass
class MediumRenderConfig:
    """Controls ray integration without exposing renderer internals."""

    far_distance: float
    chunk_pixels: int = 65536
    alpha_threshold: float = 1.0e-3
    max_density: float = 10.0
    max_optical_depth: float = 80.0
    eps: float = 1.0e-6


@dataclass
class MediumRenderOutput:
    """Renderer output compatible with densification plus medium components."""

    image: Tensor
    alpha: Tensor
    depth: Optional[Tensor]
    radii: Optional[Tensor]
    means2d: Optional[Tensor]
    metadata: Dict[str, Any]
    rgb_clear: Tensor
    rgb_object: Tensor
    rgb_medium: Tensor
    medium_attn: Tensor
    medium_bs: Tensor
    medium_rgb: Tensor
    d_ref: Tensor


class MediumRenderer:
    """Compose plain 3DGS rendering with a learned low-capacity medium field."""

    def __init__(
        self,
        base_renderer: GaussianRenderer,
        medium_field: MediumField,
        config: MediumRenderConfig,
    ) -> None:
        self.base_renderer = base_renderer
        self.medium_field = medium_field
        self.config = config

    def render(self, model: GaussianModel, camera: Camera) -> MediumRenderOutput:
        base = self.base_renderer.render(model, camera, render_mode="RGB+ED")
        if base.depth is None:
            raise RuntimeError("MediumRenderer requires depth; base renderer did not return depth.")

        rgb_clear = _remove_background(base.image, base.alpha, self.base_renderer.background, camera.device)
        d_ref = _reference_depth(base.depth, base.alpha, camera, self.config)
        ray_dirs = _camera_ray_directions(camera)
        medium_attn, medium_bs, density_l1, density_smooth = self._integrate_medium(camera.camera_center, ray_dirs, d_ref)

        medium_rgb_hwc = self.medium_field.medium_rgb.to(rgb_clear).view(1, 1, 3).expand(camera.height, camera.width, 3)
        d_hwc = d_ref.permute(1, 2, 0)
        max_optical_depth = max(float(self.config.max_optical_depth), float(self.config.eps))
        trans_attn = torch.exp(torch.clamp(-medium_attn * d_hwc, min=-max_optical_depth, max=0.0))
        trans_bs = torch.exp(torch.clamp(-medium_bs * d_hwc, min=-max_optical_depth, max=0.0))

        rgb_clear_hwc = rgb_clear.permute(1, 2, 0)
        rgb_object_hwc = rgb_clear_hwc * trans_attn
        rgb_medium_hwc = medium_rgb_hwc * (1.0 - trans_bs)
        pred_hwc = _finite_clamp(rgb_object_hwc + rgb_medium_hwc, 0.0, 1.0)

        metadata = dict(base.metadata)
        metadata.update(
            {
                "medium_density_l1": density_l1,
                "medium_density_smooth": density_smooth,
            }
        )
        return MediumRenderOutput(
            image=pred_hwc.permute(2, 0, 1).contiguous(),
            alpha=base.alpha,
            depth=base.depth,
            radii=base.radii,
            means2d=base.means2d,
            metadata=metadata,
            rgb_clear=rgb_clear,
            rgb_object=rgb_object_hwc.permute(2, 0, 1).contiguous(),
            rgb_medium=rgb_medium_hwc.permute(2, 0, 1).contiguous(),
            medium_attn=medium_attn.permute(2, 0, 1).contiguous(),
            medium_bs=medium_bs.permute(2, 0, 1).contiguous(),
            medium_rgb=medium_rgb_hwc.permute(2, 0, 1).contiguous(),
            d_ref=d_ref,
        )

    def _integrate_medium(self, origin: Tensor, ray_dirs: Tensor, d_ref: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        height, width = int(ray_dirs.shape[0]), int(ray_dirs.shape[1])
        rays = ray_dirs.reshape(-1, 3)
        distances = d_ref.permute(1, 2, 0).reshape(-1, 1)
        total_pixels = rays.shape[0]
        chunk = max(int(self.config.chunk_pixels), 1)
        steps = self.medium_field.num_samples
        factors = (torch.arange(steps, device=rays.device, dtype=rays.dtype) + 0.5) / float(steps)

        attn_chunks = []
        bs_chunks = []
        density_l1_sum = rays.new_zeros(())
        smooth_sum = rays.new_zeros(())
        smooth_count = 0
        sample_count = 0
        for start in range(0, total_pixels, chunk):
            end = min(start + chunk, total_pixels)
            ray_chunk = rays[start:end]
            dist_chunk = distances[start:end].clamp_min(float(self.config.eps))
            sample_distances = dist_chunk * factors.view(1, steps)
            points = origin.to(ray_chunk).view(1, 1, 3) + ray_chunk[:, None, :] * sample_distances[..., None]
            beta, sigma_b = self.medium_field(points.reshape(-1, 3))
            beta = beta.reshape(end - start, steps, 3)
            sigma_b = sigma_b.reshape(end - start, steps, 3)
            max_density = max(float(self.config.max_density), 0.0)
            beta = _finite_clamp(beta, 0.0, max_density)
            sigma_b = _finite_clamp(sigma_b, 0.0, max_density)

            attn_chunks.append(beta.mean(dim=1))
            bs_chunks.append(sigma_b.mean(dim=1))

            density_l1_sum = density_l1_sum + beta.mean() * beta.numel()
            sample_count += beta.numel()
            if steps > 1:
                diff = beta[:, 1:, :] - beta[:, :-1, :]
                smooth_sum = smooth_sum + diff.square().mean() * diff.numel()
                smooth_count += diff.numel()

        medium_attn = torch.cat(attn_chunks, dim=0).reshape(height, width, 3)
        medium_bs = torch.cat(bs_chunks, dim=0).reshape(height, width, 3)
        density_l1 = density_l1_sum / max(sample_count, 1)
        density_smooth = smooth_sum / max(smooth_count, 1)
        return medium_attn, medium_bs, density_l1, density_smooth


def _remove_background(image: Tensor, alpha: Tensor, background, device: torch.device) -> Tensor:
    bg = _background_tensor(background, device=device, dtype=image.dtype).view(3, 1, 1)
    return _finite_clamp(image - bg * (1.0 - alpha), 0.0, 1.0)


def _background_tensor(value, device: torch.device, dtype: torch.dtype) -> Tensor:
    if isinstance(value, Tensor):
        return value.to(device=device, dtype=dtype)
    return torch.tensor(value, dtype=dtype, device=device)


def _reference_depth(depth: Tensor, alpha: Tensor, camera: Camera, config: MediumRenderConfig) -> Tensor:
    far = float(config.far_distance)
    if not math.isfinite(far) or far <= 0.0:
        far = max(float(camera.near) * 100.0, 1.0)
    depth_map = depth[:1].detach()
    valid = (alpha[:1] > float(config.alpha_threshold)) & torch.isfinite(depth_map) & (depth_map > 0.0)
    fallback = depth_map.new_full(depth_map.shape, far)
    d_ref = torch.where(valid, depth_map, fallback)
    return d_ref.clamp(min=max(float(camera.near), float(config.eps)), max=far).contiguous()


def _camera_ray_directions(camera: Camera) -> Tensor:
    ys = torch.arange(camera.height, device=camera.device, dtype=torch.float32)
    xs = torch.arange(camera.width, device=camera.device, dtype=torch.float32)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    dirs_cam = torch.stack(
        [
            (xx - float(camera.cx)) / float(camera.fx),
            (yy - float(camera.cy)) / float(camera.fy),
            torch.ones_like(xx),
        ],
        dim=-1,
    )
    dirs_world = dirs_cam @ camera.c2w[:3, :3].T
    return torch.nn.functional.normalize(dirs_world, dim=-1)


def _finite_clamp(value: Tensor, min_value: float, max_value: float) -> Tensor:
    return torch.nan_to_num(value, nan=float(min_value), posinf=float(max_value), neginf=float(min_value)).clamp(
        float(min_value),
        float(max_value),
    )

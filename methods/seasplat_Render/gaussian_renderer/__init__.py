#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from dataclasses import dataclass
from typing import Any

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from torch import Tensor
import torch.nn.functional as F
from utils.graphics_utils import fov2focal
from utils.sh_utils import eval_sh


@dataclass(frozen=True)
class MediumRenderConfig:
    """Narrow config needed by the direct SeaSplat medium renderer."""

    medium_far: float
    chunk_pixels: int = 65536
    max_density: float = 10.0
    max_optical_depth: float = 80.0
    eps: float = 1.0e-6

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        assert False # this is by default False
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            assert False # this is by default False
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, rendered_alpha, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "alpha": rendered_alpha,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}

def homogenize_points(points):
    """Convert batched points (xyz) to (xyz1)."""
    return torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)


def render_depth(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0):
    '''
    Use the original gaussian splatting renderer but just pass in
    a color override with precomputed colors as depth
    '''
    # calculate world space or camera space z coordinates of every gaussian
    # obtain things
    points_world = pc.get_xyz # torch.Size([25837, 3]), torch.float32
    T_cam_world = viewpoint_camera.world_view_transform.T # torch.Size([4, 4]), torch.float32

    # homogenize gaussian centers
    points_world_homogenized = homogenize_points(points_world)

    # apply T_cam_world to these points
    #T_cam_world = T_world_cam.inverse()
    points_cam_homogenized = (T_cam_world @ points_world_homogenized.T).T
    points_cam = points_cam_homogenized[:, :3]

    # get the zs and use as color for rendering
    zs = points_cam[:, 2].unsqueeze(-1)
    repeated_zs = zs.repeat(1, 3)

    return render(
        viewpoint_camera=viewpoint_camera,
        pc=pc,
        pipe=pipe,
        bg_color=bg_color,
        scaling_modifier=scaling_modifier,
        override_color=repeated_zs
    )


def render_medium(
    viewpoint_camera,
    pc: GaussianModel,
    pipe,
    medium_field,
    config: MediumRenderConfig,
    scaling_modifier: float = 1.0,
) -> dict[str, Any]:
    """Render the direct SeaSplat medium formula without external rasterizers."""
    if medium_field is None:
        raise ValueError("medium_field is required for render_medium")
    far = _resolve_medium_far(config)
    device = pc.get_xyz.device
    dtype = pc.get_xyz.dtype
    zero_bg = torch.zeros(3, dtype=dtype, device=device)

    colors = _compute_rgb_colors(pc, viewpoint_camera)
    gaussian_transmittance = _integrate_gaussian_transmittance(viewpoint_camera, pc, medium_field, config)
    attenuated_colors = (colors * gaussian_transmittance).contiguous()

    with torch.no_grad():
        clear_pkg = render(
            viewpoint_camera=viewpoint_camera,
            pc=pc,
            pipe=pipe,
            bg_color=zero_bg,
            scaling_modifier=scaling_modifier,
            override_color=colors.detach(),
        )
    object_pkg = render(
        viewpoint_camera=viewpoint_camera,
        pc=pc,
        pipe=pipe,
        bg_color=zero_bg,
        scaling_modifier=scaling_modifier,
        override_color=attenuated_colors,
    )
    depth_pkg = render_depth(
        viewpoint_camera=viewpoint_camera,
        pc=pc,
        pipe=pipe,
        bg_color=zero_bg,
        scaling_modifier=scaling_modifier,
    )

    alpha = object_pkg["alpha"]
    depth = _normalize_depth(depth_pkg["render"][0:1], alpha, viewpoint_camera, far, config.eps)
    medium_maps = _integrate_pixel_medium(viewpoint_camera, medium_field, depth, config)

    rgb_object = _finite_clamp(object_pkg["render"], 0.0, 1.0)
    rgb_medium = _finite_clamp(medium_maps["rgb_medium"], 0.0, 1.0)
    pred_image = _finite_clamp(rgb_object + rgb_medium, 0.0, 1.0)

    return {
        "render": pred_image,
        "alpha": alpha,
        "depth": depth,
        "viewspace_points": object_pkg["viewspace_points"],
        "visibility_filter": object_pkg["visibility_filter"],
        "radii": object_pkg["radii"],
        "rgb_object": rgb_object,
        "rgb_clear": _finite_clamp(clear_pkg["render"], 0.0, 1.0),
        "rgb_medium": rgb_medium,
        "medium_attn": medium_maps["medium_attn"],
        "medium_bs": medium_maps["medium_bs"],
        "medium_rgb": medium_maps["medium_rgb"],
        "metadata": {
            "medium_density_l1": medium_maps["medium_density_l1"],
            "medium_density_smooth": medium_maps["medium_density_smooth"],
            "medium_far": torch.tensor(far, device=device, dtype=dtype),
            "implementation": "direct_seasplat",
        },
    }


def _compute_rgb_colors(pc: GaussianModel, viewpoint_camera) -> Tensor:
    shs_view = pc.get_features.transpose(1, 2).contiguous()
    directions = pc.get_xyz - viewpoint_camera.camera_center.view(1, 3)
    directions = F.normalize(directions, dim=1, eps=1.0e-6)
    colors = eval_sh(pc.active_sh_degree, shs_view, directions)
    return torch.clamp_min(colors + 0.5, 0.0).contiguous()


def _integrate_gaussian_transmittance(
    viewpoint_camera,
    pc: GaussianModel,
    medium_field,
    config: MediumRenderConfig,
) -> Tensor:
    means = pc.get_xyz
    if means.numel() == 0:
        return torch.empty_like(means)

    device = means.device
    dtype = means.dtype
    origin = viewpoint_camera.camera_center.to(device=device, dtype=dtype).view(1, 3)
    offsets = means - origin
    distances = torch.linalg.norm(offsets, dim=1).clamp_min(float(config.eps))
    directions = offsets / distances[:, None]

    steps = max(int(medium_field.num_samples), 1)
    factors = _sample_factors(steps, device, dtype)
    chunk = max(int(config.chunk_pixels), 1)
    transmittance_chunks = []
    for start in range(0, means.shape[0], chunk):
        end = min(start + chunk, means.shape[0])
        length_chunk = distances[start:end]
        sample_distances = length_chunk[:, None] * factors[None, :]
        points = origin.view(1, 1, 3) + directions[start:end, None, :] * sample_distances[..., None]
        beta, _sigma_b = medium_field(points.reshape(-1, 3))
        beta = _finite_clamp(beta.reshape(end - start, steps, 3), 0.0, config.max_density)
        delta = (length_chunk / float(steps)).view(-1, 1, 1)
        tau = (beta * delta).sum(dim=1).clamp(0.0, float(config.max_optical_depth))
        transmittance_chunks.append(torch.exp(-tau))

    return torch.cat(transmittance_chunks, dim=0).contiguous()


def _integrate_pixel_medium(
    viewpoint_camera,
    medium_field,
    depth: Tensor,
    config: MediumRenderConfig,
) -> dict[str, Tensor]:
    height = int(viewpoint_camera.image_height)
    width = int(viewpoint_camera.image_width)
    device = depth.device
    dtype = depth.dtype
    far = _resolve_medium_far(config)
    steps = max(int(medium_field.num_samples), 1)
    factors = _sample_factors(steps, device, dtype)
    ray_dirs, depth_to_ray_scale = _camera_rays(viewpoint_camera, device, dtype, config.eps)
    ray_lengths = (depth.reshape(-1) * depth_to_ray_scale).clamp(float(config.eps), far)

    chunk = max(int(config.chunk_pixels), 1)
    origin = viewpoint_camera.camera_center.to(device=device, dtype=dtype).view(1, 1, 3)
    attn_chunks = []
    bs_chunks = []
    density_l1_sum = torch.zeros((), device=device, dtype=dtype)
    smooth_sum = torch.zeros((), device=device, dtype=dtype)
    sample_count = 0
    smooth_count = 0

    for start in range(0, ray_dirs.shape[0], chunk):
        end = min(start + chunk, ray_dirs.shape[0])
        length_chunk = ray_lengths[start:end]
        sample_distances = length_chunk[:, None] * factors[None, :]
        points = origin + ray_dirs[start:end, None, :] * sample_distances[..., None]
        beta, sigma_b = medium_field(points.reshape(-1, 3))
        beta = _finite_clamp(beta.reshape(end - start, steps, 3), 0.0, config.max_density)
        sigma_b = _finite_clamp(sigma_b.reshape(end - start, steps, 3), 0.0, config.max_density)

        delta = (length_chunk / float(steps)).view(-1, 1, 1)
        optical_depth = beta * delta
        tau_total = optical_depth.sum(dim=1).clamp(0.0, float(config.max_optical_depth))
        tau_mid = (torch.cumsum(optical_depth, dim=1) - 0.5 * optical_depth).clamp(
            0.0,
            float(config.max_optical_depth),
        )
        trans_mid = torch.exp(-tau_mid)
        backscatter = (trans_mid * sigma_b * delta).sum(dim=1).clamp(0.0, 1.0)

        attn_chunks.append(torch.exp(-tau_total))
        bs_chunks.append(backscatter)
        density_l1_sum = density_l1_sum + beta.mean() * beta.numel()
        sample_count += beta.numel()
        if steps > 1:
            diff = beta[:, 1:, :] - beta[:, :-1, :]
            smooth_sum = smooth_sum + diff.square().mean() * diff.numel()
            smooth_count += diff.numel()

    medium_attn = torch.cat(attn_chunks, dim=0).reshape(height, width, 3).permute(2, 0, 1).contiguous()
    medium_bs = torch.cat(bs_chunks, dim=0).reshape(height, width, 3).permute(2, 0, 1).contiguous()
    medium_rgb = medium_field.medium_rgb.to(device=device, dtype=dtype).view(3, 1, 1).expand(3, height, width)
    rgb_medium = medium_rgb * medium_bs
    density_l1 = density_l1_sum / max(sample_count, 1)
    density_smooth = smooth_sum / max(smooth_count, 1)

    return {
        "medium_attn": medium_attn,
        "medium_bs": medium_bs,
        "medium_rgb": medium_rgb.contiguous(),
        "rgb_medium": rgb_medium.contiguous(),
        "medium_density_l1": density_l1,
        "medium_density_smooth": density_smooth,
    }


def _camera_rays(viewpoint_camera, device: torch.device, dtype: torch.dtype, eps: float) -> tuple[Tensor, Tensor]:
    height = int(viewpoint_camera.image_height)
    width = int(viewpoint_camera.image_width)
    fx = fov2focal(viewpoint_camera.FoVx, width)
    fy = fov2focal(viewpoint_camera.FoVy, height)
    cx = width / 2.0
    cy = height / 2.0

    ys = torch.arange(height, device=device, dtype=dtype)
    xs = torch.arange(width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    dirs_cam = torch.stack(
        [
            (xx - float(cx)) / float(fx),
            (yy - float(cy)) / float(fy),
            torch.ones_like(xx),
        ],
        dim=-1,
    )
    dirs_cam = F.normalize(dirs_cam, dim=-1, eps=eps)
    depth_to_ray_scale = 1.0 / dirs_cam[..., 2].abs().clamp_min(float(eps))

    world_to_camera = viewpoint_camera.world_view_transform.T.to(device=device, dtype=dtype)
    camera_to_world = torch.inverse(world_to_camera)
    dirs_world = dirs_cam.reshape(-1, 3) @ camera_to_world[:3, :3].T
    dirs_world = F.normalize(dirs_world, dim=1, eps=eps)
    return dirs_world.contiguous(), depth_to_ray_scale.reshape(-1).contiguous()


def _normalize_depth(
    depth_render: Tensor,
    alpha: Tensor,
    viewpoint_camera,
    far: float,
    eps: float,
) -> Tensor:
    znear = float(getattr(viewpoint_camera, "znear", eps))
    fallback = depth_render.new_full(depth_render.shape, float(far))
    depth = torch.where(alpha > float(eps), depth_render / alpha.clamp_min(float(eps)), fallback)
    depth = torch.nan_to_num(depth, nan=float(far), posinf=float(far), neginf=znear)
    return depth.clamp(znear, float(far)).contiguous()


def _sample_factors(steps: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    return (torch.arange(steps, device=device, dtype=dtype) + 0.5) / float(steps)


def _resolve_medium_far(config: MediumRenderConfig) -> float:
    far = float(config.medium_far)
    if far <= 0.0:
        raise ValueError("medium_far must be positive before render_medium is called")
    return far


def _finite_clamp(value: Tensor, min_value: float, max_value: float) -> Tensor:
    return torch.nan_to_num(
        value,
        nan=float(min_value),
        posinf=float(max_value),
        neginf=float(min_value),
    ).clamp(float(min_value), float(max_value))

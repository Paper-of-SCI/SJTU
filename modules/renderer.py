"""Thin gsplat renderer wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional

import torch
from torch import Tensor

from modules.camera import Camera
from modules.gaussian_model import GaussianModel

RenderMode = Literal["RGB", "D", "ED", "RGB+D", "RGB+ED", "d", "Ed", "RGB-d", "RGB-Ed"]


@dataclass
class RenderOutput:
    """Renderer result in CHW-friendly form for external training code."""

    image: Tensor
    alpha: Tensor
    depth: Optional[Tensor]
    radii: Optional[Tensor]
    means2d: Optional[Tensor]
    metadata: Dict[str, Any]


class GaussianRenderer:
    """Single-call wrapper around ``gsplat.rasterization``."""

    def __init__(
        self,
        background: tuple[float, float, float] | Tensor = (1.0, 1.0, 1.0),
        packed: bool = True,
        sparse_grad: bool = False,
        absgrad: bool = True,
        rasterize_mode: Literal["classic", "antialiased"] = "classic",
        radius_clip: float = 0.0,
        eps2d: float = 0.3,
    ) -> None:
        self.background = background
        self.packed = packed
        self.sparse_grad = sparse_grad
        self.absgrad = absgrad
        self.rasterize_mode = rasterize_mode
        self.radius_clip = radius_clip
        self.eps2d = eps2d

    def render(
        self,
        model: GaussianModel,
        camera: Camera,
        sh_degree: Optional[int] = None,
        render_mode: RenderMode = "RGB",
        override_colors: Optional[Tensor] = None,
    ) -> RenderOutput:
        """Render one camera from one GaussianModel."""
        try:
            from gsplat import rasterization
        except ImportError as exc:
            raise ImportError("缺少 gsplat，请先安装与当前 PyTorch/CUDA 匹配的 gsplat。") from exc

        tensors = model.activated_tensors()
        degree = model.sh_degree if sh_degree is None else min(int(sh_degree), model.sh_degree)
        raster_colors = tensors.colors
        raster_sh_degree = degree
        if override_colors is not None:
            if override_colors.ndim != 2 or override_colors.shape[0] != model.num_gaussians:
                raise ValueError("override_colors must have shape [num_gaussians, channels]")
            raster_colors = override_colors.to(device=tensors.means.device, dtype=tensors.means.dtype)
            raster_sh_degree = None
        background = _background_tensor(self.background, camera.device)
        colors, alphas, meta = rasterization(
            means=tensors.means,
            quats=tensors.quats,
            scales=tensors.scales,
            opacities=tensors.opacities,
            colors=raster_colors,
            viewmats=camera.viewmat[None, ...],
            Ks=camera.K[None, ...],
            width=camera.width,
            height=camera.height,
            near_plane=camera.near,
            far_plane=camera.far,
            radius_clip=self.radius_clip,
            eps2d=self.eps2d,
            sh_degree=raster_sh_degree,
            packed=self.packed,
            backgrounds=None,
            render_mode=render_mode,
            sparse_grad=self.sparse_grad,
            absgrad=self.absgrad,
            rasterize_mode=self.rasterize_mode,
        )

        frame = colors.reshape(-1, camera.height, camera.width, colors.shape[-1])[0]
        depth = None
        if frame.shape[-1] > 3:
            depth = frame[..., 3:].permute(2, 0, 1).contiguous()
            frame = frame[..., :3]
        image = frame.permute(2, 0, 1).contiguous()
        alpha_frame = alphas.reshape(-1, camera.height, camera.width, alphas.shape[-1])[0]
        alpha = alpha_frame.permute(2, 0, 1).contiguous()
        if image.shape[0] == 3:
            image = image + background.view(3, 1, 1) * (1.0 - alpha)
        radii = meta.get("radii")
        means2d = meta.get("means2d")
        return RenderOutput(image=image, alpha=alpha, depth=depth, radii=radii, means2d=means2d, metadata=meta)


def _background_tensor(value: tuple[float, float, float] | Tensor, device: torch.device) -> Tensor:
    if isinstance(value, Tensor):
        return value.to(device=device, dtype=torch.float32)
    return torch.tensor(value, dtype=torch.float32, device=device)

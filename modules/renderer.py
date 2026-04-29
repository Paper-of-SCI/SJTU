"""可微分高斯光栅化器，封装 diff-gaussian-rasterization 后端。

职责：
    1. 用 SH 计算视角相关颜色
    2. 计算 3D 协方差
    3. 调用 CUDA 光栅化并返回 RenderOutput

RenderOutput.screenspace_means 保留梯度，供 DensificationController 读取。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

import torch
from torch import Tensor

from modules.gaussian_model import GaussianModel
from modules.camera import Camera
from modules.spherical_harmonics import eval_sh, get_view_directions

if TYPE_CHECKING:
    pass


@dataclass
class RenderOutput:
    """一次渲染的输出结果。

    Attributes:
        image: (3, H, W) 渲染 RGB 图像，float32 [0,1]。
        alpha: (1, H, W) 累积不透明度图。
        depth: (1, H, W) 深度图（若后端支持），否则为 None。
        radii: (N,) 每个高斯的 2D 投影半径（整数），不可见高斯为 0。
        visibility_filter: (N,) bool，True 表示在本帧有非零半径。
        screenspace_means: (N, 2) 屏幕空间 2D 均值，保留梯度供致密化使用。
    """
    image: Tensor
    alpha: Tensor
    depth: Optional[Tensor]
    radii: Tensor
    visibility_filter: Tensor
    screenspace_means: Tensor


class GaussianRenderer:
    """3DGS 可微分渲染器。

    封装 diff-gaussian-rasterization，提供统一的 render 接口。
    本类无可学习参数，是纯粹的无状态函数对象。

    Args:
        sh_degree: 最大 SH 阶数（运行时可用 sh_degree_override 覆盖）。
        bg_color: (3,) 背景颜色，默认白色 [1,1,1]。
        scale_modifier: 全局尺度缩放因子（调试用，默认 1.0）。
        antialiased: 是否启用抗锯齿（需要 diff-gaussian-rasterization 支持）。

    Usage::

        renderer = GaussianRenderer(sh_degree=3)
        output = renderer.render(gaussians, camera)
        loss = photometric_loss(output.image, camera.image)
        loss.backward()
        # output.screenspace_means.grad 可在此后读取
    """

    def __init__(
        self,
        sh_degree: int = 3,
        bg_color: Optional[Tensor] = None,
        scale_modifier: float = 1.0,
        antialiased: bool = False,
    ) -> None:
        self.sh_degree = sh_degree
        self.bg_color = bg_color if bg_color is not None else torch.ones(3)
        self.scale_modifier = scale_modifier
        self.antialiased = antialiased

    def render(
        self,
        gaussians: GaussianModel,
        camera: Camera,
        sh_degree_override: Optional[int] = None,
    ) -> RenderOutput:
        """执行一次完整的高斯光栅化渲染。

        Args:
            gaussians: GaussianModel，含当前场景参数。
            camera: Camera，含相机内参、外参和图像分辨率。
            sh_degree_override: 若指定，覆盖 self.sh_degree（用于 SH 渐进训练）。

        Returns:
            RenderOutput。
        """
        try:
            from diff_gaussian_rasterization import (
                GaussianRasterizationSettings,
                GaussianRasterizer,
            )
        except ImportError:
            raise ImportError(
                "缺少 diff-gaussian-rasterization 包，请参考 README 安装：\n"
                "  pip install git+https://github.com/graphdeco-inria/diff-gaussian-rasterization"
            )

        degree = sh_degree_override if sh_degree_override is not None else self.sh_degree
        degree = min(degree, gaussians.max_sh_degree)

        device = gaussians.means.device
        bg = self.bg_color.to(device)

        # 屏幕空间均值（保留梯度，供致密化梯度统计使用）
        screenspace_means = torch.zeros_like(
            gaussians.means, requires_grad=True
        )
        try:
            screenspace_means.retain_grad()
        except Exception:
            pass

        # 视角相关颜色（SH 求值）
        view_dirs = get_view_directions(gaussians.means, camera.camera_center)  # (N, 3)
        sh_coeffs = gaussians.sh_coefficients[:, :num_sh_coefficients(degree), :]  # (N, K, 3)
        colors = eval_sh(degree, sh_coeffs, view_dirs)   # (N, 3)
        colors = colors + 0.5   # 偏移，使 DC SH 对应 0.5 灰度
        colors = colors.clamp(min=0.0)

        # 光栅化设置
        raster_settings = GaussianRasterizationSettings(
            image_height=camera.height,
            image_width=camera.width,
            tanfovx=_fov_to_tan(camera.fov_x),
            tanfovy=_fov_to_tan(camera.fov_y),
            bg=bg,
            scale_modifier=self.scale_modifier,
            viewmatrix=camera.w2c.float(),
            projmatrix=camera.full_projection_matrix.float(),
            sh_degree=degree,
            campos=camera.camera_center.float(),
            prefiltered=False,
            debug=False,
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        rendered_image, radii = rasterizer(
            means3D=gaussians.means,
            means2D=screenspace_means,
            shs=None,               # 已手动计算颜色，不传 SH
            colors_precomp=colors,
            opacities=gaussians.opacities,
            scales=gaussians.scales,
            rotations=gaussians.rotations,
            cov3D_precomp=gaussians.compute_covariance_3d(),
        )

        visibility_filter = (radii > 0)

        return RenderOutput(
            image=rendered_image,
            alpha=rendered_image.new_ones(1, camera.height, camera.width),  # 占位
            depth=None,
            radii=radii,
            visibility_filter=visibility_filter,
            screenspace_means=screenspace_means,
        )


# ---------------------------------------------------------------------------
# 内部辅助
# ---------------------------------------------------------------------------

def _fov_to_tan(fov_rad: float) -> float:
    import math
    return math.tan(fov_rad / 2)


def num_sh_coefficients(degree: int) -> int:
    return (degree + 1) ** 2

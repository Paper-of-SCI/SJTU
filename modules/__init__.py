"""modules 包：PyTorch 3DGS 功能模块。

每个模块负责单一职责，可按需独立导入和替换。
依赖 diff-gaussian-rasterization CUDA 后端（仅 renderer 模块需要）。
"""

from modules.gaussian_model import GaussianModel
from modules.camera import Camera, camera_from_scene_data, cameras_from_scene_data
from modules.renderer import GaussianRenderer, RenderOutput
from modules.densification import DensificationController, DensificationStats
from modules.losses import l1_loss, ssim_loss, photometric_loss, depth_regularization_loss, opacity_entropy_loss
from modules.spherical_harmonics import eval_sh, rgb_to_sh_dc, sh_dc_to_rgb, get_view_directions
from modules.training import (
    TrainingConfig,
    build_optimizer,
    get_expon_lr_func,
    training_step,
    evaluate,
    train,
)

__all__ = [
    # 核心模型
    "GaussianModel",
    # 相机
    "Camera", "camera_from_scene_data", "cameras_from_scene_data",
    # 渲染
    "GaussianRenderer", "RenderOutput",
    # 致密化
    "DensificationController", "DensificationStats",
    # 损失函数
    "l1_loss", "ssim_loss", "photometric_loss",
    "depth_regularization_loss", "opacity_entropy_loss",
    # 球谐函数
    "eval_sh", "rgb_to_sh_dc", "sh_dc_to_rgb", "get_view_directions",
    # 训练
    "TrainingConfig", "build_optimizer", "get_expon_lr_func",
    "training_step", "evaluate", "train",
]

"""Composable gsplat-based 3DGS modules."""

from modules.camera import Camera
from modules.densification import DensificationConfig, DensificationController, DensificationStats
from modules.gaussian_model import GaussianModel
from modules.losses import l1_loss, photometric_loss, ssim
from modules.optim import OptimConfig, build_3dgs_optimizer, exponential_lr, set_group_lr
from modules.renderer import GaussianRenderer, RenderOutput

__all__ = [
    "Camera",
    "DensificationConfig",
    "DensificationController",
    "DensificationStats",
    "GaussianModel",
    "GaussianRenderer",
    "OptimConfig",
    "RenderOutput",
    "build_3dgs_optimizer",
    "exponential_lr",
    "l1_loss",
    "photometric_loss",
    "set_group_lr",
    "ssim",
]

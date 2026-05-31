"""Feature supervision helpers for Gaussian splatting experiments."""

from .feature_splatting import SplatRenderOutput, alpha_blend_splat_2d
from .losses import feature_mse_loss, mse_loss, psnr_from_mse

__all__ = [
    "SplatRenderOutput",
    "alpha_blend_splat_2d",
    "feature_mse_loss",
    "mse_loss",
    "psnr_from_mse",
]

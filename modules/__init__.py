"""Composable gsplat-based 3DGS modules."""

from modules.camera import Camera
from modules.densification import (
    DensificationConfig,
    DensificationController,
    DensificationStats,
    PatchGuidedDensificationConfig,
    PatchGuidedDensificationController,
)
from modules.densification_modes import (
    DENSIFICATION_MODE_CHOICES,
    normalize_densification_mode,
    uses_patch_densifier,
    uses_reallocation,
    uses_semantic_importance,
    validate_semantic_importance_root,
)
from modules.evaluation import (
    BestMetricTracker,
    build_lpips_evaluator,
    compute_image_metrics,
    evaluate_cameras,
    flatten_best_metric_fields,
    mean_metrics,
    resize_to_gt_if_needed,
)
from modules.gaussian_model import GaussianModel
from modules.losses import l1_loss, photometric_loss, ssim
from modules.optim import OptimConfig, build_3dgs_optimizer, exponential_lr, set_group_lr
from modules.renderer import GaussianRenderer, RenderOutput

__all__ = [
    "Camera",
    "DensificationConfig",
    "DensificationController",
    "DensificationStats",
    "DENSIFICATION_MODE_CHOICES",
    "GaussianModel",
    "GaussianRenderer",
    "OptimConfig",
    "PatchGuidedDensificationConfig",
    "PatchGuidedDensificationController",
    "RenderOutput",
    "BestMetricTracker",
    "build_lpips_evaluator",
    "build_3dgs_optimizer",
    "compute_image_metrics",
    "evaluate_cameras",
    "exponential_lr",
    "flatten_best_metric_fields",
    "l1_loss",
    "mean_metrics",
    "normalize_densification_mode",
    "photometric_loss",
    "resize_to_gt_if_needed",
    "set_group_lr",
    "ssim",
    "uses_patch_densifier",
    "uses_reallocation",
    "uses_semantic_importance",
    "validate_semantic_importance_root",
]

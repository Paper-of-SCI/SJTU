"""Composable gsplat-based 3DGS modules."""

from modules.camera import Camera
from modules.densification import (
    DensificationConfig,
    DensificationController,
    DensificationStats,
    PatchGuidedDensificationConfig,
    PatchGuidedDensificationController,
    PatchOnlyDensificationConfig,
    PatchOnlyDensificationController,
)
from modules.densification_modes import (
    DENSIFICATION_MODE_CHOICES,
    PATCH_ONLY_DENSIFICATION_MODE_CHOICES,
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
from modules.losses import image_gradient_magnitude, l1_loss, medium_decorrelation_loss, photometric_loss, ssim, underwater_loss
from modules.medium_field import MediumField, MediumFieldConfig, medium_checkpoint_path_for_ply, scene_medium_config
from modules.medium_renderer import MediumRenderConfig, MediumRenderOutput, MediumRenderer
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
    "MediumField",
    "MediumFieldConfig",
    "MediumRenderConfig",
    "MediumRenderOutput",
    "MediumRenderer",
    "OptimConfig",
    "PatchGuidedDensificationConfig",
    "PatchGuidedDensificationController",
    "PatchOnlyDensificationConfig",
    "PatchOnlyDensificationController",
    "PATCH_ONLY_DENSIFICATION_MODE_CHOICES",
    "RenderOutput",
    "BestMetricTracker",
    "build_lpips_evaluator",
    "build_3dgs_optimizer",
    "compute_image_metrics",
    "evaluate_cameras",
    "exponential_lr",
    "flatten_best_metric_fields",
    "image_gradient_magnitude",
    "l1_loss",
    "medium_checkpoint_path_for_ply",
    "medium_decorrelation_loss",
    "mean_metrics",
    "normalize_densification_mode",
    "photometric_loss",
    "resize_to_gt_if_needed",
    "set_group_lr",
    "ssim",
    "scene_medium_config",
    "underwater_loss",
    "uses_patch_densifier",
    "uses_reallocation",
    "uses_semantic_importance",
    "validate_semantic_importance_root",
]

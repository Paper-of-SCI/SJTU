"""Patch-guided densification helpers for Gaussian splatting experiments."""

from .contracts import (
    DensificationConfig,
    DensificationDecision,
    PatchDetailConfig,
    PatchDetailResult,
)
from .densification import (
    apply_gaussian_update,
    build_child_gaussians,
    select_densification_decision,
)
from .patch_detail import (
    accumulate_patch_detail_to_gaussians,
    compute_patch_detail,
    gaussian_patch_indices,
    patch_density_from_gaussians,
)

__all__ = [
    "DensificationConfig",
    "DensificationDecision",
    "PatchDetailConfig",
    "PatchDetailResult",
    "accumulate_patch_detail_to_gaussians",
    "apply_gaussian_update",
    "build_child_gaussians",
    "compute_patch_detail",
    "gaussian_patch_indices",
    "patch_density_from_gaussians",
    "select_densification_decision",
]

"""Small spherical-harmonics helpers for 3DGS initialization.

The renderer delegates SH evaluation to gsplat.  This module only keeps the
3DGS color/SH conversion helpers needed by initialization and checkpoints.
"""

from __future__ import annotations

import torch
from torch import Tensor

SH_C0 = 0.28209479177387814


def num_sh_bases(degree: int) -> int:
    """Return the number of SH bases up to ``degree``."""
    if degree < 0:
        raise ValueError("degree must be non-negative")
    return (degree + 1) ** 2


def rgb_to_sh(rgb: Tensor) -> Tensor:
    """Convert RGB values in [0, 1] to 0th-order 3DGS SH coefficients."""
    return (rgb - 0.5) / SH_C0


def sh_to_rgb(sh_dc: Tensor) -> Tensor:
    """Convert 0th-order 3DGS SH coefficients back to RGB values."""
    return SH_C0 * sh_dc + 0.5

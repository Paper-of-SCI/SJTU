"""Loss functions for feature-supervised splatting."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def mse_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Mean squared error with an optional spatial mask."""

    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have the same shape")

    sq_error = (prediction - target).pow(2)
    if valid_mask is None:
        return sq_error.mean()

    if valid_mask.shape != prediction.shape[:-1]:
        raise ValueError("valid_mask must match prediction spatial shape")

    weighted = sq_error * valid_mask.unsqueeze(-1).to(dtype=sq_error.dtype)
    denom = valid_mask.to(dtype=sq_error.dtype).sum().clamp_min(1.0) * prediction.shape[-1]
    return weighted.sum() / denom


def feature_mse_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    valid_mask: Optional[torch.Tensor] = None,
    normalize_features: bool = True,
) -> torch.Tensor:
    """MSE between rendered and teacher feature maps."""

    if normalize_features:
        prediction = F.normalize(prediction, dim=-1)
        target = F.normalize(target, dim=-1)
    return mse_loss(prediction, target, valid_mask=valid_mask)


def psnr_from_mse(mse: torch.Tensor, *, max_value: float = 1.0) -> torch.Tensor:
    """Convert MSE to PSNR."""

    return 20.0 * torch.log10(torch.as_tensor(max_value, device=mse.device)) - 10.0 * torch.log10(
        mse.clamp_min(1e-12)
    )

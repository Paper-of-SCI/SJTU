"""Utility functions for Gaussian Attribute Encoder inputs."""

import math
from typing import Mapping

import torch
import torch.nn.functional as F
from torch import Tensor


def build_means_position_encoding(
    means: Tensor,
    scene_scale: float | Tensor,
    pos_freqs: int = 4,
) -> Tensor:
    """Build sinusoidal position features for Gaussian means.

    Args:
        means: Gaussian centers with shape [N, 3].
        scene_scale: Scene scale used to normalize the Gaussian centers.
        pos_freqs: Number of sinusoidal frequency bands.

    Returns:
        Position features with shape [N, 3 * (1 + 2 * pos_freqs)].
        With the default pos_freqs=4, the output shape is [N, 27].
    """

    if means.ndim != 2 or means.shape[-1] != 3:
        raise ValueError(f"means must have shape [N, 3], got {tuple(means.shape)}")
    if pos_freqs < 0:
        raise ValueError("pos_freqs must be >= 0")

    scene_scale = torch.as_tensor(
        scene_scale,
        device=means.device,
        dtype=means.dtype,
    ).clamp_min(1e-6)
    means_normalized = means / scene_scale

    if pos_freqs == 0:
        return means_normalized

    freq_bands = 2.0 ** torch.arange(
        pos_freqs,
        device=means.device,
        dtype=means.dtype,
    )
    encoded = means_normalized[:, None, :] * freq_bands[None, :, None] * math.pi

    return torch.cat(
        [
            means_normalized,
            encoded.sin().flatten(start_dim=1),
            encoded.cos().flatten(start_dim=1),
        ],
        dim=-1,
    )


def build_gaussian_attributes(
    splats: Mapping[str, Tensor],
    scene_scale: float | Tensor,
    pos_freqs: int = 4,
) -> Tensor:
    """Build GAE input attributes from raw Gaussian splat tensors.

    Args:
        splats: Gaussian parameter dictionary containing means, scales, quats,
            opacities, sh0, and shN.
        scene_scale: Scene scale used to normalize the Gaussian centers.
        pos_freqs: Number of sinusoidal frequency bands for means.

    Returns:
        Gaussian attributes with shape [N, 83] when pos_freqs=4 and SH degree=3.
        Layout: [position_enc, opacity, quat, scale, flattened_sh].
    """

    required_keys = ("means", "scales", "quats", "opacities", "sh0", "shN")
    missing_keys = [key for key in required_keys if key not in splats]
    if missing_keys:
        raise KeyError(f"splats is missing required keys: {missing_keys}")

    position_feat = build_means_position_encoding(
        means=splats["means"],
        scene_scale=scene_scale,
        pos_freqs=pos_freqs,
    )
    opacity_feat = torch.sigmoid(splats["opacities"]).reshape(-1, 1)
    quat_feat = F.normalize(splats["quats"], dim=-1)
    scale_feat = splats["scales"]
    sh_feat = torch.cat([splats["sh0"], splats["shN"]], dim=1).flatten(start_dim=1)

    n = splats["means"].shape[0]
    features = [position_feat, opacity_feat, quat_feat, scale_feat, sh_feat]
    for feature in features:
        if feature.shape[0] != n:
            raise ValueError(
                f"All Gaussian attributes must have leading dimension {n}, "
                f"got {tuple(feature.shape)}"
            )

    return torch.cat(features, dim=-1)

"""Local KNN utilities for Gaussian neighborhood attention."""

from __future__ import annotations

import torch
from torch import Tensor


@torch.no_grad()
def build_local_knn_indices(
    points: Tensor,
    k: int = 16,
    chunk_size: int = 2048,
    include_self: bool = False,
) -> Tensor:
    """Return KNN indices for 3D Gaussian centers.

    Args:
        points: Gaussian centers with shape [N, 3].
        k: Number of neighbors to return for each Gaussian.
        chunk_size: Number of query points processed per distance chunk.
        include_self: Whether each Gaussian may include itself as a neighbor.

    Returns:
        Long tensor with shape [N, k_eff], where k_eff is clipped by N.
    """

    if points.ndim != 2 or points.shape[-1] != 3:
        raise ValueError(f"points must have shape [N, 3], got {tuple(points.shape)}")
    if k <= 0:
        raise ValueError("k must be > 0")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")

    points = points.detach()
    n = points.shape[0]
    if n == 0:
        raise ValueError("points must contain at least one Gaussian")

    if include_self:
        k_eff = min(k, n)
        topk_count = k_eff
    else:
        if n == 1:
            raise ValueError("Cannot build non-self KNN when N == 1")
        k_eff = min(k, n - 1)
        topk_count = k_eff + 1

    knn_chunks = []
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        distances = torch.cdist(points[start:end], points)  # [chunk, N]
        indices = distances.topk(
            k=topk_count,
            dim=-1,
            largest=False,
            sorted=True,
        ).indices
        if not include_self:
            indices = indices[:, 1:]
        knn_chunks.append(indices)

    return torch.cat(knn_chunks, dim=0).long()


def gather_local_features(features: Tensor, knn_indices: Tensor) -> Tensor:
    """Gather local feature groups from precomputed neighbor indices.

    Args:
        features: Gaussian features with shape [N, D].
        knn_indices: Neighbor indices with shape [N, K].

    Returns:
        Local features with shape [N, K, D].
    """

    if features.ndim != 2:
        raise ValueError(f"features must have shape [N, D], got {tuple(features.shape)}")
    if knn_indices.ndim != 2:
        raise ValueError(
            f"knn_indices must have shape [N, K], got {tuple(knn_indices.shape)}"
        )
    if knn_indices.shape[0] != features.shape[0]:
        raise ValueError(
            f"knn_indices.shape[0] must match features.shape[0], "
            f"got {knn_indices.shape[0]} and {features.shape[0]}"
        )

    if knn_indices.device != features.device:
        knn_indices = knn_indices.to(features.device)
    return features[knn_indices.long()]


def build_local_feature_groups(
    points: Tensor,
    features: Tensor,
    k: int = 16,
    chunk_size: int = 2048,
    include_self: bool = False,
) -> Tensor:
    """Build local feature groups for downstream self-attention.

    Args:
        points: Gaussian centers with shape [N, 3].
        features: Gaussian features with shape [N, D].
        k: Number of neighbors to gather.
        chunk_size: Number of query points processed per distance chunk.
        include_self: Whether each Gaussian may include itself as a neighbor.

    Returns:
        Local features with shape [N, k_eff, D].
    """

    knn_indices = build_local_knn_indices(
        points=points,
        k=k,
        chunk_size=chunk_size,
        include_self=include_self,
    )
    return gather_local_features(features, knn_indices)


class LocalKNN:
    """Callable wrapper around :func:`build_local_knn_indices`."""

    def __init__(
        self,
        k: int = 16,
        chunk_size: int = 2048,
        include_self: bool = False,
    ) -> None:
        self.k = k
        self.chunk_size = chunk_size
        self.include_self = include_self

    def __call__(self, points: Tensor) -> Tensor:
        return build_local_knn_indices(
            points=points,
            k=self.k,
            chunk_size=self.chunk_size,
            include_self=self.include_self,
        )

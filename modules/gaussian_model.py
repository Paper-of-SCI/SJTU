"""Composable 3D Gaussian parameter container.

This module owns only the learnable Gaussian state and basic mutation
operations.  It does not know about datasets, training loops, or renderers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from modules.spherical_harmonics import num_sh_bases, rgb_to_sh


PARAMETER_NAMES = (
    "means",
    "log_scales",
    "quats",
    "logit_opacities",
    "features_dc",
    "features_rest",
)


@dataclass(frozen=True)
class GaussianTensors:
    """Activated tensors consumed by renderers or custom external code."""

    means: Tensor
    scales: Tensor
    quats: Tensor
    opacities: Tensor
    colors: Tensor


class GaussianModel(nn.Module):
    """Learnable 3DGS scene state.

    Raw parameters are stored as ``nn.Parameter`` objects.  Public properties
    expose activated values where needed: scales are positive, quaternions are
    normalized, and opacities are in ``(0, 1)``.
    """

    def __init__(self, sh_degree: int = 3) -> None:
        super().__init__()
        if sh_degree < 0:
            raise ValueError("sh_degree must be non-negative")
        self.sh_degree = int(sh_degree)
        self.means = nn.Parameter(torch.empty(0, 3))
        self.log_scales = nn.Parameter(torch.empty(0, 3))
        self.quats = nn.Parameter(torch.empty(0, 4))
        self.logit_opacities = nn.Parameter(torch.empty(0, 1))
        self.features_dc = nn.Parameter(torch.empty(0, 1, 3))
        self.features_rest = nn.Parameter(torch.empty(0, max(num_sh_bases(sh_degree) - 1, 0), 3))
        self.register_buffer("gradient_accum", torch.empty(0))
        self.register_buffer("gradient_count", torch.empty(0))

    @classmethod
    def from_point_cloud(
        cls,
        xyz: np.ndarray | Tensor,
        rgb: np.ndarray | Tensor,
        sh_degree: int = 3,
        device: str | torch.device = "cuda",
        initial_opacity: float = 0.1,
        knn: int = 3,
    ) -> "GaussianModel":
        """Initialize one Gaussian per point.

        Args:
            xyz: Point coordinates with shape ``(N, 3)``.
            rgb: Point colors as uint8 ``[0,255]`` or float ``[0,1]``.
            sh_degree: Maximum SH degree stored in the model.
            device: Target torch device.
            initial_opacity: Initial post-sigmoid opacity.
            knn: Number of neighbors used for scale initialization.
        """
        model = cls(sh_degree=sh_degree).to(device)
        xyz_t = _to_float_tensor(xyz, device)
        if xyz_t.ndim != 2 or xyz_t.shape[-1] != 3:
            raise ValueError("xyz must have shape (N, 3)")
        rgb_t = _to_rgb_tensor(rgb, device)
        if rgb_t.shape != xyz_t.shape:
            raise ValueError("rgb must have shape (N, 3)")

        n = xyz_t.shape[0]
        mean_dist = _mean_neighbor_distance(xyz_t, k=knn).clamp_min(1e-7)
        log_scales = mean_dist.log().unsqueeze(-1).expand(n, 3).contiguous()

        quats = xyz_t.new_zeros(n, 4)
        quats[:, 0] = 1.0

        opacity = float(np.clip(initial_opacity, 1e-4, 1.0 - 1e-4))
        logit_opacity = math.log(opacity / (1.0 - opacity))
        logit_opacities = xyz_t.new_full((n, 1), logit_opacity)

        features_dc = rgb_to_sh(rgb_t).unsqueeze(1)
        features_rest = xyz_t.new_zeros(n, max(num_sh_bases(sh_degree) - 1, 0), 3)
        model.replace_tensors(
            {
                "means": xyz_t,
                "log_scales": log_scales,
                "quats": quats,
                "logit_opacities": logit_opacities,
                "features_dc": features_dc,
                "features_rest": features_rest,
            }
        )
        return model

    @property
    def num_gaussians(self) -> int:
        return int(self.means.shape[0])

    @property
    def scales(self) -> Tensor:
        return self.log_scales.exp()

    @property
    def normalized_quats(self) -> Tensor:
        return torch.nn.functional.normalize(self.quats, dim=-1)

    @property
    def opacities(self) -> Tensor:
        return self.logit_opacities.sigmoid().squeeze(-1)

    @property
    def colors(self) -> Tensor:
        return torch.cat([self.features_dc, self.features_rest], dim=1)

    def activated_tensors(self) -> GaussianTensors:
        """Return the tensors expected by ``gsplat.rasterization``."""
        return GaussianTensors(
            means=self.means,
            scales=self.scales,
            quats=self.normalized_quats,
            opacities=self.opacities,
            colors=self.colors,
        )

    def parameter_map(self) -> Dict[str, nn.Parameter]:
        """Return named raw parameters used by optimizer helpers."""
        return {name: getattr(self, name) for name in PARAMETER_NAMES}

    def replace_tensors(self, tensors: Mapping[str, Tensor]) -> None:
        """Replace all raw parameter tensors and reset gradient statistics."""
        missing = [name for name in PARAMETER_NAMES if name not in tensors]
        if missing:
            raise KeyError(f"missing tensors: {missing}")
        for name in PARAMETER_NAMES:
            setattr(self, name, nn.Parameter(tensors[name].detach().clone()))
        self._reset_gradient_buffers(self.num_gaussians, self.means.device)

    def append_tensors(self, tensors: Mapping[str, Tensor]) -> int:
        """Append new Gaussians from raw parameter tensors.

        Returns:
            Number of appended Gaussians.
        """
        if not tensors:
            return 0
        n_new = _infer_count(tensors.values())
        if n_new == 0:
            return 0
        updated = {}
        for name in PARAMETER_NAMES:
            base = getattr(self, name).detach()
            extra = tensors[name].to(device=base.device, dtype=base.dtype).detach()
            updated[name] = torch.cat([base, extra], dim=0)
        self.replace_tensors(updated)
        return n_new

    def clone(self, mask: Tensor, position_jitter_scale: float = 0.0) -> int:
        """Append copies of selected Gaussians, optionally jittering child positions."""
        mask = _as_bool_mask(mask, self.num_gaussians, self.means.device)
        tensors = {name: getattr(self, name).detach()[mask] for name in PARAMETER_NAMES}
        if position_jitter_scale > 0.0 and tensors["means"].numel() > 0:
            selected = mask.nonzero(as_tuple=True)[0]
            rotations = quats_to_rotmats(self.normalized_quats.detach()[selected])
            local_offsets = torch.randn_like(tensors["means"]) * self.scales.detach()[selected] * float(position_jitter_scale)
            tensors["means"] = tensors["means"] + torch.matmul(rotations, local_offsets[..., None]).squeeze(-1)
        return self.append_tensors(tensors)

    def split(self, mask: Tensor, num_splits: int = 2, scale_shrink: float = 1.6) -> tuple[int, Tensor]:
        """Append split children and remove selected parents.

        Returns:
            ``(num_added, keep_mask_after_append)``.  The keep mask is useful
            for external optimizer-state patching.
        """
        if num_splits < 1:
            raise ValueError("num_splits must be >= 1")
        mask = _as_bool_mask(mask, self.num_gaussians, self.means.device)
        selected = mask.nonzero(as_tuple=True)[0]
        if selected.numel() == 0:
            return 0, torch.ones(self.num_gaussians, dtype=torch.bool, device=self.means.device)

        means = self.means.detach()[selected]
        log_scales = self.log_scales.detach()[selected]
        quats = self.normalized_quats.detach()[selected]
        rotations = quats_to_rotmats(quats)
        scales = log_scales.exp()

        samples = torch.randn(selected.numel(), num_splits, 3, device=self.means.device, dtype=self.means.dtype)
        local_offsets = samples * scales[:, None, :]
        offsets = torch.matmul(rotations[:, None, :, :], local_offsets[..., None]).squeeze(-1)

        new_tensors = {
            "means": (means[:, None, :] + offsets).reshape(-1, 3),
            "log_scales": (log_scales - math.log(scale_shrink))[:, None, :].expand(-1, num_splits, -1).reshape(-1, 3),
            "quats": self.quats.detach()[selected][:, None, :].expand(-1, num_splits, -1).reshape(-1, 4),
            "logit_opacities": self.logit_opacities.detach()[selected][:, None, :].expand(-1, num_splits, -1).reshape(-1, 1),
            "features_dc": self.features_dc.detach()[selected][:, None, :, :].expand(-1, num_splits, -1, -1).reshape(-1, 1, 3),
            "features_rest": self.features_rest.detach()[selected][:, None, :, :].expand(-1, num_splits, -1, -1).reshape(
                -1, self.features_rest.shape[1], 3
            ),
        }

        old_n = self.num_gaussians
        added = self.append_tensors(new_tensors)
        keep = torch.ones(old_n + added, dtype=torch.bool, device=self.means.device)
        keep[selected] = False
        self.prune(~keep)
        return added, keep

    def prune(self, remove_mask: Tensor) -> Tensor:
        """Remove Gaussians where ``remove_mask`` is true.

        Returns:
            The keep mask used after pruning.
        """
        remove_mask = _as_bool_mask(remove_mask, self.num_gaussians, self.means.device)
        keep = ~remove_mask
        updated = {name: getattr(self, name).detach()[keep] for name in PARAMETER_NAMES}
        self.replace_tensors(updated)
        return keep

    def reset_opacities(self, value: float = 0.01) -> None:
        value = float(np.clip(value, 1e-4, 1.0 - 1e-4))
        logit = math.log(value / (1.0 - value))
        with torch.no_grad():
            self.logit_opacities.fill_(logit)

    def accumulate_gradient_stats(
        self,
        means2d: Tensor,
        visibility: Optional[Tensor] = None,
        use_absgrad: bool = True,
        indices: Optional[Tensor] = None,
    ) -> None:
        """Accumulate screen-space mean gradients for densification."""
        grad = getattr(means2d, "absgrad", None) if use_absgrad else None
        if grad is None:
            grad = means2d.grad
        if grad is None:
            return
        grad = grad.reshape(-1, grad.shape[-1])[:, :2]
        norms = grad.norm(dim=-1)
        if indices is not None:
            indices = indices.reshape(-1).to(device=norms.device, dtype=torch.long)
            if indices.numel() != norms.numel():
                return
            valid = (indices >= 0) & (indices < self.num_gaussians)
            if visibility is not None:
                visibility = visibility.reshape(-1).to(device=norms.device, dtype=torch.bool)
                if visibility.numel() == norms.numel():
                    valid = valid & visibility
            if not bool(valid.any()):
                return
            indices = indices[valid]
            norms = norms[valid]
            self.gradient_accum.index_add_(0, indices.to(self.gradient_accum.device), norms.to(self.gradient_accum.device))
            self.gradient_count.index_add_(0, indices.to(self.gradient_count.device), torch.ones_like(norms, device=self.gradient_count.device))
            return
        if norms.numel() != self.num_gaussians:
            return
        if visibility is None:
            visibility = torch.ones_like(norms, dtype=torch.bool)
        else:
            visibility = visibility.reshape(-1).to(device=norms.device, dtype=torch.bool)
            if visibility.numel() != norms.numel():
                visibility = torch.ones_like(norms, dtype=torch.bool)
        self.gradient_accum[visibility] += norms[visibility]
        self.gradient_count[visibility] += 1

    def clear_gradient_stats(self) -> None:
        self.gradient_accum.zero_()
        self.gradient_count.zero_()

    def _reset_gradient_buffers(self, count: int, device: torch.device) -> None:
        self.gradient_accum = torch.zeros(count, device=device)
        self.gradient_count = torch.zeros(count, device=device)


def quats_to_rotmats(quats: Tensor) -> Tensor:
    """Convert normalized wxyz quaternions to rotation matrices."""
    q = torch.nn.functional.normalize(quats, dim=-1)
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    ).reshape(-1, 3, 3)


def _to_float_tensor(value: np.ndarray | Tensor, device: str | torch.device) -> Tensor:
    if isinstance(value, Tensor):
        return value.to(device=device, dtype=torch.float32)
    return torch.as_tensor(value, dtype=torch.float32, device=device)


def _to_rgb_tensor(value: np.ndarray | Tensor, device: str | torch.device) -> Tensor:
    if isinstance(value, Tensor):
        rgb = value.to(device=device, dtype=torch.float32)
        return rgb / 255.0 if rgb.max() > 1.0 else rgb
    arr = np.asarray(value)
    rgb = torch.as_tensor(arr, dtype=torch.float32, device=device)
    return rgb / 255.0 if arr.dtype == np.uint8 or float(rgb.max()) > 1.0 else rgb


def _mean_neighbor_distance(xyz: Tensor, k: int = 3, chunk_size: int = 4096) -> Tensor:
    n = xyz.shape[0]
    if n <= 1:
        return xyz.new_full((n,), 0.01)
    k = min(max(int(k), 1), n - 1)
    out = []
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        d = torch.cdist(xyz[start:end], xyz)
        row = torch.arange(end - start, device=xyz.device)
        d[row, torch.arange(start, end, device=xyz.device)] = float("inf")
        out.append(d.topk(k, largest=False).values.mean(dim=-1))
    return torch.cat(out, dim=0).clamp_min(1e-6)


def _as_bool_mask(mask: Tensor, count: int, device: torch.device) -> Tensor:
    mask = mask.to(device=device, dtype=torch.bool).reshape(-1)
    if mask.numel() != count:
        raise ValueError(f"mask length {mask.numel()} does not match Gaussian count {count}")
    return mask


def _infer_count(values: Iterable[Tensor]) -> int:
    iterator = iter(values)
    first = next(iterator)
    count = int(first.shape[0])
    for tensor in iterator:
        if int(tensor.shape[0]) != count:
            raise ValueError("all tensors must have the same first dimension")
    return count

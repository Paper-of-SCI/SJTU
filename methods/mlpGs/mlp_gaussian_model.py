"""MLP-parameterized Gaussian model for the first MLP-GS experiment.

This module keeps the experiment local to ``methods/mlpGs``.  The model uses
fixed Gaussian anchor positions as MLP inputs, then predicts the raw Gaussian
parameters consumed by the existing renderer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import torch
import torch.nn as nn
from torch import Tensor

from modules.gaussian_model import GaussianModel, GaussianTensors, quats_to_rotmats
from modules.spherical_harmonics import num_sh_bases


@dataclass(frozen=True)
class RawGaussianTensors:
    """Raw tensors before renderer activations."""

    means: Tensor
    log_scales: Tensor
    quats: Tensor
    logit_opacities: Tensor
    features_dc: Tensor
    features_rest: Tensor


class MLPGaussianModel(nn.Module):
    """Predict all Gaussian parameters from fixed Gaussian positions.

    The initial GaussianModel is kept as a residual base.  The final MLP layer
    is initialized to zero, so step 0 matches the standard 3DGS initialization.
    """

    def __init__(
        self,
        anchor_xyz: Tensor,
        base_tensors: RawGaussianTensors,
        sh_degree: int = 3,
        hidden_dim: int = 128,
        hidden_layers: int = 3,
        feature_dim: int = 32,
        feature_init_std: float = 0.01,
        feature_split_noise_std: float = 0.01,
    ) -> None:
        super().__init__()
        if anchor_xyz.ndim != 2 or anchor_xyz.shape[-1] != 3:
            raise ValueError("anchor_xyz must have shape (N, 3)")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if hidden_layers < 1:
            raise ValueError("hidden_layers must be >= 1")
        if feature_dim < 0:
            raise ValueError("feature_dim must be >= 0")

        self.sh_degree = int(sh_degree)
        self.hidden_dim = int(hidden_dim)
        self.hidden_layers = int(hidden_layers)
        self.feature_dim = int(feature_dim)
        self.feature_init_std = float(feature_init_std)
        self.feature_split_noise_std = float(feature_split_noise_std)
        self.rest_bases = max(num_sh_bases(self.sh_degree) - 1, 0)

        self.register_buffer("anchor_xyz", anchor_xyz.detach().clone())
        self.register_buffer("input_center", anchor_xyz.detach().mean(dim=0, keepdim=True))
        input_scale = (anchor_xyz.detach() - self.input_center).norm(dim=-1).max().clamp_min(1.0e-6)
        self.register_buffer("input_scale", input_scale)
        self.anchor_features = nn.Parameter(self._initial_anchor_features(anchor_xyz.shape[0], anchor_xyz.device, anchor_xyz.dtype))

        self.register_buffer("base_means", base_tensors.means.detach().clone())
        self.register_buffer("base_log_scales", base_tensors.log_scales.detach().clone())
        self.register_buffer("base_quats", base_tensors.quats.detach().clone())
        self.register_buffer("base_logit_opacities", base_tensors.logit_opacities.detach().clone())
        self.register_buffer("base_features_dc", base_tensors.features_dc.detach().clone())
        self.register_buffer("base_features_rest", base_tensors.features_rest.detach().clone())
        self.register_buffer("gradient_accum", torch.zeros(anchor_xyz.shape[0], device=anchor_xyz.device))
        self.register_buffer("gradient_count", torch.zeros(anchor_xyz.shape[0], device=anchor_xyz.device))
        self._conditioned_anchor_features: Tensor | None = None

        self.mlp = self._build_mlp(
            input_dim=3 + self.feature_dim,
            hidden_dim=self.hidden_dim,
            hidden_layers=self.hidden_layers,
            output_dim=self.raw_output_dim,
        )

    @classmethod
    def from_gaussian_model(
        cls,
        base_model: GaussianModel,
        hidden_dim: int = 128,
        hidden_layers: int = 3,
        feature_dim: int = 32,
        feature_init_std: float = 0.01,
        feature_split_noise_std: float = 0.01,
    ) -> "MLPGaussianModel":
        """Create an MLP-GS model from the standard point-cloud initialization."""
        base_tensors = RawGaussianTensors(
            means=base_model.means.detach(),
            log_scales=base_model.log_scales.detach(),
            quats=base_model.quats.detach(),
            logit_opacities=base_model.logit_opacities.detach(),
            features_dc=base_model.features_dc.detach(),
            features_rest=base_model.features_rest.detach(),
        )
        return cls(
            anchor_xyz=base_model.means.detach(),
            base_tensors=base_tensors,
            sh_degree=base_model.sh_degree,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            feature_dim=feature_dim,
            feature_init_std=feature_init_std,
            feature_split_noise_std=feature_split_noise_std,
        )

    @property
    def num_gaussians(self) -> int:
        return int(self.anchor_xyz.shape[0])

    @property
    def raw_output_dim(self) -> int:
        return 3 + 3 + 4 + 1 + 3 + self.rest_bases * 3

    @property
    def means(self) -> Tensor:
        return self.raw_tensors().means

    @property
    def log_scales(self) -> Tensor:
        return self.raw_tensors().log_scales

    @property
    def quats(self) -> Tensor:
        return self.raw_tensors().quats

    @property
    def logit_opacities(self) -> Tensor:
        return self.raw_tensors().logit_opacities

    @property
    def features_dc(self) -> Tensor:
        return self.raw_tensors().features_dc

    @property
    def features_rest(self) -> Tensor:
        return self.raw_tensors().features_rest

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
        raw = self.raw_tensors()
        return torch.cat([raw.features_dc, raw.features_rest], dim=1)

    def set_conditioned_anchor_features(self, anchor_features: Tensor | None) -> None:
        """Temporarily use externally fused anchor features for prediction."""
        if anchor_features is None:
            self._conditioned_anchor_features = None
            return
        if anchor_features.shape != self.anchor_features.shape:
            raise ValueError(
                f"conditioned anchor feature shape {tuple(anchor_features.shape)} "
                f"does not match {tuple(self.anchor_features.shape)}"
            )
        self._conditioned_anchor_features = anchor_features

    def clear_conditioned_anchor_features(self) -> None:
        self._conditioned_anchor_features = None

    def current_anchor_features(self) -> Tensor:
        """Return fused features when active, otherwise the learnable features."""
        if self._conditioned_anchor_features is not None:
            return self._conditioned_anchor_features
        return self.anchor_features

    def raw_tensors(self, conditioned_anchor_features: Tensor | None = None) -> RawGaussianTensors:
        """Return the current raw Gaussian tensors predicted by the MLP."""
        anchor_features = self.current_anchor_features() if conditioned_anchor_features is None else conditioned_anchor_features
        deltas = self._predict_deltas(self.anchor_xyz, anchor_features)
        return RawGaussianTensors(
            means=self.base_means + deltas.means,
            log_scales=self.base_log_scales + deltas.log_scales,
            quats=self.base_quats + deltas.quats,
            logit_opacities=self.base_logit_opacities + deltas.logit_opacities,
            features_dc=self.base_features_dc + deltas.features_dc,
            features_rest=self.base_features_rest + deltas.features_rest,
        )

    def activated_tensors(self, conditioned_anchor_features: Tensor | None = None) -> GaussianTensors:
        """Return activated tensors expected by ``gsplat.rasterization``."""
        raw = self.raw_tensors(conditioned_anchor_features)
        return GaussianTensors(
            means=raw.means,
            scales=raw.log_scales.exp(),
            quats=torch.nn.functional.normalize(raw.quats, dim=-1),
            opacities=raw.logit_opacities.sigmoid().squeeze(-1),
            colors=torch.cat([raw.features_dc, raw.features_rest], dim=1),
        )

    def export_tensors(self, conditioned_anchor_features: Tensor | None = None) -> RawGaussianTensors:
        """Return detached raw tensors suitable for PLY checkpoint writing."""
        raw = self.raw_tensors(conditioned_anchor_features)
        return RawGaussianTensors(
            means=raw.means.detach(),
            log_scales=raw.log_scales.detach(),
            quats=torch.nn.functional.normalize(raw.quats.detach(), dim=-1),
            logit_opacities=raw.logit_opacities.detach(),
            features_dc=raw.features_dc.detach(),
            features_rest=raw.features_rest.detach(),
        )

    def checkpoint_payload(self, extra: Dict[str, object] | None = None) -> Dict[str, object]:
        """Build a resumable torch checkpoint payload for this experiment."""
        payload: Dict[str, object] = {
            "model_state_dict": self.state_dict(),
            "sh_degree": self.sh_degree,
            "hidden_dim": self.hidden_dim,
            "hidden_layers": self.hidden_layers,
            "feature_dim": self.feature_dim,
            "feature_init_std": self.feature_init_std,
            "feature_split_noise_std": self.feature_split_noise_std,
            "num_gaussians": self.num_gaussians,
            "raw_output_dim": self.raw_output_dim,
        }
        if extra:
            payload.update(extra)
        return payload

    def accumulate_gradient_stats(
        self,
        means2d: Tensor,
        visibility: Optional[Tensor] = None,
        use_absgrad: bool = True,
        indices: Optional[Tensor] = None,
    ) -> None:
        """Accumulate screen-space mean gradients for MLP-GS densification."""
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

    def clone(self, mask: Tensor) -> int:
        """Append one new MLP input anchor for each selected Gaussian."""
        mask = _as_bool_mask(mask, self.num_gaussians, self.anchor_xyz.device)
        if not bool(mask.any()):
            return 0
        raw = self.export_tensors()
        target = _index_raw_tensors(raw, mask)
        new_features = self.anchor_features.detach()[mask]
        return self.append_gaussians(target.means, target, new_features)

    def split(self, mask: Tensor, num_splits: int = 2, scale_shrink: float = 1.6) -> int:
        """Split selected Gaussians and remove their parents."""
        if num_splits < 1:
            raise ValueError("num_splits must be >= 1")
        mask = _as_bool_mask(mask, self.num_gaussians, self.anchor_xyz.device)
        selected = mask.nonzero(as_tuple=True)[0]
        if selected.numel() == 0:
            return 0

        raw = self.export_tensors()
        means = raw.means[selected]
        log_scales = raw.log_scales[selected]
        quats = torch.nn.functional.normalize(raw.quats[selected], dim=-1)
        rotations = quats_to_rotmats(quats)
        scales = log_scales.exp()

        samples = torch.randn(selected.numel(), num_splits, 3, device=self.anchor_xyz.device, dtype=self.anchor_xyz.dtype)
        local_offsets = samples * scales[:, None, :]
        offsets = torch.matmul(rotations[:, None, :, :], local_offsets[..., None]).squeeze(-1)
        new_means = (means[:, None, :] + offsets).reshape(-1, 3)

        target = RawGaussianTensors(
            means=new_means,
            log_scales=(log_scales - math.log(scale_shrink))[:, None, :].expand(-1, num_splits, -1).reshape(-1, 3),
            quats=raw.quats[selected][:, None, :].expand(-1, num_splits, -1).reshape(-1, 4),
            logit_opacities=raw.logit_opacities[selected][:, None, :].expand(-1, num_splits, -1).reshape(-1, 1),
            features_dc=raw.features_dc[selected][:, None, :, :].expand(-1, num_splits, -1, -1).reshape(-1, 1, 3),
            features_rest=raw.features_rest[selected][:, None, :, :].expand(-1, num_splits, -1, -1).reshape(
                -1, self.rest_bases, 3
            ),
        )
        new_features = self.anchor_features.detach()[selected][:, None, :].expand(-1, num_splits, -1).reshape(-1, self.feature_dim)
        if self.feature_split_noise_std > 0 and self.feature_dim > 0:
            new_features = new_features + torch.randn_like(new_features) * self.feature_split_noise_std
        added = self.append_gaussians(new_means, target, new_features)

        remove_mask = torch.zeros(self.num_gaussians, dtype=torch.bool, device=self.anchor_xyz.device)
        remove_mask[selected] = True
        self.prune(remove_mask)
        return added

    def prune(self, remove_mask: Tensor) -> Tensor:
        """Remove Gaussian anchors where ``remove_mask`` is true."""
        remove_mask = _as_bool_mask(remove_mask, self.num_gaussians, self.anchor_xyz.device)
        keep = ~remove_mask
        if bool(keep.all()):
            return keep
        self._replace_gaussian_buffers(
            anchor_xyz=self.anchor_xyz.detach()[keep],
            anchor_features=self.anchor_features.detach()[keep],
            base_tensors=RawGaussianTensors(
                means=self.base_means.detach()[keep],
                log_scales=self.base_log_scales.detach()[keep],
                quats=self.base_quats.detach()[keep],
                logit_opacities=self.base_logit_opacities.detach()[keep],
                features_dc=self.base_features_dc.detach()[keep],
                features_rest=self.base_features_rest.detach()[keep],
            ),
        )
        return keep

    def reset_opacities(self, value: float = 0.01) -> None:
        """Set current predicted opacities by shifting the residual base logits."""
        value = float(min(max(value, 1.0e-4), 1.0 - 1.0e-4))
        logit = math.log(value / (1.0 - value))
        deltas = self._predict_deltas(self.anchor_xyz)
        self.base_logit_opacities = self.base_logit_opacities.new_full((self.num_gaussians, 1), logit) - deltas.logit_opacities.detach()

    def append_gaussians(self, anchor_xyz: Tensor, target_tensors: RawGaussianTensors, anchor_features: Tensor | None = None) -> int:
        """Append new anchors whose first forward pass equals ``target_tensors``."""
        if anchor_features is None:
            anchor_features = self._initial_anchor_features(anchor_xyz.shape[0], anchor_xyz.device, anchor_xyz.dtype)
        n_new = _infer_count(
            [
                anchor_xyz,
                anchor_features,
                target_tensors.means,
                target_tensors.log_scales,
                target_tensors.quats,
                target_tensors.logit_opacities,
                target_tensors.features_dc,
                target_tensors.features_rest,
            ]
        )
        if n_new == 0:
            return 0
        anchor_xyz = anchor_xyz.to(device=self.anchor_xyz.device, dtype=self.anchor_xyz.dtype).detach()
        anchor_features = anchor_features.to(device=self.anchor_xyz.device, dtype=self.anchor_xyz.dtype).detach()
        target_tensors = _raw_to(target_tensors, device=self.anchor_xyz.device, dtype=self.anchor_xyz.dtype)
        base_new = self._base_tensors_for_targets(anchor_xyz, anchor_features, target_tensors)
        self._replace_gaussian_buffers(
            anchor_xyz=torch.cat([self.anchor_xyz.detach(), anchor_xyz], dim=0),
            anchor_features=torch.cat([self.anchor_features.detach(), anchor_features], dim=0),
            base_tensors=RawGaussianTensors(
                means=torch.cat([self.base_means.detach(), base_new.means], dim=0),
                log_scales=torch.cat([self.base_log_scales.detach(), base_new.log_scales], dim=0),
                quats=torch.cat([self.base_quats.detach(), base_new.quats], dim=0),
                logit_opacities=torch.cat([self.base_logit_opacities.detach(), base_new.logit_opacities], dim=0),
                features_dc=torch.cat([self.base_features_dc.detach(), base_new.features_dc], dim=0),
                features_rest=torch.cat([self.base_features_rest.detach(), base_new.features_rest], dim=0),
            ),
        )
        return n_new

    def normalized_inputs(self) -> Tensor:
        return self.mlp_inputs_for(self.anchor_xyz, self.anchor_features)

    def normalized_inputs_for(self, anchor_xyz: Tensor) -> Tensor:
        return (anchor_xyz - self.input_center) / self.input_scale

    def mlp_inputs_for(self, anchor_xyz: Tensor, anchor_features: Tensor) -> Tensor:
        xyz_inputs = self.normalized_inputs_for(anchor_xyz)
        if self.feature_dim == 0:
            return xyz_inputs
        return torch.cat([xyz_inputs, anchor_features], dim=-1)

    def _predict_deltas(self, anchor_xyz: Tensor, anchor_features: Tensor | None = None) -> RawGaussianTensors:
        if anchor_features is None:
            anchor_features = self.anchor_features
        deltas = self.mlp(self.mlp_inputs_for(anchor_xyz, anchor_features))
        return self._unpack_deltas(deltas)

    def _unpack_deltas(self, deltas: Tensor) -> RawGaussianTensors:
        n = int(deltas.shape[0])

        offset = 0
        delta_means = deltas[:, offset : offset + 3]
        offset += 3
        delta_log_scales = deltas[:, offset : offset + 3]
        offset += 3
        delta_quats = deltas[:, offset : offset + 4]
        offset += 4
        delta_logit_opacities = deltas[:, offset : offset + 1]
        offset += 1
        delta_features_dc = deltas[:, offset : offset + 3].reshape(n, 1, 3)
        offset += 3

        if self.rest_bases > 0:
            delta_features_rest = deltas[:, offset:].reshape(n, self.rest_bases, 3)
        else:
            delta_features_rest = deltas.new_zeros(n, 0, 3)
        return RawGaussianTensors(
            means=delta_means,
            log_scales=delta_log_scales,
            quats=delta_quats,
            logit_opacities=delta_logit_opacities,
            features_dc=delta_features_dc,
            features_rest=delta_features_rest,
        )

    def _base_tensors_for_targets(
        self,
        anchor_xyz: Tensor,
        anchor_features: Tensor,
        target_tensors: RawGaussianTensors,
    ) -> RawGaussianTensors:
        deltas = self._predict_deltas(anchor_xyz, anchor_features)
        return RawGaussianTensors(
            means=target_tensors.means.detach() - deltas.means.detach(),
            log_scales=target_tensors.log_scales.detach() - deltas.log_scales.detach(),
            quats=target_tensors.quats.detach() - deltas.quats.detach(),
            logit_opacities=target_tensors.logit_opacities.detach() - deltas.logit_opacities.detach(),
            features_dc=target_tensors.features_dc.detach() - deltas.features_dc.detach(),
            features_rest=target_tensors.features_rest.detach() - deltas.features_rest.detach(),
        )

    def _replace_gaussian_buffers(self, anchor_xyz: Tensor, anchor_features: Tensor, base_tensors: RawGaussianTensors) -> None:
        self.anchor_xyz = anchor_xyz.detach().clone()
        self.anchor_features = nn.Parameter(anchor_features.detach().clone())
        self._conditioned_anchor_features = None
        self.base_means = base_tensors.means.detach().clone()
        self.base_log_scales = base_tensors.log_scales.detach().clone()
        self.base_quats = base_tensors.quats.detach().clone()
        self.base_logit_opacities = base_tensors.logit_opacities.detach().clone()
        self.base_features_dc = base_tensors.features_dc.detach().clone()
        self.base_features_rest = base_tensors.features_rest.detach().clone()
        self._reset_gradient_buffers(self.num_gaussians, self.anchor_xyz.device)

    def _reset_gradient_buffers(self, count: int, device: torch.device) -> None:
        self.gradient_accum = torch.zeros(count, device=device)
        self.gradient_count = torch.zeros(count, device=device)

    def _initial_anchor_features(self, count: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        if self.feature_dim == 0:
            return torch.empty(count, 0, device=device, dtype=dtype)
        features = torch.empty(count, self.feature_dim, device=device, dtype=dtype)
        nn.init.normal_(features, mean=0.0, std=self.feature_init_std)
        return features

    @staticmethod
    def _build_mlp(input_dim: int, hidden_dim: int, hidden_layers: int, output_dim: int) -> nn.Sequential:
        layers: list[nn.Module] = []
        dim = input_dim
        for _ in range(hidden_layers):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.SiLU())
            dim = hidden_dim
        output = nn.Linear(dim, output_dim)
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)
        layers.append(output)
        return nn.Sequential(*layers)


def _index_raw_tensors(raw: RawGaussianTensors, mask: Tensor) -> RawGaussianTensors:
    return RawGaussianTensors(
        means=raw.means[mask],
        log_scales=raw.log_scales[mask],
        quats=raw.quats[mask],
        logit_opacities=raw.logit_opacities[mask],
        features_dc=raw.features_dc[mask],
        features_rest=raw.features_rest[mask],
    )


def _raw_to(raw: RawGaussianTensors, device: torch.device, dtype: torch.dtype) -> RawGaussianTensors:
    return RawGaussianTensors(
        means=raw.means.to(device=device, dtype=dtype).detach(),
        log_scales=raw.log_scales.to(device=device, dtype=dtype).detach(),
        quats=raw.quats.to(device=device, dtype=dtype).detach(),
        logit_opacities=raw.logit_opacities.to(device=device, dtype=dtype).detach(),
        features_dc=raw.features_dc.to(device=device, dtype=dtype).detach(),
        features_rest=raw.features_rest.to(device=device, dtype=dtype).detach(),
    )


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

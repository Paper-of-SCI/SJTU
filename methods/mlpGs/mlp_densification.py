"""Densification for the one-anchor-one-Gaussian MLP-GS experiment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from mlp_gaussian_model import MLPGaussianModel
from modules.densification import DensificationConfig, DensificationStats
from modules.renderer import RenderOutput


@dataclass
class MLPDensificationController:
    """Clone/split/prune MLP-GS anchors while preserving optimizer state."""

    config: DensificationConfig

    def update(
        self,
        model: MLPGaussianModel,
        render_output: RenderOutput,
        optimizer: torch.optim.Optimizer | None,
        step: int,
    ) -> DensificationStats:
        """Accumulate screen-space gradients and optionally mutate anchors."""
        visibility = _visibility_from_output(render_output, model.num_gaussians)
        if render_output.means2d is not None:
            model.accumulate_gradient_stats(
                render_output.means2d,
                visibility=visibility,
                use_absgrad=self.config.use_absgrad,
                indices=render_output.metadata.get("gaussian_ids"),
            )

        stats = DensificationStats(total=model.num_gaussians)
        in_range = self.config.start_step <= step < self.config.stop_step
        if in_range and self.config.interval > 0 and step % self.config.interval == 0:
            with torch.no_grad():
                stats = self._densify(model, optimizer, render_output)

        if self.config.opacity_reset_interval > 0 and step > 0 and step % self.config.opacity_reset_interval == 0:
            model.reset_opacities(self.config.reset_opacity)
            if optimizer is not None and model.train_base:
                _zero_optimizer_state_for(model, optimizer, "base_logit_opacities")
            stats.opacity_reset = True
            stats.total = model.num_gaussians

        return stats

    def _densify(
        self,
        model: MLPGaussianModel,
        optimizer: torch.optim.Optimizer | None,
        render_output: RenderOutput,
    ) -> DensificationStats:
        if model.num_gaussians == 0:
            return DensificationStats(total=0)

        avg_grad = model.gradient_accum / model.gradient_count.clamp_min(1.0)
        high_grad = avg_grad >= self.config.grad_threshold
        seen = model.gradient_count > 0
        seen_grad = avg_grad[seen]
        grad_mean = float(seen_grad.mean().item()) if seen_grad.numel() > 0 else 0.0
        grad_max = float(seen_grad.max().item()) if seen_grad.numel() > 0 else 0.0

        max_scale = model.scales.detach().max(dim=-1).values
        dense_threshold = self.config.scene_extent * self.config.percent_dense
        clone_mask = high_grad & (max_scale <= dense_threshold)
        split_mask = high_grad & (max_scale > dense_threshold)

        cloned = int(clone_mask.sum().item())
        if cloned > 0:
            old = _snapshot_optimizer(model, optimizer)
            old_n = model.num_gaussians
            model.clone(clone_mask)
            _patch_optimizer_append(model, optimizer, old, old_n)

        split = int(split_mask.sum().item()) * self.config.num_splits
        if split > 0:
            if split_mask.numel() < model.num_gaussians:
                split_mask = torch.cat(
                    [
                        split_mask,
                        torch.zeros(model.num_gaussians - split_mask.numel(), dtype=torch.bool, device=split_mask.device),
                    ],
                    dim=0,
                )
            old = _snapshot_optimizer(model, optimizer)
            old_n = model.num_gaussians
            added_expected = int(split_mask.sum().item()) * self.config.num_splits
            keep_after_append = torch.ones(old_n + added_expected, dtype=torch.bool, device=model.anchor_xyz.device)
            selected = split_mask[:old_n].nonzero(as_tuple=True)[0]
            keep_after_append[selected] = False
            added = model.split(split_mask, num_splits=self.config.num_splits)
            _patch_optimizer_append_and_prune(model, optimizer, old, keep_after_append, added)

        prune_mask = model.opacities.detach() < self.config.min_opacity
        if self.config.max_screen_radius is not None and render_output.radii is not None:
            radii = render_output.radii.reshape(-1)
            if radii.numel() == model.num_gaussians:
                prune_mask = prune_mask | (radii.to(model.anchor_xyz.device) > self.config.max_screen_radius)

        pruned = int(prune_mask.sum().item())
        if pruned > 0:
            old = _snapshot_optimizer(model, optimizer)
            model.prune(prune_mask)
            keep = ~prune_mask
            _patch_optimizer_prune(model, optimizer, old, keep)

        model.clear_gradient_stats()
        return DensificationStats(
            cloned=cloned,
            split=split,
            pruned=pruned,
            total=model.num_gaussians,
            densified=True,
            high_grad=int(high_grad.sum().item()),
            grad_mean=grad_mean,
            grad_max=grad_max,
        )


def _visibility_from_output(output: RenderOutput, count: int) -> Optional[Tensor]:
    if output.radii is None:
        return None
    radii = output.radii.reshape(-1)
    if radii.numel() != count:
        return None
    return radii > 0


def _snapshot_optimizer(model: MLPGaussianModel, optimizer: torch.optim.Optimizer | None) -> dict[str, dict]:
    if optimizer is None:
        return {}
    snapshot: dict[str, dict] = {}
    for name, param in model.resizable_parameter_map().items():
        state = optimizer.state.get(param, {})
        snapshot[name] = {
            "param": param,
            "state": {key: value.detach().clone() if torch.is_tensor(value) else value for key, value in state.items()},
        }
    return snapshot


def _replace_group_param(model: MLPGaussianModel, optimizer: torch.optim.Optimizer, name: str, state: dict) -> None:
    param = model.resizable_parameter_map()[name]
    for group in optimizer.param_groups:
        if group.get("name") == name:
            for old_param in group["params"]:
                if old_param is not param:
                    optimizer.state.pop(old_param, None)
            group["params"] = [param]
            optimizer.state[param] = state
            return
    raise RuntimeError(f"optimizer is missing the {name} parameter group")


def _patch_optimizer_append(
    model: MLPGaussianModel,
    optimizer: torch.optim.Optimizer | None,
    old: dict[str, dict],
    old_n: int,
) -> None:
    if optimizer is None:
        return
    added = model.num_gaussians - old_n
    for name in model.resizable_parameter_map():
        state = _expanded_state(old.get(name, {}).get("state", {}), added)
        _replace_group_param(model, optimizer, name, state)


def _patch_optimizer_append_and_prune(
    model: MLPGaussianModel,
    optimizer: torch.optim.Optimizer | None,
    old: dict[str, dict],
    keep_after_append: Tensor,
    added: int,
) -> None:
    if optimizer is None:
        return
    keep_after_append = keep_after_append.to(model.anchor_xyz.device)
    for name in model.resizable_parameter_map():
        state = _expanded_state(old.get(name, {}).get("state", {}), added)
        state = _masked_state(state, keep_after_append)
        _replace_group_param(model, optimizer, name, state)


def _patch_optimizer_prune(
    model: MLPGaussianModel,
    optimizer: torch.optim.Optimizer | None,
    old: dict[str, dict],
    keep: Tensor,
) -> None:
    if optimizer is None:
        return
    keep = keep.to(model.anchor_xyz.device)
    for name in model.resizable_parameter_map():
        state = _masked_state(old.get(name, {}).get("state", {}), keep)
        _replace_group_param(model, optimizer, name, state)


def _expanded_state(state: dict, added: int) -> dict:
    out = {}
    for key, value in state.items():
        if torch.is_tensor(value) and value.ndim > 0 and added > 0:
            pad = torch.zeros((added, *value.shape[1:]), dtype=value.dtype, device=value.device)
            out[key] = torch.cat([value, pad], dim=0)
        else:
            out[key] = value
    return out


def _masked_state(state: dict, keep: Tensor) -> dict:
    out = {}
    for key, value in state.items():
        if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == keep.numel():
            out[key] = value[keep.to(value.device)]
        else:
            out[key] = value
    return out


def _zero_optimizer_state_for(model: MLPGaussianModel, optimizer: torch.optim.Optimizer, name: str) -> None:
    param = model.resizable_parameter_map().get(name)
    if param is None:
        return
    for value in optimizer.state.get(param, {}).values():
        if torch.is_tensor(value):
            value.zero_()

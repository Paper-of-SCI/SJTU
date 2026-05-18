"""Composable adaptive densification helpers.

The controller mutates a GaussianModel only when the caller invokes ``update``.
It does not own the training loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import Tensor

from modules.gaussian_model import GaussianModel, PARAMETER_NAMES
from modules.renderer import RenderOutput


@dataclass
class DensificationStats:
    cloned: int = 0
    split: int = 0
    pruned: int = 0
    opacity_reset: bool = False
    total: int = 0
    densified: bool = False
    high_grad: int = 0
    grad_mean: float = 0.0
    grad_max: float = 0.0


@dataclass
class DensificationConfig:
    start_step: int = 500
    stop_step: int = 15_000
    interval: int = 100
    grad_threshold: float = 2.0e-4
    scene_extent: float = 1.0
    percent_dense: float = 0.01
    min_opacity: float = 0.005
    max_screen_radius: Optional[float] = None
    num_splits: int = 2
    opacity_reset_interval: int = 3000
    reset_opacity: float = 0.01
    use_absgrad: bool = True


class DensificationController:
    """Stateless-enough densification controller for caller-owned loops."""

    def __init__(self, config: DensificationConfig | None = None) -> None:
        self.config = config or DensificationConfig()

    def update(
        self,
        model: GaussianModel,
        render_output: RenderOutput,
        optimizer: torch.optim.Optimizer,
        step: int,
    ) -> DensificationStats:
        """Accumulate gradients and optionally clone/split/prune."""
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
            stats = self._densify(model, optimizer, render_output)

        if self.config.opacity_reset_interval > 0 and step > 0 and step % self.config.opacity_reset_interval == 0:
            model.reset_opacities(self.config.reset_opacity)
            _zero_optimizer_state_for(model, optimizer, "logit_opacities")
            stats.opacity_reset = True
            stats.total = model.num_gaussians
        return stats

    def _densify(self, model: GaussianModel, optimizer: torch.optim.Optimizer, render_output: RenderOutput) -> DensificationStats:
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
        split = int(split_mask.sum().item()) * self.config.num_splits

        if cloned > 0:
            old = _snapshot_optimizer(model, optimizer)
            old_n = model.num_gaussians
            model.clone(clone_mask)
            _patch_optimizer_append(model, optimizer, old, old_n)

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
            added, keep_after_append = model.split(split_mask, num_splits=self.config.num_splits)
            _patch_optimizer_append_and_prune(model, optimizer, old, keep_after_append, added)

        prune_mask = model.opacities.detach() < self.config.min_opacity
        if self.config.max_screen_radius is not None and render_output.radii is not None:
            radii = render_output.radii.reshape(-1)
            if radii.numel() == model.num_gaussians:
                prune_mask = prune_mask | (radii.to(model.means.device) > self.config.max_screen_radius)

        pruned = int(prune_mask.sum().item())
        if pruned > 0:
            old = _snapshot_optimizer(model, optimizer)
            keep = model.prune(prune_mask)
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


def _snapshot_optimizer(model: GaussianModel, optimizer: torch.optim.Optimizer) -> Dict[str, dict]:
    params = model.parameter_map()
    snapshot: Dict[str, dict] = {}
    for name, param in params.items():
        state = optimizer.state.get(param, {})
        snapshot[name] = {
            "param": param,
            "state": {key: value.detach().clone() if torch.is_tensor(value) else value for key, value in state.items()},
        }
    return snapshot


def _replace_group_param(model: GaussianModel, optimizer: torch.optim.Optimizer, name: str, state: dict) -> None:
    param = getattr(model, name)
    for group in optimizer.param_groups:
        if group.get("name") == name:
            for old_param in group["params"]:
                if old_param is not param:
                    optimizer.state.pop(old_param, None)
            group["params"] = [param]
            break
    optimizer.state[param] = state


def _patch_optimizer_append(model: GaussianModel, optimizer: torch.optim.Optimizer, old: Dict[str, dict], old_n: int) -> None:
    added = model.num_gaussians - old_n
    for name in PARAMETER_NAMES:
        state = _expanded_state(old[name]["state"], added)
        _replace_group_param(model, optimizer, name, state)


def _patch_optimizer_append_and_prune(
    model: GaussianModel,
    optimizer: torch.optim.Optimizer,
    old: Dict[str, dict],
    keep_after_append: Tensor,
    added: int,
) -> None:
    keep_after_append = keep_after_append.to(model.means.device)
    for name in PARAMETER_NAMES:
        state = _expanded_state(old[name]["state"], added)
        state = _masked_state(state, keep_after_append)
        _replace_group_param(model, optimizer, name, state)


def _patch_optimizer_prune(model: GaussianModel, optimizer: torch.optim.Optimizer, old: Dict[str, dict], keep: Tensor) -> None:
    keep = keep.to(model.means.device)
    for name in PARAMETER_NAMES:
        state = _masked_state(old[name]["state"], keep)
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


def _zero_optimizer_state_for(model: GaussianModel, optimizer: torch.optim.Optimizer, name: str) -> None:
    param = getattr(model, name)
    state = optimizer.state.get(param, {})
    for value in state.values():
        if torch.is_tensor(value):
            value.zero_()

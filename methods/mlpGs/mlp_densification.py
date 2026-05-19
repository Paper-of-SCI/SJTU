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
    """Clone/split/prune MLP-GS anchors without touching MLP optimizer state."""

    config: DensificationConfig

    def update(self, model: MLPGaussianModel, render_output: RenderOutput, step: int) -> DensificationStats:
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
                stats = self._densify(model, render_output)

        return stats

    def _densify(self, model: MLPGaussianModel, render_output: RenderOutput) -> DensificationStats:
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
            model.clone(clone_mask)

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
            model.split(split_mask, num_splits=self.config.num_splits)

        prune_mask = model.opacities.detach() < self.config.min_opacity
        if self.config.max_screen_radius is not None and render_output.radii is not None:
            radii = render_output.radii.reshape(-1)
            if radii.numel() == model.num_gaussians:
                prune_mask = prune_mask | (radii.to(model.anchor_xyz.device) > self.config.max_screen_radius)

        pruned = int(prune_mask.sum().item())
        if pruned > 0:
            model.prune(prune_mask)

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

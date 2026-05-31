from __future__ import annotations

import math
import unittest

import torch

from src.gs_patch_densification import (
    DensificationConfig,
    PatchDetailConfig,
    accumulate_patch_detail_to_gaussians,
    apply_gaussian_update,
    build_child_gaussians,
    compute_patch_detail,
    gaussian_patch_indices,
    select_densification_decision,
)


class PatchGuidedDensificationTest(unittest.TestCase):
    def test_patch_detail_shape_and_semantic_base(self) -> None:
        rendered = torch.zeros((16, 16, 3), dtype=torch.float32)
        target = torch.zeros_like(rendered)
        target[:8, :8] = 1.0
        semantic = torch.zeros((16, 16), dtype=torch.float32)

        base = compute_patch_detail(
            rendered,
            target,
            PatchDetailConfig(patch_size=8, edge_weight=0.0, use_semantic=False),
        )
        guided = compute_patch_detail(
            rendered,
            target,
            PatchDetailConfig(patch_size=8, edge_weight=0.0, semantic_base=0.25, use_semantic=True),
            semantic_importance=semantic,
        )

        self.assertEqual(base.detail.shape, (2, 2))
        self.assertEqual(guided.detail.shape, (2, 2))
        self.assertGreater(float(base.detail[0, 0]), 0.0)
        torch.testing.assert_close(guided.detail[0, 0], base.detail[0, 0] * 0.25)

    def test_gaussian_patch_indices_and_accumulation_boundaries(self) -> None:
        means = torch.tensor(
            [
                [0.0, 0.0],
                [15.9, 15.9],
                [-0.1, 1.0],
                [16.0, 2.0],
                [8.0, 8.0],
            ],
            dtype=torch.float32,
        )
        patch_detail = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)

        indices, valid = gaussian_patch_indices(means, (16, 16), 8)
        self.assertEqual(indices.tolist(), [0, 3, 0, 1, 3])
        self.assertEqual(valid.tolist(), [True, True, False, False, True])

        per_gaussian = accumulate_patch_detail_to_gaussians(means, patch_detail, (16, 16), 8)
        torch.testing.assert_close(per_gaussian, torch.tensor([1.0, 4.0, 0.0, 0.0, 4.0]))

    def test_clone_jitter_and_split_scale_rules(self) -> None:
        means = torch.tensor([[4.0, 4.0], [8.0, 8.0]], dtype=torch.float32)
        log_scales = torch.log(torch.tensor([[2.0, 2.0], [5.0, 5.0]], dtype=torch.float32))
        opacity = torch.tensor([-0.5, -0.25], dtype=torch.float32)
        colors = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=torch.float32)
        gradients = torch.tensor([2.0, 1.0], dtype=torch.float32)
        detail = torch.zeros(2, dtype=torch.float32)
        config = DensificationConfig(
            gradient_threshold=0.0,
            max_new_gaussians=2,
            large_scale_threshold=4.0,
            clone_jitter_fraction=0.10,
            split_scale_shrink=0.5,
        )
        decision = select_densification_decision(
            position_gradients=gradients,
            log_scales_xy=log_scales,
            opacity_logits=opacity,
            per_gaussian_detail=detail,
            config=config,
        )

        self.assertEqual(decision.clone_indices.tolist(), [0])
        self.assertEqual(decision.split_indices.tolist(), [1])
        generator = torch.Generator().manual_seed(123)
        child_means, child_logs, child_opacity, child_colors = build_child_gaussians(
            means_xy=means,
            log_scales_xy=log_scales,
            opacity_logits=opacity,
            color_logits=colors,
            decision=decision,
            config=config,
            generator=generator,
        )

        self.assertEqual(child_means.shape, (2, 2))
        self.assertFalse(torch.equal(child_means[0], means[0]))
        torch.testing.assert_close(child_logs[0], log_scales[0])
        torch.testing.assert_close(child_logs[1], log_scales[1] + math.log(0.5))
        torch.testing.assert_close(child_opacity, opacity)
        torch.testing.assert_close(child_colors, colors)

    def test_prune_and_reallocation_keep_budget(self) -> None:
        means = torch.stack((torch.arange(10, dtype=torch.float32), torch.arange(10, dtype=torch.float32)), dim=-1)
        log_scales = torch.log(torch.full((10, 2), 2.0, dtype=torch.float32))
        opacity = torch.tensor([-5.0, -4.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
        colors = torch.zeros((10, 3), dtype=torch.float32)
        gradients = torch.arange(10, dtype=torch.float32)
        detail = torch.tensor([0.0, 0.05, 0.2, 0.2, 0.3, 0.3, 0.5, 0.6, 1.0, 1.0], dtype=torch.float32)
        contribution = torch.tensor([0.0, 0.01, 0.5, 0.5, 0.5, 0.5, 0.7, 0.7, 1.0, 1.0])
        config = DensificationConfig(
            gradient_threshold=7.0,
            max_new_gaussians=2,
            large_scale_threshold=4.0,
            prune_fraction=0.2,
        )

        decision = select_densification_decision(
            position_gradients=gradients,
            log_scales_xy=log_scales,
            opacity_logits=opacity,
            per_gaussian_detail=detail,
            contribution_scores=contribution,
            config=config,
        )
        self.assertEqual(set(decision.prune_indices.tolist()), {0, 1})
        self.assertEqual(decision.clone_indices.numel(), 2)

        child_tensors = build_child_gaussians(
            means_xy=means,
            log_scales_xy=log_scales,
            opacity_logits=opacity,
            color_logits=colors,
            decision=decision,
            config=config,
            generator=torch.Generator().manual_seed(7),
        )
        new_state = apply_gaussian_update(
            means_xy=means,
            log_scales_xy=log_scales,
            opacity_logits=opacity,
            color_logits=colors,
            decision=decision,
            child_means_xy=child_tensors[0],
            child_log_scales_xy=child_tensors[1],
            child_opacity_logits=child_tensors[2],
            child_color_logits=child_tensors[3],
        )
        self.assertEqual(new_state[0].shape[0], 10)


if __name__ == "__main__":
    unittest.main()

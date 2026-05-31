from __future__ import annotations

import math
import unittest

import torch

from modules.densification import (
    DensificationConfig,
    PatchGuidedDensificationConfig,
    PatchGuidedDensificationController,
    _compute_patch_detail,
    _project_patch_detail_to_gaussians,
)
from modules.gaussian_model import GaussianModel, PARAMETER_NAMES
from modules.optim import build_3dgs_optimizer
from modules.renderer import RenderOutput


class ThreeDGSPatchDensificationTest(unittest.TestCase):
    def test_patch_detail_shape_and_normalization(self) -> None:
        render = torch.zeros((3, 17, 18), dtype=torch.float32)
        gt = torch.zeros_like(render)
        gt[:, :9, :9] = 1.0

        detail = _compute_patch_detail(render, gt, patch_size=8, edge_weight=0.75)

        self.assertEqual(detail.shape, (3, 3))
        self.assertGreaterEqual(float(detail.min()), 0.0)
        self.assertLessEqual(float(detail.max()), 1.0)
        self.assertGreater(float(detail[0, 0]), 0.0)

    def test_projection_accumulates_patch_detail_with_boundaries(self) -> None:
        detail = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
        means2d = torch.tensor(
            [
                [0.0, 0.0],
                [15.9, 15.9],
                [-0.1, 1.0],
                [16.0, 2.0],
                [8.0, 8.0],
            ],
            dtype=torch.float32,
        )

        values, counts = _project_patch_detail_to_gaussians(
            detail,
            means2d,
            gaussian_ids=None,
            radii=torch.ones(5),
            gaussian_count=5,
            image_height=16,
            image_width=16,
            patch_size=8,
        )

        torch.testing.assert_close(values, torch.tensor([1.0, 4.0, 0.0, 0.0, 4.0]))
        torch.testing.assert_close(counts, torch.tensor([1.0, 1.0, 0.0, 0.0, 1.0]))

    def test_clone_jitter_and_copy_rules(self) -> None:
        exact = make_model(2)
        exact.clone(torch.tensor([True, False]), position_jitter_scale=0.0)

        torch.testing.assert_close(exact.means[2], exact.means[0])
        torch.testing.assert_close(exact.log_scales[2], exact.log_scales[0])
        torch.testing.assert_close(exact.logit_opacities[2], exact.logit_opacities[0])
        torch.testing.assert_close(exact.features_dc[2], exact.features_dc[0])

        jittered = make_model(2)
        torch.manual_seed(123)
        jittered.clone(torch.tensor([True, False]), position_jitter_scale=0.25)

        self.assertFalse(torch.equal(jittered.means[2], jittered.means[0]))
        torch.testing.assert_close(jittered.log_scales[2], jittered.log_scales[0])
        torch.testing.assert_close(jittered.logit_opacities[2], jittered.logit_opacities[0])
        torch.testing.assert_close(jittered.features_dc[2], jittered.features_dc[0])

    def test_patch_controller_update_keeps_optimizer_lengths_consistent(self) -> None:
        model = make_model(4)
        optimizer = build_3dgs_optimizer(model)
        initialize_adam_state(model, optimizer)

        means2d = torch.tensor([[2.0, 2.0], [6.0, 2.0], [2.0, 6.0], [6.0, 6.0]], requires_grad=True)
        (means2d[:, 0].sum() + means2d[:, 1].sum()).backward()
        render = RenderOutput(
            image=torch.zeros((3, 8, 8), dtype=torch.float32),
            alpha=torch.ones((1, 8, 8), dtype=torch.float32),
            depth=None,
            radii=torch.ones(4),
            means2d=means2d,
            metadata={},
        )
        gt = torch.ones((3, 8, 8), dtype=torch.float32)
        controller = PatchGuidedDensificationController(
            PatchGuidedDensificationConfig(
                densification=DensificationConfig(
                    start_step=1,
                    stop_step=10,
                    interval=1,
                    grad_threshold=0.0,
                    scene_extent=1.0,
                    percent_dense=1.0,
                    min_opacity=0.0,
                ),
                patch_size=4,
                clone_jitter_scale=0.0,
            )
        )

        stats = controller.update(model, render, optimizer, step=1, gt_image=gt)

        self.assertTrue(stats.densified)
        self.assertEqual(stats.cloned, 4)
        self.assertEqual(model.num_gaussians, 8)
        assert_optimizer_lengths(model, optimizer)

    def test_reallocation_keeps_budget_and_optimizer_lengths_consistent(self) -> None:
        model = make_model(8)
        optimizer = build_3dgs_optimizer(model)
        initialize_adam_state(model, optimizer)
        model.gradient_accum = torch.linspace(0.1, 0.8, steps=8)
        model.gradient_count = torch.ones(8)

        controller = PatchGuidedDensificationController(
            PatchGuidedDensificationConfig(
                densification=DensificationConfig(
                    start_step=1,
                    stop_step=10,
                    interval=1,
                    grad_threshold=10.0,
                    scene_extent=1.0,
                    percent_dense=1.0,
                    min_opacity=0.0,
                    num_splits=2,
                ),
                reallocate_fraction=0.25,
                clone_jitter_scale=0.0,
            )
        )
        controller._detail_accum = torch.tensor([0.0, 0.0, 0.1, 0.2, 0.4, 0.6, 1.0, 1.0])
        controller._detail_count = torch.ones(8)
        render = RenderOutput(
            image=torch.zeros((3, 8, 8), dtype=torch.float32),
            alpha=torch.ones((1, 8, 8), dtype=torch.float32),
            depth=None,
            radii=torch.ones(8),
            means2d=None,
            metadata={},
        )

        stats = controller._densify(model, optimizer, render)

        self.assertEqual(stats.reallocated, 2)
        self.assertEqual(stats.cloned, 2)
        self.assertEqual(stats.pruned, 2)
        self.assertEqual(model.num_gaussians, 8)
        assert_optimizer_lengths(model, optimizer)


def make_model(count: int) -> GaussianModel:
    model = GaussianModel(sh_degree=0)
    means = torch.stack(
        [
            torch.arange(count, dtype=torch.float32),
            torch.zeros(count, dtype=torch.float32),
            torch.zeros(count, dtype=torch.float32),
        ],
        dim=-1,
    )
    model.replace_tensors(
        {
            "means": means,
            "log_scales": torch.full((count, 3), math.log(0.01), dtype=torch.float32),
            "quats": torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32).expand(count, 4).clone(),
            "logit_opacities": torch.zeros((count, 1), dtype=torch.float32),
            "features_dc": torch.arange(count * 3, dtype=torch.float32).reshape(count, 1, 3) / 100.0,
            "features_rest": torch.zeros((count, 0, 3), dtype=torch.float32),
        }
    )
    return model


def initialize_adam_state(model: GaussianModel, optimizer: torch.optim.Optimizer) -> None:
    optimizer.zero_grad(set_to_none=True)
    loss = sum(parameter.sum() * 0.0 for parameter in model.parameter_map().values())
    loss.backward()
    optimizer.step()


def assert_optimizer_lengths(model: GaussianModel, optimizer: torch.optim.Optimizer) -> None:
    params = model.parameter_map()
    for name in PARAMETER_NAMES:
        param = params[name]
        self_state = optimizer.state.get(param, {})
        for value in self_state.values():
            if torch.is_tensor(value) and value.ndim > 0:
                assert value.shape[0] == model.num_gaussians, name
        group = next(group for group in optimizer.param_groups if group.get("name") == name)
        assert group["params"][0] is param, name


if __name__ == "__main__":
    unittest.main()

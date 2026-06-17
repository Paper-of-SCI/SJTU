from __future__ import annotations

import unittest

import numpy as np
import torch

from modules import (
    DensificationConfig,
    GaussianModel,
    PatchOnlyDensificationConfig,
    PatchOnlyDensificationController,
    RenderOutput,
)


def _build_named_optimizer(model: GaussianModel) -> torch.optim.Adam:
    return torch.optim.Adam(
        [
            {"params": [model.means], "lr": 1.0e-3, "name": "means"},
            {"params": [model.features_dc], "lr": 1.0e-3, "name": "features_dc"},
            {"params": [model.features_rest], "lr": 1.0e-3, "name": "features_rest"},
            {"params": [model.logit_opacities], "lr": 1.0e-3, "name": "logit_opacities"},
            {"params": [model.log_scales], "lr": 1.0e-3, "name": "log_scales"},
            {"params": [model.quats], "lr": 1.0e-3, "name": "quats"},
        ],
        eps=1.0e-15,
    )


class PatchOnlyDensificationTest(unittest.TestCase):
    def test_patch_only_update_uses_gt_image_without_semantic_input(self) -> None:
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                [0.0, 0.1, 0.0],
                [0.1, 0.1, 0.0],
            ],
            dtype=np.float32,
        )
        colors = np.full((4, 3), 0.5, dtype=np.float32)
        model = GaussianModel.from_point_cloud(points, colors, sh_degree=0, device="cpu")
        optimizer = _build_named_optimizer(model)

        means2d = torch.tensor(
            [
                [0.5, 0.5],
                [2.5, 0.5],
                [0.5, 2.5],
                [2.5, 2.5],
            ],
            dtype=torch.float32,
            requires_grad=True,
        )
        means2d.sum().backward()

        render_image = torch.zeros(3, 4, 4)
        gt_image = torch.zeros(3, 4, 4)
        gt_image[:, :2, :2] = 1.0
        render_output = RenderOutput(
            image=render_image,
            alpha=torch.ones(1, 4, 4),
            depth=None,
            radii=torch.ones(model.num_gaussians),
            means2d=means2d,
            metadata={},
        )

        controller = PatchOnlyDensificationController(
            PatchOnlyDensificationConfig(
                densification=DensificationConfig(
                    start_step=1,
                    stop_step=10,
                    interval=1,
                    grad_threshold=0.0,
                    scene_extent=10.0,
                    percent_dense=1.0,
                    min_opacity=0.0,
                    opacity_reset_interval=0,
                    use_absgrad=False,
                ),
                patch_size=2,
                edge_weight=0.75,
                detail_lambda=2.0,
                clone_jitter_scale=0.0,
            )
        )

        initial_count = model.num_gaussians
        stats = controller.update(model, render_output, optimizer, step=1, gt_image=gt_image)

        self.assertTrue(stats.densified)
        self.assertGreater(stats.total, initial_count)
        self.assertGreaterEqual(stats.patch_detail_mean, 0.0)
        self.assertGreater(stats.patch_detail_max, 0.0)
        for group in optimizer.param_groups:
            self.assertEqual(group["params"][0].shape[0], stats.total)


if __name__ == "__main__":
    unittest.main()

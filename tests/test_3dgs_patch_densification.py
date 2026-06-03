from __future__ import annotations

import csv
import importlib
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from methods.semantic_importance import SemanticImportanceProvider
from modules import (
    BestMetricTracker,
    flatten_best_metric_fields,
    uses_reallocation,
    uses_semantic_importance,
    validate_semantic_importance_root,
)
from modules.densification import (
    DensificationConfig,
    DensificationController,
    PatchGuidedDensificationConfig,
    PatchGuidedDensificationController,
    _compute_patch_detail,
    _project_patch_detail_to_gaussians,
)
from modules.gaussian_model import GaussianModel, PARAMETER_NAMES
from modules.optim import build_3dgs_optimizer
from modules.renderer import RenderOutput

benchmark_patch_densification = importlib.import_module("methods.3dgs.benchmark_patch_densification")
train_3dgs_scene = importlib.import_module("methods.3dgs.train_3dgs_scene")


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

    def test_patch_detail_semantic_weighting_and_validation(self) -> None:
        render = torch.zeros((3, 8, 8), dtype=torch.float32)
        gt = torch.zeros_like(render)
        gt[:, :4, :4] = 1.0
        gt[:, :4, 4:] = 0.5

        plain = _compute_patch_detail(render, gt, patch_size=4, edge_weight=0.0)
        all_semantic = torch.ones((8, 8), dtype=torch.float32)
        weighted_all = _compute_patch_detail(
            render,
            gt,
            patch_size=4,
            edge_weight=0.0,
            semantic_importance=all_semantic,
            semantic_base=0.2,
        )
        torch.testing.assert_close(weighted_all, plain)

        zero_semantic = torch.zeros((8, 8), dtype=torch.float32)
        weighted_zero = _compute_patch_detail(
            render,
            gt,
            patch_size=4,
            edge_weight=0.0,
            semantic_importance=zero_semantic,
            semantic_base=0.2,
        )
        torch.testing.assert_close(weighted_zero, plain * 0.2)

        spatial_semantic = torch.ones((8, 8), dtype=torch.float32)
        spatial_semantic[:4, :4] = 0.0
        weighted_spatial = _compute_patch_detail(
            render,
            gt,
            patch_size=4,
            edge_weight=0.0,
            semantic_importance=spatial_semantic,
            semantic_base=0.2,
        )
        self.assertLess(float(weighted_spatial[0, 0]), float(plain[0, 0]))
        self.assertGreater(float(weighted_spatial[0, 1]), float(weighted_spatial[0, 0]))

        with self.assertRaises(ValueError):
            _compute_patch_detail(render, gt, patch_size=4, edge_weight=0.0, semantic_importance=torch.ones((4, 4)))

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

    def test_opacity_reset_only_runs_inside_densification_phase(self) -> None:
        model = make_model(2)
        optimizer = build_3dgs_optimizer(model)
        render = RenderOutput(
            image=torch.zeros((3, 4, 4), dtype=torch.float32),
            alpha=torch.ones((1, 4, 4), dtype=torch.float32),
            depth=None,
            radii=torch.ones(2),
            means2d=None,
            metadata={},
        )
        controller = DensificationController(
            DensificationConfig(
                start_step=1,
                stop_step=3,
                interval=0,
                opacity_reset_interval=3,
                reset_opacity=0.01,
            )
        )

        stats = controller.update(model, render, optimizer, step=3)

        self.assertFalse(stats.opacity_reset)
        torch.testing.assert_close(model.opacities, torch.full((2,), 0.5))

    def test_semantic_importance_provider_loads_named_mask(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            scene_dir = Path(tmp_dir) / "SceneA"
            scene_dir.mkdir(parents=True)
            mask = np.array([[0, 255], [128, 64]], dtype=np.uint8)
            Image.fromarray(mask).save(scene_dir / "frame_001.png")

            provider = SemanticImportanceProvider(Path(tmp_dir), "SceneA", torch.device("cpu"))
            loaded = provider.load("/any/path/frame_001.jpg", width=4, height=3)

            self.assertEqual(tuple(loaded.shape), (3, 4))
            self.assertGreaterEqual(float(loaded.min()), 0.0)
            self.assertLessEqual(float(loaded.max()), 1.0)

            with self.assertRaises(FileNotFoundError):
                provider.load("/any/path/missing.jpg", width=4, height=3)

    def test_densification_mode_contracts_keep_semantic_and_reallocate_separate(self) -> None:
        self.assertFalse(uses_semantic_importance("patch_reallocate"))
        self.assertTrue(uses_reallocation("patch_reallocate"))
        self.assertTrue(uses_semantic_importance("patch_reallocate_semantic"))
        self.assertTrue(uses_reallocation("patch_reallocate_semantic"))

        validate_semantic_importance_root(["patch_reallocate"], "")
        with self.assertRaisesRegex(ValueError, "patch_reallocate_semantic"):
            validate_semantic_importance_root(["patch_reallocate_semantic"], "")

    def test_best_metric_tracker_tracks_maxima_and_lpips_minimum(self) -> None:
        tracker = BestMetricTracker(include_lpips=True)

        first = tracker.update(100, {"psnr": 20.0, "ssim": 0.70, "l1": 0.05, "lpips": 0.40})
        self.assertEqual(set(first), {"psnr", "ssim", "lpips"})
        tracker.set_checkpoint("psnr", "best/best_psnr.ply")
        tracker.set_checkpoint("ssim", "best/best_ssim.ply")
        tracker.set_checkpoint("lpips", "best/best_lpips.ply")

        second = tracker.update(200, {"psnr": 19.0, "ssim": 0.75, "l1": 0.04, "lpips": 0.35})
        self.assertEqual(set(second), {"ssim", "lpips"})
        tracker.set_checkpoint("ssim", "best/best_ssim_step200.ply")
        tracker.set_checkpoint("lpips", "best/best_lpips_step200.ply")

        payload = tracker.to_dict()
        fields = flatten_best_metric_fields(payload)
        self.assertEqual(fields["best_psnr"], 20.0)
        self.assertEqual(fields["best_psnr_step"], 100)
        self.assertEqual(fields["best_psnr_checkpoint"], "best/best_psnr.ply")
        self.assertEqual(fields["best_ssim_step"], 200)
        self.assertEqual(fields["best_lpips"], 0.35)
        self.assertEqual(len(payload["history"]), 2)

    def test_semantic_base_values_expand_only_semantic_variants(self) -> None:
        specs = benchmark_patch_densification.build_run_specs(
            ["standard_3dgs", "patch_guided", "patch_guided_semantic"],
            semantic_base=0.2,
            semantic_base_values=[0.2, 0.4],
        )

        self.assertEqual(
            [(spec.variant, spec.semantic_base, spec.semantic_base_label) for spec in specs],
            [
                ("standard_3dgs", 0.2, ""),
                ("patch_guided", 0.2, ""),
                ("patch_guided_semantic", 0.2, "base_0p20"),
                ("patch_guided_semantic", 0.4, "base_0p40"),
            ],
        )

    def test_csv_only_skip_existing_does_not_require_final_ply(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            final_ply = root / "train" / "final.ply"
            metrics_csv = root / "test_renders" / "metrics.csv"
            training_summary = root / "train" / "training_summary.json"
            metrics_csv.parent.mkdir(parents=True)
            training_summary.parent.mkdir(parents=True)
            metrics_csv.write_text("image,psnr\nframe.png,20\n", encoding="utf-8")
            training_summary.write_text("{}", encoding="utf-8")

            self.assertTrue(benchmark_patch_densification.is_run_complete(True, final_ply, metrics_csv, training_summary))
            self.assertFalse(benchmark_patch_densification.is_run_complete(False, final_ply, metrics_csv, training_summary))

    def test_compact_summary_hides_path_and_checkpoint_fields_and_writes_aggregate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir)
            rows = [
                {
                    "scene": "cur",
                    "variant": "patch_guided_semantic",
                    "semantic_base": 0.2,
                    "seed": 1,
                    "psnr": 20.0,
                    "ssim": 0.8,
                    "lpips": 0.2,
                    "l1": 0.05,
                    "gaussian_count": 100,
                    "train_seconds": 10.0,
                    "total_seconds": 12.0,
                    "final_ply": "/tmp/final.ply",
                    "best_psnr_checkpoint": "/tmp/best.ply",
                },
                {
                    "scene": "cur",
                    "variant": "patch_guided_semantic",
                    "semantic_base": 0.2,
                    "seed": 2,
                    "psnr": 22.0,
                    "ssim": 0.9,
                    "lpips": 0.1,
                    "l1": 0.03,
                    "gaussian_count": 120,
                    "train_seconds": 14.0,
                    "total_seconds": 16.0,
                },
            ]

            benchmark_patch_densification.write_summary(out_dir, rows, compact=True)

            with (out_dir / "summary.csv").open(newline="", encoding="utf-8") as handle:
                header = next(csv.reader(handle))
            self.assertEqual(header, benchmark_patch_densification.COMPACT_SUMMARY_FIELDS)
            self.assertNotIn("final_ply", header)
            self.assertNotIn("best_psnr_checkpoint", header)

            with (out_dir / "summary_agg.csv").open(newline="", encoding="utf-8") as handle:
                aggregate_rows = list(csv.DictReader(handle))
            self.assertEqual(len(aggregate_rows), 1)
            self.assertEqual(aggregate_rows[0]["count"], "2")
            self.assertAlmostEqual(float(aggregate_rows[0]["psnr_mean"]), 21.0)

    def test_save_every_zero_disables_step_artifacts(self) -> None:
        self.assertFalse(train_3dgs_scene.should_save_training_artifacts(step=1, iterations=100, save_every=0))
        self.assertTrue(train_3dgs_scene.should_save_training_artifacts(step=1, iterations=100, save_every=100))
        self.assertTrue(train_3dgs_scene.should_save_training_artifacts(step=100, iterations=100, save_every=100))

    def test_cleanup_csv_only_artifacts_keeps_metric_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir)
            keep_files = [
                run_dir / "train" / "training_summary.json",
                run_dir / "test_renders" / "metrics.csv",
            ]
            delete_files = [
                run_dir / "train" / "final.ply",
                run_dir / "train" / "checkpoints" / "step_000001.ply",
                run_dir / "train" / "best" / "best_psnr.ply",
                run_dir / "train" / "previews" / "step_000001.png",
                run_dir / "test_renders" / "000_frame_render.png",
                run_dir / "test_renders" / "000_frame_gt.png",
            ]
            for path in keep_files + delete_files:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("x", encoding="utf-8")

            benchmark_patch_densification.cleanup_csv_only_artifacts(run_dir)

            self.assertTrue(all(path.exists() for path in keep_files))
            self.assertFalse(any(path.exists() for path in delete_files))

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

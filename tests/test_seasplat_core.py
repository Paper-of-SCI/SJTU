from __future__ import annotations

import inspect
import unittest

import torch

from methods.seasplat.seasplat_core import (
    AttenuateNetV3,
    BackscatterNetV2,
    SeaSplatConfig,
    SeaSplatForwardOutput,
    SeaSplatState,
    compute_seasplat_loss,
    dark_channel_prior_v3,
    gray_world_loss,
    rgb_saturation_loss,
    should_update_medium,
    smooth_depth_loss,
)
from modules.renderer import GaussianRenderer, RenderOutput


class SeaSplatCoreTest(unittest.TestCase):
    def test_medium_network_shapes(self) -> None:
        depth = torch.linspace(0.0, 1.0, 20, dtype=torch.float32).reshape(1, 1, 4, 5)
        backscatter = BackscatterNetV2()
        attenuate = AttenuateNetV3()

        self.assertEqual(backscatter(depth).shape, (1, 3, 4, 5))
        self.assertEqual(attenuate(depth).shape, (1, 3, 4, 5))

    def test_underwater_image_range_and_loss_parts_are_finite(self) -> None:
        torch.manual_seed(1)
        config = SeaSplatConfig(seathru_from_step=0)
        state = SeaSplatState(config, device="cpu")
        base = torch.rand(3, 8, 9)
        depth = torch.rand(1, 8, 9)
        gt = torch.rand(3, 8, 9)
        attenuation = state.attenuate(depth.unsqueeze(0))
        backscatter = state.backscatter(depth.unsqueeze(0))
        direct = base.unsqueeze(0) * attenuation
        predicted = torch.clamp(direct + backscatter, 0.0, 1.0).squeeze(0)
        dummy = RenderOutput(
            image=base,
            alpha=torch.ones(1, 8, 9),
            depth=None,
            radii=None,
            means2d=None,
            metadata={},
        )
        output = SeaSplatForwardOutput(
            base_render=dummy,
            depth_render=dummy,
            base_image=base,
            predicted_image=predicted,
            depth=depth,
            attenuation=attenuation,
            backscatter=backscatter,
            direct=direct,
            seathru_active=True,
        )

        self.assertGreaterEqual(float(predicted.detach().min()), 0.0)
        self.assertLessEqual(float(predicted.detach().max()), 1.0)
        loss, parts = compute_seasplat_loss(output, gt, state, step=1)

        self.assertTrue(torch.isfinite(loss))
        for value in parts.values():
            self.assertTrue(torch.isfinite(value))

    def test_regularizers_return_finite_scalars(self) -> None:
        rgb = torch.rand(1, 3, 6, 7)
        depth = torch.rand(1, 1, 6, 7)

        for value in [
            smooth_depth_loss(rgb, depth),
            dark_channel_prior_v3(rgb - 0.5),
            gray_world_loss(rgb),
            rgb_saturation_loss(rgb, saturation_val=0.7),
        ]:
            self.assertEqual(value.ndim, 0)
            self.assertTrue(torch.isfinite(value))

    def test_renderer_accepts_override_colors_parameter(self) -> None:
        signature = inspect.signature(GaussianRenderer.render)
        self.assertIn("override_colors", signature.parameters)
        self.assertEqual(signature.parameters["override_colors"].default, None)

    def test_medium_update_schedule(self) -> None:
        config = SeaSplatConfig(seathru_from_step=10, update_bs_at_interval=4, update_bs_at_count=2)

        self.assertFalse(should_update_medium(config, 10))
        self.assertTrue(should_update_medium(config, 11))
        self.assertTrue(should_update_medium(config, 12))
        self.assertFalse(should_update_medium(config, 13))
        self.assertFalse(should_update_medium(config, 14))
        self.assertTrue(should_update_medium(config, 15))


if __name__ == "__main__":
    unittest.main()

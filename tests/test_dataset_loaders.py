from __future__ import annotations

import unittest

from utils.dataset_loaders import _scaled_resolution, _split_indices


class DatasetLoaderResolutionTest(unittest.TestCase):
    def test_factor_scales_resolution_and_intrinsics(self) -> None:
        width, height, scale = _scaled_resolution(1776, 1182, factor=4)

        self.assertEqual((width, height), (444, 295))
        self.assertAlmostEqual(scale, 0.25)

    def test_target_height_takes_precedence_over_factor(self) -> None:
        width, height, scale = _scaled_resolution(1776, 1182, factor=4, target_height=720)

        self.assertEqual((width, height), (1082, 720))
        self.assertAlmostEqual(scale, 720 / 1182)

    def test_target_width_takes_precedence_over_factor(self) -> None:
        width, height, scale = _scaled_resolution(1776, 1182, factor=4, target_width=720)

        self.assertEqual((width, height), (720, 479))
        self.assertAlmostEqual(scale, 720 / 1776)

    def test_target_width_and_height_are_mutually_exclusive(self) -> None:
        with self.assertRaises(ValueError):
            _scaled_resolution(1776, 1182, factor=4, target_height=720, target_width=720)

    def test_invalid_source_resolution_fails_loudly(self) -> None:
        with self.assertRaises(ValueError):
            _scaled_resolution(0, 1182, factor=1)

    def test_holdout_offset_shifts_test_indices(self) -> None:
        values = list(range(10))

        self.assertEqual(_split_indices(values, "test", holdout=4, holdout_offset=0), [0, 4, 8])
        self.assertEqual(_split_indices(values, "test", holdout=4, holdout_offset=2), [2, 6])

    def test_holdout_offset_shifts_train_indices(self) -> None:
        values = list(range(10))

        self.assertEqual(_split_indices(values, "train", holdout=4, holdout_offset=2), [0, 1, 3, 4, 5, 7, 8, 9])


if __name__ == "__main__":
    unittest.main()

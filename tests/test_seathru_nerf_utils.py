from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from methods.seathru_NeRF.seathru_nerf_utils import (
    ColmapCameraRecord,
    prepare_llff_scene,
    read_cameras_binary,
    select_holdout_indices,
    write_cameras_binary,
)


class SeaThruNeRFUtilsTest(unittest.TestCase):
    def test_select_holdout_indices(self) -> None:
        self.assertEqual(select_holdout_indices(18, "test", 8, 0), [0, 8, 16])
        self.assertEqual(select_holdout_indices(18, "train", 8, 0), [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 17])
        self.assertEqual(select_holdout_indices(10, "test", 4, 1), [1, 5, 9])

    def test_prepare_llff_scene_resizes_images_and_intrinsics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            scene = Path(tmp) / "scene"
            image_dir = scene / "images_wb"
            sparse_dir = scene / "sparse" / "0"
            image_dir.mkdir(parents=True)
            sparse_dir.mkdir(parents=True)
            for index in range(3):
                Image.new("RGB", (10, 5), color=(index * 20, 40, 60)).save(image_dir / f"{index:03d}.png")
            poses_bounds = np.zeros((3, 17), dtype=np.float64)
            poses = poses_bounds[:, :15].reshape(-1, 3, 5)
            poses[:, 0, 4] = 5.0
            poses[:, 1, 4] = 10.0
            poses[:, 2, 4] = 7.0
            np.save(scene / "poses_bounds.npy", poses_bounds)
            write_cameras_binary(
                sparse_dir / "cameras.bin",
                [ColmapCameraRecord(1, "OPENCV", 10, 5, np.array([8.0, 9.0, 5.0, 2.5, 0.1, 0.2, 0.3, 0.4]))],
            )

            prepared = Path(tmp) / "prepared"
            info = prepare_llff_scene(scene, prepared, target_width=20)

            self.assertEqual(info.width, 20)
            self.assertEqual(info.height, 10)
            with Image.open(prepared / "images_wb" / "000.png") as image:
                self.assertEqual(image.size, (20, 10))
            scaled_poses = np.load(prepared / "poses_bounds.npy")[:, :15].reshape(-1, 3, 5)
            self.assertEqual(float(scaled_poses[0, 0, 4]), 10.0)
            self.assertEqual(float(scaled_poses[0, 1, 4]), 20.0)
            self.assertEqual(float(scaled_poses[0, 2, 4]), 14.0)
            camera = read_cameras_binary(prepared / "sparse" / "0" / "cameras.bin")[0]
            self.assertEqual(camera.width, 20)
            self.assertEqual(camera.height, 10)
            np.testing.assert_allclose(camera.params[:4], [16.0, 18.0, 10.0, 5.0])

    def test_prepare_llff_scene_can_keep_original_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            scene = Path(tmp) / "scene"
            image_dir = scene / "Images_wb"
            sparse_dir = scene / "sparse" / "0"
            image_dir.mkdir(parents=True)
            sparse_dir.mkdir(parents=True)
            Image.new("RGB", (11, 7)).save(image_dir / "000.jpg")
            poses_bounds = np.zeros((1, 17), dtype=np.float64)
            poses = poses_bounds[:, :15].reshape(-1, 3, 5)
            poses[:, 0, 4] = 7.0
            poses[:, 1, 4] = 11.0
            poses[:, 2, 4] = 9.0
            np.save(scene / "poses_bounds.npy", poses_bounds)
            write_cameras_binary(sparse_dir / "cameras.bin", [ColmapCameraRecord(1, "PINHOLE", 11, 7, np.array([9.0, 10.0, 5.5, 3.5]))])

            prepared = Path(tmp) / "prepared"
            info = prepare_llff_scene(scene, prepared, target_width=720, keep_original_resolution=True)

            self.assertEqual((info.width, info.height), (11, 7))
            with Image.open(prepared / "images_wb" / "000.jpg") as image:
                self.assertEqual(image.size, (11, 7))


if __name__ == "__main__":
    unittest.main()

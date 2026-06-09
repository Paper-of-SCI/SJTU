from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from methods.watersplatting.render_watersplatting_views import collect_render_pairs, save_metrics_csv
from methods.watersplatting.watersplatting_utils import (
    ColmapCameraRecord,
    prepare_colmap_scene,
    read_cameras_binary,
    select_holdout_indices,
    write_cameras_binary,
)


class WaterSplattingUtilsTest(unittest.TestCase):
    def test_select_holdout_indices(self) -> None:
        self.assertEqual(select_holdout_indices(18, "test", 8, 0), [0, 8, 16])
        self.assertEqual(select_holdout_indices(10, "test", 4, 1), [1, 5, 9])
        self.assertEqual(select_holdout_indices(10, "train", 4, 1), [0, 2, 3, 4, 6, 7, 8])

    def test_prepare_colmap_scene_resizes_intrinsics_and_split_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            scene = Path(tmp) / "scene"
            image_dir = scene / "Images_wb"
            sparse_dir = scene / "sparse" / "0"
            image_dir.mkdir(parents=True)
            sparse_dir.mkdir(parents=True)
            for index in range(10):
                Image.new("RGB", (10, 5), color=(index * 10, 40, 60)).save(image_dir / f"{index:03d}.png")
            write_cameras_binary(
                sparse_dir / "cameras.bin",
                [ColmapCameraRecord(1, "OPENCV", 10, 5, np.array([8.0, 9.0, 5.0, 2.5, 0.1, 0.2, 0.3, 0.4]))],
            )

            prepared = Path(tmp) / "prepared"
            info = prepare_colmap_scene(scene, prepared, target_width=20, holdout=4, holdout_offset=1)

            self.assertEqual(info.width, 20)
            self.assertEqual(info.height, 10)
            self.assertEqual(info.test_count, 3)
            with Image.open(prepared / "images" / "000.png") as image:
                self.assertEqual(image.size, (20, 10))
            camera = read_cameras_binary(prepared / "sparse" / "0" / "cameras.bin")[0]
            self.assertEqual(camera.width, 20)
            self.assertEqual(camera.height, 10)
            np.testing.assert_allclose(camera.params[:4], [16.0, 18.0, 10.0, 5.0])
            self.assertEqual((prepared / "test_list.txt").read_text(encoding="utf-8").splitlines(), ["001.png", "005.png", "009.png"])
            self.assertEqual(
                (prepared / "train_list.txt").read_text(encoding="utf-8").splitlines(),
                ["000.png", "002.png", "003.png", "004.png", "006.png", "007.png", "008.png"],
            )

    def test_prepare_colmap_scene_can_keep_original_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            scene = Path(tmp) / "scene"
            image_dir = scene / "images_wb"
            sparse_dir = scene / "sparse" / "0"
            image_dir.mkdir(parents=True)
            sparse_dir.mkdir(parents=True)
            Image.new("RGB", (11, 7)).save(image_dir / "000.jpg")
            write_cameras_binary(sparse_dir / "cameras.bin", [ColmapCameraRecord(1, "PINHOLE", 11, 7, np.array([9.0, 10.0, 5.5, 3.5]))])

            prepared = Path(tmp) / "prepared"
            info = prepare_colmap_scene(scene, prepared, target_width=720, holdout=8, holdout_offset=0, keep_original_resolution=True)

            self.assertEqual((info.width, info.height), (11, 7))
            with Image.open(prepared / "images" / "000.jpg") as image:
                self.assertEqual(image.size, (11, 7))

    def test_collect_render_pairs_and_save_metrics_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "renders"
            rgb_dir = root / "test" / "rgb"
            gt_dir = root / "test" / "gt-rgb"
            rgb_dir.mkdir(parents=True)
            gt_dir.mkdir(parents=True)
            Image.new("RGB", (4, 3), color=(10, 20, 30)).save(rgb_dir / "000.png")
            Image.new("RGB", (4, 3), color=(10, 20, 30)).save(gt_dir / "000.png")

            pairs = collect_render_pairs(root)
            self.assertEqual(len(pairs), 1)
            self.assertEqual(pairs[0][0], "000.png")
            metrics_path = Path(tmp) / "metrics.csv"
            save_metrics_csv(metrics_path, [{"image": "000.png", "psnr": 99.0, "ssim": 1.0, "l1": 0.0, "lpips": None}], include_lpips=False)
            text = metrics_path.read_text(encoding="utf-8")
            self.assertIn("image,psnr,ssim,l1", text)
            self.assertIn("000.png", text)


if __name__ == "__main__":
    unittest.main()

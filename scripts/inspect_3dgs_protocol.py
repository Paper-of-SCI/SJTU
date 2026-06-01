"""Inspect the dataset protocol used by the local 3DGS benchmark.

Owner:
    scripts boundary for read-only experiment diagnostics.

Responsibility:
    Report COLMAP camera model, selected image directory, train/test split,
    target resolution, and point count before launching a benchmark run.

Input:
    One or more COLMAP scene directories.

Output:
    Human-readable protocol summary on stdout. The script does not write files.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.colmap_reader import read_colmap_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect COLMAP 3DGS data protocol before training.")
    parser.add_argument("scenes", nargs="+", help="Scene directories to inspect.")
    parser.add_argument("--factor", type=int, default=1, help="Fallback integer downscale factor.")
    parser.add_argument("--target-height", type=int, default=0, help="Target image height; mutually exclusive with --target-width.")
    parser.add_argument("--target-width", type=int, default=0, help="Target image width; mutually exclusive with --target-height.")
    parser.add_argument("--holdout", type=int, default=8, help="Every Nth image is selected for test.")
    parser.add_argument("--holdout-offset", type=int, default=0, help="Offset for every-Nth holdout selection.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.target_height > 0 and args.target_width > 0:
        raise ValueError("target_height 和 target_width 只能设置一个")
    for scene_arg in args.scenes:
        inspect_scene(
            Path(scene_arg).expanduser(),
            factor=args.factor,
            target_height=args.target_height,
            target_width=args.target_width,
            holdout=args.holdout,
            holdout_offset=args.holdout_offset,
        )


def inspect_scene(
    scene_dir: Path,
    factor: int,
    target_height: int,
    target_width: int,
    holdout: int,
    holdout_offset: int,
) -> None:
    if not scene_dir.exists():
        raise FileNotFoundError(f"找不到场景目录: {scene_dir}")
    sparse_dir = scene_dir / "sparse" / "0"
    cameras, images, points = read_colmap_model(str(sparse_dir))
    ordered_images = [images[image_id] for image_id in sorted(images, key=lambda item: images[item].name)]
    test_names = [
        image.name
        for index, image in enumerate(ordered_images)
        if holdout > 0 and (index - holdout_offset) % holdout == 0
    ]
    train_count = len(ordered_images) - len(test_names) if holdout > 0 else len(ordered_images)
    image_dir = find_image_dir(scene_dir, factor, target_height, target_width)

    print(f"scene: {scene_dir}")
    print(f"image_dir: {image_dir}")
    print(f"images: total={len(ordered_images)} train={train_count} test={len(test_names)}")
    print(f"holdout: interval={holdout} offset={holdout_offset}")
    print(f"test_images: {', '.join(test_names) if test_names else '<none>'}")
    print(f"points3D: {len(points)}")
    for camera_id, camera in sorted(cameras.items()):
        width, height, scale = scaled_resolution(camera.width, camera.height, factor, target_height, target_width)
        print(
            "camera: "
            f"id={camera_id} model={camera.model} source={camera.width}x{camera.height} "
            f"target={width}x{height} scale={scale:.6f} "
            f"fx={camera.fx * scale:.6f} fy={camera.fy * scale:.6f} "
            f"cx={camera.cx * scale:.6f} cy={camera.cy * scale:.6f}"
        )
    print()


def find_image_dir(scene_dir: Path, factor: int, target_height: int, target_width: int) -> Path:
    names: list[str] = []
    if target_height <= 0 and target_width <= 0 and factor > 1:
        names.extend([f"images_{factor}", f"Images_{factor}"])
    names.extend(["images_wb", "Images_wb", "images", "Images"])
    for name in names:
        path = scene_dir / name
        if path.is_dir():
            return path
    raise FileNotFoundError(f"未找到图像目录: {scene_dir}")


def scaled_resolution(
    width: int,
    height: int,
    factor: int,
    target_height: int,
    target_width: int,
) -> tuple[int, int, float]:
    if width <= 0 or height <= 0:
        raise ValueError(f"图像尺寸非法: {width}x{height}")
    if target_height > 0:
        scale = float(target_height) / float(height)
        return max(int(round(width * scale)), 1), max(int(target_height), 1), scale
    if target_width > 0:
        scale = float(target_width) / float(width)
        return max(int(target_width), 1), max(int(round(height * scale)), 1), scale
    safe_factor = max(int(factor), 1)
    scale = 1.0 / float(safe_factor)
    return max(int(width // safe_factor), 1), max(int(height // safe_factor), 1), scale


if __name__ == "__main__":
    main()

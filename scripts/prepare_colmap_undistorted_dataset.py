from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path


DEFAULT_SCENES = ("Curasao", "IUI3-RedSea", "JapaneseGradens-RedSea", "Panama")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare COLMAP-undistorted SeaThru-NeRF scenes for 3DGS experiments.")
    parser.add_argument("--data-root", default="src/datasets/SeathruNeRF_dataset", help="Root containing original scene folders.")
    parser.add_argument("--out-root", default="outputs/aligned_datasets/SeathruNeRF_undistorted", help="Output root for undistorted scenes.")
    parser.add_argument("--scenes", nargs="+", default=list(DEFAULT_SCENES), help="Scene folders to process.")
    parser.add_argument("--image-dir", default="images_wb", help="Input image directory inside each scene.")
    parser.add_argument("--max-image-size", type=int, default=2000, help="COLMAP undistortion max image size; 0 keeps COLMAP default.")
    parser.add_argument("--colmap-bin", default="colmap", help="COLMAP executable.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output scenes.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    for scene in args.scenes:
        prepare_scene(
            colmap_bin=args.colmap_bin,
            source_scene=data_root / scene,
            output_scene=out_root / scene,
            image_dir=args.image_dir,
            max_image_size=args.max_image_size,
            overwrite=args.overwrite,
        )


def prepare_scene(
    *,
    colmap_bin: str,
    source_scene: Path,
    output_scene: Path,
    image_dir: str,
    max_image_size: int,
    overwrite: bool,
) -> None:
    image_path = find_image_dir(source_scene, image_dir)
    sparse_path = source_scene / "sparse" / "0"
    if not sparse_path.is_dir():
        raise FileNotFoundError(f"COLMAP sparse/0 目录不存在: {sparse_path}")
    if output_scene.exists():
        if not overwrite:
            raise FileExistsError(f"输出目录已存在，传入 --overwrite 覆盖: {output_scene}")
        shutil.rmtree(output_scene)

    with tempfile.TemporaryDirectory(prefix=f"{source_scene.name}_undistort_") as temp_dir:
        temp_output = Path(temp_dir) / source_scene.name
        command = [
            colmap_bin,
            "image_undistorter",
            "--image_path",
            str(image_path),
            "--input_path",
            str(sparse_path),
            "--output_path",
            str(temp_output),
            "--output_type",
            "COLMAP",
        ]
        if max_image_size > 0:
            command.extend(["--max_image_size", str(max_image_size)])
        print("运行:", " ".join(command))
        subprocess.run(command, check=True)
        normalize_colmap_undistort_output(source_scene, temp_output, output_scene)
    print(f"完成: {output_scene}")


def find_image_dir(source_scene: Path, preferred_name: str) -> Path:
    for name in (preferred_name, "images_wb", "Images_wb", "images", "Images"):
        image_path = source_scene / name
        if image_path.is_dir():
            return image_path
    raise FileNotFoundError(f"输入图像目录不存在: {source_scene}")


def normalize_colmap_undistort_output(source_scene: Path, colmap_output: Path, output_scene: Path) -> None:
    output_scene.mkdir(parents=True)
    shutil.copytree(colmap_output / "images", output_scene / "images")
    sparse_zero = output_scene / "sparse" / "0"
    sparse_zero.mkdir(parents=True)
    for file_name in ("cameras.bin", "images.bin", "points3D.bin", "frames.bin", "rigs.bin"):
        source_file = colmap_output / "sparse" / file_name
        if source_file.exists():
            shutil.copy2(source_file, sparse_zero / file_name)
    poses_bounds = source_scene / "poses_bounds.npy"
    if poses_bounds.exists():
        shutil.copy2(poses_bounds, output_scene / "poses_bounds.npy")


if __name__ == "__main__":
    main()

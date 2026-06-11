"""Utilities for running the official WaterSplatting implementation."""

from __future__ import annotations

import json
import os
import shutil
import struct
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
METHOD_DIR = Path(__file__).resolve().parent
VENDOR_DIR = METHOD_DIR / "vendor"
VENDOR_COMMIT = "0c2d94383438f8e948892e7aaeeed16577b3c2b6"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
NERFSTUDIO_METHOD_CONFIGS = ",".join(
    [
        "water-splatting=water_splatting.water_splatting_config:water_splatting_method",
        "water-splatting-big=water_splatting.water_splatting_config:water_splatting_method_big",
    ]
)

CAMERA_MODEL_IDS = {
    0: "SIMPLE_PINHOLE",
    1: "PINHOLE",
    2: "SIMPLE_RADIAL",
    3: "RADIAL",
    4: "OPENCV",
    5: "OPENCV_FISHEYE",
    6: "FULL_OPENCV",
    7: "FOV",
    8: "SIMPLE_RADIAL_FISHEYE",
    9: "RADIAL_FISHEYE",
    10: "THIN_PRISM_FISHEYE",
}
CAMERA_MODEL_NAMES = {name: model_id for model_id, name in CAMERA_MODEL_IDS.items()}
CAMERA_MODEL_NUM_PARAMS = {
    "SIMPLE_PINHOLE": 3,
    "PINHOLE": 4,
    "SIMPLE_RADIAL": 4,
    "RADIAL": 5,
    "OPENCV": 8,
    "OPENCV_FISHEYE": 8,
    "FULL_OPENCV": 12,
    "FOV": 5,
    "SIMPLE_RADIAL_FISHEYE": 4,
    "RADIAL_FISHEYE": 5,
    "THIN_PRISM_FISHEYE": 12,
}


@dataclass(frozen=True)
class ColmapCameraRecord:
    camera_id: int
    model: str
    width: int
    height: int
    params: np.ndarray


@dataclass(frozen=True)
class PreparedSceneInfo:
    source_data: str
    prepared_data: str
    image_count: int
    train_count: int
    test_count: int
    source_width: int
    source_height: int
    width: int
    height: int
    scale_x: float
    scale_y: float
    target_width: int
    keep_original_resolution: bool
    holdout: int
    holdout_offset: int

    def to_dict(self) -> dict:
        return asdict(self)


def resolve_input_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.exists():
        return candidate.resolve()
    root_candidate = ROOT / candidate
    if root_candidate.exists():
        return root_candidate.resolve()
    raise FileNotFoundError(f"找不到路径: {path}；也尝试过 {root_candidate}")


def resolve_output_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return ROOT / candidate


def scene_output_dir(data_dir: Path, target_width: int, keep_original_resolution: bool) -> Path:
    suffix = "original" if keep_original_resolution or target_width <= 0 else f"{int(target_width)}w"
    return ROOT / "outputs" / f"watersplatting_{data_dir.name}_{suffix}"


def select_holdout_indices(count: int, split: str, holdout: int, holdout_offset: int = 0) -> list[int]:
    values = list(range(int(count)))
    if split in {"test", "val"}:
        if holdout <= 0:
            return []
        return [index for index in values if (index - int(holdout_offset)) % int(holdout) == 0]
    if holdout <= 0:
        return values
    return [index for index in values if (index - int(holdout_offset)) % int(holdout) != 0]


def prepare_colmap_scene(
    data_dir: Path,
    prepared_dir: Path,
    target_width: int,
    holdout: int,
    holdout_offset: int,
    keep_original_resolution: bool = False,
    overwrite: bool = False,
) -> PreparedSceneInfo:
    """Create a nerfstudio COLMAP scene with images and intrinsics scaled together."""
    data_dir = data_dir.resolve()
    prepared_dir = prepared_dir.resolve()
    image_dir = find_image_dir(data_dir)
    image_paths = list_image_paths(image_dir)
    if not image_paths:
        raise FileNotFoundError(f"图像目录为空: {image_dir}")
    sparse_dir = data_dir / "sparse" / "0"
    if not sparse_dir.is_dir():
        raise FileNotFoundError(f"WaterSplatting 需要 COLMAP sparse/0: {sparse_dir}")

    with Image.open(image_paths[0]) as first_image:
        source_width, source_height = first_image.size
    width, height = target_size(source_width, source_height, target_width, keep_original_resolution)
    scale_x = float(width) / float(source_width)
    scale_y = float(height) / float(source_height)
    train_indices = select_holdout_indices(len(image_paths), "train", holdout, holdout_offset)
    test_indices = select_holdout_indices(len(image_paths), "test", holdout, holdout_offset)

    info = PreparedSceneInfo(
        source_data=str(data_dir),
        prepared_data=str(prepared_dir),
        image_count=len(image_paths),
        train_count=len(train_indices),
        test_count=len(test_indices),
        source_width=int(source_width),
        source_height=int(source_height),
        width=int(width),
        height=int(height),
        scale_x=float(scale_x),
        scale_y=float(scale_y),
        target_width=int(target_width),
        keep_original_resolution=bool(keep_original_resolution),
        holdout=int(holdout),
        holdout_offset=int(holdout_offset),
    )
    manifest_path = prepared_dir / "watersplatting_prepared_manifest.json"
    train_names = [image_paths[index].name for index in train_indices]
    test_names = [image_paths[index].name for index in test_indices]
    if manifest_path.exists() and not overwrite:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if _manifest_matches(existing, info):
            write_split_file(prepared_dir / "train_list.txt", train_names)
            write_split_file(prepared_dir / "test_list.txt", test_names)
            write_split_file(prepared_dir / "val_list.txt", test_names)
            return info
        raise ValueError(f"prepared data 已存在但配置不同: {prepared_dir}；需要重建请加 --overwrite-prepared-data")

    if prepared_dir.exists() and overwrite:
        shutil.rmtree(prepared_dir)
    prepared_image_dir = prepared_dir / "images"
    prepared_sparse_dir = prepared_dir / "sparse" / "0"
    prepared_image_dir.mkdir(parents=True, exist_ok=True)
    prepared_sparse_dir.parent.mkdir(parents=True, exist_ok=True)

    for image_path in image_paths:
        resize_or_copy_image(image_path, prepared_image_dir / image_path.name, width, height)

    if prepared_sparse_dir.exists():
        shutil.rmtree(prepared_sparse_dir)
    shutil.copytree(sparse_dir, prepared_sparse_dir)
    scale_colmap_cameras(prepared_sparse_dir, width, height, scale_x, scale_y)
    write_split_file(prepared_dir / "train_list.txt", train_names)
    write_split_file(prepared_dir / "test_list.txt", test_names)
    write_split_file(prepared_dir / "val_list.txt", test_names)
    manifest_path.write_text(json.dumps(info.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return info


def find_image_dir(data_dir: Path) -> Path:
    for name in ("images_wb", "Images_wb", "images", "Images"):
        candidate = data_dir / name
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"未找到图像目录: {data_dir}")


def list_image_paths(image_dir: Path) -> list[Path]:
    return sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)


def target_size(source_width: int, source_height: int, target_width: int, keep_original_resolution: bool) -> tuple[int, int]:
    if keep_original_resolution or int(target_width) <= 0:
        return int(source_width), int(source_height)
    width = int(target_width)
    if width <= 0:
        raise ValueError("target_width 必须为正数，或使用 --keep-original-resolution")
    scale = float(width) / float(source_width)
    height = max(int(round(float(source_height) * scale)), 1)
    return width, height


def resize_or_copy_image(source: Path, target: Path, width: int, height: int) -> None:
    with Image.open(source) as image:
        image = image.convert("RGB")
        if image.size != (width, height):
            image = image.resize((width, height), Image.Resampling.LANCZOS)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.suffix.lower() in {".jpg", ".jpeg"}:
            image.save(target, quality=95)
        else:
            image.save(target)


def write_split_file(path: Path, image_names: list[str]) -> None:
    path.write_text("\n".join(image_names) + ("\n" if image_names else ""), encoding="utf-8")


def scale_colmap_cameras(sparse_dir: Path, width: int, height: int, scale_x: float, scale_y: float) -> None:
    cameras_bin = sparse_dir / "cameras.bin"
    cameras_txt = sparse_dir / "cameras.txt"
    found = False
    if cameras_bin.exists():
        cameras = read_cameras_binary(cameras_bin)
        write_cameras_binary(cameras_bin, [_scale_camera(camera, width, height, scale_x, scale_y) for camera in cameras])
        found = True
    if cameras_txt.exists():
        cameras = read_cameras_text(cameras_txt)
        write_cameras_text(cameras_txt, [_scale_camera(camera, width, height, scale_x, scale_y) for camera in cameras])
        found = True
    if not found:
        raise FileNotFoundError(f"未找到 COLMAP cameras.bin 或 cameras.txt: {sparse_dir}")


def read_cameras_binary(path: Path) -> list[ColmapCameraRecord]:
    cameras = []
    with open(path, "rb") as handle:
        count = struct.unpack("<Q", handle.read(8))[0]
        for _ in range(count):
            camera_id, model_id, width, height = struct.unpack("<IiQQ", handle.read(24))
            model = CAMERA_MODEL_IDS[model_id]
            num_params = CAMERA_MODEL_NUM_PARAMS[model]
            params = np.array(struct.unpack("<" + "d" * num_params, handle.read(8 * num_params)), dtype=np.float64)
            cameras.append(ColmapCameraRecord(int(camera_id), model, int(width), int(height), params))
    return cameras


def write_cameras_binary(path: Path, cameras: list[ColmapCameraRecord]) -> None:
    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", len(cameras)))
        for camera in sorted(cameras, key=lambda item: item.camera_id):
            model_id = CAMERA_MODEL_NAMES[camera.model]
            handle.write(struct.pack("<IiQQ", int(camera.camera_id), int(model_id), int(camera.width), int(camera.height)))
            params = [float(value) for value in camera.params]
            handle.write(struct.pack("<" + "d" * len(params), *params))


def read_cameras_text(path: Path) -> list[ColmapCameraRecord]:
    cameras = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.split()
            cameras.append(ColmapCameraRecord(int(parts[0]), parts[1], int(parts[2]), int(parts[3]), np.array(parts[4:], dtype=np.float64)))
    return cameras


def write_cameras_text(path: Path, cameras: list[ColmapCameraRecord]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("# Camera list with one line of data per camera:\n")
        handle.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        handle.write(f"# Number of cameras: {len(cameras)}\n")
        for camera in sorted(cameras, key=lambda item: item.camera_id):
            params = " ".join(f"{float(value):.17g}" for value in camera.params)
            handle.write(f"{camera.camera_id} {camera.model} {camera.width} {camera.height} {params}\n")


def _scale_camera(camera: ColmapCameraRecord, width: int, height: int, scale_x: float, scale_y: float) -> ColmapCameraRecord:
    params = np.asarray(camera.params, dtype=np.float64).copy()
    focal_scale = 0.5 * (float(scale_x) + float(scale_y))
    if camera.model == "SIMPLE_PINHOLE":
        params[0] *= focal_scale
        params[1] *= scale_x
        params[2] *= scale_y
    elif camera.model == "PINHOLE":
        params[0] *= scale_x
        params[1] *= scale_y
        params[2] *= scale_x
        params[3] *= scale_y
    elif camera.model in {"SIMPLE_RADIAL", "SIMPLE_RADIAL_FISHEYE"}:
        params[0] *= focal_scale
        params[1] *= scale_x
        params[2] *= scale_y
    elif camera.model in {"RADIAL", "RADIAL_FISHEYE", "FOV"}:
        params[0] *= focal_scale
        params[1] *= scale_x
        params[2] *= scale_y
    elif camera.model in {"OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV", "THIN_PRISM_FISHEYE"}:
        params[0] *= scale_x
        params[1] *= scale_y
        params[2] *= scale_x
        params[3] *= scale_y
    else:
        raise ValueError(f"暂不支持缩放 COLMAP camera model: {camera.model}")
    return ColmapCameraRecord(camera.camera_id, camera.model, int(width), int(height), params)


def check_watersplatting_python(python_executable: str) -> None:
    script = """
import importlib.util
from pathlib import Path
import shutil
import sys
required = {
    "nerfstudio": "nerfstudio==1.1.4",
    "pytorch_msssim": "pytorch-msssim",
    "sklearn": "scikit-learn",
    "tinycudann": "tiny-cuda-nn",
}
missing = [package for module, package in required.items() if importlib.util.find_spec(module) is None]
water_spec = importlib.util.find_spec("water_splatting")
package_dirs = []
if water_spec is None:
    missing.append("water_splatting")
else:
    package_dirs = [Path(path) for path in water_spec.submodule_search_locations or []]
    if not any((path / "water_splatting_config.py").exists() for path in package_dirs):
        missing.append("water_splatting.water_splatting_config")
try:
    import torch
except Exception:
    torch = None
    missing.append("torch==2.1.2+cu118")
if torch is not None and not torch.cuda.is_available():
    missing.append("CUDA GPU")
has_prebuilt_cuda = any(
    list(path.glob("csrc*.so")) or list(path.glob("csrc*.pyd")) or list(path.glob("csrc*.dll"))
    for path in package_dirs
)
has_jit_cuda = False
if torch is not None:
    try:
        from torch.utils.cpp_extension import _get_build_directory
        build_dir = Path(_get_build_directory("water_splatting_cuda", verbose=False))
        has_jit_cuda = (build_dir / "water_splatting_cuda.so").exists() or (build_dir / "water_splatting_cuda.lib").exists()
    except Exception:
        has_jit_cuda = False
if not (has_prebuilt_cuda or has_jit_cuda or shutil.which("nvcc") is not None):
    missing.append("water_splatting CUDA extension or nvcc")
if missing:
    raise SystemExit("missing:" + ",".join(missing))
"""
    env = watersplatting_env()
    result = subprocess.run([python_executable, "-c", script], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    if result.returncode != 0:
        missing_text = result.stderr.strip() or result.stdout.strip()
        if missing_text.startswith("missing:"):
            missing_text = missing_text.removeprefix("missing:")
        raise RuntimeError(
            "WaterSplatting 官方代码需要单独 nerfstudio/CUDA 环境；"
            f"--python={python_executable} 缺少依赖或不可用：{missing_text}"
        )


def watersplatting_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(VENDOR_DIR) + os.pathsep + env.get("PYTHONPATH", "")
    env["NERFSTUDIO_METHOD_CONFIGS"] = NERFSTUDIO_METHOD_CONFIGS
    env.setdefault("WANDB_MODE", "disabled")
    return env


def run_watersplatting_command(python_executable: str, args: list[str]) -> None:
    subprocess.run([python_executable, *args], cwd=VENDOR_DIR, env=watersplatting_env(), check=True)


def python_default() -> str:
    return sys.executable


def _manifest_matches(existing: dict, info: PreparedSceneInfo) -> bool:
    expected = info.to_dict()
    keys = [
        "source_data",
        "image_count",
        "train_count",
        "test_count",
        "source_width",
        "source_height",
        "width",
        "height",
        "target_width",
        "keep_original_resolution",
        "holdout",
        "holdout_offset",
    ]
    return all(existing.get(key) == expected.get(key) for key in keys)

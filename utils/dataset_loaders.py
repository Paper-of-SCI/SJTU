"""Dataset loaders that return data only, without constructing models."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

from utils.camera_utils import colmap_image_to_c2w, estimate_scene_extent
from utils.colmap_reader import read_colmap_model
from utils.image_utils import load_image


@dataclass(frozen=True)
class SceneData:
    image_paths: list[str]
    images: Optional[np.ndarray]
    c2w_matrices: np.ndarray
    fx: np.ndarray
    fy: np.ndarray
    cx: np.ndarray
    cy: np.ndarray
    width: int
    height: int
    near: float
    far: float
    scene_extent: float
    point_cloud_xyz: Optional[np.ndarray] = None
    point_cloud_rgb: Optional[np.ndarray] = None


def load_colmap_dataset(
    data_dir: str,
    split: str = "train",
    load_images: bool = True,
    factor: int = 1,
    target_height: int = 0,
    target_width: int = 0,
    holdout: int = 8,
    holdout_offset: int = 0,
    opengl: bool = True,
) -> SceneData:
    """Load a COLMAP scene from ``data_dir/sparse/0``."""
    cameras, images, points = read_colmap_model(os.path.join(data_dir, "sparse", "0"))
    ids = sorted(images.keys(), key=lambda item: images[item].name)
    ids = _split_indices(ids, split, holdout, holdout_offset)
    if not ids:
        raise ValueError(f"empty {split} split for {data_dir}")

    image_dir = _find_image_dir(data_dir, 1 if target_height > 0 or target_width > 0 else factor)
    c2ws = []
    paths = []
    fxs, fys, cxs, cys = [], [], [], []
    width = height = 0
    for image_id in ids:
        image = images[image_id]
        camera = cameras[image.camera_id]
        width, height, scale = _scaled_resolution(int(camera.width), int(camera.height), factor, target_height, target_width)
        fxs.append(camera.fx * scale)
        fys.append(camera.fy * scale)
        cxs.append(camera.cx * scale)
        cys.append(camera.cy * scale)
        c2ws.append(colmap_image_to_c2w(image.qvec, image.tvec, opengl=opengl))
        paths.append(os.path.join(image_dir, image.name))

    loaded = _load_images(paths, width, height) if load_images else None
    xyz = np.stack([point.xyz for point in points.values()], axis=0).astype(np.float32) if points else None
    rgb = np.stack([point.rgb for point in points.values()], axis=0).astype(np.uint8) if points else None
    c2w_matrices = np.stack(c2ws, axis=0)
    near, far = _estimate_near_far(c2w_matrices, xyz)
    return SceneData(
        image_paths=paths,
        images=loaded,
        c2w_matrices=c2w_matrices,
        fx=np.asarray(fxs, dtype=np.float64),
        fy=np.asarray(fys, dtype=np.float64),
        cx=np.asarray(cxs, dtype=np.float64),
        cy=np.asarray(cys, dtype=np.float64),
        width=width,
        height=height,
        near=near,
        far=far,
        scene_extent=estimate_scene_extent(c2w_matrices),
        point_cloud_xyz=xyz,
        point_cloud_rgb=rgb,
    )


def load_llff_dataset(
    data_dir: str,
    split: str = "train",
    load_images: bool = True,
    factor: int = 1,
    target_height: int = 0,
    target_width: int = 0,
    holdout: int = 8,
    holdout_offset: int = 0,
) -> SceneData:
    """Load LLFF ``poses_bounds.npy`` data, with COLMAP fallback for point cloud."""
    poses_path = os.path.join(data_dir, "poses_bounds.npy")
    if not os.path.exists(poses_path):
        return load_colmap_dataset(
            data_dir,
            split=split,
            load_images=load_images,
            factor=factor,
            target_height=target_height,
            target_width=target_width,
            holdout=holdout,
            holdout_offset=holdout_offset,
        )

    poses_bounds = np.load(poses_path)
    poses = poses_bounds[:, :15].reshape(-1, 3, 5)
    bounds = poses_bounds[:, 15:17]
    hwf = poses[:, :, 4]
    c2w = poses[:, :, :4]
    c2w = c2w @ np.diag([1.0, -1.0, -1.0, 1.0])
    c2w = np.concatenate([c2w, np.broadcast_to(np.array([0, 0, 0, 1.0]), (len(c2w), 1, 4))], axis=1)

    image_dir = _find_image_dir(data_dir, 1 if target_height > 0 or target_width > 0 else factor)
    names = sorted(name for name in os.listdir(image_dir) if name.lower().endswith((".png", ".jpg", ".jpeg")))
    count = min(len(names), len(c2w))
    indices = _split_indices(list(range(count)), split, holdout, holdout_offset)
    width, height, scale = _scaled_resolution(int(hwf[0, 1]), int(hwf[0, 0]), factor, target_height, target_width)
    focal = hwf[:count, 2] * scale
    paths = [os.path.join(image_dir, names[i]) for i in indices]
    loaded = _load_images(paths, width, height) if load_images else None
    selected_c2w = c2w[indices]
    xyz, rgb = _try_colmap_points(data_dir)
    return SceneData(
        image_paths=paths,
        images=loaded,
        c2w_matrices=selected_c2w,
        fx=focal[indices],
        fy=focal[indices],
        cx=np.full(len(indices), width / 2.0),
        cy=np.full(len(indices), height / 2.0),
        width=width,
        height=height,
        near=float(bounds[:, 0].min() * 0.9),
        far=float(bounds[:, 1].max() * 1.1),
        scene_extent=estimate_scene_extent(selected_c2w),
        point_cloud_xyz=xyz,
        point_cloud_rgb=rgb,
    )


def _split_indices(values: list, split: str, holdout: int, holdout_offset: int = 0) -> list:
    offset = int(holdout_offset)
    if split in {"test", "val"}:
        return [value for i, value in enumerate(values) if holdout > 0 and (i - offset) % holdout == 0]
    return [value for i, value in enumerate(values) if holdout <= 0 or (i - offset) % holdout != 0]


def _scaled_resolution(width: int, height: int, factor: int, target_height: int = 0, target_width: int = 0) -> tuple[int, int, float]:
    if width <= 0 or height <= 0:
        raise ValueError(f"图像尺寸非法: {width}x{height}")
    if target_height > 0 and target_width > 0:
        raise ValueError("target_height 和 target_width 只能设置一个")
    if target_height > 0:
        scale = float(target_height) / float(height)
        return max(int(round(width * scale)), 1), max(int(target_height), 1), scale
    if target_width > 0:
        scale = float(target_width) / float(width)
        return max(int(target_width), 1), max(int(round(height * scale)), 1), scale
    safe_factor = max(int(factor), 1)
    scale = 1.0 / float(safe_factor)
    return max(int(width // safe_factor), 1), max(int(height // safe_factor), 1), scale


def _find_image_dir(data_dir: str, factor: int) -> str:
    names = []
    if factor > 1:
        names.extend([f"images_{factor}", f"Images_{factor}"])
    names.extend(["images_wb", "Images_wb", "images", "Images"])
    for name in names:
        path = os.path.join(data_dir, name)
        if os.path.isdir(path):
            return path
    raise FileNotFoundError(f"未找到图像目录: {data_dir}")


def _load_images(paths: list[str], width: int, height: int) -> np.ndarray:
    return np.stack([load_image(path, as_float=True, resize=(width, height)) for path in paths], axis=0)


def _estimate_near_far(c2w: np.ndarray, xyz: Optional[np.ndarray]) -> tuple[float, float]:
    if xyz is None or len(xyz) == 0:
        return 0.01, 1.0e10
    centers = c2w[:, :3, 3]
    dists = np.concatenate([np.linalg.norm(xyz - center[None, :], axis=-1) for center in centers])
    return max(float(np.percentile(dists, 0.1)) * 0.5, 1e-3), float(np.percentile(dists, 99.9)) * 1.5


def _try_colmap_points(data_dir: str) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    try:
        _, _, points = read_colmap_model(os.path.join(data_dir, "sparse", "0"))
    except (FileNotFoundError, ValueError):
        return None, None
    if not points:
        return None, None
    return (
        np.stack([point.xyz for point in points.values()], axis=0).astype(np.float32),
        np.stack([point.rgb for point in points.values()], axis=0).astype(np.uint8),
    )

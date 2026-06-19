from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from PIL import Image
import torch
import numpy as np


COLMAP_CAMERA_MODEL_IDS = {
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

COLMAP_CAMERA_MODEL_NUM_PARAMS = {
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
class PointCloudData:
    xyz: np.ndarray
    colors: np.ndarray


@dataclass(frozen=True)
class ColmapCameraData:
    camera_id: int
    model: str
    width: int
    height: int
    params: np.ndarray

    @property
    def fx(self) -> float:
        return float(self.params[0])

    @property
    def fy(self) -> float:
        if self.model in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL", "FOV", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE"}:
            return float(self.params[0])
        return float(self.params[1])

    @property
    def cx(self) -> float:
        if self.model == "SIMPLE_PINHOLE":
            return float(self.params[1])
        return float(self.params[2])

    @property
    def cy(self) -> float:
        if self.model == "SIMPLE_PINHOLE":
            return float(self.params[2])
        return float(self.params[3])


@dataclass(frozen=True)
class ColmapImageData:
    image_id: int
    qvec: np.ndarray
    tvec: np.ndarray
    camera_id: int
    name: str


def qvec_to_rotmat_np(qvec: np.ndarray) -> np.ndarray:
    qvec = np.asarray(qvec, dtype=np.float64)
    qvec = qvec / np.linalg.norm(qvec)
    w, x, y, z = qvec

    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
            [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
            [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def colmap_image_camera_center(image: ColmapImageData) -> np.ndarray:
    rotation_w2c = qvec_to_rotmat_np(image.qvec)
    return -rotation_w2c.T @ image.tvec


def compute_scene_extent_from_colmap_images(
    images: dict[int, ColmapImageData] | list[ColmapImageData],
    radius_scale: float = 1.1,
) -> float:
    """
    按原版 3DGS 的 NeRF++ normalization 方式计算 scene extent。

    原版逻辑是先计算所有训练相机中心的平均中心，然后取所有相机中心到平均中心的最大距离，
    最后乘以 1.1。这个 radius 就是 densification 里用的 scene_extent。
    """
    image_values = list(images.values()) if isinstance(images, dict) else list(images)
    if not image_values:
        raise ValueError("images must not be empty")

    camera_centers = np.stack(
        [colmap_image_camera_center(image) for image in image_values],
        axis=0,
    )
    center = camera_centers.mean(axis=0, keepdims=True)
    diagonal = np.linalg.norm(camera_centers - center, axis=1).max()
    return float(diagonal * radius_scale)


def load_colmap_points3d_bin(path: str | Path) -> PointCloudData:
    path = Path(path)
    point_xyz = []
    point_colors = []

    with path.open("rb") as f:
        num_points = struct.unpack("<Q", f.read(8))[0]

        for _ in range(num_points):
            _point_id = struct.unpack("<Q", f.read(8))[0]
            xyz = struct.unpack("<3d", f.read(24))
            colors = struct.unpack("<3B", f.read(3))
            _error = struct.unpack("<d", f.read(8))[0]
            track_length = struct.unpack("<Q", f.read(8))[0]
            f.read(8 * track_length)

            point_xyz.append(xyz)
            point_colors.append(colors)

    return PointCloudData(
        xyz=np.asarray(point_xyz, dtype=np.float32),
        colors=np.asarray(point_colors, dtype=np.float32) / 255.0,
    )


def load_colmap_cameras_bin(path: str | Path) -> dict[int, ColmapCameraData]:
    """读取 COLMAP cameras.bin，返回 camera_id 到相机内参的映射。"""
    path = Path(path)
    cameras: dict[int, ColmapCameraData] = {}

    with path.open("rb") as f:
        num_cameras = struct.unpack("<Q", f.read(8))[0]

        for _ in range(num_cameras):
            camera_id, model_id, width, height = struct.unpack("<IiQQ", f.read(24))
            model = COLMAP_CAMERA_MODEL_IDS[model_id]
            num_params = COLMAP_CAMERA_MODEL_NUM_PARAMS[model]
            params = np.asarray(struct.unpack("<" + "d" * num_params, f.read(8 * num_params)), dtype=np.float64)

            cameras[int(camera_id)] = ColmapCameraData(
                camera_id=int(camera_id),
                model=model,
                width=int(width),
                height=int(height),
                params=params,
            )

    return cameras


def load_colmap_images_bin(path: str | Path) -> dict[int, ColmapImageData]:
    """读取 COLMAP images.bin，返回 image_id 到每张图外参信息的映射。"""
    path = Path(path)
    images: dict[int, ColmapImageData] = {}

    with path.open("rb") as f:
        num_images = struct.unpack("<Q", f.read(8))[0]

        for _ in range(num_images):
            image_id = struct.unpack("<I", f.read(4))[0]
            qvec = np.asarray(struct.unpack("<4d", f.read(32)), dtype=np.float64)
            tvec = np.asarray(struct.unpack("<3d", f.read(24)), dtype=np.float64)
            camera_id = struct.unpack("<I", f.read(4))[0]

            name_bytes = []
            while True:
                char = f.read(1)
                if char == b"\x00":
                    break
                name_bytes.append(char)
            name = b"".join(name_bytes).decode("utf-8")

            num_points2d = struct.unpack("<Q", f.read(8))[0]
            f.read(24 * num_points2d)

            images[int(image_id)] = ColmapImageData(
                image_id=int(image_id),
                qvec=qvec,
                tvec=tvec,
                camera_id=int(camera_id),
                name=name,
            )

    return images


def load_gt_image(path: Path, camera, device):
    image = Image.open(path).convert("RGB")

    if image.size != (camera.width, camera.height):
        image = image.resize((camera.width, camera.height), Image.Resampling.BILINEAR)

    image = np.asarray(image, dtype=np.float32) / 255.0
    image = torch.from_numpy(image).to(device)

    return image.permute(2, 0, 1).contiguous()  # [H, W, 3] -> [3, H, W]

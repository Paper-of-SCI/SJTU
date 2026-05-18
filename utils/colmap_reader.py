"""Minimal COLMAP text/binary model reader."""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass
from typing import Dict

import numpy as np

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
class ColmapCamera:
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
        return float(self.params[0] if self.model.startswith("SIMPLE") or self.model in {"RADIAL", "FOV"} else self.params[1])

    @property
    def cx(self) -> float:
        return float(self.params[1] if self.model == "SIMPLE_PINHOLE" else self.params[2])

    @property
    def cy(self) -> float:
        return float(self.params[2] if self.model == "SIMPLE_PINHOLE" else self.params[3])


@dataclass(frozen=True)
class ColmapImage:
    image_id: int
    qvec: np.ndarray
    tvec: np.ndarray
    camera_id: int
    name: str
    xys: np.ndarray
    point3d_ids: np.ndarray


@dataclass(frozen=True)
class ColmapPoint3D:
    point3d_id: int
    xyz: np.ndarray
    rgb: np.ndarray
    error: float


def read_colmap_model(path: str) -> tuple[Dict[int, ColmapCamera], Dict[int, ColmapImage], Dict[int, ColmapPoint3D]]:
    """Read a COLMAP model directory containing bin or txt files."""
    if os.path.exists(os.path.join(path, "cameras.bin")):
        return read_cameras_binary(os.path.join(path, "cameras.bin")), read_images_binary(os.path.join(path, "images.bin")), read_points3d_binary(
            os.path.join(path, "points3D.bin")
        )
    return read_cameras_text(os.path.join(path, "cameras.txt")), read_images_text(os.path.join(path, "images.txt")), read_points3d_text(
        os.path.join(path, "points3D.txt")
    )


def read_cameras_binary(path: str) -> Dict[int, ColmapCamera]:
    cameras = {}
    with open(path, "rb") as handle:
        count = struct.unpack("<Q", handle.read(8))[0]
        for _ in range(count):
            camera_id, model_id, width, height = struct.unpack("<IiQQ", handle.read(24))
            model = CAMERA_MODEL_IDS[model_id]
            params = np.array(struct.unpack("<" + "d" * CAMERA_MODEL_NUM_PARAMS[model], handle.read(8 * CAMERA_MODEL_NUM_PARAMS[model])))
            cameras[camera_id] = ColmapCamera(camera_id, model, int(width), int(height), params)
    return cameras


def read_images_binary(path: str) -> Dict[int, ColmapImage]:
    images = {}
    with open(path, "rb") as handle:
        count = struct.unpack("<Q", handle.read(8))[0]
        for _ in range(count):
            image_id = struct.unpack("<I", handle.read(4))[0]
            qvec = np.array(struct.unpack("<4d", handle.read(32)), dtype=np.float64)
            tvec = np.array(struct.unpack("<3d", handle.read(24)), dtype=np.float64)
            camera_id = struct.unpack("<I", handle.read(4))[0]
            chars = []
            while True:
                char = handle.read(1)
                if char == b"\x00":
                    break
                chars.append(char)
            name = b"".join(chars).decode("utf-8")
            num_points = struct.unpack("<Q", handle.read(8))[0]
            xys = np.array(struct.unpack("<" + "d" * (2 * num_points), handle.read(16 * num_points)), dtype=np.float64).reshape(-1, 2)
            point_ids = np.array(struct.unpack("<" + "q" * num_points, handle.read(8 * num_points)), dtype=np.int64)
            images[image_id] = ColmapImage(image_id, qvec, tvec, camera_id, name, xys, point_ids)
    return images


def read_points3d_binary(path: str) -> Dict[int, ColmapPoint3D]:
    points = {}
    with open(path, "rb") as handle:
        count = struct.unpack("<Q", handle.read(8))[0]
        for _ in range(count):
            point_id = struct.unpack("<Q", handle.read(8))[0]
            xyz = np.array(struct.unpack("<3d", handle.read(24)), dtype=np.float64)
            rgb = np.array(struct.unpack("<3B", handle.read(3)), dtype=np.uint8)
            error = struct.unpack("<d", handle.read(8))[0]
            track_len = struct.unpack("<Q", handle.read(8))[0]
            handle.read(8 * track_len)
            points[point_id] = ColmapPoint3D(int(point_id), xyz, rgb, float(error))
    return points


def read_cameras_text(path: str) -> Dict[int, ColmapCamera]:
    cameras = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.split()
            camera_id = int(parts[0])
            cameras[camera_id] = ColmapCamera(camera_id, parts[1], int(parts[2]), int(parts[3]), np.array(parts[4:], dtype=np.float64))
    return cameras


def read_images_text(path: str) -> Dict[int, ColmapImage]:
    images = {}
    with open(path, "r", encoding="utf-8") as handle:
        lines = [line.strip() for line in handle if line.strip() and not line.startswith("#")]
    i = 0
    while i < len(lines):
        parts = lines[i].split()
        image_id = int(parts[0])
        point_line = lines[i + 1].split() if i + 1 < len(lines) else []
        xys = np.array([[float(point_line[j]), float(point_line[j + 1])] for j in range(0, len(point_line), 3)], dtype=np.float64)
        point_ids = np.array([int(point_line[j + 2]) for j in range(0, len(point_line), 3)], dtype=np.int64)
        images[image_id] = ColmapImage(
            image_id=image_id,
            qvec=np.array(parts[1:5], dtype=np.float64),
            tvec=np.array(parts[5:8], dtype=np.float64),
            camera_id=int(parts[8]),
            name=parts[9],
            xys=xys,
            point3d_ids=point_ids,
        )
        i += 2
    return images


def read_points3d_text(path: str) -> Dict[int, ColmapPoint3D]:
    points = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.split()
            point_id = int(parts[0])
            points[point_id] = ColmapPoint3D(
                point3d_id=point_id,
                xyz=np.array(parts[1:4], dtype=np.float64),
                rgb=np.array(parts[4:7], dtype=np.uint8),
                error=float(parts[7]),
            )
    return points

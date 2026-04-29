"""COLMAP 二进制/文本文件解析器。

支持读写 cameras.bin/txt、images.bin/txt、points3D.bin/txt。
无 PyTorch/JAX 依赖，仅用 NumPy 和标准库。
"""

import os
import struct
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np


# COLMAP 相机模型 ID → 名称
CAMERA_MODEL_IDS = {
    0:  "SIMPLE_PINHOLE",
    1:  "PINHOLE",
    2:  "SIMPLE_RADIAL",
    3:  "RADIAL",
    4:  "OPENCV",
    5:  "OPENCV_FISHEYE",
    6:  "FULL_OPENCV",
    7:  "FOV",
    8:  "SIMPLE_RADIAL_FISHEYE",
    9:  "RADIAL_FISHEYE",
    10: "THIN_PRISM_FISHEYE",
}

# 相机模型名称 → 参数数量
CAMERA_MODEL_NUM_PARAMS = {
    "SIMPLE_PINHOLE":       1,  # f
    "PINHOLE":              4,  # fx, fy, cx, cy
    "SIMPLE_RADIAL":        4,  # f, cx, cy, k1
    "RADIAL":               5,  # f, cx, cy, k1, k2
    "OPENCV":               8,  # fx, fy, cx, cy, k1, k2, p1, p2
    "OPENCV_FISHEYE":       8,  # fx, fy, cx, cy, k1, k2, k3, k4
    "FULL_OPENCV":         12,
    "FOV":                  5,
    "SIMPLE_RADIAL_FISHEYE": 4,
    "RADIAL_FISHEYE":       5,
    "THIN_PRISM_FISHEYE":  12,
}


@dataclass
class COLMAPCamera:
    camera_id: int
    model: str           # 例如 "PINHOLE"
    width: int
    height: int
    params: np.ndarray   # 依相机模型而定 [fx, fy, cx, cy, ...]

    @property
    def fx(self) -> float:
        if self.model == "SIMPLE_PINHOLE":
            return float(self.params[0])
        return float(self.params[0])

    @property
    def fy(self) -> float:
        if self.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL",
                          "RADIAL", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE"):
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


@dataclass
class COLMAPImage:
    image_id: int
    qvec: np.ndarray       # (4,) wxyz
    tvec: np.ndarray       # (3,)
    camera_id: int
    name: str
    xys: np.ndarray        # (M, 2) 二维关键点坐标
    point3D_ids: np.ndarray  # (M,) uint64，-1 表示无对应三维点


@dataclass
class COLMAPPoint3D:
    point3D_id: int
    xyz: np.ndarray        # (3,) float64
    rgb: np.ndarray        # (3,) uint8
    error: float
    image_ids: np.ndarray    # (K,) uint32
    point2D_idxs: np.ndarray  # (K,) uint32


# ---------------------------------------------------------------------------
# 读取函数（二进制）
# ---------------------------------------------------------------------------

def read_cameras_binary(path: str) -> Dict[int, COLMAPCamera]:
    cameras = {}
    with open(path, "rb") as f:
        num_cameras = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_cameras):
            # camera_id: uint32, model_id: int32, width: uint64, height: uint64
            camera_id, model_id, width, height = struct.unpack("<IiQQ", f.read(24))
            model = CAMERA_MODEL_IDS.get(model_id, f"UNKNOWN_{model_id}")
            num_params = CAMERA_MODEL_NUM_PARAMS.get(model, 0)
            params = np.array(struct.unpack(f"<{num_params}d", f.read(8 * num_params)))
            cameras[camera_id] = COLMAPCamera(camera_id, model, width, height, params)
    return cameras


def read_images_binary(path: str) -> Dict[int, COLMAPImage]:
    images = {}
    with open(path, "rb") as f:
        num_images = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_images):
            image_id = struct.unpack("<I", f.read(4))[0]
            qvec = np.array(struct.unpack("<4d", f.read(32)))   # wxyz
            tvec = np.array(struct.unpack("<3d", f.read(24)))
            camera_id = struct.unpack("<I", f.read(4))[0]
            # 读取文件名（以 \0 结尾）
            name_chars = []
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name_chars.append(c)
            name = b"".join(name_chars).decode("utf-8")
            # 读取 2D 点
            num_points2D = struct.unpack("<Q", f.read(8))[0]
            xys_flat = struct.unpack(f"<{2*num_points2D}d", f.read(16 * num_points2D))
            xys = np.array(xys_flat).reshape(-1, 2)
            point3D_ids = np.array(
                struct.unpack(f"<{num_points2D}q", f.read(8 * num_points2D)),
                dtype=np.int64,
            )
            images[image_id] = COLMAPImage(image_id, qvec, tvec, camera_id, name, xys, point3D_ids)
    return images


def read_points3D_binary(path: str) -> Dict[int, COLMAPPoint3D]:
    points = {}
    with open(path, "rb") as f:
        num_points = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_points):
            point3D_id = struct.unpack("<Q", f.read(8))[0]
            xyz = np.array(struct.unpack("<3d", f.read(24)))
            rgb = np.array(struct.unpack("<3B", f.read(3)), dtype=np.uint8)
            error = struct.unpack("<d", f.read(8))[0]
            track_length = struct.unpack("<Q", f.read(8))[0]
            track_data = struct.unpack(f"<{2*track_length}I", f.read(8 * track_length))
            image_ids   = np.array(track_data[0::2], dtype=np.uint32)
            point2D_idxs = np.array(track_data[1::2], dtype=np.uint32)
            points[point3D_id] = COLMAPPoint3D(
                point3D_id, xyz, rgb, error, image_ids, point2D_idxs
            )
    return points


# ---------------------------------------------------------------------------
# 读取函数（文本）
# ---------------------------------------------------------------------------

def read_cameras_text(path: str) -> Dict[int, COLMAPCamera]:
    cameras = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            camera_id = int(parts[0])
            model = parts[1]
            width, height = int(parts[2]), int(parts[3])
            params = np.array([float(p) for p in parts[4:]])
            cameras[camera_id] = COLMAPCamera(camera_id, model, width, height, params)
    return cameras


def read_images_text(path: str) -> Dict[int, COLMAPImage]:
    images = {}
    with open(path, "r") as f:
        lines = [l.strip() for l in f if not l.startswith("#") and l.strip()]
    i = 0
    while i < len(lines):
        parts = lines[i].split()
        image_id = int(parts[0])
        qvec = np.array([float(x) for x in parts[1:5]])
        tvec = np.array([float(x) for x in parts[5:8]])
        camera_id = int(parts[8])
        name = parts[9]
        # 下一行是 2D 点
        i += 1
        pt_parts = lines[i].split() if i < len(lines) else []
        if pt_parts:
            vals = [float(x) for x in pt_parts]
            xys_flat = vals[0::3]
            ys_flat  = vals[1::3]
            ids_flat = [int(v) for v in pt_parts[2::3]]
            xys = np.array(list(zip(xys_flat, ys_flat)))
            point3D_ids = np.array(ids_flat, dtype=np.int64)
        else:
            xys = np.zeros((0, 2))
            point3D_ids = np.zeros(0, dtype=np.int64)
        images[image_id] = COLMAPImage(image_id, qvec, tvec, camera_id, name, xys, point3D_ids)
        i += 1
    return images


def read_points3D_text(path: str) -> Dict[int, COLMAPPoint3D]:
    points = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            point3D_id = int(parts[0])
            xyz = np.array([float(x) for x in parts[1:4]])
            rgb = np.array([int(x) for x in parts[4:7]], dtype=np.uint8)
            error = float(parts[7])
            track = parts[8:]
            image_ids    = np.array([int(track[j]) for j in range(0, len(track), 2)], dtype=np.uint32)
            point2D_idxs = np.array([int(track[j]) for j in range(1, len(track), 2)], dtype=np.uint32)
            points[point3D_id] = COLMAPPoint3D(
                point3D_id, xyz, rgb, error, image_ids, point2D_idxs
            )
    return points


# ---------------------------------------------------------------------------
# 写入函数（二进制）
# ---------------------------------------------------------------------------

def write_cameras_binary(path: str, cameras: Dict[int, COLMAPCamera]) -> None:
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(cameras)))
        for cam in cameras.values():
            model_id = next(
                (k for k, v in CAMERA_MODEL_IDS.items() if v == cam.model), 1
            )
            f.write(struct.pack("<IiLL", cam.camera_id, model_id, cam.width, cam.height))
            f.write(struct.pack(f"<{len(cam.params)}d", *cam.params))


def write_images_binary(path: str, images: Dict[int, COLMAPImage]) -> None:
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(images)))
        for img in images.values():
            f.write(struct.pack("<I", img.image_id))
            f.write(struct.pack("<4d", *img.qvec))
            f.write(struct.pack("<3d", *img.tvec))
            f.write(struct.pack("<I", img.camera_id))
            f.write(img.name.encode("utf-8") + b"\x00")
            num_pts = len(img.xys)
            f.write(struct.pack("<Q", num_pts))
            for (x, y), pid in zip(img.xys, img.point3D_ids):
                f.write(struct.pack("<2d", x, y))
                f.write(struct.pack("<q", int(pid)))


def write_points3D_binary(path: str, points: Dict[int, COLMAPPoint3D]) -> None:
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(points)))
        for pt in points.values():
            f.write(struct.pack("<Q", pt.point3D_id))
            f.write(struct.pack("<3d", *pt.xyz))
            f.write(struct.pack("<3B", *pt.rgb))
            f.write(struct.pack("<d", pt.error))
            track_len = len(pt.image_ids)
            f.write(struct.pack("<Q", track_len))
            for iid, pidx in zip(pt.image_ids, pt.point2D_idxs):
                f.write(struct.pack("<2I", int(iid), int(pidx)))


# ---------------------------------------------------------------------------
# 便捷接口
# ---------------------------------------------------------------------------

def read_colmap_model(
    model_dir: str,
) -> Tuple[Dict[int, COLMAPCamera], Dict[int, COLMAPImage], Dict[int, COLMAPPoint3D]]:
    """自动检测二进制/文本格式，读取完整 COLMAP 重建。

    Args:
        model_dir: 通常为 sparse/0/ 目录。

    Returns:
        (cameras, images, points3D) 三个字典。
    """
    def _exists(name: str) -> bool:
        return os.path.exists(os.path.join(model_dir, name))

    if _exists("cameras.bin"):
        cameras  = read_cameras_binary(os.path.join(model_dir, "cameras.bin"))
        images   = read_images_binary(os.path.join(model_dir, "images.bin"))
        points3D = read_points3D_binary(os.path.join(model_dir, "points3D.bin"))
    elif _exists("cameras.txt"):
        cameras  = read_cameras_text(os.path.join(model_dir, "cameras.txt"))
        images   = read_images_text(os.path.join(model_dir, "images.txt"))
        points3D = read_points3D_text(os.path.join(model_dir, "points3D.txt"))
    else:
        raise FileNotFoundError(
            f"在 {model_dir} 中未找到 cameras.bin 或 cameras.txt"
        )
    return cameras, images, points3D


def write_colmap_model(
    model_dir: str,
    cameras: Dict[int, COLMAPCamera],
    images: Dict[int, COLMAPImage],
    points3D: Dict[int, COLMAPPoint3D],
    binary: bool = True,
) -> None:
    """写出 COLMAP 重建。"""
    os.makedirs(model_dir, exist_ok=True)
    if binary:
        write_cameras_binary(os.path.join(model_dir, "cameras.bin"), cameras)
        write_images_binary(os.path.join(model_dir, "images.bin"), images)
        write_points3D_binary(os.path.join(model_dir, "points3D.bin"), points3D)
    else:
        raise NotImplementedError("文本格式写出暂未实现")

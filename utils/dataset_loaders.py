"""多视角数据集加载器，统一返回 SceneData。

支持 COLMAP、LLFF（poses_bounds.npy）、Blender（transforms.json）格式。
无 PyTorch/JAX 依赖，仅用 NumPy 和标准库。
"""

import json
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from utils.camera_utils import (
    build_intrinsic_matrix,
    colmap_to_opengl_c2w,
    focal_to_fov,
)
from utils.colmap_reader import read_colmap_model
from utils.image_utils import load_image


@dataclass
class SceneData:
    """统一的场景数据结构，可被 PyTorch/JAX 等框架共用。"""

    image_paths: List[str]              # 绝对路径列表
    images: Optional[np.ndarray]        # (N, H, W, 3) float32 [0,1]，懒加载时为 None
    c2w_matrices: np.ndarray            # (N, 4, 4) float64，OpenGL 坐标系
    fx: np.ndarray                      # (N,) 或形状 () 的标量
    fy: np.ndarray
    cx: np.ndarray
    cy: np.ndarray
    width: int
    height: int
    distortion_params: Optional[Dict[str, float]]
    point_cloud_xyz: Optional[np.ndarray]   # (M, 3) float64 COLMAP 稀疏点云
    point_cloud_rgb: Optional[np.ndarray]   # (M, 3) uint8
    near: float = 0.01
    far: float = 100.0
    scene_scale: float = 1.0


# ---------------------------------------------------------------------------
# COLMAP 格式
# ---------------------------------------------------------------------------

def load_colmap_dataset(
    data_dir: str,
    split: str = "train",
    factor: int = 1,
    load_images: bool = True,
    llff_hold: int = 8,
) -> SceneData:
    """加载 COLMAP 重建数据集。

    Args:
        data_dir: 数据集根目录，sparse/0/ 在其下。
        split: "train" 或 "test"。
        factor: 下采样倍数（1 = 原分辨率）。
        load_images: False 则跳过图像加载（懒加载）。
        llff_hold: 每 N 张取一张作为测试集。

    Returns:
        SceneData。
    """
    model_dir = os.path.join(data_dir, "sparse", "0")
    cameras, images_data, points3D = read_colmap_model(model_dir)

    # 按文件名排序，保证顺序可复现
    img_ids = sorted(images_data.keys(), key=lambda k: images_data[k].name)

    # 切分 train/test
    if split == "test":
        img_ids = [img_ids[i] for i in range(0, len(img_ids), llff_hold)]
    else:
        img_ids = [img_ids[i] for i in range(len(img_ids)) if i % llff_hold != 0]

    # 读取相机内参（假设共享内参）
    cam0 = cameras[images_data[img_ids[0]].camera_id]
    fx_val, fy_val = cam0.fx / factor, cam0.fy / factor
    cx_val, cy_val = cam0.cx / factor, cam0.cy / factor
    w = cam0.width  // factor
    h = cam0.height // factor

    # 构建 c2w 矩阵
    c2w_list = []
    for iid in img_ids:
        img_rec = images_data[iid]
        c2w = colmap_to_opengl_c2w(img_rec.qvec, img_rec.tvec)
        c2w_list.append(c2w)
    c2w_matrices = np.stack(c2w_list, axis=0)  # (N, 4, 4)

    # 查找图像目录
    img_dir = _find_image_dir(data_dir, factor)

    # 图像路径
    image_paths = []
    for iid in img_ids:
        name = images_data[iid].name
        image_paths.append(os.path.join(img_dir, name))

    # 加载图像
    loaded_imgs = None
    if load_images:
        loaded_imgs = _load_images(image_paths, w, h)

    # 稀疏点云
    xyz = np.stack([points3D[pid].xyz for pid in points3D], axis=0) if points3D else None
    rgb = np.stack([points3D[pid].rgb for pid in points3D], axis=0) if points3D else None

    # 场景尺度（相机位置的 90th 百分位距离）
    scale = compute_scene_scale(c2w_matrices)

    # near/far 估计
    near, far = _estimate_near_far(c2w_matrices, xyz)

    return SceneData(
        image_paths=image_paths,
        images=loaded_imgs,
        c2w_matrices=c2w_matrices,
        fx=np.full(len(img_ids), fx_val),
        fy=np.full(len(img_ids), fy_val),
        cx=np.full(len(img_ids), cx_val),
        cy=np.full(len(img_ids), cy_val),
        width=w,
        height=h,
        distortion_params=None,
        point_cloud_xyz=xyz,
        point_cloud_rgb=rgb,
        near=near,
        far=far,
        scene_scale=scale,
    )


# ---------------------------------------------------------------------------
# LLFF 格式（poses_bounds.npy）
# ---------------------------------------------------------------------------

def load_llff_dataset(
    data_dir: str,
    split: str = "train",
    factor: int = 1,
    llff_hold: int = 8,
    load_images: bool = True,
) -> SceneData:
    """加载 LLFF 格式数据集（poses_bounds.npy + images/ 或 images_wb/）。

    若不存在 poses_bounds.npy，则回退到 load_colmap_dataset。

    poses_bounds.npy 格式：(N, 17)
        [0:9]   旋转矩阵（按行展开，3x3）
        [9:12]  平移向量
        [12:15] hwf（高、宽、焦距）
        [15:16] near
        [16:17] far

    LLFF 内部坐标系 [right, down, backward]，本函数转换为
    OpenGL [right, up, forward]（即乘以 diag([1,-1,-1])）。
    """
    poses_path = os.path.join(data_dir, "poses_bounds.npy")
    if not os.path.exists(poses_path):
        return load_colmap_dataset(data_dir, split, factor, load_images, llff_hold)

    poses_bounds = np.load(poses_path)      # (N, 17)
    poses_raw = poses_bounds[:, :15].reshape(-1, 3, 5)   # (N, 3, 5)
    bounds    = poses_bounds[:, 15:17]                    # (N, 2)

    hwf = poses_raw[:, :, 4]           # (N, 3): [h, w, focal]
    c2w_raw = poses_raw[:, :, :4]      # (N, 3, 4) LLFF c2w

    # LLFF [right, down, backward] → OpenGL [right, up, forward]
    c2w_raw = c2w_raw @ np.diag([1, -1, -1, 1])   # (N, 3, 4)

    # 补齐第四行
    bottom = np.array([[[0, 0, 0, 1]]] * len(c2w_raw))   # (N,1,4)
    c2w_matrices = np.concatenate([c2w_raw, bottom], axis=1)  # (N,4,4)

    # 内参（通常每帧不同焦距，但大多数数据集共享）
    h_vals = hwf[:, 0].astype(int)
    w_vals = hwf[:, 1].astype(int)
    focal_vals = hwf[:, 2]

    h_ref = h_vals[0]
    w_ref = w_vals[0]

    cx_arr = w_vals / 2.0
    cy_arr = h_vals / 2.0
    fx_arr = focal_vals
    fy_arr = focal_vals

    if factor > 1:
        h_ref = h_ref // factor
        w_ref = w_ref // factor
        fx_arr = fx_arr / factor
        fy_arr = fy_arr / factor
        cx_arr = cx_arr / factor
        cy_arr = cy_arr / factor

    # near/far
    near = float(bounds[:, 0].min() * 0.9)
    far  = float(bounds[:, 1].max() * 1.1)

    # 查找图像文件
    img_dir = _find_image_dir(data_dir, factor)
    all_img_names = sorted([
        f for f in os.listdir(img_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".exr"))
    ])

    N = len(c2w_matrices)
    if len(all_img_names) != N:
        # 数量不匹配时只取能对上的
        N = min(N, len(all_img_names))
        c2w_matrices = c2w_matrices[:N]
        fx_arr = fx_arr[:N]
        fy_arr = fy_arr[:N]
        cx_arr = cx_arr[:N]
        cy_arr = cy_arr[:N]
        all_img_names = all_img_names[:N]

    all_paths = [os.path.join(img_dir, n) for n in all_img_names]

    # 切分 train/test
    indices = list(range(N))
    if split == "test":
        indices = [i for i in indices if i % llff_hold == 0]
    else:
        indices = [i for i in indices if i % llff_hold != 0]

    c2w_matrices = c2w_matrices[indices]
    fx_arr = fx_arr[indices]
    fy_arr = fy_arr[indices]
    cx_arr = cx_arr[indices]
    cy_arr = cy_arr[indices]
    image_paths = [all_paths[i] for i in indices]

    loaded_imgs = None
    if load_images:
        loaded_imgs = _load_images(image_paths, w_ref, h_ref)

    scale = compute_scene_scale(c2w_matrices)

    return SceneData(
        image_paths=image_paths,
        images=loaded_imgs,
        c2w_matrices=c2w_matrices,
        fx=fx_arr,
        fy=fy_arr,
        cx=cx_arr,
        cy=cy_arr,
        width=w_ref,
        height=h_ref,
        distortion_params=None,
        point_cloud_xyz=None,   # LLFF 不含点云（可从 COLMAP 独立加载）
        point_cloud_rgb=None,
        near=near,
        far=far,
        scene_scale=scale,
    )


# ---------------------------------------------------------------------------
# Blender 格式（transforms.json）
# ---------------------------------------------------------------------------

def load_blender_dataset(
    data_dir: str,
    split: str = "train",
    white_background: bool = True,
    load_images: bool = True,
) -> SceneData:
    """加载 NeRF 合成数据集（Blender/instant-ngp transforms.json 格式）。

    Args:
        data_dir: 数据集目录，包含 transforms_{split}.json 或 transforms.json。
        split: "train"、"val"、"test"。
        white_background: True 则将 Alpha 混合到白色背景。
        load_images: False 则跳过图像加载。
    """
    # 尝试读取 split-specific JSON
    json_path = os.path.join(data_dir, f"transforms_{split}.json")
    if not os.path.exists(json_path):
        json_path = os.path.join(data_dir, "transforms.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"未找到 transforms json，路径：{data_dir}")

    with open(json_path, "r") as f:
        meta = json.load(f)

    # 内参
    w = int(meta.get("w", 800))
    h = int(meta.get("h", 800))
    cx = float(meta.get("cx", w / 2.0))
    cy = float(meta.get("cy", h / 2.0))
    if "fl_x" in meta:
        fx = float(meta["fl_x"])
    else:
        fx = 0.5 * w / math.tan(0.5 * float(meta["camera_angle_x"]))
    if "fl_y" in meta:
        fy = float(meta["fl_y"])
    elif "camera_angle_y" in meta:
        fy = 0.5 * h / math.tan(0.5 * float(meta["camera_angle_y"]))
    else:
        fy = fx

    frames = meta["frames"]
    image_paths = []
    c2w_list = []

    for frame in frames:
        fpath = frame["file_path"]
        if not fpath.endswith(".png"):
            fpath = fpath + ".png"
        full_path = os.path.join(data_dir, fpath)
        if not os.path.exists(full_path):
            continue
        image_paths.append(full_path)
        c2w = np.array(frame["transform_matrix"], dtype=np.float64)
        c2w_list.append(c2w)

    c2w_matrices = np.stack(c2w_list, axis=0)  # (N, 4, 4)
    N = len(image_paths)

    loaded_imgs = None
    if load_images:
        imgs = []
        for p in image_paths:
            img = load_image(p, as_float=True)  # HxWx3 或 HxWx4
            if img.shape[-1] == 4:
                alpha = img[:, :, 3:4]
                img = img[:, :, :3]
                if white_background:
                    img = img * alpha + (1 - alpha)
            imgs.append(img)
        loaded_imgs = np.stack(imgs, axis=0)  # (N, H, W, 3)

    scale = compute_scene_scale(c2w_matrices)

    return SceneData(
        image_paths=image_paths,
        images=loaded_imgs,
        c2w_matrices=c2w_matrices,
        fx=np.full(N, fx),
        fy=np.full(N, fy),
        cx=np.full(N, cx),
        cy=np.full(N, cy),
        width=w,
        height=h,
        distortion_params=None,
        point_cloud_xyz=None,
        point_cloud_rgb=None,
        near=0.1,
        far=10.0,
        scene_scale=scale,
    )


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def compute_scene_scale(
    c2w_matrices: np.ndarray,
    percentile: float = 90.0,
) -> float:
    """用相机位置的百分位距离估算场景归一化尺度。

    Args:
        c2w_matrices: (N, 4, 4)。
        percentile: 百分位数（默认 90）。

    Returns:
        场景尺度标量。
    """
    positions = c2w_matrices[:, :3, 3]   # (N, 3)
    distances = np.linalg.norm(positions, axis=1)
    scale = float(np.percentile(distances, percentile))
    return max(scale, 1e-6)


def _find_image_dir(data_dir: str, factor: int = 1) -> str:
    """查找图像目录，优先顺序：images_wb > images > images_{factor}。"""
    candidates = []
    if factor > 1:
        candidates.append(os.path.join(data_dir, f"images_{factor}"))
    candidates.append(os.path.join(data_dir, "images_wb"))
    candidates.append(os.path.join(data_dir, "images"))
    for c in candidates:
        if os.path.isdir(c):
            return c
    raise FileNotFoundError(
        f"在 {data_dir} 下未找到图像目录（尝试过 {candidates}）"
    )


def _load_images(
    paths: List[str],
    target_w: int,
    target_h: int,
) -> np.ndarray:
    """批量加载并缩放图像，返回 (N, H, W, 3) float32。"""
    imgs = []
    for p in paths:
        img = load_image(p, as_float=True, resize=(target_w, target_h) if target_w > 0 else None)
        imgs.append(img)
    return np.stack(imgs, axis=0)


def _estimate_near_far(
    c2w_matrices: np.ndarray,
    xyz: Optional[np.ndarray],
) -> Tuple[float, float]:
    """从点云和相机位置粗略估计 near/far。"""
    if xyz is None or len(xyz) == 0:
        return 0.01, 100.0
    positions = c2w_matrices[:, :3, 3]   # 相机位置
    # 计算每个相机到每个点云点的距离
    all_dists = []
    for pos in positions:
        d = np.linalg.norm(xyz - pos[None, :], axis=1)
        all_dists.append(d)
    dists = np.concatenate(all_dists)
    near = max(float(np.percentile(dists, 0.1)) * 0.5, 0.001)
    far  = float(np.percentile(dists, 99.9)) * 1.5
    return near, far

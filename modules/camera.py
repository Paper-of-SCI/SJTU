"""相机数据类：封装内参、外参与 GT 图像张量。

Camera 是冻结数据类（immutable），所有矩阵在 GPU 上以 PyTorch Tensor 存储。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor

from utils.camera_utils import (
    build_projection_matrix,
    focal_to_fov,
)

if TYPE_CHECKING:
    from utils.dataset_loaders import SceneData


@dataclass(frozen=True)
class Camera:
    """单个相机的完整描述，含内参、外参和可选 GT 图像。

    所有 Tensor 字段均位于同一设备上。
    c2w 采用 OpenGL 坐标系（右手系，x 右、y 上、-z 前）。

    Attributes:
        image_path: GT 图像路径（可选，仅用于标识）。
        width, height: 图像分辨率（像素）。
        fx, fy, cx, cy: 相机内参（像素单位）。
        c2w: (4, 4) camera-to-world 矩阵，OpenGL 坐标系。
        znear, zfar: 近/远裁剪面。
        image: (3, H, W) float32 GT 图像 [0,1]，CHW 格式；None 表示懒加载。
    """
    image_path: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    c2w: Tensor           # (4, 4)
    znear: float = 0.01
    zfar: float = 100.0
    image: Optional[Tensor] = None  # (3, H, W)

    # ---- 派生属性 ----

    @property
    def device(self) -> torch.device:
        return self.c2w.device

    @property
    def w2c(self) -> Tensor:
        """(4, 4) world-to-camera。"""
        return torch.linalg.inv(self.c2w)

    @property
    def projection_matrix(self) -> Tensor:
        """(4, 4) OpenGL 投影矩阵。"""
        P = build_projection_matrix(
            self.fx, self.fy, self.cx, self.cy,
            self.width, self.height,
            self.znear, self.zfar,
        )
        return torch.tensor(P, dtype=torch.float32, device=self.device)

    @property
    def full_projection_matrix(self) -> Tensor:
        """(4, 4) w2c @ projection，用于 diff-gaussian-rasterization。"""
        return self.w2c.float() @ self.projection_matrix

    @property
    def camera_center(self) -> Tensor:
        """(3,) 相机中心在世界坐标系中的位置。"""
        return self.c2w[:3, 3]

    @property
    def fov_x(self) -> float:
        """水平视场角（弧度）。"""
        return focal_to_fov(self.fx, self.width)

    @property
    def fov_y(self) -> float:
        """垂直视场角（弧度）。"""
        return focal_to_fov(self.fy, self.height)


# ---------------------------------------------------------------------------
# 构造函数
# ---------------------------------------------------------------------------

def camera_from_scene_data(
    scene: "SceneData",
    index: int,
    device: str = "cuda",
    load_image: bool = True,
) -> Camera:
    """从 SceneData 的第 index 帧构造 Camera。

    Args:
        scene: utils.dataset_loaders.SceneData。
        index: 帧索引。
        device: PyTorch 设备字符串。
        load_image: True 则将图像加载为 Tensor，否则 image=None。

    Returns:
        Camera 实例。
    """
    fx = float(scene.fx[index]) if scene.fx.ndim > 0 else float(scene.fx)
    fy = float(scene.fy[index]) if scene.fy.ndim > 0 else float(scene.fy)
    cx = float(scene.cx[index]) if scene.cx.ndim > 0 else float(scene.cx)
    cy = float(scene.cy[index]) if scene.cy.ndim > 0 else float(scene.cy)

    c2w = torch.tensor(scene.c2w_matrices[index], dtype=torch.float32, device=device)

    img_tensor: Optional[Tensor] = None
    if load_image:
        if scene.images is not None:
            arr = scene.images[index]   # (H, W, 3) float32
        else:
            from utils.image_utils import load_image as _load
            arr = _load(scene.image_paths[index], as_float=True,
                        resize=(scene.width, scene.height))
        # HWC → CHW
        img_tensor = torch.tensor(arr, dtype=torch.float32, device=device).permute(2, 0, 1)

    return Camera(
        image_path=scene.image_paths[index],
        width=scene.width,
        height=scene.height,
        fx=fx, fy=fy, cx=cx, cy=cy,
        c2w=c2w,
        znear=scene.near,
        zfar=scene.far,
        image=img_tensor,
    )


def cameras_from_scene_data(
    scene: "SceneData",
    device: str = "cuda",
    load_images: bool = True,
) -> List[Camera]:
    """从 SceneData 构造所有帧的 Camera 列表。"""
    return [
        camera_from_scene_data(scene, i, device=device, load_image=load_images)
        for i in range(len(scene.image_paths))
    ]

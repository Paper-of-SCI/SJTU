"""Camera data structures for composable gsplat rendering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True)
class Camera:
    """Single pinhole camera.

    ``c2w`` is camera-to-world.  ``viewmat`` is generated as world-to-camera,
    which is the matrix expected by gsplat.
    """

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    c2w: Tensor
    image: Optional[Tensor] = None
    image_path: str = ""
    near: float = 0.01
    far: float = 1.0e10

    @classmethod
    def from_scene_data(
        cls,
        scene,
        index: int,
        device: str | torch.device = "cuda",
        load_image: bool = True,
    ) -> "Camera":
        """Build a Camera from a ``utils.dataset_loaders.SceneData`` item."""
        fx = _scalar_at(scene.fx, index)
        fy = _scalar_at(scene.fy, index)
        cx = _scalar_at(scene.cx, index)
        cy = _scalar_at(scene.cy, index)
        c2w = torch.as_tensor(scene.c2w_matrices[index], dtype=torch.float32, device=device)

        image = None
        if load_image:
            if scene.images is not None:
                arr = scene.images[index]
            else:
                from utils.image_utils import load_image

                arr = load_image(scene.image_paths[index], as_float=True, resize=(scene.width, scene.height))
            image = torch.as_tensor(arr, dtype=torch.float32, device=device).permute(2, 0, 1).contiguous()

        return cls(
            width=int(scene.width),
            height=int(scene.height),
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            c2w=c2w,
            image=image,
            image_path=scene.image_paths[index],
            near=float(scene.near),
            far=float(scene.far),
        )

    @property
    def device(self) -> torch.device:
        return self.c2w.device

    @property
    def viewmat(self) -> Tensor:
        return torch.linalg.inv(self.c2w)

    @property
    def K(self) -> Tensor:
        return torch.tensor(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
            device=self.device,
        )

    @property
    def camera_center(self) -> Tensor:
        return self.c2w[:3, 3]


def _scalar_at(value, index: int) -> float:
    arr = np.asarray(value)
    if arr.ndim == 0:
        return float(arr)
    return float(arr[index])

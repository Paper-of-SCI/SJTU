"""Semantic importance mask loading for 3DGS training entrypoints."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import Tensor


class SemanticImportanceProvider:
    """Load per-image semantic importance masks through a narrow training boundary."""

    def __init__(self, root: Path, scene_name: str, device: torch.device) -> None:
        self.root = Path(root)
        self.scene_name = str(scene_name)
        self.scene_dir = self.root / self.scene_name
        self.device = device
        if not self.scene_dir.exists():
            raise FileNotFoundError(f"找不到 semantic_importance 场景目录: {self.scene_dir}")

    def load(self, image_path: str | Path, width: int, height: int) -> Tensor:
        """Return an ``HxW`` float tensor in ``[0, 1]`` for the current image."""
        target_width = int(width)
        target_height = int(height)
        if target_width <= 0 or target_height <= 0:
            raise ValueError(f"semantic_importance 目标尺寸非法: {target_width}x{target_height}")

        mask_path = self.scene_dir / f"{Path(image_path).stem}.png"
        if not mask_path.exists():
            raise FileNotFoundError(f"找不到 semantic_importance mask: {mask_path}")

        with Image.open(mask_path) as image:
            mask = image.convert("L")
            if mask.size != (target_width, target_height):
                mask = mask.resize((target_width, target_height), _bilinear_resample())
            array = np.asarray(mask, dtype=np.float32) / 255.0

        return torch.from_numpy(array).to(device=self.device, dtype=torch.float32).clamp_(0.0, 1.0)


def _bilinear_resample() -> int:
    if hasattr(Image, "Resampling"):
        return Image.Resampling.BILINEAR
    return Image.BILINEAR

"""Image IO and metrics utilities."""

from __future__ import annotations

import math
import os
from typing import Optional, Tuple

import numpy as np
from PIL import Image


def load_image(path: str, as_float: bool = True, resize: Optional[Tuple[int, int]] = None) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    if resize is not None and image.size != resize:
        image = image.resize(resize, Image.Resampling.LANCZOS)
    array = np.asarray(image)
    if as_float:
        return array.astype(np.float32) / 255.0
    return array


def save_image(path: str, image: np.ndarray, quality: int = 95) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    array = np.asarray(image)
    if array.dtype != np.uint8:
        array = np.clip(array * 255.0, 0.0, 255.0).astype(np.uint8)
    out = Image.fromarray(array)
    ext = os.path.splitext(path)[1].lower()
    if ext in {".jpg", ".jpeg"}:
        out.save(path, quality=quality)
    else:
        out.save(path)


def compute_psnr(pred: np.ndarray, target: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    if mask is not None:
        pred = pred[mask]
        target = target[mask]
    mse = float(np.mean((np.asarray(pred, dtype=np.float64) - np.asarray(target, dtype=np.float64)) ** 2))
    return -10.0 * math.log10(max(mse, 1e-12))


def compute_ssim(pred: np.ndarray, target: np.ndarray, sigma: float = 1.5) -> float:
    """Compute image SSIM.  Uses scipy when available."""
    try:
        from scipy.ndimage import gaussian_filter
    except ImportError as exc:
        raise ImportError("compute_ssim 需要 scipy；可安装 scipy 或改用 compute_psnr。") from exc

    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    c1 = 0.01**2
    c2 = 0.03**2
    values = []
    for channel in range(pred.shape[-1]):
        x = pred[..., channel]
        y = target[..., channel]
        mux = gaussian_filter(x, sigma)
        muy = gaussian_filter(y, sigma)
        mux2 = mux * mux
        muy2 = muy * muy
        muxy = mux * muy
        sigx2 = gaussian_filter(x * x, sigma) - mux2
        sigy2 = gaussian_filter(y * y, sigma) - muy2
        sigxy = gaussian_filter(x * y, sigma) - muxy
        values.append(float((((2 * muxy + c1) * (2 * sigxy + c2)) / ((mux2 + muy2 + c1) * (sigx2 + sigy2 + c2) + 1e-12)).mean()))
    return float(np.mean(values))

"""图像读写与质量度量工具。

无 PyTorch/JAX 依赖，仅用 NumPy、Pillow、scipy，可被任何框架复用。
"""

import os
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image


def load_image(
    path: str,
    as_float: bool = True,
    resize: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """加载图像。

    Args:
        path: 图像路径。
        as_float: True 则返回 float32 [0,1]，False 则返回 uint8 [0,255]。
        resize: (width, height) 目标分辨率，None 则不缩放。

    Returns:
        HxWx3 数组。
    """
    img = Image.open(path).convert("RGB")
    if resize is not None:
        img = img.resize(resize, Image.LANCZOS)
    arr = np.array(img)
    if as_float:
        return arr.astype(np.float32) / 255.0
    return arr


def save_image(
    path: str,
    image: np.ndarray,
    quality: int = 95,
) -> None:
    """保存图像，自动根据扩展名选择格式。

    Args:
        path: 输出路径。
        image: HxWx3 float32 [0,1] 或 uint8 [0,255]。
        quality: JPEG 质量（PNG 时忽略）。
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if image.dtype != np.uint8:
        image = np.clip(image * 255, 0, 255).astype(np.uint8)
    img = Image.fromarray(image)
    ext = os.path.splitext(path)[1].lower()
    if ext in (".jpg", ".jpeg"):
        img.save(path, quality=quality)
    else:
        img.save(path)


def compute_psnr(
    pred: np.ndarray,
    gt: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> float:
    """计算 PSNR（假设像素范围 [0,1]）。

    Args:
        pred: HxWx3 float32。
        gt:   HxWx3 float32。
        mask: 可选 HxW bool 掩码，True 表示计入。

    Returns:
        PSNR（dB）。
    """
    if mask is not None:
        pred = pred[mask]
        gt   = gt[mask]
    mse = float(np.mean((pred - gt) ** 2))
    mse = max(mse, 1e-10)
    return -10.0 * np.log10(mse)


def compute_ssim(
    pred: np.ndarray,
    gt: np.ndarray,
    window_size: int = 11,
    data_range: float = 1.0,
    sigma: float = 1.5,
) -> float:
    """计算 SSIM，与 3DGS 论文评估协议保持一致。

    使用可分离高斯核，不依赖 scikit-image 或 torchmetrics。

    Args:
        pred: HxWx3 float32 [0,1]。
        gt:   HxWx3 float32 [0,1]。

    Returns:
        SSIM 标量（越高越好）。
    """
    from scipy.ndimage import gaussian_filter

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    ssim_vals = []
    for c in range(pred.shape[2]):
        p = pred[:, :, c].astype(np.float64)
        g = gt[:, :, c].astype(np.float64)

        mu1 = gaussian_filter(p, sigma)
        mu2 = gaussian_filter(g, sigma)

        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = gaussian_filter(p * p, sigma) - mu1_sq
        sigma2_sq = gaussian_filter(g * g, sigma) - mu2_sq
        sigma12   = gaussian_filter(p * g, sigma) - mu1_mu2

        num   = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
        denom = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        ssim_map = num / (denom + 1e-12)
        ssim_vals.append(float(ssim_map.mean()))

    return float(np.mean(ssim_vals))


def compute_lpips(pred: np.ndarray, gt: np.ndarray) -> float:
    """计算 LPIPS（需安装 lpips 包），若包不存在则返回 -1.0。

    Args:
        pred: HxWx3 float32 [0,1]。
        gt:   HxWx3 float32 [0,1]。
    """
    try:
        import torch
        import lpips
        loss_fn = lpips.LPIPS(net="alex")
        to_tensor = lambda x: torch.tensor(x).permute(2, 0, 1).unsqueeze(0).float() * 2 - 1
        with torch.no_grad():
            return float(loss_fn(to_tensor(pred), to_tensor(gt)).item())
    except ImportError:
        return -1.0


def images_to_video(
    frames: List[np.ndarray],
    output_path: str,
    fps: int = 30,
) -> None:
    """将一组 HxWx3 帧写出为 mp4 视频文件（需安装 opencv-python）。

    Args:
        frames: HxWx3 float32 [0,1] 或 uint8 [0,255] 帧列表。
        output_path: 输出路径，例如 "output/video.mp4"。
        fps: 帧率。
    """
    import cv2

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    if not frames:
        return
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    for frame in frames:
        if frame.dtype != np.uint8:
            frame = np.clip(frame * 255, 0, 255).astype(np.uint8)
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def linear_to_srgb(img: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """线性颜色空间 → sRGB，输入输出均为 [0,1]。"""
    img = np.maximum(img, eps)
    return np.where(
        img <= 0.0031308,
        12.92 * img,
        1.055 * np.power(img, 1.0 / 2.4) - 0.055,
    )


def srgb_to_linear(img: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """sRGB → 线性颜色空间，输入输出均为 [0,1]。"""
    img = np.maximum(img, 0.0)
    return np.where(
        img <= 0.04045,
        img / 12.92,
        np.power((img + 0.055) / 1.055, 2.4),
    )

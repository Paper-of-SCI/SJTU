#!/usr/bin/env python3
"""Generate coarse underwater semantic-importance masks.

The script is an offline annotation adapter: it reads RGB scene images and
writes grayscale importance maps plus visual overlays. It does not participate
in 3DGS training directly.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageFilter


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


@dataclass(frozen=True)
class ImportanceConfig:
    edge_weight: float = 0.45
    contrast_weight: float = 0.30
    color_structure_weight: float = 0.15
    lower_scene_prior_weight: float = 0.10
    blur_sigma: float = 3.0
    min_importance: float = 0.05
    overlay_alpha: float = 0.42


@dataclass(frozen=True)
class SceneImages:
    scene_name: str
    image_dir: Path
    image_paths: tuple[Path, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate coarse semantic-importance masks for underwater scenes.")
    parser.add_argument("--data-root", default="src/datasets/SeathruNeRF_dataset", help="Root containing scene directories.")
    parser.add_argument("--scenes", nargs="*", default=None, help="Optional scene names under --data-root. Defaults to all discoverable scenes.")
    parser.add_argument("--out", default="semantic_importance", help="Output directory for grayscale importance masks.")
    parser.add_argument("--preview-out", default="semantic_importance_preview", help="Output directory for overlay previews.")
    parser.add_argument("--edge-weight", type=float, default=0.45, help="Weight for Sobel edge strength.")
    parser.add_argument("--contrast-weight", type=float, default=0.30, help="Weight for local contrast.")
    parser.add_argument("--color-structure-weight", type=float, default=0.15, help="Weight for non-water structural color prior.")
    parser.add_argument("--lower-scene-prior-weight", type=float, default=0.10, help="Weight for lower-image spatial prior.")
    parser.add_argument("--blur-sigma", type=float, default=3.0, help="Gaussian blur radius for smoothing masks.")
    parser.add_argument("--min-importance", type=float, default=0.05, help="Minimum normalized importance value.")
    parser.add_argument("--overlay-alpha", type=float, default=0.42, help="Preview heatmap opacity.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ImportanceConfig(
        edge_weight=args.edge_weight,
        contrast_weight=args.contrast_weight,
        color_structure_weight=args.color_structure_weight,
        lower_scene_prior_weight=args.lower_scene_prior_weight,
        blur_sigma=args.blur_sigma,
        min_importance=args.min_importance,
        overlay_alpha=args.overlay_alpha,
    )
    data_root = Path(args.data_root).expanduser()
    out_root = Path(args.out).expanduser()
    preview_root = Path(args.preview_out).expanduser()
    scenes = discover_scene_images(data_root, args.scenes)
    if not scenes:
        raise FileNotFoundError(f"No scene images found under {data_root}")

    total = 0
    for scene in scenes:
        scene_out = out_root / scene.scene_name
        scene_preview = preview_root / scene.scene_name
        scene_out.mkdir(parents=True, exist_ok=True)
        scene_preview.mkdir(parents=True, exist_ok=True)
        for image_path in scene.image_paths:
            image = load_rgb_float(image_path)
            importance = compute_importance(image, config)
            mask_path = scene_out / f"{image_path.stem}.png"
            preview_path = scene_preview / f"{image_path.stem}_overlay.png"
            save_grayscale(mask_path, importance)
            save_overlay(preview_path, image, importance, alpha=config.overlay_alpha)
            total += 1
        print(f"{scene.scene_name}: {len(scene.image_paths)} masks -> {scene_out}")
    print(f"Generated {total} masks in {out_root}")
    print(f"Generated previews in {preview_root}")


def discover_scene_images(data_root: Path, scene_names: list[str] | None = None) -> list[SceneImages]:
    scenes: list[SceneImages] = []
    if scene_names:
        scene_dirs = [data_root / name for name in scene_names]
    else:
        scene_dirs = sorted(path for path in data_root.iterdir() if path.is_dir())
    for scene_dir in scene_dirs:
        if not scene_dir.is_dir():
            raise FileNotFoundError(f"Scene directory not found: {scene_dir}")
        image_dir = first_existing_dir(scene_dir / "images_wb", scene_dir / "Images_wb", scene_dir / "images", scene_dir / "Images")
        if image_dir is None:
            continue
        image_paths = tuple(sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS))
        if image_paths:
            scenes.append(SceneImages(scene_name=scene_dir.name, image_dir=image_dir, image_paths=image_paths))
    return scenes


def first_existing_dir(*paths: Path) -> Path | None:
    for path in paths:
        if path.is_dir():
            return path
    return None


def load_rgb_float(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32)
    return array / 255.0


def compute_importance(image: np.ndarray, config: ImportanceConfig) -> np.ndarray:
    gray = rgb_to_luminance(image)
    edge = robust_normalize(sobel_magnitude(gray))
    contrast = robust_normalize(np.abs(gray - gaussian_blur_array(gray, config.blur_sigma)))
    color_structure = robust_normalize(non_water_structure_prior(image))
    lower_prior = lower_image_prior(gray.shape)

    score = (
        config.edge_weight * edge
        + config.contrast_weight * contrast
        + config.color_structure_weight * color_structure
        + config.lower_scene_prior_weight * lower_prior
    )
    smoothed = gaussian_blur_array(score, max(config.blur_sigma, 0.0))
    normalized = robust_normalize(smoothed)
    minimum = float(np.clip(config.min_importance, 0.0, 1.0))
    return np.clip(minimum + (1.0 - minimum) * normalized, 0.0, 1.0).astype(np.float32)


def rgb_to_luminance(image: np.ndarray) -> np.ndarray:
    return image[..., 0] * 0.2126 + image[..., 1] * 0.7152 + image[..., 2] * 0.0722


def sobel_magnitude(gray: np.ndarray) -> np.ndarray:
    padded = np.pad(gray, ((1, 1), (1, 1)), mode="edge")
    gx = (
        padded[:-2, :-2]
        + 2.0 * padded[1:-1, :-2]
        + padded[2:, :-2]
        - padded[:-2, 2:]
        - 2.0 * padded[1:-1, 2:]
        - padded[2:, 2:]
    )
    gy = (
        padded[:-2, :-2]
        + 2.0 * padded[:-2, 1:-1]
        + padded[:-2, 2:]
        - padded[2:, :-2]
        - 2.0 * padded[2:, 1:-1]
        - padded[2:, 2:]
    )
    return np.sqrt(gx * gx + gy * gy)


def gaussian_blur_array(values: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0.0:
        return values.astype(np.float32, copy=True)
    image = Image.fromarray(np.clip(values * 255.0, 0.0, 255.0).astype(np.uint8), mode="L")
    blurred = image.filter(ImageFilter.GaussianBlur(radius=float(sigma)))
    return np.asarray(blurred, dtype=np.float32) / 255.0


def non_water_structure_prior(image: np.ndarray) -> np.ndarray:
    red = image[..., 0]
    green = image[..., 1]
    blue = image[..., 2]
    max_channel = np.maximum.reduce([red, green, blue])
    min_channel = np.minimum.reduce([red, green, blue])
    saturation = (max_channel - min_channel) / np.maximum(max_channel, 1.0e-6)

    blue_dominance = np.clip(blue - 0.5 * (red + green), 0.0, 1.0)
    not_blue_water = 1.0 - robust_normalize(blue_dominance)
    warm_or_neutral = np.clip(0.5 * (red + green) - 0.35 * blue, 0.0, 1.0)
    return 0.55 * not_blue_water + 0.30 * robust_normalize(warm_or_neutral) + 0.15 * saturation


def lower_image_prior(shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    prior = np.clip((y - 0.25) / 0.75, 0.0, 1.0)
    return np.repeat(prior, width, axis=1)


def robust_normalize(values: np.ndarray, low_percentile: float = 2.0, high_percentile: float = 98.0) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros_like(values, dtype=np.float32)
    low = float(np.percentile(finite, low_percentile))
    high = float(np.percentile(finite, high_percentile))
    if high <= low + 1.0e-8:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - low) / (high - low), 0.0, 1.0).astype(np.float32)


def save_grayscale(path: Path, importance: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(np.clip(importance * 255.0, 0.0, 255.0).astype(np.uint8), mode="L")
    image.save(path)


def save_overlay(path: Path, image: np.ndarray, importance: np.ndarray, alpha: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    heatmap = importance_to_heatmap(importance)
    blend = np.clip((1.0 - alpha) * image + alpha * heatmap, 0.0, 1.0)
    Image.fromarray((blend * 255.0).astype(np.uint8), mode="RGB").save(path)


def importance_to_heatmap(importance: np.ndarray) -> np.ndarray:
    value = np.clip(importance[..., None], 0.0, 1.0)
    low = np.array([0.05, 0.10, 0.25], dtype=np.float32)
    mid = np.array([1.00, 0.75, 0.05], dtype=np.float32)
    high = np.array([1.00, 0.05, 0.02], dtype=np.float32)
    first = np.clip(value * 2.0, 0.0, 1.0)
    second = np.clip((value - 0.5) * 2.0, 0.0, 1.0)
    base = low * (1.0 - first) + mid * first
    return base * (1.0 - second) + high * second


if __name__ == "__main__":
    main()

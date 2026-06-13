"""Compute LPIPS for saved render/GT image pairs."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.ours_denstify.lpips_backend import LPIPS_BACKEND_CHOICES, build_ours_lpips_evaluator


RENDER_SUFFIX = "_render"
GT_SUFFIX = "_gt"


@dataclass(frozen=True)
class ImagePair:
    image_name: str
    render_path: Path
    gt_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute LPIPS for saved *_render.png and *_gt.png image pairs.")
    parser.add_argument("--pairs-dir", required=True, help="Directory containing matching *_render.png and *_gt.png files.")
    parser.add_argument("--lpips-net", default="vgg", choices=["alex", "vgg", "squeeze"], help="LPIPS backbone.")
    parser.add_argument(
        "--lpips-backend",
        default="official_3dgs",
        choices=list(LPIPS_BACKEND_CHOICES),
        help="LPIPS implementation. Use seasplat to match the SeaSplat/ours_denstify backend protocol.",
    )
    parser.add_argument("--out-csv", default="", help="Optional CSV path for per-image LPIPS values.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pairs_dir = resolve_existing_dir(args.pairs_dir)
    pairs = collect_image_pairs(pairs_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluator = build_ours_lpips_evaluator(device, args.lpips_net, args.lpips_backend)

    rows = evaluate_pairs(pairs, evaluator, device)
    mean_lpips = sum(row["lpips"] for row in rows) / len(rows)

    print(f"pairs_dir={pairs_dir}")
    print(f"device={device} input_source=saved_png lpips_net={args.lpips_net} lpips_backend={args.lpips_backend} images={len(rows)}")
    for row in rows:
        print(f"{row['image']}: LPIPS={row['lpips']:.10f}")
    print(f"mean_lpips={mean_lpips:.10f}")

    if args.out_csv:
        write_rows_csv(resolve_output_path(args.out_csv), rows)


def collect_image_pairs(pairs_dir: Path) -> list[ImagePair]:
    render_by_prefix = collect_suffix_paths(pairs_dir, RENDER_SUFFIX)
    gt_by_prefix = collect_suffix_paths(pairs_dir, GT_SUFFIX)
    if not render_by_prefix:
        raise FileNotFoundError(f"找不到 render 图片: {pairs_dir / '*_render.png'}")

    missing_gt = sorted(set(render_by_prefix) - set(gt_by_prefix))
    missing_render = sorted(set(gt_by_prefix) - set(render_by_prefix))
    if missing_gt or missing_render:
        details = []
        if missing_gt:
            details.append("缺少 GT: " + ", ".join(f"{prefix}{GT_SUFFIX}.png" for prefix in missing_gt[:8]))
        if missing_render:
            details.append("缺少 render: " + ", ".join(f"{prefix}{RENDER_SUFFIX}.png" for prefix in missing_render[:8]))
        raise FileNotFoundError("; ".join(details))

    return [
        ImagePair(
            image_name=display_image_name(prefix),
            render_path=render_by_prefix[prefix],
            gt_path=gt_by_prefix[prefix],
        )
        for prefix in sorted(render_by_prefix)
    ]


def collect_suffix_paths(pairs_dir: Path, suffix: str) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for path in pairs_dir.iterdir():
        if not path.is_file() or path.suffix.lower() != ".png":
            continue
        if not path.stem.endswith(suffix):
            continue
        prefix = path.stem[: -len(suffix)]
        if prefix in paths:
            raise ValueError(f"重复图片前缀: {prefix}")
        paths[prefix] = path
    return paths


def display_image_name(prefix: str) -> str:
    match = re.match(r"^\d+_(.+)$", prefix)
    stem = match.group(1) if match else prefix
    return f"{stem}.png"


@torch.no_grad()
def evaluate_pairs(pairs: list[ImagePair], evaluator, device: torch.device) -> list[dict]:
    rows = []
    for pair in pairs:
        render = load_rgb_tensor(pair.render_path, device)
        gt = load_rgb_tensor(pair.gt_path, device)
        if render.shape != gt.shape:
            raise ValueError(
                f"render/GT 尺寸不一致: {pair.render_path.name} shape={tuple(render.shape)}; "
                f"{pair.gt_path.name} shape={tuple(gt.shape)}"
            )
        rows.append(
            {
                "image": pair.image_name,
                "render": str(pair.render_path),
                "gt": str(pair.gt_path),
                "lpips": float(evaluator(render, gt)),
            }
        )
    return rows


def load_rgb_tensor(path: Path, device: torch.device) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous().to(device=device)


def write_rows_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image", "render", "gt", "lpips"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"out_csv={path}")


def resolve_existing_dir(path: str) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    if not candidate.is_dir():
        raise NotADirectoryError(f"找不到图片目录: {path}")
    return candidate


def resolve_output_path(path: str) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return ROOT / candidate


if __name__ == "__main__":
    main()

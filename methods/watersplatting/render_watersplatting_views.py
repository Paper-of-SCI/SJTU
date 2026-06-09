"""Render and evaluate a trained WaterSplatting checkpoint on held-out views."""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.watersplatting.watersplatting_utils import (
    check_watersplatting_python,
    python_default,
    resolve_input_path,
    resolve_output_path,
    run_watersplatting_command,
)
from modules import build_lpips_evaluator, compute_image_metrics, resize_to_gt_if_needed
from utils.image_utils import save_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render WaterSplatting test views and compute metrics.")
    parser.add_argument("--data", default="src/datasets/SeathruNeRF_dataset/Curasao", help="Original scene path; recorded for consistency.")
    parser.add_argument("--prepared-data", default="outputs/watersplatting_Curasao_720w/prepared_data")
    parser.add_argument("--load-config", default="outputs/watersplatting_Curasao_720w/config.yml", help="WaterSplatting nerfstudio config.yml.")
    parser.add_argument("--out", default="outputs/watersplatting_Curasao_720w/test_renders")
    parser.add_argument("--holdout", type=int, default=8)
    parser.add_argument("--holdout-offset", type=int, default=0)
    parser.add_argument("--max-images", type=int, default=0, help="Limit metric/image export count after nerfstudio rendering; 0 means all.")
    parser.add_argument("--lpips", action="store_true")
    parser.add_argument("--lpips-net", default="vgg", choices=["alex", "vgg", "squeeze"])
    parser.add_argument("--lpips-backend", default="official_3dgs", choices=["lpips", "official_3dgs"])
    parser.add_argument("--python", default=python_default(), help="Python executable for the WaterSplatting environment.")
    parser.add_argument("--skip-dependency-check", action="store_true")
    parser.add_argument("--no-save-images", action="store_true")
    parser.add_argument("--skip-render", action="store_true", help="Reuse existing _nerfstudio_render outputs and only recompute metrics.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    resolve_input_path(args.data)
    prepared_data = resolve_input_path(args.prepared_data)
    load_config = resolve_input_path(args.load_config)
    out_dir = resolve_output_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ns_render_dir = out_dir / "_nerfstudio_render"

    if not args.skip_dependency_check and not args.skip_render:
        check_watersplatting_python(args.python)

    if not args.skip_render:
        if ns_render_dir.exists():
            shutil.rmtree(ns_render_dir)
        command = [
            "-m",
            "nerfstudio.scripts.render",
            "dataset",
            "--load-config",
            str(load_config),
            "--output-path",
            str(ns_render_dir),
            "--data",
            str(prepared_data),
            "--split",
            "test",
            "--rendered-output-names",
            "rgb",
            "gt-rgb",
            "--image-format",
            "png",
        ]
        run_watersplatting_command(args.python, command)

    pairs = collect_render_pairs(ns_render_dir)
    if args.max_images > 0:
        pairs = pairs[: args.max_images]
    if not pairs:
        raise RuntimeError(f"未找到 WaterSplatting 渲染输出: {ns_render_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lpips_evaluator = build_lpips_evaluator(device, args.lpips_net, args.lpips_backend) if args.lpips else None
    metric_rows = []
    print(
        f"WaterSplatting renders={len(pairs)} holdout={args.holdout} holdout_offset={args.holdout_offset} "
        f"lpips_backend={args.lpips_backend if args.lpips else ''}"
    )
    for index, (image_name, render_path, gt_path) in enumerate(pairs):
        image = load_rgb_tensor(render_path, device)
        gt = load_rgb_tensor(gt_path, device)
        image = resize_to_gt_if_needed(image, gt)
        metrics = compute_image_metrics(image, gt, lpips_evaluator)
        metric_rows.append(
            {
                "image": image_name,
                "psnr": metrics["psnr"],
                "ssim": metrics["ssim"],
                "l1": metrics["l1"],
                "lpips": metrics["lpips"],
            }
        )
        if not args.no_save_images:
            stem = Path(image_name).stem
            save_image(str(out_dir / f"{index:03d}_{stem}_render.png"), image.permute(1, 2, 0).cpu().numpy())
            save_image(str(out_dir / f"{index:03d}_{stem}_gt.png"), gt.permute(1, 2, 0).cpu().numpy())
        lpips_text = f" LPIPS={metrics['lpips']:.4f}" if metrics["lpips"] is not None else ""
        print(
            f"[{index + 1}/{len(pairs)}] {image_name} "
            f"PSNR={metrics['psnr']:.2f} SSIM={metrics['ssim']:.4f} L1={metrics['l1']:.5f}{lpips_text}"
        )

    save_metrics_csv(out_dir / "metrics.csv", metric_rows, include_lpips=lpips_evaluator is not None)
    print(f"指标 CSV：{out_dir / 'metrics.csv'}")


def collect_render_pairs(render_root: Path) -> list[tuple[str, Path, Path]]:
    rgb_dir = render_root / "test" / "rgb"
    gt_dir = render_root / "test" / "gt-rgb"
    if not rgb_dir.is_dir() or not gt_dir.is_dir():
        return []
    render_paths = sorted(path for path in rgb_dir.rglob("*") if path.suffix.lower() in {".png", ".jpg", ".jpeg"})
    pairs = []
    for render_path in render_paths:
        rel = render_path.relative_to(rgb_dir)
        gt_path = gt_dir / rel
        if gt_path.exists():
            pairs.append((rel.as_posix(), render_path, gt_path))
    return pairs


def load_rgb_tensor(path: Path, device: torch.device) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).to(device)


def save_metrics_csv(path: Path, rows: list[dict], include_lpips: bool) -> None:
    fieldnames = ["image", "psnr", "ssim", "l1"]
    if include_lpips:
        fieldnames.append("lpips")
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row[name] for name in fieldnames})


if __name__ == "__main__":
    main()


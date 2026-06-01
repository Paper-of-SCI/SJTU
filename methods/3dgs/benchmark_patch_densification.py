"""Benchmark standard 3DGS densification against patch-guided variants."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DENSIFY_GRAD_THRESHOLD = 2.0e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 3DGS patch densification comparison experiments.")
    parser.add_argument("--scenes", nargs="+", default=["Curasao"], help="Scene names under --data-root or explicit scene paths.")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["standard_3dgs", "patch_guided", "patch_reallocate"],
        choices=["standard", "standard_3dgs", "patch_guided", "patch_guided_semantic", "patch_reallocate"],
        help="Densification variants to compare.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0], help="Random seeds.")
    parser.add_argument("--iterations", type=int, default=7000, help="Training iterations per run.")
    parser.add_argument("--factor", type=int, default=4, help="Image downscale factor.")
    parser.add_argument("--target-height", type=int, default=0, help="Resize images to this height while preserving aspect ratio; 0 uses --factor.")
    parser.add_argument("--target-width", type=int, default=0, help="Resize images to this width while preserving aspect ratio; 0 uses --target-height or --factor.")
    parser.add_argument("--holdout", type=int, default=8, help="Holdout interval.")
    parser.add_argument("--holdout-offset", type=int, default=0, help="Offset used when selecting every Nth held-out image.")
    parser.add_argument("--lpips", action="store_true", help="Compute LPIPS during test rendering.")
    parser.add_argument("--lpips-net", default="vgg", choices=["alex", "vgg", "squeeze"], help="LPIPS backbone used when --lpips is enabled.")
    parser.add_argument("--lpips-backend", default="lpips", choices=["lpips", "official_3dgs"], help="LPIPS implementation forwarded to rendering.")
    parser.add_argument("--out", default="outputs/3dgs_patch_curasao", help="Benchmark output directory.")
    parser.add_argument("--data-root", default="src/datasets/SeathruNeRF_dataset", help="Root directory for named scenes.")
    parser.add_argument("--log-every", type=int, default=100, help="Training log interval.")
    parser.add_argument(
        "--densify-grad-threshold",
        type=float,
        default=DEFAULT_DENSIFY_GRAD_THRESHOLD,
        help="Shared densification gradient threshold; default is calibrated for this gsplat training path.",
    )
    parser.add_argument("--densify-start-step", type=int, default=500, help="First iteration that may run densification.")
    parser.add_argument("--densify-stop-step", type=int, default=0, help="Densification phase boundary forwarded to training; 0 uses training default.")
    parser.add_argument("--densify-interval", type=int, default=100, help="Densification interval forwarded to training.")
    parser.add_argument("--opacity-reset-interval", type=int, default=3000, help="Opacity reset interval during the densification phase.")
    parser.add_argument("--patch-size", type=int, default=16, help="Patch size for patch-guided variants.")
    parser.add_argument("--patch-edge-weight", type=float, default=0.75, help="Edge multiplier for patch detail scoring.")
    parser.add_argument("--patch-detail-lambda", type=float, default=2.0, help="Patch detail multiplier on gradients.")
    parser.add_argument("--semantic-importance-root", default="", help="Root directory for semantic importance masks.")
    parser.add_argument("--semantic-base", type=float, default=0.2, help="Minimum semantic multiplier for patch_guided_semantic.")
    parser.add_argument("--reallocate-fraction", type=float, default=0.10, help="Low-detail Gaussian fraction reallocated.")
    parser.add_argument("--clone-jitter-scale", type=float, default=0.05, help="Scale-relative jitter for patch-guided clones.")
    parser.add_argument("--skip-existing", action="store_true", help="Reuse runs whose final.ply and metrics.csv already exist.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if "patch_guided_semantic" in args.variants and not args.semantic_importance_root:
        raise ValueError("patch_guided_semantic 需要传入 --semantic-importance-root")
    out_dir = resolve_output_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for scene_arg in args.scenes:
        data_dir = resolve_scene_path(args.data_root, scene_arg)
        scene_name = data_dir.name
        for seed in args.seeds:
            for variant in args.variants:
                normalized_variant = "standard_3dgs" if variant == "standard" else variant
                row = run_variant(args, data_dir, scene_name, normalized_variant, int(seed), out_dir)
                rows.append(row)
                write_summary(out_dir, rows)

    print(f"汇总 CSV：{out_dir / 'summary.csv'}")
    print(f"汇总 JSON：{out_dir / 'summary.json'}")


def run_variant(args: argparse.Namespace, data_dir: Path, scene_name: str, variant: str, seed: int, out_dir: Path) -> dict:
    run_dir = out_dir / scene_name / variant / f"seed_{seed:03d}"
    train_dir = run_dir / "train"
    render_dir = run_dir / "test_renders"
    final_ply = train_dir / "final.ply"
    metrics_csv = render_dir / "metrics.csv"

    if args.skip_existing and final_ply.exists() and metrics_csv.exists():
        print(f"跳过已存在结果：scene={scene_name} variant={variant} seed={seed}")
        return build_summary_row(args, data_dir, scene_name, variant, seed, run_dir, final_ply, metrics_csv, total_seconds=0.0)

    print(f"开始训练：scene={scene_name} variant={variant} seed={seed} iterations={args.iterations}")
    started_at = time.perf_counter()
    train_cmd = [
        sys.executable,
        str(ROOT / "methods/3dgs/train_3dgs_scene.py"),
        "--data",
        str(data_dir),
        "--out",
        str(train_dir),
        "--iterations",
        str(args.iterations),
        "--factor",
        str(args.factor),
        "--target-height",
        str(args.target_height),
        "--target-width",
        str(args.target_width),
        "--holdout",
        str(args.holdout),
        "--holdout-offset",
        str(args.holdout_offset),
        "--seed",
        str(seed),
        "--densification-mode",
        variant,
        "--densify-grad-threshold",
        str(args.densify_grad_threshold),
        "--densify-start-step",
        str(args.densify_start_step),
        "--densify-stop-step",
        str(args.densify_stop_step),
        "--densify-interval",
        str(args.densify_interval),
        "--opacity-reset-interval",
        str(args.opacity_reset_interval),
        "--patch-size",
        str(args.patch_size),
        "--patch-edge-weight",
        str(args.patch_edge_weight),
        "--patch-detail-lambda",
        str(args.patch_detail_lambda),
        "--semantic-importance-root",
        str(args.semantic_importance_root),
        "--semantic-base",
        str(args.semantic_base),
        "--reallocate-fraction",
        str(args.reallocate_fraction),
        "--clone-jitter-scale",
        str(args.clone_jitter_scale),
        "--save-every",
        str(args.iterations),
        "--eval-every",
        "0",
        "--log-every",
        str(args.log_every),
    ]
    run_command(train_cmd)
    if not final_ply.exists():
        raise FileNotFoundError(f"训练未生成 final.ply: {final_ply}")

    print(f"开始测试渲染：scene={scene_name} variant={variant} seed={seed}")
    render_cmd = [
        sys.executable,
        str(ROOT / "methods/3dgs/render_3dgs_views.py"),
        "--data",
        str(data_dir),
        "--checkpoint",
        str(final_ply),
        "--out",
        str(render_dir),
        "--split",
        "test",
        "--factor",
        str(args.factor),
        "--target-height",
        str(args.target_height),
        "--target-width",
        str(args.target_width),
        "--holdout",
        str(args.holdout),
        "--holdout-offset",
        str(args.holdout_offset),
    ]
    if args.lpips:
        render_cmd.extend(["--lpips", "--lpips-net", str(args.lpips_net), "--lpips-backend", str(args.lpips_backend)])
    run_command(render_cmd)
    if not metrics_csv.exists():
        raise FileNotFoundError(f"测试渲染未生成 metrics.csv: {metrics_csv}")

    total_seconds = time.perf_counter() - started_at
    row = build_summary_row(args, data_dir, scene_name, variant, seed, run_dir, final_ply, metrics_csv, total_seconds)
    print(
        f"完成：scene={scene_name} variant={variant} seed={seed} "
        f"PSNR={row['psnr']:.3f} SSIM={row['ssim']:.4f} L1={row['l1']:.5f} "
        f"G={row['gaussian_count']} total={format_duration(total_seconds)}"
    )
    return row


def build_summary_row(
    args: argparse.Namespace,
    data_dir: Path,
    scene_name: str,
    variant: str,
    seed: int,
    run_dir: Path,
    final_ply: Path,
    metrics_csv: Path,
    total_seconds: float,
) -> dict:
    metrics = read_metric_means(metrics_csv)
    training = read_training_summary(run_dir / "train" / "training_summary.json")
    return {
        "scene": scene_name,
        "variant": variant,
        "seed": seed,
        "iterations": int(args.iterations),
        "factor": int(args.factor),
        "target_height": int(args.target_height),
        "target_width": int(args.target_width),
        "holdout": int(args.holdout),
        "holdout_offset": int(args.holdout_offset),
        "lpips_net": str(args.lpips_net) if args.lpips else "",
        "lpips_backend": str(args.lpips_backend) if args.lpips else "",
        "patch_size": int(args.patch_size),
        "patch_edge_weight": float(args.patch_edge_weight),
        "patch_detail_lambda": float(args.patch_detail_lambda),
        "semantic_importance_root": training.get("semantic_importance_root"),
        "semantic_base": training.get("semantic_base"),
        "uses_semantic_importance": training.get("uses_semantic_importance"),
        "reallocate_fraction": float(args.reallocate_fraction),
        "densify_grad_threshold": float(args.densify_grad_threshold),
        "densify_start_step": training.get("densify_start_step"),
        "densify_stop_step": training.get("densify_stop_step"),
        "effective_densify_stop_step": training.get("effective_densify_stop_step"),
        "densify_interval": training.get("densify_interval"),
        "opacity_reset_interval": training.get("opacity_reset_interval"),
        "psnr": metrics.get("psnr"),
        "ssim": metrics.get("ssim"),
        "l1": metrics.get("l1"),
        "lpips": metrics.get("lpips"),
        "gaussian_count": training.get("final_gaussians"),
        "train_seconds": training.get("elapsed_seconds"),
        "total_seconds": float(total_seconds),
        "data": str(data_dir),
        "run_dir": str(run_dir),
        "final_ply": str(final_ply),
        "metrics_csv": str(metrics_csv),
    }


def run_command(command: list[str]) -> None:
    subprocess.run(command, cwd=str(ROOT), check=True)


def read_metric_means(path: Path) -> dict:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"metrics.csv 为空: {path}")
    means = {}
    for name in ["psnr", "ssim", "l1", "lpips"]:
        values = [float(row[name]) for row in rows if name in row and row[name] not in ("", "None")]
        means[name] = float(sum(values) / len(values)) if values else None
    return means


def read_training_summary(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_summary(out_dir: Path, rows: list[dict]) -> None:
    fields = [
        "scene",
        "variant",
        "seed",
        "iterations",
        "factor",
        "target_height",
        "target_width",
        "holdout",
        "holdout_offset",
        "lpips_net",
        "lpips_backend",
        "patch_size",
        "patch_edge_weight",
        "patch_detail_lambda",
        "semantic_importance_root",
        "semantic_base",
        "uses_semantic_importance",
        "reallocate_fraction",
        "densify_grad_threshold",
        "densify_start_step",
        "densify_stop_step",
        "effective_densify_stop_step",
        "densify_interval",
        "opacity_reset_interval",
        "psnr",
        "ssim",
        "l1",
        "lpips",
        "gaussian_count",
        "train_seconds",
        "total_seconds",
        "data",
        "run_dir",
        "final_ply",
        "metrics_csv",
    ]
    with open(out_dir / "summary.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=2)


def resolve_scene_path(data_root: str, scene: str) -> Path:
    candidate = Path(scene).expanduser()
    if candidate.exists():
        return candidate.resolve()
    root = resolve_output_path(data_root)
    scene_path = root / scene
    if scene_path.exists():
        return scene_path.resolve()
    raise FileNotFoundError(f"找不到场景: {scene}；也尝试过 {scene_path}")


def resolve_output_path(path: str) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return ROOT / candidate


def format_duration(seconds: float) -> str:
    seconds = max(float(seconds), 0.0)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes > 0:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


if __name__ == "__main__":
    main()

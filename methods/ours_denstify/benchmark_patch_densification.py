"""Benchmark standard 3DGS densification against patch-guided variants."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

METHOD_DIR = Path(__file__).resolve().parent
ROOT = METHOD_DIR.parents[1]
DEFAULT_DENSIFY_GRAD_THRESHOLD = 2.0e-6

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules import (
    DENSIFICATION_MODE_CHOICES,
    flatten_best_metric_fields,
    normalize_densification_mode,
    uses_semantic_importance,
    validate_semantic_importance_root,
)
from methods.ours_denstify.lpips_backend import LPIPS_BACKEND_CHOICES, TRAIN_LPIPS_BACKEND_CHOICES


@dataclass(frozen=True)
class RunSpec:
    variant: str
    semantic_base: float
    semantic_base_label: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 3DGS patch densification comparison experiments.")
    parser.add_argument("--scenes", nargs="+", default=["Curasao"], help="Scene names under --data-root or explicit scene paths.")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["standard_3dgs", "patch_guided", "patch_reallocate"],
        choices=list(DENSIFICATION_MODE_CHOICES),
        help="Densification variants to compare.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0], help="Random seeds.")
    parser.add_argument("--iterations", type=int, default=7000, help="Training iterations per run.")
    parser.add_argument("--factor", type=int, default=-1, help="Image downscale factor; -1 keeps images at original size unless width exceeds 1600.")
    parser.add_argument("--target-height", type=int, default=0, help="Resize images to this height while preserving aspect ratio; 0 uses --factor.")
    parser.add_argument("--target-width", type=int, default=0, help="Resize images to this width while preserving aspect ratio; 0 uses --target-height or --factor.")
    parser.add_argument("--holdout", type=int, default=8, help="Holdout interval.")
    parser.add_argument("--holdout-offset", type=int, default=0, help="Offset used when selecting every Nth held-out image.")
    parser.add_argument("--lpips", action="store_true", help="Compute LPIPS during test rendering.")
    parser.add_argument("--lpips-net", default="vgg", choices=["alex", "vgg", "squeeze"], help="LPIPS backbone used when --lpips is enabled.")
    parser.add_argument("--lpips-backend", default="lpips", choices=list(LPIPS_BACKEND_CHOICES), help="LPIPS implementation forwarded to rendering.")
    parser.add_argument("--train-lpips-weight", type=float, default=0.0, help="Differentiable LPIPS loss weight forwarded to training; 0 disables it.")
    parser.add_argument("--train-lpips-start-step", type=int, default=7000, help="First training step that may include LPIPS loss.")
    parser.add_argument("--train-lpips-max-size", type=int, default=512, help="Longest side used for differentiable training LPIPS loss; 0 keeps full resolution.")
    parser.add_argument("--train-lpips-net", default="vgg", choices=["alex", "vgg", "squeeze"], help="LPIPS backbone used by training LPIPS loss.")
    parser.add_argument(
        "--train-lpips-backend",
        default="seasplat",
        choices=list(TRAIN_LPIPS_BACKEND_CHOICES),
        help="LPIPS implementation used by training loss.",
    )
    parser.add_argument("--out", default="outputs/3dgs_patch_curasao", help="Benchmark output directory.")
    parser.add_argument("--data-root", default="src/datasets/SeathruNeRF_dataset", help="Root directory for named scenes.")
    parser.add_argument("--log-every", type=int, default=100, help="Training log interval.")
    parser.add_argument("--train-eval-every", type=int, default=1000, help="Held-out training evaluation interval; 0 disables best checkpoint tracking.")
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
    parser.add_argument("--patch-perceptual-weight", type=float, default=0.0, help="VGG residual weight added to patch detail scoring; 0 disables it.")
    parser.add_argument("--patch-perceptual-max-size", type=int, default=768, help="Longest side used for VGG residual patch scoring.")
    parser.add_argument("--semantic-importance-root", default="", help="Root directory for semantic importance masks.")
    parser.add_argument("--semantic-base", type=float, default=0.2, help="Minimum semantic multiplier for semantic patch-guided variants.")
    parser.add_argument(
        "--semantic-base-values",
        nargs="+",
        type=float,
        default=[],
        help="Semantic-base sweep values applied only to semantic variants.",
    )
    parser.add_argument("--reallocate-fraction", type=float, default=0.10, help="Low-detail Gaussian fraction reallocated.")
    parser.add_argument("--clone-jitter-scale", type=float, default=0.05, help="Scale-relative jitter for patch-guided clones.")
    parser.add_argument("--skip-existing", action="store_true", help="Reuse runs whose final.ply and metrics.csv already exist.")
    parser.add_argument("--csv-only", action="store_true", help="Minimize artifacts: no train best/checkpoint previews, no render PNGs, delete final.ply after metrics.")
    parser.add_argument("--compact-summary", action="store_true", help="Write summary.csv/json with only core metrics and also write summary_agg.csv.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_semantic_importance_root(args.variants, args.semantic_importance_root)
    out_dir = resolve_output_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    run_specs = build_run_specs(args.variants, args.semantic_base, args.semantic_base_values)

    for scene_arg in args.scenes:
        data_dir = resolve_scene_path(args.data_root, scene_arg)
        scene_name = data_dir.name
        for seed in args.seeds:
            for run_spec in run_specs:
                row = run_variant(args, data_dir, scene_name, run_spec, int(seed), out_dir)
                rows.append(row)
                write_summary(out_dir, rows, compact=args.compact_summary)

    print(f"汇总 CSV：{out_dir / 'summary.csv'}")
    print(f"汇总 JSON：{out_dir / 'summary.json'}")
    if args.compact_summary:
        print(f"聚合 CSV：{out_dir / 'summary_agg.csv'}")


def run_variant(args: argparse.Namespace, data_dir: Path, scene_name: str, run_spec: RunSpec, seed: int, out_dir: Path) -> dict:
    variant = run_spec.variant
    run_dir = build_run_dir(out_dir, scene_name, run_spec, seed)
    train_dir = run_dir / "train"
    render_dir = run_dir / "test_renders"
    final_ply = train_dir / "final.ply"
    metrics_csv = render_dir / "metrics.csv"
    training_summary = train_dir / "training_summary.json"

    if args.skip_existing and is_run_complete(args.csv_only, final_ply, metrics_csv, training_summary):
        print(f"跳过已存在结果：scene={scene_name} variant={variant} seed={seed}")
        return build_summary_row(args, data_dir, scene_name, run_spec, seed, run_dir, final_ply, metrics_csv, total_seconds=0.0)

    base_text = f" semantic_base={run_spec.semantic_base:g}" if uses_semantic_importance(variant) else ""
    print(f"开始训练：scene={scene_name} variant={variant}{base_text} seed={seed} iterations={args.iterations}")
    started_at = time.perf_counter()
    train_eval_every = effective_train_eval_every(args)
    save_every = 0 if args.csv_only else args.iterations
    train_cmd = [
        sys.executable,
        str(METHOD_DIR / "train_3dgs_scene.py"),
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
        "--patch-perceptual-weight",
        str(args.patch_perceptual_weight),
        "--patch-perceptual-max-size",
        str(args.patch_perceptual_max_size),
        "--train-lpips-weight",
        str(args.train_lpips_weight),
        "--train-lpips-start-step",
        str(args.train_lpips_start_step),
        "--train-lpips-max-size",
        str(args.train_lpips_max_size),
        "--train-lpips-net",
        str(args.train_lpips_net),
        "--train-lpips-backend",
        str(args.train_lpips_backend),
        "--semantic-importance-root",
        str(args.semantic_importance_root),
        "--semantic-base",
        str(run_spec.semantic_base),
        "--reallocate-fraction",
        str(args.reallocate_fraction),
        "--clone-jitter-scale",
        str(args.clone_jitter_scale),
        "--save-every",
        str(save_every),
        "--eval-every",
        str(train_eval_every),
        "--log-every",
        str(args.log_every),
    ]
    if args.lpips and train_eval_every > 0:
        train_cmd.extend(["--eval-lpips", "--lpips-net", str(args.lpips_net), "--lpips-backend", str(args.lpips_backend)])
    run_command(train_cmd)
    if not final_ply.exists():
        raise FileNotFoundError(f"训练未生成 final.ply: {final_ply}")

    print(f"开始测试渲染：scene={scene_name} variant={variant} seed={seed}")
    render_cmd = [
        sys.executable,
        str(METHOD_DIR / "render_3dgs_views.py"),
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
    if args.csv_only:
        render_cmd.append("--no-save-images")
    run_command(render_cmd)
    if not metrics_csv.exists():
        raise FileNotFoundError(f"测试渲染未生成 metrics.csv: {metrics_csv}")

    total_seconds = time.perf_counter() - started_at
    row = build_summary_row(args, data_dir, scene_name, run_spec, seed, run_dir, final_ply, metrics_csv, total_seconds)
    if args.csv_only:
        cleanup_csv_only_artifacts(run_dir)
        row["final_ply"] = ""
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
    run_spec: RunSpec,
    seed: int,
    run_dir: Path,
    final_ply: Path,
    metrics_csv: Path,
    total_seconds: float,
) -> dict:
    variant = run_spec.variant
    metrics = read_metric_means(metrics_csv)
    training = read_training_summary(run_dir / "train" / "training_summary.json")
    best_fields = flatten_best_metric_fields(read_best_metrics(run_dir / "train" / "best_metrics.json"))
    semantic_enabled = bool(training.get("uses_semantic_importance", uses_semantic_importance(variant)))
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
        "train_eval_every": training.get("eval_every", int(args.train_eval_every)),
        "train_eval_lpips": training.get("eval_lpips"),
        "patch_size": int(args.patch_size),
        "patch_edge_weight": float(args.patch_edge_weight),
        "patch_detail_lambda": float(args.patch_detail_lambda),
        "patch_perceptual_weight": training.get("patch_perceptual_weight", float(args.patch_perceptual_weight)),
        "patch_perceptual_max_size": training.get("patch_perceptual_max_size", int(args.patch_perceptual_max_size)),
        "train_lpips_weight": training.get("train_lpips_weight", float(args.train_lpips_weight)),
        "train_lpips_start_step": training.get("train_lpips_start_step", int(args.train_lpips_start_step)),
        "train_lpips_max_size": training.get("train_lpips_max_size", int(args.train_lpips_max_size)),
        "train_lpips_net": training.get("train_lpips_net", str(args.train_lpips_net) if args.train_lpips_weight > 0.0 else ""),
        "train_lpips_backend": training.get("train_lpips_backend", str(args.train_lpips_backend) if args.train_lpips_weight > 0.0 else ""),
        "semantic_importance_root": training.get("semantic_importance_root"),
        "semantic_base": training.get("semantic_base", run_spec.semantic_base) if semantic_enabled else "",
        "uses_semantic_importance": semantic_enabled,
        "reallocate_fraction": training.get("reallocate_fraction", float(args.reallocate_fraction)),
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
        "best_psnr": best_fields["best_psnr"],
        "best_psnr_step": best_fields["best_psnr_step"],
        "best_psnr_checkpoint": best_fields["best_psnr_checkpoint"],
        "best_ssim": best_fields["best_ssim"],
        "best_ssim_step": best_fields["best_ssim_step"],
        "best_ssim_checkpoint": best_fields["best_ssim_checkpoint"],
        "best_lpips": best_fields["best_lpips"],
        "best_lpips_step": best_fields["best_lpips_step"],
        "best_lpips_checkpoint": best_fields["best_lpips_checkpoint"],
        "best_metrics_json": training.get("best_metrics_json", str(run_dir / "train" / "best_metrics.json") if (run_dir / "train" / "best_metrics.json").exists() else ""),
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


def build_run_specs(variants: list[str], semantic_base: float, semantic_base_values: list[float]) -> list[RunSpec]:
    specs: list[RunSpec] = []
    for variant in variants:
        normalized_variant = normalize_densification_mode(variant)
        if uses_semantic_importance(normalized_variant) and semantic_base_values:
            for base in semantic_base_values:
                specs.append(RunSpec(normalized_variant, float(base), format_semantic_base_label(float(base))))
        else:
            specs.append(RunSpec(normalized_variant, float(semantic_base), ""))
    return specs


def build_run_dir(out_dir: Path, scene_name: str, run_spec: RunSpec, seed: int) -> Path:
    if run_spec.semantic_base_label:
        return out_dir / scene_name / run_spec.variant / run_spec.semantic_base_label / f"seed_{seed:03d}"
    return out_dir / scene_name / run_spec.variant / f"seed_{seed:03d}"


def format_semantic_base_label(value: float) -> str:
    return f"base_{value:.2f}".replace("-", "m").replace(".", "p")


def effective_train_eval_every(args: argparse.Namespace) -> int:
    return 0 if bool(args.csv_only) else max(int(args.train_eval_every), 0)


def is_run_complete(csv_only: bool, final_ply: Path, metrics_csv: Path, training_summary: Path) -> bool:
    if csv_only:
        return metrics_csv.exists() and training_summary.exists()
    return final_ply.exists() and metrics_csv.exists()


def cleanup_csv_only_artifacts(run_dir: Path) -> None:
    patterns = [
        "train/final.ply",
        "train/checkpoints/*.ply",
        "train/best/*.ply",
        "train/previews/*.png",
        "test_renders/*_render.png",
        "test_renders/*_gt.png",
    ]
    for pattern in patterns:
        for path in run_dir.glob(pattern):
            if path.is_file():
                path.unlink()


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


def read_best_metrics(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


FULL_SUMMARY_FIELDS = [
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
        "train_eval_every",
        "train_eval_lpips",
        "patch_size",
        "patch_edge_weight",
        "patch_detail_lambda",
        "patch_perceptual_weight",
        "patch_perceptual_max_size",
        "train_lpips_weight",
        "train_lpips_start_step",
        "train_lpips_max_size",
        "train_lpips_net",
        "train_lpips_backend",
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
        "best_psnr",
        "best_psnr_step",
        "best_psnr_checkpoint",
        "best_ssim",
        "best_ssim_step",
        "best_ssim_checkpoint",
        "best_lpips",
        "best_lpips_step",
        "best_lpips_checkpoint",
        "best_metrics_json",
        "gaussian_count",
        "train_seconds",
        "total_seconds",
        "data",
        "run_dir",
        "final_ply",
        "metrics_csv",
]

COMPACT_SUMMARY_FIELDS = [
    "scene",
    "variant",
    "patch_perceptual_weight",
    "patch_perceptual_max_size",
    "train_lpips_weight",
    "semantic_base",
    "seed",
    "psnr",
    "ssim",
    "lpips",
    "l1",
    "gaussian_count",
    "train_seconds",
    "total_seconds",
]

AGGREGATE_METRICS = ["psnr", "ssim", "lpips", "l1", "gaussian_count", "train_seconds", "total_seconds"]


def write_summary(out_dir: Path, rows: list[dict], compact: bool = False) -> None:
    fields = COMPACT_SUMMARY_FIELDS if compact else FULL_SUMMARY_FIELDS
    output_rows = [project_row(row, fields) for row in rows]
    with open(out_dir / "summary.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in output_rows:
            writer.writerow(row)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(output_rows, handle, ensure_ascii=False, indent=2)
    if compact:
        write_aggregate_summary(out_dir / "summary_agg.csv", rows)


def write_aggregate_summary(path: Path, rows: list[dict]) -> None:
    fields = ["scene", "variant", "semantic_base", "seeds", "count"]
    for metric in AGGREGATE_METRICS:
        fields.extend([f"{metric}_mean", f"{metric}_std"])
    grouped: dict[tuple[str, str, str], list[dict]] = {}
    for row in rows:
        key = (str(row.get("scene", "")), str(row.get("variant", "")), str(row.get("semantic_base", "")))
        grouped.setdefault(key, []).append(row)

    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for key in sorted(grouped):
            group_rows = grouped[key]
            output = {
                "scene": key[0],
                "variant": key[1],
                "semantic_base": key[2],
                "seeds": " ".join(str(row.get("seed", "")) for row in sorted(group_rows, key=lambda item: int(item.get("seed", 0)))),
                "count": len(group_rows),
            }
            for metric in AGGREGATE_METRICS:
                values = [value for value in (to_float(row.get(metric)) for row in group_rows) if value is not None]
                output[f"{metric}_mean"] = float(sum(values) / len(values)) if values else ""
                output[f"{metric}_std"] = sample_std(values) if len(values) > 1 else (0.0 if values else "")
            writer.writerow(output)


def project_row(row: dict, fields: list[str]) -> dict:
    return {field: row.get(field, "") for field in fields}


def to_float(value) -> float | None:
    if value in ("", None):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def sample_std(values: list[float]) -> float:
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


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

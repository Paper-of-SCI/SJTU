"""Minimal gsplat 3DGS training entrypoint for a COLMAP scene."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_DENSIFY_STOP_STEP = 15_000
DEFAULT_DENSIFY_GRAD_THRESHOLD = 2.0e-6

from modules import (
    BestMetricTracker,
    DENSIFICATION_MODE_CHOICES,
    Camera,
    DensificationConfig,
    DensificationController,
    GaussianModel,
    GaussianRenderer,
    OptimConfig,
    PatchGuidedDensificationConfig,
    PatchGuidedDensificationController,
    build_3dgs_optimizer,
    evaluate_cameras,
    exponential_lr,
    flatten_best_metric_fields,
    normalize_densification_mode,
    photometric_loss,
    set_group_lr,
    uses_patch_densifier,
    uses_reallocation,
    uses_semantic_importance,
    validate_semantic_importance_root,
)
from methods.ours_denstify.lpips_backend import LPIPS_BACKEND_CHOICES, build_ours_lpips_evaluator
from methods.semantic_importance import SemanticImportanceProvider
from utils.dataset_loaders import load_colmap_dataset
from utils.image_utils import compute_psnr, save_image
from utils.ply_io import gaussians_to_ply_dict, write_ply


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a minimal 3DGS model on a COLMAP scene.")
    parser.add_argument("--data", default="src/datasets/SeathruNeRF_dataset/Curasao", help="COLMAP scene directory.")
    parser.add_argument("--out", default="outputs/3dgs_scene", help="Output directory.")
    parser.add_argument("--iterations", type=int, default=7000, help="Training iterations.")
    parser.add_argument("--factor", type=int, default=-1, help="Image downscale factor for training; -1 keeps images at original size unless width exceeds 1600.")
    parser.add_argument("--target-height", type=int, default=0, help="Resize images to this height while preserving aspect ratio; 0 uses --factor.")
    parser.add_argument("--target-width", type=int, default=0, help="Resize images to this width while preserving aspect ratio; 0 uses --target-height or --factor.")
    parser.add_argument("--holdout", type=int, default=8, help="Every Nth image is held out by the loader.")
    parser.add_argument("--holdout-offset", type=int, default=0, help="Offset used when selecting every Nth held-out image.")
    parser.add_argument("--sh-degree", type=int, default=3, help="Maximum spherical harmonics degree.")
    parser.add_argument("--lambda-dssim", type=float, default=0.2, help="Photometric DSSIM weight.")
    parser.add_argument("--save-every", type=int, default=1000, help="Save preview/checkpoint interval.")
    parser.add_argument("--log-every", type=int, default=50, help="Console log interval.")
    parser.add_argument("--eval-every", type=int, default=500, help="Evaluate held-out test views every N iterations; set 0 to disable.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--densification-mode",
        default="standard_3dgs",
        choices=list(DENSIFICATION_MODE_CHOICES),
        help="Densification strategy. standard and standard_3dgs are the baseline path.",
    )
    parser.add_argument(
        "--densify-grad-threshold",
        type=float,
        default=DEFAULT_DENSIFY_GRAD_THRESHOLD,
        help="Screen-space gradient threshold for clone/split; default is calibrated for this gsplat training path.",
    )
    parser.add_argument("--densify-start-step", type=int, default=500, help="First iteration that may run densification.")
    parser.add_argument("--densify-stop-step", type=int, default=0, help="Densification phase boundary; 0 uses an official-3DGS-style default.")
    parser.add_argument("--densify-interval", type=int, default=100, help="Densification interval in iterations.")
    parser.add_argument("--opacity-reset-interval", type=int, default=3000, help="Opacity reset interval during the densification phase; 0 disables resets.")
    parser.add_argument("--patch-size", type=int, default=16, help="Patch size for patch-guided densification.")
    parser.add_argument("--patch-edge-weight", type=float, default=0.75, help="Edge multiplier for patch detail scoring.")
    parser.add_argument("--patch-detail-lambda", type=float, default=2.0, help="Patch detail multiplier on screen-space gradients.")
    parser.add_argument("--semantic-importance-root", default="", help="Root directory for per-image semantic importance masks.")
    parser.add_argument("--semantic-base", type=float, default=0.2, help="Minimum semantic multiplier for semantic patch-guided densification.")
    parser.add_argument("--reallocate-fraction", type=float, default=0.10, help="Low-detail Gaussian fraction to reallocate in patch_reallocate modes.")
    parser.add_argument("--clone-jitter-scale", type=float, default=0.05, help="Scale-relative position jitter for patch-guided clones.")
    parser.add_argument("--eval-lpips", action="store_true", help="Compute LPIPS during held-out training evaluation.")
    parser.add_argument("--lpips-net", default="vgg", choices=["alex", "vgg", "squeeze"], help="LPIPS backbone used when --eval-lpips is enabled.")
    parser.add_argument(
        "--lpips-backend",
        default="lpips",
        choices=list(LPIPS_BACKEND_CHOICES),
        help="LPIPS implementation used during held-out training evaluation. seasplat calls methods/seasplat/lpipsPyTorch.",
    )
    parser.add_argument("--disable-densification", action="store_true", help="Turn off clone/split/prune.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.densification_mode = normalize_densification_mode(args.densification_mode)
    validate_semantic_importance_root([args.densification_mode], args.semantic_importance_root)
    if not torch.cuda.is_available():
        raise RuntimeError("当前训练脚本需要 CUDA；gsplat 渲染训练建议在 GPU 上运行。")

    # 固定随机种子，保证随机选相机、split 等训练行为尽量可复现。
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")

    # 解析输入输出路径。输入是 COLMAP 场景目录，输出包括预览图、阶段 checkpoint 和 final.ply。
    data_dir = resolve_input_path(args.data)
    out_dir = resolve_output_path(args.out)
    preview_dir = out_dir / "previews"
    checkpoint_dir = out_dir / "checkpoints"
    best_dir = out_dir / "best"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.save_every > 0:
        preview_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 读取 COLMAP 数据：相机内参、相机位姿、图像路径、稀疏点云。
    # opengl=False 表示保留 COLMAP/OpenCV 相机坐标约定，和当前 gsplat 投影链路一致。
    scene = load_colmap_dataset(
        str(data_dir),
        split="train",
        load_images=False,
        factor=args.factor,
        target_height=args.target_height,
        target_width=args.target_width,
        holdout=args.holdout,
        holdout_offset=args.holdout_offset,
        opengl=False,
    )
    test_scene = load_colmap_dataset(
        str(data_dir),
        split="test",
        load_images=False,
        factor=args.factor,
        target_height=args.target_height,
        target_width=args.target_width,
        holdout=args.holdout,
        holdout_offset=args.holdout_offset,
        opengl=False,
    )
    if scene.point_cloud_xyz is None or scene.point_cloud_rgb is None:
        raise RuntimeError("训练需要 COLMAP 点云初始化 GaussianModel。")

    # 用 COLMAP 稀疏点云初始化 3D Gaussian：
    # 点坐标 -> means，点颜色 -> SH 颜色，近邻距离 -> 初始尺度。
    model = GaussianModel.from_point_cloud(
        scene.point_cloud_xyz,
        scene.point_cloud_rgb,
        sh_degree=args.sh_degree,
        device=device,
    )
    # renderer 只负责把 Gaussian + Camera 渲染成图片；白色背景和多数 3DGS 实现保持一致。
    renderer = GaussianRenderer(background=(1.0, 1.0, 1.0))

    # optimizer 直接优化 Gaussian 的 raw 参数：means、SH 颜色、opacity、scale、rotation。
    optimizer = build_3dgs_optimizer(model, OptimConfig())

    # 位置学习率单独做指数衰减：前期让 Gaussian 多移动，后期收小步长精修。
    position_lr = exponential_lr(1.6e-4, 1.6e-6, max_steps=args.iterations, delay_steps=1000, delay_mult=0.01)

    semantic_enabled = uses_semantic_importance(args.densification_mode)
    semantic_importance_root = ""
    semantic_provider = None
    if semantic_enabled:
        semantic_root = resolve_input_path(args.semantic_importance_root)
        semantic_importance_root = str(semantic_root)
        semantic_provider = SemanticImportanceProvider(semantic_root, data_dir.name, device)

    # densifier 根据屏幕空间梯度动态 clone/split/prune Gaussian，提高细节表达能力。
    effective_densify_stop_step = resolve_densify_stop_step(args.iterations, args.densify_stop_step)
    densify_config = DensificationConfig(
        start_step=max(int(args.densify_start_step), 0),
        stop_step=effective_densify_stop_step,
        interval=max(int(args.densify_interval), 0),
        grad_threshold=args.densify_grad_threshold,
        scene_extent=float(scene.scene_extent),
        percent_dense=0.01,
        min_opacity=0.005,
        opacity_reset_interval=max(int(args.opacity_reset_interval), 0),
    )
    patch_densifier_enabled = uses_patch_densifier(args.densification_mode)
    effective_reallocate_fraction = args.reallocate_fraction if uses_reallocation(args.densification_mode) else 0.0
    if patch_densifier_enabled:
        densifier = PatchGuidedDensificationController(
            PatchGuidedDensificationConfig(
                densification=densify_config,
                patch_size=args.patch_size,
                edge_weight=args.patch_edge_weight,
                detail_lambda=args.patch_detail_lambda,
                semantic_base=args.semantic_base,
                reallocate_fraction=effective_reallocate_fraction,
                clone_jitter_scale=args.clone_jitter_scale,
            )
        )
    else:
        densifier = DensificationController(densify_config)

    # 把 SceneData 中的每张图封装成 Camera。Camera 内部会按需读取 GT 图像到 GPU。
    train_cameras = build_cameras(scene, device)
    test_cameras = build_cameras(test_scene, device)
    eval_lpips_enabled = bool(args.eval_lpips and args.eval_every > 0)
    lpips_evaluator = build_ours_lpips_evaluator(device, args.lpips_net, args.lpips_backend) if eval_lpips_enabled else None
    best_tracker = BestMetricTracker(include_lpips=eval_lpips_enabled) if args.eval_every > 0 else None
    best_metrics_path = out_dir / "best_metrics.json"
    if best_tracker is not None:
        best_dir.mkdir(parents=True, exist_ok=True)
    camera_indices = list(range(len(train_cameras)))
    viewpoint_stack: list[int] = []
    gpu_name = torch.cuda.get_device_name(device)
    total_mem_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)
    print(f"设备：CUDA GPU='{gpu_name}' 显存={total_mem_gb:.2f}GB")
    print(f"数据集：{data_dir}")
    print(
        f"数据划分：训练={len(train_cameras)} 张，测试={len(test_cameras)} 张，"
        f"holdout={args.holdout}，holdout_offset={args.holdout_offset}，分辨率={scene.width}x{scene.height}，"
        f"target_height={args.target_height}，target_width={args.target_width}，降采样 factor={args.factor}"
    )
    print(
        f"初始 Gaussian 数量：{model.num_gaussians}，训练步数：{args.iterations}，"
        f"启用致密化：{not args.disable_densification}，densify_grad_threshold={args.densify_grad_threshold:g}"
    )
    print(
        f"densify_start_step={densify_config.start_step} densify_stop_step={densify_config.stop_step} "
        f"densify_interval={densify_config.interval} opacity_reset_interval={densify_config.opacity_reset_interval}"
    )
    print(
        f"densification_mode={args.densification_mode} patch_size={args.patch_size} "
        f"edge_weight={args.patch_edge_weight:g} detail_lambda={args.patch_detail_lambda:g} "
        f"semantic_base={args.semantic_base:g} reallocate_fraction={effective_reallocate_fraction:g}"
    )
    if semantic_enabled:
        print(f"semantic_importance_root={semantic_importance_root}")
    print(
        f"heldout_eval_every={args.eval_every} eval_lpips={eval_lpips_enabled} "
        f"lpips_backend={args.lpips_backend if eval_lpips_enabled else ''}"
    )
    print(f"输出目录：{out_dir}")

    started_at = time.perf_counter()
    progress = tqdm(range(1, args.iterations + 1), desc="训练进度", unit="步", dynamic_ncols=True)
    for step in progress:
        step_started_at = time.perf_counter()

        # 和官方 3DGS 一样：每轮把训练图随机不放回地用一遍，用完再重新装满。
        if not viewpoint_stack:
            viewpoint_stack = camera_indices.copy()
        camera_index = viewpoint_stack.pop(random.randrange(len(viewpoint_stack)))
        camera = train_cameras[camera_index]
        gt_image = require_camera_image(camera)

        optimizer.zero_grad(set_to_none=True)
        set_group_lr(optimizer, "means", position_lr(step))

        # 前向：当前 Gaussian 从当前相机视角渲染一张图。
        render = renderer.render(model, camera)

        # 图像监督：渲染图和 GT 图计算 L1 + DSSIM。
        loss, parts = photometric_loss(render.image.clamp(0.0, 1.0), gt_image, lambda_dssim=args.lambda_dssim)

        # 反向传播会把图像误差传回 Gaussian 参数，然后 Adam 更新这些参数。
        loss.backward()
        optimizer.step()

        stats = None
        if not args.disable_densification:
            # 用本轮反传得到的屏幕空间梯度决定是否 clone/split/prune。
            if patch_densifier_enabled:
                semantic_importance = None
                if semantic_provider is not None:
                    semantic_importance = semantic_provider.load(camera.image_path, camera.width, camera.height)
                stats = densifier.update(model, render, optimizer, step, gt_image, semantic_importance=semantic_importance)
            else:
                stats = densifier.update(model, render, optimizer, step)

        if step == 1 or step % args.log_every == 0:
            with torch.no_grad():
                # 这里是当前训练视角 PSNR，只用于看训练过程是否在变好。
                psnr = compute_psnr(
                    render.image.detach().clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy(),
                    gt_image.detach().permute(1, 2, 0).cpu().numpy(),
                )
            densify_text = ""
            if stats is not None and stats.densified:
                densify_text = (
                    f" clone={stats.cloned} split={stats.split} prune={stats.pruned} realloc={stats.reallocated}"
                    f" total={stats.total} high_grad={stats.high_grad} grad_max={stats.grad_max:.2e}"
                    f" detail_max={stats.patch_detail_max:.3f}"
                )
            if stats is not None and stats.opacity_reset:
                densify_text += " opacity_reset=1"
            progress.set_postfix(
                {
                    "loss": f"{float(loss.detach()):.4f}",
                    "训练PSNR": f"{psnr:.2f}",
                    "G": model.num_gaussians,
                    "单步": f"{time.perf_counter() - step_started_at:.2f}s",
                }
            )
            tqdm.write(
                f"第 {step:06d} 步 | loss={float(loss.detach()):.6f} | "
                f"L1={float(parts['l1'].detach()):.6f} | SSIM={float(parts['ssim'].detach()):.4f} | "
                f"训练PSNR={psnr:.2f} | Gaussian={model.num_gaussians}{densify_text}"
            )

        if best_tracker is not None and (step % args.eval_every == 0 or step == args.iterations):
            # 测试集评估不反传，只衡量 held-out 视角渲染质量，并保存各指标最优 checkpoint。
            eval_metrics = evaluate_cameras(model, renderer, test_cameras, lpips_evaluator)
            improved = best_tracker.update(step, eval_metrics)
            for metric_name in improved:
                best_checkpoint = best_dir / f"best_{metric_name}.ply"
                save_checkpoint(best_checkpoint, model)
                best_tracker.set_checkpoint(metric_name, str(best_checkpoint))
            save_best_metrics(best_metrics_path, best_tracker.to_dict())
            improved_text = ",".join(improved) if improved else "-"
            tqdm.write(
                f"测试评估 | 第 {step:06d} 步 | 测试图像={len(test_cameras)} | "
                f"{format_eval_metrics(eval_metrics)} | best更新={improved_text}"
            )

        if should_save_training_artifacts(step, args.iterations, args.save_every):
            # 保存当前训练视角预览图和 Gaussian 参数 checkpoint。
            save_preview(preview_dir / f"step_{step:06d}.png", render.image)
            save_checkpoint(checkpoint_dir / f"step_{step:06d}.ply", model)

    elapsed = time.perf_counter() - started_at
    save_checkpoint(out_dir / "final.ply", model)
    save_training_summary(
        out_dir / "training_summary.json",
        args,
        data_dir,
        out_dir,
        model.num_gaussians,
        elapsed,
        best_tracker.to_dict() if best_tracker is not None else None,
    )
    print(f"训练完成：最终模型已保存到 {out_dir / 'final.ply'}，总耗时 {format_duration(elapsed)}")


def build_cameras(scene, device: torch.device) -> list[Camera]:
    """Convert all SceneData entries into renderable Camera objects."""
    return [Camera.from_scene_data(scene, index, device=device, load_image=True) for index in range(len(scene.image_paths))]


def require_camera_image(camera: Camera) -> Tensor:
    """Return the GT image tensor, or fail loudly if the camera was built without one."""
    if camera.image is None:
        raise RuntimeError(f"相机缺少 GT 图像，无法计算图像损失: {camera.image_path}")
    return camera.image


def save_preview(path: Path, image: torch.Tensor) -> None:
    """Save a rendered CHW tensor as an RGB preview image."""
    array = image.detach().clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()
    save_image(str(path), array)


def save_checkpoint(path: Path, model: GaussianModel) -> None:
    """Bake the current GaussianModel parameters into a 3DGS PLY checkpoint."""
    with torch.no_grad():
        data = gaussians_to_ply_dict(
            model.means.detach().cpu().numpy(),
            model.log_scales.detach().cpu().numpy(),
            model.quats.detach().cpu().numpy(),
            model.logit_opacities.detach().cpu().numpy(),
            model.features_dc.detach().cpu().numpy(),
            model.features_rest.detach().cpu().numpy(),
        )
    write_ply(str(path), data)


def should_save_training_artifacts(step: int, iterations: int, save_every: int) -> bool:
    """Return whether this step should write preview/checkpoint artifacts."""
    return int(save_every) > 0 and (int(step) == 1 or int(step) % int(save_every) == 0 or int(step) == int(iterations))


def save_best_metrics(path: Path, best_metrics: dict) -> None:
    """Persist best metric records and evaluation history."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(best_metrics, handle, ensure_ascii=False, indent=2)


def save_training_summary(
    path: Path,
    args: argparse.Namespace,
    data_dir: Path,
    out_dir: Path,
    final_gaussians: int,
    elapsed_seconds: float,
    best_metrics: dict | None = None,
) -> None:
    """Persist narrow run metadata consumed by benchmark aggregation."""
    effective_reallocate_fraction = args.reallocate_fraction if uses_reallocation(args.densification_mode) else 0.0
    best_fields = flatten_best_metric_fields(best_metrics)
    summary = {
        "data": str(data_dir),
        "out": str(out_dir),
        "iterations": int(args.iterations),
        "factor": int(args.factor),
        "target_height": int(args.target_height),
        "target_width": int(args.target_width),
        "holdout": int(args.holdout),
        "holdout_offset": int(args.holdout_offset),
        "seed": int(args.seed),
        "densification_mode": args.densification_mode,
        "disable_densification": bool(args.disable_densification),
        "eval_every": int(args.eval_every),
        "eval_lpips": bool(args.eval_lpips and args.eval_every > 0),
        "lpips_net": str(args.lpips_net) if args.eval_lpips and args.eval_every > 0 else "",
        "lpips_backend": str(args.lpips_backend) if args.eval_lpips and args.eval_every > 0 else "",
        "densify_grad_threshold": float(args.densify_grad_threshold),
        "densify_start_step": int(args.densify_start_step),
        "densify_stop_step": int(args.densify_stop_step),
        "effective_densify_stop_step": int(resolve_densify_stop_step(args.iterations, args.densify_stop_step)),
        "densify_interval": int(args.densify_interval),
        "opacity_reset_interval": int(args.opacity_reset_interval),
        "patch_size": int(args.patch_size),
        "patch_edge_weight": float(args.patch_edge_weight),
        "patch_detail_lambda": float(args.patch_detail_lambda),
        "semantic_importance_root": (
            str(resolve_input_path(args.semantic_importance_root))
            if args.semantic_importance_root and uses_semantic_importance(args.densification_mode)
            else ""
        ),
        "semantic_base": float(args.semantic_base),
        "uses_semantic_importance": bool(uses_semantic_importance(args.densification_mode)),
        "reallocate_fraction": float(effective_reallocate_fraction),
        "clone_jitter_scale": float(args.clone_jitter_scale),
        "best_metrics_json": str(out_dir / "best_metrics.json") if best_metrics else "",
        "final_gaussians": int(final_gaussians),
        "elapsed_seconds": float(elapsed_seconds),
    }
    summary.update(best_fields)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)


def format_eval_metrics(metrics: dict[str, float | None]) -> str:
    parts = []
    for name, label, precision in [("psnr", "PSNR", 2), ("ssim", "SSIM", 4), ("l1", "L1", 5), ("lpips", "LPIPS", 4)]:
        value = metrics.get(name)
        if value is None:
            continue
        parts.append(f"{label}={value:.{precision}f}")
    return " ".join(parts) if parts else "无有效指标"


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


def resolve_input_path(path: str) -> Path:
    candidate = Path(path).expanduser()
    if candidate.exists():
        return candidate
    root_candidate = ROOT / candidate
    if root_candidate.exists():
        return root_candidate
    raise FileNotFoundError(f"找不到路径: {path}；也尝试过 {root_candidate}")


def resolve_output_path(path: str) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return ROOT / candidate


def resolve_densify_stop_step(iterations: int, requested_stop_step: int) -> int:
    """Return the exclusive stop boundary for densification and opacity resets."""
    if requested_stop_step > 0:
        return int(requested_stop_step)
    return max(min(int(iterations) - 500, DEFAULT_DENSIFY_STOP_STEP), 501)


if __name__ == "__main__":
    main()

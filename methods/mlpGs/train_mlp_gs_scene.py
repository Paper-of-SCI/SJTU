"""Train the first MLP-parameterized 3DGS model on a COLMAP scene."""

from __future__ import annotations

import argparse
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gaussian_image_fusion import FusionCameraBatch, GaussianImageFusion
from mlp_densification import MLPDensificationController
from mlp_gaussian_model import MLPGaussianModel
from modules import (
    Camera,
    DensificationConfig,
    GaussianModel,
    GaussianRenderer,
    exponential_lr,
    photometric_loss,
    set_group_lr,
    ssim,
)
from utils.dataset_loaders import load_colmap_dataset
from utils.image_utils import compute_psnr, save_image
from utils.ply_io import gaussians_to_ply_dict, write_ply
from vit_patch_memory import FrozenViTPatchExtractor, ViTPatchMemory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an MLP-parameterized 3DGS model on a COLMAP scene.")
    parser.add_argument("--data", default="src/datasets/SeathruNeRF_dataset/Curasao", help="COLMAP scene directory.")
    parser.add_argument("--out", default="outputs/mlpGs_scene", help="Output directory.")
    parser.add_argument("--iterations", type=int, default=7000, help="Training iterations.")
    parser.add_argument("--factor", type=int, default=4, help="Image downscale factor for training.")
    parser.add_argument("--holdout", type=int, default=8, help="Every Nth image is held out by the loader.")
    parser.add_argument("--sh-degree", type=int, default=3, help="Maximum spherical harmonics degree.")
    parser.add_argument("--lambda-dssim", type=float, default=0.2, help="Photometric DSSIM weight.")
    parser.add_argument("--save-every", type=int, default=1000, help="Save preview/checkpoint interval.")
    parser.add_argument("--log-every", type=int, default=50, help="Console log interval.")
    parser.add_argument("--eval-every", type=int, default=500, help="Evaluate held-out test views every N iterations; set 0 to disable.")
    parser.add_argument("--eval-lpips", action="store_true", help="Also compute LPIPS during eval and save best_test_lpips checkpoints.")
    parser.add_argument("--eval-lpips-size", type=int, default=512, help="Resize long side before training-time LPIPS; 0 keeps full resolution.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--mlp-hidden-dim", type=int, default=128, help="Hidden width of the Gaussian-parameter MLP.")
    parser.add_argument("--mlp-hidden-layers", type=int, default=3, help="Number of hidden layers in the MLP.")
    parser.add_argument("--mlp-lr", type=float, default=1.0e-3, help="Adam learning rate for MLP parameters.")
    parser.add_argument("--feature-dim", type=int, default=32, help="Per-Gaussian learnable feature dimension.")
    parser.add_argument("--feature-lr", type=float, default=1.0e-2, help="Adam learning rate for per-Gaussian features.")
    parser.add_argument("--feature-init-std", type=float, default=0.01, help="Initial stddev for per-Gaussian features.")
    parser.add_argument("--feature-split-noise-std", type=float, default=0.01, help="Noise stddev added to split-child features.")
    parser.add_argument("--mlp-weight-decay", type=float, default=0.0, help="Adam weight decay for MLP parameters.")
    parser.add_argument("--freeze-gaussian-base", action="store_true", help="Freeze direct Gaussian raw tensors and reproduce the older pure-MLP residual mode.")
    parser.add_argument("--position-lr-init", type=float, default=1.6e-4, help="Initial LR for trainable Gaussian base means.")
    parser.add_argument("--position-lr-final", type=float, default=1.6e-6, help="Final LR for trainable Gaussian base means.")
    parser.add_argument("--position-lr-delay-steps", type=int, default=1000, help="Delay steps for trainable Gaussian base means LR.")
    parser.add_argument("--position-lr-delay-mult", type=float, default=0.01, help="Delay multiplier for trainable Gaussian base means LR.")
    parser.add_argument("--base-feature-lr", type=float, default=2.5e-3, help="LR for trainable Gaussian base DC SH features.")
    parser.add_argument("--base-feature-rest-lr-scale", type=float, default=0.05, help="LR multiplier for trainable non-DC SH features.")
    parser.add_argument("--base-opacity-lr", type=float, default=5.0e-2, help="LR for trainable Gaussian base opacities.")
    parser.add_argument("--base-scaling-lr", type=float, default=5.0e-3, help="LR for trainable Gaussian base scales.")
    parser.add_argument("--base-rotation-lr", type=float, default=1.0e-3, help="LR for trainable Gaussian base rotations.")
    parser.add_argument("--use-vit-memory", action="store_true", help="Condition anchors on frozen ViT patch memory from training views.")
    parser.add_argument("--vit-weights", default="DEFAULT", help="torchvision ViT_B_16 weights name; use 'none' for random weights.")
    parser.add_argument("--vit-image-size", type=int, default=224, help="Square ViT input size; larger values create denser patch memory.")
    parser.add_argument("--vit-batch-size", type=int, default=4, help="Batch size for precomputing frozen ViT patch tokens.")
    parser.add_argument("--vit-topk-views", type=int, default=4, help="Top-K visible source views used by each Gaussian.")
    parser.add_argument("--vit-patch-window", type=int, default=1, help="Patch window radius around each projected Gaussian.")
    parser.add_argument("--fusion-dim", type=int, default=64, help="Hidden dimension for Gaussian-image cross-attention fusion.")
    parser.add_argument("--fusion-heads", type=int, default=4, help="Attention heads for Gaussian-image fusion.")
    parser.add_argument("--fusion-lr", type=float, default=1.0e-3, help="Adam learning rate for ViT-memory fusion parameters.")
    parser.add_argument("--fusion-chunk-size", type=int, default=4096, help="Number of Gaussians fused per chunk.")
    parser.add_argument("--fusion-residual-scale", type=float, default=1.0, help="Multiplier for the ViT fusion residual added to anchor features.")
    parser.add_argument("--fusion-output-init-std", type=float, default=0.0, help="Stddev for nonzero fusion output init; 0 keeps the old zero-init behavior.")
    parser.add_argument("--densify-grad-threshold", type=float, default=2.0e-5, help="Screen-space gradient threshold for clone/split.")
    parser.add_argument("--densify-from", type=int, default=500, help="Start MLP-GS densification at this step.")
    parser.add_argument("--densify-until", type=int, default=0, help="Stop densification at this step; 0 means iterations - 500.")
    parser.add_argument("--densification-interval", type=int, default=100, help="Run densification every N steps.")
    parser.add_argument("--disable-densification", action="store_true", help="Turn off MLP-GS clone/split/prune.")
    parser.add_argument("--min-opacity", type=float, default=0.005, help="Prune Gaussians with opacity below this threshold.")
    parser.add_argument("--max-screen-radius", type=float, default=0.0, help="Prune Gaussians larger than this screen radius; 0 disables it.")
    return parser.parse_args()


@dataclass(frozen=True)
class EvalMetrics:
    psnr: float
    ssim: float
    l1: float
    lpips: float | None = None


@dataclass
class ViTConditioner:
    fusion: GaussianImageFusion
    camera_batch: FusionCameraBatch
    patch_memory: ViTPatchMemory
    chunk_size: int


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("当前训练脚本需要 CUDA；gsplat 渲染训练建议在 GPU 上运行。")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")

    data_dir = resolve_input_path(args.data)
    out_dir = resolve_output_path(args.out)
    preview_dir = out_dir / "previews"
    checkpoint_dir = out_dir / "checkpoints"
    preview_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    scene = load_colmap_dataset(str(data_dir), split="train", load_images=False, factor=args.factor, holdout=args.holdout, opengl=False)
    test_scene = load_colmap_dataset(str(data_dir), split="test", load_images=False, factor=args.factor, holdout=args.holdout, opengl=False)
    if scene.point_cloud_xyz is None or scene.point_cloud_rgb is None:
        raise RuntimeError("训练需要 COLMAP 点云初始化 MLP-GS。")

    base_model = GaussianModel.from_point_cloud(
        scene.point_cloud_xyz,
        scene.point_cloud_rgb,
        sh_degree=args.sh_degree,
        device=device,
    )
    model = MLPGaussianModel.from_gaussian_model(
        base_model,
        hidden_dim=args.mlp_hidden_dim,
        hidden_layers=args.mlp_hidden_layers,
        feature_dim=args.feature_dim,
        feature_init_std=args.feature_init_std,
        feature_split_noise_std=args.feature_split_noise_std,
        train_base=not args.freeze_gaussian_base,
    ).to(device)
    renderer = GaussianRenderer(background=(1.0, 1.0, 1.0))
    lpips_evaluator = LPIPSEvaluator(device, max_size=args.eval_lpips_size) if args.eval_lpips else None

    train_cameras = build_cameras(scene, device)
    test_cameras = build_cameras(test_scene, device)
    conditioner = build_vit_conditioner(args, train_cameras, model, device)
    optimizer = build_optimizer(model, args, conditioner)
    position_lr = exponential_lr(
        args.position_lr_init,
        args.position_lr_final,
        max_steps=args.iterations,
        delay_steps=args.position_lr_delay_steps,
        delay_mult=args.position_lr_delay_mult,
    )
    densify_until = args.densify_until if args.densify_until > 0 else max(args.iterations - 500, args.densify_from + 1)
    densifier = MLPDensificationController(
        DensificationConfig(
            start_step=args.densify_from,
            stop_step=densify_until,
            interval=args.densification_interval,
            grad_threshold=args.densify_grad_threshold,
            scene_extent=float(scene.scene_extent),
            percent_dense=0.01,
            min_opacity=args.min_opacity,
            max_screen_radius=args.max_screen_radius if args.max_screen_radius > 0 else None,
            opacity_reset_interval=0,
        )
    )

    camera_indices = list(range(len(train_cameras)))
    viewpoint_stack: list[int] = []

    gpu_name = torch.cuda.get_device_name(device)
    total_mem_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)
    model_params = count_trainable_parameters(model)
    fusion_params = count_trainable_parameters(conditioner.fusion) if conditioner is not None else 0
    trainable_params = model_params + fusion_params
    print(f"设备：CUDA GPU='{gpu_name}' 显存={total_mem_gb:.2f}GB")
    print(f"数据集：{data_dir}")
    print(
        f"数据划分：训练={len(train_cameras)} 张，测试={len(test_cameras)} 张，"
        f"holdout={args.holdout}，分辨率={scene.width}x{scene.height}，降采样 factor={args.factor}"
    )
    print(
        f"MLP-GS：初始 Gaussian 数量={model.num_gaussians}，可训练参数量={trainable_params:,}，"
        f"hidden_dim={args.mlp_hidden_dim}，hidden_layers={args.mlp_hidden_layers}，"
        f"feature_dim={args.feature_dim}，mlp_lr={args.mlp_lr:g}，feature_lr={args.feature_lr:g}"
    )
    print(
        f"Gaussian base：{'可训练' if model.train_base else '冻结'}，"
        f"position_lr={args.position_lr_init:g}->{args.position_lr_final:g}，"
        f"base_feature_lr={args.base_feature_lr:g}，opacity_lr={args.base_opacity_lr:g}，"
        f"scaling_lr={args.base_scaling_lr:g}，rotation_lr={args.base_rotation_lr:g}"
    )
    if conditioner is not None:
        print(
            f"ViT-memory：启用，source_views={conditioner.patch_memory.num_views}，"
            f"vit_image_size={conditioner.patch_memory.image_size}，"
            f"patch={conditioner.patch_memory.grid_width}x{conditioner.patch_memory.grid_height}，"
            f"token_dim={conditioner.patch_memory.token_dim}，topk={args.vit_topk_views}，"
            f"window={(2 * args.vit_patch_window + 1)}x{(2 * args.vit_patch_window + 1)}，"
            f"fusion_params={fusion_params:,}，fusion_lr={args.fusion_lr:g}，"
            f"fusion_residual_scale={args.fusion_residual_scale:g}，fusion_output_init_std={args.fusion_output_init_std:g}"
        )
    else:
        print("ViT-memory：未启用")
    print(
        f"致密化：{not args.disable_densification}，start={args.densify_from}，stop={densify_until}，"
        f"interval={args.densification_interval}，grad_threshold={args.densify_grad_threshold:g}，"
        f"min_opacity={args.min_opacity:g}"
    )
    print("注意：这里一个 anchor 就是一个 Gaussian；clone/split 新增的是下一轮会输入 MLP 的高斯位置。")
    print(f"输出目录：{out_dir}")

    started_at = time.perf_counter()
    best_loss = float("inf")
    best_loss_step = 0
    best_test_psnr = float("-inf")
    best_test_psnr_step = 0
    best_test_ssim = float("-inf")
    best_test_ssim_step = 0
    best_test_l1 = float("inf")
    best_test_l1_step = 0
    best_test_lpips = float("inf")
    best_test_lpips_step = 0
    progress = tqdm(range(1, args.iterations + 1), desc="MLP-GS 训练进度", unit="步", dynamic_ncols=True)
    for step in progress:
        step_started_at = time.perf_counter()

        if not viewpoint_stack:
            viewpoint_stack = camera_indices.copy()
        camera_index = viewpoint_stack.pop(random.randrange(len(viewpoint_stack)))
        camera = train_cameras[camera_index]
        gt_image = require_camera_image(camera)

        optimizer.zero_grad(set_to_none=True)
        if model.train_base:
            set_group_lr(optimizer, "base_means", position_lr(step))
        apply_vit_conditioning(model, conditioner, enable_grad=True)
        render = renderer.render(model, camera)
        loss, parts = photometric_loss(render.image.clamp(0.0, 1.0), gt_image, lambda_dssim=args.lambda_dssim)
        loss.backward()
        optimizer.step()

        stats = None
        if not args.disable_densification:
            apply_vit_conditioning(model, conditioner, enable_grad=False)
            stats = densifier.update(model, render, optimizer, step)

        current_loss = float(loss.detach())
        if current_loss < best_loss:
            best_loss = current_loss
            best_loss_step = step
            save_preview(out_dir / "best_loss.png", render.image)
            save_ply_checkpoint(out_dir / "best_loss.ply", model, conditioner)
            save_mlp_checkpoint(
                out_dir / "best_loss_mlp.pt",
                model,
                args,
                conditioner,
                extra={
                    "best_loss": best_loss,
                    "best_loss_step": best_loss_step,
                    "best_loss_camera_index": camera_index,
                    "best_loss_image_path": str(camera.image_path),
                },
            )

        if step == 1 or step % args.log_every == 0:
            with torch.no_grad():
                psnr = compute_psnr(
                    render.image.detach().clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy(),
                    gt_image.detach().permute(1, 2, 0).cpu().numpy(),
                )
            densify_text = ""
            if stats is not None and stats.densified:
                densify_text = (
                    f" clone={stats.cloned} split={stats.split} prune={stats.pruned} total={stats.total}"
                    f" high_grad={stats.high_grad} grad_max={stats.grad_max:.2e}"
                )
            progress.set_postfix(
                {
                    "loss": f"{float(loss.detach()):.4f}",
                    "best": f"{best_loss:.4f}",
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

        if args.eval_every > 0 and (step == 1 or step % args.eval_every == 0 or step == args.iterations):
            eval_metrics = evaluate_metrics(model, renderer, test_cameras, lpips_evaluator, conditioner)
            lpips_text = f" | 测试LPIPS={eval_metrics.lpips:.4f}" if eval_metrics.lpips is not None else ""
            tqdm.write(
                f"测试评估 | 第 {step:06d} 步 | 测试图像={len(test_cameras)} | "
                f"测试PSNR={eval_metrics.psnr:.2f} | 测试SSIM={eval_metrics.ssim:.4f} | "
                f"测试L1={eval_metrics.l1:.5f}{lpips_text}"
            )
            if is_finite(eval_metrics.psnr) and eval_metrics.psnr > best_test_psnr:
                best_test_psnr = eval_metrics.psnr
                best_test_psnr_step = step
                save_eval_checkpoint(out_dir, "best_test_psnr", model, args, step, eval_metrics, conditioner)
                tqdm.write(f"保存 best_test_psnr | 第 {step:06d} 步 | PSNR={best_test_psnr:.2f}")
            if is_finite(eval_metrics.ssim) and eval_metrics.ssim > best_test_ssim:
                best_test_ssim = eval_metrics.ssim
                best_test_ssim_step = step
                save_eval_checkpoint(out_dir, "best_test_ssim", model, args, step, eval_metrics, conditioner)
                tqdm.write(f"保存 best_test_ssim | 第 {step:06d} 步 | SSIM={best_test_ssim:.4f}")
            if is_finite(eval_metrics.l1) and eval_metrics.l1 < best_test_l1:
                best_test_l1 = eval_metrics.l1
                best_test_l1_step = step
                save_eval_checkpoint(out_dir, "best_test_l1", model, args, step, eval_metrics, conditioner)
                tqdm.write(f"保存 best_test_l1 | 第 {step:06d} 步 | L1={best_test_l1:.5f}")
            if eval_metrics.lpips is not None and is_finite(eval_metrics.lpips) and eval_metrics.lpips < best_test_lpips:
                best_test_lpips = eval_metrics.lpips
                best_test_lpips_step = step
                save_eval_checkpoint(out_dir, "best_test_lpips", model, args, step, eval_metrics, conditioner)
                tqdm.write(f"保存 best_test_lpips | 第 {step:06d} 步 | LPIPS={best_test_lpips:.4f}")

        if step == 1 or step % args.save_every == 0 or step == args.iterations:
            save_preview(preview_dir / f"step_{step:06d}.png", render.image)
            save_ply_checkpoint(checkpoint_dir / f"step_{step:06d}.ply", model, conditioner)

    save_ply_checkpoint(out_dir / "final.ply", model, conditioner)
    save_mlp_checkpoint(out_dir / "final_mlp.pt", model, args, conditioner)
    best_eval_text = ""
    if best_test_psnr_step > 0:
        best_eval_text = (
            f"，best_test_psnr.ply={out_dir / 'best_test_psnr.ply'}，"
            f"best_test_psnr={best_test_psnr:.2f}@step={best_test_psnr_step}，"
            f"best_test_ssim={best_test_ssim:.4f}@step={best_test_ssim_step}，"
            f"best_test_l1={best_test_l1:.5f}@step={best_test_l1_step}"
        )
        if best_test_lpips_step > 0:
            best_eval_text += f"，best_test_lpips={best_test_lpips:.4f}@step={best_test_lpips_step}"
    print(
        f"训练完成：final.ply={out_dir / 'final.ply'}，final_mlp.pt={out_dir / 'final_mlp.pt'}，"
        f"best_loss.ply={out_dir / 'best_loss.ply'}，best_loss={best_loss:.6f}@step={best_loss_step}，"
        f"总耗时 {format_duration(time.perf_counter() - started_at)}{best_eval_text}"
    )


def build_cameras(scene, device: torch.device) -> list[Camera]:
    """Convert all SceneData entries into renderable Camera objects."""
    return [Camera.from_scene_data(scene, index, device=device, load_image=True) for index in range(len(scene.image_paths))]


def require_camera_image(camera: Camera) -> Tensor:
    """Return the GT image tensor, or fail loudly if the camera was built without one."""
    if camera.image is None:
        raise RuntimeError(f"相机缺少 GT 图像，无法计算图像损失: {camera.image_path}")
    return camera.image


def build_vit_conditioner(
    args: argparse.Namespace,
    train_cameras: list[Camera],
    model: MLPGaussianModel,
    device: torch.device,
) -> ViTConditioner | None:
    """Build frozen ViT patch memory and the trainable fusion module."""
    if not args.use_vit_memory:
        return None
    if model.feature_dim <= 0:
        raise ValueError("--use-vit-memory 需要 --feature-dim > 0")

    print("ViT-memory：开始提取训练图 patch tokens（ViT 主干冻结）...")
    extractor = FrozenViTPatchExtractor(weights=args.vit_weights, image_size=args.vit_image_size, device=device)
    patch_memory = extractor.build_memory(train_cameras, batch_size=args.vit_batch_size)
    del extractor
    torch.cuda.empty_cache()

    camera_batch = FusionCameraBatch.from_cameras(train_cameras, device=device)
    fusion = GaussianImageFusion(
        anchor_feature_dim=model.feature_dim,
        vit_token_dim=patch_memory.token_dim,
        fusion_dim=args.fusion_dim,
        num_heads=args.fusion_heads,
        topk_views=args.vit_topk_views,
        patch_window=args.vit_patch_window,
        residual_scale=args.fusion_residual_scale,
        output_init_std=args.fusion_output_init_std,
    ).to(device)
    return ViTConditioner(
        fusion=fusion,
        camera_batch=camera_batch,
        patch_memory=patch_memory,
        chunk_size=args.fusion_chunk_size,
    )


def apply_vit_conditioning(model: MLPGaussianModel, conditioner: ViTConditioner | None, enable_grad: bool) -> None:
    """Set current fused anchor features on the MLP-GS model."""
    if conditioner is None:
        model.clear_conditioned_anchor_features()
        return
    with torch.set_grad_enabled(enable_grad):
        conditioned_features = conditioner.fusion(
            model.anchor_xyz,
            model.normalized_inputs_for(model.anchor_xyz),
            model.anchor_features,
            conditioner.camera_batch,
            conditioner.patch_memory,
            chunk_size=conditioner.chunk_size,
        )
    model.set_conditioned_anchor_features(conditioned_features)


def build_optimizer(model: MLPGaussianModel, args: argparse.Namespace, conditioner: ViTConditioner | None) -> torch.optim.Adam:
    """Build Adam for shared MLP weights plus per-Gaussian learnable features."""
    param_groups = [
        {"params": list(model.mlp.parameters()), "lr": args.mlp_lr, "weight_decay": args.mlp_weight_decay, "name": "mlp"},
        {"params": [model.anchor_features], "lr": args.feature_lr, "weight_decay": 0.0, "name": "anchor_features"},
    ]
    if model.train_base:
        param_groups.extend(
            [
                {"params": [model.base_means], "lr": args.position_lr_init, "weight_decay": 0.0, "name": "base_means"},
                {"params": [model.base_features_dc], "lr": args.base_feature_lr, "weight_decay": 0.0, "name": "base_features_dc"},
                {
                    "params": [model.base_features_rest],
                    "lr": args.base_feature_lr * args.base_feature_rest_lr_scale,
                    "weight_decay": 0.0,
                    "name": "base_features_rest",
                },
                {"params": [model.base_logit_opacities], "lr": args.base_opacity_lr, "weight_decay": 0.0, "name": "base_logit_opacities"},
                {"params": [model.base_log_scales], "lr": args.base_scaling_lr, "weight_decay": 0.0, "name": "base_log_scales"},
                {"params": [model.base_quats], "lr": args.base_rotation_lr, "weight_decay": 0.0, "name": "base_quats"},
            ]
        )
    if conditioner is not None:
        param_groups.append(
            {"params": list(conditioner.fusion.parameters()), "lr": args.fusion_lr, "weight_decay": 0.0, "name": "image_fusion"}
        )
    return torch.optim.Adam(
        param_groups,
        eps=1.0e-15,
    )

def count_trainable_parameters(module: torch.nn.Module) -> int:
    return sum(param.numel() for param in module.parameters() if param.requires_grad)


def save_preview(path: Path, image: Tensor) -> None:
    """Save a rendered CHW tensor as an RGB preview image."""
    array = image.detach().clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()
    save_image(str(path), array)


def save_ply_checkpoint(path: Path, model: MLPGaussianModel, conditioner: ViTConditioner | None = None) -> None:
    """Bake the current MLP-predicted Gaussian parameters into a 3DGS PLY."""
    apply_vit_conditioning(model, conditioner, enable_grad=False)
    with torch.no_grad():
        raw = model.export_tensors()
        data = gaussians_to_ply_dict(
            raw.means.cpu().numpy(),
            raw.log_scales.cpu().numpy(),
            raw.quats.cpu().numpy(),
            raw.logit_opacities.cpu().numpy(),
            raw.features_dc.cpu().numpy(),
            raw.features_rest.cpu().numpy(),
        )
    write_ply(str(path), data)


def save_mlp_checkpoint(
    path: Path,
    model: MLPGaussianModel,
    args: argparse.Namespace,
    conditioner: ViTConditioner | None = None,
    extra: dict | None = None,
) -> None:
    """Save the MLP weights plus fixed anchors/base tensors for future reuse."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload_extra = {
        "args": vars(args),
        "checkpoint_format": "mlpGs.hybrid_base.v2" if conditioner is None else "mlpGs.hybrid_base.vit_memory.v2",
    }
    if conditioner is not None:
        payload_extra.update(
            {
                "fusion_state_dict": conditioner.fusion.state_dict(),
                "fusion_config": {
                    "fusion_dim": conditioner.fusion.fusion_dim,
                    "fusion_heads": conditioner.fusion.num_heads,
                    "vit_topk_views": conditioner.fusion.topk_views,
                    "vit_patch_window": conditioner.fusion.patch_window,
                    "fusion_chunk_size": conditioner.chunk_size,
                    "fusion_residual_scale": conditioner.fusion.residual_scale,
                    "fusion_output_init_std": conditioner.fusion.output_init_std,
                },
                "vit_memory": conditioner.patch_memory.metadata(),
            }
        )
    if extra:
        payload_extra.update(extra)
    torch.save(
        model.checkpoint_payload(payload_extra),
        path,
    )


def save_eval_checkpoint(
    out_dir: Path,
    name: str,
    model: MLPGaussianModel,
    args: argparse.Namespace,
    step: int,
    metrics: EvalMetrics,
    conditioner: ViTConditioner | None = None,
) -> None:
    """Save the baked PLY and resumable MLP checkpoint for a best eval metric."""
    save_ply_checkpoint(out_dir / f"{name}.ply", model, conditioner)
    save_mlp_checkpoint(
        out_dir / f"{name}_mlp.pt",
        model,
        args,
        conditioner,
        extra={
            "best_eval_name": name,
            "best_eval_step": step,
            "eval_psnr": metrics.psnr,
            "eval_ssim": metrics.ssim,
            "eval_l1": metrics.l1,
            "eval_lpips": metrics.lpips,
        },
    )


@torch.no_grad()
def evaluate_metrics(
    model: MLPGaussianModel,
    renderer: GaussianRenderer,
    cameras: list[Camera],
    lpips_evaluator: "LPIPSEvaluator | None" = None,
    conditioner: ViTConditioner | None = None,
) -> EvalMetrics:
    """Render held-out cameras and return mean test metrics."""
    apply_vit_conditioning(model, conditioner, enable_grad=False)
    psnr_values = []
    ssim_values = []
    l1_values = []
    lpips_values = []
    for camera in cameras:
        gt_image = require_camera_image(camera)
        render = renderer.render(model, camera)
        image = render.image.detach().clamp(0.0, 1.0)
        psnr_values.append(
            compute_psnr(
                image.permute(1, 2, 0).cpu().numpy(),
                gt_image.detach().permute(1, 2, 0).cpu().numpy(),
            )
        )
        ssim_values.append(float(ssim(image, gt_image).detach()))
        l1_values.append(float(torch.mean(torch.abs(image - gt_image)).detach()))
        if lpips_evaluator is not None:
            lpips_values.append(lpips_evaluator(image, gt_image))
    return EvalMetrics(
        psnr=float(np.mean(psnr_values)) if psnr_values else float("nan"),
        ssim=float(np.mean(ssim_values)) if ssim_values else float("nan"),
        l1=float(np.mean(l1_values)) if l1_values else float("nan"),
        lpips=float(np.mean(lpips_values)) if lpips_values else None,
    )


class LPIPSEvaluator:
    """Optional LPIPS wrapper used only when --eval-lpips is enabled."""

    def __init__(self, device: torch.device, max_size: int = 512) -> None:
        try:
            import lpips
        except ImportError as exc:
            raise ImportError("计算 eval LPIPS 需要安装 lpips：pip install lpips") from exc
        self.model = lpips.LPIPS(net="vgg").to(device).eval()
        self.max_size = int(max_size)

    @torch.no_grad()
    def __call__(self, image: Tensor, gt: Tensor) -> float:
        image_bchw = image.unsqueeze(0) * 2.0 - 1.0
        gt_bchw = gt.unsqueeze(0) * 2.0 - 1.0
        image_bchw, gt_bchw = self._resize_for_eval(image_bchw, gt_bchw)
        return float(self.model(image_bchw, gt_bchw).detach().reshape(-1)[0])

    def _resize_for_eval(self, image: Tensor, gt: Tensor) -> tuple[Tensor, Tensor]:
        if self.max_size <= 0:
            return image, gt
        _, _, height, width = image.shape
        long_side = max(height, width)
        if long_side <= self.max_size:
            return image, gt
        scale = self.max_size / float(long_side)
        new_height = max(1, int(round(height * scale)))
        new_width = max(1, int(round(width * scale)))
        size = (new_height, new_width)
        image = F.interpolate(image, size=size, mode="bilinear", align_corners=False)
        gt = F.interpolate(gt, size=size, mode="bilinear", align_corners=False)
        return image, gt


def is_finite(value: float) -> bool:
    return bool(np.isfinite(value))


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


if __name__ == "__main__":
    main()

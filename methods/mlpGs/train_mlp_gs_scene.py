"""Train the first MLP-parameterized 3DGS model on a COLMAP scene."""

from __future__ import annotations

import argparse
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

from mlp_densification import MLPDensificationController
from mlp_gaussian_model import MLPGaussianModel
from modules import Camera, DensificationConfig, GaussianModel, GaussianRenderer, photometric_loss
from utils.dataset_loaders import load_colmap_dataset
from utils.image_utils import compute_psnr, save_image
from utils.ply_io import gaussians_to_ply_dict, write_ply


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
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--mlp-hidden-dim", type=int, default=128, help="Hidden width of the Gaussian-parameter MLP.")
    parser.add_argument("--mlp-hidden-layers", type=int, default=3, help="Number of hidden layers in the MLP.")
    parser.add_argument("--mlp-lr", type=float, default=1.0e-3, help="Adam learning rate for MLP parameters.")
    parser.add_argument("--feature-dim", type=int, default=32, help="Per-Gaussian learnable feature dimension.")
    parser.add_argument("--feature-lr", type=float, default=1.0e-2, help="Adam learning rate for per-Gaussian features.")
    parser.add_argument("--feature-init-std", type=float, default=0.01, help="Initial stddev for per-Gaussian features.")
    parser.add_argument("--feature-split-noise-std", type=float, default=0.01, help="Noise stddev added to split-child features.")
    parser.add_argument("--mlp-weight-decay", type=float, default=0.0, help="Adam weight decay for MLP parameters.")
    parser.add_argument("--densify-grad-threshold", type=float, default=2.0e-5, help="Screen-space gradient threshold for clone/split.")
    parser.add_argument("--densify-from", type=int, default=500, help="Start MLP-GS densification at this step.")
    parser.add_argument("--densify-until", type=int, default=0, help="Stop densification at this step; 0 means iterations - 500.")
    parser.add_argument("--densification-interval", type=int, default=100, help="Run densification every N steps.")
    parser.add_argument("--disable-densification", action="store_true", help="Turn off MLP-GS clone/split/prune.")
    parser.add_argument("--min-opacity", type=float, default=0.005, help="Prune Gaussians with opacity below this threshold.")
    parser.add_argument("--max-screen-radius", type=float, default=0.0, help="Prune Gaussians larger than this screen radius; 0 disables it.")
    parser.add_argument(
        "--opacity-reset-interval",
        type=int,
        default=0,
        help="Reset predicted opacities every N steps; disabled by default for MLP-GS.",
    )
    return parser.parse_args()


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
    ).to(device)
    renderer = GaussianRenderer(background=(1.0, 1.0, 1.0))
    optimizer = build_optimizer(model, args)
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
            opacity_reset_interval=args.opacity_reset_interval,
        )
    )

    train_cameras = build_cameras(scene, device)
    test_cameras = build_cameras(test_scene, device)
    camera_indices = list(range(len(train_cameras)))
    viewpoint_stack: list[int] = []

    gpu_name = torch.cuda.get_device_name(device)
    total_mem_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print(f"设备：CUDA GPU='{gpu_name}' 显存={total_mem_gb:.2f}GB")
    print(f"数据集：{data_dir}")
    print(
        f"数据划分：训练={len(train_cameras)} 张，测试={len(test_cameras)} 张，"
        f"holdout={args.holdout}，分辨率={scene.width}x{scene.height}，降采样 factor={args.factor}"
    )
    print(
        f"MLP-GS：初始 Gaussian 数量={model.num_gaussians}，MLP 参数量={trainable_params:,}，"
        f"hidden_dim={args.mlp_hidden_dim}，hidden_layers={args.mlp_hidden_layers}，"
        f"feature_dim={args.feature_dim}，mlp_lr={args.mlp_lr:g}，feature_lr={args.feature_lr:g}"
    )
    print(
        f"致密化：{not args.disable_densification}，start={args.densify_from}，stop={densify_until}，"
        f"interval={args.densification_interval}，grad_threshold={args.densify_grad_threshold:g}，"
        f"min_opacity={args.min_opacity:g}"
    )
    print("注意：这里一个 anchor 就是一个 Gaussian；clone/split 新增的是下一轮会输入 MLP 的高斯位置。")
    print(f"输出目录：{out_dir}")

    started_at = time.perf_counter()
    progress = tqdm(range(1, args.iterations + 1), desc="MLP-GS 训练进度", unit="步", dynamic_ncols=True)
    for step in progress:
        step_started_at = time.perf_counter()

        if not viewpoint_stack:
            viewpoint_stack = camera_indices.copy()
        camera_index = viewpoint_stack.pop(random.randrange(len(viewpoint_stack)))
        camera = train_cameras[camera_index]
        gt_image = require_camera_image(camera)

        optimizer.zero_grad(set_to_none=True)
        render = renderer.render(model, camera)
        loss, parts = photometric_loss(render.image.clamp(0.0, 1.0), gt_image, lambda_dssim=args.lambda_dssim)
        loss.backward()
        optimizer.step()

        stats = None
        if not args.disable_densification:
            previous_anchor_features = model.anchor_features
            stats = densifier.update(model, render, step)
            sync_anchor_feature_optimizer(optimizer, previous_anchor_features, model.anchor_features)

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

        if args.eval_every > 0 and (step == 1 or step % args.eval_every == 0 or step == args.iterations):
            eval_psnr = evaluate_psnr(model, renderer, test_cameras)
            tqdm.write(f"测试评估 | 第 {step:06d} 步 | 测试图像={len(test_cameras)} | 测试PSNR={eval_psnr:.2f}")

        if step == 1 or step % args.save_every == 0 or step == args.iterations:
            save_preview(preview_dir / f"step_{step:06d}.png", render.image)
            save_ply_checkpoint(checkpoint_dir / f"step_{step:06d}.ply", model)

    save_ply_checkpoint(out_dir / "final.ply", model)
    save_mlp_checkpoint(out_dir / "final_mlp.pt", model, args)
    print(
        f"训练完成：final.ply={out_dir / 'final.ply'}，final_mlp.pt={out_dir / 'final_mlp.pt'}，"
        f"总耗时 {format_duration(time.perf_counter() - started_at)}"
    )


def build_cameras(scene, device: torch.device) -> list[Camera]:
    """Convert all SceneData entries into renderable Camera objects."""
    return [Camera.from_scene_data(scene, index, device=device, load_image=True) for index in range(len(scene.image_paths))]


def require_camera_image(camera: Camera) -> Tensor:
    """Return the GT image tensor, or fail loudly if the camera was built without one."""
    if camera.image is None:
        raise RuntimeError(f"相机缺少 GT 图像，无法计算图像损失: {camera.image_path}")
    return camera.image


def build_optimizer(model: MLPGaussianModel, args: argparse.Namespace) -> torch.optim.Adam:
    """Build Adam for shared MLP weights plus per-Gaussian learnable features."""
    return torch.optim.Adam(
        [
            {"params": list(model.mlp.parameters()), "lr": args.mlp_lr, "weight_decay": args.mlp_weight_decay, "name": "mlp"},
            {"params": [model.anchor_features], "lr": args.feature_lr, "weight_decay": 0.0, "name": "anchor_features"},
        ],
        eps=1.0e-15,
    )


def sync_anchor_feature_optimizer(
    optimizer: torch.optim.Optimizer,
    previous_anchor_features: torch.nn.Parameter,
    current_anchor_features: torch.nn.Parameter,
) -> None:
    """Point the optimizer at the resized anchor feature parameter after densification."""
    if previous_anchor_features is current_anchor_features:
        return
    for group in optimizer.param_groups:
        if group.get("name") == "anchor_features":
            optimizer.state.pop(previous_anchor_features, None)
            group["params"] = [current_anchor_features]
            optimizer.state.setdefault(current_anchor_features, {})
            return
    raise RuntimeError("optimizer is missing the anchor_features parameter group")


def save_preview(path: Path, image: Tensor) -> None:
    """Save a rendered CHW tensor as an RGB preview image."""
    array = image.detach().clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()
    save_image(str(path), array)


def save_ply_checkpoint(path: Path, model: MLPGaussianModel) -> None:
    """Bake the current MLP-predicted Gaussian parameters into a 3DGS PLY."""
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


def save_mlp_checkpoint(path: Path, model: MLPGaussianModel, args: argparse.Namespace) -> None:
    """Save the MLP weights plus fixed anchors/base tensors for future reuse."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        model.checkpoint_payload(
            {
                "args": vars(args),
                "checkpoint_format": "mlpGs.v1",
            }
        ),
        path,
    )


@torch.no_grad()
def evaluate_psnr(model: MLPGaussianModel, renderer: GaussianRenderer, cameras: list[Camera]) -> float:
    """Render held-out cameras and return their mean PSNR."""
    values = []
    for camera in cameras:
        gt_image = require_camera_image(camera)
        render = renderer.render(model, camera)
        values.append(
            compute_psnr(
                render.image.detach().clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy(),
                gt_image.detach().permute(1, 2, 0).cpu().numpy(),
            )
        )
    return float(np.mean(values)) if values else float("nan")


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

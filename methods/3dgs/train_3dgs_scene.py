"""Minimal gsplat 3DGS training entrypoint for a COLMAP scene."""

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

from modules import (
    Camera,
    DensificationConfig,
    DensificationController,
    GaussianModel,
    GaussianRenderer,
    OptimConfig,
    build_3dgs_optimizer,
    exponential_lr,
    photometric_loss,
    set_group_lr,
)
from utils.dataset_loaders import load_colmap_dataset
from utils.image_utils import compute_psnr, save_image
from utils.ply_io import gaussians_to_ply_dict, write_ply


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a minimal 3DGS model on a COLMAP scene.")
    parser.add_argument("--data", default="src/datasets/SeathruNeRF_dataset/Curasao", help="COLMAP scene directory.")
    parser.add_argument("--out", default="outputs/3dgs_scene", help="Output directory.")
    parser.add_argument("--iterations", type=int, default=7000, help="Training iterations.")
    parser.add_argument("--factor", type=int, default=4, help="Image downscale factor for training.")
    parser.add_argument("--holdout", type=int, default=8, help="Every Nth image is held out by the loader.")
    parser.add_argument("--sh-degree", type=int, default=3, help="Maximum spherical harmonics degree.")
    parser.add_argument("--lambda-dssim", type=float, default=0.2, help="Photometric DSSIM weight.")
    parser.add_argument("--save-every", type=int, default=1000, help="Save preview/checkpoint interval.")
    parser.add_argument("--log-every", type=int, default=50, help="Console log interval.")
    parser.add_argument("--eval-every", type=int, default=500, help="Evaluate held-out test views every N iterations; set 0 to disable.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--densify-grad-threshold", type=float, default=2.0e-5, help="Screen-space gradient threshold for clone/split.")
    parser.add_argument("--disable-densification", action="store_true", help="Turn off clone/split/prune.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
    preview_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 读取 COLMAP 数据：相机内参、相机位姿、图像路径、稀疏点云。
    # opengl=False 表示保留 COLMAP/OpenCV 相机坐标约定，和当前 gsplat 投影链路一致。
    scene = load_colmap_dataset(str(data_dir), split="train", load_images=False, factor=args.factor, holdout=args.holdout, opengl=False)
    test_scene = load_colmap_dataset(str(data_dir), split="test", load_images=False, factor=args.factor, holdout=args.holdout, opengl=False)
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

    # densifier 根据屏幕空间梯度动态 clone/split/prune Gaussian，提高细节表达能力。
    densifier = DensificationController(
        DensificationConfig(
            start_step=500,
            stop_step=max(args.iterations - 500, 501),
            interval=100,
            grad_threshold=args.densify_grad_threshold,
            scene_extent=float(scene.scene_extent),
            percent_dense=0.01,
            min_opacity=0.005,
            opacity_reset_interval=3000,
        )
    )

    # 把 SceneData 中的每张图封装成 Camera。Camera 内部会按需读取 GT 图像到 GPU。
    train_cameras = build_cameras(scene, device)
    test_cameras = build_cameras(test_scene, device)
    camera_indices = list(range(len(train_cameras)))
    viewpoint_stack: list[int] = []
    gpu_name = torch.cuda.get_device_name(device)
    total_mem_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)
    print(f"设备：CUDA GPU='{gpu_name}' 显存={total_mem_gb:.2f}GB")
    print(f"数据集：{data_dir}")
    print(
        f"数据划分：训练={len(train_cameras)} 张，测试={len(test_cameras)} 张，"
        f"holdout={args.holdout}，分辨率={scene.width}x{scene.height}，降采样 factor={args.factor}"
    )
    print(
        f"初始 Gaussian 数量：{model.num_gaussians}，训练步数：{args.iterations}，"
        f"启用致密化：{not args.disable_densification}，densify_grad_threshold={args.densify_grad_threshold:g}"
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
            # 测试集评估不反传，只衡量 held-out 视角渲染质量。
            eval_psnr = evaluate_psnr(model, renderer, test_cameras)
            tqdm.write(f"测试评估 | 第 {step:06d} 步 | 测试图像={len(test_cameras)} | 测试PSNR={eval_psnr:.2f}")

        if step == 1 or step % args.save_every == 0 or step == args.iterations:
            # 保存当前训练视角预览图和 Gaussian 参数 checkpoint。
            save_preview(preview_dir / f"step_{step:06d}.png", render.image)
            save_checkpoint(checkpoint_dir / f"step_{step:06d}.ply", model)

    save_checkpoint(out_dir / "final.ply", model)
    print(f"训练完成：最终模型已保存到 {out_dir / 'final.ply'}，总耗时 {format_duration(time.perf_counter() - started_at)}")


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


@torch.no_grad()
def evaluate_psnr(model: GaussianModel, renderer: GaussianRenderer, cameras: list[Camera]) -> float:
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

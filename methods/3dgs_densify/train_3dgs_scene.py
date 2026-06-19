"""Minimal gsplat 3DGS training entrypoint for a COLMAP scene.

复现顺序建议：

1. 先确认环境和数据。
   本项目在 WSL 的 sfquant 环境里跑：

       /home/leo/miniconda3/bin/conda run -n sfquant python methods/3dgs/train_3dgs_scene.py --help

   输入数据应是 COLMAP/SeathruNeRF 风格目录，常见结构包括：

       <scene>/images
       <scene>/sparse/0

   例如：

       src/datasets/SeathruNeRF_dataset/Curasao/undistorted_pinhole

2. 先跑 1 step smoke test，确认 CUDA、数据加载、渲染、PLY 保存都通：

       /home/leo/miniconda3/bin/conda run -n sfquant python methods/3dgs/train_3dgs_scene.py \
         --data src/datasets/SeathruNeRF_dataset/Curasao/undistorted_pinhole \
         --out outputs/3dgs_smoke_curasao \
         --iterations 1 \
         --save-every 1 \
         --eval-every 0

   成功后应至少看到：

       outputs/3dgs_smoke_curasao/final.ply
       outputs/3dgs_smoke_curasao/training_summary.json

3. 再跑正式训练。默认 ``--factor -1`` 的含义是：原图宽度不超过 1600
   就用原图，超过 1600 自动压到 1600；不要为了正式实验随手加
   ``--target-width``，除非只是为了显存调试。

       /home/leo/miniconda3/bin/conda run -n sfquant python methods/3dgs/train_3dgs_scene.py \
         --data src/datasets/SeathruNeRF_dataset/Curasao/undistorted_pinhole \
         --out outputs/3dgs_curasao_baseline \
         --iterations 7000 \
         --factor -1 \
         --eval-every 500 \
         --save-every 1000

4. 训练后用 render 脚本评估 test split：

       /home/leo/miniconda3/bin/conda run -n sfquant python methods/3dgs/render_3dgs_views.py \
         --data src/datasets/SeathruNeRF_dataset/Curasao/undistorted_pinhole \
         --checkpoint outputs/3dgs_curasao_baseline/final.ply \
         --out outputs/3dgs_curasao_baseline/test_renders \
         --split test \
         --factor -1

   重点看 ``metrics.csv``、``best_metrics.json`` 和 ``training_summary.json``。
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.rule import Rule
from rich.table import Table
import numpy as np
import torch
from torch import Tensor

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_DENSIFY_STOP_STEP = 15_000
DEFAULT_DENSIFY_GRAD_THRESHOLD = 2.0e-6
SYSTEM_STATUS_SAMPLE_INTERVAL = 2.0

from modules import (
    BestMetricTracker,
    PATCH_ONLY_DENSIFICATION_MODE_CHOICES,
    Camera,
    DensificationConfig,
    DensificationController,
    GaussianModel,
    GaussianRenderer,
    OptimConfig,
    PatchOnlyDensificationConfig,
    PatchOnlyDensificationController,
    build_lpips_evaluator,
    build_3dgs_optimizer,
    evaluate_cameras,
    exponential_lr,
    flatten_best_metric_fields,
    normalize_densification_mode,
    photometric_loss,
    set_group_lr,
)
from utils.dataset_loaders import load_colmap_dataset
from utils.image_utils import compute_psnr, save_image
from utils.ply_io import gaussians_to_ply_dict, write_ply


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="3DGS训练."
    )

    # 数据和输出路径。复现实验时通常只改 --data、--out、--iterations。
    # --data 指向单个 COLMAP scene；--out 每次实验都建议用新目录，避免覆盖旧结果。
    parser.add_argument(
        "--data", 
        default="src/datasets/SeathruNeRF_dataset/Curasao/undistorted_pinhole", 
        help="COLMAP scene directory."
    )
    parser.add_argument(
        "--out", 
        default="outputs/3dgs_Curasao_densifyVersion", 
        help="Output directory."
    )
    parser.add_argument(
        "--iterations", 
        type=int, 
        default=19999, 
        help="Training iterations."
    )

    # 分辨率控制优先级：
    # 1) 如果设置 --target-height 或 --target-width，就按目标边长缩放。
    # 2) 否则如果 --factor=-1，宽度 <=1600 用原图，宽度 >1600 自动压到 1600。
    # 3) 否则按整数 factor 做下采样。
    # 正式复现实验建议保留 --factor -1；显存不够时才临时用 target-width 降配。
    parser.add_argument(
        "--factor", 
        type=int, 
        default=-1, 
        help="Image downscale factor for training; -1 keeps images at original size unless width exceeds 1600."
    )
    parser.add_argument(
        "--target-height", 
        type=int, 
        default=0, 
        help="Resize images to this height while preserving aspect ratio; 0 uses --factor."
    )
    parser.add_argument(
        "--target-width", 
        type=int, 
        default=0, 
        help="Resize images to this width while preserving aspect ratio; 0 uses --target-height or --factor."
    )

    # holdout 决定 train/test split。默认每 8 张取 1 张做 test；
    # render_3dgs_views.py 评估时必须使用相同 holdout 参数，否则 test 集不一致。
    parser.add_argument(
        "--holdout", 
        type=int, 
        default=8, 
        help="Every Nth image is held out by the loader."
    )
    parser.add_argument(
        "--holdout-offset", 
        type=int, 
        default=0, 
        help="Offset used when selecting every Nth held-out image."
    )

    # 模型和基础 loss。这里是 baseline 3DGS：只优化 Gaussian 参数，不含 medium field。
    # photometric_loss = L1 + DSSIM，其中 --lambda-dssim 控制 DSSIM 权重。
    parser.add_argument(
        "--sh-degree", 
        type=int, 
        default=3, 
        help="Maximum spherical harmonics degree."
    )
    parser.add_argument(
        "--lambda-dssim", 
        type=float, 
        default=0.2, 
        help="Photometric DSSIM weight."
    )

    # 训练过程输出：
    # --save-every 控制 preview PNG 和 checkpoint PLY；
    # --eval-every 控制 held-out test 评估和 best checkpoint 保存；
    # --log-every 只影响控制台日志频率。
    parser.add_argument("--save-every", type=int, default=1000, help="Save preview/checkpoint interval.")
    parser.add_argument("--log-every", type=int, default=50, help="Console log interval.")
    parser.add_argument("--eval-every", type=int, default=500, help="Evaluate held-out test views every N iterations; set 0 to disable.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")

    # densification 是 3DGS 训练的关键阶段：根据屏幕空间梯度 clone/split/prune Gaussian。
    # patch_guided 是默认主路径；standard_3dgs 仅保留为 baseline 对照。
    parser.add_argument(
        "--densification-mode",
        default="patch_guided",
        choices=list(PATCH_ONLY_DENSIFICATION_MODE_CHOICES),
        help="Densification strategy. patch_guided uses error/edge patch detail; standard and standard_3dgs are baseline paths.",
    )
    parser.add_argument(
        "--densify-grad-threshold",
        type=float,
        default=DEFAULT_DENSIFY_GRAD_THRESHOLD,
        help="Screen-space gradient threshold for clone/split; default is calibrated for this gsplat training path.",
    )
    parser.add_argument("--densify-start-step", type=int, default=1000, help="First iteration that may run densification.")
    parser.add_argument("--densify-stop-step", type=int, default=7500, help="Densification phase boundary; 0 uses an official-3DGS-style default.")
    parser.add_argument("--densify-interval", type=int, default=100, help="Densification interval in iterations.")
    parser.add_argument("--opacity-reset-interval", type=int, default=0, help="Opacity reset interval during the densification phase; 0 disables resets.")

    # patch-guided densification 参数只在 patch_guided 下生效。
    # standard_3dgs 下保留这些参数不会改变训练结果。
    parser.add_argument("--patch-size", type=int, default=32, help="Patch size for patch-guided densification.")
    parser.add_argument("--patch-edge-weight", type=float, default=0.75, help="Edge multiplier for patch detail scoring.")
    parser.add_argument("--patch-detail-lambda", type=float, default=2.0, help="Patch detail multiplier on screen-space gradients.")
    parser.add_argument("--clone-jitter-scale", type=float, default=0.05, help="Scale-relative position jitter for patch-guided clones.")

    # eval LPIPS 只用于 held-out evaluation，不参与训练反传。
    # 如果只想快速验证训练链路，可以不加 --eval-lpips，避免额外模型加载和耗时。
    parser.add_argument("--eval-lpips", action="store_false", help="Compute LPIPS during held-out training evaluation.")
    parser.add_argument("--lpips-net", default="vgg", choices=["alex", "vgg", "squeeze"], help="LPIPS backbone used when --eval-lpips is enabled.")
    parser.add_argument(
        "--lpips-backend",
        default="lpips",
        choices=["lpips", "official_3dgs"],
        help="LPIPS implementation used during held-out training evaluation.",
    )
    parser.add_argument("--disable-densification", action="store_true", help="Turn off clone/split/prune.")
    return parser.parse_args()


def main() -> None:
    # 入口主流程按复现顺序组织：
    # 1. 解析参数和检查环境；
    # 2. 读取 COLMAP 数据；
    # 3. 初始化 Gaussian、renderer、optimizer、densifier；
    # 4. 循环训练；
    # 5. 保存 checkpoint 和 summary。
    args = parse_args()
    args.densification_mode = normalize_densification_mode(args.densification_mode)
    
    if not torch.cuda.is_available():
        raise RuntimeError("没发现cuda设备，请检查CUDA安装和环境配置。")

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
    # load_images=False 表示这里先只读相机和路径；真正的 GT 图像由 Camera 在 build_cameras 时加载。
    # 同一个 data_dir 会被读两次：split=train 用于优化，split=test 用于 held-out 评估。
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
    # 这里是从零开始训练 baseline 3DGS 的关键入口；没有读取外部 checkpoint。
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
    # 默认 stop_step=0 时，resolve_densify_stop_step 会给出官方 3DGS 风格边界：
    # 训练末尾前 500 步停止 densification，且最多到 DEFAULT_DENSIFY_STOP_STEP。
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
    patch_densifier_enabled = args.densification_mode == "patch_guided"
    if patch_densifier_enabled:
        densifier = PatchOnlyDensificationController(
            PatchOnlyDensificationConfig(
                densification=densify_config,
                patch_size=args.patch_size,
                edge_weight=args.patch_edge_weight,
                detail_lambda=args.patch_detail_lambda,
                clone_jitter_scale=args.clone_jitter_scale,
            )
        )
    else:
        densifier = DensificationController(densify_config)

    # 把 SceneData 中的每张图封装成 Camera。Camera 内部会按需读取 GT 图像到 GPU。
    # 如果显存紧张，优先降低分辨率；不要在训练 loop 内频繁改 Camera 构造逻辑。
    train_cameras = build_cameras(scene, device)
    test_cameras = build_cameras(test_scene, device)
 

    # best_tracker 只在 eval_every > 0 时启用。它会跟踪 PSNR/SSIM 最大值；
    # 如果 --eval-lpips 开启，也会跟踪 LPIPS 最小值。
    eval_lpips_enabled = bool(args.eval_lpips and args.eval_every > 0)
    lpips_evaluator = build_lpips_evaluator(device, args.lpips_net, args.lpips_backend) if eval_lpips_enabled else None
    best_tracker = BestMetricTracker(include_lpips=eval_lpips_enabled) if args.eval_every > 0 else None
    best_metrics_path = out_dir / "best_metrics.json"
    if best_tracker is not None:
        best_dir.mkdir(parents=True, exist_ok=True)
    camera_indices = list(range(len(train_cameras)))
    viewpoint_stack: list[int] = []
    gpu_name = torch.cuda.get_device_name(device)
    total_mem_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)
    rows = [
        ("设备", f"CUDA GPU='{gpu_name}'"),
        ("显存", f"{total_mem_gb:.2f}GB"),
        ("数据集", str(data_dir)),
        ("数据划分", f"训练={len(train_cameras)} 张，测试={len(test_cameras)} 张"),
        ("holdout", str(args.holdout)),
        ("holdout_offset", str(args.holdout_offset)),
        ("分辨率", f"{scene.width}x{scene.height}"),
        ("target_height", str(args.target_height)),
        ("target_width", str(args.target_width)),
        ("降采样 factor", str(args.factor)),
        ("初始 Gaussian 数量", str(model.num_gaussians)),
        ("训练步数", str(args.iterations)),
        ("启用致密化", str(not args.disable_densification)),
        ("densify_grad_threshold", f"{args.densify_grad_threshold:g}"),
        ("densify_start_step", str(densify_config.start_step)),
        ("densify_stop_step", str(densify_config.stop_step)),
        ("densify_interval", str(densify_config.interval)),
        ("opacity_reset_interval", str(densify_config.opacity_reset_interval)),
        ("densification_mode", args.densification_mode),
        ("patch_size", str(args.patch_size)),
        ("edge_weight", f"{args.patch_edge_weight:g}"),
        ("detail_lambda", f"{args.patch_detail_lambda:g}"),
        ("heldout_eval_every", str(args.eval_every)),
        ("eval_lpips", str(eval_lpips_enabled)),
        ("lpips_backend", args.lpips_backend if eval_lpips_enabled else ""),
        ("输出目录", str(out_dir)),
    ]
    print_training_config_table(rows)

    console = Console()
    console.print(Rule("[bold red]训练开始", style="blue"))
    
    started_at = time.perf_counter()
    training_progress = create_training_progress(console)
    progress_task = training_progress.add_task("训练进度", total=args.iterations)
    dashboard_state = {
        "step": f"0/{args.iterations}",
        "loss": "-",
        "l1": "-",
        "ssim": "-",
        "train_psnr": "-",
        "gaussians": str(model.num_gaussians),
        "step_time": "-",
        "camera": "-",
        "densification": "关闭" if args.disable_densification else "等待触发",
        "eval_status": "关闭" if best_tracker is None else "等待首次评估",
        "eval_step": "-",
        "eval_psnr": "-",
        "eval_ssim": "-",
        "eval_l1": "-",
        "eval_lpips": "-",
        "best_update": "-",
        "artifacts": "等待保存",
        "cpu_util": "-",
        "ram_util": "-",
        "gpu_util": "-",
        "gpu_mem": "-",
        "gpu_temp": "-",
        "gpu_power": "-",
    }
    dashboard_state.update(sample_system_status())
    last_system_sample_at = time.perf_counter()
    with Live(
        render_training_dashboard(training_progress, dashboard_state),
        console=console,
        refresh_per_second=4,
        transient=False,
    ) as live:
        for step in range(1, args.iterations + 1):
            step_started_at = time.perf_counter()

            # 和官方 3DGS 一样：每轮把训练图随机不放回地用一遍，用完再重新装满。
            # 这样每个 epoch 内不会重复抽同一张图，随机性只来自 seed 和 pop 的顺序。
            if not viewpoint_stack:
                viewpoint_stack = camera_indices.copy()
            camera_index = viewpoint_stack.pop(random.randrange(len(viewpoint_stack)))
            
            camera = train_cameras[camera_index]
            gt_image = require_camera_image(camera)

            optimizer.zero_grad(set_to_none=True)
            set_group_lr(optimizer, "means", position_lr(step))

            # 前向：当前 Gaussian 从当前相机视角渲染一张图。
            # render.image 是 CHW RGB；render.means2d/radii 会在 densification 中继续使用。
            render = renderer.render(model, camera)
            
            

            # 图像监督：渲染图和 GT 图计算 L1 + DSSIM。
            # clamp 只约束送入 loss 的 RGB 范围，避免异常值扩大损失；GT 已在 loader 中归一化到 [0, 1]。
            loss, parts = photometric_loss(render.image.clamp(0.0, 1.0), gt_image, lambda_dssim=args.lambda_dssim)

            # 反向传播会把图像误差传回 Gaussian 参数，然后 Adam 更新这些参数。
            # 注意 densification 用的是本轮 backward 后累积到 means2d 的屏幕空间梯度。
            loss.backward()
            optimizer.step()

            stats = None
            if not args.disable_densification:
                # 用本轮反传得到的屏幕空间梯度决定是否 clone/split/prune。
                if patch_densifier_enabled:
                    stats = densifier.update(model, render, optimizer, step, gt_image)
                else:
                    stats = densifier.update(model, render, optimizer, step)

            step_elapsed = time.perf_counter() - step_started_at
            
            
            dashboard_state.update(
                {
                    "step": f"{step}/{args.iterations}",
                    "loss": f"{float(loss.detach()):.4f}",
                    "l1": f"{float(parts['l1'].detach()):.5f}",
                    "ssim": f"{float(parts['ssim'].detach()):.4f}",
                    "gaussians": str(model.num_gaussians),
                    "step_time": f"{step_elapsed:.2f}s",
                    "camera": Path(camera.image_path).name,
                }
            )
            densification_text = format_densification_stats(stats)
            if densification_text:
                dashboard_state["densification"] = densification_text

            now = time.perf_counter()
            if now - last_system_sample_at >= SYSTEM_STATUS_SAMPLE_INTERVAL:
                dashboard_state.update(sample_system_status())
                last_system_sample_at = now

            if step == 1 or step % args.log_every == 0:
                with torch.no_grad():
                    # 这里是当前训练视角 PSNR，只用于看训练过程是否在变好。
                    psnr = compute_psnr(
                        render.image.detach().clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy(),
                        gt_image.detach().permute(1, 2, 0).cpu().numpy(),
                    )
                dashboard_state["train_psnr"] = f"{psnr:.2f}"
                live.console.log(
                    f"训练 | step={step:06d} loss={float(loss.detach()):.6f} "
                    f"L1={float(parts['l1'].detach()):.6f} SSIM={float(parts['ssim'].detach()):.4f} "
                    f"PSNR={psnr:.2f} Gaussian={model.num_gaussians} "
                    f"densify={dashboard_state['densification']}"
                )

            if best_tracker is not None and (step % args.eval_every == 0 or step == args.iterations):
                # 测试集评估不反传，只衡量 held-out 视角渲染质量，并保存各指标最优 checkpoint。
                # 这里的 best checkpoint 是复现实验最常用的结果，不要只看 final.ply。
                dashboard_state["eval_status"] = "评估中..."
                dashboard_state["eval_step"] = str(step)
                live.update(render_training_dashboard(training_progress, dashboard_state), refresh=True)
                eval_metrics = evaluate_cameras(model, renderer, test_cameras, lpips_evaluator)
                improved = best_tracker.update(step, eval_metrics)
                for metric_name in improved:
                    best_checkpoint = best_dir / f"best_{metric_name}.ply"
                    save_checkpoint(best_checkpoint, model)
                    best_tracker.set_checkpoint(metric_name, str(best_checkpoint))
                save_best_metrics(best_metrics_path, best_tracker.to_dict())
                improved_text = ",".join(improved) if improved else "-"
                dashboard_state.update(
                    {
                        "eval_status": "完成",
                        "eval_step": str(step),
                        "eval_psnr": format_metric_value(eval_metrics.get("psnr"), 2),
                        "eval_ssim": format_metric_value(eval_metrics.get("ssim"), 4),
                        "eval_l1": format_metric_value(eval_metrics.get("l1"), 5),
                        "eval_lpips": format_metric_value(eval_metrics.get("lpips"), 4),
                        "best_update": improved_text,
                    }
                )
                live.console.log(
                    f"测试评估 | step={step:06d} 测试图像={len(test_cameras)} "
                    f"{format_eval_metrics(eval_metrics)} best更新={improved_text}"
                )

            if should_save_training_artifacts(step, args.iterations, args.save_every):
                # 保存当前训练视角预览图和 Gaussian 参数 checkpoint。
                # preview 用来肉眼检查训练是否崩坏；checkpoint 可用于中途恢复分析，但本脚本不实现 resume。
                save_preview(preview_dir / f"step_{step:06d}.png", render.image)
                save_checkpoint(checkpoint_dir / f"step_{step:06d}.ply", model)
                dashboard_state["artifacts"] = f"step={step} preview + checkpoint"

            training_progress.update(progress_task, completed=step)
            live.update(render_training_dashboard(training_progress, dashboard_state), refresh=True)

    elapsed = time.perf_counter() - started_at
    # final.ply 是最后一步模型；如果启用了 eval_every，best/*.ply 可能比 final.ply 指标更好。
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
    console.print(Rule("[bold blue]训练结束", style="blue"))
    console.print(f"训练完成：最终模型已保存到 {out_dir / 'final.ply'}，总耗时 {format_duration(elapsed)}")


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
        "semantic_importance_root": "",
        "semantic_base": "",
        "uses_semantic_importance": False,
        "reallocate_fraction": 0.0,
        "clone_jitter_scale": float(args.clone_jitter_scale),
        "best_metrics_json": str(out_dir / "best_metrics.json") if best_metrics else "",
        "final_gaussians": int(final_gaussians),
        "elapsed_seconds": float(elapsed_seconds),
    }
    summary.update(best_fields)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)


def create_training_progress(console: Console) -> Progress:
    """Create the Rich progress bar used by the training dashboard."""
    return Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=None),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        expand=True,
    )


def render_training_dashboard(progress: Progress, state: dict[str, str]) -> Panel:
    """Render the live training dashboard from narrow string state."""
    training_metrics = Table(title="训练实时", expand=True, show_lines=False)
    training_metrics.add_column("项目", style="cyan", no_wrap=True)
    training_metrics.add_column("值", overflow="fold")
    for key, label in [
        ("step", "step"),
        ("loss", "loss"),
        ("l1", "L1"),
        ("ssim", "SSIM"),
        ("train_psnr", "训练 PSNR"),
        ("gaussians", "Gaussian"),
        ("step_time", "单步耗时"),
        ("camera", "当前图像"),
    ]:
        training_metrics.add_row(label, state.get(key, "-"))

    eval_metrics = Table(title="评测 / 状态", expand=True, show_lines=False)
    eval_metrics.add_column("项目", style="magenta", no_wrap=True)
    eval_metrics.add_column("值", overflow="fold")
    for key, label in [
        ("eval_status", "评测状态"),
        ("eval_step", "评测 step"),
        ("eval_psnr", "测试 PSNR"),
        ("eval_ssim", "测试 SSIM"),
        ("eval_l1", "测试 L1"),
        ("eval_lpips", "测试 LPIPS"),
        ("best_update", "best 更新"),
        ("densification", "densification"),
        ("artifacts", "最近保存"),
    ]:
        eval_metrics.add_row(label, state.get(key, "-"))

    system_metrics = Table(title="系统资源", expand=True, show_lines=False)
    system_metrics.add_column("项目", style="green", no_wrap=True)
    system_metrics.add_column("值", overflow="fold")
    for key, label in [
        ("cpu_util", "CPU"),
        ("ram_util", "RAM"),
        ("gpu_util", "GPU"),
        ("gpu_mem", "GPU 显存"),
        ("gpu_temp", "GPU 温度"),
        ("gpu_power", "GPU 功耗"),
    ]:
        system_metrics.add_row(label, state.get(key, "-"))

    right_column = Table.grid(expand=True)
    right_column.add_row(eval_metrics)
    right_column.add_row(system_metrics)

    columns = Table.grid(expand=True)
    columns.add_column(ratio=1)
    columns.add_column(ratio=1)
    columns.add_row(
        Panel(training_metrics, border_style="cyan"),
        Panel(right_column, border_style="magenta"),
    )

    body = Table.grid(expand=True)
    body.add_row(progress)
    body.add_row(columns)
    return Panel(body, title="3DGS DensifyVesion 训练监控", border_style="blue")


def sample_system_status() -> dict[str, str]:
    """Return a narrow, best-effort snapshot of CPU/RAM/GPU resource status."""
    status = {
        "cpu_util": "-",
        "ram_util": "-",
        "gpu_util": "-",
        "gpu_mem": "-",
        "gpu_temp": "-",
        "gpu_power": "-",
    }

    try:
        import psutil

        cpu_percent = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory()
        status["cpu_util"] = f"{cpu_percent:.0f}%"
        status["ram_util"] = f"{memory.percent:.0f}% ({format_gib(memory.used)}/{format_gib(memory.total)})"
    except Exception:
        pass

    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return status

    try:
        output = subprocess.check_output(
            [
                nvidia_smi,
                "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=1.0,
        ).strip()
        first_gpu = output.splitlines()[0]
        gpu_util, mem_used, mem_total, temp, power = [part.strip() for part in first_gpu.split(",", maxsplit=4)]
    except Exception:
        return status

    status.update(
        {
            "gpu_util": f"{gpu_util}%",
            "gpu_mem": f"{mem_used}/{mem_total} MB",
            "gpu_temp": f"{temp}C",
            "gpu_power": f"{power}W",
        }
    )
    return status


def format_gib(value: int | float) -> str:
    return f"{float(value) / (1024**3):.1f}GB"


def format_densification_stats(stats) -> str:
    """Return a compact densification status string for logs and dashboard."""
    if stats is None:
        return ""
    parts = []
    if getattr(stats, "densified", False):
        parts.extend(
            [
                f"clone={stats.cloned}",
                f"split={stats.split}",
                f"prune={stats.pruned}",
                f"total={stats.total}",
                f"high_grad={stats.high_grad}",
                f"grad_max={stats.grad_max:.2e}",
                f"detail_max={stats.patch_detail_max:.3f}",
            ]
        )
    if getattr(stats, "opacity_reset", False):
        parts.append("opacity_reset=1")
    return " ".join(parts)


def format_eval_metrics(metrics: dict[str, float | None]) -> str:
    parts = []
    for name, label, precision in [("psnr", "PSNR", 2), ("ssim", "SSIM", 4), ("l1", "L1", 5), ("lpips", "LPIPS", 4)]:
        value = metrics.get(name)
        if value is None:
            continue
        parts.append(f"{label}={value:.{precision}f}")
    return " ".join(parts) if parts else "无有效指标"


def format_metric_value(value: float | None, precision: int) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{precision}f}"


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


def print_training_config_table(rows: list[tuple[str, str]]) -> None:
    """Render startup configuration rows as a terminal table."""
    console = Console()
    table = Table(title="3DGS 训练配置", show_lines=False)
    table.add_column("项目", style="cyan", no_wrap=True)
    table.add_column("值", overflow="fold")
    for key, value in rows:
        table.add_row(key, value)
    console.print(Rule("[bold green]训练配置"))
    console.print(table)


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

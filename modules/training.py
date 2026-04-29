"""3DGS 训练主循环。

提供：
  - TrainingConfig：所有超参数的数据类
  - build_optimizer：每参数组独立学习率的 Adam 优化器
  - get_expon_lr_func：指数衰减学习率调度
  - training_step：单步前向/反向/致密化
  - evaluate：无梯度评估（PSNR / SSIM）
  - train：端到端训练入口
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.optim as optim
from torch import Tensor

from modules.gaussian_model import GaussianModel
from modules.renderer import GaussianRenderer, RenderOutput
from modules.camera import Camera, camera_from_scene_data, cameras_from_scene_data
from modules.densification import DensificationController
from modules.losses import photometric_loss

from utils.dataset_loaders import load_colmap_dataset, load_llff_dataset, load_blender_dataset, SceneData
from utils.image_utils import compute_psnr, compute_ssim, save_image


@dataclass
class TrainingConfig:
    """3DGS 训练全部超参数。"""

    # 位置学习率（指数衰减）
    position_lr_init:        float = 1.6e-4
    position_lr_final:       float = 1.6e-6
    position_lr_delay_mult:  float = 0.01
    position_lr_max_steps:   int   = 30_000

    # 其他参数的学习率
    feature_lr:   float = 0.0025   # SH DC
    opacity_lr:   float = 0.05
    scaling_lr:   float = 0.005
    rotation_lr:  float = 0.001

    # 训练长度
    num_iterations: int = 30_000

    # SH 渐进训练
    sh_degree:            int = 3
    sh_degree_up_interval: int = 1000   # 每隔多少步提升一阶 SH

    # 致密化
    densify_from_iter:      int   = 500
    densify_until_iter:     int   = 15_000
    densify_interval:       int   = 100
    densify_grad_threshold: float = 0.0002
    opacity_reset_interval: int   = 3000
    min_opacity:            float = 0.005
    max_screen_size:        Optional[float] = None
    max_world_size_percent: float = 0.01

    # 损失
    lambda_dssim: float = 0.2

    # 背景色（默认白色，Blender 数据集用）
    bg_color: Tuple[float, float, float] = (1.0, 1.0, 1.0)

    # 检查点与评估
    save_interval: int = 5000
    eval_interval: int = 1000

    # 输出与设备
    output_dir: str = "output"
    device:     str = "cuda"


# ---------------------------------------------------------------------------
# 优化器
# ---------------------------------------------------------------------------

def build_optimizer(
    gaussians: GaussianModel,
    config: TrainingConfig,
) -> optim.Adam:
    """为 GaussianModel 构建逐参数组学习率的 Adam 优化器。

    参数组命名与 DensificationController 中的修补逻辑对应：
        _means      → position_lr_init
        _sh_dc      → feature_lr
        _sh_rest    → feature_lr / 20
        _opacities  → opacity_lr
        _scales     → scaling_lr
        _rotations  → rotation_lr
    """
    param_groups = [
        {"params": [gaussians._means],     "lr": config.position_lr_init, "name": "_means"},
        {"params": [gaussians._sh_dc],     "lr": config.feature_lr,       "name": "_sh_dc"},
        {"params": [gaussians._sh_rest],   "lr": config.feature_lr / 20,  "name": "_sh_rest"},
        {"params": [gaussians._opacities], "lr": config.opacity_lr,       "name": "_opacities"},
        {"params": [gaussians._scales],    "lr": config.scaling_lr,       "name": "_scales"},
        {"params": [gaussians._rotations], "lr": config.rotation_lr,      "name": "_rotations"},
    ]
    return optim.Adam(param_groups, eps=1e-15)


def get_expon_lr_func(
    lr_init: float,
    lr_final: float,
    lr_delay_steps: int = 0,
    lr_delay_mult: float = 1.0,
    max_steps: int = 1_000_000,
) -> Callable[[int], float]:
    """返回一个 step → lr 的指数衰减函数，可选 warmup。

    Args:
        lr_init:      初始学习率。
        lr_final:     最终学习率。
        lr_delay_steps: warmup 步数（0 = 无 warmup）。
        lr_delay_mult:  warmup 期间初始乘数。
        max_steps:    衰减结束步数。

    Returns:
        callable(step) -> float。
    """
    def lr_fn(step: int) -> float:
        if step < 0 or (lr_init == 0 and lr_final == 0):
            return 0.0
        if lr_delay_steps > 0:
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * math.sin(
                0.5 * math.pi * min(step / lr_delay_steps, 1.0)
            )
        else:
            delay_rate = 1.0
        t = min(step / max_steps, 1.0)
        log_lerp = math.exp(math.log(lr_init) * (1 - t) + math.log(max(lr_final, 1e-15)) * t)
        return delay_rate * log_lerp

    return lr_fn


def _update_position_lr(
    optimizer: optim.Adam,
    step: int,
    lr_func: Callable[[int], float],
) -> None:
    """更新位置参数组的学习率。"""
    for group in optimizer.param_groups:
        if group.get("name") == "_means":
            group["lr"] = lr_func(step)


# ---------------------------------------------------------------------------
# 单步训练
# ---------------------------------------------------------------------------

def training_step(
    gaussians: GaussianModel,
    renderer: GaussianRenderer,
    camera: Camera,
    optimizer: optim.Adam,
    densification_ctrl: DensificationController,
    config: TrainingConfig,
    step: int,
    lr_func: Optional[Callable[[int], float]] = None,
) -> Dict[str, float]:
    """执行单次训练迭代。

    流程：
        1. 更新位置学习率
        2. 渲染 → RenderOutput
        3. 计算光度损失并 backward
        4. 致密化控制器更新（读取梯度统计）
        5. optimizer.step() + zero_grad()

    Args:
        gaussians: 当前高斯模型。
        renderer:  渲染器。
        camera:    当前帧相机。
        optimizer: Adam 优化器。
        densification_ctrl: 致密化控制器。
        config:    训练配置。
        step:      当前步（从 1 开始）。
        lr_func:   位置学习率函数，None 则跳过更新。

    Returns:
        含损失各分量的 float 字典，键：'l1'、'ssim'、'dssim'、'total'。
    """
    if lr_func is not None:
        _update_position_lr(optimizer, step, lr_func)

    # SH 渐进：每 sh_degree_up_interval 步提升一阶
    sh_degree_current = min(step // config.sh_degree_up_interval, config.sh_degree)

    # 前向渲染
    render_out: RenderOutput = renderer.render(
        gaussians, camera, sh_degree_override=sh_degree_current
    )

    # 光度损失
    assert camera.image is not None, "训练帧必须加载 GT 图像"
    total_loss, components = photometric_loss(
        render_out.image, camera.image, lambda_dssim=config.lambda_dssim
    )

    # 反向传播（须在 densification_ctrl.update 之前）
    total_loss.backward()

    # 致密化（读取 screenspace_means.grad）
    densification_ctrl.update(step, render_out, camera, optimizer)

    # 参数更新
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    return {k: float(v.item()) for k, v in components.items()}


# ---------------------------------------------------------------------------
# 评估
# ---------------------------------------------------------------------------

def evaluate(
    gaussians: GaussianModel,
    renderer: GaussianRenderer,
    cameras: List[Camera],
    output_dir: Optional[str] = None,
) -> Dict[str, float]:
    """对所有相机进行无梯度渲染并计算 PSNR / SSIM。

    Args:
        gaussians:  训练好的高斯模型。
        renderer:   渲染器。
        cameras:    测试相机列表。
        output_dir: 若指定，将渲染图像保存于此目录。

    Returns:
        {"psnr": float, "ssim": float}。
    """
    psnr_vals, ssim_vals = [], []

    with torch.no_grad():
        for i, cam in enumerate(cameras):
            out = renderer.render(gaussians, cam)
            img_pred = out.image.permute(1, 2, 0).cpu().numpy()  # HxWx3

            if cam.image is not None:
                img_gt = cam.image.permute(1, 2, 0).cpu().numpy()
                psnr_vals.append(compute_psnr(img_pred, img_gt))
                ssim_vals.append(compute_ssim(img_pred, img_gt))

            if output_dir is not None:
                os.makedirs(output_dir, exist_ok=True)
                save_image(os.path.join(output_dir, f"{i:05d}.png"), img_pred)

    result = {}
    if psnr_vals:
        result["psnr"] = sum(psnr_vals) / len(psnr_vals)
        result["ssim"] = sum(ssim_vals) / len(ssim_vals)
    return result


# ---------------------------------------------------------------------------
# 端到端训练入口
# ---------------------------------------------------------------------------

def train(
    data_dir: str,
    config: TrainingConfig,
    scene_format: str = "colmap",
) -> GaussianModel:
    """完整的 3DGS 训练流程。

    1. 加载数据集（COLMAP / LLFF / Blender）
    2. 从 COLMAP 稀疏点云初始化 GaussianModel
    3. 构建优化器、渲染器、致密化控制器
    4. 训练循环（随机采样相机）
    5. 定期保存 PLY checkpoint 和评估

    Args:
        data_dir:     数据集根目录。
        config:       TrainingConfig。
        scene_format: "colmap"、"llff" 或 "blender"。

    Returns:
        训练完成的 GaussianModel。
    """
    # ---- 加载数据 ----
    loader_map = {
        "colmap":  load_colmap_dataset,
        "llff":    load_llff_dataset,
        "blender": load_blender_dataset,
    }
    loader = loader_map.get(scene_format, load_colmap_dataset)
    train_scene: SceneData = loader(data_dir, split="train", load_images=True)
    test_scene:  SceneData = loader(data_dir, split="test",  load_images=True)

    train_cameras = cameras_from_scene_data(train_scene, device=config.device)
    test_cameras  = cameras_from_scene_data(test_scene,  device=config.device)

    # ---- 初始化高斯模型 ----
    import numpy as np
    if train_scene.point_cloud_xyz is not None and len(train_scene.point_cloud_xyz) > 0:
        xyz = train_scene.point_cloud_xyz
        rgb = train_scene.point_cloud_rgb
        if rgb is None:
            rgb = np.full((len(xyz), 3), 128, dtype=np.uint8)
    else:
        # 若无点云，在相机前方随机放少量点
        xyz = np.random.randn(10000, 3).astype(np.float32) * 0.5
        rgb = np.full((10000, 3), 128, dtype=np.uint8)

    gaussians = GaussianModel.from_point_cloud(
        xyz, rgb, max_sh_degree=config.sh_degree, device=config.device
    )
    print(f"[train] 初始高斯数量：{gaussians.num_gaussians}")

    # ---- 构建工具 ----
    bg_color = torch.tensor(config.bg_color, dtype=torch.float32, device=config.device)
    renderer = GaussianRenderer(
        sh_degree=config.sh_degree, bg_color=bg_color
    )
    optimizer = build_optimizer(gaussians, config)
    lr_func   = get_expon_lr_func(
        lr_init=config.position_lr_init,
        lr_final=config.position_lr_final,
        lr_delay_mult=config.position_lr_delay_mult,
        max_steps=config.position_lr_max_steps,
    )
    densification_ctrl = DensificationController(
        gaussians,
        densify_from_iter=config.densify_from_iter,
        densify_until_iter=config.densify_until_iter,
        densify_grad_threshold=config.densify_grad_threshold,
        densify_interval=config.densify_interval,
        opacity_reset_interval=config.opacity_reset_interval,
        min_opacity=config.min_opacity,
        max_screen_size=config.max_screen_size,
        max_world_size_percent=config.max_world_size_percent,
        scene_extent=train_scene.scene_scale,
        num_splits=2,
    )

    # ---- 训练循环 ----
    import random
    for step in range(1, config.num_iterations + 1):
        camera = random.choice(train_cameras)
        metrics = training_step(
            gaussians, renderer, camera, optimizer,
            densification_ctrl, config, step, lr_func
        )

        if step % 100 == 0 or step == 1:
            n = gaussians.num_gaussians
            print(
                f"[step {step:>6d}/{config.num_iterations}] "
                f"loss={metrics['total']:.4f}  "
                f"l1={metrics['l1']:.4f}  "
                f"gaussians={n}"
            )

        # 评估
        if step % config.eval_interval == 0 and test_cameras:
            eval_dir = os.path.join(config.output_dir, f"eval_{step:06d}")
            eval_metrics = evaluate(gaussians, renderer, test_cameras, output_dir=eval_dir)
            print(
                f"[eval  {step:>6d}] "
                f"PSNR={eval_metrics.get('psnr', -1):.2f}  "
                f"SSIM={eval_metrics.get('ssim', -1):.4f}"
            )

        # 保存 checkpoint
        if step % config.save_interval == 0 or step == config.num_iterations:
            ply_dir = os.path.join(
                config.output_dir, "point_cloud", f"iteration_{step}", "point_cloud.ply"
            )
            gaussians.save_ply(ply_dir)
            print(f"[ckpt ] 保存至 {ply_dir}（{gaussians.num_gaussians} 个高斯）")

    return gaussians

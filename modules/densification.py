"""自适应密度控制（Adaptive Density Control）。

在每个训练步骤中累积 2D 梯度统计，并在指定间隔执行：
  - clone：梯度大、尺度小 → 复制一份
  - split：梯度大、尺度大 → 分裂为两个更小的高斯
  - prune：不透明度过低 或 尺寸过大 → 删除
  - opacity_reset：定期将所有不透明度重置为接近零的值

关键细节：参数变异后必须同步修补 Adam 优化器的动量缓存，
否则不同大小的参数与动量张量会导致维度不匹配错误。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor

from modules.gaussian_model import GaussianModel
from modules.renderer import RenderOutput
from modules.camera import Camera


@dataclass
class DensificationStats:
    """单步致密化的统计信息，用于日志记录。"""
    step: int
    num_cloned: int
    num_split: int
    num_pruned: int
    opacity_reset: bool
    total_gaussians: int


class DensificationController:
    """管理 GaussianModel 的自适应密度控制循环。

    Args:
        gaussians: 待控制的 GaussianModel。
        densify_from_iter: 从第几步开始致密化。
        densify_until_iter: 到第几步停止致密化。
        densify_grad_threshold: 2D 梯度范数阈值，超过则致密化。
        densify_interval: 致密化执行间隔（步数）。
        opacity_reset_interval: 不透明度重置间隔（步数）。
        min_opacity: 低于此阈值的高斯被剪除。
        max_screen_size: 2D 半径超过此像素数的高斯被剪除（None 则自动计算）。
        max_world_size_percent: 3D 尺度超过 scene_extent * 此比例的高斯被剪除。
        scene_extent: 场景尺度（通常为相机位置的 90th 百分位距离）。
        num_splits: 每次分裂产生的新高斯数量（默认 2）。

    Usage::

        ctrl = DensificationController(gaussians, scene_extent=1.5)
        # 在训练循环中：
        stats = ctrl.update(step, render_output, camera, optimizer)
    """

    def __init__(
        self,
        gaussians: GaussianModel,
        densify_from_iter: int = 500,
        densify_until_iter: int = 15_000,
        densify_grad_threshold: float = 0.0002,
        densify_interval: int = 100,
        opacity_reset_interval: int = 3000,
        min_opacity: float = 0.005,
        max_screen_size: Optional[float] = None,
        max_world_size_percent: float = 0.01,
        scene_extent: float = 1.0,
        num_splits: int = 2,
    ) -> None:
        self.gaussians = gaussians
        self.densify_from_iter = densify_from_iter
        self.densify_until_iter = densify_until_iter
        self.densify_grad_threshold = densify_grad_threshold
        self.densify_interval = densify_interval
        self.opacity_reset_interval = opacity_reset_interval
        self.min_opacity = min_opacity
        self.max_screen_size = max_screen_size
        self.max_world_size_percent = max_world_size_percent
        self.scene_extent = scene_extent
        self.num_splits = num_splits

    def update(
        self,
        step: int,
        render_output: RenderOutput,
        camera: Camera,
        optimizer: torch.optim.Optimizer,
    ) -> DensificationStats:
        """每训练步调用一次，统计梯度并在必要时执行致密化。

        Args:
            step: 当前训练步（从 1 开始）。
            render_output: 当前步的渲染输出。
            camera: 当前步的相机。
            optimizer: Adam 优化器（致密化后需同步修补其状态）。

        Returns:
            DensificationStats。
        """
        # 累积梯度统计
        self._accumulate_stats(render_output)

        num_cloned = 0
        num_split = 0
        num_pruned = 0
        did_reset = False

        in_densify_range = (
            self.densify_from_iter <= step < self.densify_until_iter
        )

        if in_densify_range and step % self.densify_interval == 0:
            num_cloned, num_split, num_pruned = self._densify(camera, optimizer)

        if step % self.opacity_reset_interval == 0 and step > 0:
            self._reset_opacities(optimizer)
            did_reset = True

        return DensificationStats(
            step=step,
            num_cloned=num_cloned,
            num_split=num_split,
            num_pruned=num_pruned,
            opacity_reset=did_reset,
            total_gaussians=self.gaussians.num_gaussians,
        )

    # ---- 内部：梯度累积 ----

    def _accumulate_stats(self, render_output: RenderOutput) -> None:
        """从屏幕空间梯度累积每个高斯的梯度范数。"""
        if render_output.screenspace_means.grad is None:
            return
        grads = render_output.screenspace_means.grad        # (N, 3) 或 (N, 2)
        if grads.shape[1] >= 2:
            grads = grads[:, :2]    # 只取 x/y 分量
        vis = render_output.visibility_filter
        self.gaussians.update_gradient_stats(grads, vis)

    # ---- 内部：致密化主函数 ----

    def _densify(
        self,
        camera: Camera,
        optimizer: torch.optim.Optimizer,
    ):
        """执行 clone + split + prune，返回 (n_cloned, n_split, n_pruned)。"""
        accum = self.gaussians.gradient_accum       # (N,)
        denom = self.gaussians.gradient_denom       # (N,)
        grad_norms = accum / (denom.clamp(min=1))   # 均值梯度范数 (N,)

        # 判断是否超过梯度阈值
        large_grad = grad_norms >= self.densify_grad_threshold

        # 当前尺度（max over 3 axes）
        max_scale = self.gaussians.scales.max(dim=1).values   # (N,)

        # clone 条件：梯度大 + 尺度小
        scale_threshold = self.scene_extent * self.max_world_size_percent
        clone_mask = large_grad & (max_scale <= scale_threshold)

        # split 条件：梯度大 + 尺度大
        split_mask = large_grad & (max_scale > scale_threshold)

        n_cloned = int(clone_mask.sum().item())
        n_split  = int(split_mask.sum().item())

        # 保存旧参数用于修补 optimizer
        old_params = self._snapshot_params()

        if n_cloned > 0:
            self.gaussians.densify_and_clone(clone_mask)
        if n_split > 0:
            self.gaussians.densify_and_split(split_mask, num_splits=self.num_splits)

        if n_cloned > 0 or n_split > 0:
            new_params = self._snapshot_params()
            self._patch_optimizer(optimizer, old_params, new_params)

        # prune
        prune_mask = self._build_prune_mask(camera)
        n_pruned = int(prune_mask.sum().item())
        if n_pruned > 0:
            old_params2 = self._snapshot_params()
            self.gaussians.prune(prune_mask)
            new_params2 = self._snapshot_params()
            self._patch_optimizer_prune(optimizer, old_params2, new_params2, ~prune_mask)

        # 重置梯度统计
        self.gaussians.reset_gradient_stats()

        return n_cloned, n_split, n_pruned

    def _build_prune_mask(self, camera: Camera) -> Tensor:
        """构建需要剪除的高斯掩码（True = 剪除）。"""
        prune = (self.gaussians.opacities.squeeze(1) < self.min_opacity)

        # 3D 尺寸过大
        max_scale = self.gaussians.scales.max(dim=1).values
        prune |= max_scale > self.scene_extent * self.max_world_size_percent * 10

        # 2D 半径过大（仅在有 max_screen_size 时）
        if self.max_screen_size is not None:
            # 粗略估算 2D 半径：用最大尺度除以近似焦距
            focal = (camera.fx + camera.fy) / 2.0
            approx_radius = (max_scale / focal) * max(camera.width, camera.height)
            prune |= approx_radius > self.max_screen_size

        return prune

    def _reset_opacities(self, optimizer: torch.optim.Optimizer) -> None:
        """重置不透明度并清除 Adam 中对应的动量缓存。"""
        self.gaussians.reset_opacities(value=0.01)
        # 找到 _opacities 对应的参数组并清零动量
        for group in optimizer.param_groups:
            if group.get("name") == "_opacities":
                for p in group["params"]:
                    if p in optimizer.state:
                        optimizer.state[p]["exp_avg"].zero_()
                        optimizer.state[p]["exp_avg_sq"].zero_()

    # ---- 优化器状态修补 ----

    def _snapshot_params(self) -> Dict[str, Tensor]:
        """保存当前参数数据的快照（不保留梯度）。"""
        return {
            name: param.data.clone()
            for name, param in self.gaussians.named_parameters()
        }

    def _patch_optimizer(
        self,
        optimizer: torch.optim.Optimizer,
        old_params: Dict[str, Tensor],
        new_params: Dict[str, Tensor],
    ) -> None:
        """clone/split 后扩展 optimizer 的动量缓存。

        新增加的高斯对应的动量初始化为 0。
        """
        for group in optimizer.param_groups:
            param_name = group.get("name", "")
            for p in group["params"]:
                if p not in optimizer.state:
                    continue
                state = optimizer.state[p]
                old_size = old_params.get(param_name, p.data).shape[0]
                new_size = p.data.shape[0]
                added = new_size - old_size
                if added <= 0:
                    continue
                for key in ("exp_avg", "exp_avg_sq"):
                    if key in state:
                        pad = torch.zeros(
                            added, *state[key].shape[1:],
                            device=state[key].device,
                            dtype=state[key].dtype,
                        )
                        state[key] = torch.cat([state[key], pad], dim=0)

    def _patch_optimizer_prune(
        self,
        optimizer: torch.optim.Optimizer,
        old_params: Dict[str, Tensor],
        new_params: Dict[str, Tensor],
        keep_mask: Tensor,
    ) -> None:
        """prune 后裁剪 optimizer 的动量缓存，与被保留的高斯对齐。"""
        for group in optimizer.param_groups:
            for p in group["params"]:
                if p not in optimizer.state:
                    continue
                state = optimizer.state[p]
                for key in ("exp_avg", "exp_avg_sq"):
                    if key in state and state[key].shape[0] == keep_mask.shape[0]:
                        state[key] = state[key][keep_mask]

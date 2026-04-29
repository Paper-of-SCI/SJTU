"""3DGS 核心场景表示：GaussianModel。

以 nn.Module 形式持有所有可学习的高斯参数，提供激活属性访问器、
协方差计算、点云初始化、PLY 存读以及致密化变异方法。
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from utils.ply_io import gaussians_to_ply_dict, ply_dict_to_gaussians, read_ply, write_ply
from modules.spherical_harmonics import SH_C0, num_sh_coefficients


class GaussianModel(nn.Module):
    """3D 高斯场景模型。

    参数（均为 nn.Parameter）：
        _means:      (N, 3)  世界坐标位置
        _scales:     (N, 3)  log 尺度（激活后用 exp 得到真实尺度）
        _rotations:  (N, 4)  四元数 wxyz（激活后 L2 归一化）
        _opacities:  (N, 1)  pre-sigmoid 不透明度
        _sh_dc:      (N, 1, 3)  0 阶 SH 系数（视角无关颜色）
        _sh_rest:    (N, K, 3)  1~max 阶 SH 系数，K=(max_sh_degree+1)^2-1

    梯度统计缓存（register_buffer，不参与优化）：
        _gradient_accum: (N,) 2D 梯度范数累积
        _gradient_denom: (N,) 累积步数（用于求均值）
    """

    def __init__(self, max_sh_degree: int = 3) -> None:
        super().__init__()
        self.max_sh_degree = max_sh_degree
        self._active_sh_degree = 0

        # 参数会在 from_point_cloud 中通过 _reset_parameters 初始化
        self._means:     nn.Parameter
        self._scales:    nn.Parameter
        self._rotations: nn.Parameter
        self._opacities: nn.Parameter
        self._sh_dc:     nn.Parameter
        self._sh_rest:   nn.Parameter

    def _reset_parameters(
        self,
        means: Tensor,
        scales: Tensor,
        rotations: Tensor,
        opacities: Tensor,
        sh_dc: Tensor,
        sh_rest: Tensor,
    ) -> None:
        self._means     = nn.Parameter(means)
        self._scales    = nn.Parameter(scales)
        self._rotations = nn.Parameter(rotations)
        self._opacities = nn.Parameter(opacities)
        self._sh_dc     = nn.Parameter(sh_dc)
        self._sh_rest   = nn.Parameter(sh_rest)

        N = means.shape[0]
        device = means.device
        self.register_buffer("_gradient_accum", torch.zeros(N, device=device))
        self.register_buffer("_gradient_denom", torch.zeros(N, device=device))

    # ---- 激活属性 ----

    @property
    def means(self) -> Tensor:
        """(N, 3) 高斯位置（无激活，直接返回）。"""
        return self._means

    @property
    def scales(self) -> Tensor:
        """(N, 3) 正值尺度（exp 激活）。"""
        return torch.exp(self._scales)

    @property
    def rotations(self) -> Tensor:
        """(N, 4) 单位四元数（L2 归一化）。"""
        return nn.functional.normalize(self._rotations, dim=1)

    @property
    def opacities(self) -> Tensor:
        """(N, 1) 不透明度，值域 (0, 1)（sigmoid 激活）。"""
        return torch.sigmoid(self._opacities)

    @property
    def sh_coefficients(self) -> Tensor:
        """(N, (max_sh_degree+1)^2, 3) 完整 SH 系数（dc + rest 拼接）。"""
        return torch.cat([self._sh_dc, self._sh_rest], dim=1)

    @property
    def num_gaussians(self) -> int:
        return self._means.shape[0]

    @property
    def active_sh_degree(self) -> int:
        return self._active_sh_degree

    def step_sh_degree(self) -> None:
        """将激活的 SH 阶数 +1（最大到 max_sh_degree）。"""
        self._active_sh_degree = min(self._active_sh_degree + 1, self.max_sh_degree)

    # ---- 协方差 ----

    def compute_covariance_3d(self) -> Tensor:
        """计算 3D 协方差矩阵的上三角部分。

        公式：Σ = R * diag(s)^2 * R^T

        Returns:
            (N, 6) float32，顺序 [σ_xx, σ_xy, σ_xz, σ_yy, σ_yz, σ_zz]，
            与 diff-gaussian-rasterization 的 cov3D_precomp 接口匹配。
        """
        S = self.scales        # (N, 3)
        R = _qvec_to_rotmat(self.rotations)   # (N, 3, 3)

        # Σ = R S^2 R^T
        S2 = torch.diag_embed(S * S)          # (N, 3, 3)
        cov = R @ S2 @ R.transpose(1, 2)      # (N, 3, 3)

        # 取上三角
        return torch.stack([
            cov[:, 0, 0], cov[:, 0, 1], cov[:, 0, 2],
            cov[:, 1, 1], cov[:, 1, 2],
            cov[:, 2, 2],
        ], dim=1)  # (N, 6)

    # ---- 初始化 ----

    @classmethod
    def from_point_cloud(
        cls,
        xyz: np.ndarray,           # (M, 3) float64
        rgb: np.ndarray,           # (M, 3) uint8 或 float32 [0,1]
        max_sh_degree: int = 3,
        device: str = "cuda",
    ) -> "GaussianModel":
        """从 COLMAP 稀疏点云初始化 GaussianModel。

        - 位置 = 点云坐标
        - 尺度 = log(mean_knn_distance)（k=3 近邻均值距离）
        - 旋转 = 单位四元数（恒等旋转）
        - 不透明度 = inverse_sigmoid(0.1)
        - SH DC = C0 * (rgb / 255 - 0.5)，高阶 SH = 0

        Args:
            xyz: 点云坐标，(M, 3)。
            rgb: 颜色，uint8 [0,255] 或 float32 [0,1]。
            max_sh_degree: 最大 SH 阶数（0~3）。
            device: PyTorch 设备。
        """
        model = cls(max_sh_degree=max_sh_degree)

        xyz_t = torch.tensor(xyz, dtype=torch.float32, device=device)
        N = xyz_t.shape[0]

        # 尺度：用 k=3 近邻均值距离初始化（log 空间）
        dists = _mean_knn_distance(xyz_t, k=3)            # (N,)
        log_scale = torch.log(torch.sqrt(dists + 1e-8))   # (N,)
        scales = log_scale.unsqueeze(1).expand(-1, 3)      # (N, 3)

        # 旋转：恒等四元数 [1, 0, 0, 0]（wxyz）
        rotations = torch.zeros(N, 4, device=device)
        rotations[:, 0] = 1.0

        # 不透明度
        inv_sig_01 = math.log(0.1 / 0.9)   # inverse_sigmoid(0.1)
        opacities = torch.full((N, 1), inv_sig_01, dtype=torch.float32, device=device)

        # SH DC：从 RGB 颜色计算
        if rgb.dtype == np.uint8:
            rgb_f = rgb.astype(np.float32) / 255.0
        else:
            rgb_f = rgb.astype(np.float32)
        rgb_t = torch.tensor(rgb_f, dtype=torch.float32, device=device)  # (N, 3)
        sh_dc = (rgb_t - 0.5) / SH_C0   # (N, 3)
        sh_dc = sh_dc.unsqueeze(1)       # (N, 1, 3)

        # 高阶 SH：全零
        K = num_sh_coefficients(max_sh_degree) - 1
        sh_rest = torch.zeros(N, K, 3, device=device)

        model._reset_parameters(
            means=xyz_t,
            scales=scales,
            rotations=rotations,
            opacities=opacities,
            sh_dc=sh_dc,
            sh_rest=sh_rest,
        )
        return model

    # ---- 存读 PLY ----

    def save_ply(self, path: str) -> None:
        """将所有参数保存为 3DGS 格式的 PLY 文件。"""
        with torch.no_grad():
            means_np     = self._means.cpu().numpy()
            scales_np    = self._scales.cpu().numpy()
            rotations_np = self._rotations.cpu().numpy()
            opacities_np = self._opacities.cpu().numpy()
            sh_dc_np     = self._sh_dc.cpu().numpy()
            sh_rest_np   = self._sh_rest.cpu().numpy()

        ply_dict = gaussians_to_ply_dict(
            means_np, scales_np, rotations_np, opacities_np, sh_dc_np, sh_rest_np
        )
        write_ply(path, ply_dict, binary=True)

    @classmethod
    def load_ply(
        cls,
        path: str,
        device: str = "cuda",
    ) -> "GaussianModel":
        """从 PLY checkpoint 恢复 GaussianModel。"""
        data = read_ply(path)
        means, scales, rotations, opacities, sh_dc, sh_rest = ply_dict_to_gaussians(data)

        # 从 SH 系数推断 max_sh_degree
        K_rest = sh_rest.shape[1]   # K = (degree+1)^2 - 1
        max_sh_degree = 0
        for deg in range(1, 4):
            if num_sh_coefficients(deg) - 1 >= K_rest:
                max_sh_degree = deg - 1
                break
        else:
            max_sh_degree = 3

        model = cls(max_sh_degree=max_sh_degree)
        model._reset_parameters(
            means=torch.tensor(means,     dtype=torch.float32, device=device),
            scales=torch.tensor(scales,   dtype=torch.float32, device=device),
            rotations=torch.tensor(rotations, dtype=torch.float32, device=device),
            opacities=torch.tensor(opacities, dtype=torch.float32, device=device),
            sh_dc=torch.tensor(sh_dc,     dtype=torch.float32, device=device),
            sh_rest=torch.tensor(sh_rest, dtype=torch.float32, device=device),
        )
        return model

    # ---- 梯度统计 ----

    def update_gradient_stats(self, screenspace_grads: Tensor, visibility: Tensor) -> None:
        """累积 2D 屏幕空间梯度统计，供 DensificationController 使用。

        Args:
            screenspace_grads: (N, 2) 屏幕空间均值梯度。
            visibility: (N,) bool，True 表示该高斯在本帧可见。
        """
        norms = screenspace_grads.norm(dim=1)   # (N,)
        self._gradient_accum[visibility] += norms[visibility]
        self._gradient_denom[visibility] += 1

    def reset_gradient_stats(self) -> None:
        self._gradient_accum.zero_()
        self._gradient_denom.zero_()

    @property
    def gradient_accum(self) -> Tensor:
        return self._gradient_accum

    @property
    def gradient_denom(self) -> Tensor:
        return self._gradient_denom

    # ---- 变异方法（由 DensificationController 调用）----

    def densify_and_clone(self, mask: Tensor) -> None:
        """复制梯度大且尺度小的高斯（in-place）。

        Args:
            mask: (N,) bool，True 表示需要克隆的高斯。
        """
        new_means     = self._means[mask].detach()
        new_scales    = self._scales[mask].detach()
        new_rots      = self._rotations[mask].detach()
        new_opacities = self._opacities[mask].detach()
        new_sh_dc     = self._sh_dc[mask].detach()
        new_sh_rest   = self._sh_rest[mask].detach()

        self._concat_parameters(new_means, new_scales, new_rots,
                                new_opacities, new_sh_dc, new_sh_rest)

    def densify_and_split(
        self,
        mask: Tensor,
        num_splits: int = 2,
    ) -> None:
        """分裂梯度大且尺度大的高斯（in-place）。

        新高斯的位置从原高斯的 3D 分布中采样，尺度缩小 1.6 倍。

        Args:
            mask: (N,) bool，True 表示需要分裂的高斯。
            num_splits: 每个高斯分裂为多少个新高斯（默认 2）。
        """
        selected = mask.nonzero(as_tuple=True)[0]
        if len(selected) == 0:
            return

        M = len(selected)
        means_sel     = self._means[selected]      # (M, 3)
        scales_sel    = self._scales[selected]
        rots_sel      = self._rotations[selected]
        opacities_sel = self._opacities[selected]
        sh_dc_sel     = self._sh_dc[selected]
        sh_rest_sel   = self._sh_rest[selected]

        # 从原高斯分布中采样新位置
        stds = torch.exp(scales_sel).unsqueeze(1).expand(-1, num_splits, -1)  # (M, S, 3)
        rots_mat = _qvec_to_rotmat(nn.functional.normalize(rots_sel, dim=1))  # (M, 3, 3)

        # 采样 num_splits 份偏移
        samples = torch.randn(M, num_splits, 3, device=self._means.device)
        # 旋转到世界坐标
        offsets = (rots_mat.unsqueeze(1) @ (stds * samples).unsqueeze(-1)).squeeze(-1)  # (M, S, 3)
        new_means = (means_sel.unsqueeze(1) + offsets).reshape(-1, 3)  # (M*S, 3)

        # 尺度缩小
        new_scales    = (scales_sel - math.log(1.6)).unsqueeze(1).expand(-1, num_splits, -1).reshape(-1, 3)
        new_rots      = rots_sel.unsqueeze(1).expand(-1, num_splits, -1).reshape(-1, 4)
        new_opacities = opacities_sel.unsqueeze(1).expand(-1, num_splits, -1).reshape(-1, 1)
        new_sh_dc     = sh_dc_sel.unsqueeze(1).expand(-1, num_splits, -1, -1).reshape(-1, *sh_dc_sel.shape[1:])
        new_sh_rest   = sh_rest_sel.unsqueeze(1).expand(-1, num_splits, -1, -1).reshape(-1, *sh_rest_sel.shape[1:])

        # 移除原来的高斯，添加新高斯
        keep_mask = ~mask
        self._prune_with_mask(keep_mask)
        self._concat_parameters(
            new_means.detach(), new_scales.detach(), new_rots.detach(),
            new_opacities.detach(), new_sh_dc.detach(), new_sh_rest.detach()
        )

    def prune(self, mask: Tensor) -> None:
        """移除 mask 为 True 的高斯（in-place）。

        Args:
            mask: (N,) bool，True 表示需要移除的高斯。
        """
        self._prune_with_mask(~mask)

    def reset_opacities(self, value: float = 0.01) -> None:
        """将所有高斯的不透明度重置为指定值（pre-sigmoid）。"""
        inv_sig = math.log(value / (1 - value))
        with torch.no_grad():
            self._opacities.fill_(inv_sig)

    # ---- 内部工具 ----

    def _prune_with_mask(self, keep_mask: Tensor) -> None:
        """保留 keep_mask 为 True 的高斯，更新所有参数（in-place 替换）。"""
        def _masked(param: nn.Parameter) -> nn.Parameter:
            return nn.Parameter(param.data[keep_mask])

        self._means     = _masked(self._means)
        self._scales    = _masked(self._scales)
        self._rotations = _masked(self._rotations)
        self._opacities = _masked(self._opacities)
        self._sh_dc     = _masked(self._sh_dc)
        self._sh_rest   = _masked(self._sh_rest)

        self._gradient_accum = self._gradient_accum[keep_mask]
        self._gradient_denom = self._gradient_denom[keep_mask]

    def _concat_parameters(
        self,
        new_means: Tensor,
        new_scales: Tensor,
        new_rots: Tensor,
        new_opacities: Tensor,
        new_sh_dc: Tensor,
        new_sh_rest: Tensor,
    ) -> None:
        """将新高斯追加到现有参数末尾（in-place 替换）。"""
        N_new = new_means.shape[0]
        device = self._means.device

        def _cat(param: nn.Parameter, new_data: Tensor) -> nn.Parameter:
            return nn.Parameter(torch.cat([param.data, new_data], dim=0))

        self._means     = _cat(self._means,     new_means)
        self._scales    = _cat(self._scales,    new_scales)
        self._rotations = _cat(self._rotations, new_rots)
        self._opacities = _cat(self._opacities, new_opacities)
        self._sh_dc     = _cat(self._sh_dc,     new_sh_dc)
        self._sh_rest   = _cat(self._sh_rest,   new_sh_rest)

        self._gradient_accum = torch.cat([self._gradient_accum, torch.zeros(N_new, device=device)])
        self._gradient_denom = torch.cat([self._gradient_denom, torch.zeros(N_new, device=device)])


# ---------------------------------------------------------------------------
# 内部辅助
# ---------------------------------------------------------------------------

def _qvec_to_rotmat(qvec: Tensor) -> Tensor:
    """批量四元数 wxyz → 旋转矩阵。

    Args:
        qvec: (N, 4) 单位四元数。

    Returns:
        (N, 3, 3) 旋转矩阵。
    """
    w = qvec[:, 0:1]
    x = qvec[:, 1:2]
    y = qvec[:, 2:3]
    z = qvec[:, 3:4]
    R = torch.stack([
        1 - 2*(y*y + z*z),   2*(x*y - w*z),       2*(x*z + w*y),
        2*(x*y + w*z),        1 - 2*(x*x + z*z),   2*(y*z - w*x),
        2*(x*z - w*y),        2*(y*z + w*x),        1 - 2*(x*x + y*y),
    ], dim=1).reshape(-1, 3, 3)
    return R


def _mean_knn_distance(xyz: Tensor, k: int = 3) -> Tensor:
    """计算每个点到 k 近邻的均值平方距离。

    Args:
        xyz: (N, 3) 点云。
        k: 近邻数量。

    Returns:
        (N,) 均值平方距离。
    """
    N = xyz.shape[0]
    if N <= k:
        return torch.ones(N, device=xyz.device) * 0.01

    # 分批计算避免显存不足
    batch = min(4096, N)
    dists = []
    for i in range(0, N, batch):
        chunk = xyz[i:i+batch]                                    # (B, 3)
        diff = chunk.unsqueeze(1) - xyz.unsqueeze(0)             # (B, N, 3)
        sq = (diff ** 2).sum(dim=2)                              # (B, N)
        # 排除自身（设为大值）
        sq[:, i:i+batch] = sq[:, i:i+batch] + torch.eye(
            min(batch, N-i), N, device=xyz.device
        )[:, i:i+batch] * 1e9
        topk = torch.topk(sq, k, dim=1, largest=False).values   # (B, k)
        dists.append(topk.mean(dim=1))
    return torch.cat(dists)

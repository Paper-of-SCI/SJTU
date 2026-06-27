from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from methods.leo_3DGS.functions.metrics import ssim


DEFAULT_LPIPS_WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "assets" / "lpips"


@dataclass(frozen=True)
class Raw3DGSLossResult:
    loss: torch.Tensor
    l1: torch.Tensor
    ssim: torch.Tensor
    dssim: torch.Tensor
    lpips: torch.Tensor


def _to_bchw(image: torch.Tensor) -> torch.Tensor:
    if image.ndim == 3:
        if image.shape[0] == 3:
            return image.unsqueeze(0).contiguous()
        if image.shape[-1] == 3:
            return image.permute(2, 0, 1).unsqueeze(0).contiguous()
    elif image.ndim == 4:
        if image.shape[1] == 3:
            return image.contiguous()
        if image.shape[-1] == 3:
            return image.permute(0, 3, 1, 2).contiguous()

    raise ValueError(f"image must be [3,H,W], [H,W,3], [B,3,H,W], or [B,H,W,3], got {tuple(image.shape)}")


def _default_lpips_weights_path(net_type: str, version: str) -> Path:
    return DEFAULT_LPIPS_WEIGHTS_DIR / f"v{version}" / f"{net_type}.pth"


def _load_lpips_linear_state_dict(weights_path: Path) -> OrderedDict[str, torch.Tensor]:
    if not weights_path.exists():
        raise FileNotFoundError(
            f"LPIPS weights not found: {weights_path}. "
            "Put the weight file there or pass LPIPSLoss(weights_path=...)."
        )

    old_state_dict = torch.load(weights_path, map_location="cpu")
    new_state_dict: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key, value in old_state_dict.items():
        new_key = key.replace("lin", "").replace("model.", "")
        new_state_dict[new_key] = value
    return new_state_dict


class LPIPSLoss(nn.Module):
    """
    可反向传播的 LPIPS loss。

    这个类和 metrics.LPIPSEvaluator 的区别是：这里没有 torch.no_grad()，
    所以梯度可以从 LPIPS loss 传回 rendered。
    """

    def __init__(
        self,
        net_type: str = "vgg",
        version: str = "0.1",
        weights_path: str | Path | None = None,
    ):
        super().__init__()

        if version != "0.1":
            raise ValueError("only LPIPS version 0.1 is supported")

        from methods.seasplat_Render.lpipsPyTorch.modules.networks import LinLayers, get_network

        self.net_type = net_type
        self.version = version
        self.weights_path = Path(weights_path) if weights_path is not None else _default_lpips_weights_path(net_type, version)

        self.net = get_network(net_type)
        self.lin = LinLayers(self.net.n_channels_list)
        self.lin.load_state_dict(_load_lpips_linear_state_dict(self.weights_path))

        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, rendered: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        rendered = _to_bchw(rendered).clamp(0.0, 1.0)
        target = _to_bchw(target).to(device=rendered.device, dtype=rendered.dtype).clamp(0.0, 1.0)

        if rendered.shape != target.shape:
            raise ValueError(f"rendered shape {tuple(rendered.shape)} != target shape {tuple(target.shape)}")

        rendered_lpips = rendered * 2.0 - 1.0
        target_lpips = target * 2.0 - 1.0

        feat_rendered = self.net(rendered_lpips)
        feat_target = self.net(target_lpips)
        diffs = [(x - y).square() for x, y in zip(feat_rendered, feat_target)]
        values = [layer(diff).mean((2, 3), keepdim=True) for diff, layer in zip(diffs, self.lin)]
        return torch.stack(values, dim=0).sum(dim=0).reshape(rendered.shape[0], -1).mean()


def adjust_raw_3dgs_loss(
    rendered: torch.Tensor,
    target: torch.Tensor,
    lambda_dssim: float = 0.2,
    lambda_lpips: float = 0.0,
    lpips_loss_model: LPIPSLoss | None = None,
) -> Raw3DGSLossResult:
    r"""
    带可选 LPIPS 的 3DGS 图像重建 loss。

    默认仍然是原版 3DGS 主项：

    $$
    L = (1 - \lambda_{dssim}) L_1 + \lambda_{dssim}(1 - SSIM)
    $$

    如果设置 `lambda_lpips > 0`，额外加入：

    $$
    L =
    (1 - \lambda_{dssim}) L_1
    + \lambda_{dssim}(1 - SSIM)
    + \lambda_{lpips} L_{LPIPS}
    $$

    Args:
        rendered: 渲染结果，shape 通常为 [3, H, W] 或 [B, 3, H, W]，范围 [0, 1]。
        target: GT 图像，shape 与 rendered 一致，范围 [0, 1]。
        lambda_dssim: DSSIM 权重，原版默认常用 0.2。
        lambda_lpips: LPIPS 权重；0 表示关闭 LPIPS。
        lpips_loss_model: 可反传 LPIPS 模型，建议训练开始前创建一次并复用。

    Returns:
        Raw3DGSLossResult，包含总 loss 和 l1 / ssim / dssim / lpips 分项。
    """
    if not 0.0 <= lambda_dssim <= 1.0:
        raise ValueError(f"lambda_dssim must be in [0, 1], got {lambda_dssim}")

    if lambda_lpips < 0.0:
        raise ValueError(f"lambda_lpips must be >= 0, got {lambda_lpips}")

    if rendered.shape != target.shape:
        raise ValueError(f"rendered shape {tuple(rendered.shape)} != target shape {tuple(target.shape)}")

    l1_value = F.l1_loss(rendered, target)
    ssim_value = ssim(rendered, target, clamp=True, size_average=True)
    dssim_value = 1.0 - ssim_value

    lpips_value = rendered.new_tensor(0.0)
    if lambda_lpips > 0.0:
        if lpips_loss_model is None:
            raise ValueError("lpips_loss_model is required when lambda_lpips > 0")
        lpips_value = lpips_loss_model(rendered, target)

    loss = (
        (1.0 - lambda_dssim) * l1_value
        + lambda_dssim * dssim_value
        + lambda_lpips * lpips_value
    )

    return Raw3DGSLossResult(
        loss=loss,
        l1=l1_value,
        ssim=ssim_value,
        dssim=dssim_value,
        lpips=lpips_value,
    )

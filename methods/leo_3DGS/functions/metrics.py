from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn


DEFAULT_LPIPS_WEIGHTS_DIR = Path(__file__).resolve().parents[1] / "assets" / "lpips"


def _to_bchw(image: torch.Tensor) -> torch.Tensor:
    if image.ndim == 3:
        if image.shape[0] in {1, 3}:
            image = image.unsqueeze(0)
        elif image.shape[-1] in {1, 3}:
            image = image.permute(2, 0, 1).unsqueeze(0)
        else:
            raise ValueError(f"3D image must be [C,H,W] or [H,W,C], got {tuple(image.shape)}")
    elif image.ndim == 4:
        if image.shape[1] in {1, 3}:
            pass
        elif image.shape[-1] in {1, 3}:
            image = image.permute(0, 3, 1, 2)
        else:
            raise ValueError(f"4D image must be [B,C,H,W] or [B,H,W,C], got {tuple(image.shape)}")
    else:
        raise ValueError(f"image must have 3 or 4 dimensions, got {image.ndim}")

    return image.contiguous()


def _tensor_to_uint8_hwc(image: torch.Tensor) -> np.ndarray:
    image = _to_bchw(image)[0].detach().clamp(0.0, 1.0)
    image = image.permute(1, 2, 0).cpu().numpy()
    return (image * 255.0).round().astype(np.uint8)


def _prepare_pair(
    rendered: torch.Tensor,
    target: torch.Tensor,
    clamp: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    rendered = _to_bchw(rendered).float()
    target = _to_bchw(target).float().to(device=rendered.device)

    if rendered.shape != target.shape:
        raise ValueError(f"rendered shape {tuple(rendered.shape)} != target shape {tuple(target.shape)}")

    if clamp:
        rendered = rendered.clamp(0.0, 1.0)
        target = target.clamp(0.0, 1.0)

    return rendered, target


def psnr(
    rendered: torch.Tensor,
    target: torch.Tensor,
    clamp: bool = True,
    eps: float = 1e-10,
) -> torch.Tensor:
    """
    计算 PSNR，输入默认是 [0, 1] 图像。

    返回 shape 为 [B] 的 tensor；如果输入是单张图，则返回 1 个值。
    """
    rendered, target = _prepare_pair(rendered, target, clamp=clamp)
    mse = (rendered - target).square().flatten(start_dim=1).mean(dim=1).clamp_min(eps)
    return 20.0 * torch.log10(1.0 / torch.sqrt(mse))


def _gaussian_window(
    window_size: int,
    sigma: float,
    channels: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    kernel_1d = torch.exp(-(coords.square()) / (2.0 * sigma * sigma))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d[:, None] @ kernel_1d[None, :]
    return kernel_2d[None, None].expand(channels, 1, window_size, window_size).contiguous()


def ssim(
    rendered: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    clamp: bool = True,
    size_average: bool = True,
) -> torch.Tensor:
    """
    计算 SSIM，输入默认是 [0, 1] 图像。

    `size_average=True` 时返回标量；否则返回 shape 为 [B] 的 tensor。
    """
    rendered, target = _prepare_pair(rendered, target, clamp=clamp)
    channels = rendered.shape[1]
    window = _gaussian_window(
        window_size=window_size,
        sigma=sigma,
        channels=channels,
        device=rendered.device,
        dtype=rendered.dtype,
    )

    padding = window_size // 2
    mu1 = F.conv2d(rendered, window, padding=padding, groups=channels)
    mu2 = F.conv2d(target, window, padding=padding, groups=channels)

    mu1_sq = mu1.square()
    mu2_sq = mu2.square()
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(rendered * rendered, window, padding=padding, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=padding, groups=channels) - mu2_sq
    sigma12 = F.conv2d(rendered * target, window, padding=padding, groups=channels) - mu1_mu2

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    ssim_map = ((2.0 * mu1_mu2 + c1) * (2.0 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )

    if size_average:
        return ssim_map.mean()

    return ssim_map.flatten(start_dim=1).mean(dim=1)


def _default_lpips_weights_path(net_type: str, version: str) -> Path:
    return DEFAULT_LPIPS_WEIGHTS_DIR / f"v{version}" / f"{net_type}.pth"


def _load_lpips_linear_state_dict(weights_path: Path) -> OrderedDict[str, torch.Tensor]:
    if not weights_path.exists():
        raise FileNotFoundError(
            f"LPIPS weights not found: {weights_path}. "
            "Put the weight file there or pass LPIPSEvaluator(weights_path=...)."
        )

    old_state_dict = torch.load(weights_path, map_location="cpu")
    new_state_dict: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key, value in old_state_dict.items():
        new_key = key.replace("lin", "").replace("model.", "")
        new_state_dict[new_key] = value
    return new_state_dict


class _LocalLPIPS(nn.Module):
    def __init__(self, net_type: str, version: str, weights_path: Path):
        super().__init__()

        if version != "0.1":
            raise ValueError("only LPIPS version 0.1 is supported")

        from methods.seasplat_Render.lpipsPyTorch.modules.networks import LinLayers, get_network

        self.net = get_network(net_type)
        self.lin = LinLayers(self.net.n_channels_list)
        self.lin.load_state_dict(_load_lpips_linear_state_dict(weights_path))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        feat_x = self.net(x)
        feat_y = self.net(y)
        diff = [(fx - fy).square() for fx, fy in zip(feat_x, feat_y)]
        values = [layer(delta).mean((2, 3), keepdim=True) for delta, layer in zip(diff, self.lin)]
        return torch.stack(values, dim=0).sum(dim=0)


class LPIPSEvaluator:
    """
    复用 LPIPS 网络，避免每次评估都重新加载模型。

    输入图像默认是 [0, 1]，内部会转成 LPIPS 常用的 [-1, 1]。
    默认从 methods/leo_3DGS/assets/lpips/v0.1/{net_type}.pth 加载 LPIPS 线性层权重。
    """

    def __init__(
        self,
        net_type: str = "vgg",
        version: str = "0.1",
        device: torch.device | str | None = None,
        weights_path: str | Path | None = None,
    ):
        self.net_type = net_type
        self.version = version
        self.device = torch.device(device) if device is not None else None
        self.weights_path = Path(weights_path) if weights_path is not None else _default_lpips_weights_path(net_type, version)
        self.model: nn.Module | None = None

    def _load_model(self, device: torch.device) -> nn.Module:
        if self.model is not None:
            return self.model

        model = _LocalLPIPS(
            net_type=self.net_type,
            version=self.version,
            weights_path=self.weights_path,
        )

        model = model.to(device).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)

        self.model = model
        return model

    @torch.no_grad()
    def __call__(
        self,
        rendered: torch.Tensor,
        target: torch.Tensor,
        clamp: bool = True,
    ) -> torch.Tensor:
        rendered, target = _prepare_pair(rendered, target, clamp=clamp)
        device = self.device or rendered.device
        rendered = rendered.to(device)
        target = target.to(device)

        model = self._load_model(device)
        rendered_lpips = rendered * 2.0 - 1.0
        target_lpips = target * 2.0 - 1.0
        value = model(rendered_lpips, target_lpips)
        return value.reshape(value.shape[0], -1).mean(dim=1)


@dataclass(frozen=True)
class ImageMetricResult:
    psnr: float
    ssim: float
    lpips: float | None = None


@torch.no_grad()
def evaluate_image_metrics(
    rendered: torch.Tensor,
    target: torch.Tensor,
    lpips_evaluator: LPIPSEvaluator | None = None,
    clamp: bool = True,
) -> ImageMetricResult:
    rendered, target = _prepare_pair(rendered, target, clamp=clamp)

    psnr_value = psnr(rendered, target, clamp=False).mean().item()
    ssim_value = ssim(rendered, target, clamp=False, size_average=True).item()

    lpips_value = None
    if lpips_evaluator is not None:
        lpips_value = lpips_evaluator(rendered, target, clamp=False).mean().item()

    return ImageMetricResult(
        psnr=psnr_value,
        ssim=ssim_value,
        lpips=lpips_value,
    )


@torch.no_grad()
def save_gaussian_projection_debug(
    model,
    image,
    camera,
    background: torch.Tensor,
    output_path: str | Path,
    point_color: tuple[int, int, int] = (255, 0, 0),
    point_radius: int = 1,
    stride: int = 1,
) -> None:
    """
    保存 Gaussian 中心投影到图像平面的调试图。

    Args:
        model: GaussianModel。
        image: COLMAP image 参数。
        camera: 当前相机参数，必须和 background 分辨率一致。
        background: [3,H,W] 或 [H,W,3]，通常传 GT 图。
        output_path: 输出 png 路径。
        point_color: 投影点颜色，默认红色。
        point_radius: 每个点画出的半径，1 表示 3x3。
        stride: 每隔多少个有效点画一个；点太密时可以设成 5、10、20。
    """
    from methods.leo_3DGS.functions.render import project_gaussian_means

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    canvas = _tensor_to_uint8_hwc(background)
    height, width = canvas.shape[:2]

    if width != camera.width or height != camera.height:
        raise ValueError(
            f"background size {(width, height)} does not match camera size {(camera.width, camera.height)}"
        )

    pixels, _depths, valid, _xyz_cam = project_gaussian_means(model, image, camera)
    points_2d = pixels[valid][::stride].detach().cpu().numpy()

    radius = max(0, int(point_radius))
    color = np.asarray(point_color, dtype=np.uint8)

    for u, v in points_2d:
        u = int(round(float(u)))
        v = int(round(float(v)))

        if 0 <= u < width and 0 <= v < height:
            x0 = max(u - radius, 0)
            x1 = min(u + radius + 1, width)
            y0 = max(v - radius, 0)
            y1 = min(v + radius + 1, height)
            canvas[y0:y1, x0:x1] = color

    Image.fromarray(canvas).save(output_path)

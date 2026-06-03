"""Evaluation metrics and best-metric tracking for 3DGS runs."""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.camera import Camera
from modules.gaussian_model import GaussianModel
from modules.losses import ssim
from modules.renderer import GaussianRenderer
from utils.image_utils import compute_psnr

MetricDict = dict[str, Optional[float]]
LPIPSEvaluatorFn = Callable[[torch.Tensor, torch.Tensor], float]


def resize_to_gt_if_needed(image: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Match SWAGS/official metric behavior when render and GT sizes differ."""
    if image.shape[-2:] == gt.shape[-2:]:
        return image
    return F.interpolate(image.unsqueeze(0), size=gt.shape[-2:], mode="nearest").squeeze(0)


@torch.no_grad()
def compute_image_metrics(image: torch.Tensor, gt: torch.Tensor, lpips_evaluator: LPIPSEvaluatorFn | None = None) -> MetricDict:
    """Return PSNR, SSIM, L1 and optional LPIPS for CHW tensors in [0, 1]."""
    image = resize_to_gt_if_needed(image.detach().clamp(0.0, 1.0), gt.detach())
    gt = gt.detach().clamp(0.0, 1.0)
    return {
        "psnr": compute_psnr(image.permute(1, 2, 0).cpu().numpy(), gt.permute(1, 2, 0).cpu().numpy()),
        "ssim": float(ssim(image, gt).detach()),
        "l1": float(torch.mean(torch.abs(image - gt)).detach()),
        "lpips": lpips_evaluator(image, gt) if lpips_evaluator is not None else None,
    }


@torch.no_grad()
def evaluate_cameras(
    model: GaussianModel,
    renderer: GaussianRenderer,
    cameras: list[Camera],
    lpips_evaluator: LPIPSEvaluatorFn | None = None,
) -> MetricDict:
    """Render cameras and return mean held-out metrics."""
    rows: list[MetricDict] = []
    for camera in cameras:
        if camera.image is None:
            raise RuntimeError(f"相机缺少 GT 图像，无法计算评估指标: {camera.image_path}")
        render = renderer.render(model, camera)
        rows.append(compute_image_metrics(render.image, camera.image, lpips_evaluator))
    return mean_metrics(rows)


def mean_metrics(rows: list[MetricDict]) -> MetricDict:
    means: MetricDict = {}
    for name in ["psnr", "ssim", "l1", "lpips"]:
        values = [row[name] for row in rows if row.get(name) is not None and math.isfinite(float(row[name]))]
        means[name] = float(sum(values) / len(values)) if values else None
    return means


@dataclass
class BestMetricTracker:
    """Track best PSNR/SSIM maxima and optional LPIPS minimum."""

    include_lpips: bool = False
    best: dict[str, dict] = field(init=False)
    history: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        names = ["psnr", "ssim"]
        if self.include_lpips:
            names.append("lpips")
        self.best = {name: _empty_best_record() for name in names}

    def update(self, step: int, metrics: MetricDict) -> list[str]:
        snapshot = _serializable_metrics(metrics)
        self.history.append({"step": int(step), "metrics": snapshot})
        improved: list[str] = []
        for name in self.best:
            value = snapshot.get(name)
            if value is None:
                continue
            current = self.best[name].get("value")
            if current is None or _is_better_metric(name, value, float(current)):
                self.best[name] = {
                    "value": float(value),
                    "step": int(step),
                    "checkpoint": "",
                    "metrics": dict(snapshot),
                }
                improved.append(name)
        return improved

    def set_checkpoint(self, metric_name: str, checkpoint: str) -> None:
        if metric_name not in self.best:
            raise ValueError(f"未知 best metric: {metric_name}")
        self.best[metric_name]["checkpoint"] = str(checkpoint)

    def to_dict(self) -> dict:
        return {"best": self.best, "history": self.history}


def flatten_best_metric_fields(best_metrics: dict | None) -> dict:
    best = best_metrics.get("best", {}) if isinstance(best_metrics, dict) else {}
    fields = {}
    for name in ["psnr", "ssim", "lpips"]:
        record = best.get(name, {}) if isinstance(best, dict) else {}
        fields[f"best_{name}"] = record.get("value")
        fields[f"best_{name}_step"] = record.get("step")
        fields[f"best_{name}_checkpoint"] = record.get("checkpoint", "")
    return fields


def build_lpips_evaluator(device: torch.device, net: str, backend: str) -> "LPIPSEvaluator | Official3DGSLPIPSEvaluator":
    if backend == "official_3dgs":
        return Official3DGSLPIPSEvaluator(device, net)
    return LPIPSEvaluator(device, net)


class LPIPSEvaluator:
    """Optional LPIPS metric wrapper."""

    def __init__(self, device: torch.device, net: str) -> None:
        try:
            import lpips
        except ImportError as exc:
            raise ImportError("计算 LPIPS 需要安装 lpips：pip install lpips") from exc
        self.model = lpips.LPIPS(net=net).to(device).eval()

    @torch.no_grad()
    def __call__(self, image: torch.Tensor, gt: torch.Tensor) -> float:
        image_bchw = image.unsqueeze(0) * 2.0 - 1.0
        gt_bchw = gt.unsqueeze(0) * 2.0 - 1.0
        return float(self.model(image_bchw, gt_bchw).detach().reshape(-1)[0])


class Official3DGSLPIPSEvaluator:
    """LPIPS implementation compatible with graphdeco gaussian-splatting metrics.py."""

    def __init__(self, device: torch.device, net: str) -> None:
        if net not in {"alex", "vgg", "squeeze"}:
            raise ValueError("official_3dgs LPIPS net must be alex, vgg, or squeeze")
        self.model = Official3DGSLPIPS(net).to(device).eval()

    @torch.no_grad()
    def __call__(self, image: torch.Tensor, gt: torch.Tensor) -> float:
        image_bchw = image.unsqueeze(0).clamp(0.0, 1.0)
        gt_bchw = gt.unsqueeze(0).clamp(0.0, 1.0)
        return float(self.model(image_bchw, gt_bchw).detach().reshape(-1)[0])


class Official3DGSLPIPS(nn.Module):
    """Small local port of the LPIPS module vendored by official 3DGS."""

    def __init__(self, net_type: str) -> None:
        super().__init__()
        self.net = _official_lpips_network(net_type)
        self.lin = nn.ModuleList([nn.Sequential(nn.Identity(), nn.Conv2d(channels, 1, 1, 1, 0, bias=False)) for channels in self.net.n_channels_list])
        self.lin.load_state_dict(_official_lpips_state_dict(net_type))
        for parameter in self.parameters():
            parameter.requires_grad = False

    def forward(self, image: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        image_features = self.net(image)
        gt_features = self.net(gt)
        scores = [layer((image_feature - gt_feature).square()).mean((2, 3), True) for layer, image_feature, gt_feature in zip(self.lin, image_features, gt_features)]
        return torch.sum(torch.cat(scores, dim=0), dim=0, keepdim=True)


class _OfficialLPIPSNetwork(nn.Module):
    def __init__(self, layers: nn.Module, target_layers: list[int], n_channels_list: list[int]) -> None:
        super().__init__()
        self.layers = layers
        self.target_layers = target_layers
        self.n_channels_list = n_channels_list
        self.register_buffer("mean", torch.tensor([-.030, -.088, -.188], dtype=torch.float32)[None, :, None, None])
        self.register_buffer("std", torch.tensor([.458, .448, .450], dtype=torch.float32)[None, :, None, None])
        for parameter in self.parameters():
            parameter.requires_grad = False

    def forward(self, image: torch.Tensor) -> list[torch.Tensor]:
        image = (image - self.mean) / self.std
        features = []
        for index, layer in enumerate(self.layers, 1):
            image = layer(image)
            if index in self.target_layers:
                features.append(_normalize_activation(image))
            if len(features) == len(self.target_layers):
                break
        return features


def _empty_best_record() -> dict:
    return {"value": None, "step": None, "checkpoint": "", "metrics": {}}


def _serializable_metrics(metrics: MetricDict) -> MetricDict:
    serializable: MetricDict = {}
    for name in ["psnr", "ssim", "l1", "lpips"]:
        value = metrics.get(name)
        serializable[name] = float(value) if value is not None and math.isfinite(float(value)) else None
    return serializable


def _is_better_metric(name: str, value: float, current: float) -> bool:
    if name == "lpips":
        return value < current
    return value > current


def _official_lpips_network(net_type: str) -> _OfficialLPIPSNetwork:
    try:
        from torchvision import models
    except ImportError as exc:
        raise ImportError("official_3dgs LPIPS 需要 torchvision。") from exc

    if net_type == "alex":
        layers = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1).features
        return _OfficialLPIPSNetwork(layers, [2, 5, 8, 10, 12], [64, 192, 384, 256, 256])
    if net_type == "squeeze":
        layers = models.squeezenet1_1(weights=models.SqueezeNet1_1_Weights.IMAGENET1K_V1).features
        return _OfficialLPIPSNetwork(layers, [2, 5, 8, 10, 11, 12, 13], [64, 128, 256, 384, 384, 512, 512])
    layers = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
    return _OfficialLPIPSNetwork(layers, [4, 9, 16, 23, 30], [64, 128, 256, 512, 512])


def _official_lpips_state_dict(net_type: str) -> OrderedDict:
    url = f"https://raw.githubusercontent.com/richzhang/PerceptualSimilarity/master/lpips/weights/v0.1/{net_type}.pth"
    old_state = torch.hub.load_state_dict_from_url(url, progress=True, map_location=None if torch.cuda.is_available() else torch.device("cpu"))
    state = OrderedDict()
    for key, value in old_state.items():
        new_key = key.replace("lin", "").replace("model.", "")
        state[new_key] = value
    return state


def _normalize_activation(value: torch.Tensor, eps: float = 1.0e-10) -> torch.Tensor:
    return value / (torch.sqrt(torch.sum(value.square(), dim=1, keepdim=True)) + eps)

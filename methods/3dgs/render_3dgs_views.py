"""Render a trained 3DGS PLY checkpoint on dataset camera views."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules import Camera, GaussianModel, GaussianRenderer, ssim
from utils.dataset_loaders import load_colmap_dataset
from utils.image_utils import compute_psnr, save_image
from utils.ply_io import ply_dict_to_gaussians, read_ply


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a 3DGS PLY checkpoint on dataset camera views.")
    parser.add_argument("--data", default="src/datasets/SeathruNeRF_dataset/Curasao", help="COLMAP scene directory.")
    parser.add_argument("--checkpoint", default="outputs/3dgs_scene/final.ply", help="3DGS Gaussian PLY checkpoint.")
    parser.add_argument("--out", default="outputs/3dgs_scene/renders", help="Output render directory.")
    parser.add_argument("--split", default="test", choices=["train", "test", "val"], help="Camera split to render.")
    parser.add_argument("--factor", type=int, default=4, help="Image downscale factor.")
    parser.add_argument("--target-height", type=int, default=0, help="Resize images to this height while preserving aspect ratio; 0 uses --factor.")
    parser.add_argument("--target-width", type=int, default=0, help="Resize images to this width while preserving aspect ratio; 0 uses --target-height or --factor.")
    parser.add_argument("--holdout", type=int, default=8, help="Holdout interval.")
    parser.add_argument("--holdout-offset", type=int, default=0, help="Offset used when selecting every Nth held-out image.")
    parser.add_argument("--max-images", type=int, default=0, help="Limit rendered image count; 0 means all.")
    parser.add_argument("--lpips", action="store_true", help="Also compute LPIPS if the lpips package is installed.")
    parser.add_argument("--lpips-net", default="vgg", choices=["alex", "vgg", "squeeze"], help="LPIPS backbone used when --lpips is enabled.")
    parser.add_argument(
        "--lpips-backend",
        default="lpips",
        choices=["lpips", "official_3dgs"],
        help="LPIPS implementation. official_3dgs matches graphdeco gaussian-splatting metrics.py.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("渲染脚本需要 CUDA。")

    data_dir = resolve_input_path(args.data)
    checkpoint = resolve_input_path(args.checkpoint)
    out_dir = resolve_output_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda")
    scene = load_colmap_dataset(
        str(data_dir),
        split=args.split,
        load_images=False,
        factor=args.factor,
        target_height=args.target_height,
        target_width=args.target_width,
        holdout=args.holdout,
        holdout_offset=args.holdout_offset,
        opengl=False,
    )
    model = load_gaussian_checkpoint(checkpoint, device)
    renderer = GaussianRenderer(background=(1.0, 1.0, 1.0))
    lpips_evaluator = build_lpips_evaluator(device, args.lpips_net, args.lpips_backend) if args.lpips else None

    count = len(scene.image_paths) if args.max_images <= 0 else min(args.max_images, len(scene.image_paths))
    metric_rows = []
    print(f"设备：CUDA GPU='{torch.cuda.get_device_name(device)}'")
    print(f"checkpoint={checkpoint}")
    print(
        f"split={args.split} images={count}/{len(scene.image_paths)} "
        f"resolution={scene.width}x{scene.height} target_height={args.target_height} target_width={args.target_width} factor={args.factor} "
        f"holdout={args.holdout} holdout_offset={args.holdout_offset} "
        f"lpips_backend={args.lpips_backend if args.lpips else ''}"
    )
    print(f"输出目录：{out_dir}")

    for index in range(count):
        camera = Camera.from_scene_data(scene, index, device=device, load_image=True)
        with torch.no_grad():
            render = renderer.render(model, camera)
        image = render.image.detach().clamp(0.0, 1.0)
        gt = camera.image.detach()
        image = resize_to_gt_if_needed(image, gt)
        psnr = compute_psnr(image.permute(1, 2, 0).cpu().numpy(), gt.permute(1, 2, 0).cpu().numpy())
        ssim_value = float(ssim(image, gt).detach())
        l1_value = float(torch.mean(torch.abs(image - gt)).detach())
        lpips_value = lpips_evaluator(image, gt) if lpips_evaluator is not None else None
        metric_rows.append(
            {
                "image": Path(camera.image_path).name,
                "psnr": psnr,
                "ssim": ssim_value,
                "l1": l1_value,
                "lpips": lpips_value,
            }
        )

        stem = Path(camera.image_path).stem
        save_image(str(out_dir / f"{index:03d}_{stem}_render.png"), image.permute(1, 2, 0).cpu().numpy())
        save_image(str(out_dir / f"{index:03d}_{stem}_gt.png"), gt.permute(1, 2, 0).cpu().numpy())
        lpips_text = f" LPIPS={lpips_value:.4f}" if lpips_value is not None else ""
        print(f"[{index + 1}/{count}] {Path(camera.image_path).name} PSNR={psnr:.2f} SSIM={ssim_value:.4f} L1={l1_value:.5f}{lpips_text}")

    if metric_rows:
        save_metrics_csv(out_dir / "metrics.csv", metric_rows, include_lpips=lpips_evaluator is not None)
        avg_psnr = mean_metric(metric_rows, "psnr")
        avg_ssim = mean_metric(metric_rows, "ssim")
        avg_l1 = mean_metric(metric_rows, "l1")
        avg_lpips = mean_metric(metric_rows, "lpips") if lpips_evaluator is not None else None
        lpips_text = f" 平均 LPIPS={avg_lpips:.4f}" if avg_lpips is not None else ""
        print(f"平均 PSNR={avg_psnr:.2f} 平均 SSIM={avg_ssim:.4f} 平均 L1={avg_l1:.5f}{lpips_text}")
        print(f"指标 CSV：{out_dir / 'metrics.csv'}")
    print("渲染完成。")


def resize_to_gt_if_needed(image: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Match SWAGS/official metric behavior when render and GT sizes differ."""
    if image.shape[-2:] == gt.shape[-2:]:
        return image
    return F.interpolate(image.unsqueeze(0), size=gt.shape[-2:], mode="nearest").squeeze(0)


def load_gaussian_checkpoint(path: Path, device: torch.device) -> GaussianModel:
    means, log_scales, quats, logit_opacities, features_dc, features_rest = ply_dict_to_gaussians(read_ply(str(path)))
    sh_bases = 1 + features_rest.shape[1]
    sh_degree = int(round(sh_bases**0.5 - 1))
    model = GaussianModel(sh_degree=sh_degree).to(device)
    model.replace_tensors(
        {
            "means": torch.as_tensor(means, dtype=torch.float32, device=device),
            "log_scales": torch.as_tensor(log_scales, dtype=torch.float32, device=device),
            "quats": torch.as_tensor(quats, dtype=torch.float32, device=device),
            "logit_opacities": torch.as_tensor(logit_opacities, dtype=torch.float32, device=device),
            "features_dc": torch.as_tensor(features_dc, dtype=torch.float32, device=device),
            "features_rest": torch.as_tensor(features_rest, dtype=torch.float32, device=device),
        }
    )
    return model


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


def save_metrics_csv(path: Path, rows: list[dict], include_lpips: bool) -> None:
    fieldnames = ["image", "psnr", "ssim", "l1"]
    if include_lpips:
        fieldnames.append("lpips")
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row[name] for name in fieldnames})


def mean_metric(rows: list[dict], name: str) -> float | None:
    values = [row[name] for row in rows if row[name] is not None]
    if not values:
        return None
    return float(sum(values) / len(values))


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

"""Render a trained 3DGS PLY checkpoint on dataset camera views."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch

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
    parser.add_argument("--holdout", type=int, default=8, help="Holdout interval.")
    parser.add_argument("--max-images", type=int, default=0, help="Limit rendered image count; 0 means all.")
    parser.add_argument("--lpips", action="store_true", help="Also compute LPIPS if the lpips package is installed.")
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
    scene = load_colmap_dataset(str(data_dir), split=args.split, load_images=False, factor=args.factor, holdout=args.holdout, opengl=False)
    model = load_gaussian_checkpoint(checkpoint, device)
    renderer = GaussianRenderer(background=(1.0, 1.0, 1.0))
    lpips_evaluator = LPIPSEvaluator(device) if args.lpips else None

    count = len(scene.image_paths) if args.max_images <= 0 else min(args.max_images, len(scene.image_paths))
    metric_rows = []
    print(f"设备：CUDA GPU='{torch.cuda.get_device_name(device)}'")
    print(f"checkpoint={checkpoint}")
    print(f"split={args.split} images={count}/{len(scene.image_paths)} resolution={scene.width}x{scene.height}")
    print(f"输出目录：{out_dir}")

    for index in range(count):
        camera = Camera.from_scene_data(scene, index, device=device, load_image=True)
        with torch.no_grad():
            render = renderer.render(model, camera)
        image = render.image.detach().clamp(0.0, 1.0)
        gt = camera.image.detach()
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


class LPIPSEvaluator:
    """Optional LPIPS metric wrapper."""

    def __init__(self, device: torch.device) -> None:
        try:
            import lpips
        except ImportError as exc:
            raise ImportError("计算 LPIPS 需要安装 lpips：pip install lpips") from exc
        self.model = lpips.LPIPS(net="vgg").to(device).eval()

    @torch.no_grad()
    def __call__(self, image: torch.Tensor, gt: torch.Tensor) -> float:
        image_bchw = image.unsqueeze(0) * 2.0 - 1.0
        gt_bchw = gt.unsqueeze(0) * 2.0 - 1.0
        return float(self.model(image_bchw, gt_bchw).detach().reshape(-1)[0])


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

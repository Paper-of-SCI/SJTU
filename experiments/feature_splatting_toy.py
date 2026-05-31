"""Toy comparison for RGB-only vs feature-supervised Gaussian splatting.

This is not a full 3DGS benchmark. It verifies the proposed supervision path:
per-Gaussian features -> alpha blending -> patch feature map -> feature loss.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.gs_feature_supervision import alpha_blend_splat_2d, feature_mse_loss, mse_loss, psnr_from_mse


ImageSize = Tuple[int, int]


@dataclass(frozen=True)
class ToyTargets:
    rgb: torch.Tensor
    patch_features: torch.Tensor
    sparse_rgb_mask: torch.Tensor


def _meshgrid(size: ImageSize, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    height, width = size
    ys = torch.linspace(-1.0, 1.0, height, device=device)
    xs = torch.linspace(-1.0, 1.0, width, device=device)
    return torch.meshgrid(ys, xs, indexing="ij")


def build_targets(size: ImageSize, patch_size: int, device: torch.device) -> ToyTargets:
    yy, xx = _meshgrid(size, device)
    background = torch.stack(
        (
            0.05 + 0.05 * (yy + 1.0),
            0.18 + 0.04 * (xx + 1.0),
            0.28 + 0.08 * (1.0 - yy),
        ),
        dim=-1,
    )

    coral = (((xx + 0.28) / 0.43) ** 2 + ((yy + 0.03) / 0.30) ** 2) < 1.0
    branch = (torch.abs(yy + 0.42 * xx - 0.18) < 0.035) & (xx > -0.55) & (xx < 0.55)
    object_mask = (coral | branch).float()

    rgb = background * (1.0 - object_mask[..., None])
    object_color = torch.stack((0.68 + 0.10 * yy, 0.36 + 0.05 * xx, 0.22 + 0.04 * yy), dim=-1)
    rgb = rgb + object_color.clamp(0.0, 1.0) * object_mask[..., None]

    # Simulate low-visibility image supervision: only a deterministic sparse
    # subset of RGB pixels is trusted, while patch features remain available.
    pixel_ids = torch.arange(size[0] * size[1], device=device).reshape(size)
    sparse_mask = ((pixel_ids * 1103515245 + 12345) % 100) < 18
    sparse_mask = sparse_mask.float()

    # Teacher feature map stands in for a ViT patch map in this toy. It uses
    # object occupancy, edge evidence, position and low-frequency appearance.
    object_pool = F.avg_pool2d(object_mask[None, None], kernel_size=patch_size, stride=patch_size)[0, 0]
    rgb_pool = F.avg_pool2d(rgb.permute(2, 0, 1)[None], kernel_size=patch_size, stride=patch_size)[0].permute(1, 2, 0)
    edge_x = F.pad((object_pool[:, 1:] - object_pool[:, :-1]).abs(), (0, 1, 0, 0))
    edge_y = F.pad((object_pool[1:, :] - object_pool[:-1, :]).abs(), (0, 0, 0, 1))
    patch_h, patch_w = object_pool.shape
    py, px = _meshgrid((patch_h, patch_w), device)
    patch_features = torch.cat(
        (
            object_pool[..., None],
            edge_x[..., None],
            edge_y[..., None],
            ((px + 1.0) * 0.5)[..., None],
            ((py + 1.0) * 0.5)[..., None],
            rgb_pool,
        ),
        dim=-1,
    )
    return ToyTargets(rgb=rgb.clamp(0.0, 1.0), patch_features=patch_features, sparse_rgb_mask=sparse_mask)


class ToyGaussianModel(torch.nn.Module):
    def __init__(self, gaussian_count: int, feature_dim: int, image_size: ImageSize, patch_size: int) -> None:
        super().__init__()
        height, width = image_size
        means = torch.rand(gaussian_count, 2)
        means[:, 0] *= width
        means[:, 1] *= height
        self.means_xy = torch.nn.Parameter(means)
        self.log_scales_xy = torch.nn.Parameter(torch.full((gaussian_count, 2), 2.2))
        self.opacity_logits = torch.nn.Parameter(torch.full((gaussian_count,), -0.5))
        self.color_logits = torch.nn.Parameter(torch.randn(gaussian_count, 3) * 0.2)
        self.features = torch.nn.Parameter(torch.randn(gaussian_count, feature_dim) * 0.1)
        self.register_buffer("depths", torch.linspace(0.0, 1.0, gaussian_count))
        self.image_size = image_size
        self.patch_size = patch_size

    def render_rgb(self) -> torch.Tensor:
        output = alpha_blend_splat_2d(
            means_xy=self.means_xy,
            log_scales_xy=self.log_scales_xy,
            opacity_logits=self.opacity_logits,
            values=torch.sigmoid(self.color_logits),
            image_size=self.image_size,
            depths=self.depths,
            background=torch.zeros(3, device=self.means_xy.device),
        )
        return output.value.clamp(0.0, 1.0)

    def render_features(self, patch_image_size: ImageSize) -> torch.Tensor:
        scale = torch.as_tensor(float(self.patch_size), device=self.means_xy.device)
        output = alpha_blend_splat_2d(
            means_xy=self.means_xy / scale,
            log_scales_xy=self.log_scales_xy - torch.log(scale),
            opacity_logits=self.opacity_logits,
            values=self.features,
            image_size=patch_image_size,
            depths=self.depths,
            background=torch.zeros(self.features.shape[-1], device=self.means_xy.device),
        )
        return output.value


def save_rgb(path: Path, image: torch.Tensor) -> None:
    array = (image.detach().cpu().clamp(0.0, 1.0).numpy() * 255.0).astype("uint8")
    Image.fromarray(array).save(path)


def train_variant(
    *,
    name: str,
    targets: ToyTargets,
    image_size: ImageSize,
    patch_size: int,
    gaussian_count: int,
    steps: int,
    feature_weight: float,
    seed: int,
    output_dir: Path,
    device: torch.device,
) -> Dict[str, float]:
    torch.manual_seed(seed)
    model = ToyGaussianModel(
        gaussian_count=gaussian_count,
        feature_dim=targets.patch_features.shape[-1],
        image_size=image_size,
        patch_size=patch_size,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.035)
    patch_image_size = targets.patch_features.shape[:2]

    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        rendered_rgb = model.render_rgb()
        rgb_loss = mse_loss(rendered_rgb, targets.rgb, valid_mask=targets.sparse_rgb_mask)
        loss = rgb_loss
        if feature_weight > 0.0:
            rendered_features = model.render_features(patch_image_size)
            loss = loss + feature_weight * feature_mse_loss(
                rendered_features,
                targets.patch_features,
                normalize_features=False,
            )
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        rendered_rgb = model.render_rgb()
        rendered_features = model.render_features(patch_image_size)
        full_rgb_mse = mse_loss(rendered_rgb, targets.rgb)
        sparse_rgb_mse = mse_loss(rendered_rgb, targets.rgb, valid_mask=targets.sparse_rgb_mask)
        feat_mse = feature_mse_loss(rendered_features, targets.patch_features, normalize_features=False)
        save_rgb(output_dir / f"{name}_render.png", rendered_rgb)

    return {
        "full_rgb_mse": float(full_rgb_mse.detach().cpu()),
        "full_rgb_psnr": float(psnr_from_mse(full_rgb_mse).detach().cpu()),
        "sparse_rgb_mse": float(sparse_rgb_mse.detach().cpu()),
        "feature_mse": float(feat_mse.detach().cpu()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--gaussians", type=int, default=72)
    parser.add_argument("--feature-weight", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/feature_splatting_toy"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    image_size = (64, 64)
    patch_size = 8
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    targets = build_targets(image_size, patch_size, device)
    save_rgb(args.output_dir / "target.png", targets.rgb)

    baseline = train_variant(
        name="rgb_only",
        targets=targets,
        image_size=image_size,
        patch_size=patch_size,
        gaussian_count=args.gaussians,
        steps=args.steps,
        feature_weight=0.0,
        seed=args.seed,
        output_dir=args.output_dir,
        device=device,
    )
    feature_supervised = train_variant(
        name="rgb_plus_feature",
        targets=targets,
        image_size=image_size,
        patch_size=patch_size,
        gaussian_count=args.gaussians,
        steps=args.steps,
        feature_weight=args.feature_weight,
        seed=args.seed,
        output_dir=args.output_dir,
        device=device,
    )

    result = {
        "rgb_only": baseline,
        "rgb_plus_feature": feature_supervised,
        "delta": {
            "full_rgb_psnr": feature_supervised["full_rgb_psnr"] - baseline["full_rgb_psnr"],
            "feature_mse": feature_supervised["feature_mse"] - baseline["feature_mse"],
        },
    }
    result_path = args.output_dir / "metrics.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

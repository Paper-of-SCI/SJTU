"""Render novel camera paths from a trained 3DGS checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules import Camera, GaussianModel, GaussianRenderer, MediumField, MediumRenderConfig, MediumRenderer, medium_checkpoint_path_for_ply
from utils.dataset_loaders import load_colmap_dataset
from utils.image_utils import save_image
from utils.ply_io import ply_dict_to_gaussians, read_ply


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render novel 3DGS camera paths.")
    parser.add_argument("--data", default="src/datasets/SeathruNeRF_dataset/Curasao", help="Scene directory.")
    parser.add_argument("--checkpoint", default="outputs/3dgs_scene/final.ply", help="3DGS Gaussian PLY checkpoint.")
    parser.add_argument("--out", default="outputs/3dgs_scene/path_interpolate", help="Output frame directory.")
    parser.add_argument("--mode", default="interpolate", choices=["interpolate", "orbit"], help="Novel-view path type.")
    parser.add_argument("--split", default="train", choices=["train", "test", "val"], help="Reference camera split.")
    parser.add_argument("--factor", type=int, default=-1, help="Image downscale factor; -1 keeps images at original size unless width exceeds 1600.")
    parser.add_argument("--holdout", type=int, default=8, help="Holdout interval.")
    parser.add_argument("--frames", type=int, default=60, help="Number of frames to render.")
    parser.add_argument("--start", type=int, default=0, help="Start camera index for interpolate mode.")
    parser.add_argument("--end", type=int, default=-1, help="End camera index for interpolate mode; -1 means last camera.")
    parser.add_argument("--radius-scale", type=float, default=1.0, help="Orbit radius multiplier for orbit mode.")
    parser.add_argument("--disable-medium", action="store_true", help="Render the plain 3DGS checkpoint without a medium sidecar.")
    parser.add_argument("--medium-checkpoint", default="", help="Medium field checkpoint; empty auto-detects next to the PLY checkpoint.")
    parser.add_argument("--medium-far", type=float, default=0.0, help="Fallback medium integration distance; 0 uses scene_extent*4.")
    parser.add_argument("--medium-chunk-pixels", type=int, default=65536, help="Pixel chunk size for medium ray integration.")
    parser.add_argument("--medium-alpha-threshold", type=float, default=1.0e-3, help="Alpha threshold for choosing rendered depth over fallback medium far.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("路径渲染需要 CUDA。")

    data_dir = resolve_input_path(args.data)
    checkpoint = resolve_input_path(args.checkpoint)
    out_dir = resolve_output_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda")
    scene = load_colmap_dataset(str(data_dir), split=args.split, load_images=False, factor=args.factor, holdout=args.holdout, opengl=False)
    model = load_gaussian_checkpoint(checkpoint, device)
    base_renderer = GaussianRenderer(background=(1.0, 1.0, 1.0))
    medium_field, medium_path = load_medium_for_render(args, checkpoint, device)
    renderer = build_active_renderer(args, base_renderer, medium_field, scene)

    cameras = build_path_cameras(args, scene, device)
    print(f"设备：CUDA GPU='{torch.cuda.get_device_name(device)}'")
    print(f"checkpoint={checkpoint}")
    print(f"medium_checkpoint={medium_path if medium_field is not None else 'disabled'}")
    print(f"路径模式={args.mode} 帧数={len(cameras)} 分辨率={scene.width}x{scene.height}")
    print(f"输出目录：{out_dir}")

    for index, camera in enumerate(cameras):
        with torch.no_grad():
            render = renderer.render(model, camera)
        save_image(str(out_dir / f"frame_{index:04d}.png"), render.image.detach().clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy())
        print(f"[{index + 1}/{len(cameras)}] frame_{index:04d}.png")

    print("新视角路径渲染完成。")


def build_path_cameras(args: argparse.Namespace, scene, device: torch.device) -> list[Camera]:
    if args.mode == "interpolate":
        end = len(scene.image_paths) - 1 if args.end < 0 else args.end
        start = max(0, min(args.start, len(scene.image_paths) - 1))
        end = max(0, min(end, len(scene.image_paths) - 1))
        c2w_start = scene.c2w_matrices[start]
        c2w_end = scene.c2w_matrices[end]
        return [
            make_camera(scene, interpolate_pose(c2w_start, c2w_end, t), device, image_path=f"interpolate_{start}_{end}_{i}")
            for i, t in enumerate(np.linspace(0.0, 1.0, max(args.frames, 1)))
        ]
    return build_orbit_cameras(args, scene, device)


def build_orbit_cameras(args: argparse.Namespace, scene, device: torch.device) -> list[Camera]:
    c2ws = scene.c2w_matrices
    center = c2ws[:, :3, 3].mean(axis=0)
    radius = np.linalg.norm(c2ws[:, :3, 3] - center[None, :], axis=-1).mean() * args.radius_scale
    height = c2ws[:, 1, 3].mean()
    base_forward = center - c2ws[0, :3, 3]
    if np.linalg.norm(base_forward) < 1e-6:
        base_forward = np.array([0.0, 0.0, 1.0])
    cameras = []
    for i, theta in enumerate(np.linspace(0.0, 2.0 * np.pi, max(args.frames, 1), endpoint=False)):
        position = center + np.array([np.cos(theta) * radius, 0.0, np.sin(theta) * radius])
        position[1] = height
        c2w = look_at_c2w(position, center, up=np.array([0.0, 1.0, 0.0]))
        cameras.append(make_camera(scene, c2w, device, image_path=f"orbit_{i}"))
    return cameras


def make_camera(scene, c2w: np.ndarray, device: torch.device, image_path: str) -> Camera:
    return Camera(
        width=int(scene.width),
        height=int(scene.height),
        fx=float(scene.fx[0]),
        fy=float(scene.fy[0]),
        cx=float(scene.cx[0]),
        cy=float(scene.cy[0]),
        c2w=torch.as_tensor(c2w, dtype=torch.float32, device=device),
        image=None,
        image_path=image_path,
        near=float(scene.near),
        far=float(scene.far),
    )


def interpolate_pose(c2w_a: np.ndarray, c2w_b: np.ndarray, t: float) -> np.ndarray:
    qa = rotmat_to_quat(c2w_a[:3, :3])
    qb = rotmat_to_quat(c2w_b[:3, :3])
    q = slerp(qa, qb, float(t))
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = quat_to_rotmat(q)
    c2w[:3, 3] = (1.0 - t) * c2w_a[:3, 3] + t * c2w_b[:3, 3]
    return c2w


def look_at_c2w(position: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    forward = target - position
    forward = forward / max(np.linalg.norm(forward), 1e-12)
    right = np.cross(forward, up)
    right = right / max(np.linalg.norm(right), 1e-12)
    down = np.cross(forward, right)
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, 0] = right
    c2w[:3, 1] = down
    c2w[:3, 2] = forward
    c2w[:3, 3] = position
    return c2w


def rotmat_to_quat(rotmat: np.ndarray) -> np.ndarray:
    m = np.asarray(rotmat, dtype=np.float64)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        q = np.array([0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s])
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        q = np.array([(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s])
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        q = np.array([(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s])
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        q = np.array([(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s])
    return normalize_quat(q)


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = normalize_quat(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    q0 = normalize_quat(q0)
    q1 = normalize_quat(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        return normalize_quat((1.0 - t) * q0 + t * q1)
    theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
    theta = theta_0 * t
    return normalize_quat(np.sin(theta_0 - theta) / np.sin(theta_0) * q0 + np.sin(theta) / np.sin(theta_0) * q1)


def normalize_quat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    q = q / max(np.linalg.norm(q), 1e-12)
    return q if q[0] >= 0.0 else -q


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


def load_medium_for_render(args: argparse.Namespace, checkpoint: Path, device: torch.device) -> tuple[MediumField | None, Path | None]:
    if bool(args.disable_medium):
        return None, None
    if args.medium_checkpoint:
        medium_path = resolve_input_path(args.medium_checkpoint)
    else:
        medium_path = medium_checkpoint_path_for_ply(checkpoint)
        if not medium_path.exists():
            print(f"未找到 medium checkpoint，按普通 3DGS 渲染: {medium_path}")
            return None, None
    payload = torch.load(medium_path, map_location=device)
    return MediumField.from_checkpoint_payload(payload, device), medium_path


def build_active_renderer(args: argparse.Namespace, renderer: GaussianRenderer, medium_field: MediumField | None, scene):
    if medium_field is None:
        return renderer
    far_distance = float(args.medium_far) if float(args.medium_far) > 0.0 else float(scene.scene_extent) * 4.0
    return MediumRenderer(
        renderer,
        medium_field,
        MediumRenderConfig(
            far_distance=max(far_distance, 1.0e-3),
            chunk_pixels=max(int(args.medium_chunk_pixels), 1),
            alpha_threshold=max(float(args.medium_alpha_threshold), 0.0),
        ),
    )


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

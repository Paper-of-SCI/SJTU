"""Train official WaterSplatting on one COLMAP scene through nerfstudio."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.watersplatting.watersplatting_utils import (
    VENDOR_COMMIT,
    check_watersplatting_python,
    prepare_colmap_scene,
    python_default,
    resolve_input_path,
    resolve_output_path,
    run_watersplatting_command,
    scene_output_dir,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train WaterSplatting with the official nerfstudio implementation.")
    parser.add_argument("--data", default="src/datasets/SeathruNeRF_dataset/Curasao", help="Source COLMAP scene directory.")
    parser.add_argument("--out", default="", help="Output directory. Empty derives outputs/watersplatting_<scene>_<width>w.")
    parser.add_argument("--target-width", type=int, default=720, help="Prepared image width. Ignored by --keep-original-resolution.")
    parser.add_argument("--keep-original-resolution", action="store_true", help="Use original image size instead of resizing to --target-width.")
    parser.add_argument("--holdout", type=int, default=8, help="Every Nth image is held out for test.")
    parser.add_argument("--holdout-offset", type=int, default=0, help="Offset for held-out image selection.")
    parser.add_argument("--iterations", type=int, default=19999, help="WaterSplatting optimization steps.")
    parser.add_argument("--save-every", type=int, default=2000, help="nerfstudio checkpoint interval.")
    parser.add_argument("--eval-every", type=int, default=1000, help="nerfstudio eval interval.")
    parser.add_argument("--log-every", type=int, default=100, help="nerfstudio local logging interval.")
    parser.add_argument("--python", default=python_default(), help="Python executable for the WaterSplatting environment.")
    parser.add_argument("--skip-dependency-check", action="store_true")
    parser.add_argument("--overwrite-prepared-data", action="store_true")
    parser.add_argument("--prepare-only", action="store_true", help="Only create prepared_data and training_summary.json; do not launch training.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = resolve_input_path(args.data)
    out_dir = resolve_output_path(args.out) if args.out else scene_output_dir(data_dir, args.target_width, args.keep_original_resolution)
    prepared_dir = out_dir / "prepared_data"
    ns_output_dir = out_dir / "nerfstudio_outputs"
    experiment_name = "watersplatting"
    timestamp = "run"
    ns_config_path = ns_output_dir / experiment_name / "water-splatting" / timestamp / "config.yml"
    public_config_path = out_dir / "config.yml"
    out_dir.mkdir(parents=True, exist_ok=True)

    prepared_info = prepare_colmap_scene(
        data_dir=data_dir,
        prepared_dir=prepared_dir,
        target_width=args.target_width,
        holdout=args.holdout,
        holdout_offset=args.holdout_offset,
        keep_original_resolution=args.keep_original_resolution,
        overwrite=args.overwrite_prepared_data,
    )
    if not args.skip_dependency_check and not args.prepare_only:
        check_watersplatting_python(args.python)

    command = [
        "-m",
        "nerfstudio.scripts.train",
        "water-splatting",
        "--output-dir",
        str(ns_output_dir),
        "--experiment-name",
        experiment_name,
        "--timestamp",
        timestamp,
        "--vis",
        "tensorboard",
        "--max-num-iterations",
        str(int(args.iterations)),
        "--steps-per-save",
        str(max(int(args.save_every), 1)),
        "--steps-per-eval-image",
        str(max(int(args.eval_every), 1)),
        "--steps-per-eval-all-images",
        str(max(int(args.eval_every), 1)),
        "--logging.steps-per-log",
        str(max(int(args.log_every), 1)),
        "--pipeline.model.num-steps",
        str(int(args.iterations)),
        "colmap",
        "--data",
        str(prepared_dir),
        "--images-path",
        "images",
        "--colmap-path",
        "sparse/0",
        "--downscale-factor",
        "1",
        "--eval-mode",
        "filename",
    ]

    summary = {
        "method": "WaterSplatting",
        "official_repo": "https://github.com/water-splatting/water-splatting",
        "official_commit": VENDOR_COMMIT,
        "data": str(data_dir),
        "out": str(out_dir),
        "prepared": prepared_info.to_dict(),
        "split": {"holdout": int(args.holdout), "holdout_offset": int(args.holdout_offset)},
        "training": {
            "iterations": int(args.iterations),
            "save_every": int(args.save_every),
            "eval_every": int(args.eval_every),
            "log_every": int(args.log_every),
            "nerfstudio_config": str(ns_config_path),
            "public_config": str(public_config_path),
            "command": [args.python, *command],
        },
        "python": args.python,
    }
    (out_dir / "training_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"WaterSplatting prepared_data={prepared_dir}")
    print(
        f"resolution={prepared_info.width}x{prepared_info.height} target_width={args.target_width} "
        f"train={prepared_info.train_count} test={prepared_info.test_count} "
        f"holdout={args.holdout} holdout_offset={args.holdout_offset}"
    )
    if args.prepare_only:
        print(f"prepare-only 完成。summary={out_dir / 'training_summary.json'}")
        return

    run_watersplatting_command(args.python, command)
    if ns_config_path.exists():
        shutil.copy2(ns_config_path, public_config_path)
    else:
        raise FileNotFoundError(f"训练结束但找不到 nerfstudio config: {ns_config_path}")
    print(f"训练完成。config={public_config_path} summary={out_dir / 'training_summary.json'}")


if __name__ == "__main__":
    main()


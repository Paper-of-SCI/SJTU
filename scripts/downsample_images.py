from pathlib import Path
from PIL import Image, ImageOps
import argparse


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def resize_images(input_dir: Path, output_dir: Path, factor: float, overwrite: bool):
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder does not exist: {input_dir}")

    image_paths = [
        p for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]

    print(f"Found {len(image_paths)} images")
    print(f"Input : {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Factor: {factor}")

    for src in image_paths:
        rel = src.relative_to(input_dir)
        dst = output_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)

        if dst.exists() and not overwrite:
            continue

        with Image.open(src) as img:
            img = ImageOps.exif_transpose(img)

            w, h = img.size
            new_w = max(1, round(w / factor))
            new_h = max(1, round(h / factor))

            resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

            if dst.suffix.lower() in {".jpg", ".jpeg"}:
                if resized.mode in {"RGBA", "LA", "P"}:
                    resized = resized.convert("RGB")
                resized.save(dst, quality=95)
            else:
                resized.save(dst)

    print("Done")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--factor", type=float, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    resize_images(args.input, args.output, args.factor, args.overwrite)


if __name__ == "__main__":
    main()
#!/usr/bin/env python
"""
Image corruption utility for S0 experiments (levels L/M/H).
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Dict, List

import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate corrupted screenshots.")
    parser.add_argument("--in", dest="input_csv", required=True, help="Base test CSV.")
    parser.add_argument(
        "--out",
        dest="output_dir",
        required=True,
        help="Directory to store corrupted images & CSVs.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--levels",
        nargs="+",
        default=["L", "M", "H"],
        help="Corruption levels to generate.",
    )
    parser.add_argument(
        "--image-root",
        default="data/processed/screenshots",
        help="Root directory for relative img_path values.",
    )
    return parser.parse_args()


def resolve_image_path(path_str: str, image_root: Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] in {"workspace", "data"}:
        return path
    return image_root / path


def apply_corruption(img: Image.Image, level: str) -> Image.Image:
    level = level.upper()
    if level == "L":
        return img.filter(ImageFilter.GaussianBlur(radius=0.8))
    if level == "M":
        jitter = ImageEnhance.Color(img).enhance(0.6)
        return jitter.filter(ImageFilter.GaussianBlur(radius=1.2))
    # High severity: downsample + contrast jitter
    small = img.resize((img.width // 2 or 1, img.height // 2 or 1), Image.BILINEAR)
    upsampled = small.resize(img.size, Image.BILINEAR)
    contrast = ImageEnhance.Contrast(upsampled).enhance(0.5)
    return contrast.filter(ImageFilter.GaussianBlur(radius=1.5))


def save_corrupt_image(
    img: Image.Image, output_dir: Path, sample_id: str, level: str
) -> Dict[str, str]:
    target_dir = output_dir / level.upper() / "shot"
    target_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{sample_id}.jpg"
    target_path = target_dir / filename
    img.save(target_path, "JPEG", quality=95)
    sha256 = hashlib.sha256(target_path.read_bytes()).hexdigest()
    relative_path = target_path.relative_to(output_dir)
    return {
        "img_path_corrupt": str(relative_path).replace("\\", "/"),
        "img_sha256_corrupt": sha256,
    }


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    image_root = Path(args.image_root)

    if not input_path.exists():
        raise FileNotFoundError(input_path)

    df = pd.read_csv(input_path)
    if "id" not in df.columns or "img_path" not in df.columns:
        raise ValueError("CSV must contain id and img_path columns.")

    outputs: List[Path] = []

    for level in args.levels:
        records = []
        for _, row in df.iterrows():
            sample_id = str(row.get("id"))
            base_path = row.get("img_path")
            if pd.isna(base_path) or not str(base_path).strip():
                img = Image.new("RGB", (224, 224), color=(128, 128, 128))
            else:
                resolved = resolve_image_path(str(base_path), image_root)
                if resolved.exists():
                    try:
                        img = Image.open(resolved).convert("RGB")
                    except Exception:
                        img = Image.new("RGB", (224, 224), color=(128, 128, 128))
                else:
                    img = Image.new("RGB", (224, 224), color=(128, 128, 128))

            corrupted = apply_corruption(img, level)
            path_info = save_corrupt_image(corrupted, output_dir, sample_id, level)

            record = row.to_dict()
            record.update(path_info)
            record["corruption_level"] = level.upper()
            records.append(record)

        level_df = pd.DataFrame(records)
        csv_path = output_dir / f"test_corrupt_img_{level.upper()}.csv"
        level_df.to_csv(csv_path, index=False)
        outputs.append(csv_path)
        print(
            f"[corrupt_img] Level {level.upper()}: saved {len(level_df)} rows to {csv_path}"
        )

    print("Corruption CSV files:", [str(p) for p in outputs])


if __name__ == "__main__":
    main()

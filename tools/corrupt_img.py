#!/usr/bin/env python
"""
Image corruption utility for S0 experiments (levels L/M/H).
"""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path
from typing import Dict, List

import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter

# 增加PIL图像大小限制，防止DecompressionBombWarning
Image.MAX_IMAGE_PIXELS = 500_000_000  # 500MP

# Windows不允许的字符
INVALID_CHARS = re.compile(r'[<>:"/\\|?*`\r\n]')


def sanitize_filename(name: str, max_len: int = 200) -> str:
    """清理文件名，移除Windows不允许的字符，限制长度"""
    # 移除所有不允许的字符
    safe = INVALID_CHARS.sub("_", name)
    # 移除连续的下划线
    safe = re.sub(r"_+", "_", safe)
    # 移除首尾的下划线
    safe = safe.strip("_")
    # 如果文件名太长，使用hash缩短
    if len(safe) > max_len:
        name_hash = hashlib.md5(name.encode("utf-8")).hexdigest()[:8]
        # 保留前100个字符和后部分hash
        safe = safe[:100] + "_" + name_hash
    return safe if safe else "unknown"


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
    safe_id = sanitize_filename(str(sample_id))
    filename = f"{safe_id}.jpg"
    target_path = target_dir / filename

    # 确保图像大小合理（最大边不超过8192像素）
    max_dim = 8192
    if img.width > max_dim or img.height > max_dim:
        if img.width > img.height:
            new_width = max_dim
            new_height = int(img.height * max_dim / img.width)
        else:
            new_height = max_dim
            new_width = int(img.width * max_dim / img.height)
        img = img.resize((new_width, new_height), Image.Resampling.BILINEAR)

    sha256 = ""
    try:
        # 保存图像
        img.save(target_path, "JPEG", quality=95, optimize=True)
        # 计算SHA256
        sha256 = hashlib.sha256(target_path.read_bytes()).hexdigest()
    except (OSError, IOError, ValueError) as e:
        # 如果文件操作失败，尝试使用更小的quality或PNG格式
        try:
            # 尝试降低质量
            img.save(target_path, "JPEG", quality=85, optimize=True)
            sha256 = hashlib.sha256(target_path.read_bytes()).hexdigest()
        except Exception:
            # 如果还是失败，使用图像数据直接计算hash
            import io

            buffer = io.BytesIO()
            try:
                img.save(buffer, "JPEG", quality=85, format="JPEG")
                sha256 = hashlib.sha256(buffer.getvalue()).hexdigest()
            except Exception:
                # 最后尝试PNG
                buffer = io.BytesIO()
                img.save(buffer, "PNG")
                sha256 = hashlib.sha256(buffer.getvalue()).hexdigest()
                # 更改文件扩展名
                target_path = target_path.with_suffix(".png")
                filename = f"{safe_id}.png"
            print(
                f"Warning: Failed to save JPEG for {sample_id}, using alternative format: {e}"
            )

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
                        # Resize超大图像到合理大小（最大边4096像素）
                        max_size = 4096
                        if img.width > max_size or img.height > max_size:
                            if img.width > img.height:
                                new_width = max_size
                                new_height = int(img.height * max_size / img.width)
                            else:
                                new_height = max_size
                                new_width = int(img.width * max_size / img.height)
                            img = img.resize(
                                (new_width, new_height), Image.Resampling.BILINEAR
                            )
                    except Exception as e:
                        print(f"Warning: Failed to load image {resolved}: {e}")
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

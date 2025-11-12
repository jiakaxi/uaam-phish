#!/usr/bin/env python
"""
统一预处理脚本：为多模态数据集预生成HTML/URL tokens和图像缓存。

功能：
- 读取CSV文件
- 对每个样本生成：
  - {id}_html.pt (BERT tokenized: input_ids + attention_mask)
  - {id}_url.pt (character-level tokenized)
  - {id}_img_224.jpg (resized to 224x224, JPEG format)
- 更新CSV添加 *_cached 列
"""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer

# 增加PIL图像大小限制
Image.MAX_IMAGE_PIXELS = 500_000_000  # 500MP

# Windows不允许的字符（用于文件名清理）
INVALID_CHARS = re.compile(r'[<>:"/\\|?*`\r\n]')


def sanitize_filename(name: str, max_len: int = 200) -> str:
    """清理文件名，移除Windows不允许的字符，限制长度"""
    safe = INVALID_CHARS.sub("_", str(name))
    safe = re.sub(r"_+", "_", safe)
    safe = safe.strip("_")
    if len(safe) > max_len:
        name_hash = hashlib.md5(str(name).encode("utf-8")).hexdigest()[:8]
        safe = safe[:100] + "_" + name_hash
    return safe if safe else "unknown"


def load_html_text(row: pd.Series, html_root: Optional[Path] = None) -> str:
    """加载HTML文本，优先从html_text列，其次从html_path"""
    # 优先使用html_text列
    html_text = row.get("html_text", "")
    if html_text and pd.notna(html_text) and str(html_text).strip():
        return str(html_text)

    # 回退到html_path
    html_path = row.get("html_path")
    if pd.notna(html_path) and str(html_path).strip():
        path = Path(str(html_path))
        # 如果是绝对路径，直接使用
        if path.is_absolute():
            if path.exists():
                try:
                    return path.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    pass
        # 如果是相对路径，尝试从html_root解析
        elif html_root:
            full_path = html_root / path
            if full_path.exists():
                try:
                    return full_path.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    pass
        # 直接尝试路径
        elif path.exists():
            try:
                return path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                pass

    return ""


def tokenize_html(
    html_text: str, tokenizer: AutoTokenizer, max_len: int = 256
) -> Dict[str, torch.Tensor]:
    """使用BERT tokenizer对HTML进行tokenization"""
    if not html_text or not html_text.strip():
        html_text = "[EMPTY]"

    encoded = tokenizer(
        html_text,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    return {
        "input_ids": encoded["input_ids"].squeeze(0),
        "attention_mask": encoded["attention_mask"].squeeze(0),
    }


def tokenize_url(
    url_text: str, url_max_len: int = 200, url_vocab_size: int = 128
) -> torch.Tensor:
    """字符级URL tokenization"""
    if not url_text:
        url_text = ""

    # 字符级编码
    char_ids = [min(ord(c), url_vocab_size - 1) for c in url_text]

    # Padding或truncation
    if len(char_ids) > url_max_len:
        char_ids = char_ids[:url_max_len]
    else:
        char_ids += [0] * (url_max_len - len(char_ids))

    return torch.tensor(char_ids, dtype=torch.long)


def preprocess_image(
    row: pd.Series,
    image_dir: Optional[Path] = None,
    image_size: int = 224,
) -> Optional[Image.Image]:
    """加载并预处理图像到指定大小"""
    # 尝试多个可能的列名
    img_path = None
    for col in ["img_path", "image_path", "img_path_corrupt"]:
        if col in row and pd.notna(row.get(col)) and str(row.get(col)).strip():
            img_path = row.get(col)
            break

    if not img_path:
        return None

    path = Path(str(img_path))

    # 解析路径
    if path.is_absolute():
        if not path.exists():
            return None
    elif image_dir:
        full_path = image_dir / path
        if full_path.exists():
            path = full_path
        elif path.exists():
            pass
        else:
            return None
    elif not path.exists():
        return None

    try:
        img = Image.open(path).convert("RGB")
        # Resize到目标大小
        img = img.resize((image_size, image_size), Image.Resampling.BILINEAR)
        return img
    except Exception:
        return None


def process_sample(
    row: pd.Series,
    sample_id: str,
    output_dir: Path,
    html_tokenizer: AutoTokenizer,
    html_max_len: int,
    url_max_len: int,
    url_vocab_size: int,
    image_size: int,
    html_root: Optional[Path] = None,
    image_dir: Optional[Path] = None,
) -> Dict[str, Optional[str]]:
    """处理单个样本，生成所有预处理文件"""
    safe_id = sanitize_filename(sample_id)
    results = {
        "html_tokens_path": None,
        "url_tokens_path": None,
        "img_path_cached": None,
    }

    # 1. 处理HTML
    try:
        html_text = load_html_text(row, html_root)
        html_tokens = tokenize_html(html_text, html_tokenizer, html_max_len)
        html_path = output_dir / f"{safe_id}_html.pt"
        torch.save(html_tokens, html_path)
        results["html_tokens_path"] = str(html_path.relative_to(output_dir))
    except Exception as e:
        print(f"Warning: Failed to process HTML for {sample_id}: {e}")

    # 2. 处理URL
    try:
        url_text = str(row.get("url_text", row.get("url", "")))
        url_tokens = tokenize_url(url_text, url_max_len, url_vocab_size)
        url_path = output_dir / f"{safe_id}_url.pt"
        torch.save(url_tokens, url_path)
        results["url_tokens_path"] = str(url_path.relative_to(output_dir))
    except Exception as e:
        print(f"Warning: Failed to process URL for {sample_id}: {e}")

    # 3. 处理图像
    try:
        img = preprocess_image(row, image_dir, image_size)
        if img:
            img_path = output_dir / f"{safe_id}_img_{image_size}.jpg"
            img.save(img_path, "JPEG", quality=95, optimize=True)
            results["img_path_cached"] = str(img_path.relative_to(output_dir))
    except Exception as e:
        print(f"Warning: Failed to process image for {sample_id}: {e}")

    return results


def main():
    parser = argparse.ArgumentParser(description="预处理多模态数据集（HTML/URL/图像）")
    parser.add_argument("--csv", required=True, help="输入CSV文件路径")
    parser.add_argument("--output", required=True, help="输出目录")
    parser.add_argument(
        "--html-max-len", type=int, default=256, help="HTML最大token长度"
    )
    parser.add_argument("--url-max-len", type=int, default=200, help="URL最大长度")
    parser.add_argument("--url-vocab-size", type=int, default=128, help="URL词汇表大小")
    parser.add_argument("--image-size", type=int, default=224, help="图像大小")
    parser.add_argument(
        "--html-root", default=None, help="HTML文件根目录（用于解析相对路径）"
    )
    parser.add_argument(
        "--image-dir", default="data/processed/screenshots", help="图像文件根目录"
    )
    parser.add_argument(
        "--bert-model", default="bert-base-uncased", help="BERT模型名称"
    )
    parser.add_argument("--id-col", default="id", help="ID列名")
    parser.add_argument(
        "--out-csv", default=None, help="输出CSV文件路径（可选，默认保存到原CSV同目录）"
    )

    args = parser.parse_args()

    # 解析路径
    csv_path = Path(args.csv)
    output_dir = Path(args.output)
    html_root = Path(args.html_root) if args.html_root else None
    image_dir = Path(args.image_dir) if args.image_dir else None

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV文件不存在: {csv_path}")

    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载CSV
    print(f"加载CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"样本数: {len(df)}")

    # 检查必需的列
    if args.id_col not in df.columns:
        raise ValueError(f"CSV必须包含 '{args.id_col}' 列")

    # 加载tokenizer
    print(f"加载BERT tokenizer: {args.bert_model}")
    html_tokenizer = AutoTokenizer.from_pretrained(args.bert_model)

    # 处理每个样本
    print(f"开始预处理，输出目录: {output_dir}")
    cached_paths = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="预处理"):
        sample_id = str(row[args.id_col])
        results = process_sample(
            row=row,
            sample_id=sample_id,
            output_dir=output_dir,
            html_tokenizer=html_tokenizer,
            html_max_len=args.html_max_len,
            url_max_len=args.url_max_len,
            url_vocab_size=args.url_vocab_size,
            image_size=args.image_size,
            html_root=html_root,
            image_dir=image_dir,
        )
        cached_paths.append(results)

    # 更新CSV
    print("更新CSV，添加缓存路径列...")
    for key in ["html_tokens_path", "url_tokens_path", "img_path_cached"]:
        df[key] = [r[key] for r in cached_paths]

    # 保存更新的CSV
    if args.out_csv:
        output_csv = Path(args.out_csv)
    else:
        # 默认保存到原CSV同目录
        output_csv = csv_path.parent / f"{csv_path.stem}_cached.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"保存更新的CSV: {output_csv}")

    # 统计信息
    html_count = sum(1 for r in cached_paths if r["html_tokens_path"])
    url_count = sum(1 for r in cached_paths if r["url_tokens_path"])
    img_count = sum(1 for r in cached_paths if r["img_path_cached"])

    print("\n预处理完成！")
    print(f"  HTML tokens: {html_count}/{len(df)}")
    print(f"  URL tokens: {url_count}/{len(df)}")
    print(f"  Images: {img_count}/{len(df)}")
    print(f"  输出目录: {output_dir}")
    print(f"  更新CSV: {output_csv}")


if __name__ == "__main__":
    main()

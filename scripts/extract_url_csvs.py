#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 master_v2.csv 提取 URL 模态的 train/val/test CSV 文件
确保与 HTML 和 IMG 模态的数据划分一致
"""

import argparse
import sys
from pathlib import Path
import pandas as pd

# Handle Windows console encoding
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        import codecs

        sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "strict")
        sys.stderr = codecs.getwriter("utf-8")(sys.stderr.buffer, "strict")


def main():
    parser = argparse.ArgumentParser(
        description="Extract URL modality CSV files from master CSV"
    )
    parser.add_argument(
        "--master_csv",
        type=str,
        default="data/processed/master_v2.csv",
        help="Path to master CSV file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed",
        help="Output directory for URL CSV files",
    )
    parser.add_argument(
        "--url_col",
        type=str,
        default="url_text",
        help="Column name for URL text in master CSV",
    )
    parser.add_argument(
        "--split_col",
        type=str,
        default="split",
        help="Column name for split information",
    )

    args = parser.parse_args()

    # 读取 master CSV
    master_path = Path(args.master_csv)
    if not master_path.exists():
        print(f"❌ Master CSV not found: {master_path}")
        return 1

    print(f"📖 Reading master CSV: {master_path}")
    df = pd.read_csv(master_path)
    print(f"   Total samples: {len(df)}")

    # 检查必需列
    required_cols = {"id", args.url_col, "label", args.split_col}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        print(f"❌ Missing required columns: {missing_cols}")
        print(f"   Available columns: {list(df.columns)}")
        return 1

    # 选择 URL 相关列
    url_cols = ["id", args.url_col, "label"]

    # 添加可选的元数据列
    optional_cols = ["timestamp", "brand", "source", "domain"]
    for col in optional_cols:
        if col in df.columns:
            url_cols.append(col)

    print(f"📝 Extracting columns: {url_cols}")

    # 按 split 过滤并保存
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = ["train", "val", "test"]
    for split_name in splits:
        split_df = df[df[args.split_col] == split_name][url_cols].copy()

        # 重命名 url_col 为标准名称 (如果需要)
        if args.url_col != "url_text":
            split_df.rename(columns={args.url_col: "url_text"}, inplace=True)

        output_file = output_dir / f"url_{split_name}_v2.csv"
        split_df.to_csv(output_file, index=False, encoding="utf-8")

        print(f"✅ {split_name:5s} saved: {output_file}")
        print(f"   - Samples: {len(split_df)}")
        print(
            f"   - Label distribution: 0={sum(split_df['label']==0)}, 1={sum(split_df['label']==1)}"
        )

    # 统计总览
    print("\n" + "=" * 70)
    print("📊 Summary:")
    print("=" * 70)
    for split_name in splits:
        split_df = df[df[args.split_col] == split_name]
        print(f"{split_name:5s}: {len(split_df):5d} samples")
    print(f"Total: {len(df):5d} samples")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    exit(main())

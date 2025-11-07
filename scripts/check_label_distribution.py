#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查数据集的标签分布
"""

import sys
import pandas as pd
from pathlib import Path

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
    csv_path = Path("data/processed/master_v2.csv")

    print("=" * 70)
    print("数据集标签分布详情")
    print("=" * 70)

    df = pd.read_csv(csv_path)

    print(f"\n总样本数: {len(df):,}")

    # 标签分布
    print("\n标签分布:")
    label_dist = df["label"].value_counts().sort_index()
    for label, count in label_dist.items():
        label_name = "钓鱼 (Phishing)" if label == 1 else "合法 (Benign)"
        pct = count / len(df) * 100
        print(f"  Label {label} ({label_name:20s}): {count:,} ({pct:.2f}%)")

    # 差值
    diff = abs(label_dist[1] - label_dist[0])
    print(f"\n标签差值: {diff} 个样本")
    print(f"平衡度: {(1 - diff/len(df))*100:.2f}%")

    # 按来源统计
    print("\n按来源统计:")
    print("-" * 70)
    source_label = df.groupby(["source", "label"]).size().unstack(fill_value=0)
    source_label.columns = ["Benign", "Phishing"]
    source_label["Total"] = source_label.sum(axis=1)
    print(source_label)

    # Split分布
    print("\nSplit分布:")
    print("-" * 70)
    split_dist = df["split"].value_counts()
    for split_name, count in split_dist.items():
        pct = count / len(df) * 100
        print(f"  {split_name:10s}: {count:,} ({pct:.2f}%)")

    # 时间范围
    print("\n时间范围:")
    print("-" * 70)
    df["timestamp_parsed"] = pd.to_datetime(
        df["timestamp"], format="ISO8601", errors="coerce"
    )
    valid_ts = df["timestamp_parsed"].dropna()
    if len(valid_ts) > 0:
        print(f"  最早: {valid_ts.min()}")
        print(f"  最晚: {valid_ts.max()}")
        print(f"  跨度: {(valid_ts.max() - valid_ts.min()).days} 天")

    # 品牌统计
    print("\n品牌统计:")
    print("-" * 70)
    brand_counts = df["brand"].value_counts()
    print(f"  总品牌数: {len(brand_counts):,}")
    print(f"  Top 1 占比: {brand_counts.iloc[0]/len(df)*100:.2f}%")
    print(f"  Top 3 占比: {brand_counts.head(3).sum()/len(df)*100:.2f}%")
    print(f"  Top 10 占比: {brand_counts.head(10).sum()/len(df)*100:.2f}%")

    print("\n  Top 10 品牌:")
    for i, (brand, count) in enumerate(brand_counts.head(10).items(), 1):
        pct = count / len(df) * 100
        print(f"    {i:2d}. {brand:40s}: {count:5d} ({pct:.2f}%)")

    # 按标签的品牌分布
    print("\n按标签的品牌统计:")
    print("-" * 70)
    phishing_brands = df[df["label"] == 1]["brand"].nunique()
    benign_brands = df[df["label"] == 0]["brand"].nunique()
    print(f"  钓鱼样本品牌数: {phishing_brands:,}")
    print(f"  合法样本品牌数: {benign_brands:,}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()

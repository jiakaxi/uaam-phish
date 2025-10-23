#!/usr/bin/env python3
"""
创建 master.csv 用于 build_splits

如果你已有 train/val/test CSV 但没有 master.csv，运行此脚本合并它们
"""

import pandas as pd
from pathlib import Path
import sys


def main():
    data_dir = Path("data/processed")

    train_csv = data_dir / "url_train.csv"
    val_csv = data_dir / "url_val.csv"
    test_csv = data_dir / "url_test.csv"
    master_csv = data_dir / "master.csv"

    # 检查文件存在性
    missing = []
    for f in [train_csv, val_csv, test_csv]:
        if not f.exists():
            missing.append(str(f))

    if missing:
        print("❌ 缺少必需文件:")
        for f in missing:
            print(f"   - {f}")
        print("\n提示: 确保先运行数据预处理生成 train/val/test CSV")
        sys.exit(1)

    # 读取并合并
    print("📖 读取数据文件...")
    train = pd.read_csv(train_csv)
    val = pd.read_csv(val_csv)
    test = pd.read_csv(test_csv)

    print(f"   - train: {len(train)} samples")
    print(f"   - val: {len(val)} samples")
    print(f"   - test: {len(test)} samples")

    # 合并
    print("\n🔗 合并数据...")
    master = pd.concat([train, val, test], ignore_index=True)

    # 检查列
    print(f"   - Total: {len(master)} samples")
    print(f"   - Columns: {list(master.columns)}")

    # 保存
    master_csv.parent.mkdir(parents=True, exist_ok=True)
    master.to_csv(master_csv, index=False)

    print(f"\n✅ master.csv 已创建: {master_csv}")
    print(f"   - {len(master)} 样本")
    print(f"   - {len(master.columns)} 列")

    # 统计
    if "label" in master.columns:
        label_counts = master["label"].value_counts()
        print("\n📊 标签分布:")
        for label, count in label_counts.items():
            print(f"   - label={label}: {count} ({count/len(master)*100:.1f}%)")

    if "brand" in master.columns:
        brand_count = master["brand"].nunique()
        print(f"\n🏷️  品牌数: {brand_count}")

    if "timestamp" in master.columns:
        ts_count = master["timestamp"].notna().sum()
        print(
            f"\n📅 时间戳: {ts_count}/{len(master)} ({ts_count/len(master)*100:.1f}%) 非空"
        )

    print("\n✨ 现在可以运行:")
    print("   python scripts/train_hydra.py protocol=random use_build_splits=true")


if __name__ == "__main__":
    main()

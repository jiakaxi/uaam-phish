#!/usr/bin/env python
"""
迁移缓存CSV文件：将旧的缓存CSV文件移动到正确的位置并更新列名。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="迁移缓存CSV文件")
    parser.add_argument(
        "--scenario",
        choices=["iid", "brandood"],
        default="iid",
        help="数据集场景",
    )
    parser.add_argument(
        "--preprocessed-root",
        default="workspace/data/preprocessed",
        help="预处理产物根目录",
    )
    parser.add_argument(
        "--splits-root",
        default="workspace/data/splits",
        help="Splits根目录",
    )
    return parser.parse_args()


def migrate_csv(
    source_csv: Path,
    target_csv: Path,
    old_columns: dict,
    new_columns: dict,
) -> bool:
    """迁移CSV文件，更新列名"""
    if not source_csv.exists():
        print(f"  [X] 源文件不存在: {source_csv}")
        return False

    print(f"  读取: {source_csv}")
    df = pd.read_csv(source_csv)

    # 检查是否有旧列名
    has_old_columns = any(col in df.columns for col in old_columns.keys())
    if not has_old_columns:
        print("  [SKIP] 文件已使用新列名，跳过")
        return True

    # 重命名列
    rename_map = {}
    for old_col, new_col in old_columns.items():
        if old_col in df.columns:
            rename_map[old_col] = new_col
            print(f"    重命名: {old_col} -> {new_col}")

    if rename_map:
        df = df.rename(columns=rename_map)

    # 确保目标目录存在
    target_csv.parent.mkdir(parents=True, exist_ok=True)

    # 保存到新位置
    df.to_csv(target_csv, index=False)
    print(f"  保存: {target_csv}")
    print(f"  样本数: {len(df)}")

    return True


def main() -> None:
    args = parse_args()
    scenario = args.scenario
    preprocessed_root = Path(args.preprocessed_root)
    splits_root = Path(args.splits_root)

    # 列名映射
    old_columns = {
        "html_path_cached": "html_tokens_path",
        "url_path_cached": "url_tokens_path",
    }
    new_columns = {
        "html_tokens_path": "html_tokens_path",
        "url_tokens_path": "url_tokens_path",
    }

    print("=" * 70)
    print(f"迁移缓存CSV文件 - {scenario.upper()}")
    print("=" * 70)

    if scenario == "iid":
        split_names = ["train", "val", "test"]
    else:  # brandood
        split_names = ["train", "val", "test_id", "test_ood"]

    success_count = 0
    for split_name in split_names:
        print(f"\n处理 {split_name}:")
        source_csv = (
            preprocessed_root / scenario / split_name / f"{split_name}_cached.csv"
        )
        target_csv = splits_root / scenario / f"{split_name}_cached.csv"

        if migrate_csv(source_csv, target_csv, old_columns, new_columns):
            success_count += 1

    print("\n" + "=" * 70)
    print(f"完成: {success_count}/{len(split_names)} 个文件迁移成功")
    print("=" * 70)


if __name__ == "__main__":
    main()

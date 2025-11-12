#!/usr/bin/env python
"""
缓存文件完整性检查脚本：验证预处理输出的CSV列和文件数量。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="检查缓存文件完整性")
    parser.add_argument(
        "--splits-root",
        default="workspace/data/splits",
        help="Splits根目录",
    )
    parser.add_argument(
        "--preprocessed-root",
        default="workspace/data/preprocessed",
        help="预处理产物根目录",
    )
    parser.add_argument(
        "--scenario",
        choices=["iid", "brandood"],
        default="iid",
        help="数据集场景",
    )
    return parser.parse_args()


def check_cache_csv(
    csv_path: Path,
    preprocessed_dir: Path,
    required_columns: List[str],
) -> Dict[str, any]:
    """检查单个缓存CSV文件的完整性"""
    result = {
        "csv_path": str(csv_path),
        "exists": False,
        "has_columns": False,
        "columns": [],
        "total_samples": 0,
        "non_null_counts": {},
        "non_null_rates": {},
        "file_exists_counts": {},
        "file_exists_rates": {},
        "passed": False,
    }

    if not csv_path.exists():
        return result

    result["exists"] = True
    df = pd.read_csv(csv_path)
    result["total_samples"] = len(df)
    result["columns"] = list(df.columns)

    # 检查必需的列
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        result["missing_columns"] = missing_columns
        return result

    result["has_columns"] = True

    # 统计非空数量和比例
    for col in required_columns:
        non_null_count = df[col].notna().sum()
        non_null_rate = non_null_count / len(df) if len(df) > 0 else 0.0
        result["non_null_counts"][col] = int(non_null_count)
        result["non_null_rates"][col] = float(non_null_rate)

        # 检查文件是否存在
        if preprocessed_dir.exists():
            file_exists_count = 0
            for idx, row in df.iterrows():
                path_str = row.get(col)
                if pd.notna(path_str) and str(path_str).strip():
                    file_path = Path(path_str)
                    if not file_path.is_absolute():
                        file_path = preprocessed_dir / file_path
                    if file_path.exists():
                        file_exists_count += 1

            file_exists_rate = (
                file_exists_count / non_null_count if non_null_count > 0 else 0.0
            )
            result["file_exists_counts"][col] = int(file_exists_count)
            result["file_exists_rates"][col] = float(file_exists_rate)

    # 判断是否通过（非空率 > 95%）
    min_non_null_rate = (
        min(result["non_null_rates"].values()) if result["non_null_rates"] else 0.0
    )
    result["passed"] = min_non_null_rate > 0.95

    return result


def main() -> None:
    args = parse_args()
    splits_root = Path(args.splits_root)
    preprocessed_root = Path(args.preprocessed_root)
    scenario = args.scenario

    required_columns = ["img_path_cached", "html_tokens_path", "url_tokens_path"]

    print("=" * 70)
    print(f"缓存文件完整性检查 - {scenario.upper()}")
    print("=" * 70)

    if scenario == "iid":
        split_names = ["train", "val", "test"]
    else:  # brandood
        split_names = ["train", "val", "test_id", "test_ood"]

    all_passed = True
    results = {}

    for split_name in split_names:
        csv_path = splits_root / scenario / f"{split_name}_cached.csv"
        preprocessed_dir = preprocessed_root / scenario / split_name

        print(f"\n检查 {split_name}:")
        print(f"  CSV路径: {csv_path}")
        print(f"  预处理目录: {preprocessed_dir}")

        result = check_cache_csv(csv_path, preprocessed_dir, required_columns)
        results[split_name] = result

        if not result["exists"]:
            print("  [X] CSV文件不存在")
            all_passed = False
            continue

        if not result["has_columns"]:
            print(f"  [X] 缺少必需的列: {result.get('missing_columns', [])}")
            all_passed = False
            continue

        print(f"  [OK] CSV文件存在，包含 {result['total_samples']} 个样本")
        print(f"  [OK] 包含必需的列: {', '.join(required_columns)}")

        # 显示非空统计
        print("\n  非空统计:")
        for col in required_columns:
            non_null_rate = result["non_null_rates"][col]
            file_exists_rate = result["file_exists_rates"].get(col, 0.0)
            status = "[OK]" if non_null_rate > 0.95 else "[X]"
            print(f"    {status} {col}:")
            print(
                f"      非空率: {non_null_rate:.2%} ({result['non_null_counts'][col]}/{result['total_samples']})"
            )
            if file_exists_rate > 0:
                print(
                    f"      文件存在率: {file_exists_rate:.2%} ({result['file_exists_counts'][col]}/{result['non_null_counts'][col]})"
                )

        if result["passed"]:
            print("  [OK] 通过检查（非空率 > 95%）")
        else:
            print("  [X] 未通过检查（非空率 <= 95%）")
            all_passed = False

    print("\n" + "=" * 70)
    if all_passed:
        print("[OK] 所有检查通过！")
    else:
        print("[X] 部分检查未通过，请检查上述输出")
    print("=" * 70)

    # 返回退出码
    exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()

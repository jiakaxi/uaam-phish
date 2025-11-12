#!/usr/bin/env python
"""
检查master_v2.csv中每个brand的0/1分布，为Brand-OOD分割提供数据支持。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import pandas as pd
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="检查brand分布")
    parser.add_argument("--csv", required=True, help="Master CSV文件路径")
    parser.add_argument(
        "--out",
        default="workspace/reports/brand_distribution_report.json",
        help="输出JSON报告路径",
    )
    parser.add_argument(
        "--min-neg-threshold",
        type=int,
        default=10,
        help="最低负例数阈值，用于识别有效品牌",
    )
    return parser.parse_args()


def convert_numpy_types(obj):
    """将numpy类型转换为Python原生类型，用于JSON序列化"""
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


def check_brand_distribution(df: pd.DataFrame, min_neg_threshold: int = 10) -> Dict:
    """检查每个brand的类别分布"""

    # 确保brand列存在且已标准化
    if "brand" not in df.columns:
        raise ValueError("CSV文件中缺少brand列")

    # 标准化brand名称
    df = df.copy()
    df["brand"] = df["brand"].astype(str).str.strip().str.lower()

    # 统计每个brand的分布
    brand_stats = df.groupby("brand")["label"].agg(["count", "sum"]).reset_index()
    brand_stats.columns = ["brand", "total", "pos_count"]
    brand_stats["neg_count"] = brand_stats["total"] - brand_stats["pos_count"]
    brand_stats["neg_ratio"] = brand_stats["neg_count"] / brand_stats["total"]

    # 按总样本数排序
    brand_stats = brand_stats.sort_values("total", ascending=False)

    # 识别有效品牌（有足够负例）
    valid_brands = brand_stats[brand_stats["neg_count"] >= min_neg_threshold]

    # 总体统计
    total_stats = {
        "total_brands": len(brand_stats),
        "total_samples": len(df),
        "total_pos": int((df["label"] == 1).sum()),
        "total_neg": int((df["label"] == 0).sum()),
        "overall_neg_ratio": float((df["label"] == 0).sum() / len(df)),
        "valid_brands_count": len(valid_brands),
        "valid_brands_ratio": float(len(valid_brands) / len(brand_stats)),
        "min_neg_threshold": min_neg_threshold,
    }

    result = {
        "overall": total_stats,
        "brands": brand_stats.to_dict("records"),
        "valid_brands": valid_brands.to_dict("records"),
        "top_20_brands": brand_stats.head(20).to_dict("records"),
    }

    # 转换numpy类型为Python原生类型
    return convert_numpy_types(result)


def print_summary_table(report: Dict) -> None:
    """打印品牌分布摘要表格"""

    overall = report["overall"]
    top_20 = report["top_20_brands"]

    print("=" * 80)
    print("品牌分布检查报告")
    print("=" * 80)

    print("\n总体统计:")
    print(f"  总品牌数: {overall['total_brands']}")
    print(f"  总样本数: {overall['total_samples']}")
    print(
        f"  正例数: {overall['total_pos']} ({overall['total_pos']/overall['total_samples']*100:.1f}%)"
    )
    print(
        f"  负例数: {overall['total_neg']} ({overall['total_neg']/overall['total_samples']*100:.1f}%)"
    )
    print(
        f"  有效品牌数(负例≥{overall['min_neg_threshold']}): {overall['valid_brands_count']} ({overall['valid_brands_ratio']*100:.1f}%)"
    )

    print("\nTop 20品牌分布:")
    print("-" * 80)
    print(f"{'品牌':<20} {'总数':<6} {'正例':<6} {'负例':<6} {'负例比例':<10}")
    print("-" * 80)

    for brand in top_20:
        brand_name = (
            brand["brand"][:18] + ".." if len(brand["brand"]) > 20 else brand["brand"]
        )
        print(
            f"{brand_name:<20} {brand['total']:<6} {brand['pos_count']:<6} {brand['neg_count']:<6} {brand['neg_ratio']*100:<9.1f}%"
        )

    print("-" * 80)

    # 显示有效品牌信息
    valid_brands = report["valid_brands"]
    if valid_brands:
        print(f"\n有效品牌列表(负例≥{overall['min_neg_threshold']}):")
        print("-" * 50)
        for i, brand in enumerate(valid_brands[:10], 1):  # 只显示前10个
            brand_name = (
                brand["brand"][:18] + ".."
                if len(brand["brand"]) > 20
                else brand["brand"]
            )
            print(f"{i:2d}. {brand_name:<20} 负例: {brand['neg_count']}")
        if len(valid_brands) > 10:
            print(f"  ... 还有 {len(valid_brands) - 10} 个有效品牌")
    else:
        print(f"\n[WARNING] 没有找到负例数≥{overall['min_neg_threshold']}的品牌")

    print("=" * 80)


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV文件不存在: {csv_path}")

    print(f"读取CSV文件: {csv_path}")
    df = pd.read_csv(csv_path)

    # 检查必需列
    required_cols = ["brand", "label"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV文件缺少必需列: {missing_cols}")

    # 生成报告
    report = check_brand_distribution(df, args.min_neg_threshold)

    # 打印摘要表格
    print_summary_table(report)

    # 保存JSON报告
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n详细报告已保存到: {output_path}")

    # 提供建议
    overall = report["overall"]
    if overall["valid_brands_count"] < 20:
        print(
            f"\n[WARNING] 建议: 有效品牌数({overall['valid_brands_count']})少于20，考虑降低min_neg_threshold"
        )
    else:
        print(
            f"\n[OK] 有效品牌数({overall['valid_brands_count']})充足，可以使用min_neg_threshold={args.min_neg_threshold}"
        )


if __name__ == "__main__":
    main()

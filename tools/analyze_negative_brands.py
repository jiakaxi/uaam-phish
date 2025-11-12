#!/usr/bin/env python
"""
分析有负例的品牌分布
"""

import json
from pathlib import Path


def main():
    # 读取详细报告
    report_path = Path("workspace/reports/brand_distribution_detailed.json")
    with open(report_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 获取有负例的品牌（按负例数排序）
    brands_with_neg = [b for b in data["brands"] if b["neg_count"] > 0]
    brands_with_neg.sort(key=lambda x: x["neg_count"], reverse=True)

    print("有负例的品牌（按负例数排序，前50个）:")
    print("-" * 70)
    print(f'{"品牌":<25} {"总数":<6} {"正例":<6} {"负例":<6} {"负例比例":<10}')
    print("-" * 70)

    for i, brand in enumerate(brands_with_neg[:50], 1):
        brand_name = (
            brand["brand"][:22] + ".." if len(brand["brand"]) > 24 else brand["brand"]
        )
        print(
            f'{i:2d}. {brand_name:<22} {brand["total"]:<6} {brand["pos_count"]:<6} {brand["neg_count"]:<6} {brand["neg_ratio"]*100:<8.1f}%'
        )

    print("-" * 70)
    print(f"总共有 {len(brands_with_neg)} 个品牌有负例")

    # 统计不同负例数量的品牌分布
    print("\n负例数量分布:")
    print("-" * 30)
    neg_counts = {}
    for brand in brands_with_neg:
        neg_count = brand["neg_count"]
        if neg_count not in neg_counts:
            neg_counts[neg_count] = 0
        neg_counts[neg_count] += 1

    for count in sorted(neg_counts.keys(), reverse=True):
        print(f"负例数 {count:2d}: {neg_counts[count]:3d} 个品牌")

    # 推荐合适的阈值
    print("\n推荐阈值分析:")
    print("-" * 30)
    thresholds = [1, 2, 3, 5, 10]
    for threshold in thresholds:
        valid_brands = [b for b in brands_with_neg if b["neg_count"] >= threshold]
        print(f"阈值 ≥{threshold}: {len(valid_brands):3d} 个品牌")

    # 推荐策略
    print("\n推荐策略:")
    print("-" * 30)
    print("1. 使用阈值 ≥3: 有 8 个品牌，可以组成合理的in-domain集合")
    print("2. 使用阈值 ≥2: 有 15 个品牌，选择前10个作为in-domain")
    print("3. 使用阈值 ≥1: 有 7672 个品牌，但大多数只有1个负例")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
分析同时有正例和负例的品牌分布
"""

import pandas as pd


def main():
    # 读取数据
    df = pd.read_csv("data/processed/master_v2.csv")

    # 标准化brand
    df["brand"] = df["brand"].astype(str).str.strip().str.lower()

    # 统计每个品牌的类别分布
    brand_stats = df.groupby("brand")["label"].agg(["count", "sum"]).reset_index()
    brand_stats.columns = ["brand", "total", "pos_count"]
    brand_stats["neg_count"] = brand_stats["total"] - brand_stats["pos_count"]

    # 筛选出同时有正例和负例的品牌
    balanced_brands = brand_stats[
        (brand_stats["pos_count"] > 0) & (brand_stats["neg_count"] > 0)
    ].sort_values("total", ascending=False)

    print("Brands with both positive and negative samples:")
    print("-" * 60)
    print(f'{"品牌":<20} {"总数":<6} {"正例":<6} {"负例":<6} {"正例比例":<10}')
    print("-" * 60)

    for i, brand in enumerate(balanced_brands.head(50).itertuples(), 1):
        pos_ratio = brand.pos_count / brand.total * 100
        print(
            f"{i:2d}. {brand.brand:<18} {brand.total:<6} {brand.pos_count:<6} {brand.neg_count:<6} {pos_ratio:<9.1f}%"
        )

    print("-" * 60)
    print(f"Total brands with both classes: {len(balanced_brands)}")

    # 统计不同阈值下的品牌数量
    print("\nBrand counts by thresholds:")
    print("-" * 40)
    thresholds = [(1, 1), (2, 2), (3, 3), (5, 5), (1, 2), (2, 1), (1, 3), (3, 1)]

    for min_pos, min_neg in thresholds:
        count = len(
            balanced_brands[
                (balanced_brands["pos_count"] >= min_pos)
                & (balanced_brands["neg_count"] >= min_neg)
            ]
        )
        print(f"pos≥{min_pos}, neg≥{min_neg}: {count:3d} brands")

    # 推荐策略
    print("\nRecommended strategy:")
    print("-" * 40)

    # 检查不同阈值
    for min_pos, min_neg in [(1, 1), (2, 2), (3, 3)]:
        valid = balanced_brands[
            (balanced_brands["pos_count"] >= min_pos)
            & (balanced_brands["neg_count"] >= min_neg)
        ]
        if len(valid) >= 10:
            print(
                f"Use min_pos={min_pos}, min_neg={min_neg}: {len(valid)} brands available"
            )
            print("Top 10 brands:")
            for i, brand in enumerate(valid.head(10).itertuples(), 1):
                print(
                    f"  {i:2d}. {brand.brand:<18} pos: {brand.pos_count:2d}, neg: {brand.neg_count:2d}"
                )
            break


if __name__ == "__main__":
    main()

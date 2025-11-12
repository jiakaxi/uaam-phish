#!/usr/bin/env python
"""
分析in-domain数据的分布情况
"""

import pandas as pd


def main():
    # 读取数据
    df = pd.read_csv("data/processed/master_v2.csv")

    # 标准化brand
    df["brand"] = df["brand"].astype(str).str.strip().str.lower()

    # 选择有足够负例的品牌
    brand_stats = df.groupby("brand")["label"].agg(["count", "sum"]).reset_index()
    brand_stats.columns = ["brand", "total", "pos_count"]
    brand_stats["neg_count"] = brand_stats["total"] - brand_stats["pos_count"]

    # 筛选出负例数 >= 3 的品牌
    valid_brands = brand_stats[brand_stats["neg_count"] >= 3]
    valid_brands = valid_brands.sort_values("total", ascending=False).head(20)

    b_ind = valid_brands["brand"].tolist()
    print("Selected in-domain brands:")
    for i, brand in enumerate(b_ind, 1):
        brand_info = brand_stats[brand_stats["brand"] == brand].iloc[0]
        print(
            "%2d. %-20s total: %3d, pos: %3d, neg: %3d"
            % (
                i,
                brand,
                brand_info["total"],
                brand_info["pos_count"],
                brand_info["neg_count"],
            )
        )

    # 检查in-domain数据的总体分布
    df_ind = df[df["brand"].isin(b_ind)]
    print("\nIn-domain data distribution:")
    print("Total samples: %d" % len(df_ind))
    print("Positive samples: %d" % (df_ind["label"] == 1).sum())
    print("Negative samples: %d" % (df_ind["label"] == 0).sum())
    print("Negative ratio: %.1f%%" % ((df_ind["label"] == 0).sum() / len(df_ind) * 100))

    # 检查每个品牌的正负例分布
    print("\nBrand-level distribution in in-domain set:")
    print("-" * 50)
    for brand in b_ind:
        brand_data = df_ind[df_ind["brand"] == brand]
        pos_count = (brand_data["label"] == 1).sum()
        neg_count = (brand_data["label"] == 0).sum()
        print("%-20s pos: %2d, neg: %2d" % (brand, pos_count, neg_count))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全面检查数据集的重复和缺失情况
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


def check_duplicates(df: pd.DataFrame):
    """检查各种类型的重复"""
    print("\n" + "=" * 70)
    print("🔍 重复检查")
    print("=" * 70)

    # 1. ID重复
    id_dups = df["id"].duplicated().sum()
    print("\n1. ID 重复:")
    print(f"   重复数量: {id_dups}")
    if id_dups > 0:
        dup_ids = df[df["id"].duplicated(keep=False)]["id"].value_counts()
        print("   重复的ID (前10个):")
        for id_val, count in dup_ids.head(10).items():
            print(f"     - {id_val}: {count}次")

    # 2. URL重复
    url_dups = df["url_text"].duplicated().sum()
    print("\n2. URL 重复:")
    print(f"   重复数量: {url_dups}")
    if url_dups > 0:
        dup_urls = df[df["url_text"].duplicated(keep=False)]["url_text"].value_counts()
        print("   重复的URL (前5个):")
        for url, count in dup_urls.head(5).items():
            print(f"     - {url[:80]}...: {count}次")

    # 3. HTML路径重复
    html_dups = df["html_path"].duplicated().sum()
    print("\n3. HTML路径 重复:")
    print(f"   重复数量: {html_dups}")
    if html_dups > 0:
        dup_htmls = df[df["html_path"].duplicated(keep=False)][
            "html_path"
        ].value_counts()
        print("   重复的HTML路径 (前5个):")
        for path, count in dup_htmls.head(5).items():
            print(f"     - {path}: {count}次")

    # 4. IMG路径重复
    img_dups = df["img_path"].duplicated().sum()
    print("\n4. IMG路径 重复:")
    print(f"   重复数量: {img_dups}")
    if img_dups > 0:
        dup_imgs = df[df["img_path"].duplicated(keep=False)]["img_path"].value_counts()
        print("   重复的IMG路径 (前5个):")
        for path, count in dup_imgs.head(5).items():
            print(f"     - {path}: {count}次")

    # 5. 语义重复 (URL + domain + brand)
    df["semantic_key"] = (
        df["url_text"].astype(str)
        + "|"
        + df["domain"].astype(str)
        + "|"
        + df["brand"].astype(str)
    )
    semantic_dups = df["semantic_key"].duplicated().sum()
    print("\n5. 语义重复 (URL+domain+brand):")
    print(f"   重复数量: {semantic_dups}")
    if semantic_dups > 0:
        dup_semantic = df[df["semantic_key"].duplicated(keep=False)][
            ["url_text", "domain", "brand"]
        ].head(5)
        print("   重复的语义组合 (前5个):")
        for idx, row in dup_semantic.iterrows():
            print(f"     - URL: {row['url_text'][:60]}...")
            print(f"       Domain: {row['domain']}, Brand: {row['brand']}")

    # 6. 完全相同的行（所有列都相同）
    full_dups = df.duplicated().sum()
    print("\n6. 完全重复的行:")
    print(f"   重复数量: {full_dups}")

    return {
        "id": id_dups,
        "url": url_dups,
        "html_path": html_dups,
        "img_path": img_dups,
        "semantic": semantic_dups,
        "full_row": full_dups,
    }


def check_missing(df: pd.DataFrame):
    """检查缺失值"""
    print("\n" + "=" * 70)
    print("🔍 缺失值检查")
    print("=" * 70)

    total = len(df)

    for col in df.columns:
        missing = df[col].isna().sum()
        missing_pct = (missing / total) * 100

        if missing > 0:
            print(f"\n列 '{col}':")
            print(f"   缺失数量: {missing} / {total} ({missing_pct:.2f}%)")

            # 显示缺失值的样本ID
            if missing <= 10:
                missing_ids = df[df[col].isna()]["id"].tolist()
                print(f"   缺失的样本ID: {missing_ids}")

    # 检查空字符串（非NaN但为空）
    print("\n空字符串检查:")
    for col in ["url_text", "domain", "brand", "source"]:
        if col in df.columns:
            empty = (df[col] == "").sum() if df[col].dtype == "object" else 0
            if empty > 0:
                print(f"   列 '{col}': {empty} 个空字符串")


def check_path_validity(df: pd.DataFrame, sample_size: int = 100):
    """检查路径有效性"""
    print("\n" + "=" * 70)
    print("🔍 路径有效性检查")
    print("=" * 70)

    # 采样检查
    sample_df = df.sample(n=min(sample_size, len(df)), random_state=42)

    # HTML路径
    html_exists = 0
    html_missing = 0
    for path_str in sample_df["html_path"]:
        if pd.isna(path_str):
            html_missing += 1
        elif Path(path_str).exists():
            html_exists += 1
        else:
            html_missing += 1

    print(f"\nHTML路径 (采样 {len(sample_df)} 个):")
    print(f"   存在: {html_exists} ({html_exists/len(sample_df)*100:.1f}%)")
    print(f"   缺失: {html_missing} ({html_missing/len(sample_df)*100:.1f}%)")

    # IMG路径
    img_exists = 0
    img_missing = 0
    for path_str in sample_df["img_path"]:
        if pd.isna(path_str):
            img_missing += 1
        elif Path(path_str).exists():
            img_exists += 1
        else:
            img_missing += 1

    print(f"\nIMG路径 (采样 {len(sample_df)} 个):")
    print(f"   存在: {img_exists} ({img_exists/len(sample_df)*100:.1f}%)")
    print(f"   缺失: {img_missing} ({img_missing/len(sample_df)*100:.1f}%)")


def check_data_consistency(df: pd.DataFrame):
    """检查数据一致性"""
    print("\n" + "=" * 70)
    print("🔍 数据一致性检查")
    print("=" * 70)

    # 1. 标签值检查
    label_values = df["label"].unique()
    print("\n1. 标签值:")
    print(f"   唯一值: {sorted(label_values)}")
    print(f"   分布: {df['label'].value_counts().to_dict()}")
    invalid_labels = df[~df["label"].isin([0, 1])]
    if len(invalid_labels) > 0:
        print(f"   ⚠️  发现无效标签: {len(invalid_labels)} 个")

    # 2. Split值检查
    split_values = df["split"].unique()
    print("\n2. Split值:")
    print(f"   唯一值: {sorted(split_values)}")
    print(f"   分布: {df['split'].value_counts().to_dict()}")

    # 3. 时间戳格式检查
    print("\n3. 时间戳格式:")
    ts_sample = df["timestamp"].dropna().sample(n=min(5, len(df)), random_state=42)
    print("   样本 (前5个):")
    for ts in ts_sample:
        print(f"     - {ts}")

    # 检查是否有无效的时间戳
    try:
        pd.to_datetime(df["timestamp"], errors="coerce")
        invalid_ts = pd.to_datetime(df["timestamp"], errors="coerce").isna().sum()
        print(f"   无效时间戳: {invalid_ts}")
    except Exception as e:
        print(f"   时间戳解析错误: {e}")

    # 4. 品牌分布
    print("\n4. 品牌分布:")
    brand_counts = df["brand"].value_counts()
    print(f"   总品牌数: {len(brand_counts)}")
    print("   Top 10 品牌:")
    for brand, count in brand_counts.head(10).items():
        pct = count / len(df) * 100
        print(f"     - {brand}: {count} ({pct:.2f}%)")

    # 5. Domain分布
    print("\n5. Domain分布:")
    domain_counts = df["domain"].value_counts()
    print(f"   总域名数: {len(domain_counts)}")
    print("   Top 10 域名:")
    for domain, count in domain_counts.head(10).items():
        pct = count / len(df) * 100
        print(f"     - {domain}: {count} ({pct:.2f}%)")


def main():
    csv_path = Path("data/processed/master_v2.csv")

    if not csv_path.exists():
        print(f"❌ 文件不存在: {csv_path}")
        return 1

    print("=" * 70)
    print(f"📖 读取数据集: {csv_path}")
    print("=" * 70)

    df = pd.read_csv(csv_path)
    print(f"总样本数: {len(df)}")
    print(f"总列数: {len(df.columns)}")
    print(f"列名: {list(df.columns)}")

    # 执行检查
    dup_stats = check_duplicates(df)
    check_missing(df)
    check_path_validity(df, sample_size=200)
    check_data_consistency(df)

    # 总结
    print("\n" + "=" * 70)
    print("📊 检查总结")
    print("=" * 70)

    total_issues = sum(dup_stats.values())

    if total_issues == 0:
        print("\n✅ 未发现重复问题！")
    else:
        print(f"\n⚠️  发现 {total_issues} 个重复项:")
        for key, count in dup_stats.items():
            if count > 0:
                print(f"   - {key}: {count}")

    print("\n" + "=" * 70)

    return 0


if __name__ == "__main__":
    exit(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
详细分析重复项并生成清理建议
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


def analyze_url_duplicates(df: pd.DataFrame):
    """分析URL重复的详细情况"""
    print("\n" + "=" * 70)
    print("📋 URL重复详细分析")
    print("=" * 70)

    dup_urls = df[df["url_text"].duplicated(keep=False)].copy()
    dup_urls = dup_urls.sort_values("url_text")

    print(f"\n总重复URL样本数: {len(dup_urls)}")
    print(f"涉及的唯一URL数: {dup_urls['url_text'].nunique()}")

    # 按URL分组查看
    url_groups = dup_urls.groupby("url_text")

    print("\n重复URL详情 (前10个):")
    for i, (url, group) in enumerate(url_groups):
        if i >= 10:
            break
        print(f"\n{i+1}. URL: {url[:100]}...")
        print(f"   出现次数: {len(group)}")
        print(f"   标签分布: {group['label'].value_counts().to_dict()}")
        print(f"   品牌: {group['brand'].unique()[:3]}")
        print(f"   来源: {group['source'].unique()[:3]}")
        print(f"   样本ID: {list(group['id'][:3])}")

    return dup_urls


def analyze_missing_data(df: pd.DataFrame):
    """分析缺失数据"""
    print("\n" + "=" * 70)
    print("📋 缺失数据详细分析")
    print("=" * 70)

    # 问题样本：有多个缺失值的行
    problem_samples = df[
        df["url_text"].isna()
        | df["html_path"].isna()
        | df["img_path"].isna()
        | df["domain"].isna()
        | df["timestamp"].isna()
    ]

    print(f"\n问题样本总数: {len(problem_samples)}")

    if len(problem_samples) > 0:
        print("\n问题样本详情:")
        for idx, row in problem_samples.iterrows():
            print(f"\n样本ID: {row['id']}")
            print(f"  标签: {row['label']}")
            print(f"  来源: {row['source']}")
            print(f"  Split: {row['split']}")
            missing_cols = []
            if pd.isna(row["url_text"]):
                missing_cols.append("url_text")
            if pd.isna(row["html_path"]):
                missing_cols.append("html_path")
            if pd.isna(row["img_path"]):
                missing_cols.append("img_path")
            if pd.isna(row["domain"]):
                missing_cols.append("domain")
            if pd.isna(row["timestamp"]):
                missing_cols.append("timestamp")
            print(f"  缺失列: {missing_cols}")

    return problem_samples


def suggest_cleanup(
    df: pd.DataFrame, dup_urls: pd.DataFrame, problem_samples: pd.DataFrame
):
    """生成清理建议"""
    print("\n" + "=" * 70)
    print("💡 数据清理建议")
    print("=" * 70)

    print("\n1. URL重复处理建议:")
    if len(dup_urls) > 0:
        # 检查是否是合法的重复（比如相同URL但不同标签）
        url_groups = dup_urls.groupby("url_text")
        same_label_count = 0
        diff_label_count = 0

        for url, group in url_groups:
            if group["label"].nunique() == 1:
                same_label_count += 1
            else:
                diff_label_count += 1

        print(f"   - 相同URL+相同标签: {same_label_count} 个URL (应删除重复)")
        print(f"   - 相同URL+不同标签: {diff_label_count} 个URL (需人工检查)")

        if same_label_count > 0:
            print("\n   建议操作: 保留每个URL的第一个样本，删除其余重复")
            # 计算将删除的样本数
            to_remove = 0
            for url, group in url_groups:
                if group["label"].nunique() == 1:
                    to_remove += len(group) - 1
            print(f"   预计删除样本数: {to_remove}")
    else:
        print("   ✅ 无URL重复问题")

    print("\n2. 缺失数据处理建议:")
    if len(problem_samples) > 0:
        print(f"   - 发现 {len(problem_samples)} 个问题样本")
        print(f"   - 样本ID: {list(problem_samples['id'])}")
        print("   建议操作: 删除这些样本（关键字段缺失，无法使用）")
    else:
        print("   ✅ 无严重缺失问题")

    print("\n3. 路径重复处理建议:")
    html_path_dups = df["html_path"].duplicated().sum()
    img_path_dups = df["img_path"].duplicated().sum()
    if html_path_dups > 0 or img_path_dups > 0:
        print(f"   - HTML路径重复: {html_path_dups}")
        print(f"   - IMG路径重复: {img_path_dups}")
        print("   建议: 保留第一个，删除后续重复（避免数据泄露）")
    else:
        print("   ✅ 无路径重复问题（忽略少量重复是安全的）")

    print("\n4. 时间戳问题:")
    invalid_ts = pd.to_datetime(df["timestamp"], errors="coerce").isna().sum()
    if invalid_ts > 0:
        print(f"   - 发现 {invalid_ts} 个无效时间戳")
        print("   说明: 这些可能是旧数据集的时间戳格式问题")
        print("   建议: 如果不影响temporal协议，可保留；否则需要修复")

    print("\n5. 总体建议:")
    total_to_remove = len(problem_samples)
    # 计算URL重复中同标签的重复数
    if len(dup_urls) > 0:
        url_groups = dup_urls.groupby("url_text")
        for url, group in url_groups:
            if group["label"].nunique() == 1:
                total_to_remove += len(group) - 1

    print(f"   - 预计需要删除: {total_to_remove} 个样本")
    print(f"   - 清理后样本数: {len(df) - total_to_remove}")
    print(f"   - 数据质量改善: {(total_to_remove/len(df)*100):.2f}% 的问题样本将被移除")


def main():
    csv_path = Path("data/processed/master_v2.csv")

    print("=" * 70)
    print(f"📖 读取数据集: {csv_path}")
    print("=" * 70)

    df = pd.read_csv(csv_path)
    print(f"总样本数: {len(df)}")

    # 分析
    dup_urls = analyze_url_duplicates(df)
    problem_samples = analyze_missing_data(df)
    suggest_cleanup(df, dup_urls, problem_samples)

    print("\n" + "=" * 70)
    print("✅ 分析完成")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    exit(main())

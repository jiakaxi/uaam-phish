#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将数据集平衡到恰好8000个合法 + 8000个钓鱼样本
确保不重复、无缺失
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json

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
    print("=" * 70)
    print("🎯 平衡数据集到 8000 + 8000")
    print("=" * 70)

    # 读取数据
    csv_path = Path("data/processed/master_v2.csv")
    backup_path = Path("data/processed/master_v2_before_balance.csv")

    if not csv_path.exists():
        print(f"❌ 文件不存在: {csv_path}")
        return 1

    print("\n📖 读取数据集...")
    df = pd.read_csv(csv_path)
    original_count = len(df)
    print(f"   原始样本数: {original_count}")

    # 统计当前分布
    label_dist = df["label"].value_counts().sort_index()
    benign_count = label_dist[0]
    phishing_count = label_dist[1]

    print("\n当前标签分布:")
    print(f"   合法 (Label=0): {benign_count}")
    print(f"   钓鱼 (Label=1): {phishing_count}")

    target_count = 8000

    # 检查是否需要调整
    benign_diff = target_count - benign_count
    phishing_diff = target_count - phishing_count

    print(f"\n目标: 各 {target_count} 个样本")
    print(f"   合法需要: {benign_diff:+d} 个")
    print(f"   钓鱼需要: {phishing_diff:+d} 个")

    # 备份
    if benign_diff != 0 or phishing_diff != 0:
        print(f"\n📦 备份原文件到: {backup_path}")
        import shutil

        shutil.copy2(csv_path, backup_path)

    # 分离数据
    benign_df = df[df["label"] == 0].copy()
    phishing_df = df[df["label"] == 1].copy()

    # 处理合法样本
    print("\n🔧 处理合法样本...")
    if benign_diff > 0:
        print(f"   ⚠️  不足 {benign_diff} 个，无法补充（原始数据不足）")
        print(f"   当前只有 {benign_count} 个合法样本")
        print(f"   保持现有 {benign_count} 个")
        final_benign = benign_df
    elif benign_diff < 0:
        # 需要减少
        print(f"   需要减少 {-benign_diff} 个")
        print(f"   随机采样 {target_count} 个（保持品牌分布）")

        # 按品牌分层采样
        benign_brands = benign_df["brand"].value_counts()
        sampled_indices = []

        # 计算每个品牌应该采样多少
        for brand, count in benign_brands.items():
            brand_samples = benign_df[benign_df["brand"] == brand]
            # 按比例采样
            n_sample = int(count / benign_count * target_count)
            n_sample = min(n_sample, len(brand_samples))  # 不超过实际数量

            if n_sample > 0:
                sampled = brand_samples.sample(n=n_sample, random_state=42)
                sampled_indices.extend(sampled.index.tolist())

        # 如果还不够，随机补充
        if len(sampled_indices) < target_count:
            remaining = target_count - len(sampled_indices)
            remaining_candidates = benign_df.loc[~benign_df.index.isin(sampled_indices)]
            additional = remaining_candidates.sample(n=remaining, random_state=42)
            sampled_indices.extend(additional.index.tolist())

        # 如果超了，随机减少
        if len(sampled_indices) > target_count:
            np.random.seed(42)
            sampled_indices = np.random.choice(
                sampled_indices, target_count, replace=False
            ).tolist()

        final_benign = benign_df.loc[sampled_indices]
        print(f"   ✅ 采样完成: {len(final_benign)} 个")
    else:
        print(f"   ✅ 已经是 {target_count} 个，无需调整")
        final_benign = benign_df

    # 处理钓鱼样本
    print("\n🔧 处理钓鱼样本...")
    if phishing_diff > 0:
        print(f"   ⚠️  不足 {phishing_diff} 个，无法补充（原始数据不足）")
        print(f"   当前只有 {phishing_count} 个钓鱼样本")
        print(f"   保持现有 {phishing_count} 个")
        final_phishing = phishing_df
    elif phishing_diff < 0:
        # 需要减少
        print(f"   需要减少 {-phishing_diff} 个")
        print(f"   随机采样 {target_count} 个（保持品牌分布）")

        # 按品牌分层采样
        phishing_brands = phishing_df["brand"].value_counts()
        sampled_indices = []

        # 计算每个品牌应该采样多少
        for brand, count in phishing_brands.items():
            brand_samples = phishing_df[phishing_df["brand"] == brand]
            # 按比例采样
            n_sample = int(count / phishing_count * target_count)
            n_sample = min(n_sample, len(brand_samples))

            if n_sample > 0:
                sampled = brand_samples.sample(n=n_sample, random_state=42)
                sampled_indices.extend(sampled.index.tolist())

        # 如果还不够，随机补充
        if len(sampled_indices) < target_count:
            remaining = target_count - len(sampled_indices)
            remaining_candidates = phishing_df.loc[
                ~phishing_df.index.isin(sampled_indices)
            ]
            additional = remaining_candidates.sample(n=remaining, random_state=42)
            sampled_indices.extend(additional.index.tolist())

        # 如果超了，随机减少
        if len(sampled_indices) > target_count:
            np.random.seed(42)
            sampled_indices = np.random.choice(
                sampled_indices, target_count, replace=False
            ).tolist()

        final_phishing = phishing_df.loc[sampled_indices]
        print(f"   ✅ 采样完成: {len(final_phishing)} 个")
    else:
        print(f"   ✅ 已经是 {target_count} 个，无需调整")
        final_phishing = phishing_df

    # 合并
    print("\n🔀 合并数据...")
    final_df = pd.concat([final_benign, final_phishing], ignore_index=True)

    # 打乱顺序
    print("   打乱顺序...")
    final_df = final_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    # 验证
    print("\n🔍 验证结果...")
    final_label_dist = final_df["label"].value_counts().sort_index()
    final_benign = final_label_dist[0]
    final_phishing = final_label_dist[1]

    print("   最终标签分布:")
    print(f"     合法 (Label=0): {final_benign}")
    print(f"     钓鱼 (Label=1): {final_phishing}")
    print(f"     总计: {len(final_df)}")

    # 检查重复
    id_duplicates = final_df["id"].duplicated().sum()
    url_duplicates = final_df["url_text"].duplicated().sum()

    print("\n   重复检查:")
    print(f"     ID重复: {id_duplicates}")
    print(f"     URL重复: {url_duplicates}")

    if id_duplicates > 0 or url_duplicates > 0:
        print("   ⚠️  发现重复，需要处理")
        return 1

    # 检查缺失
    critical_cols = [
        "url_text",
        "html_path",
        "img_path",
        "domain",
        "timestamp",
        "brand",
    ]
    missing_check = {}
    for col in critical_cols:
        missing = final_df[col].isna().sum()
        missing_check[col] = missing
        if missing > 0:
            print(f"     {col}: {missing} 个缺失")

    total_missing = sum(missing_check.values())
    print(f"\n   缺失检查: {total_missing} 个缺失值")

    if total_missing > 0:
        print("   ⚠️  发现缺失值")
        return 1

    # 保存
    print("\n💾 保存结果...")
    final_df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"   ✅ 保存到: {csv_path}")

    # 保存日志
    log_path = Path("data/processed/balance_8k_log.json")
    balance_log = {
        "timestamp": datetime.now().isoformat(),
        "original_count": int(original_count),
        "original_benign": int(benign_count),
        "original_phishing": int(phishing_count),
        "final_count": int(len(final_df)),
        "final_benign": int(final_benign),
        "final_phishing": int(final_phishing),
        "benign_diff": int(benign_diff),
        "phishing_diff": int(phishing_diff),
        "duplicates": {"id": int(id_duplicates), "url": int(url_duplicates)},
        "missing": {k: int(v) for k, v in missing_check.items()},
    }

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(balance_log, f, indent=2, ensure_ascii=False)
    print(f"   ✅ 日志保存到: {log_path}")

    # 最终统计
    print("\n" + "=" * 70)
    print("📊 平衡完成统计")
    print("=" * 70)

    print("\n样本数变化:")
    print(f"   原始: {original_count} ({benign_count} 合法 + {phishing_count} 钓鱼)")
    print(f"   最终: {len(final_df)} ({final_benign} 合法 + {final_phishing} 钓鱼)")
    print(f"   变化: {len(final_df) - original_count:+d}")

    print("\n数据质量:")
    print(f"   ✅ ID唯一性: {len(final_df['id'].unique())}/{len(final_df)}")
    print(f"   ✅ URL唯一性: {len(final_df['url_text'].unique())}/{len(final_df)}")
    print(f"   ✅ 无重复: {id_duplicates == 0 and url_duplicates == 0}")
    print(f"   ✅ 无缺失: {total_missing == 0}")

    # 品牌多样性
    brand_count = final_df["brand"].nunique()
    top1_ratio = final_df["brand"].value_counts().iloc[0] / len(final_df) * 100
    print("\n品牌多样性:")
    print(f"   总品牌数: {brand_count}")
    print(f"   Top1占比: {top1_ratio:.2f}%")

    print("\n" + "=" * 70)
    print("✅ 数据集已平衡到 8000 + 8000！")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    exit(main())

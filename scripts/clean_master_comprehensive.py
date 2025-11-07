#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
综合清理 master_v2.csv 数据集
按照数据质量报告的6个问题依次处理
"""

import sys
import pandas as pd
from pathlib import Path
import hashlib
import json
from datetime import datetime
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import re

# Handle Windows console encoding
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        import codecs

        sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "strict")
        sys.stderr = codecs.getwriter("utf-8")(sys.stderr.buffer, "strict")


def compute_file_hash(file_path: Path) -> str:
    """计算文件SHA1哈希"""
    if not file_path.exists():
        return None
    try:
        sha1 = hashlib.sha1()
        with open(file_path, "rb") as f:
            while chunk := f.read(8192):
                sha1.update(chunk)
        return sha1.hexdigest()
    except Exception:
        return None


def compute_hashes_parallel(paths, max_workers=4):
    """并行计算哈希"""
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        hashes = list(
            tqdm(
                executor.map(compute_file_hash, paths),
                total=len(paths),
                desc="Computing hashes",
            )
        )
    return hashes


def fix_timestamp(ts_str):
    """
    修复时间戳格式，尝试多种解析方式
    """
    if pd.isna(ts_str):
        return None

    ts_str = str(ts_str).strip()

    # 如果已经是标准ISO格式，直接返回
    if re.match(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", ts_str):
        # 确保有时区信息
        if not ts_str.endswith("Z") and "+" not in ts_str and not ts_str.endswith(")"):
            ts_str += "Z"
        return ts_str

    # 尝试多种格式解析
    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%Y%m%d",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
    ]

    for fmt in formats:
        try:
            dt = datetime.strptime(ts_str, fmt)
            return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        except Exception:  # noqa: E722
            continue

    # 尝试pandas解析
    try:
        dt = pd.to_datetime(ts_str)
        if pd.notna(dt):
            return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:  # noqa: E722
        pass

    # 无法解析，返回原值
    return ts_str


def main():
    print("=" * 70)
    print("🧹 综合数据清理 - master_v2.csv")
    print("=" * 70)

    # 读取数据
    csv_path = Path("data/processed/master_v2.csv")
    backup_path = Path("data/processed/master_v2_backup.csv")
    output_path = Path("data/processed/master_v2.csv")

    if not csv_path.exists():
        print(f"❌ 文件不存在: {csv_path}")
        return 1

    # 备份原文件
    print("\n📦 备份原文件...")
    import shutil

    shutil.copy2(csv_path, backup_path)
    print(f"   备份保存到: {backup_path}")

    print("\n📖 读取数据集...")
    df = pd.read_csv(csv_path)
    original_count = len(df)
    print(f"   原始样本数: {original_count}")

    # 记录删除的样本
    removed_samples = {
        "url_duplicates": [],
        "missing_critical": [],
        "path_duplicates": [],
        "metadata_missing": [],
    }

    # ========================================================================
    # 问题1: 删除URL重复（保留第一个）
    # ========================================================================
    print("\n" + "=" * 70)
    print("🔧 问题1: 删除URL重复")
    print("=" * 70)

    # 找出同URL+同标签的重复
    url_dup_mask = df.duplicated(subset=["url_text", "label"], keep="first")
    url_dups = df[url_dup_mask]

    print(f"   发现 {len(url_dups)} 个URL重复样本")
    if len(url_dups) > 0:
        removed_samples["url_duplicates"] = url_dups["id"].tolist()
        print(f"   删除样本ID (前10个): {url_dups['id'].head(10).tolist()}")
        df = df[~url_dup_mask]
        print(f"   ✅ 删除完成，剩余: {len(df)} 个样本")

    # ========================================================================
    # 问题2: 删除关键字段缺失的样本
    # ========================================================================
    print("\n" + "=" * 70)
    print("🔧 问题2: 删除关键字段缺失样本")
    print("=" * 70)

    critical_fields = ["url_text", "html_path", "img_path", "domain", "timestamp"]
    missing_mask = df[critical_fields].isna().any(axis=1)
    missing_samples = df[missing_mask]

    print(f"   发现 {len(missing_samples)} 个关键字段缺失样本")
    if len(missing_samples) > 0:
        removed_samples["missing_critical"] = missing_samples["id"].tolist()
        print(f"   删除样本ID: {missing_samples['id'].tolist()}")
        df = df[~missing_mask]
        print(f"   ✅ 删除完成，剩余: {len(df)} 个样本")

    # ========================================================================
    # 问题3: 删除路径重复（保留第一个）
    # ========================================================================
    print("\n" + "=" * 70)
    print("🔧 问题3: 删除路径重复")
    print("=" * 70)

    # HTML路径重复
    html_dup_mask = df.duplicated(subset=["html_path"], keep="first")

    # IMG路径重复
    img_dup_mask = df.duplicated(subset=["img_path"], keep="first")

    path_dup_mask = html_dup_mask | img_dup_mask
    path_dups = df[path_dup_mask]

    print(f"   发现 {len(path_dups)} 个路径重复样本")
    print(f"     - HTML路径重复: {html_dup_mask.sum()}")
    print(f"     - IMG路径重复: {img_dup_mask.sum()}")

    if len(path_dups) > 0:
        removed_samples["path_duplicates"] = path_dups["id"].tolist()
        print(f"   删除样本ID: {path_dups['id'].tolist()}")
        df = df[~path_dup_mask]
        print(f"   ✅ 删除完成，剩余: {len(df)} 个样本")

    # ========================================================================
    # 问题4: 修复时间戳格式
    # ========================================================================
    print("\n" + "=" * 70)
    print("🔧 问题4: 修复时间戳格式")
    print("=" * 70)

    print("   开始修复时间戳...")
    df["timestamp_original"] = df["timestamp"].copy()

    # 应用修复函数
    tqdm.pandas(desc="Fixing timestamps")
    df["timestamp"] = df["timestamp"].progress_apply(fix_timestamp)

    # 验证修复结果
    valid_ts = pd.to_datetime(df["timestamp"], errors="coerce").notna().sum()
    invalid_ts = len(df) - valid_ts

    print("   修复后统计:")
    print(f"     - 有效时间戳: {valid_ts} ({valid_ts/len(df)*100:.1f}%)")
    print(f"     - 无效时间戳: {invalid_ts} ({invalid_ts/len(df)*100:.1f}%)")

    if invalid_ts > 0:
        print(f"   ⚠️  仍有 {invalid_ts} 个时间戳无法解析")
        invalid_examples = df[pd.to_datetime(df["timestamp"], errors="coerce").isna()][
            "timestamp"
        ].head(5)
        print(f"   示例: {list(invalid_examples)}")
    else:
        print("   ✅ 所有时间戳格式正确")

    # ========================================================================
    # 问题5: 删除元数据列缺失的样本
    # ========================================================================
    print("\n" + "=" * 70)
    print("🔧 问题5: 删除元数据列完全缺失的样本")
    print("=" * 70)

    metadata_fields = ["domain_source", "timestamp_source", "folder"]
    metadata_missing_mask = df[metadata_fields].isna().all(axis=1)
    metadata_missing = df[metadata_missing_mask]

    print(f"   发现 {len(metadata_missing)} 个元数据完全缺失样本")
    if len(metadata_missing) > 0:
        removed_samples["metadata_missing"] = metadata_missing["id"].tolist()
        print(f"   删除样本数: {len(metadata_missing)}")
        print("   (这些是旧数据集样本，缺少新构建脚本添加的元数据)")
        df = df[~metadata_missing_mask]
        print(f"   ✅ 删除完成，剩余: {len(df)} 个样本")

    # ========================================================================
    # 问题6: 重新计算哈希
    # ========================================================================
    print("\n" + "=" * 70)
    print("🔧 问题6: 重新计算文件哈希")
    print("=" * 70)

    print(f"   准备计算 {len(df)} 个样本的哈希...")

    # HTML哈希
    print("\n   计算HTML文件哈希...")
    html_paths = [Path(p) for p in df["html_path"]]
    df["html_sha1"] = compute_hashes_parallel(html_paths, max_workers=8)
    html_success = df["html_sha1"].notna().sum()
    print(f"   ✅ 成功: {html_success}/{len(df)} ({html_success/len(df)*100:.1f}%)")

    # IMG哈希
    print("\n   计算IMG文件哈希...")
    img_paths = [Path(p) for p in df["img_path"]]
    df["img_sha1"] = compute_hashes_parallel(img_paths, max_workers=8)
    img_success = df["img_sha1"].notna().sum()
    print(f"   ✅ 成功: {img_success}/{len(df)} ({img_success/len(df)*100:.1f}%)")

    # ========================================================================
    # 保存清理后的数据
    # ========================================================================
    print("\n" + "=" * 70)
    print("💾 保存清理后的数据")
    print("=" * 70)

    # 重新排序列（把原始时间戳移到最后）
    cols = [c for c in df.columns if c != "timestamp_original"] + ["timestamp_original"]
    df = df[cols]

    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"   ✅ 保存到: {output_path}")
    print(f"   最终样本数: {len(df)}")

    # 保存删除记录
    removed_log_path = Path("data/processed/removed_samples_log.json")
    removed_summary = {
        "timestamp": datetime.now().isoformat(),
        "original_count": original_count,
        "final_count": len(df),
        "removed_count": original_count - len(df),
        "removed_by_reason": {
            "url_duplicates": len(removed_samples["url_duplicates"]),
            "missing_critical": len(removed_samples["missing_critical"]),
            "path_duplicates": len(removed_samples["path_duplicates"]),
            "metadata_missing": len(removed_samples["metadata_missing"]),
        },
        "removed_sample_ids": removed_samples,
    }

    with open(removed_log_path, "w", encoding="utf-8") as f:
        json.dump(removed_summary, f, indent=2, ensure_ascii=False)
    print(f"   ✅ 删除记录保存到: {removed_log_path}")

    # ========================================================================
    # 最终统计
    # ========================================================================
    print("\n" + "=" * 70)
    print("📊 清理完成统计")
    print("=" * 70)

    print("\n样本数变化:")
    print(f"   原始: {original_count}")
    print(f"   最终: {len(df)}")
    print(
        f"   删除: {original_count - len(df)} ({(original_count - len(df))/original_count*100:.2f}%)"
    )

    print("\n删除原因分解:")
    for reason, ids in removed_samples.items():
        if len(ids) > 0:
            print(f"   - {reason}: {len(ids)} 个样本")

    print("\n标签分布:")
    label_dist = df["label"].value_counts()
    for label, count in label_dist.items():
        label_name = "phishing" if label == 1 else "benign"
        print(f"   - {label_name}: {count} ({count/len(df)*100:.1f}%)")

    print("\n数据质量:")
    print(f"   - URL唯一性: {df['url_text'].nunique()}/{len(df)}")
    print(
        f"   - HTML哈希完整性: {df['html_sha1'].notna().sum()}/{len(df)} ({df['html_sha1'].notna().sum()/len(df)*100:.1f}%)"
    )
    print(
        f"   - IMG哈希完整性: {df['img_sha1'].notna().sum()}/{len(df)} ({df['img_sha1'].notna().sum()/len(df)*100:.1f}%)"
    )
    print(
        f"   - 时间戳有效性: {pd.to_datetime(df['timestamp'], errors='coerce').notna().sum()}/{len(df)} ({pd.to_datetime(df['timestamp'], errors='coerce').notna().sum()/len(df)*100:.1f}%)"
    )

    print("\n" + "=" * 70)
    print("✅ 数据清理完成！")
    print("=" * 70)
    print("\n📁 生成的文件:")
    print(f"   - 清理后数据: {output_path}")
    print(f"   - 原始备份: {backup_path}")
    print(f"   - 删除记录: {removed_log_path}")

    return 0


if __name__ == "__main__":
    exit(main())

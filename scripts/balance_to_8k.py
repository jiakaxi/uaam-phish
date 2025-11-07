#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将数据集平衡到8000+8000样本
从30k数据集中随机抽取补充样本
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


def main():
    print("=" * 70)
    print("🔄 平衡数据集到 8000 + 8000")
    print("=" * 70)

    # 读取当前清理后的数据集
    current_csv = Path("data/processed/master_v2.csv")
    df_current = pd.read_csv(current_csv)

    print("\n📖 当前数据集:")
    print(f"   总样本数: {len(df_current):,}")

    current_phish = len(df_current[df_current["label"] == 1])
    current_benign = len(df_current[df_current["label"] == 0])

    print(f"   钓鱼样本: {current_phish:,}")
    print(f"   合法样本: {current_benign:,}")

    # 计算需要补充的数量
    target_phish = 8000
    target_benign = 8000

    need_phish = max(0, target_phish - current_phish)
    need_benign = max(0, target_benign - current_benign)

    print("\n📊 需要补充:")
    print(f"   钓鱼样本: {need_phish:,}")
    print(f"   合法样本: {need_benign:,}")

    if need_phish == 0 and need_benign == 0:
        print("\n✅ 数据集已经平衡到8000+8000，无需补充")
        return 0

    # 如果当前样本超过8000，需要减少
    if current_phish > target_phish or current_benign > target_benign:
        print("\n⚠️  当前某个类别超过8000，需要减少样本")

        # 随机采样到8000
        df_phish = df_current[df_current["label"] == 1].sample(
            n=min(target_phish, current_phish), random_state=42
        )
        df_benign = df_current[df_current["label"] == 0].sample(
            n=min(target_benign, current_benign), random_state=42
        )

        df_balanced = pd.concat([df_phish, df_benign], ignore_index=True)
        df_balanced = df_balanced.sample(frac=1.0, random_state=42).reset_index(
            drop=True
        )

        # 保存
        backup_path = Path("data/processed/master_v2_before_balance.csv")
        df_current.to_csv(backup_path, index=False, encoding="utf-8")
        print(f"\n📦 原数据备份到: {backup_path}")

        df_balanced.to_csv(current_csv, index=False, encoding="utf-8")

        print("\n✅ 平衡完成:")
        print(f"   总样本数: {len(df_balanced):,}")
        print(f"   钓鱼样本: {len(df_balanced[df_balanced['label']==1]):,}")
        print(f"   合法样本: {len(df_balanced[df_balanced['label']==0]):,}")

        return 0

    # 从30k数据集中补充
    print("\n🔍 从30k数据集中查找补充样本...")

    # 获取已使用的样本路径（用于去重）
    existing_html_paths = set(df_current["html_path"].dropna())
    existing_img_paths = set(df_current["img_path"].dropna())
    existing_urls = set(df_current["url_text"].dropna())

    print(
        f"   已使用路径数: HTML={len(existing_html_paths)}, IMG={len(existing_img_paths)}, URL={len(existing_urls)}"
    )

    # 扫描30k数据集找可用样本
    phish_root = Path(r"D:\one\phish_sample_30k")
    benign_root = Path(r"D:\one\benign_sample_30k")

    available_samples = []

    # 扫描钓鱼数据集
    if need_phish > 0 and phish_root.exists():
        print("\n   扫描钓鱼数据集...")
        folders = list(phish_root.iterdir())
        print(f"   总文件夹数: {len(folders)}")

        for folder in folders[:50000]:  # 限制扫描数量
            if not folder.is_dir():
                continue

            html_file = folder / "html.txt"
            if not html_file.exists():
                html_file = folder / "html.html"

            if not html_file.exists():
                continue

            # 检查是否已被使用
            if str(html_file) in existing_html_paths:
                continue

            # 检查URL（从info.txt读取）
            info_file = folder / "info.txt"
            if info_file.exists():
                try:
                    info_text = info_file.read_text(encoding="utf-8", errors="ignore")
                    if "url" in info_text.lower():
                        # 简单提取URL
                        import re

                        urls = re.findall(r'https?://[^\s\'"]+', info_text)
                        if urls and urls[0] in existing_urls:
                            continue
                except Exception:  # noqa: E722
                    pass

            available_samples.append(
                {"folder": folder, "html_path": html_file, "label": 1}
            )

            if len([s for s in available_samples if s["label"] == 1]) >= need_phish * 2:
                break

    # 扫描合法数据集
    if need_benign > 0 and benign_root.exists():
        print("\n   扫描合法数据集...")
        folders = list(benign_root.iterdir())
        print(f"   总文件夹数: {len(folders)}")

        for folder in folders[:50000]:
            if not folder.is_dir():
                continue

            html_file = folder / "html.txt"
            if not html_file.exists():
                html_file = folder / "html.html"

            if not html_file.exists():
                continue

            # 检查是否已被使用
            if str(html_file) in existing_html_paths:
                continue

            # 检查URL
            info_file = folder / "info.txt"
            if info_file.exists():
                try:
                    info_text = info_file.read_text(encoding="utf-8", errors="ignore")
                    if "http" in info_text.lower():
                        import re

                        urls = re.findall(r'https?://[^\s\'"]+', info_text)
                        if urls and urls[0] in existing_urls:
                            continue
                except Exception:  # noqa: E722
                    pass

            available_samples.append(
                {"folder": folder, "html_path": html_file, "label": 0}
            )

            if (
                len([s for s in available_samples if s["label"] == 0])
                >= need_benign * 2
            ):
                break

    # 统计可用样本
    available_phish = len([s for s in available_samples if s["label"] == 1])
    available_benign = len([s for s in available_samples if s["label"] == 0])

    print("\n📊 找到可用样本:")
    print(f"   钓鱼: {available_phish:,} (需要 {need_phish:,})")
    print(f"   合法: {available_benign:,} (需要 {need_benign:,})")

    if available_phish < need_phish:
        print(f"\n❌ 错误: 钓鱼样本不足，缺少 {need_phish - available_phish} 个")
        print("   建议: 降低目标数量或检查30k数据集")
        return 1

    if available_benign < need_benign:
        print(f"\n❌ 错误: 合法样本不足，缺少 {need_benign - available_benign} 个")
        print("   建议: 降低目标数量或检查30k数据集")
        return 1

    print("\n✅ 可用样本充足，开始补充...")

    # 建议用户使用build_from_30k.py脚本
    print("\n" + "=" * 70)
    print("💡 建议使用 build_from_30k.py 脚本进行补充")
    print("=" * 70)
    print("\n该脚本会:")
    print("  1. 自动从30k数据集中采样")
    print("  2. 执行完整的去重检查")
    print("  3. 应用品牌约束")
    print("  4. 计算文件哈希")
    print("  5. 追加到现有数据集")

    print("\n推荐命令:")
    print("python scripts/build_from_30k.py \\")
    print('  --phish_root "D:\\one\\phish_sample_30k" \\')
    print('  --benign_root "D:\\one\\benign_sample_30k" \\')
    print(f"  --k_each {target_phish} \\")
    print("  --master_csv data/processed/master_v2.csv \\")
    print("  --append \\")
    print("  --brand_alias resources/brand_alias.yaml \\")
    print("  --seed 42")

    print("\n注意: 使用 --append 模式会自动去重并补充到目标数量")

    return 0


if __name__ == "__main__":
    exit(main())

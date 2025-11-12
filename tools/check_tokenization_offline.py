#!/usr/bin/env python
"""检查tokenization是否完全离线化"""

import pandas as pd
from pathlib import Path


def check_cache_status():
    """检查缓存状态"""
    print("=" * 70)
    print("检查Tokenization离线化状态")
    print("=" * 70)

    splits = {
        "train": "workspace/data/splits/iid/train_cached.csv",
        "val": "workspace/data/splits/iid/val_cached.csv",
        "test": "workspace/data/splits/iid/test_cached.csv",
    }

    preprocessed_dirs = {
        "train": "workspace/data/preprocessed/iid/train",
        "val": "workspace/data/preprocessed/iid/val",
        "test": "workspace/data/preprocessed/iid/test",
    }

    all_offline = True

    for split_name, csv_path in splits.items():
        print(f"\n{split_name.upper()} Split:")
        print(f"  CSV路径: {csv_path}")

        if not Path(csv_path).exists():
            print("  [X] CSV文件不存在")
            all_offline = False
            continue

        df = pd.read_csv(csv_path)
        print(f"  [OK] CSV文件存在，包含 {len(df)} 个样本")

        # 检查CSV列
        required_cols = ["html_tokens_path", "url_tokens_path", "img_path_cached"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"  [X] 缺少列: {missing_cols}")
            all_offline = False
            continue

        # 检查CSV中的缓存路径
        html_cached = df["html_tokens_path"].notna().sum()
        url_cached = df["url_tokens_path"].notna().sum()
        img_cached = df["img_path_cached"].notna().sum()

        print(
            f"  HTML tokens路径: {html_cached}/{len(df)} ({html_cached/len(df)*100:.1f}%)"
        )
        print(
            f"  URL tokens路径: {url_cached}/{len(df)} ({url_cached/len(df)*100:.1f}%)"
        )
        print(f"  图像缓存路径: {img_cached}/{len(df)} ({img_cached/len(df)*100:.1f}%)")

        # 检查预处理目录
        preprocessed_dir = Path(preprocessed_dirs[split_name])
        if preprocessed_dir.exists():
            html_files = len(list(preprocessed_dir.glob("*_html.pt")))
            url_files = len(list(preprocessed_dir.glob("*_url.pt")))
            img_files = len(list(preprocessed_dir.glob("*_img_224.jpg")))

            print(f"  预处理目录: {preprocessed_dir}")
            print(f"    HTML缓存文件: {html_files}")
            print(f"    URL缓存文件: {url_files}")
            print(f"    图像缓存文件: {img_files}")

            # 检查覆盖率
            if html_cached < len(df) * 0.95 and html_files < len(df) * 0.95:
                print("  [WARNING] HTML缓存覆盖率不足95%")
                all_offline = False
            if url_cached < len(df) * 0.95 and url_files < len(df) * 0.95:
                print("  [WARNING] URL缓存覆盖率不足95%")
                all_offline = False
            if img_cached < len(df) * 0.95 and img_files < len(df) * 0.95:
                print("  [WARNING] 图像缓存覆盖率不足95%")
                all_offline = False
        else:
            print(f"  [WARNING] 预处理目录不存在: {preprocessed_dir}")
            if html_cached < len(df) * 0.95:
                print("  [WARNING] HTML缓存覆盖率不足95%")
                all_offline = False

    print("\n" + "=" * 70)
    if all_offline:
        print("[OK] Tokenization已完全离线化，可以使用多进程 (num_workers > 0)")
    else:
        print("[WARNING] Tokenization未完全离线化，建议使用单进程 (num_workers = 0)")
    print("=" * 70)

    return all_offline


if __name__ == "__main__":
    check_cache_status()

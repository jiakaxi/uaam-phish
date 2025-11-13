#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复 split CSV 文件中的图像路径，添加完整的文件路径列。

根据 protocol 和 split，将 img_path_cached（仅文件名）扩展为完整路径。
"""
import pandas as pd
from pathlib import Path
import os
import sys

# 设置输出编码
if sys.platform == "win32":
    import io

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


def fix_image_paths_for_split(csv_path: Path, preprocessed_dir: Path) -> bool:
    """
    为给定的 split CSV 文件添加完整图像路径。

    Args:
        csv_path: CSV 文件路径
        preprocessed_dir: 对应的预处理目录（如 workspace/data/preprocessed/iid/test）

    Returns:
        bool: 是否成功修复
    """
    print(f"\n处理: {csv_path}")

    if not csv_path.exists():
        print("  [X] 文件不存在")
        return False

    # 读取 CSV
    df = pd.read_csv(csv_path)
    print(f"  [INFO] 读取 {len(df)} 行数据")

    # 检查必要的列
    if "img_path_cached" not in df.columns:
        print("  [WARN] 缺少 img_path_cached 列")
        return False

    # 检查 img_path_cached 是否为空或只是文件名
    sample_path = df["img_path_cached"].iloc[0] if len(df) > 0 else ""
    print(f"  [CHECK] 示例路径: {sample_path}")

    # 构建完整路径
    def build_full_path(row):
        """根据 img_path_cached 构建完整路径"""
        filename = row["img_path_cached"]
        if pd.isna(filename) or filename == "":
            return None

        # 如果已经是完整路径，跳过
        if os.path.isabs(filename):
            full_path = Path(filename)
            if full_path.exists():
                return str(full_path)

        # 构建新的完整路径
        full_path = preprocessed_dir / filename
        return str(full_path.resolve())

    # 添加或更新 img_path_full 列
    df["img_path_full"] = df.apply(build_full_path, axis=1)

    # 验证路径
    valid_paths = 0
    missing_paths = []
    for idx, path in enumerate(df["img_path_full"]):
        if path and Path(path).exists():
            valid_paths += 1
        else:
            if len(missing_paths) < 5:  # 只记录前5个
                missing_paths.append(f"  行{idx}: {path}")

    print(f"  [OK] 有效路径: {valid_paths}/{len(df)}")
    if missing_paths:
        print("  [WARN] 缺失文件示例:")
        for mp in missing_paths:
            print(mp)

    # 保存回 CSV
    backup_path = csv_path.with_suffix(".csv.bak")
    if not backup_path.exists():
        print(f"  [BACKUP] 创建备份: {backup_path.name}")
        df_original = pd.read_csv(csv_path)
        df_original.to_csv(backup_path, index=False)

    df.to_csv(csv_path, index=False)
    print("  [OK] 已保存更新的 CSV")

    return True


def main():
    """主函数：处理所有 protocol 的 split CSV 文件"""
    workspace_root = Path(__file__).parent
    splits_dir = workspace_root / "workspace" / "data" / "splits"
    preprocessed_root = workspace_root / "workspace" / "data" / "preprocessed"

    print("=" * 60)
    print("修复图像路径工具")
    print("=" * 60)

    # 定义需要处理的文件映射
    protocols = {
        "iid": [
            ("train_cached.csv", "train"),
            ("val_cached.csv", "val"),
            ("test_cached.csv", "test"),
        ],
        "brandood": [
            ("train_cached.csv", "train"),
            ("val_cached.csv", "val"),
            ("test_id_cached.csv", "test_id"),
            ("test_ood_cached.csv", "test_ood"),
        ],
    }

    success_count = 0
    total_count = 0

    for protocol, splits in protocols.items():
        print(f"\n{'='*60}")
        print(f"Protocol: {protocol}")
        print(f"{'='*60}")

        protocol_dir = splits_dir / protocol
        preprocessed_protocol = preprocessed_root / protocol

        if not protocol_dir.exists():
            print(f"  [WARN] 目录不存在: {protocol_dir}")
            continue

        for csv_filename, split_name in splits:
            total_count += 1
            csv_path = protocol_dir / csv_filename
            preprocessed_dir = preprocessed_protocol / split_name

            if not preprocessed_dir.exists():
                print(f"\n  [WARN] 预处理目录不存在: {preprocessed_dir}")
                continue

            if fix_image_paths_for_split(csv_path, preprocessed_dir):
                success_count += 1

    # 总结
    print("\n" + "=" * 60)
    print("处理总结")
    print("=" * 60)
    print(f"  [OK] 成功: {success_count}/{total_count}")
    print(f"  [X] 失败: {total_count - success_count}/{total_count}")

    if success_count == total_count:
        print("\n所有文件处理完成！")
    else:
        print("\n部分文件处理失败，请检查上述错误信息。")


if __name__ == "__main__":
    main()

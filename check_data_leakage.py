#!/usr/bin/env python
"""检查IID数据分割是否存在数据泄漏"""

import pandas as pd
from pathlib import Path


def check_leakage():
    """检查训练集、验证集和测试集之间是否有重复样本"""

    # 读取数据文件
    train_path = Path("workspace/data/splits/iid/train.csv")
    val_path = Path("workspace/data/splits/iid/val.csv")
    test_path = Path("workspace/data/splits/iid/test.csv")

    if not train_path.exists():
        print(f"[ERROR] 文件不存在: {train_path}")
        return

    if not val_path.exists():
        print(f"[ERROR] 文件不存在: {val_path}")
        return

    if not test_path.exists():
        print(f"[ERROR] 文件不存在: {test_path}")
        return

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    print("=" * 60)
    print("IID数据分割泄漏检查")
    print("=" * 60)

    print("\n数据集大小:")
    print(f"  训练集: {len(train_df)} 样本")
    print(f"  验证集: {len(val_df)} 样本")
    print(f"  测试集: {len(test_df)} 样本")

    # 检查ID列
    if "id" not in train_df.columns:
        print("\n[WARNING] 训练集中没有'id'列，尝试使用其他标识符")
        # 尝试使用其他可能的标识列
        if "url_text" in train_df.columns:
            train_ids = set(train_df["url_text"])
            val_ids = set(val_df["url_text"])
            test_ids = set(test_df["url_text"])
            id_col = "url_text"
        else:
            print("[ERROR] 无法找到合适的标识列")
            return
    else:
        train_ids = set(train_df["id"])
        val_ids = set(val_df["id"])
        test_ids = set(test_df["id"])
        id_col = "id"

    print(f"\n使用标识列: {id_col}")
    print(f"  训练集唯一ID数: {len(train_ids)}")
    print(f"  验证集唯一ID数: {len(val_ids)}")
    print(f"  测试集唯一ID数: {len(test_ids)}")

    # 检查重叠
    train_val_overlap = train_ids & val_ids
    train_test_overlap = train_ids & test_ids
    val_test_overlap = val_ids & test_ids

    print("\n重叠检查:")
    print(f"  训练-验证重叠: {len(train_val_overlap)} 个样本")
    print(f"  训练-测试重叠: {len(train_test_overlap)} 个样本")
    print(f"  验证-测试重叠: {len(val_test_overlap)} 个样本")

    # 显示重叠的示例
    if train_val_overlap:
        print("\n[WARNING] 训练-验证重叠示例 (前5个):")
        for i, sample_id in enumerate(list(train_val_overlap)[:5]):
            print(f"    {i+1}. {sample_id}")

    if train_test_overlap:
        print("\n[WARNING] 训练-测试重叠示例 (前5个):")
        for i, sample_id in enumerate(list(train_test_overlap)[:5]):
            print(f"    {i+1}. {sample_id}")

    if val_test_overlap:
        print("\n[WARNING] 验证-测试重叠示例 (前5个):")
        for i, sample_id in enumerate(list(val_test_overlap)[:5]):
            print(f"    {i+1}. {sample_id}")

    # 总结
    total_overlap = (
        len(train_val_overlap) + len(train_test_overlap) + len(val_test_overlap)
    )

    print("\n" + "=" * 60)
    if total_overlap == 0:
        print("[SUCCESS] 无数据泄漏 - 所有数据集之间没有重复样本")
    else:
        print(f"[ERROR] 发现数据泄漏！总共 {total_overlap} 个重复样本")
        print("\n建议:")
        print("  1. 检查数据分割脚本")
        print("  2. 重新生成数据分割")
        print("  3. 确保使用正确的随机种子和分层策略")
    print("=" * 60)

    # 检查标签分布
    print("\n标签分布检查:")
    print(
        f"  训练集: {train_df['label'].value_counts().to_dict() if 'label' in train_df.columns else '无label列'}"
    )
    print(
        f"  验证集: {val_df['label'].value_counts().to_dict() if 'label' in val_df.columns else '无label列'}"
    )
    print(
        f"  测试集: {test_df['label'].value_counts().to_dict() if 'label' in test_df.columns else '无label列'}"
    )


def check_brandood_leakage():
    """检查Brand-OOD数据分割是否存在数据泄漏"""

    # 读取数据文件
    train_path = Path("workspace/data/splits/brandood/train.csv")
    val_path = Path("workspace/data/splits/brandood/val.csv")
    test_id_path = Path("workspace/data/splits/brandood/test_id.csv")
    test_ood_path = Path("workspace/data/splits/brandood/test_ood.csv")

    print("\n" + "=" * 60)
    print("Brand-OOD数据分割泄漏检查")
    print("=" * 60)

    files = {
        "train": train_path,
        "val": val_path,
        "test_id": test_id_path,
        "test_ood": test_ood_path,
    }

    datasets = {}
    for name, path in files.items():
        if path.exists():
            datasets[name] = pd.read_csv(path)
            print(f"\n{name}: {len(datasets[name])} 样本")
        else:
            print(f"\n[WARNING] 文件不存在: {path}")

    if len(datasets) < 2:
        print("[ERROR] 数据文件不足，无法检查")
        return

    # 检查重叠
    id_col = "id" if "id" in list(datasets.values())[0].columns else "url_text"
    print(f"\n使用标识列: {id_col}")

    ids_dict = {name: set(df[id_col]) for name, df in datasets.items()}

    print("\n重叠检查:")
    overlap_found = False
    for i, (name1, ids1) in enumerate(ids_dict.items()):
        for name2, ids2 in list(ids_dict.items())[i + 1 :]:
            overlap = ids1 & ids2
            if overlap:
                overlap_found = True
                print(f"  {name1}-{name2}重叠: {len(overlap)} 个样本")
                if len(overlap) <= 5:
                    print(f"    重叠ID: {list(overlap)}")
            else:
                print(f"  {name1}-{name2}重叠: 0 个样本")

    print("\n" + "=" * 60)
    if not overlap_found:
        print("[SUCCESS] 无数据泄漏 - 所有数据集之间没有重复样本")
    else:
        print("[ERROR] 发现数据泄漏！")
    print("=" * 60)

    # 检查标签分布
    print("\n标签分布检查:")
    for name, df in datasets.items():
        if "label" in df.columns:
            label_dist = df["label"].value_counts().to_dict()
            print(f"  {name}: {label_dist}")
        else:
            print(f"  {name}: 无label列")


if __name__ == "__main__":
    check_leakage()
    check_brandood_leakage()

"""
修复数据schema问题
- 删除 url_text 为空的行
- 确保 label 列为整数类型
"""

import pandas as pd
from pathlib import Path


def fix_csv(csv_path: Path):
    """修复单个CSV文件"""
    print(f"\n处理: {csv_path.name}")

    df = pd.read_csv(csv_path)
    original_count = len(df)
    print(f"  原始样本数: {original_count}")

    # 删除 url_text 为空的行
    null_count = df["url_text"].isna().sum()
    if null_count > 0:
        print(f"  发现 {null_count} 个 url_text 空值，正在删除...")
        df = df.dropna(subset=["url_text"])

    # 确保 label 为整数
    df["label"] = df["label"].astype(int)

    # 保存
    df.to_csv(csv_path, index=False)
    final_count = len(df)
    print(f"  最终样本数: {final_count}")

    if final_count < original_count:
        print(f"  已删除 {original_count - final_count} 行")
    else:
        print("  无需修改")

    return original_count, final_count


def main():
    print("=" * 70)
    print("修复数据Schema问题")
    print("=" * 70)

    data_dir = Path("data/processed")
    csv_files = ["train.csv", "val.csv", "test.csv"]

    total_removed = 0

    for csv_file in csv_files:
        csv_path = data_dir / csv_file
        if csv_path.exists():
            original, final = fix_csv(csv_path)
            total_removed += original - final

    print("\n" + "=" * 70)
    print(f"完成！共删除 {total_removed} 行空数据")
    print("=" * 70)
    print("\n请重新运行验证脚本:")
    print("  python scripts/validate_data_schema.py")


if __name__ == "__main__":
    main()

"""
数据Schema验证脚本
检查 data/processed/ 中的 CSV 文件是否符合规范
"""

import sys
from pathlib import Path
import pandas as pd
from typing import List, Tuple

# 必需列
REQUIRED_COLUMNS = ["url_text", "label"]

# 可选列
OPTIONAL_COLUMNS = ["id", "domain", "source", "split", "timestamp"]

# 有效标签值
VALID_LABELS = {0, 1}


def validate_csv(csv_path: Path) -> Tuple[bool, List[str]]:
    """
    验证单个CSV文件

    Returns:
        (is_valid, error_messages)
    """
    errors = []

    # 检查文件是否存在
    if not csv_path.exists():
        errors.append(f"[ERROR] 文件不存在: {csv_path}")
        return False, errors

    try:
        # 读取CSV
        df = pd.read_csv(csv_path)

        # 检查是否为空
        if len(df) == 0:
            errors.append(f"[ERROR] 文件为空 (样本数=0): {csv_path.name}")
            return False, errors

        # 检查必需列
        missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
        if missing_cols:
            errors.append(f"[ERROR] 缺少必需列 {missing_cols}: {csv_path.name}")
            return False, errors

        # 检查 url_text 列
        if df["url_text"].isna().any():
            null_count = df["url_text"].isna().sum()
            errors.append(
                f"[WARN] url_text 列包含 {null_count} 个空值: {csv_path.name}"
            )

        if not df["url_text"].dtype == "object":
            errors.append(f"[WARN] url_text 应为字符串类型: {csv_path.name}")

        # 检查 label 列
        if df["label"].isna().any():
            null_count = df["label"].isna().sum()
            errors.append(f"[ERROR] label 列包含 {null_count} 个空值: {csv_path.name}")
            return False, errors

        # 检查 label 值是否只包含 0 和 1
        unique_labels = set(df["label"].unique())
        invalid_labels = unique_labels - VALID_LABELS
        if invalid_labels:
            errors.append(
                f"[ERROR] label 包含无效值 {invalid_labels} (只允许 0, 1): {csv_path.name}"
            )
            return False, errors

        # 检查可选列的存在性（仅报告）
        present_optional = [col for col in OPTIONAL_COLUMNS if col in df.columns]

        # 打印信息
        print(f"\n[OK] {csv_path.name}")
        print(f"   样本数: {len(df):,}")
        print(f"   必需列: {REQUIRED_COLUMNS} [通过]")
        if present_optional:
            print(f"   可选列: {present_optional}")
        print(
            f"   标签分布: 良性={len(df[df['label']==0])} ({len(df[df['label']==0])/len(df)*100:.1f}%), "
            f"钓鱼={len(df[df['label']==1])} ({len(df[df['label']==1])/len(df)*100:.1f}%)"
        )

        # 显示列类型
        print(f"   url_text 类型: {df['url_text'].dtype}")
        print(f"   label 类型: {df['label'].dtype}")

        return len(errors) == 0, errors

    except Exception as e:
        errors.append(f"[ERROR] 读取文件失败 {csv_path.name}: {e}")
        return False, errors


def main():
    """主验证流程"""
    print("=" * 70)
    print("数据Schema验证")
    print("=" * 70)

    print("\n[Schema规范]")
    print(f"   必需列: {REQUIRED_COLUMNS}")
    print(f"   可选列: {OPTIONAL_COLUMNS}")
    print(f"   标签值: {VALID_LABELS}")
    print("   样本数: > 0")

    # 待检查的文件
    data_dir = Path("data/processed")
    csv_files = ["train.csv", "val.csv", "test.csv"]

    all_valid = True
    all_errors = []

    # 验证每个文件
    for csv_file in csv_files:
        csv_path = data_dir / csv_file
        is_valid, errors = validate_csv(csv_path)

        if not is_valid:
            all_valid = False
            all_errors.extend(errors)

    # 总结
    print("\n" + "=" * 70)
    if all_valid:
        print("[SUCCESS] 所有文件通过验证!")
        print("=" * 70)
        return 0
    else:
        print("[FAILED] 验证失败!")
        print("=" * 70)
        print("\n错误列表:")
        for error in all_errors:
            print(f"  {error}")
        print()
        return 1


if __name__ == "__main__":
    sys.exit(main())

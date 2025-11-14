"""
检查HTML文件完整性，找出缺失或空文件
"""

from pathlib import Path
from tqdm import tqdm
import pandas as pd


def check_html_integrity(csv_path: str):
    """检查CSV中所有样本的HTML完整性"""
    print(f"\n检查: {csv_path}")
    print("=" * 80)

    df = pd.read_csv(csv_path)
    print(f"总样本数: {len(df)}")

    missing_html = []
    empty_html = []
    read_errors = []
    valid_html = []

    for idx in tqdm(range(len(df)), desc="检查HTML"):
        row = df.iloc[idx]
        sample_id = row.get("id", idx)
        html_path = row.get("html_path", "")

        if not html_path or pd.isna(html_path):
            missing_html.append(
                {
                    "idx": idx,
                    "id": sample_id,
                    "reason": "path_missing",
                    "url": row.get("url_text", ""),
                }
            )
            continue

        path = Path(html_path)
        if not path.exists():
            missing_html.append(
                {
                    "idx": idx,
                    "id": sample_id,
                    "reason": "file_not_exist",
                    "path": str(path),
                    "url": row.get("url_text", ""),
                }
            )
            continue

        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
            if len(text.strip()) == 0:
                empty_html.append(
                    {
                        "idx": idx,
                        "id": sample_id,
                        "path": str(path),
                        "url": row.get("url_text", ""),
                    }
                )
            else:
                valid_html.append(idx)
        except Exception as e:
            read_errors.append(
                {
                    "idx": idx,
                    "id": sample_id,
                    "path": str(path),
                    "error": str(e),
                    "url": row.get("url_text", ""),
                }
            )

    # 打印统计
    print(f"\n[OK] 有效HTML: {len(valid_html)} ({len(valid_html)/len(df)*100:.1f}%)")
    print(
        f"[MISS] 缺失HTML: {len(missing_html)} ({len(missing_html)/len(df)*100:.1f}%)"
    )
    print(f"[EMPTY] 空HTML: {len(empty_html)} ({len(empty_html)/len(df)*100:.1f}%)")
    print(f"[ERROR] 读取错误: {len(read_errors)} ({len(read_errors)/len(df)*100:.1f}%)")

    # 打印前几个问题样本
    if missing_html:
        print("\n前5个缺失HTML的样本:")
        for item in missing_html[:5]:
            print(f"  idx={item['idx']}, id={item['id']}, reason={item['reason']}")
            print(f"    url: {item.get('url', '')[:80]}")

    if empty_html:
        print("\n前5个空HTML的样本:")
        for item in empty_html[:5]:
            print(f"  idx={item['idx']}, id={item['id']}")
            print(f"    path: {item['path']}")
            print(f"    url: {item.get('url', '')[:80]}")

    if read_errors:
        print("\n前5个读取错误的样本:")
        for item in read_errors[:5]:
            print(f"  idx={item['idx']}, id={item['id']}, error={item['error']}")
            print(f"    path: {item['path']}")

    return {
        "valid": valid_html,
        "missing": missing_html,
        "empty": empty_html,
        "errors": read_errors,
    }


if __name__ == "__main__":
    # 检查IID数据集
    print("\n" + "=" * 80)
    print("IID数据集HTML完整性检查")
    print("=" * 80)

    train_result = check_html_integrity("workspace/data/splits/iid/train_cached.csv")
    val_result = check_html_integrity("workspace/data/splits/iid/val_cached.csv")
    test_result = check_html_integrity("workspace/data/splits/iid/test_cached.csv")

    # 汇总
    print("\n" + "=" * 80)
    print("汇总统计")
    print("=" * 80)
    total_missing = (
        len(train_result["missing"])
        + len(val_result["missing"])
        + len(test_result["missing"])
    )
    total_empty = (
        len(train_result["empty"])
        + len(val_result["empty"])
        + len(test_result["empty"])
    )
    total_errors = (
        len(train_result["errors"])
        + len(val_result["errors"])
        + len(test_result["errors"])
    )

    print(f"总计缺失HTML: {total_missing}")
    print(f"总计空HTML: {total_empty}")
    print(f"总计读取错误: {total_errors}")
    print(f"总计有问题样本: {total_missing + total_empty + total_errors}")

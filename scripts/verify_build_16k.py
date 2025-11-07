#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证 build_master_16k.py 生成的数据集质量
自动检测 data/processed/ 下所有 master_*.csv 文件并执行 10 项质量检查
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

# Handle Windows console encoding
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        import codecs

        sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "strict")
        sys.stderr = codecs.getwriter("utf-8")(sys.stderr.buffer, "strict")


# =====================================================================
# Constants
# =====================================================================

REQUIRED_COLUMNS = [
    "id",
    "label",
    "url_text",
    "html_path",
    "img_path",
    "domain",
    "source",
    "split",
    "brand",
    "timestamp",
]

# 严格模式阈值
PATH_MISSING_ERROR_THRESHOLD = 0.10  # 10% 失败 → 错误
PATH_MISSING_WARN_THRESHOLD = 0.05  # 5% 失败 → 警告
BRAND_COUNT_MIN = 5  # 最少品牌数
BRAND_DOMINANCE_MAX = 0.50  # 单一品牌最大占比
TIMESTAMP_FILL_MIN = 0.70  # 时间戳非空率最低要求
LABEL_BALANCE_MIN = 0.40  # 标签平衡度（最小类占比）


# =====================================================================
# File Discovery
# =====================================================================


def discover_master_csvs(processed_dir: Path) -> List[Path]:
    """发现所有 master_*.csv 文件"""
    if not processed_dir.exists():
        return []

    csvs = sorted(processed_dir.glob("master_*.csv"))
    return csvs


def find_companion_files(csv_path: Path) -> Dict[str, Optional[Path]]:
    """查找配套的 JSON 和日志文件"""
    base_name = csv_path.stem  # e.g., "master_400_test"
    processed_dir = csv_path.parent
    logs_dir = csv_path.parent.parent / "logs"

    # 提取后缀（如 "_400_test"）
    match = re.match(r"master(.*)$", base_name)
    suffix = match.group(1) if match else ""

    companions = {
        "metadata": processed_dir / f"metadata{suffix}.json",
        "selected_ids": processed_dir / f"selected_ids{suffix}.json",
        "dropped_reasons": processed_dir / f"dropped_reasons{suffix}.json",
    }

    # 查找最新的日志文件（匹配 build_*<suffix>_*.log）
    log_pattern = f"build_*{suffix}_*.log" if suffix else "build_*.log"
    log_files = sorted(logs_dir.glob(log_pattern)) if logs_dir.exists() else []
    companions["log"] = log_files[-1] if log_files else None

    return companions


# =====================================================================
# Validation Functions
# =====================================================================


def validate_file_structure(csv_path: Path) -> Dict[str, Any]:
    """检查 1: 文件存在性"""
    result = {
        "status": "pass",
        "csv_exists": csv_path.exists(),
        "companions": {},
        "messages": [],
    }

    if not result["csv_exists"]:
        result["status"] = "fail"
        result["messages"].append(f"CSV 文件不存在: {csv_path}")
        return result

    companions = find_companion_files(csv_path)
    for key, path in companions.items():
        exists = path and path.exists()
        result["companions"][key] = exists
        if not exists and key != "log":  # 日志文件可选
            result["status"] = "warn"
            result["messages"].append(f"缺少配套文件: {key}")

    return result


def validate_csv_format(df: pd.DataFrame, csv_path: Path) -> Dict[str, Any]:
    """检查 2-4: 行数、列完整性、标签分布"""
    result = {
        "status": "pass",
        "row_count": len(df),
        "columns": list(df.columns),
        "missing_columns": [],
        "label_distribution": {},
        "messages": [],
    }

    # 检查必需列
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    result["missing_columns"] = missing
    if missing:
        result["status"] = "fail"
        result["messages"].append(f'缺少必需列: {", ".join(missing)}')
        return result

    # 检查空列名
    if "" in df.columns or df.columns.isnull().any():
        result["status"] = "fail"
        result["messages"].append("存在空列名")

    # 检查标签分布
    if "label" in df.columns:
        label_counts = df["label"].value_counts().to_dict()
        result["label_distribution"] = label_counts

        # 验证标签值仅为 {0, 1}
        invalid_labels = set(label_counts.keys()) - {0, 1}
        if invalid_labels:
            result["status"] = "fail"
            result["messages"].append(f"标签值非法: {invalid_labels}")

        # 检查平衡度
        if len(label_counts) == 2:
            total = sum(label_counts.values())
            min_ratio = min(label_counts.values()) / total
            if min_ratio < LABEL_BALANCE_MIN:
                result["status"] = "warn"
                result["messages"].append(
                    f"标签不平衡: 少数类占比 {min_ratio:.1%} < {LABEL_BALANCE_MIN:.0%}"
                )

    # 检查重复行
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        result["status"] = "warn"
        result["messages"].append(f"存在 {duplicates} 行重复数据")

    return result


def validate_paths_sample(
    df: pd.DataFrame, sample_size: int = 100, skip: bool = False
) -> Dict[str, Any]:
    """检查 5: 路径有效性（抽样）"""
    result = {
        "status": "pass",
        "html_checked": 0,
        "html_exists": 0,
        "img_checked": 0,
        "img_exists": 0,
        "missing_samples": [],
        "messages": [],
    }

    if skip:
        result["messages"].append("已跳过路径验证")
        return result

    # 抽样
    sample_df = df.sample(n=min(sample_size, len(df)), random_state=42)

    # 检查 HTML 路径
    if "html_path" in df.columns:
        for idx, row in sample_df.iterrows():
            result["html_checked"] += 1
            path = Path(row["html_path"])
            if path.exists():
                result["html_exists"] += 1
            else:
                if len(result["missing_samples"]) < 5:  # 只记录前 5 个
                    result["missing_samples"].append(("html", row.get("id", idx)))

    # 检查 IMG 路径
    if "img_path" in df.columns:
        for idx, row in sample_df.iterrows():
            result["img_checked"] += 1
            path = Path(row["img_path"])
            if path.exists():
                result["img_exists"] += 1
            else:
                if len(result["missing_samples"]) < 10:  # 只记录前 10 个
                    result["missing_samples"].append(("img", row.get("id", idx)))

    # 计算缺失率
    html_missing_rate = (
        1 - (result["html_exists"] / result["html_checked"])
        if result["html_checked"] > 0
        else 0
    )
    img_missing_rate = (
        1 - (result["img_exists"] / result["img_checked"])
        if result["img_checked"] > 0
        else 0
    )
    max_missing_rate = max(html_missing_rate, img_missing_rate)

    if max_missing_rate > PATH_MISSING_ERROR_THRESHOLD:
        result["status"] = "fail"
        result["messages"].append(
            f"路径缺失率过高: HTML {html_missing_rate:.1%}, IMG {img_missing_rate:.1%} "
            f"(阈值: {PATH_MISSING_ERROR_THRESHOLD:.0%})"
        )
    elif max_missing_rate > PATH_MISSING_WARN_THRESHOLD:
        result["status"] = "warn"
        result["messages"].append(
            f"路径缺失率警告: HTML {html_missing_rate:.1%}, IMG {img_missing_rate:.1%}"
        )

    return result


def validate_brand_distribution(df: pd.DataFrame) -> Dict[str, Any]:
    """检查 6: 品牌分布"""
    result = {
        "status": "pass",
        "brand_count": 0,
        "top_brands": [],
        "max_brand_ratio": 0.0,
        "messages": [],
    }

    if "brand" not in df.columns:
        result["status"] = "warn"
        result["messages"].append("缺少 brand 列")
        return result

    # 统计品牌分布
    brand_counts = df["brand"].dropna().value_counts()
    result["brand_count"] = len(brand_counts)
    result["top_brands"] = brand_counts.head(10).to_dict()

    if len(brand_counts) > 0:
        total = brand_counts.sum()
        result["max_brand_ratio"] = brand_counts.iloc[0] / total

        # 检查品牌数量
        if result["brand_count"] < BRAND_COUNT_MIN:
            result["status"] = "warn"
            result["messages"].append(
                f'品牌数量过少: {result["brand_count"]} < {BRAND_COUNT_MIN}'
            )

        # 检查品牌集中度
        if result["max_brand_ratio"] > BRAND_DOMINANCE_MAX:
            result["status"] = "warn"
            result["messages"].append(
                f'品牌过度集中: Top 1 占比 {result["max_brand_ratio"]:.1%} > {BRAND_DOMINANCE_MAX:.0%}'
            )

    return result


def validate_timestamp_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """检查 7: 时间戳质量"""
    result = {"status": "pass", "fill_rate": 0.0, "timestamp_range": {}, "messages": []}

    if "timestamp" not in df.columns:
        result["status"] = "warn"
        result["messages"].append("缺少 timestamp 列")
        return result

    # 计算非空率
    non_null = df["timestamp"].notna().sum()
    result["fill_rate"] = non_null / len(df)

    if result["fill_rate"] < TIMESTAMP_FILL_MIN:
        result["status"] = "warn"
        result["messages"].append(
            f'时间戳缺失过多: {result["fill_rate"]:.1%} < {TIMESTAMP_FILL_MIN:.0%}'
        )

    # 时间范围
    try:
        timestamps = pd.to_datetime(df["timestamp"].dropna(), errors="coerce")
        valid_timestamps = timestamps.dropna()
        if len(valid_timestamps) > 0:
            result["timestamp_range"] = {
                "min": valid_timestamps.min().strftime("%Y-%m-%d"),
                "max": valid_timestamps.max().strftime("%Y-%m-%d"),
                "span_days": (valid_timestamps.max() - valid_timestamps.min()).days,
            }
    except Exception as e:
        result["status"] = "warn"
        result["messages"].append(f"时间戳解析错误: {e}")

    return result


def validate_split_column(df: pd.DataFrame, csv_name: str) -> Dict[str, Any]:
    """检查 8: split 列合理性"""
    result = {"status": "pass", "split_values": {}, "messages": []}

    if "split" not in df.columns:
        result["status"] = "warn"
        result["messages"].append("缺少 split 列")
        return result

    split_counts = df["split"].value_counts().to_dict()
    result["split_values"] = split_counts

    # 判断是否为测试集（文件名包含 "test"）
    is_test_csv = "test" in csv_name.lower()

    if is_test_csv:
        # 测试集应全为 "unsplit"
        if set(split_counts.keys()) != {"unsplit"}:
            result["status"] = "warn"
            result["messages"].append(
                f'测试集 split 列应全为 "unsplit"，实际: {list(split_counts.keys())}'
            )
    else:
        # 训练集允许 train/val/test 或 unsplit
        valid_splits = {"train", "val", "test", "unsplit"}
        invalid_splits = set(split_counts.keys()) - valid_splits
        if invalid_splits:
            result["status"] = "warn"
            result["messages"].append(f"split 列包含非法值: {invalid_splits}")

    return result


def validate_metadata_files(csv_path: Path) -> Dict[str, Any]:
    """检查 9: 元数据 JSON 文件"""
    result = {
        "status": "pass",
        "metadata_valid": False,
        "dropped_reasons_valid": False,
        "messages": [],
    }

    companions = find_companion_files(csv_path)

    # 检查 metadata.json
    metadata_path = companions.get("metadata")
    if metadata_path and metadata_path.exists():
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            required_keys = [
                "total_samples",
                "brand_distribution",
                "timestamp_range",
                "modality_completeness",
            ]
            missing_keys = [k for k in required_keys if k not in metadata]

            if missing_keys:
                result["messages"].append(
                    f'metadata.json 缺少字段: {", ".join(missing_keys)}'
                )
            else:
                result["metadata_valid"] = True
        except Exception as e:
            result["status"] = "warn"
            result["messages"].append(f"metadata.json 解析失败: {e}")
    else:
        result["status"] = "warn"
        result["messages"].append("metadata.json 不存在")

    # 检查 dropped_reasons.json
    dropped_path = companions.get("dropped_reasons")
    if dropped_path and dropped_path.exists():
        try:
            with open(dropped_path, "r", encoding="utf-8") as f:
                dropped = json.load(f)

            total_dropped = dropped.get("total_dropped", 0)

            # 读取 CSV 行数
            df = pd.read_csv(csv_path)
            csv_count = len(df)

            # 丢弃数量不应超过样本数 5 倍
            if total_dropped > csv_count * 5:
                result["status"] = "warn"
                result["messages"].append(
                    f"丢弃样本数过多: {total_dropped} > {csv_count} * 5"
                )
            else:
                result["dropped_reasons_valid"] = True
        except Exception as e:
            result["status"] = "warn"
            result["messages"].append(f"dropped_reasons.json 解析失败: {e}")
    else:
        result["status"] = "warn"
        result["messages"].append("dropped_reasons.json 不存在")

    return result


def validate_log_file(csv_path: Path) -> Dict[str, Any]:
    """检查 10: 日志文件"""
    result = {
        "status": "pass",
        "log_found": False,
        "has_success_marker": False,
        "has_errors": False,
        "messages": [],
    }

    companions = find_companion_files(csv_path)
    log_path = companions.get("log")

    if not log_path or not log_path.exists():
        result["status"] = "warn"
        result["messages"].append("日志文件不存在")
        return result

    result["log_found"] = True

    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            log_content = f.read()

        # 检查成功标记
        if "Wrote" in log_content and "rows to" in log_content:
            result["has_success_marker"] = True
        else:
            result["status"] = "warn"
            result["messages"].append("日志中未找到成功标记")

        # 检查错误标记
        error_patterns = ["Traceback", "PermissionError", "Exception:", "Error:"]
        for pattern in error_patterns:
            if pattern in log_content:
                result["has_errors"] = True
                result["status"] = "warn"
                result["messages"].append(f"日志中发现错误标记: {pattern}")
                break
    except Exception as e:
        result["status"] = "warn"
        result["messages"].append(f"日志文件读取失败: {e}")

    return result


# =====================================================================
# Report Printing
# =====================================================================


def print_report(
    csv_path: Path, all_results: Dict[str, Dict[str, Any]], strict: bool
) -> int:
    """打印验证报告并返回退出码"""

    # 统计状态
    status_counts = Counter(r["status"] for r in all_results.values())
    has_failures = status_counts["fail"] > 0
    has_warnings = status_counts["warn"] > 0

    # 打印表头
    print("\n" + "╔" + "═" * 70 + "╗")
    print(f"║ 验证报告: {csv_path.name:<56} ║")
    print("╚" + "═" * 70 + "╝\n")

    # 打印各项检查结果
    checks = [
        ("file_structure", "文件存在性检查"),
        ("csv_format", "行数与格式检查"),
        ("paths", "路径有效性"),
        ("brand", "品牌分布"),
        ("timestamp", "时间戳质量"),
        ("split", "split 列"),
        ("metadata", "元数据文件"),
        ("log", "日志文件"),
    ]

    for key, label in checks:
        if key not in all_results:
            continue

        r = all_results[key]
        status = r["status"]

        # 图标
        if status == "pass":
            icon = "[✅]"
        elif status == "warn":
            icon = "[⚠️]"
        else:
            icon = "[❌]"

        # 主要信息
        if key == "file_structure":
            info = "通过" if status == "pass" else "部分缺失"
        elif key == "csv_format":
            row_count = r["row_count"]
            info = f"{row_count} 行数据"
            if r["label_distribution"]:
                phish = r["label_distribution"].get(1, 0)
                benign = r["label_distribution"].get(0, 0)
                info += f" | phishing: {phish} ({phish/(phish+benign)*100:.1f}%) | benign: {benign} ({benign/(phish+benign)*100:.1f}%)"
        elif key == "paths":
            if r.get("html_checked", 0) > 0:
                html_rate = r["html_exists"] / r["html_checked"] * 100
                img_rate = r["img_exists"] / r["img_checked"] * 100
                info = f'HTML: {r["html_exists"]}/{r["html_checked"]} ({html_rate:.0f}%) | IMG: {r["img_exists"]}/{r["img_checked"]} ({img_rate:.0f}%)'
            else:
                info = "已跳过"
        elif key == "brand":
            info = f'{r["brand_count"]} 个品牌'
            if r["max_brand_ratio"] > 0:
                info += f', Top 1 占比 {r["max_brand_ratio"]:.1%}'
        elif key == "timestamp":
            info = f'{r["fill_rate"]:.1%} 非空'
            if r["timestamp_range"]:
                tr = r["timestamp_range"]
                info += f', 跨度 {tr.get("min", "?")} ~ {tr.get("max", "?")}'
        elif key == "split":
            info = ", ".join(f"{k}: {v}" for k, v in r["split_values"].items())
        elif key == "metadata":
            valid_count = sum([r["metadata_valid"], r["dropped_reasons_valid"]])
            info = f"{valid_count}/2 文件有效"
        elif key == "log":
            info = "找到日志" if r["log_found"] else "未找到"
            if r.get("has_success_marker"):
                info += ", 包含成功标记"
        else:
            info = "通过" if status == "pass" else "有问题"

        print(f"{icon} {label:<20} {info}")

        # 打印详细消息（如果有）
        for msg in r.get("messages", []):
            print(f"    └─ {msg}")

    # 打印汇总
    print("\n" + "─" * 72)
    print(
        f"总计: {status_counts['pass']} 项通过 / {status_counts['warn']} 项警告 / {status_counts['fail']} 项失败"
    )

    # 判断最终状态
    if has_failures:
        print("状态: ❌ 验证失败，不建议用于训练")
        return 1 if strict else 0
    elif has_warnings:
        print("状态: ⚠️  有警告，建议检查后再训练")
        return 1 if strict else 0
    else:
        print("状态: ✅ 可以进入正式训练")
        return 0


# =====================================================================
# Main
# =====================================================================


def main():
    parser = argparse.ArgumentParser(
        description="验证 build_master_16k.py 生成的数据集质量",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 自动检测所有 master_*.csv
  python scripts/verify_build_16k.py

  # 指定特定文件
  python scripts/verify_build_16k.py --csv data/processed/master_400_test.csv

  # 宽松模式（仅警告，不退出）
  python scripts/verify_build_16k.py --lenient

  # 跳过路径验证（加速检查）
  python scripts/verify_build_16k.py --skip-path-check
        """,
    )
    parser.add_argument("--csv", type=Path, help="指定要验证的 CSV 文件路径")
    parser.add_argument(
        "--lenient", action="store_true", help="宽松模式: 警告和失败都不导致非零退出码"
    )
    parser.add_argument(
        "--skip-path-check", action="store_true", help="跳过路径有效性验证（加速检查）"
    )
    parser.add_argument(
        "--sample-size", type=int, default=100, help="路径验证抽样大小（默认: 100）"
    )

    args = parser.parse_args()

    # 确定要验证的文件
    if args.csv:
        if not args.csv.exists():
            print(f"错误: 文件不存在: {args.csv}", file=sys.stderr)
            return 1
        csv_files = [args.csv]
    else:
        # 自动发现
        processed_dir = Path("data/processed")
        csv_files = discover_master_csvs(processed_dir)

        if not csv_files:
            print(f"未在 {processed_dir} 中找到 master_*.csv 文件", file=sys.stderr)
            return 1

        print(f"发现 {len(csv_files)} 个 CSV 文件待验证:\n")
        for csv in csv_files:
            print(f"  - {csv.name}")
        print()

    # 验证每个文件
    exit_code = 0

    for csv_path in csv_files:
        all_results = {}

        # 1. 文件结构检查
        all_results["file_structure"] = validate_file_structure(csv_path)

        if not all_results["file_structure"]["csv_exists"]:
            print_report(csv_path, all_results, strict=not args.lenient)
            continue

        # 读取 CSV
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"\n错误: 无法读取 CSV 文件: {e}", file=sys.stderr)
            exit_code = 1 if not args.lenient else 0
            continue

        # 2-4. CSV 格式检查
        all_results["csv_format"] = validate_csv_format(df, csv_path)

        # 5. 路径有效性检查
        all_results["paths"] = validate_paths_sample(
            df, sample_size=args.sample_size, skip=args.skip_path_check
        )

        # 6. 品牌分布检查
        all_results["brand"] = validate_brand_distribution(df)

        # 7. 时间戳质量检查
        all_results["timestamp"] = validate_timestamp_quality(df)

        # 8. split 列检查
        all_results["split"] = validate_split_column(df, csv_path.name)

        # 9. 元数据文件检查
        all_results["metadata"] = validate_metadata_files(csv_path)

        # 10. 日志文件检查
        all_results["log"] = validate_log_file(csv_path)

        # 打印报告
        file_exit_code = print_report(csv_path, all_results, strict=not args.lenient)
        exit_code = max(exit_code, file_exit_code)

    return exit_code


if __name__ == "__main__":
    sys.exit(main())

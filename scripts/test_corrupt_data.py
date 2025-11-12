#!/usr/bin/env python
"""
腐败数据测试脚本：使用 IID 训练的 checkpoint 对腐败数据进行评估。

使用方法：
1. 运行测试（通过 Hydra）：
   python scripts/train_hydra.py \
     experiment=s0_iid_earlyconcat \
     trainer.max_epochs=0 \
     datamodule.test_csv=workspace/data/corrupt/url/test_corrupt_url_L.csv \
     run.name=corrupt_url_L

2. 收集结果并生成可视化：
   python scripts/test_corrupt_data.py \
     --experiment-dir experiments/s0_iid_earlyconcat_20251111_025612 \
     --corrupt-root workspace/data/corrupt \
     --output-dir experiments/corrupt_eval

衡量指标：
- AUROC（越高越好）
- FPR@TPR95（越低越好）
- ECE（越低越好）
- Brier（越低越好）

可视化：
- AUROC vs 强度(L/M/H) 的柱状或折线图（按模态分组）
- 可靠性曲线（Reliability Diagram）可选 IID vs H 这类对比
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score

from src.utils.logging import get_logger
from src.utils.metrics_v2 import (
    compute_brier_score,
    compute_ece,
    compute_fpr_at_tpr95,
)

log = get_logger(__name__)

# 设置中文字体支持
plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
sns.set_style("whitegrid")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="腐败数据测试结果收集和可视化")
    parser.add_argument(
        "--experiment-dir",
        type=str,
        required=True,
        help="IID 训练目录（包含 checkpoints/best.ckpt 和实验结果）",
    )
    parser.add_argument(
        "--modalities",
        nargs="+",
        choices=["url", "html", "img"],
        default=["url", "html", "img"],
        help="要处理的模态（默认：所有三模态）",
    )
    parser.add_argument(
        "--levels",
        nargs="+",
        choices=["L", "M", "H", "0.1", "0.3", "0.5"],
        default=None,
        help="腐败强度级别（默认：根据 test-type 自动确定）",
    )
    parser.add_argument(
        "--test-type",
        choices=["corrupt", "iid"],
        default="corrupt",
        help="测试类型：corrupt（主腐败评测 L/M/H）或 iid（轻噪声 0.1/0.3/0.5）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="输出目录（默认：experiments/corrupt_eval_<model_name>）",
    )
    parser.add_argument(
        "--collect-only",
        action="store_true",
        help="只收集结果，不生成可视化",
    )
    return parser.parse_args()


def find_predictions_files(experiment_dir: Path) -> Dict[str, Path]:
    """查找所有腐败数据的预测文件"""
    predictions = {}

    # 查找所有包含 "corrupt" 的预测文件
    for pred_file in experiment_dir.rglob("**/predictions*.csv"):
        # 检查文件名是否包含腐败数据标识
        if "corrupt" in pred_file.name.lower() or "corrupt" in str(pred_file.parent):
            # 尝试从路径或文件名提取模态和强度
            # 简化：假设文件名格式为 predictions_test_corrupt_{modality}_{intensity}.csv
            # 或者从目录结构推断
            predictions[str(pred_file)] = pred_file

    return predictions


def get_corrupt_csv_paths(
    corrupt_root: Path, modality: str, test_type: str
) -> Dict[str, Path]:
    """获取腐败数据 CSV 路径"""
    if test_type == "url":
        intensities = ["L", "M", "H"]
        base_dir = corrupt_root / modality
    else:
        intensities = ["0.1", "0.3", "0.5"]
        base_dir = corrupt_root / "iid" / modality

    csv_paths = {}
    for intensity in intensities:
        csv_path = base_dir / f"test_corrupt_{modality}_{intensity}.csv"
        if csv_path.exists():
            csv_paths[intensity] = csv_path
        else:
            log.warning(f"未找到 CSV: {csv_path}")

    return csv_paths


def collect_metrics_from_predictions(
    predictions_path: Path,
) -> Optional[Dict[str, float]]:
    """从预测 CSV 文件收集指标"""
    if not predictions_path.exists():
        log.warning(f"预测文件不存在: {predictions_path}")
        return None

    try:
        df = pd.read_csv(predictions_path)
        if not {"y_true", "prob"}.issubset(df.columns):
            log.warning(f"预测文件缺少必要列: {predictions_path}")
            return None

        y_true = df["y_true"].to_numpy()
        y_prob = df["prob"].to_numpy()

        # 计算指标
        auroc = roc_auc_score(y_true, y_prob)
        fpr95, thr95, reached = compute_fpr_at_tpr95(y_true, y_prob)
        ece, ece_bins, low_sample = compute_ece(y_true, y_prob)
        brier = compute_brier_score(y_true, y_prob)

        return {
            "auroc": float(auroc),
            "fpr_at_tpr95": float(fpr95),
            "ece": float(ece),
            "brier": float(brier),
            "ece_bins": int(ece_bins),
            "ece_low_sample": bool(low_sample),
            "tpr95_reached": bool(reached),
            "thr_at_tpr95": float(thr95),
            "y_true": y_true,
            "y_prob": y_prob,
        }
    except Exception as e:
        log.error(f"处理预测文件失败 {predictions_path}: {e}")
        return None


def search_corrupt_predictions(
    base_dir: Path, modalities: List[str], levels: List[str], test_type: str
) -> Dict[str, Dict[str, Dict]]:
    """搜索腐败数据的预测结果

    支持从以下位置查找：
    1. experiments/ 目录下的 corrupt_* 实验目录
    2. 直接根据 CSV 文件名匹配：test_corrupt_{modality}_{level}.csv
    """
    results = {}

    # 确定强度值（如果未指定，根据 test_type 自动确定）
    if not levels:
        if test_type == "corrupt":
            levels = ["L", "M", "H"]
        else:
            levels = ["0.1", "0.3", "0.5"]

    # 初始化结果结构
    for modality in modalities:
        results[modality] = {}

    # 方法1：从 experiments/ 目录查找所有 corrupt_* 实验目录
    # base_dir 是指定的实验目录（如 experiments/s0_iid_earlyconcat_20251111_025612）
    # 但腐败测试结果保存在 experiments/corrupt_* 目录中
    experiments_root = base_dir.parent if base_dir.name.startswith("s0_") else base_dir
    if not experiments_root.exists() or experiments_root.name != "experiments":
        # 如果 base_dir 本身就是 experiments 目录，使用它
        if base_dir.name == "experiments" or "experiments" in str(base_dir):
            experiments_root = base_dir
        else:
            # 尝试找到 experiments 目录
            potential_experiments = base_dir.parent / "experiments"
            if potential_experiments.exists():
                experiments_root = potential_experiments
            else:
                experiments_root = Path("experiments")

    # 查找所有包含 "corrupt" 的实验目录
    for exp_dir in experiments_root.glob("*corrupt*"):
        if not exp_dir.is_dir():
            continue

        # 在 artifacts 目录中查找预测文件
        artifacts_dir = exp_dir / "artifacts"
        if not artifacts_dir.exists():
            continue

        # 查找 predictions_test.csv 或 predictions.csv
        pred_files = list(artifacts_dir.glob("predictions*.csv"))
        if not pred_files:
            continue

        # 从实验目录名或路径推断模态和强度
        path_str = str(exp_dir).lower()
        matched_modality = None
        matched_intensity = None

        # 匹配模态
        for mod in modalities:
            if mod.lower() in path_str:
                matched_modality = mod
                break

        # 匹配强度
        for level_val in levels:
            if (
                level_val in path_str
                or f"_{level_val}" in path_str
                or f"-{level_val}" in path_str
            ):
                matched_intensity = level_val
                break

        if matched_modality and matched_intensity:
            pred_file = pred_files[0]  # 使用第一个找到的预测文件
            metrics = collect_metrics_from_predictions(pred_file)
            if metrics:
                results[matched_modality][matched_intensity] = metrics
                log.info(
                    f">> 找到预测文件: {pred_file} "
                    f"(模态={matched_modality}, 强度={matched_intensity})"
                )

    # 方法2：直接从 CSV 文件名匹配（如果 CSV 文件在已知位置）
    # 查找所有可能的预测文件，从文件名提取信息
    for pred_file in experiments_root.rglob("**/predictions*.csv"):
        path_str = str(pred_file).lower()

        # 跳过非腐败数据的预测文件
        if "corrupt" not in path_str:
            continue

        # 尝试从路径中提取模态和强度
        matched_modality = None
        matched_intensity = None

        # 匹配模态（检查完整单词，避免误匹配）
        for mod in modalities:
            mod_lower = mod.lower()
            # 检查是否包含模态名称（作为独立词或部分）
            if mod_lower in path_str:
                # 进一步验证：确保不是其他词的组成部分
                if mod == "url" and "html" in path_str:
                    continue  # 避免 url 匹配到 html
                matched_modality = mod
                break

        # 匹配强度
        for level_val in levels:
            # 检查是否在路径中（作为独立标识符）
            if (
                f"_{level_val}" in path_str
                or f"-{level_val}" in path_str
                or f"/{level_val}/" in path_str
            ):
                matched_intensity = level_val
                break

        if matched_modality and matched_intensity:
            # 如果该组合还没有结果，则添加
            if matched_intensity not in results[matched_modality]:
                metrics = collect_metrics_from_predictions(pred_file)
                if metrics:
                    results[matched_modality][matched_intensity] = metrics
                    log.info(
                        f">> 找到预测文件: {pred_file} "
                        f"(模态={matched_modality}, 强度={matched_intensity})"
                    )

    # 方法3：如果 CSV 文件路径已知，直接读取（基于 test_corrupt_{mod}_{LEVEL}.csv 模式）
    # 这适用于直接从腐败数据 CSV 推断预测文件位置的情况
    corrupt_root = experiments_root.parent / "workspace" / "data" / "corrupt"
    if not corrupt_root.exists():
        corrupt_root = Path("workspace/data/corrupt")

    if corrupt_root.exists():
        for modality in modalities:
            # 根据 test_type 确定 CSV 文件位置
            if test_type == "corrupt":
                # 主腐败评测：url 在 corrupt/url/，html 和 img 在 corrupt/html/ 和 corrupt/img/
                if modality == "url":
                    base_csv_dir = corrupt_root / "url"
                elif modality == "html":
                    base_csv_dir = corrupt_root / "html"
                elif modality == "img":
                    base_csv_dir = corrupt_root / "img"
                else:
                    continue
            else:
                # IID 轻噪声：在 corrupt/iid/{modality}/
                base_csv_dir = corrupt_root / "iid" / modality

            if not base_csv_dir.exists():
                continue

            for level in levels:
                # 如果已经有结果，跳过
                if level in results[modality]:
                    continue

                # 查找对应的 CSV 文件
                csv_file = base_csv_dir / f"test_corrupt_{modality}_{level}.csv"
                if csv_file.exists():
                    # 尝试在实验目录中查找对应的预测文件
                    # 预测文件可能在 experiments/corrupt_{modality}_{level}/artifacts/ 中
                    exp_name_pattern = f"*corrupt*{modality}*{level}*"
                    for exp_dir in experiments_root.glob(exp_name_pattern):
                        if not exp_dir.is_dir():
                            continue
                        artifacts_dir = exp_dir / "artifacts"
                        if artifacts_dir.exists():
                            pred_files = list(artifacts_dir.glob("predictions*.csv"))
                            if pred_files:
                                metrics = collect_metrics_from_predictions(
                                    pred_files[0]
                                )
                                if metrics:
                                    results[modality][level] = metrics
                                    log.info(
                                        f">> 从 CSV 推断找到预测文件: {pred_files[0]} "
                                        f"(模态={modality}, 强度={level})"
                                    )
                                    break

    return results


def plot_auroc_vs_intensity(
    results: Dict[str, Dict[str, Dict]], output_path: Path, levels: List[str]
) -> None:
    """绘制 AUROC vs 强度柱状图（按模态分组）"""
    # 准备数据
    data = []
    for modality, level_dict in results.items():
        for level, metrics in level_dict.items():
            data.append(
                {
                    "modality": modality.upper(),
                    "level": level,
                    "auroc": metrics["auroc"],
                }
            )

    if not data:
        log.warning("没有数据可绘制 AUROC vs 强度图")
        return

    df = pd.DataFrame(data)

    # 确定强度顺序
    if "L" in levels:
        level_order = ["L", "M", "H"]
        level_labels = ["Low", "Medium", "High"]
    else:
        level_order = ["0.1", "0.3", "0.5"]
        level_labels = ["0.1", "0.3", "0.5"]

    # 创建图表
    fig, ax = plt.subplots(figsize=(10, 6))

    # 按模态分组绘制
    modalities = df["modality"].unique()
    x = np.arange(len(level_order))
    width = 0.25

    for i, mod in enumerate(modalities):
        mod_data = df[df["modality"] == mod]
        auroc_values = [
            (
                mod_data[mod_data["level"] == level_val]["auroc"].values[0]
                if len(mod_data[mod_data["level"] == level_val]) > 0
                else 0
            )
            for level_val in level_order
        ]
        ax.bar(
            x + i * width,
            auroc_values,
            width,
            label=mod,
            alpha=0.8,
        )

    ax.set_xlabel("Corruption Intensity")
    ax.set_ylabel("AUROC")
    ax.set_title("AUROC vs Corruption Intensity (by Modality)")
    ax.set_xticks(x + width * (len(modalities) - 1) / 2)
    ax.set_xticklabels(level_labels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim([0, 1.05])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    log.info(f">> 保存 AUROC vs 强度图: {output_path}")


def plot_reliability_comparison(
    results: Dict[str, Dict[str, Dict]],
    output_path: Path,
    levels: List[str],
    baseline_path: Optional[Path] = None,
) -> None:
    """绘制可靠性曲线对比（IID vs H 或 IID vs 最高强度）"""
    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5))
    if len(results) == 1:
        axes = [axes]

    # 确定最高强度
    if "H" in levels:
        max_level = "H"
    elif "0.5" in levels:
        max_level = "0.5"
    else:
        max_level = levels[-1] if levels else "H"

    for idx, (modality, level_dict) in enumerate(results.items()):
        ax = axes[idx]

        # 绘制基线（IID，如果有）
        if baseline_path and baseline_path.exists():
            try:
                baseline_df = pd.read_csv(baseline_path)
                if {"y_true", "prob"}.issubset(baseline_df.columns):
                    y_true_base = baseline_df["y_true"].to_numpy()
                    y_prob_base = baseline_df["prob"].to_numpy()
                    ece_base, ece_bins, _ = compute_ece(y_true_base, y_prob_base)
                    prob_true_base, prob_pred_base = calibration_curve(
                        y_true_base, y_prob_base, n_bins=ece_bins, strategy="uniform"
                    )
                    ax.plot(
                        prob_pred_base,
                        prob_true_base,
                        "o-",
                        label=f"IID (ECE={ece_base:.4f})",
                        linewidth=2,
                        markersize=6,
                    )
            except Exception as e:
                log.warning(f"加载基线数据失败: {e}")

        # 绘制最高强度的可靠性曲线
        if max_level in level_dict:
            metrics = level_dict[max_level]
            y_true = metrics["y_true"]
            y_prob = metrics["y_prob"]
            ece = metrics["ece"]
            ece_bins = metrics["ece_bins"]

            prob_true, prob_pred = calibration_curve(
                y_true, y_prob, n_bins=ece_bins, strategy="uniform"
            )
            ax.plot(
                prob_pred,
                prob_true,
                "s-",
                label=f"{modality.upper()} {max_level} (ECE={ece:.4f})",
                linewidth=2,
                markersize=6,
            )

        # 绘制完美校准线
        ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration", linewidth=1.5)

        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Fraction of Positives")
        ax.set_title(f"Reliability Diagram - {modality.upper()}")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    log.info(f">> 保存可靠性曲线对比图: {output_path}")


def main() -> None:
    args = parse_args()

    experiment_dir = Path(args.experiment_dir)
    if not experiment_dir.exists():
        log.error(f"实验目录不存在: {experiment_dir}")
        return

    # 确定强度级别
    if args.levels:
        levels = args.levels
    else:
        if args.test_type == "corrupt":
            levels = ["L", "M", "H"]
        else:
            levels = ["0.1", "0.3", "0.5"]

    # 确定输出目录
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # 从实验目录名提取模型名称
        model_name = (
            experiment_dir.name.split("_")[0]
            if "_" in experiment_dir.name
            else "default"
        )
        output_dir = Path(f"experiments/corrupt_eval_{model_name}")

    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 70)
    log.info("腐败数据测试结果收集")
    log.info("=" * 70)
    log.info(f">> 实验目录: {experiment_dir}")
    log.info(f">> 输出目录: {output_dir}")
    log.info(f">> 测试类型: {args.test_type}")
    log.info(f">> 模态: {', '.join(args.modalities)}")
    log.info(f">> 强度级别: {', '.join(levels)}")
    log.info("=" * 70)

    # 搜索腐败数据的预测结果
    log.info("\n>> 搜索腐败数据预测结果...")
    results = search_corrupt_predictions(
        experiment_dir, args.modalities, levels, args.test_type
    )

    # 收集所有结果到 DataFrame
    all_results = []
    found_count = 0

    for modality in args.modalities:
        if modality not in results:
            log.warning(f">> 未找到 {modality.upper()} 模态的任何结果")
            continue

        for level, metrics in results[modality].items():
            record = {
                "modality": modality,
                "level": level,
                "auroc": metrics["auroc"],
                "fpr_at_tpr95": metrics["fpr_at_tpr95"],
                "ece": metrics["ece"],
                "brier": metrics["brier"],
                "ece_bins": metrics["ece_bins"],
            }
            all_results.append(record)
            found_count += 1
            log.info(
                f">> 收集结果: {modality.upper()}-{level} - "
                f"AUROC={metrics['auroc']:.4f}, ECE={metrics['ece']:.4f}, "
                f"FPR@TPR95={metrics['fpr_at_tpr95']:.4f}, Brier={metrics['brier']:.4f}"
            )

    # 检查是否所有模态和强度都有结果
    expected_count = len(args.modalities) * len(levels)
    if found_count < expected_count:
        log.warning(
            f">> 警告：期望找到 {expected_count} 个结果，实际找到 {found_count} 个"
        )
        log.info(">> 缺失的结果：")
        for modality in args.modalities:
            if modality not in results:
                log.info(f"  - {modality.upper()}: 全部缺失")
            else:
                missing = [
                    level_val
                    for level_val in levels
                    if level_val not in results[modality]
                ]
                if missing:
                    log.info(f"  - {modality.upper()}: {', '.join(missing)}")

    if not all_results:
        log.error("未找到任何腐败数据预测结果。")
        log.info("提示：请先运行测试，例如：")
        log.info("")
        log.info(f"{args.test_type.upper()} 类型测试（{', '.join(levels)}）：")
        for mod in args.modalities:
            for level in levels:
                csv_path = (
                    f"workspace/data/corrupt/{mod}/test_corrupt_{mod}_{level}.csv"
                    if args.test_type == "corrupt"
                    else f"workspace/data/corrupt/iid/{mod}/test_corrupt_{mod}_{level}.csv"
                )
                log.info(
                    f"  python scripts/train_hydra.py experiment=s0_iid_earlyconcat "
                    f"trainer.max_epochs=0 datamodule.test_csv={csv_path} "
                    f"run.name=corrupt_{mod}_{level}"
                )
        return

    # 保存结果到 CSV
    df_results = pd.DataFrame(all_results)
    results_csv = output_dir / "corrupt_metrics.csv"
    df_results.to_csv(results_csv, index=False)
    log.info(f">> 保存指标结果: {results_csv}")

    # 保存结果到 JSON
    results_json = output_dir / "corrupt_metrics.json"
    with open(results_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    log.info(f">> 保存指标结果: {results_json}")

    if args.collect_only:
        log.info(">> 仅收集模式，跳过可视化")
        return

    # 生成可视化
    log.info(">> 生成可视化...")

    # 1. AUROC vs 强度图
    auroc_plot_path = output_dir / "auroc_vs_intensity.png"
    plot_auroc_vs_intensity(results, auroc_plot_path, levels)

    # 2. 可靠性曲线对比
    # 查找基线（IID）预测结果
    baseline_path = None
    for pred_file in experiment_dir.rglob("**/predictions_test.csv"):
        if "corrupt" not in str(pred_file).lower():
            baseline_path = pred_file
            break

    reliability_plot_path = output_dir / "reliability_comparison.png"
    plot_reliability_comparison(results, reliability_plot_path, levels, baseline_path)

    log.info(f"\n>> 腐败数据测试结果已保存到: {output_dir}")


if __name__ == "__main__":
    main()

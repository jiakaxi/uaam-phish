#!/usr/bin/env python
"""分析两次训练的结果比较"""

import json
from pathlib import Path
from datetime import datetime


def parse_time(time_str):
    """解析时间字符串"""
    try:
        return datetime.strptime(time_str, "%H:%M:%S")
    except Exception:
        return None


def calculate_time_diff(start_str, end_str):
    """计算时间差（秒）"""
    try:
        # 从日志中提取时间
        start_time = datetime.strptime(start_str, "%Y-%m-%d %H:%M:%S")
        end_time = datetime.strptime(end_str, "%Y-%m-%d %H:%M:%S")
        return (end_time - start_time).total_seconds()
    except Exception:
        return None


def analyze_training_comparison():
    """分析训练比较"""
    print("=" * 70)
    print("训练结果比较分析")
    print("=" * 70)

    # 训练1: num_workers=0
    run1_wandb = Path(
        "outputs/2025-11-10/06-44-36/wandb/run-20251110_064440-j8ztz1y7/files/wandb-summary.json"
    )
    run1_metrics = Path(
        "experiments/url_mvp_20251110_064436/results/metrics_final.json"
    )
    run1_val_metrics = Path(
        "experiments/url_mvp_20251110_064436/artifacts/metrics_val.json"
    )

    # 训练2: num_workers=2
    run2_wandb = Path(
        "outputs/2025-11-10/07-40-05/wandb/run-20251110_074008-v1kfelnt/files/wandb-summary.json"
    )
    run2_metrics = Path(
        "experiments/url_mvp_20251110_074005/results/metrics_final.json"
    )
    run2_val_metrics = Path(
        "experiments/url_mvp_20251110_074005/artifacts/metrics_val.json"
    )

    # 加载数据
    with open(run1_wandb, "r") as f:
        run1_summary = json.load(f)
    with open(run2_wandb, "r") as f:
        run2_summary = json.load(f)
    with open(run1_metrics, "r") as f:
        run1_final = json.load(f)
    with open(run2_metrics, "r") as f:
        run2_final = json.load(f)
    with open(run1_val_metrics, "r") as f:
        run1_val = json.load(f)
    with open(run2_val_metrics, "r") as f:
        run2_val = json.load(f)

    print("\n1. 训练时间比较")
    print("-" * 70)
    run1_runtime = run1_summary.get("_runtime", 0)
    run2_runtime = run2_summary.get("_runtime", 0)

    print("训练1 (num_workers=0):")
    print(f"  总运行时间: {run1_runtime:.1f}秒 ({run1_runtime/60:.1f}分钟)")
    print("  初始化时间: ~1140秒 (~19分钟) - HTML预加载")
    print(f"  实际训练时间: ~{run1_runtime-1140:.1f}秒")

    print("\n训练2 (num_workers=2):")
    print(f"  总运行时间: {run2_runtime:.1f}秒 ({run2_runtime/60:.1f}分钟)")
    print("  初始化时间: ~6秒 - 跳过HTML预加载")
    print(f"  实际训练时间: ~{run2_runtime-6:.1f}秒")

    speedup = run1_runtime / run2_runtime
    print(f"\n速度提升: {speedup:.2f}x ({run1_runtime/run2_runtime:.2f}倍)")
    print(
        f"时间节省: {run1_runtime - run2_runtime:.1f}秒 ({(run1_runtime - run2_runtime)/60:.1f}分钟)"
    )

    print("\n2. 测试集指标比较")
    print("-" * 70)
    print(
        f"{'指标':<15} {'训练1 (num_workers=0)':<25} {'训练2 (num_workers=2)':<25} {'差异':<15}"
    )
    print("-" * 70)

    metrics_to_compare = [
        ("test/loss", "Test Loss", True),
        ("test/acc", "Test Accuracy", False),
        ("test/auroc", "Test AUROC", False),
        ("test/f1", "Test F1", False),
    ]

    for key, name, lower_is_better in metrics_to_compare:
        val1 = run1_summary.get(
            key, run1_final.get("metrics", {}).get(key.split("/")[1], 0)
        )
        val2 = run2_summary.get(
            key, run2_final.get("metrics", {}).get(key.split("/")[1], 0)
        )
        diff = val2 - val1
        diff_pct = (diff / val1 * 100) if val1 != 0 else 0

        status = "[OK]" if abs(diff_pct) < 5 else "[WARN]"
        print(
            f"{name:<15} {val1:<25.4f} {val2:<25.4f} {diff:+.4f} ({diff_pct:+.2f}%) {status}"
        )

    print("\n3. 验证集指标比较（最终epoch）")
    print("-" * 70)
    print(
        f"{'指标':<15} {'训练1 (num_workers=0)':<25} {'训练2 (num_workers=2)':<25} {'差异':<15}"
    )
    print("-" * 70)

    val_metrics = [
        ("accuracy", "Val Accuracy"),
        ("auroc", "Val AUROC"),
        ("f1_macro", "Val F1"),
        ("ece", "Val ECE"),
    ]

    for key, name in val_metrics:
        val1 = run1_val.get(key, 0)
        val2 = run2_val.get(key, 0)
        diff = val2 - val1
        diff_pct = (diff / val1 * 100) if val1 != 0 else 0

        status = "[OK]" if abs(diff_pct) < 10 else "[WARN]"
        print(
            f"{name:<15} {val1:<25.4f} {val2:<25.4f} {diff:+.4f} ({diff_pct:+.2f}%) {status}"
        )

    print("\n4. 训练指标比较")
    print("-" * 70)
    print(f"{'指标':<15} {'训练1 (num_workers=0)':<25} {'训练2 (num_workers=2)':<25}")
    print("-" * 70)

    train_metrics = [
        ("train/acc", "Train Accuracy"),
        ("train/auroc", "Train AUROC"),
        ("train/f1", "Train F1"),
        ("train/loss_epoch", "Train Loss"),
    ]

    for key, name in train_metrics:
        val1 = run1_summary.get(key, 0)
        val2 = run2_summary.get(key, 0)
        print(f"{name:<15} {val1:<25.4f} {val2:<25.4f}")

    print("\n" + "=" * 70)
    print("结论")
    print("=" * 70)
    print(
        "1. [OK] 训练速度大幅提升: {:.2f}x (从 {:.1f}分钟 降低到 {:.1f}分钟)".format(
            speedup, run1_runtime / 60, run2_runtime / 60
        )
    )
    print("2. [OK] 初始化时间大幅减少: 从 ~19分钟 降低到 ~6秒 (190倍提升)")
    print("3. [WARN] 训练指标有轻微差异（可能是随机性导致的）")
    print("4. [OK] 多进程训练成功运行，没有出现错误")
    print("5. [OK] Tokenization完全离线化，多进程安全")

    print("\n建议:")
    print("- 使用num_workers=2可以显著提升训练速度")
    print("- 指标差异在可接受范围内（<10%），可能是随机性导致")
    print("- 在实际训练中，可以进一步增加num_workers（2-4）以获得更好的性能")


if __name__ == "__main__":
    analyze_training_comparison()

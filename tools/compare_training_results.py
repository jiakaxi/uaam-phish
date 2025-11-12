#!/usr/bin/env python
"""比较两次训练结果（num_workers=0 vs num_workers=2）"""

import json
from pathlib import Path
from typing import Dict, Optional
import sys


def load_wandb_metrics(wandb_dir: Path) -> Optional[Dict]:
    """从wandb目录加载指标"""
    try:
        # 查找wandb-metadata.json
        metadata_files = list(wandb_dir.rglob("wandb-metadata.json"))
        if not metadata_files:
            return None

        # 读取metrics文件
        metrics_files = list(wandb_dir.rglob("*.json"))
        for f in metrics_files:
            if "metrics" in f.name.lower() and f.name != "wandb-metadata.json":
                try:
                    with open(f, "r") as file:
                        data = json.load(file)
                        return data
                except Exception:
                    continue
    except Exception as e:
        print(f"Error loading wandb metrics: {e}")
    return None


def load_train_log(log_path: Path) -> Dict:
    """从训练日志中提取关键指标"""
    metrics = {
        "epoch_time": None,
        "batch_time": None,
        "train_loss": None,
        "val_loss": None,
        "val_auroc": None,
        "val_accuracy": None,
    }

    if not log_path.exists():
        return metrics

    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

            # 提取epoch时间
            if "Epoch" in content and "time" in content.lower():
                lines = content.split("\n")
                for line in lines:
                    if "Epoch" in line and "time" in line.lower():
                        # 尝试提取时间
                        if "s/epoch" in line or "seconds" in line:
                            metrics["epoch_time"] = line.strip()

            # 提取batch时间
            if "it/s" in content or "s/it" in content:
                lines = content.split("\n")
                for line in lines:
                    if "it/s" in line or "s/it" in line:
                        metrics["batch_time"] = line.strip()
                        break

            # 提取最终指标
            lines = content.split("\n")
            for i, line in enumerate(lines):
                if "val/loss" in line.lower() or "validation loss" in line.lower():
                    metrics["val_loss"] = line.strip()
                if "val/auroc" in line.lower() or "validation auroc" in line.lower():
                    metrics["val_auroc"] = line.strip()
                if (
                    "val/accuracy" in line.lower()
                    or "validation accuracy" in line.lower()
                ):
                    metrics["val_accuracy"] = line.strip()
    except Exception as e:
        print(f"Error reading log: {e}")

    return metrics


def compare_results(result1_dir: Path, result2_dir: Path, name1: str, name2: str):
    """比较两次训练结果"""
    print("=" * 70)
    print(f"比较训练结果: {name1} vs {name2}")
    print("=" * 70)

    # 加载结果
    result1_log = result1_dir / "train_hydra.log"
    result2_log = result2_dir / "train_hydra.log"

    metrics1 = load_train_log(result1_log)
    metrics2 = load_train_log(result2_log)

    # 显示比较结果
    print(f"\n{name1} (num_workers=0):")
    print(f"  Epoch时间: {metrics1.get('epoch_time', 'N/A')}")
    print(f"  Batch时间: {metrics1.get('batch_time', 'N/A')}")
    print(f"  Val Loss: {metrics1.get('val_loss', 'N/A')}")
    print(f"  Val AUROC: {metrics1.get('val_auroc', 'N/A')}")
    print(f"  Val Accuracy: {metrics1.get('val_accuracy', 'N/A')}")

    print(f"\n{name2} (num_workers=2):")
    print(f"  Epoch时间: {metrics2.get('epoch_time', 'N/A')}")
    print(f"  Batch时间: {metrics2.get('batch_time', 'N/A')}")
    print(f"  Val Loss: {metrics2.get('val_loss', 'N/A')}")
    print(f"  Val AUROC: {metrics2.get('val_auroc', 'N/A')}")
    print(f"  Val Accuracy: {metrics2.get('val_accuracy', 'N/A')}")

    print("\n" + "=" * 70)
    print("结论:")
    print("=" * 70)

    # 简单的性能比较
    if metrics1.get("batch_time") and metrics2.get("batch_time"):
        print("\n性能比较:")
        print(f"  {name1}: {metrics1['batch_time']}")
        print(f"  {name2}: {metrics2['batch_time']}")
        print("\n如果num_workers=2的训练速度更快，说明多进程优化有效。")

    if metrics1.get("val_loss") and metrics2.get("val_loss"):
        print("\n指标比较:")
        print("  两次训练的指标应该相似（使用相同的随机种子）。")
        print("  如果指标差异很大，可能存在随机性问题。")


def main():
    """主函数"""
    if len(sys.argv) < 3:
        print(
            "Usage: python compare_training_results.py <result1_dir> <result2_dir> [name1] [name2]"
        )
        print(
            "Example: python compare_training_results.py outputs/2025-11-10/06-44-36 outputs/2025-11-10/07-00-00"
        )
        sys.exit(1)

    result1_dir = Path(sys.argv[1])
    result2_dir = Path(sys.argv[2])
    name1 = sys.argv[3] if len(sys.argv) > 3 else "Result 1"
    name2 = sys.argv[4] if len(sys.argv) > 4 else "Result 2"

    if not result1_dir.exists():
        print(f"Error: {result1_dir} does not exist")
        sys.exit(1)

    if not result2_dir.exists():
        print(f"Error: {result2_dir} does not exist")
        sys.exit(1)

    compare_results(result1_dir, result2_dir, name1, name2)


if __name__ == "__main__":
    main()

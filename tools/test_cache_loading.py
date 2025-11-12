#!/usr/bin/env python
"""
数据加载速度测试脚本：验证缓存数据加载速度。
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, Any

# 避免被 pytest 作为测试模块收集与执行
try:
    import pytest  # noqa: F401

    pytestmark = pytest.mark.skip(reason="Utility script, skip in pytest runs")
except Exception:
    pass

import torch
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from src.data.multimodal_datamodule import MultimodalDataModule
from src.utils.logging import get_logger

log = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="测试缓存数据加载速度")
    parser.add_argument(
        "--train-csv",
        default="workspace/data/splits/iid/train_cached.csv",
        help="训练CSV路径（必须使用*_cached.csv）",
    )
    parser.add_argument(
        "--preprocessed-dir",
        default="workspace/data/preprocessed/iid/train",
        help="预处理缓存目录",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch大小",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers数量（Windows建议0）",
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=10,
        help="测试的batch数量",
    )
    parser.add_argument(
        "--val-csv",
        default=None,
        help="验证CSV路径（可选，如果未提供则使用train CSV的前几行作为dummy）",
    )
    parser.add_argument(
        "--test-csv",
        default=None,
        help="测试CSV路径（可选，如果未提供则使用train CSV的前几行作为dummy）",
    )
    parser.add_argument(
        "--mode",
        choices=["train_only", "full"],
        default="train_only",
        help="测试模式：train_only（只测试train，使用dummy val/test）或 full（使用真实的val/test CSV）",
    )
    return parser.parse_args()


def create_dummy_config(args: argparse.Namespace) -> DictConfig:
    """创建用于DataModule的临时配置"""
    cfg = OmegaConf.create(
        {
            "data": {
                "url_max_len": 200,
                "url_vocab_size": 128,
                "html_max_len": 256,
            },
            "model": {
                "url_max_len": 200,
                "url_vocab_size": 128,
                "html_max_len": 256,
            },
        }
    )
    return cfg


def test_loading_speed(args: argparse.Namespace) -> Dict[str, Any]:
    """测试数据加载速度"""
    log.info("=" * 70)
    log.info("数据加载速度测试")
    log.info("=" * 70)
    log.info(f"训练CSV: {args.train_csv}")
    log.info(f"预处理目录: {args.preprocessed_dir}")
    log.info(f"Batch大小: {args.batch_size}")
    log.info(f"Workers: {args.num_workers}")
    log.info(f"测试batch数量: {args.num_batches}")
    log.info(f"测试模式: {args.mode}")
    log.info("=" * 70)

    # 检查文件是否存在
    train_csv_path = Path(args.train_csv)
    if not train_csv_path.exists():
        raise FileNotFoundError(f"训练CSV文件不存在: {train_csv_path}")

    if not train_csv_path.name.endswith("_cached.csv"):
        log.warning(f"警告: CSV文件不是*_cached.csv格式: {train_csv_path.name}")
        log.warning("这可能导致无法使用缓存，请检查配置")

    # 创建临时配置
    cfg = create_dummy_config(args)

    # 确定val和test CSV路径
    if args.mode == "full" or (args.val_csv and args.test_csv):
        # 使用真实的val和test CSV
        if args.val_csv:
            val_csv_path = Path(args.val_csv)
        else:
            # 从train CSV路径推断val CSV路径
            val_csv_path = train_csv_path.parent / train_csv_path.name.replace(
                "train", "val"
            )

        if args.test_csv:
            test_csv_path = Path(args.test_csv)
        else:
            # 从train CSV路径推断test CSV路径
            test_csv_path = train_csv_path.parent / train_csv_path.name.replace(
                "train", "test"
            )

        if not val_csv_path.exists():
            raise FileNotFoundError(f"验证CSV文件不存在: {val_csv_path}")
        if not test_csv_path.exists():
            raise FileNotFoundError(f"测试CSV文件不存在: {test_csv_path}")

        log.info("使用真实CSV:")
        log.info(f"  Val CSV: {val_csv_path}")
        log.info(f"  Test CSV: {test_csv_path}")

        # 推断预处理目录
        preprocessed_val_dir = (
            Path(args.preprocessed_dir).parent / "val"
            if args.preprocessed_dir
            else None
        )
        preprocessed_test_dir = (
            Path(args.preprocessed_dir).parent / "test"
            if args.preprocessed_dir
            else None
        )

        val_csv = str(val_csv_path)
        test_csv = str(test_csv_path)
        preprocessed_val_dir_str = (
            str(preprocessed_val_dir) if preprocessed_val_dir else None
        )
        preprocessed_test_dir_str = (
            str(preprocessed_test_dir) if preprocessed_test_dir else None
        )
    else:
        # train_only模式：使用train CSV的前几行作为dummy val和test
        import tempfile

        temp_dir = Path(tempfile.gettempdir()) / "cache_test"
        temp_dir.mkdir(parents=True, exist_ok=True)

        log.info("使用dummy CSV（train_only模式）")

        # 读取train CSV的前几行作为dummy val和test
        train_df = pd.read_csv(train_csv_path, nrows=10)
        dummy_val_csv = temp_dir / "dummy_val.csv"
        dummy_test_csv = temp_dir / "dummy_test.csv"
        train_df.to_csv(dummy_val_csv, index=False)
        train_df.to_csv(dummy_test_csv, index=False)

        val_csv = str(dummy_val_csv)
        test_csv = str(dummy_test_csv)
        preprocessed_val_dir_str = None
        preprocessed_test_dir_str = None

    # 创建DataModule
    # 注意：MultimodalDataset默认preload_html=True，但当我们有缓存时应该禁用
    # 由于MultimodalDataModule没有直接暴露preload_html参数，我们需要在Dataset创建后修改
    # 但为了测试，我们先用默认设置，看看缓存是否能正常工作
    datamodule = MultimodalDataModule(
        train_csv=str(train_csv_path),
        val_csv=val_csv,
        test_csv=test_csv,
        preprocessed_train_dir=args.preprocessed_dir,
        preprocessed_val_dir=preprocessed_val_dir_str,
        preprocessed_test_dir=preprocessed_test_dir_str,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=False,  # Windows上可能有问题
        persistent_workers=False,
        url_max_len=200,
        url_vocab_size=128,
        html_max_len=256,
        image_dir="data/processed/screenshots",
        cfg=cfg,
    )

    log.info(
        "\n注意：DataModule会预加载HTML文件到内存，但缓存系统会优先使用缓存的tokens"
    )
    log.info("如果看到'Loading HTML...'日志，说明缓存未命中，需要检查缓存路径配置")

    # 设置数据模块
    log.info("\n>> 设置DataModule...")
    datamodule.setup("fit")

    # 获取训练DataLoader
    train_loader = datamodule.train_dataloader()
    log.info(f">> 训练数据集大小: {len(datamodule.train_dataset)}")
    log.info(f">> Batch数量: {len(train_loader)}")

    # 测试加载速度
    log.info("\n>> 开始测试数据加载速度...")
    log.info(">> 跳过第1个batch（消除初始化开销）")

    batch_times = []
    batch_sizes = []
    iterator = iter(train_loader)

    try:
        # 跳过第1个batch
        batch_idx = 0
        log.info("  Skipping batch 0 (warmup)...")
        next(iterator)  # 消耗第一个batch

        # 记录开始时间
        prev_time = time.time()

        for batch_idx in range(1, args.num_batches + 1):
            # 获取batch（这会触发数据加载）
            batch = next(iterator)
            batch_end = time.time()

            # 计算从上一个batch结束到当前batch加载完成的时间
            batch_time = batch_end - prev_time
            prev_time = batch_end

            batch_times.append(batch_time)

            # 访问batch数据（健壮的访问方式）以触发实际的数据处理
            try:
                if isinstance(batch, dict):
                    # 字典格式
                    _ = batch.get("url", None)
                    _ = batch.get("html", None)
                    _ = batch.get("visual", None)
                    labels = batch.get("label", None)
                elif isinstance(batch, (list, tuple)):
                    # 元组格式
                    if len(batch) >= 2:
                        labels = batch[1]
                    if len(batch) >= 4:
                        _ = batch[0] if "url" in str(type(batch[0])) else None
                        _ = batch[1] if "html" in str(type(batch[1])) else None
                        _ = batch[2] if "visual" in str(type(batch[2])) else None
                else:
                    log.warning(f"未知的batch格式: {type(batch)}")

                # 获取batch大小
                if labels is not None:
                    if isinstance(labels, torch.Tensor):
                        batch_size = labels.shape[0]
                    else:
                        batch_size = len(labels)
                else:
                    batch_size = args.batch_size

                batch_sizes.append(batch_size)

            except Exception as e:
                log.warning(f"Batch {batch_idx} 访问出错: {e}")
                batch_size = args.batch_size
                batch_sizes.append(batch_size)

            # 实时输出
            its = 1.0 / batch_time if batch_time > 0 else 0.0
            log.info(f"  Batch {batch_idx}: {batch_time:.3f}s ({its:.2f} it/s)")

    except StopIteration:
        log.info(f"  DataLoader已耗尽（共{len(batch_times)}个batch）")

    except KeyboardInterrupt:
        log.info("\n>> 测试被用户中断")
    except Exception as e:
        log.error(f"\n>> 测试出错: {e}")
        raise

    # 计算统计信息
    if len(batch_times) == 0:
        log.error(">> 没有收集到任何batch时间数据")
        return {
            "passed": False,
            "error": "没有收集到batch时间数据",
        }

    # 计算平均速度（第2-10个batch）
    avg_time = sum(batch_times) / len(batch_times)
    avg_its = 1.0 / avg_time if avg_time > 0 else 0.0
    min_time = min(batch_times)
    max_time = max(batch_times)

    # 预估完整epoch时间
    total_batches = len(train_loader)
    estimated_epoch_time = avg_time * total_batches

    # 判断是否通过
    passed = avg_its >= 3.0 and max_time < 0.3

    # 输出结果
    log.info("\n" + "=" * 70)
    log.info("测试结果")
    log.info("=" * 70)
    log.info(f"测试batch数量: {len(batch_times)}")
    log.info(f"平均batch时间: {avg_time:.3f}s")
    log.info(f"最小batch时间: {min_time:.3f}s")
    log.info(f"最大batch时间: {max_time:.3f}s")
    log.info(f"平均速度: {avg_its:.2f} it/s")
    log.info(f"总batch数量: {total_batches}")
    log.info(
        f"预估epoch时间: {estimated_epoch_time:.1f}s ({estimated_epoch_time/60:.1f}分钟)"
    )
    log.info("=" * 70)

    # 通过标准
    log.info("\n通过标准:")
    log.info(
        f"  [OK] 平均速度 >= 3 it/s: {avg_its:.2f} it/s {'[PASS]' if avg_its >= 3.0 else '[FAIL]'}"
    )
    log.info(
        f"  [OK] 每个batch < 0.3秒: {max_time:.3f}s {'[PASS]' if max_time < 0.3 else '[FAIL]'}"
    )

    if passed:
        log.info("\n[OK] 测试通过！")
    else:
        log.info("\n[X] 测试未通过，请检查上述指标")

    log.info("=" * 70)

    return {
        "passed": passed,
        "avg_time": avg_time,
        "avg_its": avg_its,
        "min_time": min_time,
        "max_time": max_time,
        "total_batches": total_batches,
        "estimated_epoch_time": estimated_epoch_time,
        "num_test_batches": len(batch_times),
    }


def main() -> None:
    args = parse_args()
    result = test_loading_speed(args)

    # 返回退出码
    exit(0 if result.get("passed", False) else 1)


if __name__ == "__main__":
    main()

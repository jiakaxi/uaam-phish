"""
使用 Hydra 配置管理的训练脚本
支持命令行覆盖和多运行实验

用法:
    # 使用默认配置
    python scripts/train_hydra.py

    # 使用特定 trainer
    python scripts/train_hydra.py trainer=local
    python scripts/train_hydra.py trainer=server

    # 使用实验配置
    python scripts/train_hydra.py experiment=url_baseline

    # 命令行覆盖参数
    python scripts/train_hydra.py trainer=server train.bs=64 model.dropout=0.3

    # 多运行超参数搜索
    python scripts/train_hydra.py -m train.lr=1e-5,2e-5,5e-5 model.dropout=0.1,0.2,0.3
"""

import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from src.utils.seed import set_global_seed
from src.systems.url_only_module import UrlOnlySystem
from src.datamodules.url_datamodule import UrlDataModule
from src.utils.experiment_tracker import ExperimentTracker
from src.utils.callbacks import ExperimentResultsCallback, TestPredictionCollector
from src.utils.logging import get_logger

log = get_logger(__name__)


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def train(cfg: DictConfig) -> float:
    """
    Hydra 训练主函数

    Args:
        cfg: Hydra 配置对象

    Returns:
        测试集最佳指标（用于超参数优化）
    """
    # 打印配置（调试用）
    log.info("=" * 70)
    log.info("配置内容:")
    log.info(OmegaConf.to_yaml(cfg))
    log.info("=" * 70)

    # 设置随机种子
    pl.seed_everything(cfg.run.seed, workers=True)
    set_global_seed(cfg.run.seed)
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    # 初始化实验跟踪器
    exp_tracker = None
    if not cfg.get("no_save", False):
        exp_tracker = ExperimentTracker(cfg, exp_name=cfg.run.name)
        log.info(f"\n📁 实验目录: {exp_tracker.exp_dir}\n")

    # 初始化数据和模型
    dm = UrlDataModule(cfg)
    model = UrlOnlySystem(cfg)

    # 配置回调
    monitor = cfg.eval.get("monitor", "val_loss")
    patience = cfg.eval.get("patience", 3)
    mode = "max" if "f1" in monitor or "auroc" in monitor or "acc" in monitor else "min"

    callbacks = [
        EarlyStopping(monitor=monitor, mode=mode, patience=patience),
        ModelCheckpoint(
            monitor=monitor,
            mode=mode,
            save_top_k=1,
            filename=f"best-{{epoch}}-{{{monitor.replace('/', '_')}:.3f}}",
        ),
    ]

    # 添加实验结果保存回调
    if exp_tracker:
        callbacks.append(ExperimentResultsCallback(exp_tracker))
        pred_collector = TestPredictionCollector()
        callbacks.append(pred_collector)

    # 配置 Logger
    logger = None
    if "logger" in cfg:
        try:
            logger = hydra.utils.instantiate(cfg.logger)
            log.info(f"✅ 使用 Logger: {cfg.logger._target_}")
        except Exception as e:
            log.warning(f"⚠️  Logger 初始化失败: {e}")
            log.warning("   将使用默认的 CSV logger")

    # 配置训练器
    trainer = pl.Trainer(
        max_epochs=cfg.train.epochs,
        accelerator=cfg.hardware.accelerator,
        devices=cfg.hardware.devices,
        precision=cfg.hardware.precision,
        strategy=cfg.hardware.get("strategy", "auto"),
        log_every_n_steps=cfg.train.log_every,
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=1.0,
    )

    # 打印训练信息
    log.info("=" * 70)
    log.info("🚀 开始训练")
    log.info("=" * 70)
    log.info("📊 模型配置:")
    log.info(f"  - 预训练模型: {cfg.model.pretrained_name}")
    log.info(f"  - 最大长度: {cfg.data.max_length}")
    log.info(f"  - Dropout: {cfg.model.dropout}")
    log.info("\n🔧 训练配置:")
    log.info(f"  - Epochs: {cfg.train.epochs}")
    log.info(f"  - Batch size: {cfg.train.bs}")
    log.info(f"  - Learning rate: {cfg.train.lr}")
    log.info(f"  - 采样比例: {cfg.data.sample_fraction}")
    log.info("\n💻 硬件配置:")
    log.info(f"  - Accelerator: {cfg.hardware.accelerator}")
    log.info(f"  - Devices: {cfg.hardware.devices}")
    log.info(f"  - Precision: {cfg.hardware.precision}")
    log.info("\n📈 监控配置:")
    log.info(f"  - Monitor: {monitor}")
    log.info(f"  - Mode: {mode}")
    log.info(f"  - Patience: {patience}")
    log.info("=" * 70 + "\n")

    # 训练和测试
    trainer.fit(model, dm)
    test_results = trainer.test(
        model, dataloaders=dm.test_dataloader(), ckpt_path="best"
    )

    # 生成可视化图表
    if exp_tracker and not cfg.get("no_save", False):
        try:
            from src.utils.visualizer import ResultVisualizer

            lightning_log_dir = Path(trainer.log_dir)
            metrics_csv = lightning_log_dir / "metrics.csv"
            y_true, y_prob = pred_collector.get_predictions()

            if len(y_true) > 0 and metrics_csv.exists():
                log.info("\n📊 生成可视化图表...")
                ResultVisualizer.create_all_plots(
                    metrics_csv=metrics_csv,
                    y_true=y_true,
                    y_prob=y_prob,
                    output_dir=exp_tracker.results_dir,
                )
                log.info("✅ 所有图表已生成\n")
        except ImportError:
            log.warning("⚠️  matplotlib/seaborn 未安装，跳过可视化")
            log.warning('   安装命令: pip install -e ".[viz]"')
        except Exception as e:
            log.warning(f"⚠️  可视化生成失败: {e}")

    log.info("\n" + "=" * 70)
    log.info("✅ 训练完成！")
    if exp_tracker:
        log.info(f"📁 实验结果保存在: {exp_tracker.exp_dir}")
    log.info("=" * 70)

    # 返回最佳指标（用于超参数优化）
    if test_results:
        best_metric = test_results[0].get(f"test/{monitor.split('/')[-1]}", 0.0)
        return float(best_metric)
    return 0.0


if __name__ == "__main__":
    train()

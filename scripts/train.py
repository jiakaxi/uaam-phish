import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from omegaconf import OmegaConf
from pathlib import Path
import torch

from src.utils.seed import set_global_seed
from src.systems.url_only_module import UrlOnlySystem
from src.datamodules.url_datamodule import UrlDataModule
from src.utils.experiment_tracker import ExperimentTracker
from src.utils.callbacks import ExperimentResultsCallback, TestPredictionCollector
from src.utils.logging import get_logger

set_global_seed(3407)
log = get_logger(__name__)
log.info("Training start")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default=None, choices=[None, "local", "server"])
    ap.add_argument("--exp_name", default=None, help="实验名称（可选）")
    ap.add_argument("--no_save", action="store_true", help="不保存实验结果")
    args = ap.parse_args()

    # 加载配置
    cfg = OmegaConf.load("configs/default.yaml")
    if args.profile:
        prof = OmegaConf.load(f"configs/profiles/{args.profile}.yaml")
        cfg = OmegaConf.merge(cfg, prof)

    # 设置随机种子
    set_global_seed(cfg.train.seed)
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    # 初始化实验跟踪器
    exp_tracker = None
    if not args.no_save:
        exp_tracker = ExperimentTracker(cfg, exp_name=args.exp_name)
        print(f"\n📁 实验目录: {exp_tracker.exp_dir}\n")

    # 初始化数据和模型
    dm = UrlDataModule(cfg)
    model = UrlOnlySystem(cfg)

    # 配置回调
    monitor = cfg.eval.get("monitor", "val/loss")
    patience = cfg.eval.get("patience", 3)
    mode = "max" if "f1" in monitor or "auroc" in monitor else "min"

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
        # 添加预测收集器（用于生成 ROC 曲线等）
        pred_collector = TestPredictionCollector()
        callbacks.append(pred_collector)

    # 配置训练器
    trainer = pl.Trainer(
        max_epochs=cfg.train.epochs,
        accelerator=cfg.hardware.accelerator,
        devices=cfg.hardware.devices,
        precision=cfg.hardware.precision,
        strategy=cfg.hardware.get("strategy", "auto"),
        log_every_n_steps=cfg.train.log_every,
        callbacks=callbacks,
        gradient_clip_val=1.0,
    )

    # 打印配置信息
    print("=" * 70)
    print("🚀 开始训练")
    print("=" * 70)
    print("📊 模型配置:")
    print(f"  - 预训练模型: {cfg.model.pretrained_name}")
    print(f"  - 最大长度: {cfg.data.max_length}")
    print(f"  - Dropout: {cfg.model.dropout}")
    print("\n🔧 训练配置:")
    print(f"  - Epochs: {cfg.train.epochs}")
    print(f"  - Batch size: {cfg.train.bs}")
    print(f"  - Learning rate: {cfg.train.lr}")
    print(f"  - 采样比例: {cfg.data.sample_fraction}")
    print("\n💻 硬件配置:")
    print(f"  - Accelerator: {cfg.hardware.accelerator}")
    print(f"  - Devices: {cfg.hardware.devices}")
    print(f"  - Precision: {cfg.hardware.precision}")
    print("\n📈 监控配置:")
    print(f"  - Monitor: {monitor}")
    print(f"  - Mode: {mode}")
    print(f"  - Patience: {patience}")
    print("=" * 70)
    print()

    # 训练和测试
    trainer.fit(model, dm)
    test_results = trainer.test(
        model, dataloaders=dm.test_dataloader(), ckpt_path="best"
    )

    # 生成可视化图表（如果安装了 matplotlib）
    if exp_tracker and not args.no_save:
        try:
            from src.utils.visualizer import ResultVisualizer

            # 获取 Lightning 日志目录
            lightning_log_dir = Path(trainer.log_dir)
            metrics_csv = lightning_log_dir / "metrics.csv"

            # 获取测试集预测
            y_true, y_prob = pred_collector.get_predictions()

            if len(y_true) > 0 and metrics_csv.exists():
                print("\n📊 生成可视化图表...")
                ResultVisualizer.create_all_plots(
                    metrics_csv=metrics_csv,
                    y_true=y_true,
                    y_prob=y_prob,
                    output_dir=exp_tracker.results_dir,
                )
                print("✅ 所有图表已生成\n")
        except ImportError:
            print("⚠️  matplotlib/seaborn 未安装，跳过可视化")
            print('   安装命令: pip install -e ".[viz]"')
        except Exception as e:
            print(f"⚠️  可视化生成失败: {e}")

    print("\n" + "=" * 70)
    print("✅ 训练完成！")
    if exp_tracker:
        print(f"📁 实验结果保存在: {exp_tracker.exp_dir}")
    print("=" * 70)

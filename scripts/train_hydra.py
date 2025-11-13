"""
Hydra-based training entry point (S0 baseline, thesis Sec. 4.6).
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from src.utils.seed import set_global_seed
from src.utils.experiment_tracker import ExperimentTracker
from src.utils.callbacks import ExperimentResultsCallback, TestPredictionCollector
from src.utils.doc_callback import DocumentationCallback
from src.utils.logging import get_logger


log = get_logger(__name__)


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def train(cfg: DictConfig) -> float:
    """
    Launch training using Hydra configuration.

    Returns:
        Validation/test metric requested by the sweeper.
    """
    log.info("=" * 70)
    log.info("Configuration:\n%s", OmegaConf.to_yaml(cfg))
    log.info("=" * 70)

    # Sec. 4.6.3: deterministic seed handling
    pl.seed_everything(cfg.run.seed, workers=True)
    set_global_seed(cfg.run.seed)
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    exp_tracker = None
    artifact_dir: Path | None = None
    if not cfg.get("no_save", False):
        exp_tracker = ExperimentTracker(cfg, exp_name=cfg.run.name)
        artifact_dir = exp_tracker.artifacts_dir
        log.info("\n>> Experiment directory: %s\n", exp_tracker.exp_dir)

    # Instantiate DataModule
    if "datamodule" in cfg and "_target_" in cfg.datamodule:
        dm = hydra.utils.instantiate(cfg.datamodule, cfg=cfg)
        log.info(">> DataModule: %s", cfg.datamodule._target_)
    else:
        raise ValueError(
            "DataModule _target_ not specified. S0 experiments require explicit MultimodalDataModule configuration."
        )

    # Instantiate System
    if "system" in cfg and "_target_" in cfg.system:
        model = hydra.utils.instantiate(cfg.system, cfg=cfg)
        log.info(">> System: %s", cfg.system._target_)
    else:
        raise ValueError(
            "System _target_ not specified. S0 experiments require explicit MultimodalSystem configuration."
        )

    if artifact_dir and hasattr(model, "set_artifact_dir"):
        model.set_artifact_dir(artifact_dir)
    if exp_tracker and hasattr(model, "set_results_dir"):
        model.set_results_dir(exp_tracker.results_dir)

    # Callbacks (Sec. 4.6.3)
    monitor = cfg.eval.get("monitor", "val_loss")
    patience = cfg.eval.get("patience", 3)
    min_delta = cfg.eval.get("min_delta", 0.0)
    mode = (
        "max" if any(metric in monitor for metric in ("f1", "auroc", "acc")) else "min"
    )

    # Early stopping callback
    early_stopping_kwargs = {
        "monitor": monitor,
        "mode": mode,
        "patience": patience,
    }
    if min_delta > 0:
        early_stopping_kwargs["min_delta"] = min_delta

    callbacks = [
        EarlyStopping(**early_stopping_kwargs),
        ModelCheckpoint(
            monitor=monitor,
            mode=mode,
            save_top_k=1,
            save_weights_only=True,  # 只保存权重，避免完整模型序列化
            filename=f"best-{{epoch}}-{{{monitor.replace('/', '_')}:.3f}}",
        ),
    ]

    pred_collector = None
    if exp_tracker:
        callbacks.append(ExperimentResultsCallback(exp_tracker))
        pred_collector = TestPredictionCollector()
        callbacks.append(pred_collector)

        protocol = cfg.get("protocol", "presplit")

        if cfg.get("logging", {}).get("auto_append_docs", False):
            exp_name = cfg.get("exp_name", f"{protocol}_experiment")
            doc_callback = DocumentationCallback(
                feature_name=f"实验: {exp_name}",
                append_to_summary=cfg.get("logging", {}).get("append_to_summary", True),
                append_to_changes=cfg.get("logging", {}).get(
                    "append_to_changes", False
                ),
            )
            callbacks.append(doc_callback)
            log.info(">> Documentation auto-append enabled")
        else:
            log.info(
                ">> Documentation auto-append disabled (logging.auto_append_docs=false)"
            )

    # Lightning logger instantiation
    logger = None
    if "logger" in cfg:
        try:
            logger = hydra.utils.instantiate(cfg.logger)
            log.info(">> Logger: %s", cfg.logger._target_)
        except Exception as exc:
            log.warning(">> Logger init failed: %s", exc)
            log.warning("   Falling back to CSV logger")

    # 构建 Trainer 参数（支持 fast_dev_run、limit_* 等调试参数）
    # 确定max_epochs：优先使用trainer.max_epochs，否则使用train.epochs
    max_epochs = cfg.train.epochs
    if (
        "trainer" in cfg
        and "max_epochs" in cfg.trainer
        and cfg.trainer.max_epochs is not None
    ):
        max_epochs = cfg.trainer.max_epochs
        log.info(">> Using trainer.max_epochs=%s (overrides train.epochs)", max_epochs)

    trainer_kwargs = {
        "max_epochs": max_epochs,
        "accelerator": cfg.hardware.accelerator,
        "devices": cfg.hardware.devices,
        "precision": cfg.hardware.precision,
        "strategy": cfg.hardware.get("strategy", "auto"),
        "log_every_n_steps": cfg.train.log_every,
        "callbacks": callbacks,
        "logger": logger,
        "gradient_clip_val": 1.0,
        "accumulate_grad_batches": cfg.train.get("grad_accumulation", 1),
    }

    # 支持调试/测试参数（通过 trainer.* 添加）
    if "trainer" in cfg:
        for debug_param in [
            "fast_dev_run",
            "limit_train_batches",
            "limit_val_batches",
            "limit_test_batches",
            "overfit_batches",
            "max_epochs",  # 已在上面处理，但允许在trainer中设置
        ]:
            if debug_param in cfg.trainer:
                if debug_param != "max_epochs":  # max_epochs已在上面处理
                    trainer_kwargs[debug_param] = cfg.trainer[debug_param]
                    log.info(
                        ">> Debug mode enabled: %s=%s",
                        debug_param,
                        cfg.trainer[debug_param],
                    )

    trainer = pl.Trainer(**trainer_kwargs)

    log.info("=" * 70)
    log.info(">> Training starts")
    log.info("=" * 70)
    log.info(">> Model config:")
    model_name = getattr(cfg.model, "pretrained_name", "URLEncoder")
    max_len = getattr(cfg.model, "max_len", getattr(cfg.data, "max_length", 256))
    log.info("  - Model: %s", model_name)
    log.info("  - Max length: %s", max_len)
    log.info("  - Dropout: %s", cfg.model.dropout)
    log.info("\n>> Training config:")
    log.info("  - Epochs: %s", cfg.train.epochs)
    log.info("  - Batch size: %s", cfg.train.bs)
    log.info("  - Grad accumulation: %s", cfg.train.get("grad_accumulation", 1))
    log.info("  - Learning rate: %s", cfg.train.lr)
    log.info("\n>> Hardware config:")
    log.info("  - Accelerator: %s", cfg.hardware.accelerator)
    log.info("  - Devices: %s", cfg.hardware.devices)
    log.info("  - Precision: %s", cfg.hardware.precision)
    log.info("\n>> Callback config:")
    log.info("  - Monitor: %s", monitor)
    log.info("  - Mode: %s", mode)
    log.info("  - Patience: %s", patience)
    if min_delta > 0:
        log.info("  - Min delta: %s", min_delta)
    log.info("=" * 70 + "\n")

    # 如果 max_epochs=0，跳过训练，直接进行测试
    if max_epochs is None or max_epochs > 0:
        trainer.fit(model, dm)

    if hasattr(dm, "split_metadata"):
        if hasattr(model, "set_split_metadata"):
            model.set_split_metadata(dm.split_metadata)
        log.info(
            ">> DataModule split metadata keys: %s", list(dm.split_metadata.keys())
        )
    else:
        log.info(">> max_epochs=0，跳过训练，直接进行测试")
        # 即使不训练，也需要设置 split_metadata（如果有）
        if hasattr(dm, "split_metadata"):
            if hasattr(model, "set_split_metadata"):
                model.set_split_metadata(dm.split_metadata)

    dm.setup(stage="test")
    # In fast_dev_run mode, checkpoints are not saved, so we test with current weights
    # 如果配置中指定了 ckpt_path，使用它；否则使用 "best" 或 None
    if hasattr(cfg.trainer, "ckpt_path") and cfg.trainer.ckpt_path:
        ckpt_path = str(cfg.trainer.ckpt_path)
        log.info(f">> 使用指定的 checkpoint: {ckpt_path}")
    elif max_epochs is not None and max_epochs == 0:
        # max_epochs=0 时，必须指定 checkpoint
        if hasattr(cfg.trainer, "ckpt_path") and cfg.trainer.ckpt_path:
            ckpt_path = str(cfg.trainer.ckpt_path)
        else:
            log.warning(">> max_epochs=0 但未指定 checkpoint，将使用当前模型权重")
            ckpt_path = None
    elif not getattr(cfg.trainer, "fast_dev_run", False):
        ckpt_path = "best"
    else:
        ckpt_path = None
    test_results = trainer.test(
        model, dataloaders=dm.test_dataloader(), ckpt_path=ckpt_path
    )

    if exp_tracker and not cfg.get("no_save", False) and pred_collector:
        try:
            from src.utils.visualizer import ResultVisualizer

            lightning_log_dir = Path(trainer.log_dir or ".")
            metrics_csv = lightning_log_dir / "metrics.csv"
            y_true, y_prob = pred_collector.get_predictions()

            if len(y_true) > 0 and metrics_csv.exists():
                log.info("\n>> Generating visualizations...")
                ResultVisualizer.create_all_plots(
                    metrics_csv=metrics_csv,
                    y_true=y_true,
                    y_prob=y_prob,
                    output_dir=exp_tracker.results_dir,
                )
                log.info(">> Visualizations created\n")
        except ImportError:
            log.warning(">> matplotlib/seaborn not installed; skipping visualizations")
        except Exception as exc:
            log.warning(">> Visualization failed: %s", exc)

    log.info("\n" + "=" * 70)
    log.info(">> Training finished")
    if exp_tracker:
        log.info(">> Experiment assets stored at: %s", exp_tracker.exp_dir)
    log.info("=" * 70)

    if test_results:
        best_metric = test_results[0].get(f"test/{monitor.split('/')[-1]}", 0.0)
        return float(best_metric)
    return 0.0


if __name__ == "__main__":
    train()

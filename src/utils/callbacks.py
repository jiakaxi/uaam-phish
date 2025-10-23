"""
PyTorch Lightning 自定义回调
用于实验结果的自动保存
"""

import numpy as np
from pathlib import Path
from pytorch_lightning.callbacks import Callback
import torch


class ExperimentResultsCallback(Callback):
    """
    实验结果自动保存回调
    在训练结束后自动保存指标、图表和日志
    """

    def __init__(self, experiment_tracker):
        """
        Args:
            experiment_tracker: ExperimentTracker 实例
        """
        super().__init__()
        self.tracker = experiment_tracker
        self.test_outputs = []

    def on_train_start(self, trainer, pl_module):
        """训练开始时记录"""
        self.tracker.log_text("=" * 60)
        self.tracker.log_text("训练开始")
        model_name = getattr(self.tracker.cfg.model, "pretrained_name", "URLEncoder")
        self.tracker.log_text(f"模型: {model_name}")
        self.tracker.log_text(f"总轮数: {self.tracker.cfg.train.epochs}")
        self.tracker.log_text("=" * 60)

    def on_train_epoch_end(self, trainer, pl_module):
        """每个训练 epoch 结束时记录"""
        epoch = trainer.current_epoch

        # 获取当前 epoch 的指标
        metrics = trainer.callback_metrics
        log_msg = f"Epoch {epoch}: "

        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                log_msg += f"{key}={value.item():.4f} "

        self.tracker.log_text(log_msg)

    def on_train_end(self, trainer, pl_module):
        """训练结束时保存结果"""
        self.tracker.log_text("=" * 60)
        self.tracker.log_text("训练完成")
        self.tracker.log_text("=" * 60)

        # 复制检查点
        if trainer.log_dir:
            lightning_log_dir = Path(trainer.log_dir)
            self.tracker.copy_checkpoints(lightning_log_dir)

            # 保存训练历史曲线
            metrics_csv = lightning_log_dir / "metrics.csv"
            if metrics_csv.exists():
                # 生成可视化图表
                try:
                    from src.utils.visualizer import ResultVisualizer

                    ResultVisualizer.plot_training_curves(
                        metrics_csv,
                        save_path=self.tracker.results_dir / "training_curves.png",
                    )
                except ImportError:
                    print("[WARNING] matplotlib 未安装,跳过训练曲线绘制")
                except Exception as e:
                    print(f"[WARNING] 训练曲线绘制失败: {e}")

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        """收集测试批次的输出"""
        # 这个方法在每个测试批次结束时调用
        # 我们需要在 test_step 中返回需要的数据
        pass

    def on_test_end(self, trainer, pl_module):
        """测试结束时保存最终结果"""
        self.tracker.log_text("=" * 60)
        self.tracker.log_text("测试完成")
        self.tracker.log_text("=" * 60)

        # 保存最终指标
        final_metrics = {}
        for key, value in trainer.callback_metrics.items():
            if isinstance(value, torch.Tensor):
                final_metrics[key] = value.item()
            else:
                final_metrics[key] = value

        self.tracker.save_metrics(final_metrics, stage="final")

        # 生成实验总结
        summary = {
            "final_test_loss": final_metrics.get(
                "test_loss", final_metrics.get("test/loss", "N/A")
            ),
            "final_test_acc": final_metrics.get(
                "test_acc", final_metrics.get("test/acc", "N/A")
            ),
            "final_test_f1": final_metrics.get(
                "test_f1", final_metrics.get("test/f1", "N/A")
            ),
            "final_test_auroc": final_metrics.get(
                "test_auroc", final_metrics.get("test/auroc", "N/A")
            ),
            "total_epochs": trainer.current_epoch + 1,
        }

        self.tracker.save_summary(summary)

        print("\n" + "=" * 60)
        print(f"[SUCCESS] 实验结果已保存到: {self.tracker.exp_dir}")
        print("=" * 60)
        print(f"[CONFIG] 配置文件: {self.tracker.exp_dir / 'config.yaml'}")
        print(f"[METRICS] 指标文件: {self.tracker.results_dir / 'metrics_final.json'}")
        print(f"[CHECKPOINT] 检查点: {self.tracker.checkpoints_dir}")
        print(f"[LOGS] 日志: {self.tracker.logs_dir}")
        print("=" * 60)


class TestPredictionCollector(Callback):
    """
    收集测试集的预测结果，用于绘制 ROC 曲线和混淆矩阵
    """

    def __init__(self):
        super().__init__()
        self.y_true = []
        self.y_prob = []

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        """收集每个批次的预测"""
        # 需要在 LightningModule 的 test_step 中返回 predictions
        if outputs is not None and "y_true" in outputs and "y_prob" in outputs:
            self.y_true.append(outputs["y_true"].cpu().numpy())
            self.y_prob.append(outputs["y_prob"].cpu().numpy())

    def on_test_end(self, trainer, pl_module):
        """测试结束后合并所有预测"""
        if self.y_true and self.y_prob:
            self.y_true = np.concatenate(self.y_true)
            self.y_prob = np.concatenate(self.y_prob)

    def get_predictions(self):
        """获取收集的预测结果"""
        return self.y_true, self.y_prob

"""
实验结果可视化工具
自动生成训练曲线、混淆矩阵、ROC 曲线等图表
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, List
from sklearn.metrics import confusion_matrix, roc_curve, auc


# 设置中文字体支持（可选）
plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["figure.dpi"] = 100

# 设置 seaborn 样式
sns.set_style("whitegrid")
sns.set_palette("husl")


class ResultVisualizer:
    """实验结果可视化工具"""

    @staticmethod
    def plot_training_curves(metrics_csv: Path, save_path: Optional[Path] = None):
        """
        绘制训练曲线（loss, F1, AUROC, FPR）

        Args:
            metrics_csv: metrics.csv 文件路径（Lightning 输出）
            save_path: 保存路径（可选）

        Returns:
            matplotlib Figure 对象
        """
        df = pd.read_csv(metrics_csv)

        # 过滤出需要的列（支持两种命名格式）
        metric_cols = {
            "train/loss": "Train Loss",
            "val/loss": "Val Loss",
            "train_loss": "Train Loss",
            "val_loss": "Val Loss",
            "train/acc": "Train Acc",
            "val/acc": "Val Acc",
            "train_acc": "Train Acc",
            "val_acc": "Val Acc",
            "train/f1": "Train F1",
            "val/f1": "Val F1",
            "train_f1": "Train F1",
            "val_f1": "Val F1",
            "train/auroc": "Train AUROC",
            "val/auroc": "Val AUROC",
            "train_auroc": "Train AUROC",
            "val_auroc": "Val AUROC",
            "train/fpr": "Train FPR",
            "val/fpr": "Val FPR",
            "train_fpr": "Train FPR",
            "val_fpr": "Val FPR",
        }

        # 检查哪些列存在
        available_metrics = {k: v for k, v in metric_cols.items() if k in df.columns}

        if not available_metrics:
            print(f"[WARNING] 未找到可绘制的指标。可用列: {list(df.columns)}")
            return None

        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()

        # 1. Loss
        ax = axes[0]
        for col in ["train/loss", "train_loss"]:
            if col in df.columns:
                ax.plot(df["epoch"], df[col], label="Train Loss", marker="o")
                break
        for col in ["val/loss", "val_loss"]:
            if col in df.columns:
                ax.plot(df["epoch"], df[col], label="Val Loss", marker="s")
                break
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("训练和验证 Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. Accuracy (如果没有F1，显示准确率)
        ax = axes[1]
        has_f1 = False
        for col in ["train/f1", "train_f1"]:
            if col in df.columns:
                ax.plot(df["epoch"], df[col], label="Train F1", marker="o")
                has_f1 = True
                break
        for col in ["val/f1", "val_f1"]:
            if col in df.columns:
                ax.plot(df["epoch"], df[col], label="Val F1", marker="s")
                has_f1 = True
                break

        if not has_f1:
            # 如果没有F1，绘制准确率
            for col in ["train/acc", "train_acc"]:
                if col in df.columns:
                    ax.plot(df["epoch"], df[col], label="Train Acc", marker="o")
                    break
            for col in ["val/acc", "val_acc"]:
                if col in df.columns:
                    ax.plot(df["epoch"], df[col], label="Val Acc", marker="s")
                    break
            ax.set_ylabel("Accuracy")
            ax.set_title("准确率")
        else:
            ax.set_ylabel("F1 Score")
            ax.set_title("F1 分数")
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. AUROC
        ax = axes[2]
        has_auroc = False
        for col in ["train/auroc", "train_auroc"]:
            if col in df.columns:
                ax.plot(df["epoch"], df[col], label="Train AUROC", marker="o")
                has_auroc = True
                break
        for col in ["val/auroc", "val_auroc"]:
            if col in df.columns:
                ax.plot(df["epoch"], df[col], label="Val AUROC", marker="s")
                has_auroc = True
                break

        if has_auroc:
            ax.set_xlabel("Epoch")
            ax.set_ylabel("AUROC")
            ax.set_title("ROC 曲线下面积")
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(
                0.5,
                0.5,
                "No AUROC data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("AUROC (无数据)")

        # 4. FPR or other metrics
        ax = axes[3]
        has_fpr = False
        for col in ["train/fpr", "train_fpr"]:
            if col in df.columns:
                ax.plot(df["epoch"], df[col], label="Train FPR", marker="o")
                has_fpr = True
                break
        for col in ["val/fpr", "val_fpr"]:
            if col in df.columns:
                ax.plot(df["epoch"], df[col], label="Val FPR", marker="s")
                has_fpr = True
                break

        if has_fpr:
            ax.set_xlabel("Epoch")
            ax.set_ylabel("False Positive Rate")
            ax.set_title("假阳性率（越低越好）")
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(
                0.5,
                0.5,
                "No FPR data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("FPR (无数据)")

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"[SUCCESS] 训练曲线已保存: {save_path}")

        return fig

    @staticmethod
    def plot_confusion_matrix(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: List[str] = None,
        save_path: Optional[Path] = None,
    ):
        """
        绘制混淆矩阵

        Args:
            y_true: 真实标签
            y_pred: 预测标签
            class_names: 类别名称
            save_path: 保存路径

        Returns:
            matplotlib Figure 对象
        """
        if class_names is None:
            class_names = ["良性", "钓鱼"]

        cm = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
            cbar_kws={"label": "样本数"},
        )

        ax.set_xlabel("预测标签")
        ax.set_ylabel("真实标签")
        ax.set_title("混淆矩阵")

        # 添加准确率等信息
        tn, fp, fn, tp = cm.ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        info_text = f"Accuracy: {accuracy:.3f}\nPrecision: {precision:.3f}\nRecall: {recall:.3f}\nF1: {f1:.3f}"
        ax.text(
            1.4,
            0.5,
            info_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="center",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
        )

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"[SUCCESS] 混淆矩阵已保存: {save_path}")

        return fig

    @staticmethod
    def plot_roc_curve(
        y_true: np.ndarray, y_prob: np.ndarray, save_path: Optional[Path] = None
    ):
        """
        绘制 ROC 曲线

        Args:
            y_true: 真实标签
            y_prob: 预测概率
            save_path: 保存路径

        Returns:
            matplotlib Figure 对象
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=(8, 6))

        # 绘制 ROC 曲线
        ax.plot(
            fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})"
        )

        # 绘制对角线（随机分类器）
        ax.plot(
            [0, 1],
            [0, 1],
            color="navy",
            lw=2,
            linestyle="--",
            label="Random Classifier (AUC = 0.5)",
        )

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate (FPR)")
        ax.set_ylabel("True Positive Rate (TPR)")
        ax.set_title("ROC 曲线")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"[SUCCESS] ROC 曲线已保存: {save_path}")

        return fig

    @staticmethod
    def plot_threshold_analysis(
        y_true: np.ndarray, y_prob: np.ndarray, save_path: Optional[Path] = None
    ):
        """
        绘制阈值分析图（F1、Precision、Recall vs Threshold）

        Args:
            y_true: 真实标签
            y_prob: 预测概率
            save_path: 保存路径

        Returns:
            matplotlib Figure 对象
        """
        thresholds = np.linspace(0, 1, 101)
        f1_scores = []
        precisions = []
        recalls = []

        for th in thresholds:
            y_pred = (y_prob >= th).astype(int)

            tp = ((y_pred == 1) & (y_true == 1)).sum()
            fp = ((y_pred == 1) & (y_true == 0)).sum()
            fn = ((y_pred == 0) & (y_true == 1)).sum()

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0
            )

            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)

        # 找到最佳 F1 阈值
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx]
        best_f1 = f1_scores[best_idx]

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(thresholds, precisions, label="Precision", linewidth=2)
        ax.plot(thresholds, recalls, label="Recall", linewidth=2)
        ax.plot(thresholds, f1_scores, label="F1 Score", linewidth=2)

        # 标记最佳阈值
        ax.axvline(
            best_threshold,
            color="red",
            linestyle="--",
            linewidth=1.5,
            label=f"Best Threshold = {best_threshold:.3f}",
        )
        ax.scatter([best_threshold], [best_f1], color="red", s=100, zorder=5)

        ax.set_xlabel("阈值 (Threshold)")
        ax.set_ylabel("分数")
        ax.set_title(
            f"阈值分析 (最佳 F1={best_f1:.3f} @ threshold={best_threshold:.3f})"
        )
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"[SUCCESS] 阈值分析图已保存: {save_path}")

        return fig, best_threshold

    @staticmethod
    def create_all_plots(
        metrics_csv: Path, y_true: np.ndarray, y_prob: np.ndarray, output_dir: Path
    ):
        """
        创建所有可视化图表

        Args:
            metrics_csv: metrics.csv 路径
            y_true: 真实标签
            y_prob: 预测概率
            output_dir: 输出目录

        Returns:
            生成的图表文件路径列表
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_files = []

        # 1. 训练曲线
        try:
            fig = ResultVisualizer.plot_training_curves(
                metrics_csv, save_path=output_dir / "training_curves.png"
            )
            if fig:
                plt.close(fig)
                saved_files.append(output_dir / "training_curves.png")
        except Exception as e:
            print(f"[WARNING] 训练曲线绘制失败: {e}")

        # 2. ROC 曲线
        try:
            fig = ResultVisualizer.plot_roc_curve(
                y_true, y_prob, save_path=output_dir / "roc_curve.png"
            )
            plt.close(fig)
            saved_files.append(output_dir / "roc_curve.png")
        except Exception as e:
            print(f"[WARNING] ROC 曲线绘制失败: {e}")

        # 3. 混淆矩阵（使用 0.5 阈值）
        try:
            y_pred = (y_prob >= 0.5).astype(int)
            fig = ResultVisualizer.plot_confusion_matrix(
                y_true, y_pred, save_path=output_dir / "confusion_matrix.png"
            )
            plt.close(fig)
            saved_files.append(output_dir / "confusion_matrix.png")
        except Exception as e:
            print(f"[WARNING] 混淆矩阵绘制失败: {e}")

        # 4. 阈值分析
        try:
            fig, best_th = ResultVisualizer.plot_threshold_analysis(
                y_true, y_prob, save_path=output_dir / "threshold_analysis.png"
            )
            plt.close(fig)
            saved_files.append(output_dir / "threshold_analysis.png")
        except Exception as e:
            print(f"[WARNING] 阈值分析图绘制失败: {e}")

        return saved_files

    @staticmethod
    def save_roc_curve(
        y_true: np.ndarray,
        y_score: np.ndarray,
        path: Path,
        pos_label: int = 1,
        title: Optional[str] = None,
    ):
        """
        Save ROC curve to file.

        Args:
            y_true: True labels
            y_score: Predicted scores/probabilities for positive class
            path: Output path
            pos_label: Positive class label (default=1)
            title: Plot title
        """
        from sklearn.metrics import roc_curve, auc

        fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=pos_label)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(
            fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})"
        )
        ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(title or "ROC Curve")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"[SUCCESS] ROC curve saved: {path}")
        return fig

    @staticmethod
    def save_calibration_curve(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        path: Path,
        n_bins: int,
        ece_value: float,
        warn_small_sample: bool = False,
        title: Optional[str] = None,
    ):
        """
        Save calibration curve with ECE annotation.

        Args:
            y_true: True labels (0 or 1)
            y_prob: Predicted probabilities for positive class
            path: Output path
            n_bins: Number of bins used
            ece_value: ECE value to annotate
            warn_small_sample: Show warning if bins were reduced due to small sample
            title: Plot title
        """
        from sklearn.calibration import calibration_curve

        # Compute calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_prob, n_bins=n_bins, strategy="uniform"
        )

        fig, ax = plt.subplots(figsize=(8, 6))

        # Plot calibration curve
        ax.plot(
            mean_predicted_value,
            fraction_of_positives,
            "s-",
            label=f"Model (ECE={ece_value:.4f})",
            linewidth=2,
            markersize=8,
        )
        ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration", linewidth=2)

        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Fraction of Positives")
        ax.set_title(title or f"Calibration Curve (bins={n_bins})")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

        # Annotate ECE
        ax.text(
            0.05,
            0.95,
            f"ECE = {ece_value:.4f}",
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        # Warning for small sample
        if warn_small_sample:
            ax.text(
                0.5,
                0.5,
                "WARNING: Small sample: bins reduced",
                transform=ax.transAxes,
                fontsize=10,
                color="red",
                ha="center",
                va="center",
                bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.7),
            )

        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"[SUCCESS] Calibration curve saved: {path}")
        return fig

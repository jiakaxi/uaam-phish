"""
Lightning callback for automatically appending experiment results to documentation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pytorch_lightning.callbacks import Callback

from src.utils.documentation import DocumentationAppender
from src.utils.logging import get_logger

log = get_logger(__name__)


class DocumentationCallback(Callback):
    """
    Callback to automatically append experiment results to project documentation.

    This callback appends to:
    - FINAL_SUMMARY_CN.md (if enabled)
    - CHANGES_SUMMARY.md (if enabled)

    Usage:
        >>> callback = DocumentationCallback(
        ...     feature_name="URL Baseline Experiment",
        ...     append_to_summary=True,
        ...     append_to_changes=False,
        ... )
        >>> trainer = Trainer(callbacks=[callback])
    """

    def __init__(
        self,
        feature_name: str,
        append_to_summary: bool = True,
        append_to_changes: bool = False,
        append_to_manifest: bool = False,
        project_root: Optional[Path] = None,
        custom_summary: Optional[str] = None,
        custom_deliverables: Optional[list] = None,
    ):
        """
        Initialize documentation callback.

        Args:
            feature_name: Name of the feature/experiment
            append_to_summary: Whether to append to FINAL_SUMMARY_CN.md
            append_to_changes: Whether to append to CHANGES_SUMMARY.md
            append_to_manifest: Whether to append to FILES_MANIFEST.md
            project_root: Project root directory
            custom_summary: Custom summary text
            custom_deliverables: Custom deliverables list
        """
        super().__init__()
        self.feature_name = feature_name
        self.append_to_summary = append_to_summary
        self.append_to_changes = append_to_changes
        self.append_to_manifest = append_to_manifest
        self.custom_summary = custom_summary
        self.custom_deliverables = custom_deliverables

        self.doc = DocumentationAppender(root_dir=project_root)
        self.test_results = {}

    def on_test_end(self, trainer, pl_module):
        """Append results after testing completes."""
        if not (
            self.append_to_summary or self.append_to_changes or self.append_to_manifest
        ):
            return

        # Gather metrics from trainer
        metrics = trainer.logged_metrics

        test_acc = metrics.get("test_acc", metrics.get("test_accuracy", 0.0))
        test_auroc = metrics.get("test_auroc", 0.0)
        test_f1 = metrics.get("test_f1", 0.0)
        test_loss = metrics.get("test_loss", 0.0)
        test_nll = metrics.get("test_nll", 0.0)
        test_ece = metrics.get("test_ece", 0.0)

        log.info(f"\n{'='*60}")
        log.info(f"追加实验结果到文档: {self.feature_name}")
        log.info(f"{'='*60}")

        # Append to FINAL_SUMMARY_CN.md
        if self.append_to_summary:
            summary = (
                self.custom_summary
                or f"""
实验完成，模型在测试集上的性能如下：

**测试指标**:
- 准确率 (Accuracy): {test_acc:.4f}
- AUROC: {test_auroc:.4f}
- F1 Score: {test_f1:.4f}
- Loss: {test_loss:.4f}
"""
            )
            if test_nll > 0:
                summary += f"- NLL: {test_nll:.4f}\n"
            if test_ece > 0:
                summary += f"- ECE: {test_ece:.4f}\n"

            deliverables = self.custom_deliverables or [
                f"测试准确率: {test_acc:.4f}",
                f"测试 AUROC: {test_auroc:.4f}",
                f"测试 F1: {test_f1:.4f}",
            ]

            features = [
                f"[OK] 准确率: {test_acc:.2%}",
                f"[OK] AUROC: {test_auroc:.2%}",
                f"[OK] F1 Score: {test_f1:.2%}",
            ]

            self.doc.append_to_summary(
                feature_name=self.feature_name,
                status="[完成并验证]",
                summary=summary,
                deliverables=deliverables,
                features=features,
                test_results=f"[OK] 测试完成 - Acc: {test_acc:.2%}, AUROC: {test_auroc:.2%}",
            )

        # Append to CHANGES_SUMMARY.md
        if self.append_to_changes:
            stats = {
                "测试准确率": f"{test_acc:.4f}",
                "测试 AUROC": f"{test_auroc:.4f}",
                "测试 F1": f"{test_f1:.4f}",
            }

            self.doc.append_to_changes(
                feature_name=self.feature_name,
                implementation_type="实验运行",
                new_features=[
                    f"完成模型测试，准确率 {test_acc:.2%}",
                ],
                stats=stats,
            )

        log.info("[OK] 文档追加完成")
        log.info(f"{'='*60}\n")

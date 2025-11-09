"""
Visual-only Lightning Module
单模态 Visual 钓鱼检测系统，与 html_only_module.py 架构对齐
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
from omegaconf import OmegaConf

from src.models.visual_encoder import VisualEncoder
from src.utils.metrics import compute_ece, compute_nll
from src.utils.logging import get_logger

log = get_logger(__name__)


class VisualOnlyModule(pl.LightningModule):
    """
    Lightning module for Visual-only phishing detection.
    Mirrors html_only_module.py architecture for consistency.

    Key design decisions:
    - Single logit output + BCEWithLogitsLoss (matches HTML-only)
    - Metric naming: val/auroc, test/ece, test/nll (slash separator)
    - Supports both tuple and dict batch formats
    - ECE/NLL computed using src.utils.metrics (same as HTML-only)
    - Exports embeddings_test.csv in on_test_epoch_end
    """

    def __init__(self, cfg: Any) -> None:
        super().__init__()
        self.cfg = cfg

        try:
            cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        except (AttributeError, TypeError):
            cfg_dict = cfg
        self.save_hyperparameters({"cfg": cfg_dict})

        model_cfg = cfg.model

        # Visual Encoder (ResNet-50)
        self.encoder = VisualEncoder(
            pretrained=model_cfg.get("pretrained", True),
            freeze_backbone=model_cfg.get("freeze_backbone", False),
            embedding_dim=model_cfg.get("embedding_dim", 256),
            dropout=model_cfg.get("dropout", 0.1),
        )

        # 单 logit 输出（与 HTML-only 一致）
        embedding_dim = model_cfg.get("embedding_dim", 256)
        self.classifier = nn.Linear(embedding_dim, 1)

        # Loss function with optional pos_weight for class imbalance
        pos_weight = model_cfg.get("pos_weight", None)
        if pos_weight is not None:
            pos_weight = torch.tensor([float(pos_weight)])
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            log.info(f">> Using pos_weight={pos_weight.item()} for BCEWithLogitsLoss")
        else:
            self.criterion = nn.BCEWithLogitsLoss()

        # Initialize step metrics (Accuracy, AUROC, F1-macro)
        metrics_cfg = cfg.get("metrics", {})
        average = metrics_cfg.get("average", "macro")

        # Binary task metrics
        self.train_metrics = nn.ModuleDict(
            {
                "acc": torchmetrics.Accuracy(task="binary"),
                "auroc": torchmetrics.AUROC(task="binary"),
                "f1": torchmetrics.F1Score(task="binary", average=average),
            }
        )
        self.val_metrics = nn.ModuleDict(
            {
                "acc": torchmetrics.Accuracy(task="binary"),
                "auroc": torchmetrics.AUROC(task="binary"),
                "f1": torchmetrics.F1Score(task="binary", average=average),
            }
        )
        self.test_metrics = nn.ModuleDict(
            {
                "acc": torchmetrics.Accuracy(task="binary"),
                "auroc": torchmetrics.AUROC(task="binary"),
                "f1": torchmetrics.F1Score(task="binary", average=average),
            }
        )

        # For epoch-level metrics (NLL, ECE) and embeddings export
        self.validation_step_outputs: List[Dict] = []
        self.test_step_outputs: List[Dict] = []

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Returns encoded representation z_visual ∈ R^256."""
        return self.encoder(images)

    def predict_logits(self, images: torch.Tensor) -> torch.Tensor:
        """Returns single logit (batch, 1)"""
        z = self.forward(images)
        return self.classifier(z)

    def _unpack_batch(
        self, batch: Union[Tuple, Dict]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        解包批量数据，支持 tuple 或 dict 格式。

        Returns:
            images, labels
        """
        if isinstance(batch, dict):
            return batch["image"], batch["label"]
        else:
            # Tuple format: (images, labels)
            return batch[0], batch[1]

    def _shared_step(
        self, batch: Union[Tuple, Dict], stage: str
    ) -> Dict[str, torch.Tensor]:
        images, labels = self._unpack_batch(batch)

        # 转换 labels 为 float（BCEWithLogitsLoss 需要）
        labels = labels.float()

        logits = self.predict_logits(images).squeeze(1)  # (batch,)
        loss = self.criterion(logits, labels)

        # 计算概率和预测
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).long()
        acc = (preds == labels.long()).float().mean()

        self.log(
            f"{stage}_loss",
            loss,
            prog_bar=stage != "train",
            on_step=False,
            on_epoch=True,
        )
        self.log(
            f"{stage}_acc", acc, prog_bar=stage == "val", on_step=False, on_epoch=True
        )

        return {"loss": loss, "acc": acc, "logits": logits, "probs": probs}

    def training_step(self, batch, batch_idx):
        result = self._shared_step(batch, "train")
        return result["loss"]

    def validation_step(self, batch, batch_idx):
        result = self._shared_step(batch, "val")
        images, labels = self._unpack_batch(batch)
        probs = result["probs"]

        # Step metrics
        for name, metric in self.val_metrics.items():
            value = metric(probs, labels.long())
            sync_dist = (
                self.cfg.get("metrics", {}).get("dist", {}).get("sync_metrics", False)
            )
            self.log(
                f"val/{name}",  # 使用 val/ 前缀统一命名
                value,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
                sync_dist=sync_dist,
            )

        # Store for epoch-level metrics
        self.validation_step_outputs.append(
            {
                "logits": result["logits"].detach(),
                "labels": labels.detach(),
                "probs": probs.detach(),
            }
        )

        return {"val_loss": result["loss"], "val_acc": result["acc"]}

    def test_step(self, batch, batch_idx):
        result = self._shared_step(batch, "test")
        images, labels = self._unpack_batch(batch)
        probs = result["probs"]

        # Get embeddings for export
        with torch.no_grad():
            embeddings = self.forward(images)  # (batch, 256)

        # Step metrics
        for name, metric in self.test_metrics.items():
            value = metric(probs, labels.long())
            sync_dist = (
                self.cfg.get("metrics", {}).get("dist", {}).get("sync_metrics", False)
            )
            self.log(
                f"test/{name}",  # 使用 test/ 前缀统一命名
                value,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
                sync_dist=sync_dist,
            )

        # Store for epoch-level metrics and embeddings export
        self.test_step_outputs.append(
            {
                "logits": result["logits"].detach(),
                "labels": labels.detach(),
                "probs": probs.detach(),
                "embeddings": embeddings.detach(),
            }
        )

        return {
            "test_loss": result["loss"],
            "test_acc": result["acc"],
            "y_true": labels,
            "y_prob": probs,
        }

    def on_validation_epoch_end(self):
        """Compute epoch-level metrics: NLL, ECE"""
        if len(self.validation_step_outputs) == 0:
            return

        all_logits = torch.cat([x["logits"] for x in self.validation_step_outputs])
        all_labels = torch.cat([x["labels"] for x in self.validation_step_outputs])
        all_probs = torch.cat([x["probs"] for x in self.validation_step_outputs])

        # NLL（需要转换为 2-class logits 格式）
        logits_2class = torch.stack([-all_logits, all_logits], dim=1)  # (N, 2)
        nll = compute_nll(logits_2class, all_labels.long())

        # ECE with adaptive bins
        # Convert to float32 first to avoid BFloat16 numpy conversion issues
        y_true_np = all_labels.cpu().float().numpy()
        y_prob_np = all_probs.cpu().float().numpy()
        ece_value, bins_used, low_sample_warning = compute_ece(
            y_true_np, y_prob_np, n_bins=None, pos_label=1
        )

        sync_dist = (
            self.cfg.get("metrics", {}).get("dist", {}).get("sync_metrics", False)
        )
        self.log("val/nll", nll, prog_bar=False, sync_dist=sync_dist)
        self.log("val/ece", ece_value, prog_bar=False, sync_dist=sync_dist)
        self.log(
            "val/ece_low_sample_warning",
            float(low_sample_warning),
            prog_bar=False,
            sync_dist=sync_dist,
        )

        self.validation_step_outputs.clear()

    def on_test_epoch_end(self):
        """Compute epoch-level metrics: NLL, ECE, and export embeddings"""
        if len(self.test_step_outputs) == 0:
            return

        all_logits = torch.cat([x["logits"] for x in self.test_step_outputs])
        all_labels = torch.cat([x["labels"] for x in self.test_step_outputs])
        all_probs = torch.cat([x["probs"] for x in self.test_step_outputs])
        all_embeddings = torch.cat([x["embeddings"] for x in self.test_step_outputs])

        # NLL
        logits_2class = torch.stack([-all_logits, all_logits], dim=1)
        nll = compute_nll(logits_2class, all_labels.long())

        # ECE with adaptive bins
        # Convert to float32 first to avoid BFloat16 numpy conversion issues
        y_true_np = all_labels.cpu().float().numpy()
        y_prob_np = all_probs.cpu().float().numpy()
        ece_value, bins_used, low_sample_warning = compute_ece(
            y_true_np, y_prob_np, n_bins=None, pos_label=1
        )

        sync_dist = (
            self.cfg.get("metrics", {}).get("dist", {}).get("sync_metrics", False)
        )
        self.log("test/nll", nll, prog_bar=False, sync_dist=sync_dist)
        self.log("test/ece", ece_value, prog_bar=False, sync_dist=sync_dist)
        self.log("test/ece_bins", float(bins_used), prog_bar=False, sync_dist=sync_dist)
        self.log(
            "test/ece_low_sample_warning",
            float(low_sample_warning),
            prog_bar=False,
            sync_dist=sync_dist,
        )

        # Compute final test metrics for sanity check
        final_acc = (all_probs > 0.5).long().eq(all_labels.long()).float().mean().item()

        # AUROC
        try:
            from sklearn.metrics import roc_auc_score

            final_auroc = roc_auc_score(y_true_np, y_prob_np)
        except Exception:
            final_auroc = 0.0

        # Sanity check (DoD requirement)
        if final_acc < 0.70 or final_auroc < 0.75:
            log.warning("=" * 70)
            log.warning("⚠️  SANITY CHECK WARNING:")
            log.warning(f"   Test Accuracy: {final_acc:.4f} (threshold: ≥ 0.70)")
            log.warning(f"   Test AUROC:    {final_auroc:.4f} (threshold: ≥ 0.75)")
            log.warning("   Performance below expected baseline. Consider:")
            log.warning("   - Checking data quality/preprocessing")
            log.warning("   - Increasing training epochs")
            log.warning("   - Adjusting learning rate or batch size")
            log.warning("=" * 70)
        else:
            log.info("=" * 70)
            log.info("✅ SANITY CHECK PASSED:")
            log.info(f"   Test Accuracy: {final_acc:.4f} ≥ 0.70")
            log.info(f"   Test AUROC:    {final_auroc:.4f} ≥ 0.75")
            log.info("=" * 70)

        # Export embeddings_test.csv
        try:
            # Find results directory from experiment tracker
            # Try multiple possible locations
            if hasattr(self.trainer, "logger") and hasattr(
                self.trainer.logger, "log_dir"
            ):
                results_dir = (
                    Path(self.trainer.logger.log_dir).parent.parent / "results"
                )
            elif hasattr(self.trainer, "default_root_dir"):
                results_dir = Path(self.trainer.default_root_dir) / "results"
            else:
                # Fallback: create in current directory
                results_dir = Path("experiments") / "results"

            results_dir.mkdir(parents=True, exist_ok=True)

            embeddings_np = all_embeddings.cpu().numpy()
            embeddings_df = pd.DataFrame(
                embeddings_np,
                columns=[f"emb_{i}" for i in range(embeddings_np.shape[1])],
            )

            # Add sample IDs (use indices if dataset doesn't provide IDs)
            # Try to get IDs from test dataset
            test_dataset = self.trainer.datamodule.test_dataset
            if hasattr(test_dataset, "_ids"):
                embeddings_df.insert(0, "id", test_dataset._ids[: len(embeddings_df)])
            else:
                embeddings_df.insert(
                    0, "id", [str(i) for i in range(len(embeddings_df))]
                )

            embeddings_path = results_dir / "embeddings_test.csv"
            embeddings_df.to_csv(embeddings_path, index=False)
            log.info(f"[SUCCESS] Exported embeddings to: {embeddings_path}")
            log.info(f"   Shape: {embeddings_df.shape} (samples x features)")
        except Exception as e:
            log.error(f"[ERROR] Failed to export embeddings: {e}")
            import traceback

            log.error(traceback.format_exc())

        self.test_step_outputs.clear()

    def configure_optimizers(self):
        lr = self.cfg.train.get("lr", 1e-4)
        weight_decay = self.cfg.train.get("weight_decay", 1e-4)
        return torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)

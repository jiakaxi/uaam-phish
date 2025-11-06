from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from src.models.url_encoder import URLEncoder
from src.utils.metrics import get_step_metrics, compute_ece, compute_nll
from src.utils.logging import get_logger

log = get_logger(__name__)


class UrlOnlyModule(pl.LightningModule):
    """
    LightningModule wrapping a URLEncoder and a linear classifier.
    Exports embeddings_test.csv in on_test_epoch_end.
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
        self.encoder = URLEncoder(
            vocab_size=model_cfg.vocab_size,
            embedding_dim=model_cfg.embedding_dim,
            hidden_dim=model_cfg.hidden_dim,
            num_layers=model_cfg.num_layers,
            bidirectional=model_cfg.bidirectional,
            dropout=model_cfg.dropout,
            pad_id=model_cfg.pad_id,
            proj_dim=model_cfg.proj_dim,
        )
        # Safety assertion: URL encoder must remain frozen per thesis
        assert (
            self.encoder.bidirectional
            and model_cfg.num_layers == 2
            and model_cfg.hidden_dim == 128
            and model_cfg.proj_dim == 256
        ), "URL encoder must remain a 2-layer BiLSTM (char-level, 256-dim) per thesis."
        self.classifier = nn.Linear(model_cfg.proj_dim, model_cfg.num_classes)
        self.criterion = nn.CrossEntropyLoss()

        # Initialize step metrics (Accuracy, AUROC, F1-macro)
        metrics_cfg = cfg.get("metrics", {})
        sync_dist = metrics_cfg.get("dist", {}).get("sync_metrics", False)
        average = metrics_cfg.get("average", "macro")

        self.train_metrics = nn.ModuleDict(
            get_step_metrics(model_cfg.num_classes, average, sync_dist)
        )
        self.val_metrics = nn.ModuleDict(
            get_step_metrics(model_cfg.num_classes, average, sync_dist)
        )
        self.test_metrics = nn.ModuleDict(
            get_step_metrics(model_cfg.num_classes, average, sync_dist)
        )

        # For epoch-level metrics (NLL, ECE)
        self.validation_step_outputs: List[Dict] = []
        self.test_step_outputs: List[Dict] = []

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Returns encoded representation z_url ∈ R^proj_dim."""
        return self.encoder(input_ids)

    def predict_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        z = self.forward(input_ids)
        return self.classifier(z)

    def _shared_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], stage: str
    ) -> Dict[str, torch.Tensor]:
        input_ids, labels = batch
        logits = self.predict_logits(input_ids)
        loss = self.criterion(logits, labels)
        preds = logits.argmax(dim=1)
        acc = (preds == labels).float().mean()

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
        return {"loss": loss, "acc": acc, "logits": logits}

    def training_step(self, batch, batch_idx):
        result = self._shared_step(batch, "train")
        return result["loss"]

    def validation_step(self, batch, batch_idx):
        result = self._shared_step(batch, "val")
        input_ids, labels = batch
        logits = result["logits"]
        probs = torch.softmax(logits, dim=1)

        # Step metrics (Accuracy, AUROC, F1)
        # For binary classification, pass probabilities of positive class (class 1)
        y_prob = probs[:, 1] if probs.shape[1] == 2 else probs
        for name, metric in self.val_metrics.items():
            value = metric(y_prob, labels)
            sync_dist = (
                self.cfg.get("metrics", {}).get("dist", {}).get("sync_metrics", False)
            )
            self.log(
                f"val_{name}",
                value,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
                sync_dist=sync_dist,
            )

        # Store for epoch-level metrics
        self.validation_step_outputs.append(
            {
                "logits": logits.detach(),
                "labels": labels.detach(),
                "probs": probs.detach(),
            }
        )

        return {"val_loss": result["loss"], "val_acc": result["acc"]}

    def test_step(self, batch, batch_idx):
        result = self._shared_step(batch, "test")
        input_ids, labels = batch

        # Get embeddings for export
        with torch.no_grad():
            embeddings = self.forward(input_ids)  # (batch, 256)

        # 计算预测概率用于可视化
        logits = result["logits"]
        probs = torch.softmax(logits, dim=1)
        y_prob = probs[:, 1]  # 钓鱼类别的概率

        # Step metrics (Accuracy, AUROC, F1)
        # For binary classification, pass probabilities of positive class (class 1)
        for name, metric in self.test_metrics.items():
            value = metric(y_prob, labels)
            sync_dist = (
                self.cfg.get("metrics", {}).get("dist", {}).get("sync_metrics", False)
            )
            self.log(
                f"test_{name}",
                value,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
                sync_dist=sync_dist,
            )

        # Store for epoch-level metrics and embeddings export
        self.test_step_outputs.append(
            {
                "logits": logits.detach(),
                "labels": labels.detach(),
                "probs": probs.detach(),
                "embeddings": embeddings.detach(),
            }
        )

        return {
            "test_loss": result["loss"],
            "test_acc": result["acc"],
            "y_true": labels,
            "y_prob": y_prob,
        }

    def on_validation_epoch_end(self):
        """Compute epoch-level metrics: NLL, ECE"""
        if len(self.validation_step_outputs) == 0:
            return

        # Gather all outputs
        all_logits = torch.cat([x["logits"] for x in self.validation_step_outputs])
        all_labels = torch.cat([x["labels"] for x in self.validation_step_outputs])
        all_probs = torch.cat([x["probs"] for x in self.validation_step_outputs])

        # NLL
        nll = compute_nll(all_logits, all_labels)

        # ECE with adaptive bins
        y_true_np = all_labels.cpu().numpy()
        y_prob_np = all_probs[:, 1].cpu().numpy()  # Probability of positive class
        ece_value, bins_used = compute_ece(
            y_true_np, y_prob_np, n_bins=None, pos_label=1
        )

        # Log epoch metrics
        sync_dist = (
            self.cfg.get("metrics", {}).get("dist", {}).get("sync_metrics", False)
        )
        self.log("val_nll", nll, prog_bar=False, sync_dist=sync_dist)
        self.log("val_ece", ece_value, prog_bar=False, sync_dist=sync_dist)

        # Clear outputs
        self.validation_step_outputs.clear()

    def on_test_epoch_end(self):
        """Compute epoch-level metrics: NLL, ECE, and export embeddings"""
        if len(self.test_step_outputs) == 0:
            return

        # Gather all outputs
        all_logits = torch.cat([x["logits"] for x in self.test_step_outputs])
        all_labels = torch.cat([x["labels"] for x in self.test_step_outputs])
        all_probs = torch.cat([x["probs"] for x in self.test_step_outputs])
        all_embeddings = torch.cat([x["embeddings"] for x in self.test_step_outputs])

        # NLL
        nll = compute_nll(all_logits, all_labels)

        # ECE with adaptive bins
        y_true_np = all_labels.cpu().numpy()
        y_prob_np = all_probs[:, 1].cpu().numpy()  # Probability of positive class
        ece_value, bins_used = compute_ece(
            y_true_np, y_prob_np, n_bins=None, pos_label=1
        )

        # Log epoch metrics
        sync_dist = (
            self.cfg.get("metrics", {}).get("dist", {}).get("sync_metrics", False)
        )
        self.log("test_nll", nll, prog_bar=False, sync_dist=sync_dist)
        self.log("test_ece", ece_value, prog_bar=False, sync_dist=sync_dist)
        self.log("test_ece_bins", float(bins_used), prog_bar=False, sync_dist=sync_dist)

        # Log metrics summary
        log.info("=" * 70)
        log.info("[URL-only] Test Epoch Metrics Summary:")
        log.info(f"  NLL:      {nll:.4f}")
        log.info(f"  ECE:      {ece_value:.4f} (bins={bins_used})")

        # Get step metrics from test_metrics
        for name, metric in self.test_metrics.items():
            value = metric.compute()
            log.info(f"  {name.upper()}: {value:.4f}")

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
            log.info(f"[SUCCESS] Exported URL embeddings to: {embeddings_path}")
            log.info(f"   Shape: {embeddings_df.shape} (samples x features)")
        except Exception as e:
            log.error(f"[ERROR] Failed to export URL embeddings: {e}")
            import traceback

            log.error(traceback.format_exc())

        # Clear outputs
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.cfg.train.lr)


# Backwards compatibility alias
UrlOnlySystem = UrlOnlyModule

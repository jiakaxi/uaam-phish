"""
多模态拼接基线系统 (S0: Multimodal Baseline)

Early Fusion via Concatenation:
- URL Encoder (BiLSTM) -> 256-D
- HTML Encoder (BERT) -> 256-D
- Visual Encoder (ResNet-50) -> 256-D
- Fusion: Concat [768-D] -> Linear -> Logits [1-D]

Loss: BCEWithLogitsLoss (logits-only, no Sigmoid in model)
Metrics: Accuracy, AUROC, F1 (macro)
Artifacts: ROC curve, calibration curve, predictions CSV, metrics JSON
"""

from __future__ import annotations

from typing import Dict, Any, Optional

import torch
import pytorch_lightning as pl
from torchmetrics import Accuracy, AUROC, F1Score

from src.models.url_encoder import URLEncoder
from src.models.html_encoder import HTMLEncoder
from src.models.visual_encoder import VisualEncoder
from src.modules.fusion.baseline_concat import BaselineConcatFusion
from src.utils.protocol_artifacts import ArtifactsWriter
from src.utils.logging import get_logger

log = get_logger(__name__)


class MultimodalBaselineSystem(pl.LightningModule):
    """
    多模态拼接基线系统 (S0: Early Fusion Baseline).

    Architecture:
        z_url = URLEncoder(url)          # [B, 256]
        z_html = HTMLEncoder(html)       # [B, 256]
        z_visual = VisualEncoder(image)  # [B, 256]
        logits = ConcatFusion(z_url, z_html, z_visual)  # [B, 1]

    Training:
        Loss: BCEWithLogitsLoss (with optional pos_weight for class imbalance)
        Optimizer: AdamW
        Scheduler: CosineAnnealingLR

    Metrics:
        - Accuracy (binary, threshold=0.5)
        - AUROC (binary, pos_label=1)
        - F1 Score (binary, macro average)

    Artifacts (saved at validation/test epoch end):
        - preds_{val,test}.csv
        - metrics_{val,test}.json
        - roc_{val,test}.png
        - reliability_{val,test}_before_ts.png
    """

    def __init__(
        self,
        # URL Encoder
        url_vocab_size: int = 128,
        url_embed_dim: int = 64,
        url_hidden_dim: int = 128,
        url_num_layers: int = 2,
        url_max_len: int = 200,
        # HTML Encoder
        html_model_name: str = "bert-base-uncased",
        html_hidden_dim: int = 768,
        html_projection_dim: int = 256,
        html_freeze_bert: bool = False,
        # Visual Encoder
        visual_model_name: str = "resnet50",
        visual_pretrained: bool = True,
        visual_projection_dim: int = 256,
        visual_freeze_backbone: bool = False,
        # Training
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        dropout: float = 0.1,
        pos_weight: Optional[float] = None,
        # Hydra compatibility
        cfg=None,  # 兼容 Hydra 实例化时传入的 cfg 参数（忽略）
        **kwargs,  # 捕获其他未知参数
    ):
        super().__init__()
        self.save_hyperparameters()

        # === Encoders ===
        self.url_encoder = URLEncoder(
            vocab_size=url_vocab_size,
            embedding_dim=url_embed_dim,
            hidden_dim=url_hidden_dim,
            num_layers=url_num_layers,
            bidirectional=True,
            dropout=dropout,
            proj_dim=256,
        )

        self.html_encoder = HTMLEncoder(
            bert_model=html_model_name,
            hidden_dim=html_hidden_dim,
            output_dim=html_projection_dim,
            dropout=dropout,
            freeze_bert=html_freeze_bert,
        )

        self.visual_encoder = VisualEncoder(
            pretrained=visual_pretrained,
            freeze_backbone=visual_freeze_backbone,
            embedding_dim=visual_projection_dim,
            dropout=dropout,
        )

        # === Fusion Module ===
        self.fusion = BaselineConcatFusion(
            url_dim=256,
            html_dim=256,
            visual_dim=256,
            dropout=dropout,
        )

        # === Loss Function ===
        if pos_weight is not None:
            pos_weight_tensor = torch.tensor([pos_weight])
            self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        else:
            self.loss_fn = torch.nn.BCEWithLogitsLoss()

        # === Metrics ===
        # Train
        self.train_acc = Accuracy(task="binary")
        self.train_auroc = AUROC(task="binary")
        self.train_f1 = F1Score(task="binary", average="macro")

        # Val
        self.val_acc = Accuracy(task="binary")
        self.val_auroc = AUROC(task="binary")
        self.val_f1 = F1Score(task="binary", average="macro")

        # Test
        self.test_acc = Accuracy(task="binary")
        self.test_auroc = AUROC(task="binary")
        self.test_f1 = F1Score(task="binary", average="macro")

        # === Artifacts Writer ===
        self.artifacts_writer = None  # Initialized on first use

        # === Prediction Cache (for artifact generation) ===
        self._preds = {"val": [], "test": []}

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        """
        Forward pass through three encoders and fusion module.

        Args:
            batch: Dict with keys:
                - "url": [B, seq_len] (LongTensor)
                - "html": Dict with "input_ids" [B, seq_len], "attention_mask" [B, seq_len]
                - "visual": [B, 3, 224, 224] (FloatTensor)

        Returns:
            logits: [B, 1] - Raw classification logits (no Sigmoid)
        """
        # URL encoding
        z_url = self.url_encoder(batch["url"])  # [B, 256]

        # HTML encoding
        z_html = self.html_encoder(
            input_ids=batch["html"]["input_ids"],
            attention_mask=batch["html"]["attention_mask"],
        )  # [B, 256]

        # Visual encoding
        z_visual = self.visual_encoder(batch["visual"])  # [B, 256]

        # Fusion
        logits = self.fusion(z_url, z_html, z_visual)  # [B, 1]

        return logits

    def _shared_step(self, batch: Dict[str, Any], stage: str):
        """
        Shared step for train/val/test.

        Args:
            batch: Input batch
            stage: "train", "val", or "test"

        Returns:
            loss: Scalar loss value
        """
        # Forward pass
        logits = self(batch)  # [B, 1]

        # Get labels
        labels = batch["label"].float().unsqueeze(1)  # [B, 1]

        # Compute loss
        loss = self.loss_fn(logits, labels)

        # Compute probabilities and predictions
        probs = torch.sigmoid(logits).squeeze()  # [B]
        preds = (probs > 0.5).long()  # [B]
        labels_int = labels.long().squeeze()  # [B]

        # Update metrics
        if stage == "train":
            self.train_acc(preds, labels_int)
            self.train_auroc(probs, labels_int)
            self.train_f1(preds, labels_int)

            self.log_dict(
                {
                    "train/loss": loss,
                    "train/acc": self.train_acc,
                    "train/auroc": self.train_auroc,
                    "train/f1": self.train_f1,
                },
                prog_bar=True,
                on_step=False,
                on_epoch=True,
            )

        elif stage == "val":
            self.val_acc(preds, labels_int)
            self.val_auroc(probs, labels_int)
            self.val_f1(preds, labels_int)

            self.log_dict(
                {
                    "val/loss": loss,
                    "val/acc": self.val_acc,
                    "val/auroc": self.val_auroc,
                    "val/f1": self.val_f1,
                },
                prog_bar=True,
                on_step=False,
                on_epoch=True,
            )

            # Cache predictions for artifact generation
            self._preds["val"].append(
                {
                    "id": batch.get("id", None),
                    "y_true": labels_int.detach().cpu(),
                    "logit": logits.detach().cpu().squeeze(),
                    "prob": probs.detach().cpu(),
                }
            )

        elif stage == "test":
            self.test_acc(preds, labels_int)
            self.test_auroc(probs, labels_int)
            self.test_f1(preds, labels_int)

            self.log_dict(
                {
                    "test/loss": loss,
                    "test/acc": self.test_acc,
                    "test/auroc": self.test_auroc,
                    "test/f1": self.test_f1,
                },
                prog_bar=True,
                on_step=False,
                on_epoch=True,
            )

            # Cache predictions for artifact generation
            self._preds["test"].append(
                {
                    "id": batch.get("id", None),
                    "y_true": labels_int.detach().cpu(),
                    "logit": logits.detach().cpu().squeeze(),
                    "prob": probs.detach().cpu(),
                }
            )

        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    def on_validation_epoch_end(self):
        """Save validation artifacts at end of validation epoch."""
        if len(self._preds["val"]) > 0:
            # Lazy init artifacts writer
            if self.artifacts_writer is None:
                self.artifacts_writer = ArtifactsWriter(self)

            self.artifacts_writer.save_validation_artifacts(self._preds["val"])
            self._preds["val"].clear()

    def on_test_epoch_end(self):
        """Save test artifacts at end of test epoch."""
        if len(self._preds["test"]) > 0:
            # Lazy init artifacts writer
            if self.artifacts_writer is None:
                self.artifacts_writer = ArtifactsWriter(self)

            self.artifacts_writer.save_test_artifacts(self._preds["test"])
            self._preds["test"].clear()

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

        # Cosine annealing scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=1e-6,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }

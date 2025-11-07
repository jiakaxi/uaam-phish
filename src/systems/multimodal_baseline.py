"""
S0 Multimodal Baseline system (thesis Sec. 4.6).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

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
    Early-fusion concatenation baseline (Sec. 4.6.1).

    Encoders (all 256-d outputs):
        - URL: char-level BiLSTM (2-layer, hidden=128, embedding_dim=64)
        - HTML: bert-base-uncased + projection head
        - Visual: ResNet-50 (ImageNet) + projection head
    Fusion:
        z_fused = concat([z_url, z_html, z_visual]) in R^768 -> Linear(768->1) -> logits
    Training (Sec. 4.6.3):
        - Loss: BCEWithLogitsLoss
        - Optimizer: AdamW with grouped learning rates
        - Scheduler: CosineAnnealingLR(eta_min=1e-6)
    """

    def __init__(
        self,
        # URL encoder
        url_vocab_size: int = 128,
        url_embed_dim: int = 64,
        url_hidden_dim: int = 128,
        url_num_layers: int = 2,
        url_max_len: int = 200,
        # HTML encoder
        html_model_name: str = "bert-base-uncased",
        html_hidden_dim: int = 768,
        html_projection_dim: int = 256,
        html_freeze_bert: bool = False,
        # Visual encoder
        visual_model_name: str = "resnet50",
        visual_pretrained: bool = True,
        visual_projection_dim: int = 256,
        visual_freeze_backbone: bool = False,
        # Training hyperparameters
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        dropout: float = 0.1,
        pos_weight: Optional[float] = None,
        cfg: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["cfg"])
        self.cfg = cfg

        # Encoders (Sec. 4.6.1)
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

        self.fusion = BaselineConcatFusion(
            url_dim=256,
            html_dim=256,
            visual_dim=256,
            dropout=dropout,
        )

        # Loss (Sec. 4.6.3)
        if pos_weight is not None:
            pos_weight_tensor = torch.tensor([pos_weight])
            self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        else:
            self.loss_fn = torch.nn.BCEWithLogitsLoss()

        # Metrics (AUROC primary, plus accuracy/F1)
        self.train_acc = Accuracy(task="binary")
        self.train_auroc = AUROC(task="binary")
        self.train_f1 = F1Score(task="binary", average="macro")

        self.val_acc = Accuracy(task="binary")
        self.val_auroc = AUROC(task="binary")
        self.val_f1 = F1Score(task="binary", average="macro")

        self.test_acc = Accuracy(task="binary")
        self.test_auroc = AUROC(task="binary")
        self.test_f1 = F1Score(task="binary", average="macro")

        # Artifact handling (Sec. 4.6.4)
        self.artifacts_dir: Optional[Path] = None
        self.split_metadata: Dict[str, Any] = {}
        self.artifacts_writer: Optional[ArtifactsWriter] = None
        self._preds: Dict[str, list] = {"val": [], "test": []}

    # ------------------------------------------------------------------ #
    # Utility setters                                                    #
    # ------------------------------------------------------------------ #
    def set_artifact_dir(self, artifact_dir: Path | str) -> None:
        self.artifacts_dir = Path(artifact_dir)

    def set_split_metadata(self, split_metadata: Dict[str, Any]) -> None:
        self.split_metadata = split_metadata or {}
        if self.artifacts_writer:
            self.artifacts_writer.update_split_metadata(self.split_metadata)

    # ------------------------------------------------------------------ #
    # Forward & shared step                                              #
    # ------------------------------------------------------------------ #
    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        z_url = self.url_encoder(batch["url"])  # z_url ∈ R^256
        z_html = self.html_encoder(
            input_ids=batch["html"]["input_ids"],
            attention_mask=batch["html"]["attention_mask"],
        )  # z_html ∈ R^256
        z_visual = self.visual_encoder(batch["visual"])  # z_visual ∈ R^256
        z_fused = self.fusion.concat(z_url, z_html, z_visual)
        logits = self.fusion.classify(z_fused)
        return logits

    def _shared_step(
        self, batch: Dict[str, Any], stage: str
    ) -> Dict[str, torch.Tensor]:
        logits = self(batch)
        labels = batch["label"].float().unsqueeze(1)
        loss = self.loss_fn(logits, labels)

        probs = torch.sigmoid(logits).view(-1)
        preds = (probs > 0.5).long()
        labels_int = labels.long().view(-1)

        metrics_map = {
            "train": (self.train_acc, self.train_auroc, self.train_f1),
            "val": (self.val_acc, self.val_auroc, self.val_f1),
            "test": (self.test_acc, self.test_auroc, self.test_f1),
        }
        acc_metric, auroc_metric, f1_metric = metrics_map[stage]
        acc_metric(preds, labels_int)
        auroc_metric(probs, labels_int)
        f1_metric(preds, labels_int)

        self.log_dict(
            {
                f"{stage}/loss": loss,
                f"{stage}/acc": acc_metric,
                f"{stage}/auroc": auroc_metric,
                f"{stage}/f1": f1_metric,
            },
            prog_bar=stage != "train",
            on_step=False,
            on_epoch=True,
        )

        cache = self._preds.get(stage)
        if cache is not None:
            cache.append(
                {
                    "id": batch.get("id"),
                    "y_true": labels_int.detach().cpu(),
                    "logit": logits.detach().cpu(),
                    "prob": probs.detach().cpu(),
                }
            )

        return {
            "loss": loss,
            "probs": probs.detach(),
            "labels": labels_int.detach(),
        }

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        output = self._shared_step(batch, "train")
        return output["loss"]

    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        return self._shared_step(batch, "val")

    def test_step(self, batch: Dict[str, Any], batch_idx: int):
        output = self._shared_step(batch, "test")
        return {
            "loss": output["loss"],
            "y_true": output["labels"].detach().cpu(),
            "y_prob": output["probs"].detach().cpu(),
        }

    # ------------------------------------------------------------------ #
    # Artifact generation                                                #
    # ------------------------------------------------------------------ #
    def _get_artifacts_writer(self) -> ArtifactsWriter:
        if self.artifacts_writer is None:
            if self.artifacts_dir is None:
                log.warning("Artifact directory not set; defaulting to ./artifacts")
                self.artifacts_dir = Path("./artifacts")
                self.artifacts_dir.mkdir(parents=True, exist_ok=True)
            self.artifacts_writer = ArtifactsWriter(
                module=self,
                output_dir=self.artifacts_dir,
                split_metadata=self.split_metadata,
            )
        return self.artifacts_writer

    def on_validation_epoch_end(self) -> None:
        if self._preds["val"]:
            writer = self._get_artifacts_writer()
            writer.save_stage_artifacts(self._preds["val"], stage="val")
            self._preds["val"].clear()

    def on_test_epoch_end(self) -> None:
        if self._preds["test"]:
            writer = self._get_artifacts_writer()
            writer.save_stage_artifacts(self._preds["test"], stage="test")
            self._preds["test"].clear()

    # ------------------------------------------------------------------ #
    # Optimiser / scheduler (Sec. 4.6.3)                                 #
    # ------------------------------------------------------------------ #
    def configure_optimizers(self):
        bert_params = [
            p for p in self.html_encoder.bert.parameters() if p.requires_grad
        ]
        non_bert_params = []
        non_bert_params += [p for p in self.url_encoder.parameters() if p.requires_grad]
        non_bert_params += [
            p for p in self.html_encoder.projection.parameters() if p.requires_grad
        ]
        non_bert_params += [
            p for p in self.visual_encoder.parameters() if p.requires_grad
        ]
        non_bert_params += [p for p in self.fusion.parameters() if p.requires_grad]

        param_groups = []
        if bert_params:
            param_groups.append({"params": bert_params, "lr": 2e-5})
        if non_bert_params:
            param_groups.append(
                {"params": non_bert_params, "lr": self.hparams.learning_rate}
            )

        optimizer = torch.optim.AdamW(
            param_groups, weight_decay=self.hparams.weight_decay
        )

        t_max = 25
        if self.cfg and hasattr(self.cfg, "train"):
            t_max = getattr(self.cfg.train, "epochs", t_max)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=t_max,
            eta_min=1e-6,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }

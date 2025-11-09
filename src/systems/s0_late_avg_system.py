"""
Late-fusion (uniform average) multimodal baseline for S0 experiments.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch
import pytorch_lightning as pl
from torch import nn
from torchmetrics import Accuracy, AUROC, F1Score

from src.models.url_encoder import URLEncoder
from src.models.html_encoder import HTMLEncoder
from src.models.visual_encoder import VisualEncoder
from src.utils.protocol_artifacts import ArtifactsWriter
from src.utils.logging import get_logger


log = get_logger(__name__)


class S0LateAverageSystem(pl.LightningModule):
    """
    Late-fusion baseline: average the probabilities of three modality-specific heads.
    """

    def __init__(
        self,
        url_vocab_size: int = 128,
        url_embed_dim: int = 64,
        url_hidden_dim: int = 128,
        url_num_layers: int = 2,
        url_max_len: int = 200,
        html_model_name: str = "bert-base-uncased",
        html_hidden_dim: int = 768,
        html_projection_dim: int = 256,
        html_freeze_bert: bool = False,
        visual_model_name: str = "resnet50",
        visual_pretrained: bool = True,
        visual_projection_dim: int = 256,
        visual_freeze_backbone: bool = False,
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

        # Enable TF32 for faster training on Ampere+ GPUs
        if torch.cuda.is_available():
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass

        # Encoders
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

        self.url_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(256, 1))
        self.html_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(256, 1))
        self.visual_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(256, 1))

        self._pos_weight = float(pos_weight) if pos_weight is not None else None
        self.loss_fn = nn.BCEWithLogitsLoss()

        self.train_acc = Accuracy(task="binary")
        self.train_auroc = AUROC(task="binary")
        self.train_f1 = F1Score(task="binary")

        self.val_acc = Accuracy(task="binary")
        self.val_auroc = AUROC(task="binary")
        self.val_f1 = F1Score(task="binary")

        self.test_acc = Accuracy(task="binary")
        self.test_auroc = AUROC(task="binary")
        self.test_f1 = F1Score(task="binary")

        self.artifacts_dir: Optional[Path] = None
        self.split_metadata: Dict[str, Any] = {}
        self.artifacts_writer: Optional[ArtifactsWriter] = None
        self._preds: Dict[str, list] = {"val": [], "test": []}

        # Collect outputs for epoch-end metric computation (performance optimization)
        self.train_step_outputs: list = []
        self.val_step_outputs: list = []

    # ------------------------------------------------------------------ #
    def set_artifact_dir(self, artifact_dir: Path | str) -> None:
        self.artifacts_dir = Path(artifact_dir)
        log.info(">> Artifact directory set to %s", self.artifacts_dir)

    def set_split_metadata(self, split_metadata: Dict[str, Any]) -> None:
        self.split_metadata = split_metadata or {}
        if self.artifacts_writer:
            self.artifacts_writer.update_split_metadata(self.split_metadata)

    # ------------------------------------------------------------------ #
    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if self._pos_weight is None:
            return self.loss_fn(logits, labels)
        pos_weight_tensor = torch.tensor(
            [self._pos_weight], dtype=logits.dtype, device=logits.device
        )
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        return loss_fn(logits, labels)

    def _encode_modalities(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        z_url = self.url_encoder(batch["url"])
        z_html = self.html_encoder(
            input_ids=batch["html"]["input_ids"],
            attention_mask=batch["html"]["attention_mask"],
        )
        z_visual = self.visual_encoder(batch["visual"])
        return {"url": z_url, "html": z_html, "visual": z_visual}

    def _compute_logits(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        embeddings = self._encode_modalities(batch)
        return {
            "url": self.url_head(embeddings["url"]),
            "html": self.html_head(embeddings["html"]),
            "visual": self.visual_head(embeddings["visual"]),
        }

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        logits_dict = self._compute_logits(batch)
        probs = torch.stack(
            [torch.sigmoid(logits_dict[key]) for key in ("url", "html", "visual")],
            dim=0,
        )
        avg_prob = probs.mean(dim=0)
        avg_logit = torch.logit(avg_prob.clamp(min=1e-6, max=1 - 1e-6))
        return avg_logit

    def _shared_step(
        self, batch: Dict[str, Any], stage: str
    ) -> Dict[str, torch.Tensor]:
        logits_dict = self._compute_logits(batch)
        device = next(iter(logits_dict.values())).device
        labels = batch["label"].float().unsqueeze(1).to(device)
        labels_int = labels.view(-1).long()

        losses = [self._compute_loss(logits, labels) for logits in logits_dict.values()]
        loss = torch.stack(losses).mean()

        probs = torch.stack(
            [torch.sigmoid(logits) for logits in logits_dict.values()], dim=0
        ).mean(dim=0)
        preds = (probs > 0.5).long().view(-1)
        avg_logit = torch.logit(probs.clamp(min=1e-6, max=1 - 1e-6))

        # For test stage, compute metrics immediately (needed for final evaluation)
        if stage == "test":
            metrics_map = {
                "test": (self.test_acc, self.test_auroc, self.test_f1),
            }
            acc_metric, auroc_metric, f1_metric = metrics_map[stage]
            acc_metric(preds, labels_int)
            auroc_metric(probs.view(-1), labels_int)
            f1_metric(preds, labels_int)

            self.log_dict(
                {
                    f"{stage}/loss": loss,
                    f"{stage}/acc": acc_metric,
                    f"{stage}/auroc": auroc_metric,
                    f"{stage}/f1": f1_metric,
                },
                prog_bar=True,
                on_step=False,
                on_epoch=True,
            )

        # Log loss immediately (needed for training)
        self.log(
            f"{stage}/loss",
            loss,
            prog_bar=stage != "train",
            on_step=(stage == "train"),
            on_epoch=True,
            sync_dist=False,
        )

        cache = self._preds.get(stage)
        if cache is not None:
            cache.append(
                {
                    "id": batch.get("id"),
                    "y_true": labels_int.detach().cpu(),
                    "logit": avg_logit.detach().cpu(),
                    "prob": probs.detach().cpu(),
                }
            )

        return {
            "loss": loss,
            "probs": probs.detach().cpu(),
            "labels": labels_int.detach().cpu(),
            "preds": preds.detach().cpu(),
        }

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        output = self._shared_step(batch, "train")
        # Collect outputs for epoch-end metric computation
        self.train_step_outputs.append(output)
        return output["loss"]

    def on_train_epoch_end(self) -> None:
        """Compute training metrics at epoch end for performance optimization."""
        if not self.train_step_outputs:
            return

        # Concatenate all outputs
        all_probs = torch.cat(
            [out["probs"].view(-1) for out in self.train_step_outputs]
        )
        all_labels = torch.cat([out["labels"] for out in self.train_step_outputs])
        all_preds = torch.cat([out["preds"] for out in self.train_step_outputs])

        # Compute metrics
        self.train_acc(all_preds, all_labels)
        self.train_auroc(all_probs, all_labels)
        self.train_f1(all_preds, all_labels)

        # Log metrics
        self.log_dict(
            {
                "train/acc": self.train_acc,
                "train/auroc": self.train_auroc,
                "train/f1": self.train_f1,
            },
            on_step=False,
            on_epoch=True,
            sync_dist=False,
        )

        # Clear outputs
        self.train_step_outputs.clear()

    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        output = self._shared_step(batch, "val")
        # Collect outputs for epoch-end metric computation
        self.val_step_outputs.append(output)
        return output

    def test_step(self, batch: Dict[str, Any], batch_idx: int):
        output = self._shared_step(batch, "test")
        return {
            "loss": output["loss"],
            "y_true": output["labels"].detach().cpu(),
            "y_prob": output["probs"].detach().cpu(),
        }

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
        """Compute validation metrics at epoch end and save artifacts."""
        if self.val_step_outputs:
            # Concatenate all outputs
            all_probs = torch.cat(
                [out["probs"].view(-1) for out in self.val_step_outputs]
            )
            all_labels = torch.cat([out["labels"] for out in self.val_step_outputs])
            all_preds = torch.cat([out["preds"] for out in self.val_step_outputs])

            # Compute metrics
            self.val_acc(all_preds, all_labels)
            self.val_auroc(all_probs, all_labels)
            self.val_f1(all_preds, all_labels)

            # Log metrics
            self.log_dict(
                {
                    "val/acc": self.val_acc,
                    "val/auroc": self.val_auroc,
                    "val/f1": self.val_f1,
                },
                prog_bar=True,
                on_step=False,
                on_epoch=True,
                sync_dist=False,
            )

            # Clear outputs
            self.val_step_outputs.clear()

        # Save artifacts
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
        non_bert_params += [
            p
            for p in list(self.url_head.parameters())
            + list(self.html_head.parameters())
            + list(self.visual_head.parameters())
            if p.requires_grad
        ]

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

        # Use ReduceLROnPlateau scheduler if configured, otherwise use CosineAnnealingLR
        if (
            self.cfg
            and hasattr(self.cfg, "eval")
            and hasattr(self.cfg.eval, "reduce_lr_on_plateau")
        ):
            rlr_config = self.cfg.eval.reduce_lr_on_plateau
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=rlr_config.get("mode", "min"),
                factor=rlr_config.get("factor", 0.5),
                patience=rlr_config.get("patience", 2),
                min_lr=rlr_config.get("min_lr", 1e-6),
                verbose=True,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": rlr_config.get("monitor", "val/loss"),
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        else:
            # Fallback to CosineAnnealingLR if not configured
            t_max = 20
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

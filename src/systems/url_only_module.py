from __future__ import annotations

from typing import Any, Dict, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from src.models.url_encoder import URLEncoder


class UrlOnlyModule(pl.LightningModule):
    """LightningModule wrapping a URLEncoder and a linear classifier."""

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
        self.classifier = nn.Linear(model_cfg.proj_dim, model_cfg.num_classes)
        self.criterion = nn.CrossEntropyLoss()

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
        return {"val_loss": result["loss"], "val_acc": result["acc"]}

    def test_step(self, batch, batch_idx):
        result = self._shared_step(batch, "test")
        return {"test_loss": result["loss"], "test_acc": result["acc"]}

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.cfg.train.lr)


# Backwards compatibility alias
UrlOnlySystem = UrlOnlyModule

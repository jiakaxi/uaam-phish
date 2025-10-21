from typing import Dict, Optional
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.classification import BinaryF1Score, BinaryAUROC
from src.models.url_encoder import UrlBertEncoder


class UrlOnlySystem(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(ignore=["cfg"])  # logs hparams
        self.cfg = cfg
        self.encoder = UrlBertEncoder(cfg.model.pretrained_name, cfg.model.dropout)
        hidden = self.encoder.config.hidden_size
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(cfg.model.dropout),
            nn.Linear(hidden, 1),
        )
        # register pos_weight as a buffer for device-safe loss creation
        self.register_buffer(
            "pos_weight", torch.tensor([cfg.train.pos_weight], dtype=torch.float32)
        )
        self.f1 = BinaryF1Score()
        self.auroc = BinaryAUROC()
        self._val_logits = []
        self._val_y = []

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        z = self.encoder(batch)  # [B,H]
        logit = self.head(z).squeeze(1)  # [B]
        return logit

    def step(
        self,
        batch: Dict[str, torch.Tensor],
        stage: str,
        logit: Optional[torch.Tensor] = None,
    ):
        y = batch["label"].float()
        if logit is None:
            logit = self.forward(batch)
        loss = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)(logit, y)
        prob = torch.sigmoid(logit)
        pred = (prob > 0.5).int()
        # FPR = FP / (FP + TN)
        tn = ((pred == 0) & (y == 0)).sum().float()
        fp = ((pred == 1) & (y == 0)).sum().float()
        fpr = fp / (fp + tn + 1e-9)
        self.log_dict(
            {
                f"{stage}/loss": loss,
                f"{stage}/f1": self.f1(pred, y.int()),
                f"{stage}/auroc": self.auroc(prob, y.int()),
                f"{stage}/fpr": fpr,
            },
            prog_bar=True,
            on_step=(stage == "train"),
            on_epoch=True,
            sync_dist=False,
        )
        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, "train")

    def validation_step(self, batch, batch_idx):
        logit = self.forward(batch)
        self._val_logits.append(logit.detach().cpu())
        self._val_y.append(batch["label"].float().detach().cpu())
        return self.step(batch, "val", logit=logit)

    def test_step(self, batch, batch_idx):
        logit = self.forward(batch)
        loss = self.step(batch, "test", logit=logit)

        # 返回预测结果用于可视化
        y = batch["label"].float()
        prob = torch.sigmoid(logit)

        return {"y_true": y, "y_prob": prob, "loss": loss}

    def on_validation_epoch_start(self):
        self._val_logits = []
        self._val_y = []

    def on_validation_epoch_end(self):
        import torch

        logits = torch.cat(self._val_logits)
        y = torch.cat(self._val_y).int()
        prob = logits.sigmoid()
        # epoch 级 FPR
        pred05 = (prob > 0.5).int()
        tn = ((pred05 == 0) & (y == 0)).sum().float()
        fp = ((pred05 == 1) & (y == 0)).sum().float()
        fpr = fp / (fp + tn + 1e-9)
        self.log("val/fpr_epoch", fpr, prog_bar=True)

        # 粗略阈值扫描(优化 F1)
        ths = torch.linspace(0.2, 0.8, 25)
        best_f1, best_th = -1.0, 0.5
        for t in ths:
            pred = (prob > t).int()
            tp = ((pred == 1) & (y == 1)).sum().float()
            fp = ((pred == 1) & (y == 0)).sum().float()
            fn = ((pred == 0) & (y == 1)).sum().float()
            f1 = (2 * tp) / (2 * tp + fp + fn + 1e-9)
            if f1 > best_f1:
                best_f1, best_th = f1.item(), t.item()
        self.log("val/best_f1", best_f1, prog_bar=True)
        self.log("val/best_threshold", best_th, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.train.lr,
            weight_decay=getattr(self.cfg.train, "weight_decay", 0.0),
        )

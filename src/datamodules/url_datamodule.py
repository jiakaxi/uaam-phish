from __future__ import annotations

from typing import Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from src.data.url_dataset import UrlDataset


class UrlDataModule(pl.LightningDataModule):
    """Lightning DataModule wrapping the character-level UrlDataset."""

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.train_dataset: Optional[UrlDataset] = None
        self.val_dataset: Optional[UrlDataset] = None
        self.test_dataset: Optional[UrlDataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        model_cfg = self.cfg.model
        data_cfg = self.cfg.data
        max_len = model_cfg.max_len
        vocab_size = model_cfg.vocab_size
        pad_id = model_cfg.pad_id

        if stage in (None, "fit", "train"):
            self.train_dataset = UrlDataset(
                data_cfg.train_csv,
                max_len=max_len,
                vocab_size=vocab_size,
                pad_id=pad_id,
            )
        if stage in (None, "fit", "validate", "test"):
            self.val_dataset = UrlDataset(
                data_cfg.val_csv,
                max_len=max_len,
                vocab_size=vocab_size,
                pad_id=pad_id,
            )
        if stage in (None, "test"):
            self.test_dataset = UrlDataset(
                data_cfg.test_csv,
                max_len=max_len,
                vocab_size=vocab_size,
                pad_id=pad_id,
            )

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError(
                "Call setup('fit') before requesting train_dataloader()."
            )
        batch_size = self.cfg.train.get("bs", self.cfg.train.get("batch_size", 32))
        return DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.cfg.data.num_workers,
            pin_memory=False,
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise RuntimeError("Call setup('fit') before requesting val_dataloader().")
        batch_size = self.cfg.train.get("bs", self.cfg.train.get("batch_size", 32))
        return DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.cfg.data.num_workers,
            pin_memory=False,
        )

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            raise RuntimeError(
                "Call setup('test') before requesting test_dataloader()."
            )
        batch_size = self.cfg.train.get("bs", self.cfg.train.get("batch_size", 32))
        return DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.cfg.data.num_workers,
            pin_memory=False,
        )

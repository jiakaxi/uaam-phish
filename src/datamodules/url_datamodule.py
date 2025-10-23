from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from src.data.url_dataset import UrlDataset
from src.utils.logging import get_logger

log = get_logger(__name__)


class UrlDataModule(pl.LightningDataModule):
    """Lightning DataModule wrapping the character-level UrlDataset."""

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.train_dataset: Optional[UrlDataset] = None
        self.val_dataset: Optional[UrlDataset] = None
        self.test_dataset: Optional[UrlDataset] = None
        self.split_metadata: dict = {}  # Metadata from build_splits

    def setup(self, stage: Optional[str] = None) -> None:
        model_cfg = self.cfg.model
        data_cfg = self.cfg.data
        max_len = model_cfg.max_len
        vocab_size = model_cfg.vocab_size
        pad_id = model_cfg.pad_id

        # 如果启用了 use_build_splits，则调用 build_splits 生成 splits 和 metadata
        if stage in (None, "fit") and self.cfg.get("use_build_splits", False):
            try:
                from src.utils.splits import build_splits

                csv_path = Path(data_cfg.csv_path)
                if csv_path.exists():
                    log.info(
                        f">> Building splits from {csv_path} using protocol '{self.cfg.get('protocol', 'random')}'"
                    )
                    df = pd.read_csv(csv_path)
                    protocol = self.cfg.get("protocol", "random")
                    train_df, val_df, test_df, metadata = build_splits(
                        df, self.cfg, protocol=protocol
                    )

                    # 保存 splits 到文件
                    train_csv = Path(data_cfg.train_csv)
                    val_csv = Path(data_cfg.val_csv)
                    test_csv = Path(data_cfg.test_csv)

                    train_csv.parent.mkdir(parents=True, exist_ok=True)
                    train_df.to_csv(train_csv, index=False)
                    val_df.to_csv(val_csv, index=False)
                    test_df.to_csv(test_csv, index=False)

                    log.info(
                        f">> Splits saved: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
                    )

                    # 保存 metadata 供 callbacks 使用
                    self.split_metadata = metadata

                    if metadata.get("downgraded_to"):
                        log.warning(
                            f">> Protocol downgraded to '{metadata['downgraded_to']}': {metadata.get('downgrade_reason')}"
                        )
                else:
                    log.warning(
                        f">> Master CSV not found: {csv_path}, skipping build_splits"
                    )
                    self.split_metadata = {}
            except Exception as e:
                log.error(f">> Failed to build splits: {e}")
                self.split_metadata = {}

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

"""
HTML DataModule
Lightning DataModule for HTML modality, follows url_datamodule.py structure
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from src.data.html_dataset import HtmlDataset
from src.utils.logging import get_logger

log = get_logger(__name__)


class HtmlDataModule(pl.LightningDataModule):
    """
    Lightning DataModule for HTML modality.
    Follows url_datamodule.py structure for consistency.

    Supports:
    - Three split protocols (random/temporal/brand_ood) via build_splits()
    - Metadata tracking for ProtocolArtifactsCallback
    - Configurable BERT model and tokenization parameters
    """

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.train_dataset: Optional[HtmlDataset] = None
        self.val_dataset: Optional[HtmlDataset] = None
        self.test_dataset: Optional[HtmlDataset] = None
        self.split_metadata: dict = {}  # Metadata from build_splits

    def setup(self, stage: Optional[str] = None) -> None:
        model_cfg = self.cfg.model
        data_cfg = self.cfg.data

        bert_model = model_cfg.bert_model
        max_len = data_cfg.get("html_max_len", 512)  # 从 data 配置读取

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
            self.train_dataset = HtmlDataset(
                data_cfg.train_csv,
                bert_model=bert_model,
                max_len=max_len,
            )
        if stage in (None, "fit", "validate", "test"):
            self.val_dataset = HtmlDataset(
                data_cfg.val_csv,
                bert_model=bert_model,
                max_len=max_len,
            )
        if stage in (None, "test"):
            self.test_dataset = HtmlDataset(
                data_cfg.test_csv,
                bert_model=bert_model,
                max_len=max_len,
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

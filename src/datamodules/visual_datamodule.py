"""
Visual DataModule
Lightning DataModule for Visual modality, follows html_datamodule.py structure
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms

from src.data.visual_dataset import VisualDataset
from src.utils.logging import get_logger

log = get_logger(__name__)

# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class VisualDataModule(pl.LightningDataModule):
    """
    Lightning DataModule for Visual modality (screenshot images).
    Follows html_datamodule.py structure for consistency.

    Supports:
    - Three split protocols (random/temporal/brand_ood) via build_splits()
    - Metadata tracking for ProtocolArtifactsCallback
    - ImageNet-standard transforms with optional augmentation
    """

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.train_dataset: Optional[VisualDataset] = None
        self.val_dataset: Optional[VisualDataset] = None
        self.test_dataset: Optional[VisualDataset] = None
        self.split_metadata: dict = {}  # Metadata from build_splits

        # Define transforms
        self.train_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.RandomHorizontalFlip(p=0.5),  # Light augmentation
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

        self.val_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    def setup(self, stage: Optional[str] = None) -> None:
        data_cfg = self.cfg.data

        image_col = data_cfg.get("image_col", "img_path")
        label_col = data_cfg.get("label_col", "label")
        id_col = data_cfg.get("id_col", "id")

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

                    # 保存 splits 到 CSV（可选：如果需要持久化）
                    # 但这里我们直接使用 split 列，不额外生成文件

                    # 保存 metadata 供 callbacks 使用
                    self.split_metadata = metadata

                    log.info(
                        f">> Splits created: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
                    )

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

        # Create datasets using split column filtering
        csv_path = data_cfg.csv_path

        if stage in (None, "fit", "train"):
            self.train_dataset = VisualDataset(
                csv_path,
                transform=self.train_transform,
                image_col=image_col,
                label_col=label_col,
                id_col=id_col,
                split="train" if self.cfg.get("use_build_splits", False) else None,
            )
            log.info(f">> Train dataset: {len(self.train_dataset)} samples")

        if stage in (None, "fit", "validate"):
            self.val_dataset = VisualDataset(
                csv_path,
                transform=self.val_transform,
                image_col=image_col,
                label_col=label_col,
                id_col=id_col,
                split="val" if self.cfg.get("use_build_splits", False) else None,
            )
            log.info(f">> Val dataset: {len(self.val_dataset)} samples")

        if stage in (None, "test"):
            self.test_dataset = VisualDataset(
                csv_path,
                transform=self.val_transform,
                image_col=image_col,
                label_col=label_col,
                id_col=id_col,
                split="test" if self.cfg.get("use_build_splits", False) else None,
            )
            log.info(f">> Test dataset: {len(self.test_dataset)} samples")

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError(
                "Call setup('fit') before requesting train_dataloader()."
            )
        batch_size = self.cfg.train.get("bs", self.cfg.train.get("batch_size", 32))
        num_workers = self.cfg.data.get("num_workers", 4)
        return DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise RuntimeError("Call setup('fit') before requesting val_dataloader().")
        batch_size = self.cfg.train.get("bs", self.cfg.train.get("batch_size", 32))
        num_workers = self.cfg.data.get("num_workers", 4)
        return DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            raise RuntimeError(
                "Call setup('test') before requesting test_dataloader()."
            )
        batch_size = self.cfg.train.get("bs", self.cfg.train.get("batch_size", 32))
        num_workers = self.cfg.data.get("num_workers", 4)
        return DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

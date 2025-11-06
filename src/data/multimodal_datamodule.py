"""
多模态数据模块 - 整合 URL、HTML、Visual 三模态数据
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from PIL import Image
from transformers import BertTokenizer
from torchvision import transforms

from src.utils.logging import get_logger
from src.utils.splits import build_splits

log = get_logger(__name__)


class MultimodalDataset(Dataset):
    """
    多模态数据集：同时加载 URL、HTML、Visual 三模态数据。

    返回格式：
    {
        "id": sample_id,
        "url": torch.LongTensor [seq_len],
        "html": {"input_ids": ..., "attention_mask": ...},
        "visual": torch.FloatTensor [3, 224, 224],
        "label": torch.LongTensor (scalar)
    }
    """

    def __init__(
        self,
        df: pd.DataFrame,
        url_max_len: int = 200,
        url_vocab_size: int = 128,
        html_tokenizer: BertTokenizer = None,
        html_max_len: int = 512,
        visual_transform: transforms.Compose = None,
        image_dir: Path = None,
    ):
        """
        Args:
            df: DataFrame with columns [sample_id, url_text, html_text, image_path, label]
            url_max_len: Max URL character sequence length
            url_vocab_size: URL character vocabulary size (ASCII)
            html_tokenizer: BERT tokenizer
            html_max_len: Max HTML token sequence length
            visual_transform: Image transforms (ResNet-50 standard)
            image_dir: Base directory for image paths (if relative)
        """
        self.df = df.reset_index(drop=True)
        self.url_max_len = url_max_len
        self.url_vocab_size = url_vocab_size
        self.html_tokenizer = html_tokenizer
        self.html_max_len = html_max_len
        self.visual_transform = visual_transform
        self.image_dir = Path(image_dir) if image_dir else Path(".")

        # Validate required columns
        required_cols = {"sample_id", "url_text", "html_text", "image_path", "label"}
        missing = required_cols - set(df.columns)
        if missing:
            log.warning(f"Missing columns: {missing}, will use placeholders")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]

        sample_id = row.get("id", row.get("sample_id", idx))
        url_text = row.get("url_text", "")
        # 处理 NaN 情况
        if pd.isna(url_text) or not isinstance(url_text, str):
            url_text = ""

        # HTML 可能在 html_text 列，或者在 html_path 指向的文件中
        html_text = row.get("html_text", "")
        if not html_text and "html_path" in row and pd.notna(row["html_path"]):
            try:
                html_path = Path(str(row["html_path"]))
                if html_path.exists():
                    with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
                        html_text = f.read()
            except Exception as e:
                log.warning(f"Failed to read HTML from {row.get('html_path')}: {e}")
                html_text = ""

        # 图像路径可能在 image_path 或 img_path 列
        image_path = row.get("img_path", row.get("image_path", ""))
        label = int(row.get("label", 0))

        # 1. URL: Character-level tokenization
        url_ids = self._tokenize_url(url_text)

        # 2. HTML: BERT tokenization
        html_encoded = self.html_tokenizer(
            html_text,
            max_length=self.html_max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        html_input_ids = html_encoded["input_ids"].squeeze(0)  # [seq_len]
        html_attention_mask = html_encoded["attention_mask"].squeeze(0)  # [seq_len]

        # 3. Visual: Load image and apply transforms
        visual_tensor = self._load_image(image_path)

        return {
            "id": sample_id,
            "url": url_ids,
            "html": {
                "input_ids": html_input_ids,
                "attention_mask": html_attention_mask,
            },
            "visual": visual_tensor,
            "label": torch.tensor(label, dtype=torch.long),
        }

    def _tokenize_url(self, url_text: str) -> torch.LongTensor:
        """Character-level URL tokenization (ASCII-based)."""
        # Convert to ASCII codes, clamp to vocab_size
        ids = [min(ord(c), self.url_vocab_size - 1) for c in url_text]

        # Truncate or pad to max_len
        if len(ids) > self.url_max_len:
            ids = ids[: self.url_max_len]
        else:
            ids = ids + [0] * (self.url_max_len - len(ids))  # pad with 0

        return torch.tensor(ids, dtype=torch.long)

    def _load_image(self, image_path: str) -> torch.Tensor:
        """Load image from path and apply transforms."""
        try:
            # Try to load image
            full_path = self.image_dir / image_path
            if not full_path.exists():
                # Try as absolute path
                full_path = Path(image_path)

            img = Image.open(full_path).convert("RGB")

            if self.visual_transform:
                img_tensor = self.visual_transform(img)
            else:
                # Fallback: resize to 224x224 and convert to tensor
                img = img.resize((224, 224))
                img_tensor = transforms.ToTensor()(img)

            return img_tensor

        except Exception as e:
            log.warning(
                f"Failed to load image {image_path}: {e}, using black placeholder"
            )
            # Return black placeholder image
            return torch.zeros(3, 224, 224)


class MultimodalDataModule(pl.LightningDataModule):
    """
    多模态数据模块：整合 URL、HTML、Visual 三个数据源。

    数据加载策略：
    1. 加载三个 CSV 文件
    2. 以 sample_id（或 url_text）为键内连接合并
    3. 根据 split_protocol 分割数据（random / temporal / brand_ood）
    4. 返回三个 DataLoader（train/val/test）
    """

    def __init__(
        self,
        url_data_path: str = None,
        html_data_path: str = None,
        visual_data_path: str = None,
        master_csv: str = None,  # 新增：使用单个 master CSV
        split_protocol: str = "random",
        batch_size: int = 64,
        num_workers: int = 4,
        pin_memory: bool = True,
        use_augmentation: bool = False,
        url_max_len: int = 200,
        url_vocab_size: int = 128,
        html_max_len: int = 512,
        image_dir: Optional[str] = None,
        use_presplit: bool = True,  # 新增：是否使用预分割数据
        cfg=None,  # 兼容 Hydra 实例化时传入的 cfg 参数（忽略）
        **kwargs,  # 捕获其他未知参数
    ):
        super().__init__()
        self.master_csv = Path(master_csv) if master_csv else None
        self.url_data_path = Path(url_data_path) if url_data_path else None
        self.html_data_path = Path(html_data_path) if html_data_path else None
        self.visual_data_path = Path(visual_data_path) if visual_data_path else None
        self.split_protocol = split_protocol
        self.use_presplit = use_presplit
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.use_augmentation = use_augmentation
        self.url_max_len = url_max_len
        self.url_vocab_size = url_vocab_size
        self.html_max_len = html_max_len
        self.image_dir = (
            Path(image_dir) if image_dir else Path("data/processed/screenshots")
        )

        # Datasets (initialized in setup)
        self.train_dataset: Optional[MultimodalDataset] = None
        self.val_dataset: Optional[MultimodalDataset] = None
        self.test_dataset: Optional[MultimodalDataset] = None

        # Split metadata (for callbacks)
        self.split_metadata: Dict = {}

    def setup(self, stage: Optional[str] = None):
        """Load and merge three CSVs, then split into train/val/test."""

        # 使用 master CSV（所有数据已在一个文件中）
        if self.master_csv and self.master_csv.exists():
            log.info(f">> Loading multimodal data from master CSV: {self.master_csv}")
            df_merged = pd.read_csv(self.master_csv)
            log.info(f"   Total samples: {len(df_merged)}")

            # 使用预分割的数据（基于 'split' 列）
            if self.use_presplit and "split" in df_merged.columns:
                log.info(">> Using pre-split data from 'split' column")
                train_df = df_merged[df_merged["split"] == "train"].copy()
                val_df = df_merged[df_merged["split"] == "val"].copy()
                test_df = df_merged[df_merged["split"] == "test"].copy()

                self.split_metadata = {
                    "protocol": "presplit",
                    "split_stats": {
                        "train": {"count": len(train_df)},
                        "val": {"count": len(val_df)},
                        "test": {"count": len(test_df)},
                    },
                }
            else:
                # 调用 build_splits 进行分割
                log.info(f">> Building splits using protocol: {self.split_protocol}")

                class SplitConfig:
                    def __init__(self, protocol):
                        self.protocol = protocol
                        self.data = {
                            "split_ratios": {"train": 0.7, "val": 0.15, "test": 0.15}
                        }

                    def get(self, key, default=None):
                        return getattr(self, key, default)

                cfg = SplitConfig(self.split_protocol)
                train_df, val_df, test_df, metadata = build_splits(
                    df_merged, cfg, protocol=self.split_protocol
                )
                self.split_metadata = metadata

        else:
            # 兼容旧方式：从三个独立 CSV 加载
            log.info(">> Loading multimodal data...")
            df_url = pd.read_csv(self.url_data_path)
            df_html = pd.read_csv(self.html_data_path)
            df_visual = pd.read_csv(self.visual_data_path)

            log.info(f"   URL data: {len(df_url)} samples")
            log.info(f"   HTML data: {len(df_html)} samples")
            log.info(f"   Visual data: {len(df_visual)} samples")

            # 2. Inner join on sample_id (or url as fallback)
            merge_key = "sample_id" if "sample_id" in df_url.columns else "url"

            df_merged = df_url.merge(
                df_html, on=merge_key, how="inner", suffixes=("", "_html")
            )
            df_merged = df_merged.merge(
                df_visual, on=merge_key, how="inner", suffixes=("", "_vis")
            )

            log.info(f">> After inner join: {len(df_merged)} samples")

            # 使用 build_splits
            log.info(f">> Building splits using protocol: {self.split_protocol}")

            class SplitConfig:
                def __init__(self, protocol):
                    self.protocol = protocol
                    self.data = {
                        "split_ratios": {"train": 0.7, "val": 0.15, "test": 0.15}
                    }

                def get(self, key, default=None):
                    return getattr(self, key, default)

            cfg = SplitConfig(self.split_protocol)
            train_df, val_df, test_df, metadata = build_splits(
                df_merged, cfg, protocol=self.split_protocol
            )
            self.split_metadata = metadata

        log.info(
            f">> Splits: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
        )

        if self.split_metadata.get("downgraded_to"):
            log.warning(
                f">> Protocol downgraded to '{self.split_metadata['downgraded_to']}': {self.split_metadata.get('downgrade_reason')}"
            )

        # 4. Initialize tokenizers and transforms
        self.html_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        # ResNet-50 standard transforms
        if self.use_augmentation:
            # Training augmentation
            train_transform = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.RandomCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        else:
            train_transform = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

        # Val/test transform (no augmentation)
        eval_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        # 5. Create datasets
        if stage in (None, "fit", "train"):
            self.train_dataset = MultimodalDataset(
                train_df,
                url_max_len=self.url_max_len,
                url_vocab_size=self.url_vocab_size,
                html_tokenizer=self.html_tokenizer,
                html_max_len=self.html_max_len,
                visual_transform=train_transform,
                image_dir=self.image_dir,
            )
            self.val_dataset = MultimodalDataset(
                val_df,
                url_max_len=self.url_max_len,
                url_vocab_size=self.url_vocab_size,
                html_tokenizer=self.html_tokenizer,
                html_max_len=self.html_max_len,
                visual_transform=eval_transform,
                image_dir=self.image_dir,
            )

        if stage in (None, "test"):
            self.test_dataset = MultimodalDataset(
                test_df,
                url_max_len=self.url_max_len,
                url_vocab_size=self.url_vocab_size,
                html_tokenizer=self.html_tokenizer,
                html_max_len=self.html_max_len,
                visual_transform=eval_transform,
                image_dir=self.image_dir,
            )

        log.info(">> Multimodal DataModule setup complete!")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )

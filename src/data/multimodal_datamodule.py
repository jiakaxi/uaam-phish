"""
Multimodal DataModule (S0 baseline, Sec. 4.3.4 & 4.6.1).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from PIL import Image
from transformers import BertTokenizer
from torchvision import transforms

from src.utils.logging import get_logger
from src.utils.splits import build_splits


log = get_logger(__name__)

# 增加PIL图像大小限制，防止DecompressionBombWarning
Image.MAX_IMAGE_PIXELS = 500_000_000  # 500MP


class MultimodalDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        url_max_len: int,
        url_vocab_size: int,
        html_tokenizer: BertTokenizer,
        html_max_len: int,
        visual_transform: transforms.Compose,
        image_dir: Path,
        corrupt_root: Optional[Path] = None,
        preload_html: bool = True,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.url_max_len = url_max_len
        self.url_vocab_size = url_vocab_size
        self.html_tokenizer = html_tokenizer
        self.html_max_len = html_max_len
        self.visual_transform = visual_transform
        self.image_dir = image_dir
        self.corrupt_root = Path(corrupt_root) if corrupt_root else None

        # 预加载HTML文件到内存（性能优化）
        self.html_cache: Dict[int, str] = {}
        if preload_html:
            log.info(">> Preloading HTML files into memory...")
            for idx in range(len(self.df)):
                row = self.df.iloc[idx]
                html_text = self._safe_string(row.get("html_text", ""))
                if html_text:
                    self.html_cache[idx] = html_text
                else:
                    html_path = row.get("html_path")
                    if pd.notna(html_path):
                        try:
                            self.html_cache[idx] = Path(html_path).read_text(
                                encoding="utf-8", errors="ignore"
                            )
                        except Exception as exc:
                            log.debug(
                                "Failed to preload HTML from %s: %s", html_path, exc
                            )
                            self.html_cache[idx] = ""
                    else:
                        self.html_cache[idx] = ""
            log.info(">> Preloaded %d HTML files", len(self.html_cache))

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        sample_id = row.get("sample_id", row.get("id", idx))
        url_text = self._safe_string(row.get("url_text", row.get("url", "")))
        html_text = self._load_html(row, idx)
        image_tensor = self._load_image(row)
        label = int(row.get("label", 0))

        url_ids = self._tokenize_url(url_text)
        html_encoded = self.html_tokenizer(
            html_text,
            max_length=self.html_max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "id": sample_id,
            "url": url_ids,
            "html": {
                "input_ids": html_encoded["input_ids"].squeeze(0),
                "attention_mask": html_encoded["attention_mask"].squeeze(0),
            },
            "visual": image_tensor,
            "label": torch.tensor(label, dtype=torch.long),
        }

    @staticmethod
    def _safe_string(value: Any) -> str:
        if isinstance(value, str):
            return value
        return "" if pd.isna(value) else str(value)

    def _tokenize_url(self, url_text: str) -> torch.LongTensor:
        char_ids = [min(ord(c), self.url_vocab_size - 1) for c in url_text]
        if len(char_ids) > self.url_max_len:
            char_ids = char_ids[: self.url_max_len]
        else:
            char_ids += [0] * (self.url_max_len - len(char_ids))
        return torch.tensor(char_ids, dtype=torch.long)

    def _load_html(self, row: pd.Series, idx: int) -> str:
        # 使用缓存（如果可用）
        if idx in self.html_cache:
            return self.html_cache[idx]

        # 回退到原始逻辑（如果缓存未启用）
        html_text = self._safe_string(row.get("html_text", ""))
        if html_text:
            return html_text
        html_path = row.get("html_path")
        if pd.notna(html_path):
            try:
                return Path(html_path).read_text(encoding="utf-8", errors="ignore")
            except Exception as exc:
                log.warning("Failed to read HTML from %s: %s", html_path, exc)
        return ""

    def _load_image(self, row: pd.Series) -> torch.Tensor:
        img_path = row.get("img_path_corrupt")
        prefer_corrupt = True
        if pd.isna(img_path) or not str(img_path).strip():
            img_path = row.get("img_path")
            prefer_corrupt = False
        if pd.isna(img_path) or not str(img_path).strip():
            img_path = row.get("image_path")
            prefer_corrupt = False

        if pd.isna(img_path) or not img_path:
            img = Image.new("RGB", (224, 224))
            return self.visual_transform(img)

        path = self._resolve_image_path(self._safe_string(img_path), prefer_corrupt)

        try:
            if path.exists():
                img = Image.open(path)
                # 性能优化：如果图像太大，立即resize（避免内存和速度问题）
                width, height = img.size
                total_pixels = width * height
                max_pixels = 10_000_000  # 10MP阈值
                if total_pixels > max_pixels:
                    # 计算缩放比例，保持宽高比
                    scale = (max_pixels / total_pixels) ** 0.5
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    # 使用thumbnail快速resize（原地操作，更高效）
                    img.thumbnail((new_width, new_height), Image.Resampling.LANCZOS)
                img = img.convert("RGB")
            else:
                log.warning("Image path %s missing; using blank placeholder.", path)
                img = Image.new("RGB", (224, 224))
        except Exception as exc:
            log.warning(
                "Failed to load image %s (%s); using blank placeholder.", path, exc
            )
            img = Image.new("RGB", (224, 224))
        return self.visual_transform(img)

    def _resolve_image_path(self, path_str: str, prefer_corrupt: bool) -> Path:
        candidate = Path(path_str)
        if candidate.is_absolute():
            return candidate

        if candidate.parts and candidate.parts[0] in {"workspace", "data"}:
            return candidate

        if prefer_corrupt and self.corrupt_root is not None:
            return self.corrupt_root / candidate

        return self.image_dir / candidate


class MultimodalDataModule(pl.LightningDataModule):
    def __init__(
        self,
        url_data_path: Optional[str] = None,
        html_data_path: Optional[str] = None,
        visual_data_path: Optional[str] = None,
        master_csv: Optional[str] = None,
        train_csv: Optional[str] = None,
        val_csv: Optional[str] = None,
        test_csv: Optional[str] = None,
        split_protocol: str = "presplit",
        batch_size: int = 128,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: Optional[bool] = None,
        use_augmentation: bool = False,
        url_max_len: int = 200,
        url_vocab_size: int = 128,
        html_max_len: int = 256,
        image_dir: Optional[str] = None,
        corrupt_root: Optional[str] = "workspace/data/corrupt",
        use_presplit: bool = True,
        cfg: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.master_csv = Path(master_csv) if master_csv else None
        self.train_csv = Path(train_csv) if train_csv else None
        self.val_csv = Path(val_csv) if val_csv else None
        self.test_csv = Path(test_csv) if test_csv else None
        self.url_data_path = Path(url_data_path) if url_data_path else None
        self.html_data_path = Path(html_data_path) if html_data_path else None
        self.visual_data_path = Path(visual_data_path) if visual_data_path else None
        self.split_protocol = split_protocol
        self.use_presplit = use_presplit
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = (
            bool(persistent_workers)
            if persistent_workers is not None
            else self.num_workers > 0
        )
        self.use_augmentation = use_augmentation
        self.url_max_len = url_max_len
        self.url_vocab_size = url_vocab_size
        self.html_max_len = html_max_len
        self.image_dir = (
            Path(image_dir) if image_dir else Path("data/processed/screenshots")
        )
        self.corrupt_root = Path(corrupt_root) if corrupt_root else None

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None
        self.split_metadata: Dict[str, Any] = {}

    # ------------------------------------------------------------------ #
    def setup(self, stage: Optional[str] = None) -> None:
        full_df = None
        if not (self.train_csv and self.val_csv and self.test_csv):
            full_df = self._load_dataframe()
        train_df, val_df, test_df = self._determine_splits(full_df)

        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        train_transform, eval_transform = self._build_transforms()

        if stage in (None, "fit", "train"):
            self.train_dataset = MultimodalDataset(
                train_df,
                url_max_len=self.url_max_len,
                url_vocab_size=self.url_vocab_size,
                html_tokenizer=tokenizer,
                html_max_len=self.html_max_len,
                visual_transform=train_transform,
                image_dir=self.image_dir,
                corrupt_root=self.corrupt_root,
            )
            self.val_dataset = MultimodalDataset(
                val_df,
                url_max_len=self.url_max_len,
                url_vocab_size=self.url_vocab_size,
                html_tokenizer=tokenizer,
                html_max_len=self.html_max_len,
                visual_transform=eval_transform,
                image_dir=self.image_dir,
                corrupt_root=self.corrupt_root,
            )

        if stage in (None, "test"):
            self.test_dataset = MultimodalDataset(
                test_df,
                url_max_len=self.url_max_len,
                url_vocab_size=self.url_vocab_size,
                html_tokenizer=tokenizer,
                html_max_len=self.html_max_len,
                visual_transform=eval_transform,
                image_dir=self.image_dir,
                corrupt_root=self.corrupt_root,
            )

    # ------------------------------------------------------------------ #
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    # ------------------------------------------------------------------ #
    def _load_dataframe(self) -> pd.DataFrame:
        if self.master_csv and self.master_csv.exists():
            log.info(">> Loading master CSV: %s", self.master_csv)
            return pd.read_csv(self.master_csv)

        if not all(
            path and path.exists()
            for path in (self.url_data_path, self.html_data_path, self.visual_data_path)
        ):
            raise FileNotFoundError(
                "Master CSV not provided and modality CSVs missing."
            )

        log.info(">> Loading modality CSVs individually.")
        df_url = pd.read_csv(self.url_data_path)
        df_html = pd.read_csv(self.html_data_path)
        df_visual = pd.read_csv(self.visual_data_path)

        merge_key = "sample_id" if "sample_id" in df_url.columns else "url"
        df_merged = df_url.merge(
            df_html, on=merge_key, how="inner", suffixes=("", "_html")
        )
        df_merged = df_merged.merge(
            df_visual, on=merge_key, how="inner", suffixes=("", "_vis")
        )
        log.info(">> After inner join: %d samples", len(df_merged))
        return df_merged

    def _determine_splits(
        self, df: Optional[pd.DataFrame]
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if self.train_csv and self.val_csv and self.test_csv:
            log.info(
                ">> Loading explicit split CSVs: %s / %s / %s",
                self.train_csv,
                self.val_csv,
                self.test_csv,
            )
            train_df = pd.read_csv(self.train_csv)
            val_df = pd.read_csv(self.val_csv)
            test_df = pd.read_csv(self.test_csv)
            self.split_metadata = self._summarize_splits(
                train_df,
                val_df,
                test_df,
                protocol="presplit",
                details={"source": "explicit_csv"},
            )
            return train_df, val_df, test_df

        if df is None:
            raise ValueError(
                "Base dataframe is required when explicit CSVs are not provided."
            )

        protocol = self.split_protocol
        if self.use_presplit and "split" in df.columns:
            train_df = df[df["split"] == "train"].copy()
            val_df = df[df["split"] == "val"].copy()
            test_df = df[df["split"] == "test"].copy()
            self.split_metadata = self._summarize_splits(
                train_df,
                val_df,
                test_df,
                protocol="presplit",
                details={"source": "csv_split_column"},
            )
            return train_df, val_df, test_df

        class SplitCfg:
            def __init__(self, proto: str):
                self.protocol = proto
                self.data = {"split_ratios": {"train": 0.7, "val": 0.15, "test": 0.15}}

            def get(self, key: str, default=None):
                return getattr(self, key, default)

        cfg_stub = SplitCfg(protocol)
        train_df, val_df, test_df, metadata = build_splits(
            df, cfg_stub, protocol=protocol if protocol != "presplit" else "random"
        )

        if metadata.get("downgraded_to"):
            log.warning(
                "Split protocol downgraded from %s to %s (%s).",
                metadata["protocol"],
                metadata["downgraded_to"],
                metadata.get("downgrade_reason"),
            )

        actual_protocol = metadata.get("downgraded_to") or metadata.get(
            "protocol", "random"
        )
        self.split_metadata = self._summarize_splits(
            train_df,
            val_df,
            test_df,
            protocol=actual_protocol,
            details={
                "requested_protocol": protocol,
                "metadata": metadata,
            },
        )
        return train_df, val_df, test_df

    def _build_transforms(self) -> Tuple[transforms.Compose, transforms.Compose]:
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
        if not self.use_augmentation:
            return eval_transform, eval_transform
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
        return train_transform, eval_transform

    def _summarize_splits(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        protocol: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        summary = {
            "protocol": protocol,
            "details": details or {},
            "splits": {
                "train": self._summarize_split(train_df),
                "val": self._summarize_split(val_df),
                "test": self._summarize_split(test_df),
            },
        }
        return summary

    def _summarize_split(self, df: pd.DataFrame) -> Dict[str, Any]:
        ids = self._extract_ids(df)
        summary = {
            "count": int(len(df)),
            "positive": (
                int((df["label"] == 1).sum()) if "label" in df.columns else None
            ),
            "negative": (
                int((df["label"] == 0).sum()) if "label" in df.columns else None
            ),
            "brands": (
                sorted(df["brand"].dropna().astype(str).unique().tolist())
                if "brand" in df.columns
                else []
            ),
            "timestamp_range": self._timestamp_range(df),
            "ids": ids,
        }
        return summary

    @staticmethod
    def _extract_ids(df: pd.DataFrame) -> list:
        for key in ("sample_id", "id", "url"):
            if key in df.columns:
                return df[key].astype(str).tolist()
        return df.index.astype(str).tolist()

    @staticmethod
    def _timestamp_range(df: pd.DataFrame) -> Dict[str, Optional[str]]:
        if "timestamp" not in df.columns:
            return {"min": None, "max": None}
        ts = pd.to_datetime(df["timestamp"], errors="coerce")
        return {
            "min": None if ts.isna().all() else str(ts.min()),
            "max": None if ts.isna().all() else str(ts.max()),
        }

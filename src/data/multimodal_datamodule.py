"""
Multimodal DataModule (S0 baseline, Sec. 4.3.4 & 4.6.1).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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


def multimodal_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function to handle string fields (image_path, id) properly.
    PyTorch's default collate_fn cannot stack strings.
    """
    # Initialize output dict
    collated = {}

    # Handle each field
    for key in batch[0].keys():
        values = [item[key] for item in batch]

        if key in ("id", "image_path", "url_text", "html_path"):
            # Keep strings as list
            collated[key] = values
        elif key == "html":
            # Handle nested dict
            collated[key] = {
                "input_ids": torch.stack([item[key]["input_ids"] for item in batch]),
                "attention_mask": torch.stack(
                    [item[key]["attention_mask"] for item in batch]
                ),
            }
        elif key == "meta":
            # Handle meta dict - keep string fields as lists
            collated[key] = {
                "scenario": [item[key]["scenario"] for item in batch],
                "corruption_level": [item[key]["corruption_level"] for item in batch],
                "protocol": [item[key]["protocol"] for item in batch],
            }
        elif isinstance(values[0], torch.Tensor):
            # Stack tensors
            collated[key] = torch.stack(values)
        else:
            # Keep as list for other types
            collated[key] = values

    return collated


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
        cache_root: Optional[Path] = None,  # 缓存根目录
        protocol: str = "iid",  # Protocol: iid, brandood, temporal, etc.
        scenario: Optional[str] = None,  # Explicit scenario override
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.url_max_len = url_max_len
        self.url_vocab_size = url_vocab_size
        self.html_tokenizer = html_tokenizer
        self.html_max_len = html_max_len
        self.visual_transform = visual_transform
        self.image_dir = image_dir
        self.corrupt_root = Path(corrupt_root) if corrupt_root else None
        self.cache_root = Path(cache_root) if cache_root else None
        self.protocol = protocol
        self.scenario_override = scenario

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

        # 缓存优先加载：先尝试加载缓存，失败则回退到原始逻辑
        url_ids = self._load_cached_url(row)
        if url_ids is not None:
            log.debug(f">> 样本 {sample_id}: URL缓存命中")
        else:
            url_text = self._safe_string(row.get("url_text", row.get("url", "")))
            url_ids = self._tokenize_url(url_text)
            log.debug(f">> 样本 {sample_id}: URL缓存未命中，使用tokenizer")

        html_encoded = self._load_cached_html(row)
        if html_encoded is not None:
            log.debug(f">> 样本 {sample_id}: HTML缓存命中")
        else:
            html_text = self._load_html(row, idx)
            html_encoded = self.html_tokenizer(
                html_text,
                max_length=self.html_max_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            log.debug(f">> 样本 {sample_id}: HTML缓存未命中，使用tokenizer")

        image_tensor = self._load_cached_image(row)
        if image_tensor is not None:
            log.debug(f">> 样本 {sample_id}: 图像缓存命中")
        else:
            image_tensor = self._load_image(row)
            log.debug(f">> 样本 {sample_id}: 图像缓存未命中，使用原始图像")

        label = int(row.get("label", 0))

        image_path_str = self._select_image_path(row)

        # 处理HTML编码，确保返回正确的格式
        if hasattr(html_encoded, "keys") and "input_ids" in html_encoded:
            # 来自tokenizer的BatchEncoding
            html_input_ids = html_encoded["input_ids"].squeeze(0)
            html_attention_mask = html_encoded["attention_mask"].squeeze(0)
        else:
            # 来自缓存的tensor
            html_input_ids = html_encoded
            html_attention_mask = torch.ones_like(html_input_ids)

        # Determine scenario and corruption level
        scenario, corruption_level = self._get_scenario(idx, row)

        # Extract raw text fields for C-Module
        url_text_str = self._safe_string(row.get("url_text", row.get("url", "")))
        html_path_str = self._safe_string(row.get("html_path", ""))

        # Brand presence flag (for C-Module gating); default to 0 if missing
        try:
            bp_val = int(row.get("brand_present", 0))
        except Exception:
            bp_val = 0

        return {
            "id": sample_id,
            "url": url_ids,
            "html": {
                "input_ids": html_input_ids,
                "attention_mask": html_attention_mask,
            },
            "visual": image_tensor,
            "label": torch.tensor(label, dtype=torch.long),
            "brand_present": torch.tensor(bp_val, dtype=torch.long),
            "image_path": image_path_str,  # For C-Module OCR
            "url_text": url_text_str,  # For C-Module brand extraction
            "html_path": html_path_str,  # For C-Module brand extraction
            "meta": {
                "scenario": scenario,
                "corruption_level": corruption_level,
                "protocol": self.protocol,
            },
        }

    @staticmethod
    def _safe_string(value: Any) -> str:
        if isinstance(value, str):
            return value
        return "" if pd.isna(value) else str(value)

    def _get_scenario(self, idx: int, row: pd.Series) -> Tuple[str, str]:
        """
        Determine scenario and corruption level for a sample.

        Returns:
            Tuple[str, str]: (scenario, corruption_level)
            - scenario: 'clean', 'light', 'medium', 'heavy', 'brandood'
            - corruption_level: 'clean', 'light', 'medium', 'heavy'
        """
        # If scenario is explicitly overridden (e.g., for multi-corruption test sets)
        if self.scenario_override:
            corruption_level = (
                self.scenario_override
                if self.scenario_override != "brandood"
                else "clean"
            )
            return self.scenario_override, corruption_level

        # Check for explicit corruption_level column in CSV
        if "corruption_level" in row and pd.notna(row["corruption_level"]):
            corr_level = str(row["corruption_level"]).lower()
            if corr_level in ["light", "medium", "heavy"]:
                return corr_level, corr_level
            elif corr_level == "clean":
                return "clean", "clean"

        # Infer from image path fields (check all candidate fields)
        path_fields = [
            "img_path",
            "img_path_corrupt",
            "img_path_full",
            "img_path_cached",
            "image_path",
        ]
        for field in path_fields:
            if field in row and pd.notna(row[field]):
                path_str = self._safe_string(row[field])
                if path_str:
                    path_lower = path_str.lower()
                    if "corrupt" in path_lower or "corruption" in path_lower:
                        # Try to infer level from path
                        if (
                            "light" in path_lower
                            or "_l_" in path_lower
                            or "_l/" in path_lower
                            or "/l/" in path_lower
                        ):
                            return "light", "light"
                        elif (
                            "medium" in path_lower
                            or "_m_" in path_lower
                            or "_m/" in path_lower
                            or "/m/" in path_lower
                        ):
                            return "medium", "medium"
                        elif (
                            "heavy" in path_lower
                            or "_h_" in path_lower
                            or "_h/" in path_lower
                            or "/h/" in path_lower
                        ):
                            return "heavy", "heavy"
                        else:
                            # Generic corruption without level
                            return "medium", "medium"

        # Also try the selected image path (if files exist)
        image_path_str = self._select_image_path(row)
        if image_path_str:
            path_lower = image_path_str.lower()
            if "corrupt" in path_lower or "corruption" in path_lower:
                # Try to infer level from path
                if (
                    "light" in path_lower
                    or "_l_" in path_lower
                    or "_l/" in path_lower
                    or "/l/" in path_lower
                ):
                    return "light", "light"
                elif (
                    "medium" in path_lower
                    or "_m_" in path_lower
                    or "_m/" in path_lower
                    or "/m/" in path_lower
                ):
                    return "medium", "medium"
                elif (
                    "heavy" in path_lower
                    or "_h_" in path_lower
                    or "_h/" in path_lower
                    or "/h/" in path_lower
                ):
                    return "heavy", "heavy"
                else:
                    # Generic corruption without level
                    return "medium", "medium"

        # Check protocol-specific scenarios
        if self.protocol == "brandood":
            return "brandood", "clean"

        # Default: clean IID scenario
        return "clean", "clean"

    def _tokenize_url(self, url_text: str) -> torch.LongTensor:
        char_ids = [min(ord(c), self.url_vocab_size - 1) for c in url_text]
        if len(char_ids) > self.url_max_len:
            char_ids = char_ids[: self.url_max_len]
        else:
            char_ids += [0] * (self.url_max_len - len(char_ids))
        return torch.tensor(char_ids, dtype=torch.long)

    def _load_cached_url(self, row: pd.Series) -> Optional[torch.Tensor]:
        """
        加载缓存的URL tokens。
        如果url_tokens_path存在且文件存在，则加载缓存的tensor。
        """
        if "url_tokens_path" in row and pd.notna(row["url_tokens_path"]):
            cached_path = self._resolve_cached_path(row["url_tokens_path"])
            if cached_path and cached_path.exists():
                try:
                    cached_tensor = torch.load(cached_path)
                    log.debug(f">> 加载缓存的URL tokens: {cached_path}")
                    return cached_tensor
                except Exception as exc:
                    log.warning(f"加载URL缓存失败 {cached_path}: {exc}")
        return None

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

    def _resolve_cached_path(self, cached_rel: str) -> Optional[Path]:
        """
        解析缓存文件路径，将相对路径转换为绝对路径。
        """
        if not cached_rel or pd.isna(cached_rel):
            return None
        path = Path(cached_rel)
        if not path.is_absolute() and self.cache_root:
            path = self.cache_root / path
        return path if path.exists() else None

    def _select_image_path(self, row: pd.Series) -> Optional[str]:
        """
        根据可用字段挑选一个存在的图像路径，供视觉 OCR 使用。
        优先顺序（针对OCR优化，需要高分辨率原图）：
            1. img_path (原始全尺寸图像 - 最适合OCR)
            2. img_path_corrupt
            3. img_path_full (预处理后的224x224图像 - 对OCR来说太小)
            4. img_path_cached
            5. image_path
        """
        # 优先检查 img_path（原始全尺寸图像，最适合OCR）
        candidates = [
            ("img_path", False, False),  # 原始图像优先用于OCR
            ("img_path_corrupt", True, False),
            ("img_path_full", False, False),  # 预处理图像作为备选
            ("img_path_cached", False, True),
            ("image_path", False, False),
        ]
        for field, prefer_corrupt, is_cached in candidates:
            value = row.get(field)
            if value is None or (isinstance(value, float) and pd.isna(value)):
                continue
            value_str = self._safe_string(value).strip()
            if not value_str:
                continue
            if is_cached:
                resolved = self._resolve_cached_path(value_str)
            else:
                resolved = self._resolve_image_path(value_str, prefer_corrupt)
            if resolved and resolved.exists():
                return str(resolved)
        return None

    def _load_cached_html(self, row: pd.Series) -> Optional[torch.Tensor]:
        """
        加载缓存的HTML tokens。
        如果html_tokens_path存在且文件存在，则加载缓存的tensor。
        """
        if "html_tokens_path" in row and pd.notna(row["html_tokens_path"]):
            cached_path = self._resolve_cached_path(row["html_tokens_path"])
            if cached_path and cached_path.exists():
                try:
                    cached_tensor = torch.load(cached_path)
                    log.debug(f">> 加载缓存的HTML tokens: {cached_path}")
                    return cached_tensor
                except Exception as exc:
                    log.warning(f"加载HTML缓存失败 {cached_path}: {exc}")
        return None

    def _load_image(self, row: pd.Series) -> torch.Tensor:
        resolved = self._select_image_path(row)
        if not resolved:
            img = Image.new("RGB", (224, 224))
            return self.visual_transform(img)

        try:
            path = Path(resolved)
            if path.exists():
                img = Image.open(path)
                width, height = img.size
                total_pixels = width * height
                max_pixels = 10_000_000  # 10MP阈值
                if total_pixels > max_pixels:
                    scale = (max_pixels / total_pixels) ** 0.5
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    img.thumbnail((new_width, new_height), Image.Resampling.LANCZOS)
                img = img.convert("RGB")
            else:
                log.warning("Image path %s missing; using blank placeholder.", path)
                img = Image.new("RGB", (224, 224))
        except Exception as exc:
            log.warning(
                "Failed to load image %s (%s); using blank placeholder.", resolved, exc
            )
            img = Image.new("RGB", (224, 224))
        return self.visual_transform(img)

    def _load_cached_image(self, row: pd.Series) -> Optional[torch.Tensor]:
        """
        加载缓存的图像。
        支持JPG和PT格式，JPG需要transform，PT直接加载tensor。
        """
        if "img_path_cached" in row and pd.notna(row["img_path_cached"]):
            cached_path = self._resolve_cached_path(row["img_path_cached"])
            if cached_path and cached_path.exists():
                try:
                    if cached_path.suffix.lower() in [".pt", ".pth"]:
                        # 直接加载tensor
                        cached_tensor = torch.load(cached_path)
                        log.debug(f">> 加载缓存的图像tensor: {cached_path}")
                        return cached_tensor
                    elif cached_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                        # 加载JPG图像并应用transform
                        img = Image.open(cached_path).convert("RGB")
                        log.debug(f">> 加载缓存的JPG图像: {cached_path}")
                        return self.visual_transform(img)
                    else:
                        log.warning(f"不支持的缓存图像格式: {cached_path.suffix}")
                except Exception as exc:
                    log.warning(f"加载图像缓存失败 {cached_path}: {exc}")
        return None

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
        test_ood_csv: Optional[str] = None,  # OOD测试集备用
        split_protocol: str = "presplit",
        batch_size: int = 128,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: Optional[bool] = None,
        prefetch_factor: Optional[int] = None,
        use_augmentation: bool = False,
        url_max_len: int = 200,
        url_vocab_size: int = 128,
        html_max_len: int = 256,
        image_dir: Optional[str] = None,
        corrupt_root: Optional[str] = "workspace/data/corrupt",
        use_presplit: bool = True,
        preload_html: Optional[bool] = None,
        preprocessed_train_dir: Optional[str] = None,
        preprocessed_val_dir: Optional[str] = None,
        preprocessed_test_dir: Optional[str] = None,
        cfg: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.master_csv = Path(master_csv) if master_csv else None
        self.train_csv = Path(train_csv) if train_csv else None
        self.val_csv = Path(val_csv) if val_csv else None
        self.test_csv = Path(test_csv) if test_csv else None
        self.test_ood_csv = Path(test_ood_csv) if test_ood_csv else None
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
        self.prefetch_factor = prefetch_factor
        self.use_augmentation = use_augmentation
        self.url_max_len = url_max_len
        self.url_vocab_size = url_vocab_size
        self.html_max_len = html_max_len
        self.image_dir = (
            Path(image_dir) if image_dir else Path("data/processed/screenshots")
        )
        self.corrupt_root = Path(corrupt_root) if corrupt_root else None

        # 预处理目录：如果提供了预处理目录，自动禁用HTML预加载
        self.preprocessed_train_dir = (
            Path(preprocessed_train_dir) if preprocessed_train_dir else None
        )
        self.preprocessed_val_dir = (
            Path(preprocessed_val_dir) if preprocessed_val_dir else None
        )
        self.preprocessed_test_dir = (
            Path(preprocessed_test_dir) if preprocessed_test_dir else None
        )

        # 如果提供了预处理目录，自动设置preload_html为False
        if preload_html is None:
            if any(
                [
                    self.preprocessed_train_dir,
                    self.preprocessed_val_dir,
                    self.preprocessed_test_dir,
                ]
            ):
                self.preload_html = False
                log.info(
                    ">> Preprocessed directories provided, disabling HTML preloading"
                )
            else:
                self.preload_html = True  # 默认行为
        else:
            self.preload_html = preload_html

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None
        self.split_metadata: Dict[str, Any] = {}

    def _maybe_use_cached(self) -> None:
        """
        自动检测并使用缓存CSV文件。
        如果存在对应的*_cached.csv文件，则自动切换train/val/test_csv路径。
        """
        # 检查并切换train_csv
        if self.train_csv and self.train_csv.exists():
            cached_train_csv = (
                self.train_csv.parent / f"{self.train_csv.stem}_cached.csv"
            )
            if cached_train_csv.exists():
                log.info(f">> 检测到缓存训练CSV，切换到: {cached_train_csv}")
                self.train_csv = cached_train_csv
            else:
                log.info(f">> 未找到缓存训练CSV，使用原始文件: {self.train_csv}")

        # 检查并切换val_csv
        if self.val_csv and self.val_csv.exists():
            cached_val_csv = self.val_csv.parent / f"{self.val_csv.stem}_cached.csv"
            if cached_val_csv.exists():
                log.info(f">> 检测到缓存验证CSV，切换到: {cached_val_csv}")
                self.val_csv = cached_val_csv
            else:
                log.info(f">> 未找到缓存验证CSV，使用原始文件: {self.val_csv}")

        # 检查并切换test_csv
        if self.test_csv and self.test_csv.exists():
            cached_test_csv = self.test_csv.parent / f"{self.test_csv.stem}_cached.csv"
            if cached_test_csv.exists():
                log.info(f">> 检测到缓存测试CSV，切换到: {cached_test_csv}")
                self.test_csv = cached_test_csv
            else:
                log.info(f">> 未找到缓存测试CSV，使用原始文件: {self.test_csv}")

    # ------------------------------------------------------------------ #
    def setup(self, stage: Optional[str] = None) -> None:
        # 自动检测并使用缓存CSV文件
        self._maybe_use_cached()

        full_df = None
        if not (self.train_csv and self.val_csv and self.test_csv):
            full_df = self._load_dataframe()
        train_df, val_df, test_df = self._determine_splits(full_df)

        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        train_transform, eval_transform = self._build_transforms()

        # Infer protocol from split metadata or use default
        protocol = self.split_metadata.get("protocol", self.split_protocol)
        if protocol == "presplit":
            # Try to infer from CSV paths
            if self.train_csv and "brandood" in str(self.train_csv).lower():
                protocol = "brandood"
            elif self.train_csv and "temporal" in str(self.train_csv).lower():
                protocol = "temporal"
            else:
                protocol = "iid"

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
                preload_html=self.preload_html,
                cache_root=self.preprocessed_train_dir,  # 训练集缓存目录
                protocol=protocol,
                scenario=None,  # Will be inferred per sample
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
                preload_html=self.preload_html,
                cache_root=self.preprocessed_val_dir,  # 验证集缓存目录
                protocol=protocol,
                scenario=None,  # Will be inferred per sample
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
                preload_html=self.preload_html,
                cache_root=self.preprocessed_test_dir,  # 测试集缓存目录
                protocol=protocol,
                scenario=None,  # Will be inferred per sample
            )

    # ------------------------------------------------------------------ #
    def train_dataloader(self) -> DataLoader:
        loader_kwargs = {
            "batch_size": self.batch_size,
            "shuffle": True,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "persistent_workers": self.persistent_workers,
            "collate_fn": multimodal_collate_fn,  # Use custom collate for string fields
        }
        # prefetch_factor只在num_workers > 0时有效
        if self.num_workers > 0 and self.prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = self.prefetch_factor
        return DataLoader(self.train_dataset, **loader_kwargs)

    def val_dataloader(self) -> DataLoader:
        loader_kwargs = {
            "batch_size": self.batch_size,
            "shuffle": False,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "persistent_workers": self.persistent_workers,
            "collate_fn": multimodal_collate_fn,  # Use custom collate for string fields
        }
        # prefetch_factor只在num_workers > 0时有效
        if self.num_workers > 0 and self.prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = self.prefetch_factor
        return DataLoader(self.val_dataset, **loader_kwargs)

    def test_dataloader(self) -> DataLoader:
        loader_kwargs = {
            "batch_size": self.batch_size,
            "shuffle": False,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "persistent_workers": self.persistent_workers,
            "collate_fn": multimodal_collate_fn,  # Use custom collate for string fields
        }
        # prefetch_factor只在num_workers > 0时有效
        if self.num_workers > 0 and self.prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = self.prefetch_factor
        return DataLoader(self.test_dataset, **loader_kwargs)

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

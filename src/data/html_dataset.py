"""
HTML 数据集模块
用于 BERT-based HTML 钓鱼检测
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset

from src.utils.html_clean import load_html_from_path, clean_html


class HtmlDataset(Dataset):
    """
    HTML modality dataset with BERT tokenization.
    Reads HTML files from paths in CSV and tokenizes with transformers.

    Returns:
        Tuple of (input_ids, attention_mask, label) - all torch.Tensor

    Example:
        >>> dataset = HtmlDataset(
        ...     "data/processed/html_train_v2.csv",
        ...     bert_model="bert-base-uncased",
        ...     max_len=512
        ... )
        >>> input_ids, attention_mask, label = dataset[0]
        >>> print(input_ids.shape)  # (512,)
    """

    def __init__(
        self,
        csv_path: str | Path,
        *,
        bert_model: str = "bert-base-uncased",
        max_len: int = 512,
        html_col: str = "html_path",
        label_col: str = "label",
    ) -> None:
        """
        Args:
            csv_path: Path to CSV containing html_path and label columns
            bert_model: Hugging Face model name for tokenizer
            max_len: Max token length (BERT max = 512)
            html_col: Column name for HTML file paths
            label_col: Column name for labels (0=benign, 1=phishing)
        """
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")

        self.max_len = max_len
        self.html_col = html_col
        self.label_col = label_col

        # 在 __init__ 中加载 tokenizer（避免每次 __getitem__ 重复加载）
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(bert_model)

        # 加载数据
        frame = pd.read_csv(self.csv_path)
        required = {html_col, label_col}
        if not required.issubset(frame.columns):
            missing = ", ".join(sorted(required - set(frame.columns)))
            raise ValueError(f"Missing required columns: {missing}")

        self._html_paths = frame[html_col].fillna("").astype(str).tolist()
        self._labels = frame[label_col].astype(int).tolist()

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(
        self, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            input_ids: (max_len,) - BERT token IDs
            attention_mask: (max_len,) - BERT attention mask
            label: (,) - Binary label (0 or 1)
        """
        html_path = self._html_paths[index]
        label = self._labels[index]

        # 加载并清洗 HTML
        html_text = load_html_from_path(html_path)
        clean_text = clean_html(html_text)

        # 如果清洗后为空，使用占位符（避免 BERT tokenizer 报错）
        if not clean_text:
            clean_text = "[EMPTY]"

        # BERT tokenization
        tokens = self.tokenizer(
            clean_text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # 返回 tuple 格式（与 URL dataset 对齐）
        return (
            tokens["input_ids"].squeeze(0),  # (max_len,)
            tokens["attention_mask"].squeeze(0),  # (max_len,)
            torch.tensor(label, dtype=torch.long),  # (,)
        )

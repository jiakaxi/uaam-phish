from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset


def encode_url(
    text: str,
    *,
    max_len: int,
    vocab_size: int,
    pad_id: int,
) -> List[int]:
    text = (text or "")[:max_len]
    tokens: List[int] = []
    for ch in text:
        code = ord(ch)
        if code < 0:
            code = 0
        if code >= vocab_size:
            code = vocab_size - 1
        tokens.append(code)
    if len(tokens) < max_len:
        tokens.extend([pad_id] * (max_len - len(tokens)))
    return tokens


class UrlDataset(Dataset):
    """Character-level URL dataset used by the URL-only Lightning pipeline."""

    def __init__(
        self,
        csv_path: str | Path,
        *,
        max_len: int = 256,
        vocab_size: int = 128,
        pad_id: int = 0,
    ) -> None:
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")

        self.max_len = max_len
        self.vocab_size = vocab_size
        self.pad_id = pad_id

        frame = pd.read_csv(self.csv_path)
        required = {"url_text", "label"}
        if not required.issubset(frame.columns):
            missing = ", ".join(sorted(required - set(frame.columns)))
            raise ValueError(f"Missing required columns: {missing}")
        self._texts = frame["url_text"].fillna("").astype(str).tolist()
        self._labels = frame["label"].astype(int).tolist()

    def __len__(self) -> int:
        return len(self._texts)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        text = self._texts[index]
        label = self._labels[index]
        encoded = encode_url(
            text,
            max_len=self.max_len,
            vocab_size=self.vocab_size,
            pad_id=self.pad_id,
        )
        return torch.tensor(encoded, dtype=torch.long), torch.tensor(
            label, dtype=torch.long
        )

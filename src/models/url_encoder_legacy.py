"""
Legacy URL 编码器 - 基于 HuggingFace Transformers
保留用于向后兼容多模态实验
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel


class UrlBertEncoder(nn.Module):
    """
    Legacy HuggingFace-based encoder kept for backward compatibility with
    multimodal experiments.

    ⚠️  不建议用于新实验，请使用 URLEncoder（字符级）
    """

    def __init__(
        self, pretrained_name: str = "bert-base-uncased", dropout: float = 0.1
    ):
        super().__init__()
        self.config = AutoConfig.from_pretrained(
            pretrained_name, output_hidden_states=False
        )
        self.backbone = AutoModel.from_pretrained(pretrained_name, config=self.config)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Identity()

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        out = self.backbone(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch.get("token_type_ids"),
            return_dict=True,
        )
        x = out.last_hidden_state[:, 0, :]
        x = self.dropout(x)
        return self.proj(x)

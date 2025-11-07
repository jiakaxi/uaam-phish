"""
Archived legacy URL encoder (BERT-based).

Retained for reference; not used in S0 baseline.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel


class UrlBertEncoder(nn.Module):
    def __init__(
        self, pretrained_name: str = "bert-base-uncased", dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.config = AutoConfig.from_pretrained(
            pretrained_name, output_hidden_states=False
        )
        self.backbone = AutoModel.from_pretrained(pretrained_name, config=self.config)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Identity()

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        outputs = self.backbone(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch.get("token_type_ids"),
            return_dict=True,
        )
        cls_token = outputs.last_hidden_state[:, 0, :]
        cls_token = self.dropout(cls_token)
        return self.proj(cls_token)

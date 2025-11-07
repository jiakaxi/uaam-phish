"""
HTML encoder (Sec. 4.6.1): bert-base-uncased + 2-layer projection to 256-d.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel


class HTMLEncoder(nn.Module):
    def __init__(
        self,
        bert_model: str = "bert-base-uncased",
        hidden_dim: int = 768,
        output_dim: int = 256,
        dropout: float = 0.1,
        freeze_bert: bool = False,
    ) -> None:
        super().__init__()

        self.bert = AutoModel.from_pretrained(bert_model)
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        projection_hidden = hidden_dim // 2
        self.projection = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, projection_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(projection_hidden, output_dim),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = outputs.last_hidden_state[:, 0, :]
        z_html = self.projection(cls_token)
        return z_html

"""
URL 编码器 - 字符级 BiLSTM 实现
用于 URL-only 钓鱼检测基线
"""

from __future__ import annotations

import torch
import torch.nn as nn


class URLEncoder(nn.Module):
    """Character-level BiLSTM encoder producing vectors in R^256 by default."""

    def __init__(
        self,
        vocab_size: int = 128,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.1,
        pad_id: int = 0,
        proj_dim: int = 256,
    ) -> None:
        super().__init__()
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_id,
        )
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        output_dim = hidden_dim * (2 if bidirectional else 1)
        self.project = nn.Linear(output_dim, proj_dim)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedding(input_ids)
        embeddings = self.dropout(embeddings)
        _, (hidden_states, _) = self.lstm(embeddings)
        if self.bidirectional:
            forward = hidden_states[-2]
            backward = hidden_states[-1]
            features = torch.cat([forward, backward], dim=1)
        else:
            features = hidden_states[-1]
        features = self.dropout(features)
        return self.project(features)

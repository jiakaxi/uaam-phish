"""
HTML 编码器 - BERT-base 实现
用于 HTML-only 钓鱼检测基线
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel


class HTMLEncoder(nn.Module):
    """
    BERT-base encoder for HTML semantic content.
    Extracts [CLS] token embedding and projects to 256-D.

    Architecture per thesis §3.3:
    - BERT-base-uncased (110M params) 或 distilbert-base-uncased (66M params)
    - [CLS] token extraction
    - Linear projection: 768 -> 256
    - Dropout regularization

    Output dimension matches URLEncoder (256-D) for future fusion compatibility.
    """

    def __init__(
        self,
        bert_model: str = "bert-base-uncased",
        hidden_dim: int = 768,
        output_dim: int = 256,
        dropout: float = 0.1,
        freeze_bert: bool = False,
    ) -> None:
        """
        Args:
            bert_model: Hugging Face model name (bert-base-uncased, distilbert-base-uncased)
            hidden_dim: BERT output dimension (768 for base/distilbert)
            output_dim: Projection dimension (must be 256 for fusion)
            dropout: Dropout rate
            freeze_bert: Whether to freeze BERT weights (faster training, lower memory)
        """
        super().__init__()

        # Load pretrained BERT
        self.bert = AutoModel.from_pretrained(bert_model)

        # Optionally freeze BERT parameters
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        # Projection layer: 768 -> 256
        self.projection = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(hidden_dim, output_dim)
        )

    def forward(
        self,
        input_ids: torch.Tensor,  # (batch, seq_len)
        attention_mask: torch.Tensor,  # (batch, seq_len)
    ) -> torch.Tensor:
        """
        Extract [CLS] token and project to 256-D.

        Args:
            input_ids: BERT token IDs (batch, seq_len)
            attention_mask: BERT attention mask (batch, seq_len)

        Returns:
            z_html: (batch, 256) - Projected HTML embeddings
        """
        # BERT forward pass
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # Extract [CLS] token (first token)
        cls_embedding = bert_output.last_hidden_state[:, 0, :]  # (batch, 768)

        # Project to 256-D
        z_html = self.projection(cls_embedding)  # (batch, 256)

        return z_html

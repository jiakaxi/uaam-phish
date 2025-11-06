"""
多模态拼接融合基线 (S0: Early Fusion via Concatenation)

严格遵循论文设计：
- 无注意力机制
- 无门控机制
- 无自适应权重
- 纯线性投影分类头

输出 logits（不含 Sigmoid），方便后续温度缩放与校准评估。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class BaselineConcatFusion(nn.Module):
    """
    Early Fusion (Concatenation) baseline for multimodal phishing detection.

    Architecture:
        z_fused = concat([z_url, z_html, z_visual])  # [B, 768]
        logits = Linear(768 -> 1)(Dropout(z_fused))  # [B, 1]

    Args:
        url_dim: URL encoder output dimension (default: 256)
        html_dim: HTML encoder output dimension (default: 256)
        visual_dim: Visual encoder output dimension (default: 256)
        dropout: Dropout rate before classifier (default: 0.1)

    Input:
        z_url: [B, url_dim] - URL embeddings from BiLSTM
        z_html: [B, html_dim] - HTML embeddings from BERT
        z_visual: [B, visual_dim] - Visual embeddings from ResNet-50

    Output:
        logits: [B, 1] - Raw logits (no Sigmoid), for BCEWithLogitsLoss
    """

    def __init__(
        self,
        url_dim: int = 256,
        html_dim: int = 256,
        visual_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.url_dim = url_dim
        self.html_dim = html_dim
        self.visual_dim = visual_dim
        concat_dim = url_dim + html_dim + visual_dim  # 768

        # Classifier: Dropout -> Linear(768 -> 1)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(concat_dim, 1))

    def forward(
        self,
        z_url: torch.Tensor,  # [B, 256]
        z_html: torch.Tensor,  # [B, 256]
        z_visual: torch.Tensor,  # [B, 256]
    ) -> torch.Tensor:
        """
        Forward pass: concatenate three modalities and classify.

        Args:
            z_url: URL embeddings [B, 256]
            z_html: HTML embeddings [B, 256]
            z_visual: Visual embeddings [B, 256]

        Returns:
            logits: [B, 1] - Raw classification logits
        """
        # Concatenate along feature dimension
        z_fused = torch.cat([z_url, z_html, z_visual], dim=1)  # [B, 768]

        # Classify
        logits = self.classifier(z_fused)  # [B, 1]

        return logits

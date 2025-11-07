"""
Early-fusion concatenation module (Sec. 4.6.1).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class BaselineConcatFusion(nn.Module):
    """
    Concatenate modality embeddings and produce logits.

    Architecture:
        z_fused = [z_url; z_html; z_visual] in R^768
        logits = Linear(768->1)(Dropout(z_fused))
    """

    def __init__(
        self,
        url_dim: int = 256,
        html_dim: int = 256,
        visual_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        concat_dim = url_dim + html_dim + visual_dim
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(concat_dim, 1))

    def concat(
        self,
        z_url: torch.Tensor,
        z_html: torch.Tensor,
        z_visual: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat([z_url, z_html, z_visual], dim=1)

    def classify(self, z_fused: torch.Tensor) -> torch.Tensor:
        return self.classifier(z_fused)

    def forward(
        self,
        z_url: torch.Tensor,
        z_html: torch.Tensor,
        z_visual: torch.Tensor,
    ) -> torch.Tensor:
        return self.classifier(self.concat(z_url, z_html, z_visual))

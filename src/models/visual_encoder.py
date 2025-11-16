"""
Visual 编码器 - ResNet-50 实现
用于 Visual-only 钓鱼检测基线
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class VisualEncoder(nn.Module):
    """
    ResNet-50 encoder for visual modality (screenshot images).
    Extracts CNN features and projects to 256-D.

    Architecture per thesis §3.2.1 "Visual Encoder (ResNet-50)":
    - ResNet-50 pretrained on ImageNet (110M params)
    - Replace classification head with: AdaptiveAvgPool2d -> Dropout -> Linear(2048->256)
    - Optional backbone freezing for warm-up

    Output dimension matches URLEncoder/HTMLEncoder (256-D) for future fusion compatibility.
    """

    def __init__(
        self,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        embedding_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        """
        Args:
            pretrained: Whether to load ImageNet pretrained weights
            freeze_backbone: Whether to freeze ResNet backbone (faster training, lower memory)
            embedding_dim: Output embedding dimension (must be 256 for fusion)
            dropout: Dropout rate before projection layer
        """
        super().__init__()

        # Load pretrained ResNet-50
        if pretrained:
            weights = models.ResNet50_Weights.IMAGENET1K_V1
            self.resnet = models.resnet50(weights=weights)
        else:
            self.resnet = models.resnet50(weights=None)

        # Remove original classification head (fc layer)
        # ResNet-50 has 2048-D features before fc
        self.resnet.fc = nn.Identity()

        # Optionally freeze backbone parameters
        if freeze_backbone:
            for param in self.resnet.parameters():
                param.requires_grad = False

        # Custom projection head: 2048 -> 256
        self.projection = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(2048, embedding_dim)
        )

    def forward(
        self, images: torch.Tensor, return_logits: bool = False  # (batch, 3, 224, 224)
    ) -> torch.Tensor:
        """
        Extract visual features and project to 256-D.

        Args:
            images: Input images (batch, 3, 224, 224)
            return_logits: Unused (kept for API compatibility with future fusion)

        Returns:
            z_visual: (batch, 256) - Projected visual embeddings
        """
        # ResNet forward pass (without fc layer)
        features = self.resnet(images)  # (batch, 2048)

        # Project to 256-D
        z_visual = self.projection(features)  # (batch, 256)

        return z_visual



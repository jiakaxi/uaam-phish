"""
Adaptive Fusion Module for S4 RCAF Full.

Implements the complete S4 fusion pipeline with learned lambda_c weights.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.modules.fusion.lambda_gate import LambdaGate
from src.utils.logging import get_logger


log = get_logger(__name__)


class AdaptiveFusion(nn.Module):
    """
    Adaptive fusion module that combines modality predictions using
    learned lambda_c weights and reliability/consistency scores.

    Forward pipeline:
        1. lambda_c = LambdaGate(r_m, c_m)  # [B, M]
        2. U_m = r_m + lambda_c * c_m       # [B, M] unified trust scores
        3. alpha_m = softmax(gamma * U_m)   # [B, M] fusion weights
        4. p_fused = sum(alpha_m * p_m)     # [B, num_classes]

    Args:
        num_modalities: Number of modalities (default: 3 for URL/HTML/Visual)
        num_classes: Number of output classes (default: 2 for binary)
        hidden_dim: Hidden dimension for lambda gate (default: 16)
        temperature: Temperature scaling for softmax (gamma, default: 2.0)
    """

    def __init__(
        self,
        num_modalities: int = 3,
        num_classes: int = 2,
        hidden_dim: int = 16,
        temperature: float = 2.0,
    ) -> None:
        super().__init__()
        self.num_modalities = num_modalities
        self.num_classes = num_classes
        self.temperature = temperature

        # Lambda gate for learning adaptive weights
        self.lambda_gate = LambdaGate(
            hidden_dim=hidden_dim,
            num_modalities=num_modalities,
        )

    def forward(
        self,
        probs_list: List[torch.Tensor],
        r_m: torch.Tensor,
        c_m: torch.Tensor,
        modality_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform adaptive fusion of modality predictions.

        Args:
            probs_list: List of M tensors, each [B, num_classes] containing
                       probability distributions from each modality
            r_m: Reliability scores [B, M]
            c_m: Consistency scores [B, M], should be in [0, 1] range
            modality_mask: Optional boolean mask [B, M] indicating valid modalities
                          (True = valid, False = missing/NaN)

        Returns:
            Tuple of:
                - p_fused: Fused probabilities [B, num_classes]
                - alpha_m: Fusion weights [B, M]
                - lambda_c: Adaptive consistency weights [B, M]
                - U_m: Unified trust scores [B, M]
        """
        num_modalities = len(probs_list)

        # Validate inputs
        assert (
            num_modalities == self.num_modalities
        ), f"Expected {self.num_modalities} modalities, got {num_modalities}"
        assert (
            r_m.shape[1] == num_modalities
        ), f"r_m has {r_m.shape[1]} modalities, expected {num_modalities}"
        assert (
            c_m.shape == r_m.shape
        ), f"c_m shape {c_m.shape} must match r_m shape {r_m.shape}"

        # Stack probabilities [M, B, num_classes] -> [B, M, num_classes]
        probs_stacked = torch.stack(probs_list, dim=1)  # [B, M, num_classes]

        # Determine valid modalities if mask not provided
        if modality_mask is None:
            # Infer from NaN/Inf in inputs
            r_valid = torch.isfinite(r_m)
            c_valid = torch.isfinite(c_m)
            probs_valid = torch.all(torch.isfinite(probs_stacked), dim=-1)  # [B, M]
            modality_mask = r_valid & c_valid & probs_valid

        # Handle case where all modalities are invalid (shouldn't happen, but be safe)
        num_valid = modality_mask.sum(dim=1, keepdim=True)  # [B, 1]
        if torch.any(num_valid == 0):
            # Find which samples have no valid modalities
            invalid_samples = (num_valid == 0).squeeze()
            num_invalid = invalid_samples.sum().item()
            log.warning(
                "Some samples have no valid modalities! Using uniform weights. "
                f"Affected: {num_invalid}/{len(invalid_samples)} samples in batch. "
                f"r_m stats: mean={r_m.mean():.4f}, has_nan={torch.isnan(r_m).any()}, "
                f"c_m stats: mean={c_m.mean():.4f}, has_nan={torch.isnan(c_m).any()}"
            )
            modality_mask = torch.ones_like(modality_mask)
            num_valid = modality_mask.sum(dim=1, keepdim=True)

        # 1. Compute adaptive lambda_c using the gate
        lambda_c = self.lambda_gate(r_m, c_m, mask=modality_mask)  # [B, M]

        # 2. Compute unified trust scores U_m = r_m + lambda_c * c_m
        # Clean inputs first
        r_m_clean = torch.nan_to_num(r_m, nan=0.0, posinf=0.0, neginf=0.0)
        c_m_clean = torch.nan_to_num(c_m, nan=0.0, posinf=0.0, neginf=0.0)

        U_m = r_m_clean + lambda_c * c_m_clean  # [B, M]

        # Apply mask to U_m - set invalid modalities to -inf so they get 0 weight
        U_m_masked = torch.where(
            modality_mask, U_m, torch.full_like(U_m, float("-inf"))
        )

        # 3. Compute fusion weights alpha_m = softmax(gamma * U_m)
        # Temperature scaling
        U_m_scaled = self.temperature * U_m_masked  # [B, M]

        # Softmax over modalities (automatically handles -inf by giving 0 weight)
        alpha_m = F.softmax(U_m_scaled, dim=1)  # [B, M]

        # Safety check: if all modalities were masked, use uniform weights
        alpha_m = torch.where(
            torch.isfinite(alpha_m), alpha_m, torch.ones_like(alpha_m) / num_valid
        )

        # 4. Fuse predictions: p_fused = sum(alpha_m * p_m)
        # Expand alpha_m for broadcasting: [B, M] -> [B, M, 1]
        alpha_m_expanded = alpha_m.unsqueeze(-1)  # [B, M, 1]

        # Weighted sum: [B, M, num_classes] * [B, M, 1] -> [B, M, num_classes] -> [B, num_classes]
        p_fused = (probs_stacked * alpha_m_expanded).sum(dim=1)  # [B, num_classes]

        # Normalize to ensure valid probability distribution
        p_fused = p_fused / p_fused.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        return p_fused, alpha_m, lambda_c, U_m

    def extra_repr(self) -> str:
        """String representation for debugging."""
        return (
            f"num_modalities={self.num_modalities}, "
            f"num_classes={self.num_classes}, "
            f"temperature={self.temperature}"
        )


if __name__ == "__main__":
    # Quick test
    print("Testing AdaptiveFusion...")

    batch_size = 4
    num_modalities = 3
    num_classes = 2

    # Create sample modality predictions (probabilities)
    probs_url = F.softmax(torch.randn(batch_size, num_classes), dim=-1)
    probs_html = F.softmax(torch.randn(batch_size, num_classes), dim=-1)
    probs_visual = F.softmax(torch.randn(batch_size, num_classes), dim=-1)
    probs_list = [probs_url, probs_html, probs_visual]

    # Create sample reliability and consistency scores
    r_m = torch.rand(batch_size, num_modalities) * 0.5 + 0.3  # [0.3, 0.8]
    c_m = torch.rand(batch_size, num_modalities) * 0.6 + 0.2  # [0.2, 0.8]

    # Create fusion module
    fusion = AdaptiveFusion(
        num_modalities=num_modalities,
        num_classes=num_classes,
        hidden_dim=16,
        temperature=2.0,
    )

    # Forward pass
    p_fused, alpha_m, lambda_c, U_m = fusion(probs_list, r_m, c_m)

    print("\nInput shapes:")
    print(f"  probs_url: {probs_url.shape}")
    print(f"  r_m: {r_m.shape}")
    print(f"  c_m: {c_m.shape}")

    print("\nOutput shapes:")
    print(f"  p_fused: {p_fused.shape}")
    print(f"  alpha_m: {alpha_m.shape}")
    print(f"  lambda_c: {lambda_c.shape}")
    print(f"  U_m: {U_m.shape}")

    print("\nSample outputs (first sample):")
    print(f"  lambda_c[0]: {lambda_c[0]}")
    print(f"  U_m[0]: {U_m[0]}")
    print(f"  alpha_m[0]: {alpha_m[0]} (sum={alpha_m[0].sum():.4f})")
    print(f"  p_fused[0]: {p_fused[0]} (sum={p_fused[0].sum():.4f})")

    # Test with mask (missing visual modality for sample 0)
    mask = torch.tensor(
        [
            [True, True, False],  # Missing visual
            [True, False, True],  # Missing HTML
            [True, True, True],  # All present
            [True, True, True],  # All present
        ]
    )

    p_fused_masked, alpha_m_masked, lambda_c_masked, U_m_masked = fusion(
        probs_list, r_m, c_m, modality_mask=mask
    )

    print("\nWith modality mask:")
    print(f"  Mask[0]: {mask[0]} (missing visual)")
    print(f"  alpha_m[0]: {alpha_m_masked[0]} (sum={alpha_m_masked[0].sum():.4f})")
    print(f"  lambda_c[0]: {lambda_c_masked[0]}")

    # Test gradient flow
    loss = p_fused.sum()
    loss.backward()
    print("\nGradient flow test:")
    print(
        f"  lambda_gate.fc1.weight.grad exists: {fusion.lambda_gate.fc1.weight.grad is not None}"
    )

    # Test lambda_c variability (non-constant check)
    lambda_c_std = lambda_c.std()
    print("\nLambda_c variability:")
    print(f"  std: {lambda_c_std:.4f} (should be > 0.01 for non-constant)")

    print("\n[OK] AdaptiveFusion test passed!")

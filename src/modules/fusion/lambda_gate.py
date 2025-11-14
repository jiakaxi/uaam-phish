"""
Lambda Gate Network for S4 Adaptive Fusion.

Learns per-sample, per-modality lambda_c weights that balance
reliability (r_m) and consistency (c_m) scores.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.logging import get_logger


log = get_logger(__name__)


class LambdaGate(nn.Module):
    """
    Small attention network that computes per-sample, per-modality lambda_c.

    Architecture:
        Input: [r_m, c_m] concatenated -> [B, M, 2]
        -> Linear(2, hidden_dim)
        -> ReLU
        -> Linear(hidden_dim, 1)
        -> Sigmoid
        -> squeeze -> [B, M]

    Args:
        hidden_dim: Hidden dimension for the MLP (default: 16)
        num_modalities: Number of modalities (default: 3 for URL/HTML/Visual)
    """

    def __init__(
        self,
        hidden_dim: int = 16,
        num_modalities: int = 3,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_modalities = num_modalities

        # MLP layers
        self.fc1 = nn.Linear(2, hidden_dim)  # Input: [r_m, c_m]
        self.fc2 = nn.Linear(hidden_dim, 1)  # Output: lambda_c

        # Initialize with Xavier/He for stability
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize weights using Xavier/He initialization for stability."""
        # Use He initialization for ReLU layers
        nn.init.kaiming_normal_(self.fc1.weight, mode="fan_in", nonlinearity="relu")
        nn.init.zeros_(self.fc1.bias)

        # Use Xavier initialization for the output layer
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(
        self,
        r_m: torch.Tensor,
        c_m: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute per-sample, per-modality lambda_c weights.

        Args:
            r_m: Reliability scores [B, M] where M is number of modalities
            c_m: Consistency scores [B, M], normalized to [0, 1]
            mask: Optional boolean mask [B, M] indicating valid modalities
                  (True = valid, False = missing/NaN)

        Returns:
            lambda_c: Adaptive weights [B, M] in range (0, 1)
                     For masked modalities, returns 0.0
        """
        # Validate inputs
        assert (
            r_m.shape == c_m.shape
        ), f"r_m and c_m must have same shape, got {r_m.shape} vs {c_m.shape}"
        assert r_m.dim() == 2, f"r_m must be 2D [B, M], got shape {r_m.shape}"

        # Handle NaN/Inf in inputs - replace with zeros for masked modalities
        r_m_clean = torch.nan_to_num(r_m, nan=0.0, posinf=0.0, neginf=0.0)
        c_m_clean = torch.nan_to_num(c_m, nan=0.0, posinf=0.0, neginf=0.0)

        # Concatenate [r_m, c_m] -> [B, M, 2]
        inputs = torch.stack([r_m_clean, c_m_clean], dim=-1)  # [B, M, 2]

        # MLP forward pass
        # [B, M, 2] -> [B, M, hidden_dim]
        hidden = self.fc1(inputs)
        hidden = F.relu(hidden)

        # [B, M, hidden_dim] -> [B, M, 1]
        logits = self.fc2(hidden)

        # Sigmoid to get values in (0, 1), then squeeze
        lambda_c = torch.sigmoid(logits).squeeze(-1)  # [B, M]

        # Apply mask if provided - set masked modalities to 0
        if mask is not None:
            assert (
                mask.shape == lambda_c.shape
            ), f"Mask shape {mask.shape} must match lambda_c shape {lambda_c.shape}"
            lambda_c = torch.where(mask, lambda_c, torch.zeros_like(lambda_c))

        return lambda_c

    def extra_repr(self) -> str:
        """String representation for debugging."""
        return f"hidden_dim={self.hidden_dim}, num_modalities={self.num_modalities}"


if __name__ == "__main__":
    # Quick test
    print("Testing LambdaGate...")

    # Create sample inputs
    batch_size = 4
    num_modalities = 3

    r_m = torch.rand(batch_size, num_modalities)  # [4, 3]
    c_m = torch.rand(batch_size, num_modalities)  # [4, 3]

    # Create gate
    gate = LambdaGate(hidden_dim=16, num_modalities=num_modalities)

    # Forward pass
    lambda_c = gate(r_m, c_m)

    print(f"r_m shape: {r_m.shape}")
    print(f"c_m shape: {c_m.shape}")
    print(f"lambda_c shape: {lambda_c.shape}")
    print(f"lambda_c values:\n{lambda_c}")
    print(f"lambda_c range: [{lambda_c.min():.3f}, {lambda_c.max():.3f}]")

    # Test with mask
    mask = torch.tensor(
        [
            [True, True, False],
            [True, False, True],
            [False, True, True],
            [True, True, True],
        ]
    )
    lambda_c_masked = gate(r_m, c_m, mask=mask)
    print(f"\nWith mask:\n{lambda_c_masked}")

    # Test with NaN
    r_m_nan = r_m.clone()
    r_m_nan[0, 2] = float("nan")
    lambda_c_nan = gate(r_m_nan, c_m)
    print(f"\nWith NaN in r_m:\n{lambda_c_nan}")

    # Test gradient flow
    lambda_c.sum().backward()
    print(
        f"\nGradient flow test: fc1.weight.grad exists = {gate.fc1.weight.grad is not None}"
    )

    print("\n[OK] LambdaGate test passed!")

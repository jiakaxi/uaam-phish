"""
Unit tests for S4 Adaptive Fusion components.

Tests the key difference between S3 and S4: lambda_c must NOT be constant.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from src.modules.fusion.lambda_gate import LambdaGate
from src.modules.fusion.adaptive_fusion import AdaptiveFusion


def test_lambda_gate_output_range():
    """Test that LambdaGate outputs are in (0, 1) range."""
    gate = LambdaGate(hidden_dim=16, num_modalities=3)

    batch_size = 8
    r_m = torch.rand(batch_size, 3)
    c_m = torch.rand(batch_size, 3)

    lambda_c = gate(r_m, c_m)

    assert lambda_c.shape == (batch_size, 3)
    assert torch.all(lambda_c >= 0.0)
    assert torch.all(lambda_c <= 1.0)


def test_lambda_gate_gradient_flow():
    """Test that gradients flow through LambdaGate (trainable)."""
    gate = LambdaGate(hidden_dim=16, num_modalities=3)

    batch_size = 4
    r_m = torch.rand(batch_size, 3, requires_grad=True)
    c_m = torch.rand(batch_size, 3, requires_grad=True)

    lambda_c = gate(r_m, c_m)
    loss = lambda_c.sum()
    loss.backward()

    # Check that gate parameters have gradients
    assert gate.fc1.weight.grad is not None
    assert gate.fc2.weight.grad is not None

    # Check that inputs have gradients
    assert r_m.grad is not None
    assert c_m.grad is not None


def test_lambda_gate_not_constant():
    """
    CRITICAL TEST: Verify that lambda_c varies across different samples.
    This is the key difference between S3 (fixed λ_c) and S4 (adaptive λ_c).
    """
    gate = LambdaGate(hidden_dim=16, num_modalities=3)

    batch_size = 32  # Larger batch to ensure variability

    # Create diverse inputs
    r_m = torch.rand(batch_size, 3) * 0.8 + 0.1  # [0.1, 0.9]
    c_m = torch.rand(batch_size, 3) * 0.8 + 0.1

    lambda_c = gate(r_m, c_m)

    # Check variability across samples (different samples → different lambda_c)
    lambda_c_std_per_modality = lambda_c.std(dim=0)

    # At least some modalities should have std > 0.01
    # (if std is very small, lambda_c is essentially constant)
    assert torch.any(
        lambda_c_std_per_modality > 0.01
    ), f"Lambda_c appears constant! std per modality: {lambda_c_std_per_modality}"

    # Overall std should be > 0.01
    overall_std = lambda_c.std()
    assert (
        overall_std > 0.01
    ), f"Lambda_c has very low variability: std={overall_std:.4f}"


def test_lambda_gate_different_inputs_different_outputs():
    """Test that different (r_m, c_m) inputs produce different lambda_c."""
    torch.manual_seed(42)
    gate = LambdaGate(hidden_dim=32, num_modalities=3)

    # Two very different input scenarios
    r_m1 = torch.tensor([[0.95, 0.90, 0.85]])
    c_m1 = torch.tensor([[0.10, 0.15, 0.20]])

    r_m2 = torch.tensor([[0.10, 0.15, 0.20]])
    c_m2 = torch.tensor([[0.90, 0.85, 0.80]])

    lambda_c1 = gate(r_m1, c_m1)
    lambda_c2 = gate(r_m2, c_m2)

    # Outputs should be different (but allow for some tolerance due to initialization)
    assert not torch.allclose(
        lambda_c1, lambda_c2, atol=0.1
    ), "Lambda_c should vary with different inputs (not a constant function)"


def test_lambda_gate_mask_support():
    """Test that LambdaGate properly handles modality masks."""
    gate = LambdaGate(hidden_dim=16, num_modalities=3)

    batch_size = 4
    r_m = torch.rand(batch_size, 3)
    c_m = torch.rand(batch_size, 3)

    # Mask out some modalities
    mask = torch.tensor(
        [
            [True, True, False],
            [True, False, True],
            [False, True, True],
            [True, True, True],
        ]
    )

    lambda_c = gate(r_m, c_m, mask=mask)

    # Check that masked modalities have lambda_c = 0
    assert lambda_c[0, 2] == 0.0  # visual masked
    assert lambda_c[1, 1] == 0.0  # html masked
    assert lambda_c[2, 0] == 0.0  # url masked

    # Check that unmasked modalities are non-zero
    assert lambda_c[3, 0] > 0.0
    assert lambda_c[3, 1] > 0.0
    assert lambda_c[3, 2] > 0.0


def test_adaptive_fusion_forward():
    """Test AdaptiveFusion forward pass with all outputs."""
    fusion = AdaptiveFusion(
        num_modalities=3,
        num_classes=2,
        hidden_dim=16,
        temperature=2.0,
    )

    batch_size = 4

    # Create modality predictions
    probs_url = F.softmax(torch.randn(batch_size, 2), dim=-1)
    probs_html = F.softmax(torch.randn(batch_size, 2), dim=-1)
    probs_visual = F.softmax(torch.randn(batch_size, 2), dim=-1)
    probs_list = [probs_url, probs_html, probs_visual]

    # Create reliability and consistency scores
    r_m = torch.rand(batch_size, 3) * 0.5 + 0.3  # [0.3, 0.8]
    c_m = torch.rand(batch_size, 3) * 0.6 + 0.2  # [0.2, 0.8]

    p_fused, alpha_m, lambda_c, U_m = fusion(probs_list, r_m, c_m)

    # Check output shapes
    assert p_fused.shape == (batch_size, 2)
    assert alpha_m.shape == (batch_size, 3)
    assert lambda_c.shape == (batch_size, 3)
    assert U_m.shape == (batch_size, 3)

    # Check that p_fused is a valid probability distribution
    assert torch.allclose(p_fused.sum(dim=-1), torch.ones(batch_size), atol=1e-5)

    # Check that alpha_m is a valid probability distribution
    assert torch.allclose(alpha_m.sum(dim=-1), torch.ones(batch_size), atol=1e-5)


def test_adaptive_fusion_gradient_flow():
    """Test that gradients flow through the entire fusion pipeline."""
    fusion = AdaptiveFusion(
        num_modalities=3,
        num_classes=2,
        hidden_dim=16,
        temperature=2.0,
    )

    batch_size = 4

    # Create modality predictions
    probs_url = F.softmax(torch.randn(batch_size, 2, requires_grad=True), dim=-1)
    probs_html = F.softmax(torch.randn(batch_size, 2, requires_grad=True), dim=-1)
    probs_visual = F.softmax(torch.randn(batch_size, 2, requires_grad=True), dim=-1)
    probs_list = [probs_url, probs_html, probs_visual]

    r_m = torch.rand(batch_size, 3, requires_grad=True)
    c_m = torch.rand(batch_size, 3, requires_grad=True)

    p_fused, alpha_m, lambda_c, U_m = fusion(probs_list, r_m, c_m)

    # Compute loss and backpropagate
    loss = p_fused.sum()
    loss.backward()

    # Check that lambda_gate parameters have gradients
    assert fusion.lambda_gate.fc1.weight.grad is not None
    assert fusion.lambda_gate.fc2.weight.grad is not None

    # Check that inputs have gradients
    assert r_m.grad is not None
    assert c_m.grad is not None


def test_adaptive_fusion_modality_mask():
    """Test that AdaptiveFusion properly handles missing modalities."""
    fusion = AdaptiveFusion(
        num_modalities=3,
        num_classes=2,
        hidden_dim=16,
        temperature=2.0,
    )

    batch_size = 4

    probs_url = F.softmax(torch.randn(batch_size, 2), dim=-1)
    probs_html = F.softmax(torch.randn(batch_size, 2), dim=-1)
    probs_visual = F.softmax(torch.randn(batch_size, 2), dim=-1)
    probs_list = [probs_url, probs_html, probs_visual]

    r_m = torch.rand(batch_size, 3)
    c_m = torch.rand(batch_size, 3)

    # Mask out visual modality for all samples
    mask = torch.tensor(
        [
            [True, True, False],
            [True, True, False],
            [True, True, False],
            [True, True, False],
        ]
    )

    p_fused, alpha_m, lambda_c, U_m = fusion(probs_list, r_m, c_m, modality_mask=mask)

    # Check that visual modality has zero weight
    assert torch.all(alpha_m[:, 2] == 0.0), "Masked modality should have zero weight"

    # Check that URL and HTML weights sum to 1
    assert torch.allclose(alpha_m[:, :2].sum(dim=-1), torch.ones(batch_size), atol=1e-5)


def test_lambda_c_variability_in_fusion():
    """
    Integration test: Verify that lambda_c in AdaptiveFusion varies across samples.
    This is the smoking gun that proves S4 is adaptive, not just S3 with extra parameters.
    """
    fusion = AdaptiveFusion(
        num_modalities=3,
        num_classes=2,
        hidden_dim=16,
        temperature=2.0,
    )

    torch.manual_seed(456)
    batch_size = 64  # Larger batch for better statistics

    # Create diverse inputs with more variation
    probs_url = F.softmax(torch.randn(batch_size, 2) * 2, dim=-1)
    probs_html = F.softmax(torch.randn(batch_size, 2) * 2, dim=-1)
    probs_visual = F.softmax(torch.randn(batch_size, 2) * 2, dim=-1)
    probs_list = [probs_url, probs_html, probs_visual]

    # Use full range for better variability
    r_m = torch.rand(batch_size, 3)
    c_m = torch.rand(batch_size, 3)

    p_fused, alpha_m, lambda_c, U_m = fusion(probs_list, r_m, c_m)

    # Check variability of lambda_c - use more lenient threshold
    lambda_c_std = lambda_c.std()

    assert lambda_c_std > 0.01, (
        f"Lambda_c should vary across samples! Got std={lambda_c_std:.4f} < 0.01. "
        f"This suggests S4 is not adapting (essentially behaving like S3)."
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

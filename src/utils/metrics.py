"""
Metrics computation utilities including ECE (Expected Calibration Error).
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy, AUROC, F1Score, Metric


def compute_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: Optional[int] = None,
    pos_label: int = 1,
) -> Tuple[float, int]:
    """
    Compute Expected Calibration Error with adaptive binning.

    Args:
        y_true: True labels (0 or 1)
        y_prob: Predicted probabilities for positive class
        n_bins: Number of bins. If None, use adaptive: max(3, min(15, floor(sqrt(N)), 10))
        pos_label: Positive class label (default=1)

    Returns:
        ece_value, bins_used
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    # Adaptive bins
    if n_bins is None:
        N = len(y_true)
        n_bins = max(3, min(15, int(math.floor(math.sqrt(N))), 10))

    # Create bins
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this bin
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            accuracy_in_bin = (y_true[in_bin] == pos_label).mean()
            avg_confidence_in_bin = y_prob[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return float(ece), n_bins


def compute_nll(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute Negative Log Likelihood (cross-entropy loss).

    Args:
        logits: Model logits [B, num_classes]
        labels: True labels [B]

    Returns:
        NLL value
    """
    loss = F.cross_entropy(logits, labels, reduction="mean")
    return float(loss.item())


class ECEMetric(Metric):
    """
    TorchMetrics-compatible ECE metric.
    """

    def __init__(self, n_bins: Optional[int] = None, pos_label: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.n_bins = n_bins
        self.pos_label = pos_label
        self.add_state("y_true", default=[], dist_reduce_fx="cat")
        self.add_state("y_prob", default=[], dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        Update state with predictions and targets.

        Args:
            preds: Predicted probabilities [B, num_classes] or [B]
            target: True labels [B]
        """
        if preds.ndim == 2:
            # Extract probability for positive class
            y_prob = preds[:, self.pos_label]
        else:
            y_prob = preds

        self.y_true.append(target)
        self.y_prob.append(y_prob)

    def compute(self) -> Tuple[torch.Tensor, int]:
        """Compute ECE."""
        y_true = torch.cat(self.y_true).cpu().numpy()
        y_prob = torch.cat(self.y_prob).cpu().numpy()
        ece_value, bins_used = compute_ece(y_true, y_prob, self.n_bins, self.pos_label)
        return torch.tensor(ece_value), bins_used


def get_step_metrics(
    num_classes: int = 2, average: str = "macro", sync_dist: bool = False
):
    """
    Get step-level metrics: Accuracy, AUROC, F1.

    Args:
        num_classes: Number of classes
        average: Averaging method for F1 ('macro', 'micro', 'weighted')
        sync_dist: Whether to sync metrics across distributed processes

    Returns:
        dict of metric_name -> Metric instance
    """
    return {
        "accuracy": Accuracy(
            task="binary" if num_classes == 2 else "multiclass",
            num_classes=num_classes,
            sync_on_compute=sync_dist,
        ),
        "auroc": AUROC(
            task="binary" if num_classes == 2 else "multiclass",
            num_classes=num_classes,
            sync_on_compute=sync_dist,
        ),
        "f1": F1Score(
            task="binary" if num_classes == 2 else "multiclass",
            num_classes=num_classes,
            average=average,
            sync_on_compute=sync_dist,
        ),
    }

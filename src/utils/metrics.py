"""
Metrics computation utilities including ECE (Expected Calibration Error).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy, AUROC, F1Score, Metric
from sklearn.metrics import roc_curve


def compute_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: Optional[int] = None,
    pos_label: int = 1,
) -> Tuple[float, int, bool]:
    """
    Compute Expected Calibration Error using fixed 15-bin partition.

    The first bin is left-closed to ensure probabilities equal to 0.0 are not skipped.

    Args:
        y_true: True labels (0 or 1).
        y_prob: Predicted probabilities for the positive class.
        n_bins: Number of bins (ignored; fixed to 15 to stabilize comparisons).
        pos_label: Positive class label (default=1).

    Returns:
        Tuple of (ece_value, bins_used, low_sample_warning)
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    N = len(y_true)
    low_sample_warning = N < 150
    n_bins = 15

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    for idx, (bin_lower, bin_upper) in enumerate(zip(bin_lowers, bin_uppers)):
        if idx == 0:
            in_bin = (y_prob >= bin_lower) & (y_prob <= bin_upper)
        else:
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            accuracy_in_bin = (y_true[in_bin] == pos_label).mean()
            avg_confidence_in_bin = y_prob[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return float(ece), n_bins, low_sample_warning


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
        ece_value, bins_used, _ = compute_ece(
            y_true, y_prob, self.n_bins, self.pos_label
        )
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


def compute_fpr_at_tpr95(
    y_true: np.ndarray, y_prob: np.ndarray
) -> Tuple[float, float, bool]:
    """
    Compute FPR when TPR reaches 95% using full ROC curve (drop_intermediate=False).

    Returns:
        fpr_at_95, threshold_at_95, reached_flag
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob, drop_intermediate=False)
    if not np.all(np.diff(tpr) >= 0):
        raise AssertionError("TPR should be non-decreasing")

    if tpr.max() < 0.95:
        idx = np.argmax(tpr)
        return float(fpr[idx]), float(thresholds[idx]), False

    fpr_95 = np.interp(0.95, tpr, fpr)
    thr_95 = np.interp(0.95, tpr, thresholds)
    return float(fpr_95), float(thr_95), True

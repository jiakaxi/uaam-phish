"""
Metrics computation utilities including ECE (Expected Calibration Error).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy, AUROC, F1Score, Metric
from sklearn.metrics import roc_curve


def _to_numpy(values: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy()
    return np.asarray(values)


def ece(
    probs: np.ndarray | torch.Tensor,
    targets: np.ndarray | torch.Tensor,
    n_bins: int = 15,
    pos_label: int = 1,
) -> Tuple[float, Dict[str, object]]:
    """
    Compute Expected Calibration Error with adaptive bin fallback.

    When the overall sample count is <150 or any bin has <20 samples, the number
    of bins is reduced to 10 and the reason flag is surfaced for downstream logs.
    """
    y_prob = _to_numpy(probs).reshape(-1)
    y_true = _to_numpy(targets).reshape(-1)
    total = y_true.shape[0]

    def _compute(bin_count: int) -> Tuple[float, np.ndarray]:
        boundaries = np.linspace(0.0, 1.0, bin_count + 1)
        ece_val = 0.0
        counts = np.zeros(bin_count, dtype=int)
        for idx in range(bin_count):
            lower, upper = boundaries[idx], boundaries[idx + 1]
            if idx == 0:
                mask = (y_prob >= lower) & (y_prob <= upper)
            else:
                mask = (y_prob > lower) & (y_prob <= upper)
            count = int(mask.sum())
            counts[idx] = count
            if count > 0:
                accuracy = (y_true[mask] == pos_label).mean()
                confidence = y_prob[mask].mean()
                ece_val += abs(confidence - accuracy) * (count / total)
        return float(ece_val), counts

    bins_used = n_bins
    ece_reason = None
    ece_value, counts = _compute(bins_used)
    min_count = counts.min() if counts.size else 0
    if total < 150 or (counts.size and min_count < 20):
        bins_used = 10
        ece_reason = "low_sample_bins"
        ece_value, counts = _compute(bins_used)
        min_count = counts.min() if counts.size else 0

    stats = {
        "bins_used": int(bins_used),
        "ece_reason": ece_reason,
        "n_samples": int(total),
        "min_bin_count": int(min_count),
        "bin_counts": counts.tolist(),
    }
    return ece_value, stats


def brier_score(
    probs: np.ndarray | torch.Tensor, targets: np.ndarray | torch.Tensor
) -> float:
    y_prob = _to_numpy(probs).reshape(-1)
    y_true = _to_numpy(targets).reshape(-1)
    return float(np.mean((y_prob - y_true) ** 2))


def fpr_at_tpr(
    probs: np.ndarray | torch.Tensor,
    targets: np.ndarray | torch.Tensor,
    tpr: float = 0.95,
) -> Tuple[float, float, bool]:
    y_prob = _to_numpy(probs).reshape(-1)
    y_true = _to_numpy(targets).reshape(-1)
    fpr, roc_tpr, thresholds = roc_curve(y_true, y_prob, drop_intermediate=False)
    if not np.all(np.diff(roc_tpr) >= 0):
        raise AssertionError("TPR should be non-decreasing")
    if roc_tpr.max() < tpr:
        idx = int(np.argmax(roc_tpr))
        return float(fpr[idx]), float(thresholds[idx]), False
    fpr_val = float(np.interp(tpr, roc_tpr, fpr))
    thr_val = float(np.interp(tpr, roc_tpr, thresholds))
    return fpr_val, thr_val, True


def compute_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: Optional[int] = None,
    pos_label: int = 1,
) -> Tuple[float, int, bool]:
    ece_value, stats = ece(y_prob, y_true, n_bins or 15, pos_label)
    low_sample_warning = stats.get("ece_reason") == "low_sample_bins"
    return ece_value, int(stats["bins_used"]), low_sample_warning


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
        ece_value, stats = ece(y_prob, y_true, self.n_bins or 15, self.pos_label)
        return torch.tensor(ece_value), stats["bins_used"]


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
    return fpr_at_tpr(y_prob, y_true, tpr=0.95)

"""
S0-specific metric helpers (calibration, Brier score, FPR@TPR95).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.metrics import roc_curve


def compute_fpr_at_tpr95(
    y_true: np.ndarray, y_prob: np.ndarray
) -> Tuple[float, float, bool]:
    fpr, tpr, thresholds = roc_curve(y_true, y_prob, drop_intermediate=False)
    if not np.all(np.diff(tpr) >= 0):
        raise AssertionError("TPR should be non-decreasing")
    if tpr.max() < 0.95:
        idx = np.argmax(tpr)
        return float(fpr[idx]), float(thresholds[idx]), False
    fpr_95 = np.interp(0.95, tpr, fpr)
    thr_95 = np.interp(0.95, tpr, thresholds)
    return float(fpr_95), float(thr_95), True


def compute_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    return float(np.mean((y_prob - y_true) ** 2))


def compute_ece(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15, pos_label: int = 1
) -> Tuple[float, int, bool]:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    n_bins = 15
    low_sample_warning = len(y_true) < 150
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    for i, (lo, hi) in enumerate(zip(bin_lowers, bin_uppers)):
        if i == 0:
            in_bin = (y_prob >= lo) & (y_prob <= hi)
        else:
            in_bin = (y_prob > lo) & (y_prob <= hi)
        prop = in_bin.mean()
        if prop > 0:
            acc = (y_true[in_bin] == pos_label).mean()
            conf = y_prob[in_bin].mean()
            ece += abs(conf - acc) * prop
    return float(ece), n_bins, low_sample_warning

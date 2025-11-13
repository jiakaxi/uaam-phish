#!/usr/bin/env python
"""
Generate S0/S2 distribution plots and consistency metrics for thesis figures.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot S0 visual similarity and S2 consistency distributions."
    )
    parser.add_argument(
        "--runs_dir",
        type=Path,
        default=Path("workspace/runs"),
        help="Root directory that contains experiment runs.",
    )
    parser.add_argument(
        "--s0",
        type=Path,
        default=None,
        help="Explicit path to baseline (S0) run directory (seed_xxx). "
        "If omitted, the latest folder matching --s0-pattern is used.",
    )
    parser.add_argument(
        "--s2",
        type=Path,
        default=None,
        help="Explicit path to S2 run directory (seed_xxx). "
        "If omitted, the latest folder matching --s2-pattern is used.",
    )
    parser.add_argument(
        "--s0-pattern",
        type=str,
        default="s0_*",
        help="Glob pattern (under runs_dir) to locate S0 run when --s0 is not provided.",
    )
    parser.add_argument(
        "--s2-pattern",
        type=str,
        default="s2_*",
        help="Glob pattern (under runs_dir) to locate S2 run when --s2 is not provided.",
    )
    parser.add_argument(
        "--s0-column",
        type=str,
        default="prob",
        help="Column to treat as 'visual similarity' for the S0 histogram.",
    )
    parser.add_argument(
        "--thresh",
        type=float,
        default=0.60,
        help="Consistency threshold τ_s for mismatch analysis.",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("figures"),
        help="Directory to save generated figures.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory to save the consistency_report.json file.",
    )
    return parser.parse_args()


def find_latest_run(runs_dir: Path, pattern: str) -> Path:
    candidates = []
    for root in runs_dir.glob(pattern):
        if not root.is_dir():
            continue
        for seed_dir in root.glob("seed_*"):
            if seed_dir.is_dir():
                candidates.append(seed_dir)
    if not candidates:
        raise FileNotFoundError(f"No runs found under {runs_dir} matching {pattern}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def resolve_run_path(runs_dir: Path, explicit: Optional[Path], pattern: str) -> Path:
    if explicit:
        path = explicit if explicit.is_absolute() else (runs_dir / explicit)
        if not path.exists():
            raise FileNotFoundError(f"Run directory {path} does not exist")
        return path
    return find_latest_run(runs_dir, pattern)


def load_predictions(run_dir: Path) -> pd.DataFrame:
    csv_path = run_dir / "artifacts" / "predictions_test.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing predictions_test.csv at {csv_path}")
    df = pd.read_csv(csv_path)
    if "y_true" not in df.columns:
        raise ValueError(f"'y_true' column missing from {csv_path}")
    return df


def compute_overlap(a: np.ndarray, b: np.ndarray, bins: int = 40) -> float:
    lo = float(min(a.min(), b.min()))
    hi = float(max(a.max(), b.max()))
    if hi == lo:
        return 1.0
    hist_a, edges = np.histogram(a, bins=bins, range=(lo, hi), density=True)
    hist_b, _ = np.histogram(b, bins=bins, range=(lo, hi), density=True)
    width = edges[1] - edges[0]
    return float(np.clip(np.minimum(hist_a, hist_b).sum() * width, 0.0, 1.0))


def compute_ks(a: np.ndarray, b: np.ndarray) -> float:
    a_sorted = np.sort(a)
    b_sorted = np.sort(b)
    data = np.sort(np.concatenate([a_sorted, b_sorted]))
    cdf_a = np.searchsorted(a_sorted, data, side="right") / a_sorted.size
    cdf_b = np.searchsorted(b_sorted, data, side="right") / b_sorted.size
    return float(np.max(np.abs(cdf_a - cdf_b)))


def mean_ci(values: np.ndarray) -> Tuple[float, Tuple[float, float]]:
    if values.size == 0:
        return float("nan"), (float("nan"), float("nan"))
    mean_val = float(values.mean())
    if values.size == 1:
        return mean_val, (mean_val, mean_val)
    std = float(values.std(ddof=1))
    margin = 1.96 * std / np.sqrt(values.size)
    return mean_val, (mean_val - margin, mean_val + margin)


def plot_histogram(
    values_a: np.ndarray,
    values_b: np.ndarray,
    labels: Tuple[str, str],
    title: str,
    xlabel: str,
    outfile: Path,
    thresh: Optional[float] = None,
    annotations: Optional[Dict[str, str]] = None,
) -> None:
    bins = min(60, max(10, max(values_a.size, values_b.size) // 4))
    plt.figure(figsize=(8, 5))
    plt.hist(
        values_a,
        bins=bins,
        alpha=0.6,
        label=labels[0],
        density=True,
        color="#4c72b0",
    )
    plt.hist(
        values_b,
        bins=bins,
        alpha=0.6,
        label=labels[1],
        density=True,
        color="#dd8452",
    )
    if thresh is not None:
        plt.axvline(
            thresh, color="black", linestyle="--", linewidth=1.2, label=r"$\tau_s$"
        )
    if annotations:
        for idx, (key, text) in enumerate(annotations.items()):
            plt.text(
                0.02,
                0.90 - idx * 0.08,
                f"{key}: {text}",
                transform=plt.gca().transAxes,
                fontsize=10,
            )
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Density")
    plt.legend()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    plt.close()


def summarize_distribution(df: pd.DataFrame, column: str) -> Dict[str, Any]:
    scores = df[column].to_numpy(dtype=np.float32)
    mask = np.isfinite(scores)
    if not mask.any():
        raise ValueError(f"No finite values found in column '{column}'")
    scores = scores[mask]  # Filter scores to match y
    y = df.loc[mask, "y_true"].to_numpy()
    legit = scores[(y == 0)]
    phish = scores[(y == 1)]
    if legit.size == 0 or phish.size == 0:
        raise ValueError("Both classes must have at least one sample for analysis.")
    bins = min(60, max(10, max(legit.size, phish.size) // 4))
    metrics = {
        "ovl": compute_overlap(legit, phish, bins),
        "ks": compute_ks(legit, phish),
        "auc": float(roc_auc_score(y, scores)),
    }
    mean_legit, ci_legit = mean_ci(legit)
    mean_phish, ci_phish = mean_ci(phish)
    metrics.update(
        {
            "mean_legit": mean_legit,
            "ci_legit": list(ci_legit),
            "mean_phish": mean_phish,
            "ci_phish": list(ci_phish),
        }
    )
    return metrics


def main() -> None:
    args = parse_args()
    s0_dir = resolve_run_path(args.runs_dir, args.s0, args.s0_pattern)
    s2_dir = resolve_run_path(args.runs_dir, args.s2, args.s2_pattern)

    df_s0 = load_predictions(s0_dir)
    if args.s0_column not in df_s0.columns:
        raise ValueError(
            f"Column '{args.s0_column}' not found in {s0_dir}/artifacts/predictions_test.csv"
        )
    df_s2 = load_predictions(s2_dir)
    if "c_mean" not in df_s2.columns:
        raise ValueError(
            f"Column 'c_mean' missing from {s2_dir}/artifacts/predictions_test.csv"
        )

    metrics_s0 = summarize_distribution(df_s0, args.s0_column)
    metrics_s2 = summarize_distribution(df_s2, "c_mean")
    thresh = args.thresh
    scores_s2 = df_s2["c_mean"].to_numpy(dtype=np.float32)
    valid_mask = np.isfinite(scores_s2)
    scores_s2 = scores_s2[valid_mask]
    labels_s2 = df_s2.loc[valid_mask, "y_true"].to_numpy()
    mr_overall = float(np.mean(scores_s2 < thresh))
    mr_phish = float(np.mean(scores_s2[labels_s2 == 1] < thresh))
    fpr_legit = float(np.mean(scores_s2[labels_s2 == 0] < thresh))
    metrics_s2.update(
        {
            "thresh": thresh,
            "acs": float(np.nanmean(scores_s2)),
            "mr_overall": mr_overall,
            "mr_phish": mr_phish,
            "fpr_legit": fpr_legit,
        }
    )

    # Plot S0 histogram (visual similarity proxy)
    values_s0 = df_s0[args.s0_column].to_numpy(dtype=np.float32)
    mask_s0 = np.isfinite(values_s0)
    plot_histogram(
        values_s0[(df_s0["y_true"].to_numpy() == 0) & mask_s0],
        values_s0[(df_s0["y_true"].to_numpy() == 1) & mask_s0],
        labels=("Legitimate", "Phishing"),
        title="S0: Visual similarity distribution",
        xlabel=args.s0_column,
        outfile=args.figures_dir / "s0_vis_similarity_hist.png",
    )

    # Plot S2 histogram
    plot_histogram(
        scores_s2[labels_s2 == 0],
        scores_s2[labels_s2 == 1],
        labels=("Legitimate", "Phishing"),
        title="S2: Cross-modal consistency (c_mean)",
        xlabel="c_mean",
        outfile=args.figures_dir / "s2_consistency_hist.png",
        thresh=thresh,
        annotations={
            "ACS": f"{metrics_s2['acs']:.3f}",
            "MR (all)": f"{mr_overall:.3f}",
            "MR (phish)": f"{mr_phish:.3f}",
            "FPR (legit)": f"{fpr_legit:.3f}",
        },
    )

    report = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "runs": {
            "s0": {
                "path": str(s0_dir),
                "column": args.s0_column,
                **metrics_s0,
            },
            "s2": {
                "path": str(s2_dir),
                "column": "c_mean",
                **metrics_s2,
            },
        },
        "meta": {
            "thresh": thresh,
            "figures": {
                "s0": str(args.figures_dir / "s0_vis_similarity_hist.png"),
                "s2": str(args.figures_dir / "s2_consistency_hist.png"),
            },
        },
    }
    report["baseline"] = report["runs"]["s0"]
    report["current"] = report["runs"]["s2"]

    args.results_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.results_dir / "consistency_report.json"
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"[INFO] Saved report to {report_path}")
    print(f"[INFO] Figures saved to {args.figures_dir.resolve()}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
S3 Final Visualization Script
Generates visualizations for the two fixed S3 experiments (seed=100).
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

log = logging.getLogger(__name__)


def setup_plotting():
    """Set up matplotlib/seaborn style."""
    sns.set_context("paper", font_scale=1.3)
    sns.set_style("whitegrid")
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["font.family"] = "sans-serif"


def load_s3_experiment(exp_dir: Path) -> Dict:
    """Load S3 experiment data including predictions and summary."""
    result = {
        "name": exp_dir.name,
        "protocol": "iid" if "iid" in exp_dir.name.lower() else "brandood",
    }

    # Load predictions
    pred_file = exp_dir / "results" / "predictions_test.csv"
    if pred_file.exists():
        result["predictions"] = pd.read_csv(pred_file)

    # Load summary
    summary_file = exp_dir / "results" / "eval_summary.json"
    if summary_file.exists():
        with open(summary_file) as f:
            result["summary"] = json.load(f)

    return result


def plot_alpha_distribution(iid_data: Dict, brandood_data: Dict, output_dir: Path):
    """Plot alpha weight distributions for IID vs Brand-OOD."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    modalities = ["url", "html", "visual"]

    for idx, mod in enumerate(modalities):
        ax = axes[idx]

        iid_alpha = iid_data["predictions"][f"alpha_{mod}"].dropna()
        brandood_alpha = brandood_data["predictions"][f"alpha_{mod}"].dropna()

        # Plot violin plots
        data_to_plot = pd.DataFrame({"IID": iid_alpha, "Brand-OOD": brandood_alpha})

        sns.violinplot(data=data_to_plot, ax=ax, palette="Set2")
        ax.set_title(f"{mod.capitalize()} Modality")
        ax.set_ylabel("Alpha Weight" if idx == 0 else "")
        ax.set_ylim(0, 1)
        ax.axhline(y=1 / 3, color="r", linestyle="--", alpha=0.5, label="Uniform (1/3)")

        if idx == 0:
            ax.legend()

    plt.tight_layout()
    output_path = output_dir / "s3_alpha_distribution.png"
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    log.info(f"Saved alpha distribution plot: {output_path}")


def plot_performance_comparison(iid_data: Dict, brandood_data: Dict, output_dir: Path):
    """Plot performance metrics comparison between IID and Brand-OOD."""
    metrics = ["auroc", "f1_macro", "ece", "accuracy"]
    metric_labels = ["AUROC", "F1 (macro)", "ECE", "Accuracy"]

    # Extract S3 metrics
    iid_s3 = iid_data["summary"].get("s3", {})
    brandood_s3 = brandood_data["summary"].get("s3", {})

    iid_values = [iid_s3.get(m, 0) for m in metrics]
    brandood_values = [brandood_s3.get(m, 0) for m in metrics]

    # Plot
    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, iid_values, width, label="IID", color="steelblue")
    bars2 = ax.bar(
        x + width / 2, brandood_values, width, label="Brand-OOD", color="coral"
    )

    ax.set_xlabel("Metrics")
    ax.set_ylabel("Score")
    ax.set_title("S3 Performance: IID vs Brand-OOD")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.tight_layout()
    output_path = output_dir / "s3_performance_comparison.png"
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    log.info(f"Saved performance comparison plot: {output_path}")


def plot_alpha_stats(iid_data: Dict, brandood_data: Dict, output_dir: Path):
    """Plot alpha statistics (mean and std) for each modality."""
    modalities = ["url", "html", "visual"]

    iid_stats = iid_data["summary"].get("s3", {}).get("alpha_stats", {})
    brandood_stats = brandood_data["summary"].get("s3", {}).get("alpha_stats", {})

    iid_means = [iid_stats.get(f"alpha_{m}_mean", 0) for m in modalities]
    iid_stds = [iid_stats.get(f"alpha_{m}_std", 0) for m in modalities]

    brandood_means = [brandood_stats.get(f"alpha_{m}_mean", 0) for m in modalities]
    brandood_stds = [brandood_stats.get(f"alpha_{m}_std", 0) for m in modalities]

    x = np.arange(len(modalities))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(
        x - width / 2,
        iid_means,
        width,
        yerr=iid_stds,
        label="IID",
        color="steelblue",
        alpha=0.8,
        capsize=5,
    )
    ax.bar(
        x + width / 2,
        brandood_means,
        width,
        yerr=brandood_stds,
        label="Brand-OOD",
        color="coral",
        alpha=0.8,
        capsize=5,
    )

    ax.set_xlabel("Modality")
    ax.set_ylabel("Alpha Weight (mean ± std)")
    ax.set_title("S3 Alpha Statistics: IID vs Brand-OOD")
    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in modalities])
    ax.legend()
    ax.axhline(y=1 / 3, color="r", linestyle="--", alpha=0.5, label="Uniform (1/3)")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / "s3_alpha_stats.png"
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    log.info(f"Saved alpha stats plot: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate S3 final visualizations")
    parser.add_argument(
        "--iid-exp",
        type=str,
        default="experiments/s3_iid_fixed_*_seed100",
        help="IID experiment directory pattern",
    )
    parser.add_argument(
        "--brandood-exp",
        type=str,
        default="experiments/s3_brandood_fixed_*_seed100",
        help="Brand-OOD experiment directory pattern",
    )
    parser.add_argument(
        "--output-dir", type=str, default="figures", help="Output directory for plots"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    setup_plotting()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Find experiment directories
    from glob import glob

    iid_dirs = sorted(glob(args.iid_exp))
    brandood_dirs = sorted(glob(args.brandood_exp))

    if not iid_dirs:
        log.error(f"No IID experiment found matching: {args.iid_exp}")
        return
    if not brandood_dirs:
        log.error(f"No Brand-OOD experiment found matching: {args.brandood_exp}")
        return

    iid_dir = Path(iid_dirs[-1])  # Use latest
    brandood_dir = Path(brandood_dirs[-1])  # Use latest

    log.info(f"Loading IID experiment: {iid_dir}")
    iid_data = load_s3_experiment(iid_dir)

    log.info(f"Loading Brand-OOD experiment: {brandood_dir}")
    brandood_data = load_s3_experiment(brandood_dir)

    # Check if predictions are available
    if "predictions" not in iid_data or "predictions" not in brandood_data:
        log.error("Missing predictions data. Experiments may still be running.")
        return

    # Check if alpha columns exist
    alpha_cols = [f"alpha_{m}" for m in ["url", "html", "visual"]]
    iid_has_alpha = all(col in iid_data["predictions"].columns for col in alpha_cols)
    brandood_has_alpha = all(
        col in brandood_data["predictions"].columns for col in alpha_cols
    )

    if not iid_has_alpha or not brandood_has_alpha:
        log.warning(
            "Alpha columns missing. Fixed fusion may not have executed correctly."
        )

    # Generate plots
    log.info("Generating alpha distribution plot...")
    plot_alpha_distribution(iid_data, brandood_data, output_dir)

    log.info("Generating performance comparison plot...")
    plot_performance_comparison(iid_data, brandood_data, output_dir)

    log.info("Generating alpha statistics plot...")
    plot_alpha_stats(iid_data, brandood_data, output_dir)

    log.info(f"All visualizations saved to: {output_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
Generate visualizations for S3 (fixed fusion) analysis.

Creates three types of plots:
1. Alpha distribution analysis (violin/box plots) - IID vs Brand-OOD
2. Performance comparison bar charts (S0/S1/S2/S3)
3. Corruption robustness line plots (if corruption experiments exist)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.logging import get_logger

log = get_logger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 150


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize S3 fixed fusion results")
    parser.add_argument(
        "--experiments-dir",
        type=Path,
        default=Path("experiments"),
        help="Root directory containing experiment results",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures"),
        help="Output directory for figures",
    )
    parser.add_argument(
        "--baselines",
        type=Path,
        default=Path("experiments/synergy_baselines.json"),
        help="Synergy baselines JSON file",
    )
    return parser.parse_args()


def load_predictions(exp_dir: Path, stage: str = "test") -> Optional[pd.DataFrame]:
    """Load predictions CSV from experiment directory."""
    pred_file = exp_dir / "artifacts" / f"predictions_{stage}.csv"

    if not pred_file.exists():
        log.warning(f"Predictions file not found: {pred_file}")
        return None

    try:
        return pd.read_csv(pred_file)
    except Exception as exc:
        log.warning(f"Failed to load {pred_file}: {exc}")
        return None


def load_metrics(exp_dir: Path, stage: str = "test") -> Optional[Dict[str, float]]:
    """Load metrics JSON from experiment directory."""
    metrics_file = exp_dir / "artifacts" / f"metrics_{stage}.json"

    if not metrics_file.exists():
        return None

    try:
        with open(metrics_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        log.warning(f"Failed to load {metrics_file}: {exc}")
        return None


def plot_alpha_distribution(
    experiments_dir: Path,
    output_dir: Path,
) -> None:
    """Plot alpha distribution across IID and Brand-OOD protocols."""
    log.info("Generating alpha distribution plots...")

    data_records = []

    for protocol in ["iid", "brandood"]:
        pattern = f"s3_{protocol}_fixed_"
        exp_dirs = list(experiments_dir.glob(f"{pattern}*"))

        if not exp_dirs:
            log.warning(f"No S3 {protocol} experiments found")
            continue

        log.info(f"Found {len(exp_dirs)} S3 {protocol} experiments")

        for exp_dir in exp_dirs:
            if not exp_dir.is_dir():
                continue

            df = load_predictions(exp_dir)
            if df is None:
                continue

            # Check for alpha columns
            alpha_cols = ["alpha_url", "alpha_html", "alpha_img"]
            if not all(col in df.columns for col in alpha_cols):
                log.warning(f"Missing alpha columns in {exp_dir.name}")
                continue

            # Collect alpha values
            for col in alpha_cols:
                modality = col.replace("alpha_", "")
                for value in df[col].dropna():
                    data_records.append(
                        {
                            "protocol": "IID" if protocol == "iid" else "Brand-OOD",
                            "modality": (
                                modality.upper() if modality != "img" else "Visual"
                            ),
                            "alpha": float(value),
                            "experiment": exp_dir.name,
                        }
                    )

    if not data_records:
        log.warning("No alpha data collected. Skipping alpha distribution plot.")
        return

    alpha_df = pd.DataFrame(data_records)

    # Create violin plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Grouped by protocol
    sns.violinplot(
        data=alpha_df,
        x="modality",
        y="alpha",
        hue="protocol",
        split=False,
        ax=axes[0],
        palette="Set2",
    )
    axes[0].set_title(
        "Fusion Weight Distribution by Protocol", fontsize=14, fontweight="bold"
    )
    axes[0].set_xlabel("Modality", fontsize=12)
    axes[0].set_ylabel("Fusion Weight (α)", fontsize=12)
    axes[0].legend(title="Protocol", fontsize=10)
    axes[0].axhline(
        y=1 / 3, color="red", linestyle="--", alpha=0.5, label="Uniform (1/3)"
    )
    axes[0].grid(alpha=0.3)

    # Plot 2: Box plot with individual protocols
    sns.boxplot(
        data=alpha_df,
        x="protocol",
        y="alpha",
        hue="modality",
        ax=axes[1],
        palette="Set1",
    )
    axes[1].set_title(
        "Fusion Weight Distribution by Modality", fontsize=14, fontweight="bold"
    )
    axes[1].set_xlabel("Protocol", fontsize=12)
    axes[1].set_ylabel("Fusion Weight (α)", fontsize=12)
    axes[1].legend(title="Modality", fontsize=10)
    axes[1].axhline(y=1 / 3, color="red", linestyle="--", alpha=0.5)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / "s3_alpha_distribution.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    log.info(f"✓ Saved alpha distribution plot: {output_path}")

    # Print statistics
    print("\n" + "=" * 70)
    print("ALPHA DISTRIBUTION STATISTICS")
    print("=" * 70)
    grouped = alpha_df.groupby(["protocol", "modality"])["alpha"].agg(
        ["mean", "std", "count"]
    )
    print(grouped)
    print("=" * 70)


def plot_performance_comparison(
    experiments_dir: Path,
    baselines_path: Path,
    output_dir: Path,
) -> None:
    """Plot S0/S1/S2/S3 performance comparison."""
    log.info("Generating performance comparison plots...")

    # Load baselines
    if not baselines_path.exists():
        log.warning(
            f"Baselines file not found: {baselines_path}. Run collect_synergy_baselines.py first."
        )
        return

    with open(baselines_path, "r", encoding="utf-8") as f:
        baseline_data = json.load(f)

    baselines = baseline_data.get("baselines", baseline_data)

    # Collect S0 and S3 metrics
    all_metrics = {}

    # S0 metrics
    for protocol in ["iid", "brandood"]:
        pattern = f"s0_{protocol}_lateavg_"
        exp_dirs = list(experiments_dir.glob(f"{pattern}*"))

        if exp_dirs:
            metrics_list = []
            for exp_dir in exp_dirs:
                if not exp_dir.is_dir():
                    continue
                metrics = load_metrics(exp_dir)
                if metrics:
                    metrics_list.append(metrics)

            if metrics_list:
                aggregated = {}
                for key in ["auroc", "f1_macro", "ece", "brier"]:
                    values = [
                        m[key] for m in metrics_list if key in m and np.isfinite(m[key])
                    ]
                    if values:
                        aggregated[key] = np.mean(values)
                        aggregated[f"{key}_std"] = (
                            np.std(values, ddof=1) if len(values) > 1 else 0.0
                        )
                all_metrics[f"s0_{protocol}"] = aggregated

    # S3 metrics
    for protocol in ["iid", "brandood"]:
        pattern = f"s3_{protocol}_fixed_"
        exp_dirs = list(experiments_dir.glob(f"{pattern}*"))

        if exp_dirs:
            metrics_list = []
            for exp_dir in exp_dirs:
                if not exp_dir.is_dir():
                    continue

                # Try to load from eval_summary.json (has S3 specific metrics)
                eval_file = exp_dir / "eval_summary.json"
                if eval_file.exists():
                    try:
                        with open(eval_file, "r", encoding="utf-8") as f:
                            eval_data = json.load(f)
                        if "s3" in eval_data:
                            metrics_list.append(eval_data["s3"])
                    except Exception:
                        pass

                # Fallback to regular metrics
                if not metrics_list:
                    metrics = load_metrics(exp_dir)
                    if metrics:
                        metrics_list.append(metrics)

            if metrics_list:
                aggregated = {}
                for key in ["auroc", "f1_macro", "ece", "brier"]:
                    values = [
                        m[key] for m in metrics_list if key in m and np.isfinite(m[key])
                    ]
                    if values:
                        aggregated[key] = np.mean(values)
                        aggregated[f"{key}_std"] = (
                            np.std(values, ddof=1) if len(values) > 1 else 0.0
                        )
                all_metrics[f"s3_{protocol}"] = aggregated

    # Merge with baselines
    all_metrics.update(baselines)

    if not all_metrics:
        log.warning("No metrics collected. Skipping performance comparison.")
        return

    # Create comparison plots
    metrics_to_plot = ["auroc", "f1_macro", "ece", "brier"]
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()

    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx]

        # Prepare data for plotting
        plot_data = []
        for protocol in ["iid", "brandood"]:
            for system in ["s0", "s1", "s2", "s3"]:
                key = f"{system}_{protocol}"
                if key in all_metrics:
                    metrics = all_metrics[key]
                    if metric in metrics:
                        plot_data.append(
                            {
                                "System": system.upper(),
                                "Protocol": "IID" if protocol == "iid" else "Brand-OOD",
                                "Value": metrics[metric],
                                "Std": metrics.get(f"{metric}_std", 0.0),
                            }
                        )

        if not plot_data:
            continue

        plot_df = pd.DataFrame(plot_data)

        # Create grouped bar chart
        x_positions = np.arange(len(plot_df["System"].unique()))
        width = 0.35

        iid_data = plot_df[plot_df["Protocol"] == "IID"]
        ood_data = plot_df[plot_df["Protocol"] == "Brand-OOD"]

        if not iid_data.empty:
            ax.bar(
                x_positions - width / 2,
                iid_data["Value"],
                width,
                yerr=iid_data["Std"],
                label="IID",
                alpha=0.8,
                capsize=5,
            )

        if not ood_data.empty:
            ax.bar(
                x_positions + width / 2,
                ood_data["Value"],
                width,
                yerr=ood_data["Std"],
                label="Brand-OOD",
                alpha=0.8,
                capsize=5,
            )

        ax.set_xlabel("System", fontsize=11)
        ax.set_ylabel(metric.upper().replace("_", " "), fontsize=11)
        ax.set_title(
            f"{metric.upper().replace('_', ' ')} Comparison",
            fontsize=12,
            fontweight="bold",
        )
        ax.set_xticks(x_positions)
        ax.set_xticklabels(plot_df["System"].unique())
        ax.legend()
        ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    output_path = output_dir / "s3_performance_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    log.info(f"✓ Saved performance comparison plot: {output_path}")


def plot_corruption_robustness(
    experiments_dir: Path,
    output_dir: Path,
) -> None:
    """Plot corruption robustness analysis (if corrupt experiments exist)."""
    log.info("Checking for corruption robustness data...")

    # Look for corrupt experiments
    corrupt_patterns = ["corrupt_url_", "corrupt_html_", "corrupt_img_"]

    data_records = []

    for pattern in corrupt_patterns:
        for level in ["L", "M", "H"]:
            exp_pattern = f"{pattern}{level}_"
            exp_dirs = list(experiments_dir.glob(f"{exp_pattern}*"))

            if not exp_dirs:
                continue

            modality = pattern.replace("corrupt_", "").replace("_", "").upper()

            for exp_dir in exp_dirs:
                if not exp_dir.is_dir():
                    continue

                metrics = load_metrics(exp_dir)
                if metrics:
                    data_records.append(
                        {
                            "modality": modality,
                            "level": level,
                            "auroc": metrics.get("auroc", np.nan),
                            "ece": metrics.get("ece", np.nan),
                            "experiment": exp_dir.name,
                        }
                    )

    if not data_records:
        log.info(
            "No corruption experiments found. Skipping corruption robustness plot."
        )
        return

    corrupt_df = pd.DataFrame(data_records)

    # Create line plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot AUROC
    for modality in corrupt_df["modality"].unique():
        data = corrupt_df[corrupt_df["modality"] == modality]
        grouped = data.groupby("level")["auroc"].agg(["mean", "std"])
        levels = ["L", "M", "H"]
        axes[0].plot(
            levels, grouped.loc[levels, "mean"], marker="o", label=modality, linewidth=2
        )
        axes[0].fill_between(
            range(len(levels)),
            grouped.loc[levels, "mean"] - grouped.loc[levels, "std"],
            grouped.loc[levels, "mean"] + grouped.loc[levels, "std"],
            alpha=0.2,
        )

    axes[0].set_title("AUROC vs Corruption Level", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Corruption Level", fontsize=12)
    axes[0].set_ylabel("AUROC", fontsize=12)
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Plot ECE
    for modality in corrupt_df["modality"].unique():
        data = corrupt_df[corrupt_df["modality"] == modality]
        grouped = data.groupby("level")["ece"].agg(["mean", "std"])
        levels = ["L", "M", "H"]
        axes[1].plot(
            levels, grouped.loc[levels, "mean"], marker="o", label=modality, linewidth=2
        )
        axes[1].fill_between(
            range(len(levels)),
            grouped.loc[levels, "mean"] - grouped.loc[levels, "std"],
            grouped.loc[levels, "mean"] + grouped.loc[levels, "std"],
            alpha=0.2,
        )

    axes[1].set_title("ECE vs Corruption Level", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Corruption Level", fontsize=12)
    axes[1].set_ylabel("ECE", fontsize=12)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / "s3_corruption_robustness.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    log.info(f"✓ Saved corruption robustness plot: {output_path}")


def main() -> None:
    args = parse_args()

    log.info("=" * 70)
    log.info("S3 Visualization Generator")
    log.info("=" * 70)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Generate visualizations
    plot_alpha_distribution(args.experiments_dir, args.output_dir)
    plot_performance_comparison(args.experiments_dir, args.baselines, args.output_dir)
    plot_corruption_robustness(args.experiments_dir, args.output_dir)

    log.info("=" * 70)
    log.info(f"✓ All visualizations saved to: {args.output_dir}")
    log.info("=" * 70)


if __name__ == "__main__":
    main()

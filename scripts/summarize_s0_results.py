#!/usr/bin/env python
"""
Generate summary table with mean ± std for S0 experiments.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize S0 results with statistics."
    )
    parser.add_argument(
        "--eval_csv",
        default="workspace/tables/s0_eval_summary.csv",
        help="Path to evaluation CSV file.",
    )
    parser.add_argument(
        "--out_table",
        default="workspace/tables/s0_results_summary.md",
        help="Path to output summary table.",
    )
    return parser.parse_args()


def calculate_statistics(df: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
    """Calculate mean ± std for each model and metric."""
    results = []

    for model in df["model"].unique():
        model_data = df[df["model"] == model]
        row = {"Model": model}

        for metric in metrics:
            values = model_data[metric].values
            mean_val = np.mean(values)
            std_val = np.std(values)

            # Format as mean ± std
            if metric in ["accuracy", "f1", "auroc", "brier", "ece"]:
                # Format with appropriate precision
                if metric in ["accuracy", "f1", "auroc"]:
                    row[metric] = f"{mean_val:.4f} ± {std_val:.4f}"
                else:
                    row[metric] = f"{mean_val:.4f} ± {std_val:.4f}"
            elif metric == "fpr_at_tpr95":
                row[metric] = f"{mean_val:.4f} ± {std_val:.4f}"

        results.append(row)

    return pd.DataFrame(results)


def main() -> None:
    args = parse_args()

    # Read evaluation results
    df = pd.read_csv(args.eval_csv)

    # Define metrics to include in the table
    metrics = ["accuracy", "f1", "auroc", "brier", "ece", "fpr_at_tpr95"]

    # Calculate statistics
    summary_df = calculate_statistics(df, metrics)

    # Create markdown table
    markdown_table = "# S0 Model Performance Summary (IID/Test)\n\n"
    markdown_table += "**Metrics: Mean ± Standard Deviation (3 seeds)**\n\n"

    # Create table header
    headers = ["Model", "Accuracy", "F1", "AUROC", "Brier", "ECE", "FPR@95"]
    markdown_table += "| " + " | ".join(headers) + " |\n"
    markdown_table += "|-" + "-|-" * (len(headers) - 1) + "-|\n"

    # Add rows
    for _, row in summary_df.iterrows():
        table_row = [
            row["Model"],
            row["accuracy"],
            row["f1"],
            row["auroc"],
            row["brier"],
            row["ece"],
            row["fpr_at_tpr95"],
        ]
        markdown_table += "| " + " | ".join(table_row) + " |\n"

    # Add sample sizes
    markdown_table += "\n**Sample size per model: 3 seeds**\n"
    markdown_table += "**Total test samples: 2400**\n"

    # Write to file
    out_path = Path(args.out_table)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(markdown_table)

    print(f"[summarize_s0_results] Summary table saved to {out_path}")

    # Also print to console
    print("\n" + "=" * 80)
    print("S0 MODEL PERFORMANCE SUMMARY")
    print("=" * 80)
    print(markdown_table)


if __name__ == "__main__":
    main()

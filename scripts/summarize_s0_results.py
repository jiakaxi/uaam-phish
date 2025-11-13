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
    """Calculate mean ± 95% CI for each model and metric."""
    results = []

    for model in sorted(df["model"].unique()):
        model_data = df[df["model"] == model]
        row = {"Model": model, "n": len(model_data)}

        for metric in metrics:
            if metric not in model_data:
                row[metric] = "N/A"
                continue
            values = model_data[metric].dropna().to_numpy()
            if values.size == 0:
                row[metric] = "N/A"
                continue
            mean_val = np.mean(values)
            if values.size > 1:
                std_val = np.std(values, ddof=1)
                ci = 1.96 * std_val / np.sqrt(values.size)
            else:
                ci = 0.0
            row[metric] = f"{mean_val:.4f} ± {ci:.4f}"

        results.append(row)

    return pd.DataFrame(results)


def main() -> None:
    args = parse_args()

    # Read evaluation results
    df = pd.read_csv(args.eval_csv)

    metrics = ["auroc", "ece_pre_fused", "ece_post_fused", "brier_post_fused"]

    # Calculate statistics
    summary_df = calculate_statistics(df, metrics)

    # Create markdown table
    markdown_table = "# S0 / S1 Model Performance Summary\n\n"
    markdown_table += "**Metrics: Mean ± 95% CI (per model)**\n\n"

    # Create table header
    headers = ["Model", "AUROC", "ECE_pre", "ECE_post", "Brier_post"]
    markdown_table += "| " + " | ".join(headers) + " |\n"
    markdown_table += "|-" + "-|-" * (len(headers) - 1) + "-|\n"

    # Add rows
    for _, row in summary_df.iterrows():
        table_row = [
            row["Model"],
            row["auroc"],
            row["ece_pre_fused"],
            row["ece_post_fused"],
            row["brier_post_fused"],
        ]
        markdown_table += "| " + " | ".join(table_row) + " |\n"

    markdown_table += "\n**Entries per model:** "
    markdown_table += ", ".join(
        f"{row['Model']} (n={row['n']})" for _, row in summary_df.iterrows()
    )
    markdown_table += "\n"

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

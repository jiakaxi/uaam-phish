#!/usr/bin/env python
"""
Summarize S0 evaluation results across seeds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize S0 results.")
    parser.add_argument(
        "--runs_dir", default="workspace/runs", help="Root directory containing runs."
    )
    parser.add_argument(
        "--out_tables",
        default="workspace/tables",
        help="Directory to store summary tables.",
    )
    parser.add_argument(
        "--out_figs",
        default="workspace/figs",
        help="Directory to store plots.",
    )
    return parser.parse_args()


def collect_eval_records(runs_root: Path) -> List[Dict[str, object]]:
    records = []
    for model_dir in runs_root.glob("*"):
        if not model_dir.is_dir():
            continue
        for run_dir in model_dir.glob("seed_*"):
            summary_path = run_dir / "eval_summary.json"
            if not summary_path.exists():
                continue
            with open(summary_path, "r", encoding="utf-8") as f:
                record = json.load(f)
            records.append(record)
    return records


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "accuracy",
        "f1",
        "auroc",
        "brier",
        "ece",
        "fpr_at_tpr95",
    ]
    grouped = (
        df.groupby("model")[metrics]
        .agg(["mean", "std"])
        .reset_index()
        .sort_values(("auroc", "mean"), ascending=False)
    )
    grouped.columns = ["_".join(col).strip("_") for col in grouped.columns.values]
    return grouped


def plot_auroc(df: pd.DataFrame, out_path: Path) -> None:
    models = df["model"].unique()
    means = df.groupby("model")["auroc"].mean().loc[models]
    stds = df.groupby("model")["auroc"].std().loc[models].fillna(0.0)

    plt.figure(figsize=(6, 4))
    bars = plt.bar(models, means, yerr=stds, capsize=5)
    plt.ylim(0.0, 1.0)
    plt.ylabel("AUROC")
    plt.title("S0 Evaluation AUROC (mean ± std)")
    for bar, value in zip(bars, means):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{value:.3f}",
            ha="center",
            va="bottom",
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    runs_root = Path(args.runs_dir)
    out_tables = Path(args.out_tables)
    out_figs = Path(args.out_figs)

    records = collect_eval_records(runs_root)
    if not records:
        print("[summarize_s0_results] No eval_summary.json files found.")
        return

    df = pd.DataFrame(records)
    out_tables.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_tables / "s0_eval_all_runs.csv", index=False)

    summary_df = build_summary(df)
    summary_path = out_tables / "s0_eval_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"[summarize_s0_results] Saved summary table to {summary_path}")

    plot_path = out_figs / "s0_auroc.png"
    plot_auroc(df, plot_path)
    print(f"[summarize_s0_results] Saved AUROC plot to {plot_path}")


if __name__ == "__main__":
    main()

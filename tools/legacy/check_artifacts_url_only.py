"""
Archived helper script for URL-only baseline artifact checks.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Archive-only URL artifact checker.")
    parser.add_argument(
        "experiment_dir", type=Path, help="Path to experiment directory."
    )
    args = parser.parse_args()

    metrics_path = args.experiment_dir / "metrics.json"
    predictions_path = args.experiment_dir / "predictions.csv"

    if metrics_path.exists():
        print(metrics_path.read_text(encoding="utf-8"))
    else:
        print("No metrics.json found.")

    if predictions_path.exists():
        df = pd.read_csv(predictions_path)
        print(df.head())
    else:
        print("No predictions.csv found.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
Aggregate evaluation metrics from saved prediction CSVs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from src.utils.metrics_v2 import (
    compute_brier_score,
    compute_ece,
    compute_fpr_at_tpr95,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize S0 evaluation metrics.")
    parser.add_argument(
        "--runs_dir", default="workspace/runs", help="Root directory containing runs."
    )
    parser.add_argument(
        "--out_csv",
        default="workspace/tables/s0_eval_summary.csv",
        help="Path to aggregated CSV.",
    )
    return parser.parse_args()


def extract_metadata(run_dir: Path) -> Dict[str, str]:
    model_name = run_dir.parent.name
    seed_token = run_dir.name
    return {
        "model": model_name,
        "run_dir": str(run_dir),
        "seed": seed_token.replace("seed_", ""),
    }


def evaluate_predictions(csv_path: Path) -> Dict[str, float | bool]:
    df = pd.read_csv(csv_path)
    if not {"y_true", "prob"}.issubset(df.columns):
        raise ValueError(f"{csv_path} missing required columns y_true/prob.")

    y_true = df["y_true"].to_numpy()
    y_prob = df["prob"].to_numpy()
    y_pred = (y_prob > 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auroc = roc_auc_score(y_true, y_prob)
    brier = compute_brier_score(y_true, y_prob)
    ece, ece_bins, low_sample = compute_ece(y_true, y_prob)
    fpr95, thr95, reached = compute_fpr_at_tpr95(y_true, y_prob)

    return {
        "accuracy": acc,
        "f1": f1,
        "auroc": auroc,
        "brier": brier,
        "ece": ece,
        "ece_bins": ece_bins,
        "ece_low_sample": bool(low_sample),
        "fpr_at_tpr95": fpr95,
        "thr_at_tpr95": thr95,
        "tpr95_reached": bool(reached),
    }


def main() -> None:
    args = parse_args()
    runs_root = Path(args.runs_dir)
    summary_records: List[Dict[str, object]] = []

    for model_dir in runs_root.glob("*"):
        if not model_dir.is_dir():
            continue
        for run_dir in model_dir.glob("seed_*"):
            artifacts_dir = run_dir / "artifacts"
            preds_path = artifacts_dir / "predictions_test.csv"
            if not preds_path.exists():
                continue
            metrics = evaluate_predictions(preds_path)
            record = extract_metadata(run_dir)
            record.update(metrics)
            summary_records.append(record)

            out_path = run_dir / "eval_summary.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(record, f, indent=2, ensure_ascii=False)
            print(f"[evaluate_s0] Saved {out_path}")

    if not summary_records:
        print("[evaluate_s0] No prediction files found; nothing to summarize.")
        return

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_records).to_csv(out_path, index=False)
    print(f"[evaluate_s0] Wrote aggregate metrics to {out_path}")


if __name__ == "__main__":
    main()

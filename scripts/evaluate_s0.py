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
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=None,
        help="Filter by scenario (e.g., 'iid', 'brandood'). If not specified, process all scenarios.",
    )
    return parser.parse_args()


def extract_metadata(run_dir: Path) -> Dict[str, str]:
    # 从实验目录名提取模型名称和种子
    exp_name = run_dir.name
    if "earlyconcat" in exp_name:
        model_name = "s0_earlyconcat"
    elif "lateavg" in exp_name:
        model_name = "s0_lateavg"
    else:
        model_name = exp_name.split("_")[0]  # 默认取第一个部分

    # 从目录名提取种子（假设格式为 ..._seed_42_...）
    seed = "unknown"
    if "seed_" in exp_name:
        seed_parts = exp_name.split("seed_")
        if len(seed_parts) > 1:
            seed = seed_parts[1].split("_")[0]

    # 从目录名提取场景（iid或brandood）
    scenario = "unknown"
    if "brandood" in exp_name or "brand_ood" in exp_name:
        scenario = "brandood"
    elif "iid" in exp_name:
        scenario = "iid"

    return {
        "model": model_name,
        "scenario": scenario,
        "run_dir": str(run_dir),
        "seed": seed,
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

    # 搜索所有包含predictions_test.csv的实验目录
    # 同时搜索experiments目录（如果runs_dir不是experiments）
    search_paths = [runs_root]
    if runs_root.name != "experiments" and Path("experiments").exists():
        search_paths.append(Path("experiments"))

    for search_path in search_paths:
        for preds_path in search_path.glob("**/predictions_test.csv"):
            artifacts_dir = preds_path.parent
            run_dir = artifacts_dir.parent

            # 提取元数据
            record = extract_metadata(run_dir)

            # 如果指定了scenarios参数，进行筛选
            if args.scenarios is not None:
                if record.get("scenario", "unknown") not in args.scenarios:
                    continue

            metrics = evaluate_predictions(preds_path)
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
    df = pd.DataFrame(summary_records)
    df.to_csv(out_path, index=False)
    print(f"[evaluate_s0] Wrote aggregate metrics to {out_path}")
    print(f"[evaluate_s0] Total experiments: {len(summary_records)}")
    if args.scenarios:
        print(f"[evaluate_s0] Filtered by scenarios: {args.scenarios}")


if __name__ == "__main__":
    main()

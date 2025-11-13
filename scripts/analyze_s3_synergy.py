#!/usr/bin/env python
"""
Comprehensive S3 synergy analysis and comparison tool.

Performs:
1. Statistical comparison between S3 and S1/S2 baselines
2. Synergy metrics calculation (Δ vs best baseline)
3. Generate paper-ready comparison tables
4. Statistical significance testing
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import pandas as pd
from scipy import stats

from src.utils.logging import get_logger

log = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze S3 synergy and generate comparison tables"
    )
    parser.add_argument(
        "--experiments-dir",
        type=Path,
        default=Path("experiments"),
        help="Root directory containing experiment results",
    )
    parser.add_argument(
        "--baselines",
        type=Path,
        default=Path("experiments/synergy_baselines.json"),
        help="Synergy baselines JSON file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Output directory for analysis results",
    )
    parser.add_argument(
        "--stage",
        default="test",
        choices=["val", "test"],
        help="Evaluation stage",
    )
    return parser.parse_args()


def load_experiment_metrics(
    experiments_dir: Path,
    pattern: str,
    stage: str,
) -> List[Dict[str, float]]:
    """Load metrics from all experiments matching pattern."""
    exp_dirs = list(experiments_dir.glob(f"{pattern}*"))
    metrics_list = []

    for exp_dir in exp_dirs:
        if not exp_dir.is_dir():
            continue

        # Try eval_summary.json first (for S3)
        eval_file = exp_dir / "eval_summary.json"
        if eval_file.exists():
            try:
                with open(eval_file, "r", encoding="utf-8") as f:
                    eval_data = json.load(f)
                if "s3" in eval_data:
                    metrics_list.append(eval_data["s3"])
                    continue
            except Exception:
                pass

        # Fallback to metrics file
        metrics_file = exp_dir / "artifacts" / f"metrics_{stage}.json"
        if metrics_file.exists():
            try:
                with open(metrics_file, "r", encoding="utf-8") as f:
                    metrics = json.load(f)
                metrics_list.append(metrics)
            except Exception as exc:
                log.warning(f"Failed to load {metrics_file}: {exc}")

    return metrics_list


def aggregate_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, Any]:
    """Aggregate metrics with mean, std, and individual values."""
    if not metrics_list:
        return {}

    metric_keys = ["auroc", "f1_macro", "ece", "brier", "accuracy"]
    aggregated = {}

    for key in metric_keys:
        values = [m[key] for m in metrics_list if key in m and np.isfinite(m[key])]
        if values:
            aggregated[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "n": len(values),
                "values": values,
            }

    return aggregated


def compute_synergy(
    s3_metrics: Dict[str, Any],
    s1_metrics: Dict[str, Any],
    s2_metrics: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute synergy metrics (improvement vs best baseline)."""
    synergy = {}

    for metric in ["auroc", "f1_macro", "ece", "brier"]:
        if metric not in s3_metrics:
            continue

        s3_value = s3_metrics[metric]["mean"]

        # Find best baseline (higher is better for most metrics, lower for ECE/Brier)
        is_lower_better = metric in ["ece", "brier"]

        baseline_values = []
        baseline_labels = []

        if metric in s1_metrics:
            baseline_values.append(s1_metrics[metric]["mean"])
            baseline_labels.append("S1")

        if metric in s2_metrics:
            baseline_values.append(s2_metrics[metric]["mean"])
            baseline_labels.append("S2")

        if not baseline_values:
            continue

        if is_lower_better:
            best_idx = np.argmin(baseline_values)
            delta = baseline_values[best_idx] - s3_value  # Positive means S3 is better
        else:
            best_idx = np.argmax(baseline_values)
            delta = s3_value - baseline_values[best_idx]  # Positive means S3 is better

        best_label = baseline_labels[best_idx]
        best_value = baseline_values[best_idx]

        # Compute relative improvement
        if is_lower_better:
            rel_improvement = (delta / best_value) * 100 if best_value != 0 else 0.0
        else:
            rel_improvement = (delta / best_value) * 100 if best_value != 0 else 0.0

        synergy[metric] = {
            "s3_value": s3_value,
            "best_baseline": best_label,
            "best_value": best_value,
            "delta": delta,
            "relative_improvement_pct": rel_improvement,
        }

    return synergy


def statistical_test(
    s3_values: List[float],
    baseline_values: List[float],
    test_type: str = "paired_t",
) -> Dict[str, float]:
    """Perform statistical significance test."""
    if len(s3_values) != len(baseline_values):
        log.warning("Unequal sample sizes, using independent t-test")
        test_type = "independent_t"

    if len(s3_values) < 2 or len(baseline_values) < 2:
        return {"p_value": np.nan, "test": "insufficient_samples"}

    try:
        if test_type == "paired_t":
            stat, p_value = stats.ttest_rel(s3_values, baseline_values)
        elif test_type == "wilcoxon":
            stat, p_value = stats.wilcoxon(s3_values, baseline_values)
        else:
            stat, p_value = stats.ttest_ind(s3_values, baseline_values)

        # Cohen's d effect size
        pooled_std = np.sqrt(
            (np.std(s3_values, ddof=1) ** 2 + np.std(baseline_values, ddof=1) ** 2) / 2
        )
        cohens_d = (
            (np.mean(s3_values) - np.mean(baseline_values)) / pooled_std
            if pooled_std > 0
            else 0.0
        )

        return {
            "test": test_type,
            "statistic": float(stat),
            "p_value": float(p_value),
            "cohens_d": float(cohens_d),
            "significant": p_value < 0.05,
        }
    except Exception as exc:
        log.warning(f"Statistical test failed: {exc}")
        return {"p_value": np.nan, "test": "failed"}


def generate_comparison_table(
    all_systems: Dict[str, Dict[str, Dict[str, Any]]],
    output_dir: Path,
) -> None:
    """Generate comparison table (CSV and LaTeX)."""
    rows = []

    for protocol in ["iid", "brandood"]:
        protocol_label = "IID" if protocol == "iid" else "Brand-OOD"

        for system in ["s0", "s1", "s2", "s3"]:
            key = f"{system}_{protocol}"
            if key not in all_systems:
                continue

            metrics = all_systems[key]
            row = {
                "Protocol": protocol_label,
                "System": system.upper(),
            }

            for metric in ["auroc", "f1_macro", "ece", "brier"]:
                if metric in metrics:
                    mean = metrics[metric]["mean"]
                    std = metrics[metric]["std"]
                    row[metric.upper()] = f"{mean:.4f} ± {std:.4f}"
                else:
                    row[metric.upper()] = "-"

            rows.append(row)

    df = pd.DataFrame(rows)

    # Save CSV
    csv_path = output_dir / "s3_comparison_table.csv"
    df.to_csv(csv_path, index=False)
    log.info(f"✓ Saved comparison table: {csv_path}")

    # Generate LaTeX
    latex_path = output_dir / "s3_comparison_table.tex"
    with open(latex_path, "w", encoding="utf-8") as f:
        f.write("% S0/S1/S2/S3 Comparison Table\n")
        f.write("% Generated by analyze_s3_synergy.py\n\n")
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Performance Comparison: S0/S1/S2/S3}\n")
        f.write("\\label{tab:s3_comparison}\n")
        f.write("\\begin{tabular}{llcccc}\n")
        f.write("\\hline\n")
        f.write("Protocol & System & AUROC & F1-Macro & ECE & Brier \\\\\n")
        f.write("\\hline\n")

        for _, row in df.iterrows():
            f.write(
                f"{row['Protocol']} & {row['System']} & {row['AUROC']} & {row['F1_MACRO']} & {row['ECE']} & {row['BRIER']} \\\\\n"
            )

        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    log.info(f"✓ Saved LaTeX table: {latex_path}")

    # Print to console
    print("\n" + "=" * 100)
    print("COMPARISON TABLE")
    print("=" * 100)
    print(df.to_string(index=False))
    print("=" * 100)


def main() -> None:
    args = parse_args()

    log.info("=" * 70)
    log.info("S3 Synergy Analysis")
    log.info("=" * 70)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load baselines
    if not args.baselines.exists():
        log.error(f"Baselines file not found: {args.baselines}")
        log.error("Run collect_synergy_baselines.py first!")
        return

    with open(args.baselines, "r", encoding="utf-8") as f:
        baseline_data = json.load(f)

    baselines = baseline_data.get("baselines", baseline_data)

    # Collect all system metrics
    all_systems = {}

    # S0
    for protocol in ["iid", "brandood"]:
        metrics_list = load_experiment_metrics(
            args.experiments_dir,
            f"s0_{protocol}_lateavg_",
            args.stage,
        )
        if metrics_list:
            all_systems[f"s0_{protocol}"] = aggregate_metrics(metrics_list)

    # S1, S2 from baselines
    for key, metrics in baselines.items():
        # Convert baseline format to aggregated format
        aggregated = {}
        for metric_key, value in metrics.items():
            if not metric_key.endswith(("_std", "_n")):
                aggregated[metric_key] = {
                    "mean": value,
                    "std": metrics.get(f"{metric_key}_std", 0.0),
                    "n": metrics.get(f"{metric_key}_n", 1),
                    "values": [value],  # Single value placeholder
                }
        all_systems[key] = aggregated

    # S3
    for protocol in ["iid", "brandood"]:
        metrics_list = load_experiment_metrics(
            args.experiments_dir,
            f"s3_{protocol}_fixed_",
            args.stage,
        )
        if metrics_list:
            all_systems[f"s3_{protocol}"] = aggregate_metrics(metrics_list)

    # Generate comparison table
    if all_systems:
        generate_comparison_table(all_systems, args.output_dir)

    # Compute synergy for each protocol
    synergy_results = {}

    for protocol in ["iid", "brandood"]:
        s3_key = f"s3_{protocol}"
        s1_key = f"s1_{protocol}"
        s2_key = f"s2_{protocol}"

        if s3_key not in all_systems:
            log.warning(f"S3 {protocol} metrics not found")
            continue

        s3_metrics = all_systems[s3_key]
        s1_metrics = all_systems.get(s1_key, {})
        s2_metrics = all_systems.get(s2_key, {})

        synergy = compute_synergy(s3_metrics, s1_metrics, s2_metrics)

        if synergy:
            synergy_results[protocol] = synergy

            # Print synergy report
            print(f"\n{'='*70}")
            print(f"SYNERGY ANALYSIS - {protocol.upper()}")
            print(f"{'='*70}")
            for metric, data in synergy.items():
                print(f"\n{metric.upper()}:")
                print(f"  S3: {data['s3_value']:.4f}")
                print(
                    f"  Best Baseline ({data['best_baseline']}): {data['best_value']:.4f}"
                )
                print(f"  Delta: {data['delta']:+.4f}")
                print(
                    f"  Relative Improvement: {data['relative_improvement_pct']:+.2f}%"
                )
            print(f"{'='*70}")

    # Save synergy analysis
    synergy_path = args.output_dir / "s3_synergy_analysis.json"
    with open(synergy_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "synergy_by_protocol": synergy_results,
                "all_systems": {
                    k: {
                        m: {"mean": v["mean"], "std": v["std"], "n": v["n"]}
                        for m, v in metrics.items()
                    }
                    for k, metrics in all_systems.items()
                },
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    log.info(f"✓ Saved synergy analysis: {synergy_path}")

    log.info("=" * 70)
    log.info("✓ Analysis complete!")
    log.info("=" * 70)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
Collect S1 (U-Module) and S2 (C-Module) baseline metrics for S3 synergy comparison.

Scans experiments/ directory for S1 and S2 results, aggregates key metrics,
and generates synergy_baselines.json for automatic loading by S3 experiments.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any
import numpy as np

from src.utils.logging import get_logger

log = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect S1/S2 baseline metrics for S3 synergy analysis"
    )
    parser.add_argument(
        "--experiments-dir",
        type=Path,
        default=Path("experiments"),
        help="Root directory containing experiment results",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/synergy_baselines.json"),
        help="Output JSON file path",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["auroc", "f1_macro", "ece", "brier", "accuracy"],
        help="Metrics to collect",
    )
    parser.add_argument(
        "--stage",
        default="test",
        choices=["val", "test"],
        help="Evaluation stage to extract metrics from",
    )
    return parser.parse_args()


def find_experiment_dirs(root: Path, pattern: str) -> List[Path]:
    """Find all experiment directories matching pattern."""
    matches = []
    for path in root.glob(f"{pattern}*"):
        if path.is_dir() and not path.name.endswith((".tar", ".zip")):
            matches.append(path)
    return sorted(matches)


def extract_metrics_from_experiment(
    exp_dir: Path, stage: str, metric_names: List[str]
) -> Dict[str, float]:
    """Extract metrics from experiment artifacts."""
    metrics_file = exp_dir / "artifacts" / f"metrics_{stage}.json"

    if not metrics_file.exists():
        log.warning(f"Metrics file not found: {metrics_file}")
        return {}

    try:
        with open(metrics_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        log.warning(f"Failed to load {metrics_file}: {exc}")
        return {}

    result = {}
    for metric in metric_names:
        if metric in data:
            try:
                result[metric] = float(data[metric])
            except (TypeError, ValueError):
                log.warning(f"Invalid {metric} value in {exp_dir.name}: {data[metric]}")

    return result


def aggregate_metrics(
    exp_dirs: List[Path], stage: str, metric_names: List[str]
) -> Dict[str, Any]:
    """Aggregate metrics across multiple seeds."""
    all_metrics: Dict[str, List[float]] = {m: [] for m in metric_names}

    for exp_dir in exp_dirs:
        metrics = extract_metrics_from_experiment(exp_dir, stage, metric_names)
        for metric, value in metrics.items():
            if not np.isnan(value) and np.isfinite(value):
                all_metrics[metric].append(value)

    # Compute mean and std
    aggregated = {}
    for metric, values in all_metrics.items():
        if values:
            aggregated[f"{metric}"] = float(np.mean(values))
            if len(values) > 1:
                aggregated[f"{metric}_std"] = float(np.std(values, ddof=1))
            aggregated[f"{metric}_n"] = len(values)
        else:
            log.warning(f"No valid values for metric: {metric}")

    return aggregated


def collect_baselines(
    experiments_dir: Path,
    stage: str,
    metric_names: List[str],
) -> Dict[str, Any]:
    """Collect all baseline metrics."""
    baselines = {}

    # S1 experiments (U-Module only)
    for protocol in ["iid", "brandood"]:
        pattern = f"s1_{protocol}_"
        exp_dirs = find_experiment_dirs(experiments_dir, pattern)

        if not exp_dirs:
            log.warning(f"No S1 {protocol} experiments found")
            continue

        log.info(
            f"Found {len(exp_dirs)} S1 {protocol} experiments: {[d.name for d in exp_dirs]}"
        )
        metrics = aggregate_metrics(exp_dirs, stage, metric_names)

        if metrics:
            baselines[f"s1_{protocol}"] = metrics
            log.info(f"S1 {protocol} aggregated: {metrics}")

    # S2 experiments (C-Module only)
    for protocol in ["iid", "brandood"]:
        pattern = f"s2_{protocol}_"
        exp_dirs = find_experiment_dirs(experiments_dir, pattern)

        if not exp_dirs:
            log.warning(f"No S2 {protocol} experiments found")
            continue

        log.info(
            f"Found {len(exp_dirs)} S2 {protocol} experiments: {[d.name for d in exp_dirs]}"
        )
        metrics = aggregate_metrics(exp_dirs, stage, metric_names)

        if metrics:
            baselines[f"s2_{protocol}"] = metrics
            log.info(f"S2 {protocol} aggregated: {metrics}")

    return baselines


def main() -> None:
    args = parse_args()

    log.info("=" * 70)
    log.info("Collecting Synergy Baselines (S1/S2 Metrics)")
    log.info("=" * 70)
    log.info(f"Experiments directory: {args.experiments_dir}")
    log.info(f"Output file: {args.output}")
    log.info(f"Stage: {args.stage}")
    log.info(f"Metrics: {args.metrics}")

    if not args.experiments_dir.exists():
        log.error(f"Experiments directory not found: {args.experiments_dir}")
        return

    # Collect baselines
    baselines = collect_baselines(
        args.experiments_dir,
        args.stage,
        args.metrics,
    )

    if not baselines:
        log.error("No baselines collected! Check experiment directories.")
        return

    # Add metadata
    output_data = {
        "baselines": baselines,
        "metadata": {
            "stage": args.stage,
            "metrics": args.metrics,
            "experiments_dir": str(args.experiments_dir),
        },
    }

    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    log.info("=" * 70)
    log.info(f"✓ Synergy baselines saved to: {args.output}")
    log.info("=" * 70)

    # Print summary
    print("\n" + "=" * 70)
    print("BASELINE SUMMARY")
    print("=" * 70)
    for key, metrics in baselines.items():
        print(f"\n{key.upper()}:")
        for metric, value in sorted(metrics.items()):
            if not metric.endswith(("_std", "_n")):
                std = metrics.get(f"{metric}_std", 0.0)
                n = metrics.get(f"{metric}_n", 1)
                print(f"  {metric:12s}: {value:.4f} ± {std:.4f} (n={n})")
    print("=" * 70)


if __name__ == "__main__":
    main()

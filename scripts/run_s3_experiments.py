#!/usr/bin/env python
"""
Run S3 (fixed fusion) experiments across protocols and seeds.

Orchestrates batch execution of S3 experiments with:
- Multiple protocols (IID, Brand-OOD)
- Multiple random seeds
- Error handling and progress tracking
- Automatic result summarization
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime
from typing import List, Dict, Any

from src.utils.logging import get_logger

log = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run S3 fixed fusion experiments")
    parser.add_argument(
        "--protocols",
        nargs="+",
        choices=["iid", "brandood"],
        default=["iid", "brandood"],
        help="Protocols to run",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 43, 44],
        help="Random seeds",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue running remaining experiments even if one fails",
    )
    parser.add_argument(
        "--wandb-offline",
        action="store_true",
        help="Use WandB in offline mode",
    )
    return parser.parse_args()


def build_command(
    protocol: str,
    seed: int,
    wandb_offline: bool = False,
) -> List[str]:
    """Build training command for S3 experiment."""
    config_name = f"s3_{protocol}_fixed"

    cmd = [
        sys.executable,
        "scripts/train_hydra.py",
        f"experiment={config_name}",
        f"run.seed={seed}",
    ]

    if wandb_offline:
        cmd.append("logger.wandb.offline=true")

    return cmd


def run_experiment(
    cmd: List[str],
    protocol: str,
    seed: int,
    exp_num: int,
    total: int,
) -> Dict[str, Any]:
    """Run a single experiment and track results."""
    start_time = time.time()

    log.info("=" * 70)
    log.info(
        f"[{exp_num}/{total}] Starting S3 {protocol.upper()} experiment (seed={seed})"
    )
    log.info("Command: " + " ".join(cmd))
    log.info("=" * 70)

    try:
        subprocess.run(
            cmd,
            check=True,
            capture_output=False,
        )
        elapsed = time.time() - start_time

        log.info("=" * 70)
        log.info(
            f"✓ [{exp_num}/{total}] S3 {protocol.upper()} seed={seed} completed successfully"
        )
        log.info(f"  Duration: {elapsed/60:.1f} minutes")
        log.info("=" * 70)

        return {
            "protocol": protocol,
            "seed": seed,
            "status": "success",
            "duration": elapsed,
            "exit_code": 0,
        }

    except subprocess.CalledProcessError as exc:
        elapsed = time.time() - start_time

        log.error("=" * 70)
        log.error(f"✗ [{exp_num}/{total}] S3 {protocol.upper()} seed={seed} FAILED")
        log.error(f"  Exit code: {exc.returncode}")
        log.error(f"  Duration: {elapsed/60:.1f} minutes")
        log.error("=" * 70)

        return {
            "protocol": protocol,
            "seed": seed,
            "status": "failed",
            "duration": elapsed,
            "exit_code": exc.returncode,
        }

    except Exception as exc:
        elapsed = time.time() - start_time

        log.error("=" * 70)
        log.error(f"✗ [{exp_num}/{total}] S3 {protocol.upper()} seed={seed} ERROR")
        log.error(f"  Error: {exc}")
        log.error(f"  Duration: {elapsed/60:.1f} minutes")
        log.error("=" * 70)

        return {
            "protocol": protocol,
            "seed": seed,
            "status": "error",
            "duration": elapsed,
            "exit_code": -1,
            "error": str(exc),
        }


def print_summary(results: List[Dict[str, Any]]) -> None:
    """Print experiment summary."""
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)

    for idx, result in enumerate(results, 1):
        status_icon = "✓" if result["status"] == "success" else "✗"
        print(
            f"{status_icon} [{idx}/{len(results)}] "
            f"S3 {result['protocol'].upper()} seed={result['seed']}: "
            f"{result['status'].upper()} "
            f"({result['duration']/60:.1f} min)"
        )

    success_count = sum(1 for r in results if r["status"] == "success")
    failed_count = len(results) - success_count

    total_duration = sum(r["duration"] for r in results)

    print("=" * 70)
    print(f"Total: {len(results)} experiments")
    print(f"  Success: {success_count}")
    print(f"  Failed: {failed_count}")
    print(f"  Total time: {total_duration/3600:.2f} hours")
    print("=" * 70)


def main() -> None:
    args = parse_args()

    log.info("=" * 70)
    log.info("S3 Fixed Fusion Experiment Runner")
    log.info("=" * 70)
    log.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info(f"Protocols: {args.protocols}")
    log.info(f"Seeds: {args.seeds}")
    log.info(f"Dry run: {args.dry_run}")
    log.info(f"Continue on error: {args.continue_on_error}")
    log.info("=" * 70)

    # Build experiment list
    experiments = []
    for protocol in args.protocols:
        for seed in args.seeds:
            experiments.append(
                {
                    "protocol": protocol,
                    "seed": seed,
                    "cmd": build_command(protocol, seed, args.wandb_offline),
                }
            )

    log.info(f"\nTotal experiments to run: {len(experiments)}")

    if args.dry_run:
        print("\n" + "=" * 70)
        print("DRY RUN - Commands to be executed:")
        print("=" * 70)
        for idx, exp in enumerate(experiments, 1):
            print(f"[{idx}/{len(experiments)}] {' '.join(exp['cmd'])}")
        print("=" * 70)
        return

    # Run experiments
    results = []
    start_time = time.time()

    for idx, exp in enumerate(experiments, 1):
        result = run_experiment(
            exp["cmd"],
            exp["protocol"],
            exp["seed"],
            idx,
            len(experiments),
        )
        results.append(result)

        if result["status"] != "success" and not args.continue_on_error:
            log.error("\nExperiment failed and --continue-on-error not set. Stopping.")
            break

        # Brief pause between experiments
        if idx < len(experiments):
            time.sleep(5)

    total_elapsed = time.time() - start_time

    # Print summary
    print_summary(results)

    log.info(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info(f"Total elapsed: {total_elapsed/3600:.2f} hours")

    # Exit with error if any experiments failed
    if any(r["status"] != "success" for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Monitor S3 experiments and automatically generate visualizations when both are complete.
"""

import argparse
import subprocess
import time
from pathlib import Path


def check_experiment_complete(exp_pattern: str) -> bool:
    """Check if an experiment is complete by looking for eval_summary.json."""
    from glob import glob

    exp_dirs = sorted(glob(exp_pattern))
    if not exp_dirs:
        return False

    exp_dir = Path(exp_dirs[-1])
    summary_file = exp_dir / "results" / "eval_summary.json"
    return summary_file.exists()


def main():
    parser = argparse.ArgumentParser(
        description="Monitor S3 experiments and generate visualizations"
    )
    parser.add_argument(
        "--check-interval",
        type=int,
        default=60,
        help="Check interval in seconds (default: 60)",
    )
    parser.add_argument(
        "--max-wait",
        type=int,
        default=7200,
        help="Maximum wait time in seconds (default: 7200 = 2 hours)",
    )
    args = parser.parse_args()

    iid_pattern = "experiments/s3_iid_fixed_*"
    brandood_pattern = "experiments/s3_brandood_fixed_*"

    print("=" * 70)
    print("S3 Experiment Monitor")
    print("=" * 70)
    print("Monitoring for:")
    print(f"  - IID: {iid_pattern}")
    print(f"  - Brand-OOD: {brandood_pattern}")
    print(f"Check interval: {args.check_interval}s")
    print(f"Max wait time: {args.max_wait}s")
    print("=" * 70)

    start_time = time.time()
    iid_complete = False
    brandood_complete = False

    while (time.time() - start_time) < args.max_wait:
        # Check IID
        if not iid_complete:
            iid_complete = check_experiment_complete(iid_pattern)
            status = "COMPLETE" if iid_complete else "RUNNING"
            print(f"[{time.strftime('%H:%M:%S')}] IID: {status}")

        # Check Brand-OOD
        if not brandood_complete:
            brandood_complete = check_experiment_complete(brandood_pattern)
            status = "COMPLETE" if brandood_complete else "RUNNING"
            print(f"[{time.strftime('%H:%M:%S')}] Brand-OOD: {status}")

        # Both complete?
        if iid_complete and brandood_complete:
            print("\n" + "=" * 70)
            print("Both experiments complete! Generating visualizations...")
            print("=" * 70)

            # Run visualization script
            try:
                result = subprocess.run(
                    ["python", "scripts/visualize_s3_final.py"],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                print(result.stdout)
                print("\nVisualization complete!")
                return 0
            except subprocess.CalledProcessError as e:
                print(f"Error running visualization: {e}")
                print(f"stderr: {e.stderr}")
                return 1

        # Wait before next check
        time.sleep(args.check_interval)

    # Timeout
    print(f"\nTimeout reached ({args.max_wait}s). Experiments still running:")
    print(f"  - IID: {'complete' if iid_complete else 'incomplete'}")
    print(f"  - Brand-OOD: {'complete' if brandood_complete else 'incomplete'}")
    return 2


if __name__ == "__main__":
    exit(main())

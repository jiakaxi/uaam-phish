#!/usr/bin/env python
"""
Orchestrate S0 experiments (multiple models × seeds).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


SCENARIO_TO_CONFIG = {
    "iid": {
        "s0_earlyconcat": "s0_iid_earlyconcat",
        "s0_lateavg": "s0_iid_lateavg",
    },
    "brandood": {
        "s0_earlyconcat": "s0_brandood_earlyconcat",
        "s0_lateavg": "s0_brandood_lateavg",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run S0 Hydra experiments.")
    parser.add_argument(
        "--scenario",
        choices=["iid", "brandood"],
        default="iid",
        help="Dataset scenario.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["s0_earlyconcat", "s0_lateavg"],
        help="Model keys (s0_earlyconcat / s0_lateavg).",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42],
        help="Random seeds to iterate.",
    )
    parser.add_argument(
        "--runs-root",
        default="workspace/runs",
        help="Root directory for Lightning outputs.",
    )
    parser.add_argument(
        "--logger",
        default=None,
        help="Optional Hydra logger override (e.g., wandb or csv).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue running remaining experiments even if one fails.",
    )
    parser.add_argument(
        "--wandb-offline",
        action="store_true",
        help="Use WandB in offline mode to avoid connection issues.",
    )
    args, unknown = parser.parse_known_args()
    args.hydra_overrides = unknown
    return args


def build_command(
    config_name: str,
    seed: int,
    runs_root: Path,
    logger: str | None,
    hydra_overrides: List[str],
    wandb_offline: bool = False,
) -> List[str]:
    output_dir = runs_root / config_name / f"seed_{seed}"
    cmd = [
        sys.executable,
        "scripts/train_hydra.py",
        f"experiment={config_name}",
        f"run.seed={seed}",
        f"paths.output_dir={output_dir.as_posix()}",
    ]
    if logger:
        cmd.append(f"logger={logger}")
        if wandb_offline and logger == "wandb":
            cmd.append("logger.offline=true")
    cmd.extend(hydra_overrides)
    return cmd


def main() -> None:
    args = parse_args()
    runs_root = Path(args.runs_root)
    runs_root.mkdir(parents=True, exist_ok=True)

    scenario_configs = SCENARIO_TO_CONFIG[args.scenario]
    commands: List[List[str]] = []

    for model_key in args.models:
        if model_key not in scenario_configs:
            raise ValueError(
                f"Unknown model key '{model_key}' for scenario {args.scenario}"
            )
        config_name = scenario_configs[model_key]
        for seed in args.seeds:
            cmd = build_command(
                config_name=config_name,
                seed=seed,
                runs_root=runs_root,
                logger=args.logger,
                hydra_overrides=args.hydra_overrides,
                wandb_offline=args.wandb_offline,
            )
            commands.append(cmd)

    failed_commands: List[List[str]] = []
    for i, cmd in enumerate(commands, 1):
        print(f"[run_s0_experiments] [{i}/{len(commands)}] >", " ".join(cmd))
        if args.dry_run:
            continue
        completed = subprocess.run(cmd, check=False)
        if completed.returncode != 0:
            print(
                f"[run_s0_experiments] ERROR: Command failed with exit code {completed.returncode}",
                file=sys.stderr,
            )
            failed_commands.append(cmd)
            if not args.continue_on_error:
                raise SystemExit(completed.returncode)

    if failed_commands:
        print(
            f"\n[run_s0_experiments] WARNING: {len(failed_commands)} experiment(s) failed:",
            file=sys.stderr,
        )
        for cmd in failed_commands:
            print(f"  - {' '.join(cmd)}", file=sys.stderr)
        if args.continue_on_error:
            print(
                f"\n[run_s0_experiments] INFO: {len(commands) - len(failed_commands)} experiment(s) completed successfully.",
                file=sys.stderr,
            )
            sys.exit(1)


if __name__ == "__main__":
    main()

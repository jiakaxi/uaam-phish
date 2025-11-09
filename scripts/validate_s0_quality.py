#!/usr/bin/env python
"""
Quality gate checks for S0 assets (splits, corruption data, run artifacts).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict

import pandas as pd

REQUIRED_SPLIT_COLS = [
    "id",
    "label",
    "url_text",
    "html_path",
    "img_path",
    "brand",
    "timestamp",
    "etld_plus_one",
    "source",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate S0 data quality gates.")
    parser.add_argument(
        "--splits_root", default="workspace/data/splits", help="Splits root directory."
    )
    parser.add_argument(
        "--corrupt_root", default="workspace/data/corrupt", help="Corruption root."
    )
    parser.add_argument(
        "--runs_dir", default="workspace/runs", help="Runs directory for metrics."
    )
    parser.add_argument(
        "--out_report",
        default="workspace/reports/quality_report.json",
        help="Output JSON report path.",
    )
    return parser.parse_args()


def check_split(csv_path: Path) -> Dict[str, object]:
    if not csv_path.exists():
        return {"status": "missing", "path": str(csv_path)}
    df = pd.read_csv(csv_path)
    missing_cols = [col for col in REQUIRED_SPLIT_COLS if col not in df.columns]
    if missing_cols:
        return {
            "status": "failed",
            "path": str(csv_path),
            "reason": f"missing columns {missing_cols}",
        }
    if df.isna().any().any():
        return {"status": "warning", "path": str(csv_path), "reason": "contains NaN"}
    return {"status": "passed", "path": str(csv_path), "rows": len(df)}


def check_image_corruption(csv_path: Path, base_dir: Path) -> Dict[str, object]:
    if not csv_path.exists():
        return {"status": "missing", "path": str(csv_path)}
    df = pd.read_csv(csv_path)
    required = {"img_path_corrupt", "img_sha256_corrupt"}
    if not required.issubset(df.columns):
        return {
            "status": "failed",
            "path": str(csv_path),
            "reason": "missing corruption columns",
        }
    mismatches = []
    for _, row in df.iterrows():
        rel_path = row["img_path_corrupt"]
        full_path = base_dir / rel_path
        if not full_path.exists():
            mismatches.append(str(rel_path))
            continue
        sha = hashlib.sha256(full_path.read_bytes()).hexdigest()
        if sha != str(row["img_sha256_corrupt"]):
            mismatches.append(str(rel_path))
    if mismatches:
        return {
            "status": "failed",
            "path": str(csv_path),
            "reason": f"sha mismatch or missing files ({len(mismatches)})",
        }
    return {"status": "passed", "path": str(csv_path), "rows": len(df)}


def check_runs(runs_root: Path) -> Dict[str, object]:
    missing = []
    total = 0
    for model_dir in runs_root.glob("*"):
        if not model_dir.is_dir():
            continue
        for run_dir in model_dir.glob("seed_*"):
            total += 1
            if not (run_dir / "eval_summary.json").exists():
                missing.append(str(run_dir))
    if missing:
        return {
            "status": "warning",
            "reason": f"{len(missing)} run(s) missing eval_summary.json",
            "missing": missing,
            "total_runs": total,
        }
    return {"status": "passed", "total_runs": total}


def main() -> None:
    args = parse_args()
    splits_root = Path(args.splits_root)
    corrupt_root = Path(args.corrupt_root)
    runs_root = Path(args.runs_dir)

    report: Dict[str, object] = {"splits": {}, "corruption": {}, "runs": {}}

    iid_splits = ["train.csv", "val.csv", "test.csv"]
    brand_splits = ["train.csv", "val.csv", "test_id.csv", "test_ood.csv"]

    for fname in iid_splits:
        path = splits_root / "iid" / fname
        report["splits"][f"iid/{fname}"] = check_split(path)

    for fname in brand_splits:
        path = splits_root / "brandood" / fname
        report["splits"][f"brandood/{fname}"] = check_split(path)

    image_csvs = sorted((corrupt_root / "img").glob("test_corrupt_img_*.csv"))
    for csv_path in image_csvs:
        report["corruption"][csv_path.name] = check_image_corruption(
            csv_path, corrupt_root / "img"
        )

    report["runs"] = check_runs(runs_root)

    def section_passed(section: object) -> bool:
        if isinstance(section, dict) and "status" in section:
            return section.get("status") == "passed"
        if isinstance(section, dict):
            return all(item.get("status") == "passed" for item in section.values())
        return False

    report["passed"] = all(
        section_passed(section)
        for section in [report["splits"], report["corruption"], report["runs"]]
    )

    out_path = Path(args.out_report)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"[validate_s0_quality] Report saved to {out_path}")


if __name__ == "__main__":
    main()

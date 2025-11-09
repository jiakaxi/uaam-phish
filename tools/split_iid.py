#!/usr/bin/env python
"""
IID split utility for S0 baselines.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd
from sklearn.model_selection import train_test_split
import tldextract

REQUIRED_COLUMNS = [
    "id",
    "label",
    "url_text",
    "html_path",
    "img_path",
    "brand",
    "timestamp",
]

OPTIONAL_COLUMNS = ["etld_plus_one", "source", "domain"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create IID splits (70/15/15).")
    parser.add_argument(
        "--in", dest="input_csv", required=True, help="Master CSV path."
    )
    parser.add_argument(
        "--out",
        dest="output_dir",
        required=True,
        help="Output directory for IID splits (train/val/test).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--val-ratio", type=float, default=0.15, help="Validation ratio."
    )
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test ratio.")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only validate schema without writing files.",
    )
    return parser.parse_args()


def ensure_required_columns(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def ensure_etld_column(df: pd.DataFrame) -> pd.DataFrame:
    if "etld_plus_one" in df.columns and df["etld_plus_one"].notna().any():
        return df

    def compute_etld(url: str) -> str:
        try:
            ext = tldextract.extract(url or "")
            domain = ".".join([p for p in [ext.domain, ext.suffix] if p])
            return domain or ""
        except Exception:
            return ""

    df["etld_plus_one"] = df["url_text"].astype(str).apply(compute_etld)
    return df


def ensure_source_column(df: pd.DataFrame, source_hint: str) -> pd.DataFrame:
    if "source" not in df.columns:
        df["source"] = source_hint
    df["source"] = df["source"].fillna(source_hint)
    return df


def normalize_brand(df: pd.DataFrame) -> pd.DataFrame:
    df["brand"] = df["brand"].astype(str).str.strip().str.lower()
    return df


def stratified_split(
    df: pd.DataFrame, val_ratio: float, test_ratio: float, seed: int
) -> List[pd.DataFrame]:
    if not 0 < val_ratio < 1 or not 0 < test_ratio < 1:
        raise ValueError("val_ratio and test_ratio must be between 0 and 1.")
    if val_ratio + test_ratio >= 1:
        raise ValueError("val_ratio + test_ratio must be < 1.")

    train_df, temp_df = train_test_split(
        df,
        test_size=val_ratio + test_ratio,
        stratify=df["label"],
        random_state=seed,
    )
    relative_val = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=1 - relative_val,
        stratify=temp_df["label"],
        random_state=seed,
    )
    return [
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    ]


def write_split(df: pd.DataFrame, path: Path) -> None:
    cols = REQUIRED_COLUMNS + [col for col in OPTIONAL_COLUMNS if col in df.columns]
    unique_cols = []
    for col in cols:
        if col not in unique_cols:
            unique_cols.append(col)
    extra_cols = [col for col in df.columns if col not in unique_cols]
    ordered_cols = unique_cols + extra_cols
    df.loc[:, ordered_cols].to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_csv)
    output_dir = Path(args.output_dir)

    if not input_path.exists():
        raise FileNotFoundError(input_path)

    df = pd.read_csv(input_path)
    ensure_required_columns(df)
    df = ensure_etld_column(df)
    df = ensure_source_column(df, source_hint=str(input_path.parent))
    df = normalize_brand(df)

    if args.check_only:
        print("[OK] Schema validation passed.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    train_df, val_df, test_df = stratified_split(
        df, args.val_ratio, args.test_ratio, args.seed
    )

    write_split(train_df, output_dir / "train.csv")
    write_split(val_df, output_dir / "val.csv")
    write_split(test_df, output_dir / "test.csv")

    summary = {
        "train": len(train_df),
        "val": len(val_df),
        "test": len(test_df),
    }
    print("IID split summary:", summary)


if __name__ == "__main__":
    main()

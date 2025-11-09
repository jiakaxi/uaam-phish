#!/usr/bin/env python
"""
Brand-OOD split utility (train/val/test_id/test_ood + brand sets JSON).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

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
    parser = argparse.ArgumentParser(description="Create Brand-OOD splits.")
    parser.add_argument(
        "--in", dest="input_csv", required=True, help="Master CSV path."
    )
    parser.add_argument(
        "--out",
        dest="output_dir",
        required=True,
        help="Output directory for Brand-OOD splits.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--top_k", type=int, default=20, help="Number of in-domain brands."
    )
    parser.add_argument(
        "--ood_ratio",
        type=float,
        default=0.25,
        help="Desired ratio (#test_ood ≈ ood_ratio * #test_id).",
    )
    return parser.parse_args()


def ensure_required(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def ensure_etld(df: pd.DataFrame) -> pd.DataFrame:
    if "etld_plus_one" in df.columns and df["etld_plus_one"].notna().any():
        return df

    def compute(url: str) -> str:
        try:
            ext = tldextract.extract(url or "")
            domain = ".".join([p for p in [ext.domain, ext.suffix] if p])
            return domain or ""
        except Exception:
            return ""

    df["etld_plus_one"] = df["url_text"].astype(str).apply(compute)
    return df


def normalize_brand(df: pd.DataFrame) -> pd.DataFrame:
    df["brand"] = df["brand"].astype(str).str.strip().str.lower()
    return df


def ensure_source(df: pd.DataFrame, hint: str) -> pd.DataFrame:
    if "source" not in df.columns:
        df["source"] = hint
    df["source"] = df["source"].fillna(hint)
    return df


def select_brand_sets(df: pd.DataFrame, top_k: int) -> Tuple[List[str], List[str]]:
    counts = df["brand"].value_counts()
    b_ind = counts.head(top_k).index.tolist()
    b_ood = [b for b in counts.index.tolist() if b not in b_ind]
    return b_ind, b_ood


def stratified_split(
    df: pd.DataFrame, seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df, temp_df = train_test_split(
        df, test_size=0.3, stratify=df["label"], random_state=seed
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df["label"], random_state=seed
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def write_csv(df: pd.DataFrame, path: Path) -> None:
    cols = REQUIRED_COLUMNS + [c for c in OPTIONAL_COLUMNS if c in df.columns]
    unique_cols = []
    for col in cols:
        if col not in unique_cols:
            unique_cols.append(col)
    extra_cols = [col for col in df.columns if col not in unique_cols]
    ordered = unique_cols + extra_cols
    df.loc[:, ordered].to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    ensure_required(df)
    df = ensure_etld(df)
    df = ensure_source(df, str(input_path.parent))
    df = normalize_brand(df)

    b_ind, b_ood = select_brand_sets(df, args.top_k)
    if not b_ind:
        raise ValueError("No brands available for in-domain set.")

    df_ind = df[df["brand"].isin(b_ind)].copy()
    df_ood = df[df["brand"].isin(b_ood)].copy()

    train_df, val_df, test_id_df = stratified_split(df_ind, args.seed)

    if len(df_ood) == 0:
        test_ood_df = pd.DataFrame(columns=df.columns)
    else:
        target_ood = int(round(len(test_id_df) * args.ood_ratio))
        if target_ood <= 0:
            target_ood = min(len(df_ood), len(test_id_df))
        target_ood = min(len(df_ood), target_ood) or len(df_ood)
        test_ood_df = (
            df_ood.sample(n=target_ood, random_state=args.seed)
            .reset_index(drop=True)
            .copy()
        )

    write_csv(train_df, output_dir / "train.csv")
    write_csv(val_df, output_dir / "val.csv")
    write_csv(test_id_df, output_dir / "test_id.csv")
    write_csv(test_ood_df, output_dir / "test_ood.csv")

    brand_sets = {"b_ind": sorted(b_ind), "b_ood": sorted(b_ood)}
    with open(output_dir / "brand_sets.json", "w", encoding="utf-8") as f:
        json.dump(brand_sets, f, indent=2, ensure_ascii=False)

    summary = {
        "train": len(train_df),
        "val": len(val_df),
        "test_id": len(test_id_df),
        "test_ood": len(test_ood_df),
        "num_in_domain_brands": len(b_ind),
        "num_ood_brands": len(b_ood),
    }
    print("Brand-OOD split summary:", summary)


if __name__ == "__main__":
    main()

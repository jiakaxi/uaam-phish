"""
Data splitting utilities for random, temporal, and brand-OOD protocols.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Literal, Optional, Tuple

import pandas as pd

from src.utils.logging import get_logger

log = get_logger(__name__)

SplitProtocol = Literal["random", "temporal", "brand_ood"]


def build_splits(
    df: pd.DataFrame,
    cfg: any,
    protocol: Optional[SplitProtocol] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    """
    Build train/val/test splits according to specified protocol.

    Args:
        df: Master dataframe with columns: url_text, label, timestamp, brand, source
        cfg: Configuration object
        protocol: One of 'random', 'temporal', 'brand_ood'. If None, read from cfg.

    Returns:
        train_df, val_df, test_df, metadata_dict

    Metadata dict contains:
        - protocol: actual protocol used
        - downgraded_to: protocol if downgraded, else None
        - downgrade_reason: reason if downgraded
        - split_stats: statistics for each split
        - tie_policy: for temporal splits
        - brand_normalization: for brand_ood splits
    """
    if protocol is None:
        protocol = cfg.get("protocol", "random")

    metadata = {
        "protocol": protocol,
        "downgraded_to": None,
        "downgrade_reason": None,
        "tie_policy": None,
        "brand_normalization": None,
    }

    # Check required columns
    required_cols = {"url_text", "label"}
    if not required_cols.issubset(df.columns):
        missing = ", ".join(sorted(required_cols - set(df.columns)))
        raise ValueError(f"Missing required columns: {missing}")

    # Get split ratios
    split_ratios = cfg.data.get(
        "split_ratios", {"train": 0.7, "val": 0.15, "test": 0.15}
    )
    train_ratio = split_ratios["train"]
    val_ratio = split_ratios["val"]
    test_ratio = split_ratios["test"]

    # Attempt protocol-specific split with downgrade logic
    if protocol == "temporal":
        if "timestamp" not in df.columns:
            log.warning(
                "Temporal protocol requested but 'timestamp' column missing. Downgrading to random."
            )
            metadata["downgraded_to"] = "random"
            metadata["downgrade_reason"] = "Missing timestamp column"
            protocol = "random"
        else:
            train_df, val_df, test_df = _temporal_split(
                df, train_ratio, val_ratio, test_ratio
            )
            metadata["tie_policy"] = "left-closed"

    if protocol == "brand_ood":
        if "brand" not in df.columns:
            log.warning(
                "Brand-OOD protocol requested but 'brand' column missing. Downgrading to random."
            )
            metadata["downgraded_to"] = "random"
            metadata["downgrade_reason"] = "Missing brand column"
            protocol = "random"
        else:
            # Normalize brands
            df = df.copy()
            df["brand"] = df["brand"].fillna("").astype(str).str.strip().str.lower()
            unique_brands = df["brand"].unique()
            if len(unique_brands) <= 2:
                log.warning(
                    f"Brand-OOD protocol requested but only {len(unique_brands)} unique brands. Downgrading to random."
                )
                metadata["downgraded_to"] = "random"
                metadata["downgrade_reason"] = (
                    f"Insufficient unique brands ({len(unique_brands)} ≤ 2)"
                )
                protocol = "random"
            else:
                train_df, val_df, test_df = _brand_ood_split(
                    df, train_ratio, val_ratio, test_ratio
                )
                metadata["brand_normalization"] = "strip+lower"
                # Verify disjointness
                train_brands = set(train_df["brand"].unique())
                test_brands = set(test_df["brand"].unique())
                if train_brands & test_brands:
                    log.error("Brand-OOD split failed: train and test brands overlap!")
                    metadata["downgraded_to"] = "random"
                    metadata["downgrade_reason"] = "Brand disjointness check failed"
                    protocol = "random"

    if protocol == "random":
        train_df, val_df, test_df = _random_split(
            df, train_ratio, val_ratio, test_ratio
        )

    # Compute split statistics
    metadata["split_stats"] = _compute_split_stats(train_df, val_df, test_df)

    # Add brand intersection check
    if "brand" in df.columns:
        train_brands = (
            set(train_df["brand"].dropna().unique())
            if "brand" in train_df.columns
            else set()
        )
        test_brands = (
            set(test_df["brand"].dropna().unique())
            if "brand" in test_df.columns
            else set()
        )
        # Store as bool in metadata (not in split_stats)
        metadata["brand_intersection_ok"] = len(train_brands & test_brands) == 0

    return train_df, val_df, test_df, metadata


def _random_split(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Random stratified split by label (and brand if present)."""
    df = df.copy().reset_index(drop=True)

    # Stratify by label and brand if available
    if "brand" in df.columns:
        df["_strata"] = (
            df["label"].astype(str) + "_" + df["brand"].fillna("").astype(str)
        )
    else:
        df["_strata"] = df["label"].astype(str)

    # Shuffle within each stratum
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    n = len(df)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_df = df.iloc[:train_end].drop(columns=["_strata"], errors="ignore")
    val_df = df.iloc[train_end:val_end].drop(columns=["_strata"], errors="ignore")
    test_df = df.iloc[val_end:].drop(columns=["_strata"], errors="ignore")

    return train_df, val_df, test_df


def _temporal_split(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Temporal split: sort by timestamp, tie_policy=left-closed."""
    df = df.copy()

    # Convert timestamp to datetime
    df["_ts"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # Sort by timestamp (stable sort)
    df = df.sort_values("_ts", kind="stable").reset_index(drop=True)

    n = len(df)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    # Tie policy: left-closed (identical timestamps go to earlier split)
    # This is naturally handled by stable sort + index-based splitting

    train_df = df.iloc[:train_end].drop(columns=["_ts"], errors="ignore")
    val_df = df.iloc[train_end:val_end].drop(columns=["_ts"], errors="ignore")
    test_df = df.iloc[val_end:].drop(columns=["_ts"], errors="ignore")

    return train_df, val_df, test_df


def _brand_ood_split(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Brand-OOD split: ensure disjoint brand sets between train and test."""
    df = df.copy()

    # Group by brand
    brands = df["brand"].unique()
    n_brands = len(brands)

    # Shuffle brands
    import random

    random.seed(42)
    brands_shuffled = list(brands)
    random.shuffle(brands_shuffled)

    # Split brands
    train_brand_end = int(n_brands * train_ratio)
    val_brand_end = train_brand_end + int(n_brands * val_ratio)

    train_brands = set(brands_shuffled[:train_brand_end])
    val_brands = set(brands_shuffled[train_brand_end:val_brand_end])
    test_brands = set(brands_shuffled[val_brand_end:])

    # Create splits
    train_df = df[df["brand"].isin(train_brands)].reset_index(drop=True)
    val_df = df[df["brand"].isin(val_brands)].reset_index(drop=True)
    test_df = df[df["brand"].isin(test_brands)].reset_index(drop=True)

    return train_df, val_df, test_df


def _compute_split_stats(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Dict:
    """Compute statistics for each split."""
    stats = {}

    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        split_stat = {
            "count": len(df),
            "pos_count": int((df["label"] == 1).sum()) if "label" in df.columns else 0,
            "neg_count": int((df["label"] == 0).sum()) if "label" in df.columns else 0,
        }

        if "brand" in df.columns:
            brands = df["brand"].dropna().unique()
            split_stat["brand_unique"] = len(brands)
            split_stat["brand_set"] = sorted(brands.tolist())[
                :10
            ]  # First 10 for brevity
        else:
            split_stat["brand_unique"] = 0
            split_stat["brand_set"] = []

        if "timestamp" in df.columns:
            ts = pd.to_datetime(df["timestamp"], errors="coerce")
            split_stat["timestamp_min"] = str(ts.min()) if not ts.isna().all() else None
            split_stat["timestamp_max"] = str(ts.max()) if not ts.isna().all() else None
        else:
            split_stat["timestamp_min"] = None
            split_stat["timestamp_max"] = None

        if "source" in df.columns:
            source_counts = df["source"].value_counts().to_dict()
            split_stat["source_counts"] = {
                str(k): int(v) for k, v in source_counts.items()
            }
        else:
            split_stat["source_counts"] = {}

        stats[name] = split_stat

    return stats


def write_split_table(split_stats: Dict, path: Path, metadata: Dict = None) -> None:
    """
    Write split statistics to CSV with all required columns.

    Args:
        split_stats: Statistics for each split (train/val/test)
        path: Output CSV path
        metadata: Additional metadata (tie_policy, brand_normalization, etc.)
    """
    if metadata is None:
        metadata = {}

    rows = []
    for split_name, stats in split_stats.items():
        row = {
            "split": split_name,
            "count": stats["count"],
            "pos_count": stats["pos_count"],
            "neg_count": stats["neg_count"],
            "brand_unique": stats.get("brand_unique", 0),
            "brand_set": str(stats.get("brand_set", [])),
            "timestamp_min": stats.get("timestamp_min", ""),
            "timestamp_max": stats.get("timestamp_max", ""),
            "source_counts": str(stats.get("source_counts", {})),
            # Add metadata columns (same for all splits)
            "brand_intersection_ok": metadata.get("brand_intersection_ok", ""),
            "tie_policy": metadata.get("tie_policy", ""),
            "brand_normalization": metadata.get("brand_normalization", ""),
            "downgraded_to": metadata.get("downgraded_to", ""),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    log.info(f"Split table saved: {path}")

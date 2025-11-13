#!/usr/bin/env python3
"""Quick script to check S3 experiment results"""
import json
from pathlib import Path

exp_dir = Path("experiments/s3_iid_fixed_20251114_013853")

print("=" * 70)
print("S3 Experiment Results Check")
print("=" * 70)
print()

# Check metrics_final.json
metrics_file = exp_dir / "results" / "metrics_final.json"
if metrics_file.exists():
    with open(metrics_file) as f:
        data = json.load(f)

    print("[1] Final Metrics:")
    metrics = data.get("metrics", {})
    for key, value in metrics.items():
        if "alpha" in key or "fusion" in key:
            print(f"  {key}: {value}")
    print()
else:
    print("[1] metrics_final.json not found")
    print()

# Check if predictions_test.csv exists
pred_file = exp_dir / "results" / "predictions_test.csv"
if pred_file.exists():
    import pandas as pd

    df = pd.read_csv(pred_file)

    print("[2] Predictions Analysis:")

    if "alpha_url" in df.columns:
        print(
            f"  Alpha URL:    mean={df['alpha_url'].mean():.4f}, std={df['alpha_url'].std():.4f}"
        )
    if "alpha_html" in df.columns:
        print(
            f"  Alpha HTML:   mean={df['alpha_html'].mean():.4f}, std={df['alpha_html'].std():.4f}"
        )
    if "alpha_visual" in df.columns:
        print(
            f"  Alpha Visual: mean={df['alpha_visual'].mean():.4f}, std={df['alpha_visual'].std():.4f}"
        )
    print()

    if "brand_vis" in df.columns:
        non_empty = df["brand_vis"].notna() & (df["brand_vis"] != "")
        print(
            f"  Brand Visual: {non_empty.sum()}/{len(df)} ({non_empty.mean():.1%}) non-empty"
        )

    if "c_visual" in df.columns:
        valid_c = df["c_visual"].notna() & (~df["c_visual"].isna())
        print(f"  C Visual:     {valid_c.sum()}/{len(df)} ({valid_c.mean():.1%}) valid")
        if valid_c.any():
            print(
                f"                min={df.loc[valid_c, 'c_visual'].min():.3f}, max={df.loc[valid_c, 'c_visual'].max():.3f}"
            )
    print()
else:
    print("[2] predictions_test.csv not found")
    print()

# Check SUMMARY.md
summary_file = exp_dir / "SUMMARY.md"
if summary_file.exists():
    with open(summary_file) as f:
        content = f.read()
    if "alpha" in content.lower() or "fusion" in content.lower():
        print("[3] Summary mentions fusion/alpha")
    print()

print("=" * 70)
print("For full logs, check:")
print(f"  {exp_dir / 'logs'}")
print("  outputs/2025-11-14/01-38-53/")
print("=" * 70)

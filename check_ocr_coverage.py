#!/usr/bin/env python3
"""
Check OCR coverage in predictions_test.csv
"""
import sys
from pathlib import Path
import pandas as pd
import glob


def check_ocr_coverage(predictions_csv: str):
    """Check brand_vis and c_visual coverage in predictions"""
    if not Path(predictions_csv).exists():
        print(f"ERROR: File not found: {predictions_csv}")
        return False

    df = pd.read_csv(predictions_csv)
    total = len(df)

    print("=" * 70)
    print("OCR Coverage Analysis")
    print("=" * 70)
    print(f"\nTotal samples: {total}")
    print()

    # Brand extraction rates
    print("Brand Extraction Rates:")
    for col in ["brand_url", "brand_html", "brand_vis"]:
        if col in df.columns:
            non_empty = (df[col].notna() & (df[col] != "")).sum()
            rate = non_empty / total * 100
            status = "✓" if rate > 50 else "⚠" if rate > 10 else "✗"
            print(f"  {status} {col:15s}: {non_empty:4d}/{total} ({rate:5.1f}%)")
    print()

    # Consistency scores
    print("Consistency Score Validity:")
    for col in ["c_url", "c_html", "c_visual"]:
        if col in df.columns:
            valid = df[col].notna() & (~df[col].isna())
            valid_count = valid.sum()
            rate = valid_count / total * 100

            if valid_count > 0:
                c_values = df.loc[valid, col]
                mean_val = c_values.mean()
                min_val = c_values.min()
                max_val = c_values.max()
                status = "✓" if rate > 50 else "⚠" if rate > 10 else "✗"
                print(
                    f"  {status} {col:15s}: {valid_count:4d}/{total} ({rate:5.1f}%) "
                    f"[{min_val:.3f}, {max_val:.3f}], mean={mean_val:.3f}"
                )
            else:
                print(f"  ✗ {col:15s}: ALL NaN")
    print()

    # Reliability scores
    print("Reliability Score Validity:")
    for col in ["r_url", "r_html", "r_img"]:
        if col in df.columns:
            valid = df[col].notna() & (~df[col].isna())
            valid_count = valid.sum()
            rate = valid_count / total * 100

            if valid_count > 0:
                r_values = df.loc[valid, col]
                mean_val = r_values.mean()
                status = "✓" if rate > 50 else "⚠" if rate > 10 else "✗"
                print(
                    f"  {status} {col:15s}: {valid_count:4d}/{total} ({rate:5.1f}%), mean={mean_val:.3f}"
                )
            else:
                print(f"  ✗ {col:15s}: ALL NaN or empty")
    print()

    # Alpha weights
    print("Fusion Weights (Alpha):")
    alpha_cols = [c for c in df.columns if "alpha" in c]
    if alpha_cols:
        for col in sorted(alpha_cols):
            if col in df.columns:
                valid = df[col].notna() & (~df[col].isna())
                if valid.any():
                    mean_val = df.loc[valid, col].mean()
                    print(f"  {col:20s}: mean={mean_val:.6f}")
                else:
                    print(f"  {col:20s}: ALL NaN")
    else:
        print("  No alpha columns found")
    print()

    # Diagnosis
    print("=" * 70)
    print("Diagnosis:")
    print("=" * 70)

    # Check brand_vis
    if "brand_vis" in df.columns:
        brand_vis_rate = (
            (df["brand_vis"].notna() & (df["brand_vis"] != "")).sum() / total * 100
        )
        if brand_vis_rate == 0:
            print("✗ brand_vis is 0% - OCR not extracting brands")
            print("  Possible reasons:")
            print("    1. image_path not being passed to C-Module")
            print("    2. Tesseract not found or not working")
            print("    3. Image files don't exist at the paths")
        elif brand_vis_rate < 30:
            print(f"⚠ brand_vis is only {brand_vis_rate:.1f}% - OCR partially working")
            print("  Consider checking image quality or OCR configuration")
        else:
            print(f"✓ brand_vis is {brand_vis_rate:.1f}% - OCR working well")

    # Check r_img
    if "r_img" in df.columns:
        r_img_valid = (df["r_img"].notna() & (~df["r_img"].isna())).sum()
        if r_img_valid == 0:
            print("✗ r_img is ALL NaN - MC Dropout not generating visual reliability")
            print("  Possible reasons:")
            print("    1. Visual branch has no Dropout layers")
            print("    2. MC Dropout not being executed for visual")
            print("    3. umodule not enabled")
        else:
            print(f"✓ r_img is valid for {r_img_valid}/{total} samples")

    # Check c_visual
    if "c_visual" in df.columns:
        c_visual_valid = (df["c_visual"].notna() & (~df["c_visual"].isna())).sum()
        if c_visual_valid == 0:
            print("✗ c_visual is ALL NaN - consistency not being computed")
        elif c_visual_valid < total * 0.3:
            print(
                f"⚠ c_visual only valid for {c_visual_valid}/{total} - limited brand extraction"
            )
        else:
            print(f"✓ c_visual valid for {c_visual_valid}/{total}")

    # Check alpha_visual
    if "alpha_visual" in df.columns:
        alpha_visual_valid = df["alpha_visual"].notna() & (~df["alpha_visual"].isna())
        if alpha_visual_valid.any():
            mean_alpha = df.loc[alpha_visual_valid, "alpha_visual"].mean()
            if mean_alpha < 0.001:
                print("✗ alpha_visual ≈ 0 - visual modality excluded from fusion")
                print("  Need BOTH r_img and c_visual for visual to participate")
            else:
                print(
                    f"✓ alpha_visual = {mean_alpha:.4f} - visual participating in fusion!"
                )
        else:
            print("✗ alpha_visual is ALL NaN - fusion may have failed completely")

    print("=" * 70)

    return True


if __name__ == "__main__":
    # Find latest S3 experiment
    exp_dirs = glob.glob("experiments/s3_*_fixed_*")
    if not exp_dirs:
        print("No S3 experiments found")
        sys.exit(1)

    latest = max(exp_dirs, key=lambda x: x.split("_")[-1])
    predictions_file = f"{latest}/artifacts/predictions_test.csv"

    print(f"Checking: {predictions_file}")
    print()

    if not check_ocr_coverage(predictions_file):
        sys.exit(1)

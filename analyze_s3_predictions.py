#!/usr/bin/env python3
import pandas as pd

df = pd.read_csv(
    "experiments/s3_iid_fixed_20251114_013853/artifacts/predictions_test.csv"
)

print("=" * 70)
print("S3 IID Experiment Analysis")
print("=" * 70)
print()

print(f"Total samples: {len(df)}")
print()

# Check alpha weights
alpha_cols = [c for c in df.columns if "alpha" in c]
if alpha_cols:
    print("Alpha Weights:")
    for c in sorted(alpha_cols):
        print(f"  {c:20s}: mean={df[c].mean():.6f}, std={df[c].std():.6f}")
    print()
else:
    print("WARNING: No alpha columns found!")
    print()

# Check brand extraction
brand_cols = [c for c in df.columns if "brand" in c]
if brand_cols:
    print("Brand Extraction:")
    for c in sorted(brand_cols):
        non_empty = (df[c].notna() & (df[c] != "")).sum()
        pct = non_empty / len(df) * 100
        print(f"  {c:20s}: {non_empty:3d}/{len(df)} ({pct:5.1f}%)")
    print()

# Check c scores
c_cols = [c for c in df.columns if c.startswith("c_") and c not in ["c_mean"]]
if c_cols:
    print("Consistency Scores:")
    for c in sorted(c_cols):
        valid = df[c].notna() & (~df[c].isna())
        if valid.any():
            print(
                f"  {c:20s}: min={df.loc[valid, c].min():.3f}, max={df.loc[valid, c].max():.3f}, mean={df.loc[valid, c].mean():.3f}"
            )
        else:
            print(f"  {c:20s}: ALL NaN")
    print()

print("=" * 70)
print("Summary:")
if alpha_cols:
    has_visual = any("visual" in c and df[c].mean() > 0.001 for c in alpha_cols)
    if has_visual:
        print("  [SUCCESS] Visual modality is participating (alpha_visual > 0)!")
    else:
        print("  [WARNING] Visual modality still excluded (alpha_visual = 0)")
else:
    print("  [WARNING] No fusion weights found - may have fallen back to LateAvg")

if brand_cols:
    brand_vis_col = [c for c in brand_cols if "vis" in c]
    if brand_vis_col:
        non_empty = (df[brand_vis_col[0]].notna() & (df[brand_vis_col[0]] != "")).sum()
        if non_empty > 0:
            print(f"  [SUCCESS] Visual brand extraction working ({non_empty} samples)!")
        else:
            print("  [WARNING] Visual brand extraction still at 0%")

print("=" * 70)

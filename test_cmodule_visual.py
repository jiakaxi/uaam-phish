#!/usr/bin/env python3
"""Test C-Module visual brand extraction"""
import sys
import pandas as pd
from pathlib import Path

# Import C-Module
from src.modules.c_module import CModule

print("=" * 70)
print("C-Module Visual Brand Extraction Test")
print("=" * 70)
print()

# Initialize C-Module
print("[1/4] Initializing C-Module...")
try:
    c_module = CModule(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        use_ocr=True,
        thresh=0.60,
        brand_lexicon_path="resources/brand_lexicon.txt",
    )
    print(f"  OK - use_ocr={c_module.use_ocr}")
except Exception as e:
    print(f"  FAIL: {e}")
    sys.exit(1)

# Load test data
print("\n[2/4] Loading test data...")
test_csv = "workspace/data/splits/iid/test_cached.csv"
if not Path(test_csv).exists():
    print(f"  ERROR: {test_csv} not found")
    sys.exit(1)

df = pd.read_csv(test_csv)
print(f"  Loaded {len(df)} samples")

# Find image path column
img_col = None
for col in ["img_path", "img_path_corrupt", "image_path"]:
    if col in df.columns:
        img_col = col
        break

if not img_col:
    print("  ERROR: No image path column found!")
    sys.exit(1)

print(f"  Using column: {img_col}")

# Test visual brand extraction on first 5 samples
print("\n[3/4] Testing visual brand extraction...")
samples_to_test = min(5, len(df))

for i in range(samples_to_test):
    row = df.iloc[i]
    sample_id = row.get("sample_id", row.get("id", i))
    img_path = row[img_col]

    print(f"\n  Sample {i+1}/{samples_to_test} (id={sample_id}):")
    print(f"    Path: {img_path}")

    # Check if file exists
    if pd.isna(img_path) or not str(img_path).strip():
        print("    Result: Empty path")
        continue

    path = Path(img_path)
    if not path.exists():
        print("    Result: File NOT found")
        continue

    print("    File exists: YES")

    # Call C-Module
    try:
        brand, meta = c_module._brand_from_visual(str(path))
        print(f"    Brand: '{brand}'")
        print(f"    Method: {meta.get('method', 'N/A')}")
        print(f"    Reason: {meta.get('reason', 'N/A')}")
        if brand:
            print(f"    *** SUCCESS! Extracted: {brand}")
    except Exception as e:
        print(f"    ERROR: {e}")

# Summary
print("\n[4/4] Summary")
print("=" * 70)
print("If all brands are None, possible issues:")
print("  1. Tesseract not found (check path in c_module.py)")
print("  2. OCR returns empty text")
print("  3. Brand lexicon doesn't match OCR output")
print("=" * 70)

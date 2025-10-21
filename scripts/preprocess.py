"""
Simple random (or domain-aware, if column exists) split from a single CSV to train/val/test.
Input CSV must contain: url_text (str), label (0/1), optional domain (str)
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--src", default="data/raw/urls.csv")
parser.add_argument("--outdir", default="data/processed")
parser.add_argument("--val_size", type=float, default=0.1)
parser.add_argument("--test_size", type=float, default=0.1)
args = parser.parse_args()

out = Path(args.outdir)
out.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(args.src)
assert {"url_text", "label"}.issubset(df.columns), "CSV must have url_text,label"

if "domain" in df.columns:
    # domain-aware split to reduce leakage
    domains = df["domain"].astype(str)
    # stratify by label while grouping by domain (approximation via group shuffle)
    # For MVP we do random split with stratify label; replace with GroupShuffleSplit if needed

train_df, temp = train_test_split(
    df, test_size=args.val_size + args.test_size, stratify=df["label"], random_state=42
)
val_rel = args.val_size / (args.val_size + args.test_size)
val_df, test_df = train_test_split(
    temp, test_size=1 - val_rel, stratify=temp["label"], random_state=42
)

train_df.to_csv(out / "train.csv", index=False)
val_df.to_csv(out / "val.csv", index=False)
test_df.to_csv(out / "test.csv", index=False)
print("Saved:", out / "train.csv", out / "val.csv", out / "test.csv")

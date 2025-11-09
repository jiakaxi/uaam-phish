#!/usr/bin/env python
"""
URL corruption utility (text-only) for S0 experiments.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List

import pandas as pd

HOMOGLYPHS = {
    "a": ["@", "4"],
    "o": ["0"],
    "e": ["3"],
    "i": ["1", "!"],
    "l": ["1"],
    "s": ["5", "$"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate corrupted URL text CSVs.")
    parser.add_argument("--in", dest="input_csv", required=True, help="Base CSV path.")
    parser.add_argument(
        "--out",
        dest="output_dir",
        required=True,
        help="Directory to store corrupted CSV files.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--levels",
        nargs="+",
        default=["L", "M", "H"],
        help="Corruption levels to synthesize.",
    )
    return parser.parse_args()


def corrupt_low(url: str) -> str:
    if "?" in url:
        return f"{url}&utm_ref=s0"
    return f"{url}?ref=secure-update"


def corrupt_mid(url: str) -> str:
    return url.replace(".", "-").replace("/", "//", 1)


def corrupt_high(url: str, rng: random.Random) -> str:
    chars = list(url)
    for idx, ch in enumerate(chars):
        glyphs = HOMOGLYPHS.get(ch.lower())
        if glyphs and rng.random() < 0.4:
            replacement = rng.choice(glyphs)
            chars[idx] = replacement
    if rng.random() < 0.5:
        chars.insert(rng.randint(0, len(chars)), rng.choice(["-", "_", "~"]))
    return "".join(chars)


def main() -> None:
    args = parse_args()
    base_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    if not base_csv.exists():
        raise FileNotFoundError(base_csv)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(base_csv)
    if "url_text" not in df.columns:
        raise ValueError("CSV must contain url_text column.")

    rng = random.Random(args.seed)
    outputs: List[Path] = []

    for level in args.levels:
        level = level.upper()
        records = []
        for _, row in df.iterrows():
            url = str(row.get("url_text", ""))
            if level == "L":
                corrupted = corrupt_low(url)
            elif level == "M":
                corrupted = corrupt_mid(url)
            else:
                corrupted = corrupt_high(url, rng)
            record = row.to_dict()
            record["url_text_corrupt"] = corrupted
            record["corruption_level"] = level
            records.append(record)

        level_df = pd.DataFrame(records)
        csv_path = output_dir / f"test_corrupt_url_{level}.csv"
        level_df.to_csv(csv_path, index=False)
        outputs.append(csv_path)
        print(f"[corrupt_url] Level {level}: saved {len(level_df)} rows to {csv_path}")

    print("Corrupted URL CSV files:", [str(p) for p in outputs])


if __name__ == "__main__":
    main()

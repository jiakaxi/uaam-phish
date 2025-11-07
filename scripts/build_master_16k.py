#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build master CSV of 16k samples from two folders (phish / benign).
Production-ready with full validation, brand ratio enforcement, and quality reporting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import tldextract
from dateutil import parser as dparser
from PIL import Image
from tqdm import tqdm

# Handle Windows console encoding
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        # Python < 3.7
        import codecs

        sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "strict")
        sys.stderr = codecs.getwriter("utf-8")(sys.stderr.buffer, "strict")


# =====================================================================
# Helper Functions - File Discovery
# =====================================================================


def find_html_file(folder: Path) -> Optional[Path]:
    """Find HTML file with priority: html.html > html.txt > *.html"""
    for name in ("html.html", "html.txt", "page.html", "index.html"):
        p = folder / name
        if p.exists():
            return p
    # Fallback: any .html or .htm
    for p in folder.glob("*.htm*"):
        return p
    return None


def find_image_file(folder: Path) -> Optional[Path]:
    """Find image file with priority: shot.png > screenshot.png > *.png > *.jpg"""
    for name in (
        "shot.png",
        "screenshot.png",
        "shot.jpg",
        "screenshot.jpg",
        "page.png",
    ):
        p = folder / name
        if p.exists():
            return p
    # Fallback: any png/jpg
    for p in folder.glob("*.png"):
        return p
    for p in folder.glob("*.jpg"):
        return p
    for p in folder.glob("*.jpeg"):
        return p
    return None


# =====================================================================
# Helper Functions - Info Parsing
# =====================================================================


def read_info_txt(path: Path) -> Dict[str, Any]:
    """Parse info.txt - supports JSON, Python dict, and key:value formats"""
    if not path.exists():
        return {}

    data = {}
    try:
        text = path.read_text(encoding="utf-8", errors="ignore").strip()

        # Try JSON first
        try:
            data = json.loads(text)
            return data
        except json.JSONDecodeError:
            pass

        # Try Python dict (ast.literal_eval)
        try:
            import ast

            data = ast.literal_eval(text)
            if isinstance(data, dict):
                return data
        except (ValueError, SyntaxError):
            pass

        # Try key: value format
        for line in text.splitlines():
            line = line.strip()
            if not line or ":" not in line:
                continue
            k, v = line.split(":", 1)
            data[k.strip().lower()] = v.strip()
    except Exception:
        pass

    return data


def find_url(folder: Path, info: Dict[str, Any]) -> Optional[str]:
    """Extract URL from url.txt or info dict"""
    # Try url.txt file
    url_file = folder / "url.txt"
    if url_file.exists():
        try:
            return url_file.read_text(encoding="utf-8", errors="ignore").strip()
        except Exception:
            pass

    # Try info dict
    for key in ("url", "url_text", "request_url"):
        if key in info:
            return str(info[key]).strip()

    return None


# =====================================================================
# Helper Functions - Normalization
# =====================================================================


def normalize_url(url: Optional[str]) -> Optional[str]:
    """Normalize URL: remove trailing slash, utm params, and fragment"""
    if not url:
        return url

    # Remove fragment (#...)
    if "#" in url:
        url = url.split("#")[0]

    # Remove trailing slash
    url = url.rstrip("/")

    # Remove utm parameters
    if "?" in url:
        base, params = url.split("?", 1)
        clean_params = "&".join(
            [p for p in params.split("&") if not p.startswith("utm_")]
        )
        url = f"{base}?{clean_params}" if clean_params else base

    return url


def norm_domain(
    url: Optional[str], info_domain: Optional[str] = None
) -> Tuple[Optional[str], str]:
    """Extract registered domain using tldextract. Returns (domain, source)"""
    if info_domain:
        return info_domain.strip().lower(), "info"

    if not url:
        return None, "failed"

    try:
        ext = tldextract.extract(url)
        if ext.registered_domain:
            return ext.registered_domain.lower(), "url"
    except Exception:
        pass

    # Fallback: simple regex
    try:
        m = re.search(r"://([^/]+)", url)
        if m:
            return m.group(1).lower(), "url"
    except Exception:
        pass

    return None, "failed"


def standardize_brand(
    brand_raw: Any, brand_map: Optional[Dict[str, str]] = None
) -> Optional[str]:
    """Standardize brand name using mapping table and normalization"""
    if not brand_raw or pd.isna(brand_raw):
        return None

    brand = str(brand_raw).strip().lower()

    # Use mapping table if provided
    if brand_map and brand in brand_map:
        return brand_map[brand]

    # Default normalization
    brand = re.sub(r"[^\w\s]", "", brand)  # Remove special chars
    brand = re.sub(r"\s+", "", brand)  # Remove spaces

    return brand if brand else None


def parse_timestamp(ts_str: Any) -> Optional[str]:
    """Parse timestamp to ISO format"""
    if not ts_str or pd.isna(ts_str):
        return None

    ts_str = str(ts_str).strip()

    try:
        # Handle epoch timestamps
        if re.fullmatch(r"\d{10,13}", ts_str):
            t = int(ts_str)
            if len(ts_str) == 13:
                t = t / 1000
            return datetime.utcfromtimestamp(t).isoformat() + "Z"

        # Parse ISO or other formats
        dt = dparser.parse(ts_str)
        return dt.isoformat() + "Z"
    except Exception:
        return None


def get_file_mtime(folder: Path) -> Optional[str]:
    """Get folder modification time as fallback timestamp"""
    try:
        mtime = folder.stat().st_mtime
        return datetime.utcfromtimestamp(mtime).isoformat() + "Z"
    except Exception:
        return None


# =====================================================================
# Helper Functions - File Validation
# =====================================================================


def validate_file_exists_and_size(
    path: Optional[Path], min_size: int = 0
) -> Tuple[bool, str]:
    """Validate file exists and meets size requirement"""
    if not path:
        return False, "missing"

    if not path.exists():
        return False, "not_found"

    try:
        size = path.stat().st_size
        if size <= min_size:
            return False, "empty" if size == 0 else "too_small"
        return True, "valid"
    except Exception:
        return False, "stat_error"


def validate_html_quality(html_path: Path, min_size: int = 200) -> Tuple[bool, str]:
    """Validate HTML file quality: size and tag count"""
    valid, reason = validate_file_exists_and_size(html_path, min_size)
    if not valid:
        return False, f"html_{reason}"

    try:
        # Check tag count
        html_content = html_path.read_text(encoding="utf-8", errors="ignore")
        tag_count = len(re.findall(r"<[a-zA-Z]+", html_content))
        if tag_count < 5:
            return False, "html_insufficient_tags"
        return True, "valid"
    except Exception:
        return False, "html_read_error"


def validate_image_quality(img_path: Path) -> Tuple[bool, str]:
    """Validate image file using Pillow"""
    valid, reason = validate_file_exists_and_size(img_path, 0)
    if not valid:
        return False, f"img_{reason}"

    try:
        with Image.open(img_path) as img:
            img.verify()
        return True, "valid"
    except Exception:
        return False, "img_corrupt"


# =====================================================================
# Helper Functions - Hashing
# =====================================================================


def compute_sha1(path: Path) -> Optional[str]:
    """Compute SHA1 hash of file"""
    try:
        return hashlib.sha1(path.read_bytes()).hexdigest()
    except Exception:
        return None


def compute_hashes_parallel(
    paths: List[Path], max_workers: int = 4
) -> List[Optional[str]]:
    """Compute SHA1 hashes in parallel using ThreadPoolExecutor"""
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(compute_sha1, paths))
    return results


# =====================================================================
# Main Scanning Function
# =====================================================================


def scan_folder(
    root: Path,
    label: int,
    source_tag: str,
    brand_map: Optional[Dict[str, str]] = None,
    min_html_size: int = 200,
    use_fallback_timestamp: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Scan folder and extract metadata with validation.
    Returns: (samples_list, dropped_reasons_count)
    """
    samples = []
    dropped_reasons = defaultdict(int)

    # Get all child folders
    children = [p for p in root.iterdir() if p.is_dir()]

    debug_first = True  # Debug first sample
    for child in tqdm(children, desc=f"Scanning {source_tag}"):
        stem = child.name

        # Read info.txt
        info = read_info_txt(child / "info.txt")

        # Extract URL
        url_text = find_url(child, info)
        if not url_text:
            if debug_first:
                print(f"\n[DEBUG] First sample dropped: missing_url - {child.name}")
                print(f"  info keys: {list(info.keys())}")
                debug_first = False
            dropped_reasons["missing_url"] += 1
            continue

        url_text = normalize_url(url_text)

        # Extract domain
        domain, domain_source = norm_domain(
            url_text, info.get("domain") or info.get("host")
        )
        if not domain:
            if debug_first:
                print(
                    f"\n[DEBUG] First sample dropped: domain_extraction_failed - {child.name}"
                )
                print(f"  url_text: {url_text}")
                print(f"  info domain/host: {info.get('domain')}, {info.get('host')}")
                debug_first = False
            dropped_reasons["domain_extraction_failed"] += 1
            continue

        # Extract and standardize brand
        brand_raw = info.get("brand") or info.get("vendor")
        brand = standardize_brand(brand_raw, brand_map)

        # Extract timestamp
        timestamp = parse_timestamp(
            info.get("isotime") or info.get("timestamp") or info.get("time")
        )
        timestamp_source = "info"
        if not timestamp and use_fallback_timestamp:
            timestamp = get_file_mtime(child)
            timestamp_source = "mtime"

        # Find files
        html_path = find_html_file(child)
        img_path = find_image_file(child)

        # Validate HTML
        if not html_path:
            if debug_first:
                print(f"\n[DEBUG] First sample dropped: missing_html - {child.name}")
                debug_first = False
            dropped_reasons["missing_html"] += 1
            continue

        html_valid, html_reason = validate_html_quality(html_path, min_html_size)
        if not html_valid:
            if debug_first:
                print(f"\n[DEBUG] First sample dropped: {html_reason} - {child.name}")
                print(f"  html_path: {html_path}")
                print(
                    f"  exists: {html_path.exists()}, size: {html_path.stat().st_size if html_path.exists() else 'N/A'}"
                )
                debug_first = False
            dropped_reasons[html_reason] += 1
            continue

        # Validate image
        if not img_path:
            if debug_first:
                print(f"\n[DEBUG] First sample dropped: missing_img - {child.name}")
                debug_first = False
            dropped_reasons["missing_img"] += 1
            continue

        img_valid, img_reason = validate_image_quality(img_path)
        if not img_valid:
            if debug_first:
                print(f"\n[DEBUG] First sample dropped: {img_reason} - {child.name}")
                print(f"  img_path: {img_path}")
                print(
                    f"  exists: {img_path.exists()}, size: {img_path.stat().st_size if img_path.exists() else 'N/A'}"
                )
                debug_first = False
            dropped_reasons[img_reason] += 1
            continue

        # Create sample
        sample_id = f"{source_tag}__{stem}"
        samples.append(
            {
                "id": sample_id,
                "stem": stem,
                "label": label,
                "url_text": url_text,
                "html_path": str(html_path.resolve()),
                "img_path": str(img_path.resolve()),
                "domain": domain,
                "domain_source": domain_source,
                "brand": brand,
                "brand_raw": brand_raw,
                "timestamp": timestamp,
                "timestamp_source": timestamp_source,
                "source": source_tag,
                "folder": str(child.resolve()),
            }
        )

    return samples, dict(dropped_reasons)


# =====================================================================
# Brand Sampling Functions
# =====================================================================


def enforce_brand_ratio(
    df: pd.DataFrame, target_n: int, max_ratio: float = 0.30, seed: int = 42
) -> pd.DataFrame:
    """Enforce brand ratio cap: no single brand > max_ratio of total"""
    brand_counts = df["brand"].value_counts()
    max_per_brand = int(max_ratio * target_n)

    over_brands = brand_counts[brand_counts > max_per_brand].index.tolist()
    if not over_brands:
        return df

    print(
        f"  [WARNING] Brand ratio enforcement: {len(over_brands)} brands exceed {max_ratio:.0%} cap"
    )
    print(f"            Capping each brand to {max_per_brand} samples")

    # Cap each brand
    capped = []
    for b in df["brand"].unique():
        b_df = df[df["brand"] == b]
        cap = min(len(b_df), max_per_brand)
        capped.append(b_df.sample(n=cap, random_state=seed))

    return pd.concat(capped).sample(frac=1.0, random_state=seed)


def sample_with_brand_controls(
    df: pd.DataFrame,
    n_samples: int,
    min_per_brand: int = 50,
    brand_cap: int = 500,
    max_ratio: float = 0.30,
    seed: int = 42,
) -> pd.DataFrame:
    """Sample with brand diversity controls: min samples, cap per brand, and max ratio"""
    initial_len = len(df)

    # Filter low-frequency brands
    brand_counts = df["brand"].value_counts()
    print("\n  Brand distribution before filtering:")
    print(f"    Total brands: {len(brand_counts)}")
    print(
        f"    Brands with >= {min_per_brand} samples: {(brand_counts >= min_per_brand).sum()}"
    )

    valid_brands = brand_counts[brand_counts >= min_per_brand].index
    df_filtered = df[df["brand"].isin(valid_brands)]

    if len(df_filtered) < n_samples:
        # Auto-downgrade min_per_brand
        new_min = 20
        print(f"  [WARNING] Insufficient samples ({len(df_filtered)} < {n_samples})")
        print(
            f"            Auto-downgrading min_per_brand from {min_per_brand} to {new_min}"
        )
        valid_brands = brand_counts[brand_counts >= new_min].index
        df_filtered = df[df["brand"].isin(valid_brands)]
        min_per_brand = new_min

    # Apply brand cap
    df_capped = df_filtered.groupby("brand", group_keys=False).head(brand_cap)

    # Apply brand ratio enforcement
    df_ratio_enforced = enforce_brand_ratio(df_capped, n_samples, max_ratio, seed)

    # Sample exactly n_samples
    if len(df_ratio_enforced) >= n_samples:
        result = df_ratio_enforced.sample(n=n_samples, random_state=seed)
    else:
        # Need to backfill from remaining samples
        remaining = n_samples - len(df_ratio_enforced)
        extra = df_filtered[~df_filtered.index.isin(df_ratio_enforced.index)]
        if len(extra) > 0:
            extra_sample = extra.sample(n=min(remaining, len(extra)), random_state=seed)
            result = pd.concat([df_ratio_enforced, extra_sample]).sample(
                frac=1.0, random_state=seed
            )
        else:
            result = df_ratio_enforced

    print(f"\n  [OK] Sampled {len(result)} from {initial_len} candidates")
    print(f"       Brand count: {result['brand'].nunique()}")
    print(
        f"       Max brand ratio: {result['brand'].value_counts().iloc[0] / len(result):.2%}"
    )

    return result


# =====================================================================
# Quality Report Generation
# =====================================================================


def get_timestamp_histogram(df: pd.DataFrame) -> Dict[str, int]:
    """Get timestamp histogram by month"""
    df_ts = df[df["timestamp"].notna()].copy()
    if len(df_ts) == 0:
        return {}

    try:
        df_ts["month"] = pd.to_datetime(
            df_ts["timestamp"], errors="coerce"
        ).dt.to_period("M")
        return df_ts["month"].value_counts().sort_index().astype(str).to_dict()
    except Exception:
        return {}


def generate_quality_report(
    df: pd.DataFrame, dropped_reasons: Dict[str, int], args: Any
) -> Dict[str, Any]:
    """Generate comprehensive quality report"""
    report = {
        "total_samples": len(df),
        "phishing_count": int((df["label"] == 1).sum()),
        "legitimate_count": int((df["label"] == 0).sum()),
        # Brand distribution
        "brand_distribution_top10": df["brand"].value_counts().head(10).to_dict(),
        "brand_count": int(df["brand"].nunique()),
        "max_brand_ratio": (
            float(df["brand"].value_counts().iloc[0] / len(df)) if len(df) > 0 else 0
        ),
        # Timestamp coverage
        "timestamp_coverage": float(df["timestamp"].notna().mean()),
        "timestamp_range": {
            "min": (
                str(df["timestamp"].min()) if df["timestamp"].notna().any() else None
            ),
            "max": (
                str(df["timestamp"].max()) if df["timestamp"].notna().any() else None
            ),
        },
        "timestamp_histogram_by_month": get_timestamp_histogram(df),
        "timestamp_sources": (
            df["timestamp_source"].value_counts().to_dict()
            if "timestamp_source" in df.columns
            else {}
        ),
        # Modality completeness
        "modality_completeness": {
            "url": float(df["url_text"].notna().mean()),
            "html": float(df["html_path"].notna().mean()),
            "image": float(df["img_path"].notna().mean()),
        },
        # Domain extraction
        "domain_extraction": (
            df["domain_source"].value_counts().to_dict()
            if "domain_source" in df.columns
            else {}
        ),
        # Dropped reasons
        "dropped_reasons": dropped_reasons,
        "total_dropped": sum(dropped_reasons.values()),
        # Hash info
        "hash_computed": "html_sha1" in df.columns,
        "hash_mode": args.compute_hash if hasattr(args, "compute_hash") else None,
        # Build parameters
        "build_parameters": {
            "min_per_brand": args.min_per_brand,
            "brand_cap": args.brand_cap,
            "max_brand_ratio": args.max_brand_ratio,
            "min_html_size": args.min_html_size,
            "random_seed": args.seed,
        },
        "created_at": datetime.now().isoformat() + "Z",
        "random_seed": args.seed,
    }

    return report


# =====================================================================
# Validation Functions
# =====================================================================


def validate_final_dataset(
    df: pd.DataFrame, expected_count: int, dropped_reasons: Dict[str, int]
) -> bool:
    """Run comprehensive validation assertions"""
    print("\n" + "=" * 70)
    print("[VALIDATION] Running final dataset checks...")
    print("=" * 70)

    issues = []

    # 1. Row count
    try:
        assert len(df) == expected_count, f"Expected {expected_count}, got {len(df)}"
        print("[PASS] 1. Row count check passed")
    except AssertionError as e:
        issues.append(str(e))
        print(f"[FAIL] 1. Row count check FAILED: {e}")

    # 2. Label distribution
    try:
        phish_count = (df["label"] == 1).sum()
        legit_count = (df["label"] == 0).sum()
        assert (
            phish_count == expected_count // 2
        ), f"Phishing count {phish_count} != {expected_count // 2}"
        assert (
            legit_count == expected_count // 2
        ), f"Legitimate count {legit_count} != {expected_count // 2}"
        print(
            f"[PASS] 2. Label distribution check passed (phish={phish_count}, legit={legit_count})"
        )
    except AssertionError as e:
        issues.append(str(e))
        print(f"[FAIL] 2. Label distribution check FAILED: {e}")

    # 3. Split consistency
    try:
        assert (df["split"] == "unsplit").all(), "All split values must be 'unsplit'"
        print("[PASS] 3. Split consistency check passed (all='unsplit')")
    except AssertionError as e:
        issues.append(str(e))
        print(f"[FAIL] 3. Split consistency check FAILED: {e}")

    # 4. Deduplication
    try:
        dup_count = df.duplicated(subset=["url_text", "domain", "brand"]).sum()
        assert dup_count == 0, f"Found {dup_count} duplicates"
        print("[PASS] 4. Deduplication check passed (0 duplicates)")
    except AssertionError as e:
        issues.append(str(e))
        print(f"[FAIL] 4. Deduplication check FAILED: {e}")

    # 5. Brand ratio cap
    try:
        brand_dist = df["brand"].value_counts()
        max_brand_ratio = brand_dist.iloc[0] / len(df)
        assert max_brand_ratio < 0.30, f"Max brand ratio {max_brand_ratio:.2%} >= 30%"
        print(f"[PASS] 5. Brand ratio cap check passed (max={max_brand_ratio:.2%})")
    except AssertionError as e:
        issues.append(str(e))
        print(f"[FAIL] 5. Brand ratio cap check FAILED: {e}")

    # 6. File validation (sample)
    try:
        sample_size = min(50, len(df))
        sample_df = df.sample(sample_size)
        for idx, row in sample_df.iterrows():
            html_p = Path(row["html_path"])
            img_p = Path(row["img_path"])
            assert (
                html_p.exists() and html_p.stat().st_size > 200
            ), f"HTML invalid: {row['id']}"
            assert (
                img_p.exists() and img_p.stat().st_size > 0
            ), f"Image invalid: {row['id']}"
        print(f"[PASS] 6. File validation check passed ({sample_size} samples)")
    except AssertionError as e:
        issues.append(str(e))
        print(f"[FAIL] 6. File validation check FAILED: {e}")

    # 7. Timestamp coverage
    try:
        ts_coverage = df["timestamp"].notna().mean()
        assert ts_coverage >= 0.70, f"Timestamp coverage {ts_coverage:.2%} < 70%"
        print(f"[PASS] 7. Timestamp coverage check passed ({ts_coverage:.2%})")
    except AssertionError as e:
        issues.append(str(e))
        print(f"[FAIL] 7. Timestamp coverage check FAILED: {e}")

    # 8. Domain completeness
    try:
        domain_completeness = df["domain"].notna().mean()
        assert (
            domain_completeness == 1.0
        ), f"Domain completeness {domain_completeness:.2%} < 100%"
        print("[PASS] 8. Domain completeness check passed")
    except AssertionError as e:
        issues.append(str(e))
        print(f"[FAIL] 8. Domain completeness check FAILED: {e}")

    print("=" * 70)
    if not issues:
        print("==> ALL VALIDATIONS PASSED!")
        return True
    else:
        print(f"[WARNING] {len(issues)} VALIDATION(S) FAILED:")
        for i, issue in enumerate(issues, 1):
            print(f"   {i}. {issue}")
        return False


# =====================================================================
# Main Build Function
# =====================================================================


def main():
    parser = argparse.ArgumentParser(description="Build 16k multimodal dataset")
    parser.add_argument("--phish_root", required=True, help="Path to phishing dataset")
    parser.add_argument("--benign_root", required=True, help="Path to benign dataset")
    parser.add_argument("--k_each", type=int, default=8000, help="Samples per class")
    parser.add_argument(
        "--out_csv", default="data/processed/master_16k.csv", help="Output CSV path"
    )
    parser.add_argument(
        "--out_meta",
        default="data/processed/metadata_16k.json",
        help="Output metadata JSON",
    )
    parser.add_argument(
        "--out_selected_ids",
        default="data/processed/selected_ids_16k.json",
        help="Selected IDs JSON",
    )
    parser.add_argument(
        "--out_dropped_reasons",
        default="data/processed/dropped_reasons_16k.json",
        help="Dropped reasons JSON",
    )
    parser.add_argument(
        "--brand_map", default="resources/brand_map.json", help="Brand mapping JSON"
    )
    parser.add_argument(
        "--min_per_brand", type=int, default=50, help="Min samples per brand"
    )
    parser.add_argument(
        "--brand_cap", type=int, default=500, help="Max samples per brand"
    )
    parser.add_argument(
        "--max_brand_ratio", type=float, default=0.30, help="Max brand ratio"
    )
    parser.add_argument(
        "--min_html_size", type=int, default=200, help="Min HTML file size (bytes)"
    )
    parser.add_argument(
        "--compute_hash",
        choices=["none", "html", "all"],
        default="none",
        help="Compute file hashes",
    )
    parser.add_argument(
        "--use_fallback_timestamp",
        action="store_true",
        default=True,
        help="Use file mtime as timestamp fallback",
    )
    parser.add_argument(
        "--validate", action="store_true", default=True, help="Run validation checks"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--log", default=None, help="Log file path")

    args = parser.parse_args()

    # Setup logging
    if not args.log:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.log = f"logs/build_16k_{timestamp}.log"

    # Create output directories
    for path in [
        args.out_csv,
        args.out_meta,
        args.out_selected_ids,
        args.out_dropped_reasons,
        args.log,
    ]:
        Path(path).parent.mkdir(parents=True, exist_ok=True)

    # Load brand map
    brand_map = {}
    if Path(args.brand_map).exists():
        with open(args.brand_map, "r", encoding="utf-8") as f:
            brand_map = json.load(f)
        print(f"[OK] Loaded brand map with {len(brand_map)} entries")

    print("\n" + "=" * 70)
    print("==> Starting 16k Dataset Build")
    print("=" * 70)
    print(f"Phishing root:  {args.phish_root}")
    print(f"Benign root:    {args.benign_root}")
    print(f"Samples/class:  {args.k_each}")
    print(f"Min/brand:      {args.min_per_brand}")
    print(f"Brand cap:      {args.brand_cap}")
    print(f"Max ratio:      {args.max_brand_ratio:.0%}")
    print(f"Random seed:    {args.seed}")
    print("=" * 70)

    # Scan phishing folder
    print("\n[Step 1/6] Scanning phishing samples...")
    phish_samples, phish_dropped = scan_folder(
        Path(args.phish_root),
        label=1,
        source_tag="phish",
        brand_map=brand_map,
        min_html_size=args.min_html_size,
        use_fallback_timestamp=args.use_fallback_timestamp,
    )
    print(f"   Found: {len(phish_samples)} valid samples")
    print(f"   Dropped: {sum(phish_dropped.values())} samples")

    # Scan benign folder
    print("\n[Step 2/6] Scanning benign samples...")
    benign_samples, benign_dropped = scan_folder(
        Path(args.benign_root),
        label=0,
        source_tag="benign",
        brand_map=brand_map,
        min_html_size=args.min_html_size,
        use_fallback_timestamp=args.use_fallback_timestamp,
    )
    print(f"   Found: {len(benign_samples)} valid samples")
    print(f"   Dropped: {sum(benign_dropped.values())} samples")

    # Combine dropped reasons
    all_dropped = defaultdict(int)
    for k, v in phish_dropped.items():
        all_dropped[k] += v
    for k, v in benign_dropped.items():
        all_dropped[k] += v

    # Convert to DataFrame
    print("\n[Step 3/6] Processing and deduplicating...")
    phish_df = pd.DataFrame(phish_samples)
    benign_df = pd.DataFrame(benign_samples)

    # Deduplicate within each class
    phish_len_before = len(phish_df)
    phish_df = phish_df.drop_duplicates(
        subset=["url_text", "domain", "brand"], keep="first"
    )
    phish_dup_count = phish_len_before - len(phish_df)

    benign_len_before = len(benign_df)
    benign_df = benign_df.drop_duplicates(
        subset=["url_text", "domain", "brand"], keep="first"
    )
    benign_dup_count = benign_len_before - len(benign_df)

    all_dropped["duplicate_url_domain_brand"] = phish_dup_count + benign_dup_count

    print(
        f"   Phishing: {phish_len_before} → {len(phish_df)} (removed {phish_dup_count} duplicates)"
    )
    print(
        f"   Benign:   {benign_len_before} → {len(benign_df)} (removed {benign_dup_count} duplicates)"
    )

    # Sample with brand controls
    print("\n[Step 4/6] Sampling with brand controls...")
    print("\n  Phishing samples:")
    phish_sampled = sample_with_brand_controls(
        phish_df,
        args.k_each,
        min_per_brand=args.min_per_brand,
        brand_cap=args.brand_cap,
        max_ratio=args.max_brand_ratio,
        seed=args.seed,
    )

    print("\n  Benign samples:")
    benign_sampled = sample_with_brand_controls(
        benign_df,
        args.k_each,
        min_per_brand=args.min_per_brand,
        brand_cap=args.brand_cap,
        max_ratio=args.max_brand_ratio,
        seed=args.seed,
    )

    # Combine
    df_final = pd.concat([phish_sampled, benign_sampled], ignore_index=True)
    df_final = df_final.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    # Compute hashes if requested
    if args.compute_hash in ["html", "all"]:
        print("\n[Step 5/6] Computing file hashes...")
        print(f"   Mode: {args.compute_hash}")

        html_paths = [Path(p) for p in df_final["html_path"]]
        print("   Computing HTML hashes...")
        df_final["html_sha1"] = compute_hashes_parallel(html_paths, max_workers=4)

        if args.compute_hash == "all":
            img_paths = [Path(p) for p in df_final["img_path"]]
            print("   Computing image hashes...")
            df_final["img_sha1"] = compute_hashes_parallel(img_paths, max_workers=4)
    else:
        print("\n[Step 5/6] Skipping hash computation (use --compute_hash to enable)")

    # Set split to unsplit
    df_final["split"] = "unsplit"

    # Remove temporary columns
    cols_to_keep = [
        "id",
        "stem",
        "label",
        "url_text",
        "html_path",
        "img_path",
        "domain",
        "brand",
        "timestamp",
        "source",
        "split",
    ]
    if "html_sha1" in df_final.columns:
        cols_to_keep.append("html_sha1")
    if "img_sha1" in df_final.columns:
        cols_to_keep.append("img_sha1")

    df_output = df_final[cols_to_keep].copy()

    # Save CSV with explicit dtypes
    print("\n[Step 6/6] Saving outputs...")
    df_output.to_csv(args.out_csv, index=False, encoding="utf-8-sig")
    print(f"   [OK] Saved CSV: {args.out_csv}")

    # Generate and save metadata
    quality_report = generate_quality_report(df_final, dict(all_dropped), args)
    with open(args.out_meta, "w", encoding="utf-8") as f:
        json.dump(quality_report, f, indent=2, ensure_ascii=False)
    print(f"   [OK] Saved metadata: {args.out_meta}")

    # Save selected IDs
    selected_ids = {
        "phishing": df_final[df_final["label"] == 1]["id"].tolist(),
        "legitimate": df_final[df_final["label"] == 0]["id"].tolist(),
    }
    with open(args.out_selected_ids, "w", encoding="utf-8") as f:
        json.dump(selected_ids, f, indent=2, ensure_ascii=False)
    print(f"   [OK] Saved selected IDs: {args.out_selected_ids}")

    # Save dropped reasons
    with open(args.out_dropped_reasons, "w", encoding="utf-8") as f:
        json.dump(dict(all_dropped), f, indent=2, ensure_ascii=False)
    print(f"   [OK] Saved dropped reasons: {args.out_dropped_reasons}")

    # Validate if requested
    if args.validate:
        validate_final_dataset(df_output, args.k_each * 2, dict(all_dropped))

    print("\n" + "=" * 70)
    print("==> Build complete!")
    print("Final stats:")
    print(f"   Total samples: {len(df_output)}")
    print(f"   Phishing: {(df_output['label'] == 1).sum()}")
    print(f"   Legitimate: {(df_output['label'] == 0).sum()}")
    print(f"   Brands: {df_output['brand'].nunique()}")
    print(f"   Timestamp coverage: {df_output['timestamp'].notna().mean():.2%}")
    print("=" * 70)


if __name__ == "__main__":
    main()

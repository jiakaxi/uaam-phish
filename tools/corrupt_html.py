#!/usr/bin/env python
"""
HTML corruption utility for S0 experiments.
"""

from __future__ import annotations

import argparse
import hashlib
import random
import re
from pathlib import Path
from typing import Dict, List

import pandas as pd


SCRIPT_RE = re.compile(r"<script.*?>.*?</script>", re.IGNORECASE | re.DOTALL)
TAG_RE = re.compile(r"<[^>]+>")

# Windows不允许的字符
INVALID_CHARS = re.compile(r'[<>:"/\\|?*`\r\n]')


def sanitize_filename(name: str, max_len: int = 200) -> str:
    """清理文件名，移除Windows不允许的字符，限制长度"""
    # 移除所有不允许的字符
    safe = INVALID_CHARS.sub("_", name)
    # 移除连续的下划线
    safe = re.sub(r"_+", "_", safe)
    # 移除首尾的下划线
    safe = safe.strip("_")
    # 如果文件名太长，使用hash缩短
    if len(safe) > max_len:
        name_hash = hashlib.md5(name.encode("utf-8")).hexdigest()[:8]
        # 保留前100个字符和后部分hash
        safe = safe[:100] + "_" + name_hash
    return safe if safe else "unknown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate corrupted HTML files.")
    parser.add_argument("--in", dest="input_csv", required=True, help="Base CSV path.")
    parser.add_argument(
        "--out",
        dest="output_dir",
        required=True,
        help="Directory to save corrupted HTML + CSVs.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--levels",
        nargs="+",
        default=["L", "M", "H"],
        help="Corruption levels to create.",
    )
    parser.add_argument(
        "--html-root",
        default=".",
        help="Root path for resolving relative html_path values.",
    )
    return parser.parse_args()


def resolve_html_path(path_str: str, html_root: Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] in {"workspace", "data"}:
        return path
    return html_root / path


def corrupt_low(html: str) -> str:
    return SCRIPT_RE.sub("", html)


def corrupt_mid(html: str) -> str:
    stripped = TAG_RE.sub("", html)
    return stripped[: len(stripped) // 2]


def corrupt_high(html: str, rng: random.Random) -> str:
    chunk = html[: max(200, len(html) // 3)]
    noise = "".join(rng.choice(["#", "*", "%", "&"]) for _ in range(50))
    return f"<div>{chunk}</div><!--noise:{noise}-->"


def save_html(
    content: str, output_dir: Path, sample_id: str, level: str
) -> Dict[str, str]:
    target_dir = output_dir / level.upper() / "html"
    target_dir.mkdir(parents=True, exist_ok=True)
    safe_id = sanitize_filename(str(sample_id))
    filename = f"{safe_id}.html"
    target_path = target_dir / filename

    try:
        # 写入文件
        target_path.write_text(content, encoding="utf-8", errors="ignore")
        # 计算SHA256
        sha256 = hashlib.sha256(target_path.read_bytes()).hexdigest()
    except (OSError, IOError) as e:
        # 如果文件操作失败，使用内容直接计算hash
        sha256 = hashlib.sha256(content.encode("utf-8", errors="ignore")).hexdigest()
        # 如果写入失败，记录错误但不中断流程
        print(f"Warning: Failed to save HTML for {sample_id}: {e}")

    relative_path = target_path.relative_to(output_dir)
    return {
        "html_path_corrupt": str(relative_path).replace("\\", "/"),
        "html_sha256_corrupt": sha256,
    }


def main() -> None:
    args = parse_args()
    base_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    html_root = Path(args.html_root)
    if not base_csv.exists():
        raise FileNotFoundError(base_csv)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(base_csv)
    if "html_path" not in df.columns or "id" not in df.columns:
        raise ValueError("CSV must contain id and html_path columns.")

    rng = random.Random(args.seed)
    outputs: List[Path] = []

    for level in args.levels:
        level = level.upper()
        records = []
        for _, row in df.iterrows():
            html_path = row.get("html_path")
            if pd.isna(html_path) or not str(html_path).strip():
                html_content = "<html></html>"
            else:
                resolved = resolve_html_path(str(html_path), html_root)
                if resolved.exists():
                    try:
                        html_content = resolved.read_text(
                            encoding="utf-8", errors="ignore"
                        )
                    except Exception:
                        html_content = "<html></html>"
                else:
                    html_content = "<html></html>"

            if level == "L":
                corrupted = corrupt_low(html_content)
            elif level == "M":
                corrupted = corrupt_mid(html_content)
            else:
                corrupted = corrupt_high(html_content, rng)

            path_info = save_html(corrupted, output_dir, str(row.get("id")), level)
            record = row.to_dict()
            record.update(path_info)
            record["corruption_level"] = level
            records.append(record)

        level_df = pd.DataFrame(records)
        csv_path = output_dir / f"test_corrupt_html_{level}.csv"
        level_df.to_csv(csv_path, index=False)
        outputs.append(csv_path)
        print(f"[corrupt_html] Level {level}: saved {len(level_df)} rows to {csv_path}")

    print("Corrupted HTML CSV files:", [str(p) for p in outputs])


if __name__ == "__main__":
    main()

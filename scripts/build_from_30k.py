#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 30k 数据集构建高质量数据集并追加到 master CSV
支持：文件夹名解析、品牌规范化、分标签约束、四级去重、自适应阈值
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tldextract
import yaml
from PIL import Image
from tqdm import tqdm

# Handle Windows console encoding
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        import codecs

        sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "strict")
        sys.stderr = codecs.getwriter("utf-8")(sys.stderr.buffer, "strict")


# =====================================================================
# File Discovery
# =====================================================================


def find_html_file(folder: Path) -> Optional[Path]:
    """查找 HTML 文件，优先 html.txt，回退 html.html"""
    for name in ("html.txt", "html.html", "page.html", "index.html"):
        p = folder / name
        if p.exists():
            return p
    # Fallback: any .html or .htm
    for p in folder.glob("*.htm*"):
        return p
    return None


def find_image_file(folder: Path) -> Optional[Path]:
    """查找图像文件"""
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
    for p in folder.glob("*.png"):
        return p
    for p in folder.glob("*.jpg"):
        return p
    for p in folder.glob("*.jpeg"):
        return p
    return None


# =====================================================================
# Info Parsing - 安全解析
# =====================================================================


def read_info_txt(path: Path) -> Dict[str, Any]:
    """使用 ast.literal_eval 安全解析 Python dict"""
    if not path.exists():
        return {}

    try:
        text = path.read_text(encoding="utf-8", errors="ignore").strip()

        # 尝试安全解析 Python dict
        data = ast.literal_eval(text)
        if isinstance(data, dict):
            return data
    except (ValueError, SyntaxError):
        pass

    # 回退：尝试 JSON
    try:
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    # 回退：key: value 格式
    try:
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        data = {}
        for line in text.splitlines():
            line = line.strip()
            if not line or ":" not in line:
                continue
            k, v = line.split(":", 1)
            data[k.strip().lower()] = v.strip()
        if data:
            return data
    except Exception:
        pass

    return {}


def find_url(folder: Path, info: Dict[str, Any]) -> Optional[str]:
    """提取 URL：优先 info dict，回退 url.txt 或 info.txt 纯文本"""
    # Try info dict
    for key in ("url", "url_text", "request_url"):
        if key in info:
            return str(info[key]).strip()

    # Try url.txt file
    url_file = folder / "url.txt"
    if url_file.exists():
        try:
            url_text = url_file.read_text(encoding="utf-8", errors="ignore").strip()
            if url_text and ("http://" in url_text or "https://" in url_text):
                return url_text
        except Exception:
            pass

    # Try info.txt as plain URL (for benign dataset)
    info_file = folder / "info.txt"
    if info_file.exists():
        try:
            text = info_file.read_text(encoding="utf-8", errors="ignore").strip()
            # 如果是纯URL文本（不是dict）
            if (
                text
                and ("http://" in text or "https://" in text)
                and not text.startswith("{")
            ):
                # 取第一行
                first_line = text.split("\n")[0].strip()
                return first_line
        except Exception:
            pass

    return None


# =====================================================================
# Timestamp Parsing - 鲁棒解析
# =====================================================================


def parse_ts_any(ts_str: Any) -> Optional[str]:
    """鲁棒的时间戳解析，支持多种格式，统一返回 ISO 8601 + Z"""
    if not ts_str or (isinstance(ts_str, float) and np.isnan(ts_str)):
        return None

    s = str(ts_str).strip()
    if not s:
        return None

    # 尝试多种格式
    patterns = [
        ("%Y-%m-%dT%H:%M:%SZ", None),  # ISO 8601
        ("%Y-%m-%d-%H`%M`%S", "`"),  # 2019-07-28-22`34`40
        ("%Y-%m-%d-%H-%M-%S", None),  # 2019-07-28-22-34-40
        ("%Y/%m/%d %H:%M:%S", None),  # 2019/07/28 22:34:40
        ("%Y-%m-%d %H:%M:%S", None),  # 2019-07-28 22:34:40
        ("%Y-%m-%dT%H:%M:%S", None),  # ISO without Z
    ]

    for fmt, replace_char in patterns:
        try:
            s_clean = s.replace(replace_char, ":") if replace_char else s
            dt = datetime.strptime(s_clean, fmt)
            return dt.isoformat() + "Z"
        except ValueError:
            continue

    # 回退：使用 dateutil.parser
    try:
        from dateutil import parser as dparser

        dt = dparser.parse(s)
        return dt.isoformat() + "Z"
    except Exception:  # noqa: E722
        pass

    return None


def parse_folder_timestamp(folder_name: str) -> Optional[str]:
    """从文件夹名解析时间戳：Brand+YYYY-MM-DD-HH`MM`SS"""
    if "+" not in folder_name:
        return None

    parts = folder_name.split("+")
    if len(parts) < 2:
        return None

    ts_part = parts[1]
    return parse_ts_any(ts_part)


def get_timestamp_fallback(folder: Path) -> Tuple[str, str]:
    """回退到文件 mtime，返回 (timestamp, source)"""
    try:
        mtime = folder.stat().st_mtime
        dt = datetime.fromtimestamp(mtime, tz=timezone.utc)
        return dt.isoformat(), "fs_mtime"
    except Exception:
        return datetime.now(tz=timezone.utc).isoformat(), "fs_mtime"


# =====================================================================
# Brand Normalization - 品牌规范化
# =====================================================================


def load_brand_aliases(alias_path: Path) -> Dict[str, str]:
    """加载品牌别名映射"""
    if not alias_path or not alias_path.exists():
        return {}

    try:
        with open(alias_path, "r", encoding="utf-8") as f:
            aliases = yaml.safe_load(f) or {}
        return aliases
    except Exception as e:
        print(f"警告: 无法加载品牌别名文件 {alias_path}: {e}")
        return {}


def normalize_brand(brand_raw: Any, aliases: Dict[str, str]) -> Optional[str]:
    """品牌规范化：去空格、转小写、应用别名、清洗特殊字符"""
    if not brand_raw or (isinstance(brand_raw, float) and np.isnan(brand_raw)):
        return None

    # 去除奇怪字符（全角空格、换行、制表符）
    brand = str(brand_raw).strip()
    brand = brand.replace("\u3000", " ")  # 全角空格
    brand = brand.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    brand = " ".join(brand.split())  # 压缩多余空格
    brand = brand.lower()

    # 应用别名映射
    if brand in aliases:
        brand = aliases[brand]

    # 移除特殊字符，只保留字母数字
    brand = re.sub(r"[^\w\s]", "", brand)
    brand = re.sub(r"\s+", "", brand)

    return brand if brand else None


def extract_brand_from_benign_domain(
    domain: str, aliases: Dict[str, str]
) -> Optional[str]:
    """从合法数据集域名提取品牌，清洗无效字符"""
    try:
        ext = tldextract.extract(domain)
        brand = ext.domain

        if not brand:
            return None

        # 清洗：仅保留字母数字
        brand = re.sub(r"[^a-z0-9]", "", brand.lower())

        # 过滤无效品牌（数字开头、过短、纯数字）
        if not brand or brand[0].isdigit() or len(brand) < 2 or brand.isdigit():
            return None

        # 应用别名
        brand = normalize_brand(brand, aliases)

        return brand
    except Exception:
        return None


def parse_folder_brand(folder_name: str) -> Optional[str]:
    """从钓鱼数据集文件夹名提取品牌：Brand+Timestamp"""
    if "+" not in folder_name:
        return None

    parts = folder_name.split("+")
    brand = parts[0].strip()
    return brand if brand else None


# =====================================================================
# URL Normalization
# =====================================================================


def normalize_url(url: Optional[str]) -> Optional[str]:
    """标准化 URL: 移除尾斜杠、fragment、utm 参数"""
    if not url:
        return url

    url = url.strip()

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


def normalize_url_short(url: str, max_len: int = 128) -> str:
    """标准化 URL 并截断，用于模糊去重"""
    url = url.lower().strip()
    url = url.rstrip("/")
    # 移除查询参数
    if "?" in url:
        url = url.split("?")[0]
    return url[:max_len]


# =====================================================================
# Domain Extraction
# =====================================================================


def extract_domain(
    url: Optional[str], info_domain: Optional[str] = None
) -> Tuple[Optional[str], str]:
    """提取域名，返回 (domain, source)"""
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


# =====================================================================
# File Validation
# =====================================================================


def validate_path(p: Path) -> bool:
    """验证路径，兼容软链接和权限问题"""
    try:
        return p.exists() and p.stat().st_size > 0
    except (PermissionError, OSError):
        return False


def validate_html_quality(html_path: Path, min_size: int = 200) -> Tuple[bool, str]:
    """验证 HTML 文件质量"""
    if not validate_path(html_path):
        return False, "html_missing"

    try:
        size = html_path.stat().st_size
        if size < min_size:
            return False, "html_too_small"
        return True, "valid"
    except Exception:
        return False, "html_error"


def validate_image_quality(img_path: Path) -> Tuple[bool, str]:
    """验证图像文件"""
    if not validate_path(img_path):
        return False, "img_missing"

    try:
        with Image.open(img_path) as img:
            img.verify()
        return True, "valid"
    except Exception:
        return False, "img_corrupt"


# =====================================================================
# Hash Computation (Optional)
# =====================================================================


def compute_file_hash(path: Path) -> Optional[str]:
    """计算文件 SHA1 哈希"""
    try:
        sha1 = hashlib.sha1()
        with open(path, "rb") as f:
            while chunk := f.read(8192):
                sha1.update(chunk)
        return sha1.hexdigest()
    except Exception:
        return None


# =====================================================================
# Main Scanning Function
# =====================================================================


def scan_folder(
    root: Path,
    label: int,
    source_tag: str,
    brand_aliases: Dict[str, str],
    min_html_size: int = 200,
    compute_hash: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """扫描文件夹并提取样本"""

    samples = []
    dropped_reasons = defaultdict(int)

    # 获取子文件夹
    children = [child for child in root.iterdir() if child.is_dir()]

    print(f"\n扫描 {source_tag} 数据集: {len(children)} 个文件夹")

    for child in tqdm(children, desc=f"Processing {source_tag}"):
        stem = child.name

        # 读取 info.txt
        info = read_info_txt(child / "info.txt")

        # 提取 URL
        url_text = find_url(child, info)
        if not url_text:
            dropped_reasons["missing_url"] += 1
            continue

        url_text = normalize_url(url_text)

        # 提取域名
        domain, domain_source = extract_domain(
            url_text, info.get("domain") or info.get("host")
        )
        if not domain:
            dropped_reasons["domain_extraction_failed"] += 1
            continue

        # 提取品牌
        if label == 1:  # 钓鱼数据集
            # 优先级: info['brand'] → 文件夹名
            brand_raw = info.get("brand") or parse_folder_brand(stem)
            brand = normalize_brand(brand_raw, brand_aliases)
        else:  # 合法数据集
            # 从域名提取（文件夹名即域名）
            brand_raw = stem  # 文件夹名就是域名
            brand = extract_brand_from_benign_domain(stem, brand_aliases)

        # 提取时间戳
        # 优先级: info['isotime'] → 文件夹名 → 文件 mtime
        timestamp = parse_ts_any(
            info.get("isotime") or info.get("timestamp") or info.get("time")
        )
        timestamp_source = "info"

        if not timestamp:
            timestamp = parse_folder_timestamp(stem)
            timestamp_source = "folder_name"

        if not timestamp:
            timestamp, timestamp_source = get_timestamp_fallback(child)

        # 查找文件
        html_path = find_html_file(child)
        img_path = find_image_file(child)

        # 验证 HTML
        if not html_path:
            dropped_reasons["missing_html"] += 1
            continue

        html_valid, html_reason = validate_html_quality(html_path, min_html_size)
        if not html_valid:
            dropped_reasons[html_reason] += 1
            continue

        # 验证图像
        if not img_path:
            dropped_reasons["missing_img"] += 1
            continue

        img_valid, img_reason = validate_image_quality(img_path)
        if not img_valid:
            dropped_reasons[img_reason] += 1
            continue

        # 计算哈希（可选）
        html_sha1 = None
        img_sha1 = None
        if compute_hash:
            html_sha1 = compute_file_hash(html_path)
            img_sha1 = compute_file_hash(img_path)

        # 创建样本
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
                "brand_raw": brand_raw if label == 1 else domain,
                "timestamp": timestamp,
                "timestamp_source": timestamp_source,
                "source": source_tag,
                "folder": str(child.resolve()),
                "html_sha1": html_sha1,
                "img_sha1": img_sha1,
            }
        )

    print(f"  找到 {len(samples)} 个有效样本")
    print(f"  丢弃 {sum(dropped_reasons.values())} 个样本")

    return samples, dict(dropped_reasons)


# =====================================================================
# Deduplication - 四级去重
# =====================================================================


def deduplicate_samples(
    samples: List[Dict], compute_hash: bool = False
) -> Tuple[List[Dict], Dict[str, int]]:
    """四级去重，记录每级丢弃数"""

    dropped_reasons = defaultdict(int)
    original_count = len(samples)

    # Level 1: 哈希去重（如果启用）
    if compute_hash:
        seen_hashes = set()
        deduped = []
        for s in samples:
            hash_key = (s.get("html_sha1"), s.get("img_sha1"))
            if hash_key != (None, None) and hash_key in seen_hashes:
                dropped_reasons["duplicate_hash"] += 1
            else:
                if hash_key != (None, None):
                    seen_hashes.add(hash_key)
                deduped.append(s)
        samples = deduped
        print(
            f"  哈希去重: {original_count} → {len(samples)} (丢弃 {dropped_reasons['duplicate_hash']})"
        )

    # Level 2: 路径去重
    seen_paths = set()
    deduped = []
    for s in samples:
        path_key = s["html_path"]
        if path_key in seen_paths:
            dropped_reasons["duplicate_path"] += 1
        else:
            seen_paths.add(path_key)
            deduped.append(s)
    samples = deduped
    print(
        f"  路径去重: {len(samples) + dropped_reasons['duplicate_path']} → {len(samples)} (丢弃 {dropped_reasons['duplicate_path']})"
    )

    # Level 3: 语义键去重（url + domain + brand）
    seen_semantic = set()
    deduped = []
    for s in samples:
        sem_key = (s["url_text"], s["domain"], s.get("brand"))
        if sem_key in seen_semantic:
            dropped_reasons["duplicate_semantic"] += 1
        else:
            seen_semantic.add(sem_key)
            deduped.append(s)
    samples = deduped
    print(
        f"  语义去重: {len(samples) + dropped_reasons['duplicate_semantic']} → {len(samples)} (丢弃 {dropped_reasons['duplicate_semantic']})"
    )

    # Level 4: URL 短键去重
    seen_url_short = set()
    deduped = []
    for s in samples:
        url_short = normalize_url_short(s["url_text"])
        if url_short in seen_url_short:
            dropped_reasons["duplicate_url_short"] += 1
        else:
            seen_url_short.add(url_short)
            deduped.append(s)
    samples = deduped
    print(
        f"  URL短键去重: {len(samples) + dropped_reasons['duplicate_url_short']} → {len(samples)} (丢弃 {dropped_reasons['duplicate_url_short']})"
    )

    print(
        f"  总去重: {original_count} → {len(samples)} (丢弃 {sum(dropped_reasons.values())})"
    )

    return samples, dict(dropped_reasons)


# =====================================================================
# Brand Constraints - 分标签品牌约束 + 自适应阈值
# =====================================================================


def get_adaptive_thresholds(brand_count: int) -> Tuple[float, float]:
    """根据品牌数返回 (max_top1, max_top3)"""
    if brand_count >= 30:
        return 0.30, 0.60
    elif brand_count >= 10:
        return 0.35, 0.70
    else:
        return 0.40, 1.0  # 不检查 Top3


def enforce_brand_constraints(
    df: pd.DataFrame,
    target_n: int,
    label: int,
    min_per_brand: int = 50,
    brand_cap: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    """品牌约束 + 自适应 Top-N 保护（单标签）"""

    label_name = "phishing" if label == 1 else "benign"
    print(f"\n品牌约束 ({label_name}, 目标={target_n}):")

    # 检查是否有 brand 列
    if "brand" not in df.columns:
        print("  警告: 缺少 brand 列，返回原始数据")
        if len(df) > target_n:
            return df.sample(n=target_n, random_state=seed)
        return df

    # 过滤空品牌
    df = df[df["brand"].notna() & (df["brand"] != "")].copy()

    if len(df) == 0:
        print("  警告: 无有效品牌样本")
        return df

    brand_counts = df["brand"].value_counts()
    print(f"  品牌总数: {len(brand_counts)}")
    print(f"  Top 5 品牌: {dict(brand_counts.head(5))}")

    # 自适应阈值
    max_top1, max_top3 = get_adaptive_thresholds(len(brand_counts))
    print(f"  自适应阈值: Top1≤{max_top1:.0%}, Top3≤{max_top3:.0%}")

    # 策略 1: 限制单一品牌最大样本数
    sampled = []
    for brand, count in brand_counts.items():
        brand_df = df[df["brand"] == brand]
        n_take = min(count, brand_cap)
        sampled.append(brand_df.sample(n=n_take, random_state=seed))

    df = pd.concat(sampled, ignore_index=True)

    # 策略 2: 检查 Top 1 占比
    brand_counts = df["brand"].value_counts()
    if len(brand_counts) > 0:
        top1_ratio = brand_counts.iloc[0] / len(df)

        iteration = 0
        while top1_ratio > max_top1 and len(df) > target_n and iteration < 10:
            iteration += 1
            # 移除 Top 1 品牌的部分样本
            top_brand = brand_counts.index[0]
            top_df = df[df["brand"] == top_brand]
            remove_n = max(1, int(len(top_df) * 0.1))  # 每次移除 10%
            drop_indices = top_df.sample(
                n=remove_n, random_state=seed + iteration
            ).index
            df = df.drop(drop_indices)

            brand_counts = df["brand"].value_counts()
            top1_ratio = brand_counts.iloc[0] / len(df) if len(brand_counts) > 0 else 0

        print(f"  Top1 占比调整后: {top1_ratio:.1%}")

    # 策略 3: 检查 Top 3 累计占比
    if len(brand_counts) >= 3 and max_top3 < 1.0:
        top3_ratio = brand_counts.iloc[:3].sum() / len(df)

        iteration = 0
        while top3_ratio > max_top3 and len(df) > target_n and iteration < 10:
            iteration += 1
            # 减少 Top 3 样本
            for brand in brand_counts.index[:3]:
                brand_df = df[df["brand"] == brand]
                remove_n = max(1, int(len(brand_df) * 0.05))  # 每次移除 5%
                if len(brand_df) > remove_n:
                    drop_indices = brand_df.sample(
                        n=remove_n, random_state=seed + iteration
                    ).index
                    df = df.drop(drop_indices)

            brand_counts = df["brand"].value_counts()
            top3_ratio = (
                brand_counts.iloc[:3].sum() / len(df) if len(brand_counts) >= 3 else 0
            )

        print(f"  Top3 累计占比调整后: {top3_ratio:.1%}")

    # 最终抽样到目标数量
    if len(df) > target_n:
        df = df.sample(n=target_n, random_state=seed)
    elif len(df) < target_n:
        print(f"  警告: 样本不足 ({len(df)} < {target_n})")

    print(f"  最终样本数: {len(df)}")

    return df


# =====================================================================
# Append Mode - 追加并去重
# =====================================================================


def append_with_dedup(new_df: pd.DataFrame, master_csv: Path) -> pd.DataFrame:
    """追加新样本到现有 master CSV，严格去重"""

    if not master_csv.exists():
        print(f"\n{master_csv} 不存在，将创建新文件")
        return new_df

    print(f"\n追加模式: 读取现有 {master_csv}")
    existing_df = pd.read_csv(master_csv, encoding="utf-8", encoding_errors="ignore")
    print(f"  现有样本数: {len(existing_df)}")

    # 基于路径去重
    new_df = new_df[~new_df["html_path"].isin(existing_df["html_path"])]
    print(f"  路径去重后: {len(new_df)} 个新样本")

    # 基于 URL 去重
    if "url_text" in existing_df.columns:
        new_df = new_df[~new_df["url_text"].isin(existing_df["url_text"])]
        print(f"  URL去重后: {len(new_df)} 个新样本")

    # 合并
    combined = pd.concat([existing_df, new_df], ignore_index=True)
    print(f"  合并后总样本数: {len(combined)}")

    return combined


# =====================================================================
# Main Function
# =====================================================================


def main():
    parser = argparse.ArgumentParser(
        description="从 30k 数据集构建高质量数据集",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--phish_root", required=True, help="钓鱼数据集路径")
    parser.add_argument("--benign_root", required=True, help="合法数据集路径")
    parser.add_argument("--k_each", type=int, default=8000, help="每类样本数")
    parser.add_argument(
        "--out_csv", default="data/processed/master_30k.csv", help="输出 CSV 路径"
    )
    parser.add_argument("--master_csv", help="追加模式：现有 master CSV 路径")
    parser.add_argument("--append", action="store_true", help="追加模式")
    parser.add_argument(
        "--brand_alias", default="resources/brand_alias.yaml", help="品牌别名文件"
    )
    parser.add_argument("--min_per_brand", type=int, default=50, help="品牌最小样本数")
    parser.add_argument("--brand_cap", type=int, default=500, help="单品牌最大样本数")
    parser.add_argument(
        "--max_single_ratio", type=float, default=0.30, help="Top1 品牌最大占比"
    )
    parser.add_argument(
        "--max_top3_ratio", type=float, default=0.60, help="Top3 品牌累计最大占比"
    )
    parser.add_argument(
        "--min_html_size", type=int, default=200, help="HTML 最小字节数"
    )
    parser.add_argument(
        "--compute_hash", action="store_true", help="计算文件哈希（慢）"
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    args = parser.parse_args()

    # 设置随机种子
    np.random.seed(args.seed)

    # 加载品牌别名
    brand_aliases = load_brand_aliases(Path(args.brand_alias))
    print(f"加载品牌别名: {len(brand_aliases)} 条")

    # 创建输出目录
    output_path = Path(args.master_csv if args.append else args.out_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("==> 30k 数据集构建")
    print("=" * 70)
    print(f"钓鱼数据集: {args.phish_root}")
    print(f"合法数据集: {args.benign_root}")
    print(f"目标样本数: {args.k_each} 每类")
    print(f"品牌约束: min={args.min_per_brand}, cap={args.brand_cap}")
    print(f"Top占比: Top1≤{args.max_single_ratio:.0%}, Top3≤{args.max_top3_ratio:.0%}")
    print(f"计算哈希: {'是' if args.compute_hash else '否'}")
    print(f"追加模式: {'是' if args.append else '否'}")
    print(f"随机种子: {args.seed}")
    print("=" * 70)

    # 扫描钓鱼数据集
    print("\n[步骤 1/6] 扫描钓鱼数据集...")
    phish_samples, phish_dropped = scan_folder(
        Path(args.phish_root),
        label=1,
        source_tag="phish",
        brand_aliases=brand_aliases,
        min_html_size=args.min_html_size,
        compute_hash=args.compute_hash,
    )

    # 扫描合法数据集
    print("\n[步骤 2/6] 扫描合法数据集...")
    benign_samples, benign_dropped = scan_folder(
        Path(args.benign_root),
        label=0,
        source_tag="benign",
        brand_aliases=brand_aliases,
        min_html_size=args.min_html_size,
        compute_hash=args.compute_hash,
    )

    # 转 DataFrame
    print("\n[步骤 3/6] 转换为 DataFrame...")
    phish_df = pd.DataFrame(phish_samples)
    benign_df = pd.DataFrame(benign_samples)
    print(f"  钓鱼: {len(phish_df)} 样本")
    print(f"  合法: {len(benign_df)} 样本")

    # 分别去重
    print("\n[步骤 4/6] 四级去重...")
    print("\n钓鱼数据集去重:")
    phish_df_list, phish_dup = deduplicate_samples(phish_samples, args.compute_hash)
    phish_df = pd.DataFrame(phish_df_list)

    print("\n合法数据集去重:")
    benign_df_list, benign_dup = deduplicate_samples(benign_samples, args.compute_hash)
    benign_df = pd.DataFrame(benign_df_list)

    # 分别应用品牌约束
    print("\n[步骤 5/6] 分标签品牌约束...")
    phish_final = enforce_brand_constraints(
        phish_df,
        target_n=args.k_each,
        label=1,
        min_per_brand=args.min_per_brand,
        brand_cap=args.brand_cap,
        seed=args.seed,
    )

    benign_final = enforce_brand_constraints(
        benign_df,
        target_n=args.k_each,
        label=0,
        min_per_brand=args.min_per_brand,
        brand_cap=args.brand_cap,
        seed=args.seed,
    )

    # 合并
    print("\n[步骤 6/6] 合并并保存...")
    final_df = pd.concat([phish_final, benign_final], ignore_index=True)

    # 添加 split 列（unsplit）
    final_df["split"] = "unsplit"

    # 追加模式
    if args.append and args.master_csv:
        final_df = append_with_dedup(final_df, Path(args.master_csv))

    # 保存（Windows下 pandas 会自动处理换行符）
    final_df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"\n✅ 保存到: {output_path}")
    print(f"   总样本数: {len(final_df)}")
    print(f"   钓鱼: {len(final_df[final_df['label']==1])}")
    print(f"   合法: {len(final_df[final_df['label']==0])}")

    # 保存元数据
    metadata = {
        "total_samples": len(final_df),
        "phishing_count": len(final_df[final_df["label"] == 1]),
        "benign_count": len(final_df[final_df["label"] == 0]),
        "brand_distribution": final_df["brand"].value_counts().head(20).to_dict(),
        "timestamp_range": {
            "min": final_df["timestamp"].min(),
            "max": final_df["timestamp"].max(),
        },
        "dropped_reasons": {
            "phishing": phish_dropped,
            "benign": benign_dropped,
            "dedup_phishing": phish_dup,
            "dedup_benign": benign_dup,
        },
        "parameters": vars(args),
    }

    metadata_path = (
        output_path.parent / f'metadata_{output_path.stem.replace("master_", "")}.json'
    )
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"✅ 元数据保存到: {metadata_path}")

    print("\n" + "=" * 70)
    print("构建完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()

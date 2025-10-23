#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件匹配和复制脚本 V2 - 优化版
匹配 data/raw/dataset 和 benign_sample_30k 文件夹中的对应文件
"""

import os
import hashlib
import shutil
from pathlib import Path
from urllib.parse import urlparse
import json
from datetime import datetime


def get_file_hash(file_path):
    """计算文件的MD5哈希值"""
    if not os.path.exists(file_path):
        return None

    hash_md5 = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception:
        return None


def extract_domain(url_text):
    """从URL中提取域名"""
    url_text = url_text.strip()
    if not url_text:
        return None

    # 如果没有协议，添加http://
    if not url_text.startswith(("http://", "https://")):
        url_text = "http://" + url_text

    try:
        parsed = urlparse(url_text)
        domain = parsed.netloc.lower()
        # 移除 www. 前缀
        if domain.startswith("www."):
            domain = domain[4:]
        return domain
    except Exception:
        return None


def check_html_match(path_a, path_b):
    """检查HTML文件是否匹配"""
    html_a = os.path.join(path_a, "html.html")
    html_b = os.path.join(path_b, "html.txt")

    if not os.path.exists(html_a) or not os.path.exists(html_b):
        return False

    hash_a = get_file_hash(html_a)
    hash_b = get_file_hash(html_b)

    return hash_a is not None and hash_a == hash_b


def check_shot_match(path_a, path_b):
    """检查shot.png是否匹配"""
    shot_a = os.path.join(path_a, "shot.png")
    shot_b = os.path.join(path_b, "shot.png")

    if not os.path.exists(shot_a) or not os.path.exists(shot_b):
        return False

    hash_a = get_file_hash(shot_a)
    hash_b = get_file_hash(shot_b)

    return hash_a is not None and hash_a == hash_b


def find_matching_folder(dir_a, url_domain, path_b_root, dirs_b_dict):
    """
    查找匹配的B路径文件夹
    返回: (是否匹配, 匹配的文件夹路径, 匹配详情)
    """
    # 候选文件夹列表
    candidates = []

    # 1. 先尝试直接域名匹配
    if url_domain and url_domain in dirs_b_dict:
        candidates.append(url_domain)

    # 2. 如果没找到，尝试部分匹配
    if not candidates and url_domain:
        for b_name in dirs_b_dict.keys():
            if url_domain in b_name or b_name in url_domain:
                candidates.append(b_name)
                if len(candidates) >= 5:  # 限制候选数量
                    break

    # 对每个候选进行详细检查
    for candidate in candidates:
        dir_b = dirs_b_dict[candidate]

        # 检查HTML和Shot匹配
        html_match = check_html_match(str(dir_a), str(dir_b))
        shot_match = check_shot_match(str(dir_a), str(dir_b))

        match_details = {
            "url": True,  # 已经通过URL筛选了候选
            "html": html_match,
            "shot": shot_match,
        }

        # 至少2个匹配才算成功
        match_count = sum(match_details.values())
        if match_count >= 2:
            return True, dir_b, candidate, match_details

    return False, None, None, None


def copy_files(src_dir, dst_dir, files_to_copy):
    """复制文件，返回成功复制的文件列表和缺失的文件列表"""
    copied = []
    missing = []

    for file_name in files_to_copy:
        src_file = os.path.join(src_dir, file_name)
        dst_file = os.path.join(dst_dir, file_name)

        # 检查源文件是否存在
        if not os.path.exists(src_file):
            missing.append(file_name)
            continue

        # 检查目标文件是否已存在（不覆盖）
        if os.path.exists(dst_file):
            continue

        # 复制文件
        try:
            shutil.copy2(src_file, dst_file)
            copied.append(file_name)
        except Exception:
            missing.append(file_name)

    return copied, missing


def main():
    # 路径配置
    path_a_root = Path("data/raw/fish_dataset")
    path_b_root = Path(r"D:\one\生活\qwq\资料\学校个人资料\数据集资料\phish_sample_30k")

    # 要复制的文件列表
    files_to_copy = ["yolo_coords.txt", "info.txt", "ocr.txt"]

    # 统计信息
    stats = {
        "total_a": 0,
        "matched": 0,
        "unmatched": [],
        "copied": [],
        "missing_files": [],
    }

    print("=" * 80)
    print("Match and Copy Script V2 - Optimized")
    print("=" * 80)
    print(f"A Path: {path_a_root.absolute()}")
    print(f"B Path: {path_b_root.absolute()}")
    print(f"Files to copy: {', '.join(files_to_copy)}")
    print("=" * 80)

    # 检查路径是否存在
    if not path_a_root.exists():
        print(f"[ERROR] A path not found: {path_a_root}")
        return

    if not path_b_root.exists():
        print(f"[ERROR] B path not found: {path_b_root}")
        return

    # 获取所有A路径的文件夹
    dirs_a = sorted([d for d in path_a_root.iterdir() if d.is_dir()])
    stats["total_a"] = len(dirs_a)

    print(f"\nFound {len(dirs_a)} folders in A path")

    # 预先加载B路径的文件夹字典（文件夹名 -> 路径）
    print("Loading B path folders...")
    dirs_b_dict = {d.name: d for d in path_b_root.iterdir() if d.is_dir()}
    print(f"Found {len(dirs_b_dict)} folders in B path")
    print("\nStarting matching process...\n")

    # 遍历A路径的每个文件夹
    for idx, dir_a in enumerate(dirs_a, 1):
        print(f"[{idx}/{len(dirs_a)}] Processing: {dir_a.name}")

        # 读取URL并提取域名
        url_file = dir_a / "url.txt"
        url_domain = None
        if url_file.exists():
            try:
                with open(url_file, "r", encoding="utf-8", errors="ignore") as f:
                    url_text = f.read().strip()
                url_domain = extract_domain(url_text)
                print(f"  URL domain: {url_domain}")
            except Exception:
                pass

        if not url_domain:
            print("  [SKIP] Cannot extract domain from URL")
            stats["unmatched"].append(dir_a.name)
            continue

        # 查找匹配的文件夹
        matched, dir_b, dir_b_name, match_details = find_matching_folder(
            dir_a, url_domain, path_b_root, dirs_b_dict
        )

        if matched:
            print(f"  [MATCH] Found: {dir_b_name}")
            print(
                f"    Details: URL={match_details['url']}, HTML={match_details['html']}, Shot={match_details['shot']}"
            )

            # 复制文件
            copied, missing = copy_files(str(dir_b), str(dir_a), files_to_copy)

            stats["matched"] += 1
            stats["copied"].append(
                {
                    "folder_a": dir_a.name,
                    "folder_b": dir_b_name,
                    "copied_files": copied,
                    "missing_files": missing,
                    "match_details": match_details,
                }
            )

            if copied:
                print(f"    [COPY] Copied {len(copied)} files: {', '.join(copied)}")
            if missing:
                print(f"    [MISSING] Files not found in B: {', '.join(missing)}")
                stats["missing_files"].append(
                    {"folder_a": dir_a.name, "folder_b": dir_b_name, "missing": missing}
                )
        else:
            print("  [NOT MATCHED]")
            stats["unmatched"].append(dir_a.name)

        print()

    # 生成报告
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total folders in A: {stats['total_a']}")
    print(f"Matched: {stats['matched']}")
    print(f"Not matched: {len(stats['unmatched'])}")
    print(f"Folders with missing files: {len(stats['missing_files'])}")
    print("=" * 80)

    # 保存详细报告
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"match_report_{timestamp}.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"\nDetailed report saved to: {report_file}")

    # 输出未匹配的文件夹
    if stats["unmatched"]:
        unmatched_file = f"unmatched_folders_{timestamp}.txt"
        with open(unmatched_file, "w", encoding="utf-8") as f:
            f.write("Unmatched folders in A path:\n")
            f.write("=" * 60 + "\n")
            for folder in stats["unmatched"]:
                f.write(f"{folder}\n")
        print(f"Unmatched folders list saved to: {unmatched_file}")

    # 输出文件不全的记录
    if stats["missing_files"]:
        missing_file = f"missing_files_{timestamp}.txt"
        with open(missing_file, "w", encoding="utf-8") as f:
            f.write("Folders with missing files in B path:\n")
            f.write("=" * 60 + "\n")
            for item in stats["missing_files"]:
                f.write(f"\nA folder: {item['folder_a']}\n")
                f.write(f"B folder: {item['folder_b']}\n")
                f.write(f"Missing files: {', '.join(item['missing'])}\n")
        print(f"Missing files log saved to: {missing_file}")


if __name__ == "__main__":
    main()

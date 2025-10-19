#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
import tldextract
from urllib.parse import urlparse

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
HTML_EXTS = {".html", ".htm"}
TXT_EXTS = {".txt"}

def _parse_domain(url: Optional[str]) -> str:
    if not url or not isinstance(url, str) or len(url) < 4:
        return ""
    try:
        ext = tldextract.extract(url)
        if ext.domain:
            return ".".join([p for p in [ext.domain, ext.suffix] if p])
    except Exception:
        pass
    try:
        netloc = urlparse(url).netloc
        if netloc:
            parts = netloc.split(":")[0].split(".")
            return ".".join(parts[-2:]) if len(parts) >= 2 else netloc
    except Exception:
        pass
    return ""

def _find_first_with_ext(dirpath: Path, exts: set[str], prefer=("html","shot")) -> Optional[Path]:
    for pref in prefer:
        for ext in exts:
            p = dirpath / f"{pref}{ext}"
            if p.exists(): return p
    for p in dirpath.iterdir():
        if p.is_file() and p.suffix.lower() in exts:
            return p
    return None

def _collect_rows_sampledirs(root: Path, label: int) -> List[Dict]:
    rows: List[Dict] = []
    if not root.exists(): return rows
    for sub in sorted([p for p in root.iterdir() if p.is_dir()]):
        html_p = _find_first_with_ext(sub, HTML_EXTS)
        img_p  = _find_first_with_ext(sub, IMG_EXTS)
        url_text = ""
        url_p = sub / "url.txt"
        if url_p.exists():
            try:
                first = url_p.read_text(encoding="utf-8", errors="ignore").strip().splitlines()
                if first: url_text = first[0].strip()
            except Exception: pass
        if not url_text:
            for p in sub.iterdir():
                if p.is_file() and p.suffix.lower() in TXT_EXTS:
                    try:
                        first = p.read_text(encoding="utf-8", errors="ignore").strip().splitlines()
                        if first: url_text = first[0].strip(); break
                    except Exception: continue
        stem = sub.name
        rows.append({
            "id": f"{root.name}_{stem}",
            "stem": stem,
            "label": int(label),
            "url_text": url_text,
            "html_path": str(html_p) if html_p else "",
            "img_path": str(img_p) if img_p else "",
            "domain": _parse_domain(url_text),
            "source": str(root),
        })
    return rows

def _collect_rows_tridirs(root: Path, label: int) -> List[Dict]:
    url_dir, html_dir, img_dir = root / "url", root / "html", root / "img"
    def list_stems(dirpath: Path, exts: set[str]) -> set[str]:
        stems = set()
        if dirpath.exists():
            for p in dirpath.iterdir():
                if p.is_file() and p.suffix.lower() in exts:
                    stems.add(p.stem)
        return stems
    url_map: Dict[str, str] = {}
    if url_dir.exists():
        for p in url_dir.iterdir():
            if p.is_file() and p.suffix.lower() in TXT_EXTS:
                try:
                    first = p.read_text(encoding="utf-8", errors="ignore").strip().splitlines()
                    if first: url_map[p.stem] = first[0].strip()
                except Exception: pass
        for p in url_dir.iterdir():
            if p.suffix.lower() == ".csv":
                try:
                    df = pd.read_csv(p)
                    cols = {c.lower(): c for c in df.columns}
                    stem_col = cols.get("stem") or cols.get("id") or cols.get("name")
                    url_col  = cols.get("url_text") or cols.get("url")
                    if stem_col and url_col:
                        for s, u in zip(df[stem_col].astype(str), df[url_col].astype(str)):
                            url_map[str(Path(s).stem)] = u
                except Exception: pass
    stems = sorted(list(list_stems(html_dir, HTML_EXTS) | list_stems(img_dir, IMG_EXTS) | set(url_map.keys())))
    rows: List[Dict] = []
    for s in stems:
        html_path = next((str(html_dir / f"{s}{ext}") for ext in HTML_EXTS if (html_dir / f"{s}{ext}").exists()), "")
        img_path  = next((str(img_dir  / f"{s}{ext}") for ext in IMG_EXTS  if (img_dir  / f"{s}{ext}").exists()), "")
        url_text  = url_map.get(s, "")
        rows.append({
            "id": f"{root.name}_{s}",
            "stem": s,
            "label": int(label),
            "url_text": url_text,
            "html_path": html_path,
            "img_path": img_path,
            "domain": _parse_domain(url_text),
            "source": str(root),
        })
    return rows

def _collect_rows(root: Path, label: int) -> List[Dict]:
    has_tri = (root / "url").is_dir() or (root / "html").is_dir() or (root / "img").is_dir()
    return _collect_rows_tridirs(root, label) if has_tri else _collect_rows_sampledirs(root, label)

def _groupwise_split(df: pd.DataFrame, val_size: float, test_size: float, seed: int = 42) -> pd.Series:
    groups = df["domain"].fillna("")
    if groups.eq("").all(): groups = df["stem"].astype(str)
    gss = GroupShuffleSplit(n_splits=1, test_size=val_size + test_size, random_state=seed)
    idx_train, idx_temp = next(gss.split(df, groups=groups))
    temp = df.iloc[idx_temp]
    val_rel = val_size / max(val_size + test_size, 1e-9)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=1 - val_rel, random_state=seed)
    idx_val_rel, idx_test_rel = next(gss2.split(temp, groups=temp["domain"]))
    idx_val = temp.index[idx_val_rel]; idx_test = temp.index[idx_test_rel]
    split = pd.Series(index=df.index, dtype="string")
    split.loc[idx_train] = "train"; split.loc[idx_val] = "val"; split.loc[idx_test] = "test"
    return split

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benign", required=True, help="Path to benign root (dataset)")
    ap.add_argument("--phish",  required=True, help="Path to phishing root (fish_dataset)")
    ap.add_argument("--outdir", required=True, help="Output directory for master & splits")
    ap.add_argument("--val_size", type=float, default=0.15)
    ap.add_argument("--test_size", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    benign_root = Path(args.benign); phish_root = Path(args.phish)
    print(f"Processing benign data from: {benign_root}")
    print(f"Processing phish data from: {phish_root}")
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory created: {outdir}")

    rows: List[Dict] = []
    print("Collecting benign samples...")
    rows += _collect_rows(benign_root, label=0)
    print(f"Found {len(rows)} benign samples")
    
    print("Collecting phishing samples...")
    rows += _collect_rows(phish_root,  label=1)
    print(f"Total samples: {len(rows)}")

    df = pd.DataFrame(rows).drop_duplicates(subset=["id"]).reset_index(drop=True)
    df["split"] = _groupwise_split(df, val_size=args.val_size, test_size=args.test_size, seed=args.seed)

    master_path = outdir / "master.csv"
    df.to_csv(master_path, index=False)

    for part in ("train","val","test"):
        part_df = df[df["split"]==part][["url_text","label"]].copy()
        part_df = part_df[part_df["url_text"].astype(str).str.len() > 0]
        part_df.to_csv(outdir / f"{part}.csv", index=False)

    print("Saved:", master_path)
    print("Counts:\n", df["split"].value_counts())

if __name__ == "__main__":
    main()
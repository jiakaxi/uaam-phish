#!/usr/bin/env python
"""
Brand-OOD split utility (train/val/test_id/test_ood + brand sets JSON).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
import tldextract

REQUIRED_COLUMNS = [
    "id",
    "label",
    "url_text",
    "html_path",
    "img_path",
    "brand",
    "timestamp",
]
OPTIONAL_COLUMNS = ["etld_plus_one", "source", "domain"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create Brand-OOD splits.")
    parser.add_argument(
        "--in", dest="input_csv", required=True, help="Master CSV path."
    )
    parser.add_argument(
        "--out",
        dest="output_dir",
        required=True,
        help="Output directory for Brand-OOD splits.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--top_k", type=int, default=20, help="Number of in-domain brands."
    )
    parser.add_argument(
        "--min-neg-per-brand",
        type=int,
        default=1,
        help="Minimum number of negative samples per brand for in-domain selection.",
    )
    parser.add_argument(
        "--min-pos-per-brand",
        type=int,
        default=1,
        help="Minimum number of positive samples per brand for in-domain selection.",
    )
    parser.add_argument(
        "--ood-ratio",
        type=float,
        default=0.25,
        help="Desired ratio (#test_ood ≈ ood_ratio * #test_id).",
    )
    return parser.parse_args()


def ensure_required(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def ensure_etld(df: pd.DataFrame) -> pd.DataFrame:
    if "etld_plus_one" in df.columns and df["etld_plus_one"].notna().any():
        return df

    def compute(url: str) -> str:
        try:
            ext = tldextract.extract(url or "")
            domain = ".".join([p for p in [ext.domain, ext.suffix] if p])
            return domain or ""
        except Exception:
            return ""

    df["etld_plus_one"] = df["url_text"].astype(str).apply(compute)
    return df


def normalize_brand(df: pd.DataFrame) -> pd.DataFrame:
    df["brand"] = df["brand"].astype(str).str.strip().str.lower()
    return df


def ensure_source(df: pd.DataFrame, hint: str) -> pd.DataFrame:
    if "source" not in df.columns:
        df["source"] = hint
    df["source"] = df["source"].fillna(hint)
    return df


def select_balanced_brand_sets(
    df: pd.DataFrame, top_k: int, min_neg_per_brand: int = 1, min_pos_per_brand: int = 1
) -> Tuple[List[str], List[str]]:
    """
    选择包含足够负例和正例的in-domain brands
    """
    # 统计每个品牌的类别分布
    brand_stats = df.groupby("brand")["label"].agg(["count", "sum"]).reset_index()
    brand_stats.columns = ["brand", "total", "pos_count"]
    brand_stats["neg_count"] = brand_stats["total"] - brand_stats["pos_count"]

    # 筛选出有足够负例和正例的品牌
    valid_brands = brand_stats[
        (brand_stats["neg_count"] >= min_neg_per_brand)
        & (brand_stats["pos_count"] >= min_pos_per_brand)
    ]

    # 按总样本数排序，选择top_k个
    valid_brands = valid_brands.sort_values("total", ascending=False).head(top_k)

    b_ind = valid_brands["brand"].tolist()

    # 将单侧品牌（只有正例或只有负例）放入OOD集
    single_class_brands = brand_stats[
        (brand_stats["pos_count"] == 0) | (brand_stats["neg_count"] == 0)
    ]["brand"].tolist()

    # 其他品牌放入OOD集
    b_ood = [b for b in df["brand"].unique() if b not in b_ind]

    # 输出品牌选择信息
    print(
        f"Selected {len(b_ind)} in-domain brands with ≥{min_neg_per_brand} negative and ≥{min_pos_per_brand} positive samples:"
    )
    for i, brand in enumerate(b_ind, 1):
        brand_info = brand_stats[brand_stats["brand"] == brand].iloc[0]
        print(
            f"  {i:2d}. {brand:<20} total: {brand_info['total']:3d}, pos: {brand_info['pos_count']:2d}, neg: {brand_info['neg_count']:2d}"
        )

    print(f"Single-class brands (moved to OOD): {len(single_class_brands)}")

    if len(b_ind) < top_k:
        print(
            f"[WARNING] Only {len(b_ind)} brands meet the criteria (requested {top_k})"
        )
        if len(b_ind) == 0:
            print(
                "[WARNING] No brands meet the criteria. Falling back to brands with any positive and negative samples."
            )
            # 回退策略：选择有正例和负例的品牌（不限制数量）
            fallback_brands = brand_stats[
                (brand_stats["pos_count"] > 0) & (brand_stats["neg_count"] > 0)
            ]
            fallback_brands = fallback_brands.sort_values(
                "total", ascending=False
            ).head(top_k)
            b_ind = fallback_brands["brand"].tolist()
            b_ood = [b for b in df["brand"].unique() if b not in b_ind]
            print(
                f"[INFO] Selected {len(b_ind)} fallback brands with both positive and negative samples:"
            )
            for i, brand in enumerate(b_ind, 1):
                brand_info = brand_stats[brand_stats["brand"] == brand].iloc[0]
                print(
                    f"  {i:2d}. {brand:<20} total: {brand_info['total']:3d}, pos: {brand_info['pos_count']:2d}, neg: {brand_info['neg_count']:2d}"
                )

    return b_ind, b_ood


def stratified_split_by_brand_label(
    df: pd.DataFrame, seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    按品牌和标签的组合进行分层采样，处理样本数太少的组合
    """
    df = df.copy()

    # 创建品牌+标签的组合作为分层变量
    df["strata"] = df["brand"].astype(str) + "_" + df["label"].astype(str)

    # 统计每个strata的样本数
    strata_counts = df["strata"].value_counts()

    # 将样本数少于2的strata合并到"OTHER"组
    small_strata = strata_counts[strata_counts < 2].index.tolist()
    if small_strata:
        print(f"[INFO] 合并 {len(small_strata)} 个样本数少于2的strata到OTHER组")
        df.loc[df["strata"].isin(small_strata), "strata"] = "OTHER"

    # 检查合并后的strata分布
    final_strata_counts = df["strata"].value_counts()
    print(f"[INFO] 分层采样使用 {len(final_strata_counts)} 个strata")

    # 检查是否所有strata至少有2个样本
    min_strata_count = final_strata_counts.min()
    if min_strata_count < 2:
        print("[WARNING] 仍有strata样本数少于2，继续合并到OTHER组")
        # 将所有样本数少于2的strata合并到OTHER
        small_final = final_strata_counts[final_strata_counts < 2].index.tolist()
        if "OTHER" in small_final:
            small_final.remove("OTHER")
        if small_final:
            df.loc[df["strata"].isin(small_final), "strata"] = "OTHER"
            final_strata_counts = df["strata"].value_counts()

    # 如果OTHER组样本数仍然少于2，或者只有1个strata，使用随机采样
    if len(final_strata_counts) == 1 or final_strata_counts.min() < 2:
        print("[WARNING] 无法进行分层采样，使用随机采样（按label分层）")
        train_df, temp_df = train_test_split(
            df, test_size=0.3, stratify=df["label"], random_state=seed
        )
        val_df, test_df = train_test_split(
            temp_df, test_size=0.5, stratify=temp_df["label"], random_state=seed
        )
    else:
        # 使用分层采样
        try:
            train_df, temp_df = train_test_split(
                df, test_size=0.3, stratify=df["strata"], random_state=seed
            )
            val_df, test_df = train_test_split(
                temp_df, test_size=0.5, stratify=temp_df["strata"], random_state=seed
            )
        except ValueError as e:
            print(f"[WARNING] 分层采样失败: {e}")
            print("[WARNING] 回退到按label分层采样")
            # 回退到按label分层采样
            train_df, temp_df = train_test_split(
                df, test_size=0.3, stratify=df["label"], random_state=seed
            )
            val_df, test_df = train_test_split(
                temp_df, test_size=0.5, stratify=temp_df["label"], random_state=seed
            )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def check_split_distribution(df: pd.DataFrame, split_name: str) -> None:
    """检查split的类别分布"""
    if len(df) == 0:
        print(f"[WARNING] {split_name} split is empty!")
        return

    pos_count = (df["label"] == 1).sum()
    neg_count = (df["label"] == 0).sum()
    total = len(df)

    print(
        f"{split_name:10} | Total: {total:4d} | Pos: {pos_count:4d} ({pos_count/total*100:5.1f}%) | Neg: {neg_count:4d} ({neg_count/total*100:5.1f}%)"
    )

    # 检查是否只有单一类别
    if pos_count == 0 or neg_count == 0:
        print(
            f"[ERROR] {split_name} split has only one class! This will break evaluation."
        )
        raise ValueError(f"{split_name} split has only one class")


def write_csv(df: pd.DataFrame, path: Path) -> None:
    cols = REQUIRED_COLUMNS + [c for c in OPTIONAL_COLUMNS if c in df.columns]
    unique_cols = []
    for col in cols:
        if col not in unique_cols:
            unique_cols.append(col)
    extra_cols = [col for col in df.columns if col not in unique_cols]
    ordered = unique_cols + extra_cols
    df.loc[:, ordered].to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    ensure_required(df)
    df = ensure_etld(df)
    df = ensure_source(df, str(input_path.parent))
    df = normalize_brand(df)

    # 使用新的品牌选择函数
    b_ind, b_ood = select_balanced_brand_sets(
        df, args.top_k, args.min_neg_per_brand, args.min_pos_per_brand
    )
    if not b_ind:
        raise ValueError("No brands available for in-domain set.")

    df_ind = df[df["brand"].isin(b_ind)].copy()
    df_ood = df[df["brand"].isin(b_ood)].copy()

    # 使用新的分层采样函数
    train_df, val_df, test_id_df = stratified_split_by_brand_label(df_ind, args.seed)

    if len(df_ood) == 0:
        test_ood_df = pd.DataFrame(columns=df.columns)
    else:
        target_ood = int(round(len(test_id_df) * args.ood_ratio))
        if target_ood <= 0:
            target_ood = min(len(df_ood), len(test_id_df))
        target_ood = min(len(df_ood), target_ood) or len(df_ood)
        test_ood_df = (
            df_ood.sample(n=target_ood, random_state=args.seed)
            .reset_index(drop=True)
            .copy()
        )

    # 检查每个split的分布
    print("\nSplit Distribution Check:")
    print("-" * 60)
    check_split_distribution(train_df, "Train")
    check_split_distribution(val_df, "Val")
    check_split_distribution(test_id_df, "Test_ID")
    check_split_distribution(test_ood_df, "Test_OOD")
    print("-" * 60)

    # 保存CSV文件
    write_csv(train_df, output_dir / "train.csv")
    write_csv(val_df, output_dir / "val.csv")
    write_csv(test_id_df, output_dir / "test_id.csv")
    write_csv(test_ood_df, output_dir / "test_ood.csv")

    brand_sets = {"b_ind": sorted(b_ind), "b_ood": sorted(b_ood)}
    with open(output_dir / "brand_sets.json", "w", encoding="utf-8") as f:
        json.dump(brand_sets, f, indent=2, ensure_ascii=False)

    # 保存分布统计
    split_stats = {
        "train": {
            "count": len(train_df),
            "pos_count": int((train_df["label"] == 1).sum()),
            "neg_count": int((train_df["label"] == 0).sum()),
            "neg_ratio": float((train_df["label"] == 0).sum() / len(train_df)),
        },
        "val": {
            "count": len(val_df),
            "pos_count": int((val_df["label"] == 1).sum()),
            "neg_count": int((val_df["label"] == 0).sum()),
            "neg_ratio": float((val_df["label"] == 0).sum() / len(val_df)),
        },
        "test_id": {
            "count": len(test_id_df),
            "pos_count": int((test_id_df["label"] == 1).sum()),
            "neg_count": int((test_id_df["label"] == 0).sum()),
            "neg_ratio": float((test_id_df["label"] == 0).sum() / len(test_id_df)),
        },
        "test_ood": {
            "count": len(test_ood_df),
            "pos_count": int((test_ood_df["label"] == 1).sum()),
            "neg_count": int((test_ood_df["label"] == 0).sum()),
            "neg_ratio": float((test_ood_df["label"] == 0).sum() / len(test_ood_df)),
        },
        "parameters": {
            "top_k": args.top_k,
            "min_neg_per_brand": args.min_neg_per_brand,
            "min_pos_per_brand": args.min_pos_per_brand,
            "seed": args.seed,
            "ood_ratio": args.ood_ratio,
        },
    }

    with open(
        output_dir / "split_distribution_report.json", "w", encoding="utf-8"
    ) as f:
        json.dump(split_stats, f, indent=2, ensure_ascii=False)

    summary = {
        "train": len(train_df),
        "val": len(val_df),
        "test_id": len(test_id_df),
        "test_ood": len(test_ood_df),
        "num_in_domain_brands": len(b_ind),
        "num_ood_brands": len(b_ood),
    }
    print("\nBrand-OOD split summary:", summary)


if __name__ == "__main__":
    main()

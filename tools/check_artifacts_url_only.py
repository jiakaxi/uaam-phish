#!/usr/bin/env python3
"""
URL-Only 产物校验脚本
验证三协议（random/temporal/brand_ood）的四件套产物是否符合规范

运行: python tools/check_artifacts_url_only.py [experiment_dir]
"""

import json
import os
import sys
import glob
import csv
from pathlib import Path

REQUIRED_SPLIT_COLUMNS = [
    "split",
    "count",
    "pos_count",
    "neg_count",
    "brand_unique",
    "brand_set",
    "timestamp_min",
    "timestamp_max",
    "source_counts",
    "brand_intersection_ok",
    "tie_policy",
    "brand_normalization",
    "downgraded_to",
]


def read_json(p):
    """读取JSON文件"""
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def check_metrics_json(p, protocol):
    """检查 metrics_{protocol}.json 的 schema"""
    print(f"  [CHECK] Metrics JSON: {p}")
    j = read_json(p)

    required_top = [
        "accuracy",
        "auroc",
        "f1_macro",
        "nll",
        "ece",
        "ece_bins_used",
        "positive_class",
        "artifacts",
        "warnings",
    ]
    for k in required_top:
        assert k in j, f"{p}: missing key `{k}`"

    assert (
        j["positive_class"] == "phishing"
    ), f"{p}: positive_class must be 'phishing', got '{j['positive_class']}'"

    arts = j["artifacts"]
    for k in ["roc_path", "calib_path", "splits_path"]:
        assert k in arts, f"{p}: artifacts missing key `{k}`"
        # Note: paths might be relative, so we just check they exist
        art_path = arts[k]
        if art_path and not os.path.isabs(art_path):
            # Try relative to results dir
            results_dir = os.path.dirname(p)
            art_path = os.path.join(os.path.dirname(results_dir), art_path)
        if art_path and not os.path.exists(art_path):
            print(f"    [WARNING] Artifact path not found: {arts[k]}")

    bins = j["ece_bins_used"]
    assert isinstance(bins, (int, float)), f"{p}: ece_bins_used must be numeric"
    bins = int(bins)
    assert 3 <= bins <= 15, f"{p}: ece_bins_used out of range [3,15], got {bins}"

    print("    [OK] Metrics JSON schema valid")
    print(f"       - accuracy: {j['accuracy']:.4f}")
    print(f"       - auroc: {j['auroc']:.4f}")
    print(f"       - ece: {j['ece']:.4f} (bins={bins})")
    return j


def check_splits_csv(p, protocol):
    """检查 splits_{protocol}.csv 的列完整性"""
    print(f"  [CHECK] Splits CSV: {p}")
    with open(p, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        header = r.fieldnames or []
        for col in REQUIRED_SPLIT_COLUMNS:
            assert col in header, f"{p}: missing column `{col}`"
        rows = list(r)

    print(f"    [OK] Splits CSV has all required columns ({len(rows)} splits)")

    # 协议特定检查
    if protocol == "brand_ood":
        # brand_intersection_ok 应该为 true（品牌不相交）
        bool_vals = [
            str(row.get("brand_intersection_ok", "")).strip().lower() for row in rows
        ]
        has_true = any(v in ("true", "1", "yes") for v in bool_vals)
        if not has_true:
            print(
                f"    ⚠️  WARNING: brand_intersection_ok should be 'true' for brand_ood (got: {bool_vals})"
            )
        else:
            print("       - brand_intersection_ok: [OK] true")

    if protocol == "temporal":
        # tie_policy 应该包含 left-closed
        tie_vals = [str(row.get("tie_policy", "")).lower() for row in rows]
        has_left_closed = any(
            "left-closed" in v or "left_closed" in v for v in tie_vals
        )
        if not has_left_closed:
            print(
                f"    ⚠️  WARNING: tie_policy should include 'left-closed' for temporal (got: {tie_vals})"
            )
        else:
            print("       - tie_policy: [OK] left-closed")

    # 检查降级情况
    downgraded = [str(row.get("downgraded_to", "")).strip() for row in rows]
    if any(downgraded):
        print(f"       - downgraded_to: {downgraded[0] if downgraded[0] else 'None'}")

    return rows


def check_image_exists(p, name):
    """检查图片文件存在"""
    print(f"  [CHECK] {name}: {p}")
    assert os.path.exists(p), f"Missing {name}: {p}"
    assert os.path.getsize(p) > 0, f"{name} is empty: {p}"
    print(f"    [OK] {name} exists ({os.path.getsize(p)} bytes)")


def guess_run_root(exp_dir=None):
    """猜测最新的 experiments/*/results 目录"""
    if exp_dir:
        results = Path(exp_dir) / "results"
        if results.exists():
            return str(results)
        elif Path(exp_dir).name == "results":
            return str(exp_dir)

    # 取最新的 experiments/*/results 目录
    candidates = sorted(glob.glob("experiments/*/results"), key=os.path.getmtime)
    if not candidates:
        raise SystemExit(
            "[ERROR] No results directories found under experiments/*/results"
        )
    return candidates[-1]


def check_protocol(run_root, protocol):
    """检查单个协议的四件套"""
    print(f"\n{'='*60}")
    print(f"Protocol: {protocol}")
    print(f"{'='*60}")

    roc = os.path.join(run_root, f"roc_{protocol}.png")
    cal = os.path.join(run_root, f"calib_{protocol}.png")
    sp = os.path.join(run_root, f"splits_{protocol}.csv")
    mj = os.path.join(run_root, f"metrics_{protocol}.json")

    # 检查四件套存在
    for p, name in [
        (roc, "ROC"),
        (cal, "Calibration"),
        (sp, "Splits CSV"),
        (mj, "Metrics JSON"),
    ]:
        if not os.path.exists(p):
            print(f"  [MISSING] {name} at {p}")
            return False

    # 逐项检查
    try:
        check_image_exists(roc, "ROC curve")
        check_image_exists(cal, "Calibration curve")
        check_splits_csv(sp, protocol)
        check_metrics_json(mj, protocol)

        print(f"\n[SUCCESS] Protocol '{protocol}' artifacts validated!\n")
        return True
    except AssertionError as e:
        print(f"\n[FAILED] Protocol '{protocol}' validation FAILED: {e}\n")
        return False
    except Exception as e:
        print(f"\n[ERROR] Protocol '{protocol}' validation ERROR: {e}\n")
        return False


def main():
    """主函数"""
    print("=" * 60)
    print("URL-Only 产物校验脚本")
    print("=" * 60)

    exp_dir = sys.argv[1] if len(sys.argv) > 1 else None
    run_root = guess_run_root(exp_dir)
    print(f"\nValidating results in: {run_root}\n")

    protocols = ["random", "temporal", "brand_ood"]
    results = {}

    for protocol in protocols:
        results[protocol] = check_protocol(run_root, protocol)

    # 汇总
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for protocol, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {protocol:15s}: {status}")

    all_passed = all(results.values())
    if all_passed:
        print("\n[SUCCESS] All protocols passed validation!")
        sys.exit(0)
    else:
        print("\n[WARNING] Some protocols failed validation. Check logs above.")
        sys.exit(1)


if __name__ == "__main__":
    main()

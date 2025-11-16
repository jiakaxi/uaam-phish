#!/usr/bin/env python
"""Check experiment types from config files."""
import yaml
from pathlib import Path

exps = sorted(Path("experiments").glob("url_mvp_20251110_1*"))
for exp in exps:
    config_path = exp / "config.yaml"
    if not config_path.exists():
        continue
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    system_target = config.get("system", {}).get("_target_", "unknown").split(".")[-1]
    tags = config.get("run", {}).get("tags", [])
    train_csv = config.get("datamodule", {}).get("train_csv", "unknown")
    dataset = "unknown"
    if "brandood" in train_csv:
        dataset = "Brand-OOD"
    elif "iid" in train_csv:
        dataset = "IID"
    print(f"{exp.name}:")
    print(f"  System: {system_target}")
    print(f"  Tags: {tags}")
    print(f"  Dataset: {dataset}")
    print(f"  Train CSV: {train_csv.split('/')[-1]}")
    print()



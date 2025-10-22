from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from omegaconf import OmegaConf

from src.data.url_dataset import encode_url
from src.systems.url_only_module import UrlOnlyModule


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict with the URL-only Lightning model."
    )
    parser.add_argument(
        "--config-path",
        default="configs",
        help="Directory containing the configuration files.",
    )
    parser.add_argument(
        "--config-name",
        default="default.yaml",
        help="Base config file name located under --config-path.",
    )
    parser.add_argument(
        "--profile",
        default=None,
        help="Optional profile (under configs/profiles/) to merge.",
    )
    parser.add_argument(
        "--checkpoint",
        default="experiments/url_only/checkpoints/url-only-best.ckpt",
        help="Path to the Lightning checkpoint.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--url", help="Single URL string to classify.")
    group.add_argument(
        "--test",
        help="CSV file for batch inference. Must contain 'url_text' and optional 'label'.",
    )
    parser.add_argument(
        "--out",
        help="Output CSV path when using --test.",
    )
    return parser.parse_args()


def load_config(
    config_path: str, config_name: str, profile: Optional[str]
) -> OmegaConf:
    base_path = Path(config_path) / config_name
    # 确保有 .yaml 扩展名
    if not str(base_path).endswith(".yaml"):
        base_path = Path(str(base_path) + ".yaml")

    if not base_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {base_path}")

    cfg = OmegaConf.load(base_path)
    if profile:
        profile_path = Path(config_path) / "profiles" / f"{profile}.yaml"
        profile_cfg = OmegaConf.load(profile_path)
        cfg = OmegaConf.merge(cfg, profile_cfg)
    return cfg


def load_model(checkpoint_path: str | Path, cfg: OmegaConf) -> UrlOnlyModule:
    return UrlOnlyModule.load_from_checkpoint(
        str(checkpoint_path), cfg=cfg, map_location="cpu"
    )


def predict_single(module: UrlOnlyModule, text: str, cfg) -> list[float]:
    module.eval()
    with torch.no_grad():
        input_tensor = torch.tensor(
            encode_url(
                text,
                max_len=cfg.model.max_len,
                vocab_size=cfg.model.vocab_size,
                pad_id=cfg.model.pad_id,
            ),
            dtype=torch.long,
        ).unsqueeze(0)
        logits = module.predict_logits(input_tensor)
        probs = torch.softmax(logits, dim=-1).squeeze(0)
        return [float(probs[0]), float(probs[1])]


def predict_batch(
    module: UrlOnlyModule, csv_path: str | Path, out_path: str | Path, cfg
) -> None:
    """批量预测并保存结果到CSV"""
    df = pd.read_csv(csv_path)
    if "url_text" not in df.columns:
        raise ValueError("Input CSV must contain a 'url_text' column.")

    print(f"正在处理 {len(df)} 条样本...")
    records = []
    module.eval()
    with torch.no_grad():
        for idx, row in df.iterrows():
            text = str(row.get("url_text", ""))
            encoded = torch.tensor(
                encode_url(
                    text,
                    max_len=cfg.model.max_len,
                    vocab_size=cfg.model.vocab_size,
                    pad_id=cfg.model.pad_id,
                ),
                dtype=torch.long,
            ).unsqueeze(0)
            logits = module.predict_logits(encoded)
            probs = torch.softmax(logits, dim=-1).squeeze(0)
            entry = {
                "idx": int(idx),
                "legit_prob": float(probs[0]),
                "phish_prob": float(probs[1]),
            }
            if "label" in df.columns:
                entry["label"] = int(row["label"])
            records.append(entry)

            # 简单进度提示
            if (idx + 1) % 10 == 0:
                print(f"  已处理: {idx + 1}/{len(df)}", end="\r")

    print("\n预测完成！")

    columns = ["idx"]
    if "label" in df.columns:
        columns.append("label")
    columns.extend(["legit_prob", "phish_prob"])
    output = pd.DataFrame(records)[columns]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(out_path, index=False)
    print(f"结果已保存到: {out_path}")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config_path, args.config_name, args.profile)
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    module = load_model(checkpoint_path, cfg)

    if args.url is not None:
        probs = predict_single(module, args.url, cfg)
        print(json.dumps(probs))
    else:
        if not args.out:
            raise ValueError("--out is required when using --test.")
        predict_batch(module, args.test, args.out, cfg)


if __name__ == "__main__":
    main()

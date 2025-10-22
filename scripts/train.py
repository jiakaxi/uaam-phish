from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning import seed_everything
from omegaconf import OmegaConf

from src.datamodules.url_datamodule import UrlDataModule
from src.systems.url_only_module import UrlOnlyModule


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the URL-only Lightning model.")
    parser.add_argument(
        "--config-path",
        default="configs",
        help="Directory containing the base config.",
    )
    parser.add_argument(
        "--config-name",
        default="default.yaml",
        help="Config file name located under --config-path.",
    )
    parser.add_argument(
        "--profile",
        default=None,
        help="Optional profile name under configs/profiles/ to merge.",
    )
    return parser.parse_args()


def load_config(
    config_path: str, config_name: str, profile: Optional[str]
) -> OmegaConf:
    # Add .yaml extension if not present
    if not config_name.endswith(".yaml"):
        config_name = f"{config_name}.yaml"
    base_path = Path(config_path) / config_name
    cfg = OmegaConf.load(base_path)
    if profile:
        profile_path = Path(config_path) / "profiles" / f"{profile}.yaml"
        profile_cfg = OmegaConf.load(profile_path)
        cfg = OmegaConf.merge(cfg, profile_cfg)
    return cfg


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config_path, args.config_name, args.profile)

    seed_everything(cfg.seed, workers=True)

    datamodule = UrlDataModule(cfg)
    model = UrlOnlyModule(cfg)

    checkpoint_dir = Path("experiments/url_only/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=cfg.train.patience, mode="min"),
        ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            dirpath=str(checkpoint_dir),
            filename="url-only-best",
        ),
    ]

    trainer = pl.Trainer(
        max_epochs=cfg.train.epochs,
        accelerator="cpu",
        devices=1,
        deterministic=True,
        log_every_n_steps=10,
        callbacks=callbacks,
        default_root_dir="experiments",
        limit_train_batches=cfg.train.get("limit_train_batches", 1.0),
        limit_val_batches=cfg.train.get("limit_val_batches", 1.0),
    )

    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()

import os
from types import SimpleNamespace

import pandas as pd
import torch

from src.datamodules.url_datamodule import UrlDataModule


def test_datamodule_smoke(tmp_path, monkeypatch):
    # Create tiny CSVs
    df = pd.DataFrame({"url_text": ["a.com", "b.com"], "label": [0, 1]})
    train_csv = tmp_path / "train.csv"
    val_csv = tmp_path / "val.csv"
    df.to_csv(train_csv, index=False)
    df.to_csv(val_csv, index=False)

    cache_dir = tmp_path / "hf-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Create minimal config
    cfg = SimpleNamespace(
        paths=SimpleNamespace(
            train_index=str(train_csv),
            val_index=str(val_csv),
            test_index=str(val_csv),
        ),
        data=SimpleNamespace(
            text_col="url_text",
            label_col="label",
            max_length=8,
            sample_fraction=1.0,
        ),
        model=SimpleNamespace(
            pretrained_name="prajjwal1/bert-tiny",
            cache_dir=str(cache_dir),
            local_files_only=True,
        ),
        train=SimpleNamespace(
            bs=2,
            seed=1,
        ),
        hardware=SimpleNamespace(
            num_workers=0,
        ),
    )

    class DummyTokenizer:
        def __call__(self, text, padding, truncation, max_length, return_tensors):
            token_ids = torch.arange(max_length, dtype=torch.long).unsqueeze(0)
            return {
                "input_ids": token_ids,
                "attention_mask": torch.ones_like(token_ids),
            }

    monkeypatch.setattr(
        "src.datamodules.url_datamodule.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: DummyTokenizer(),
    )

    dm = UrlDataModule(cfg)
    dm.setup(stage="fit")
    batch = next(iter(dm.train_dataloader()))
    assert "input_ids" in batch and "label" in batch

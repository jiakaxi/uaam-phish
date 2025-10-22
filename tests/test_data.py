from omegaconf import OmegaConf
import pandas as pd
import torch

from src.datamodules.url_datamodule import UrlDataModule


def _build_cfg(train_csv: str, val_csv: str, test_csv: str):
    return OmegaConf.create(
        {
            "seed": 42,
            "data": {
                "train_csv": train_csv,
                "val_csv": val_csv,
                "test_csv": test_csv,
                "num_workers": 0,
            },
            "model": {
                "vocab_size": 128,
                "pad_id": 0,
                "max_len": 32,
                "embedding_dim": 16,
                "hidden_dim": 8,
                "num_layers": 1,
                "bidirectional": True,
                "dropout": 0.0,
                "proj_dim": 16,
                "num_classes": 2,
            },
            "train": {
                "batch_size": 2,
                "lr": 1e-3,
                "epochs": 1,
                "patience": 1,
            },
        }
    )


def test_datamodule_returns_padded_batches(tmp_path):
    df = pd.DataFrame({"url_text": ["abc", "xyz"], "label": [0, 1]})
    train_csv = tmp_path / "train.csv"
    val_csv = tmp_path / "val.csv"
    df.to_csv(train_csv, index=False)
    df.to_csv(val_csv, index=False)

    cfg = _build_cfg(str(train_csv), str(val_csv), str(val_csv))
    dm = UrlDataModule(cfg)
    dm.setup("fit")

    batch = next(iter(dm.train_dataloader()))
    inputs, labels = batch
    assert inputs.shape == (2, 32)
    assert inputs.dtype == torch.long
    # Labels may be shuffled, so check they contain both 0 and 1
    assert set(labels.tolist()) == {0, 1}
    assert torch.all(inputs[:, -1] == torch.tensor([0, 0]))

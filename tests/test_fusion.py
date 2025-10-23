from omegaconf import OmegaConf
import torch

from src.systems.url_only_module import UrlOnlyModule


def _cfg():
    return OmegaConf.create(
        {
            "seed": 123,
            "data": {
                "train_csv": "data/processed/url_train.csv",
                "val_csv": "data/processed/url_val.csv",
                "test_csv": "data/processed/url_test.csv",
                "num_workers": 0,
            },
            "model": {
                "vocab_size": 64,
                "pad_id": 0,
                "max_len": 16,
                "embedding_dim": 8,
                "hidden_dim": 128,
                "num_layers": 2,
                "bidirectional": True,
                "dropout": 0.0,
                "proj_dim": 256,
                "num_classes": 2,
            },
            "train": {
                "batch_size": 4,
                "lr": 1e-3,
                "epochs": 1,
                "patience": 1,
            },
        }
    )


def test_training_step_and_validation_step_logs_metrics():
    cfg = _cfg()
    module = UrlOnlyModule(cfg)

    batch_size = 4
    input_ids = torch.randint(
        0, cfg.model.vocab_size, (batch_size, cfg.model.max_len), dtype=torch.long
    )
    labels = torch.randint(0, 2, (batch_size,), dtype=torch.long)

    train_loss = module.training_step((input_ids, labels), 0)
    assert torch.isfinite(train_loss)

    val_out = module.validation_step((input_ids, labels), 0)
    assert "val_loss" in val_out
    assert "val_acc" in val_out

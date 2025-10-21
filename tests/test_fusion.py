import torch
import torch.nn as nn
from types import SimpleNamespace

from src.systems.url_only_module import UrlOnlySystem


def test_url_only_system_step(monkeypatch):
    # Create minimal config
    cfg = SimpleNamespace(
        model=SimpleNamespace(
            pretrained_name="prajjwal1/bert-tiny",
            dropout=0.1,
            cache_dir="tests/.cache",
            local_files_only=True,
        ),
        train=SimpleNamespace(
            lr=1e-3,
            weight_decay=0.0,
            pos_weight=1.0,
        ),
    )

    class DummyEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(hidden_size=8)

        def forward(self, batch):
            batch_size = batch["input_ids"].size(0)
            return torch.zeros(batch_size, self.config.hidden_size)

    monkeypatch.setattr(
        "src.systems.url_only_module.UrlBertEncoder",
        lambda *args, **kwargs: DummyEncoder(),
    )

    sys = UrlOnlySystem(cfg)

    # Create a simple batch (simulating tokenizer output)
    batch = {
        "input_ids": torch.randint(0, 1000, (2, 8)),
        "attention_mask": torch.ones(2, 8, dtype=torch.long),
        "label": torch.tensor([0.0, 1.0]),
    }

    loss = sys.step(batch, "train")
    assert torch.isfinite(loss)

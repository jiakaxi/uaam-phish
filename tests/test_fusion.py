import torch
from types import SimpleNamespace
from src.systems.url_only_module import UrlOnlySystem


def test_url_only_system_step():
    # Create minimal config
    cfg = SimpleNamespace(
        model=SimpleNamespace(
            pretrained_name="prajjwal1/bert-tiny",
            dropout=0.1,
        ),
        train=SimpleNamespace(
            lr=1e-3,
            weight_decay=0.0,
            pos_weight=1.0,
        ),
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

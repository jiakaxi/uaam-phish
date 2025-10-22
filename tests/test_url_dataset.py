import pandas as pd
import torch

from src.data.url_dataset import UrlDataset


def test_url_dataset_encodes_ascii_and_pads(tmp_path):
    csv_path = tmp_path / "sample.csv"
    pd.DataFrame({"url_text": ["abc", "Ωmega"], "label": [0, 1]}).to_csv(
        csv_path, index=False
    )

    dataset = UrlDataset(csv_path=csv_path, max_len=6, vocab_size=128, pad_id=0)
    inputs, label = dataset[1]

    assert inputs.shape == (6,)
    assert inputs.dtype == torch.long
    assert label.item() == 1
    # Non-ASCII should be clipped to vocab_size - 1 (127)
    assert inputs[0].item() == 127
    # Padding applied
    assert inputs[-1].item() == 0

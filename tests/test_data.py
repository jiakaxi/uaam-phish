import os
import pandas as pd
from src.datamodules.url_datamodule import UrlDataModule

def test_datamodule_smoke(tmp_path):
    # Create tiny CSV
    df = pd.DataFrame({"url_text": ["a.com", "b.com"], "label": [0, 1]})
    csv = tmp_path / "tiny.csv"
    df.to_csv(csv, index=False)

    dm = UrlDataModule(
        csv_path=str(csv),
        tokenizer_name="roberta-base",
        text_col="url_text",
        label_col="label",
        max_length=8,
        sample_fraction=1.0,
        seed=1,
        cache_dir=os.environ.get("HF_CACHE_DIR"),
        local_files_only=bool(int(os.environ.get("HF_LOCAL_ONLY", "0"))),
    )
    dm.setup(stage="fit")
    batch = next(iter(dm.train_dataloader()))
    assert "input_ids" in batch and "label" in batch

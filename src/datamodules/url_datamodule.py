from typing import Dict, Optional
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from transformers import AutoTokenizer

class UrlDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        tokenizer_name: str,
        text_col: str,
        label_col: str,
        max_length: int = 128,
        sample_fraction: float = 1.0,
        seed: int = 42,
        cache_dir: Optional[str] = None,
        local_files_only: bool = False,
    ):
        super().__init__()
        df = pd.read_csv(csv_path)
        if 0 < sample_fraction < 1.0:
            df = df.sample(frac=sample_fraction, random_state=seed).reset_index(drop=True)
        assert text_col in df.columns and label_col in df.columns, f"CSV must contain {text_col},{label_col}"
        self.texts = df[text_col].astype(str).tolist()
        self.labels = df[label_col].astype(int).tolist()
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name, cache_dir=cache_dir, local_files_only=local_files_only
            )
        except OSError as exc:
            raise RuntimeError(
                f"Tokenizer '{tokenizer_name}' is not cached locally. "
                "Download it first or set model.cache_dir / HF_CACHE_DIR."
            ) from exc
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        lab  = self.labels[idx]
        tok = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        item = {k: v.squeeze(0) for k, v in tok.items()}
        item["label"] = torch.tensor(lab, dtype=torch.float32)
        return item

class UrlDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.num_workers = cfg.hardware.num_workers
        self.sample_fraction = cfg.data.sample_fraction
        self.seed = cfg.train.seed

    def setup(self, stage=None):
        c = self.cfg
        mk = dict(
            tokenizer_name=c.model.pretrained_name,
            text_col=c.data.text_col,
            label_col=c.data.label_col,
            max_length=c.data.max_length,
            sample_fraction=self.sample_fraction,
            seed=self.seed,
            cache_dir=c.model.get("cache_dir"),
            local_files_only=bool(c.model.get("local_files_only", False)),
        )
        self.train_set = UrlDataset(c.paths.train_index, **mk)
        self.val_set   = UrlDataset(c.paths.val_index,   **mk)
        self.test_set  = UrlDataset(c.paths.test_index,  **mk)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.cfg.train.bs, shuffle=True,
                          num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.cfg.eval.bs, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.cfg.eval.bs, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True)

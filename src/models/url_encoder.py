from typing import Dict
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class UrlBertEncoder(nn.Module):
    """
    Wraps a pretrained BERT to produce a [B, H] embedding for a given tokenized URL string.
    Expects batch dict with keys: input_ids, attention_mask (and optional token_type_ids).
    """

    def __init__(
        self, pretrained_name: str = "bert-base-uncased", dropout: float = 0.1
    ):
        super().__init__()
        self.config = AutoConfig.from_pretrained(
            pretrained_name, output_hidden_states=False
        )
        self.backbone = AutoModel.from_pretrained(pretrained_name, config=self.config)
        self.dropout = nn.Dropout(dropout)
        # Optional projection if you want; for MVP we keep identity
        self.proj = nn.Identity()

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        out = self.backbone(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch.get("token_type_ids", None),
            return_dict=True,
        )
        # CLS representation
        x = out.last_hidden_state[:, 0, :]
        x = self.dropout(x)
        return self.proj(x)

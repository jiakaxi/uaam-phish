"""
Archived batch utilities (unused in S0 baseline).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import torch


def _unpack_batch(
    batch: Union[Tuple, Dict],
    batch_format: str = "tuple",
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Optional[Any]]]:
    meta = {"timestamp": None, "brand": None, "source": None}

    if batch_format == "tuple":
        if len(batch) == 2:
            inputs, labels = batch
        elif len(batch) == 3:
            inputs, labels, batch_meta = batch
            meta.update(batch_meta)
        else:
            raise ValueError(f"Expected batch of length 2 or 3, got {len(batch)}")
    elif batch_format == "dict":
        inputs = batch["inputs"]
        labels = batch["labels"]
        for key in ["timestamp", "brand", "source"]:
            if key in batch:
                meta[key] = batch[key]
    else:
        raise ValueError(f"Unknown batch_format: {batch_format}")

    return inputs, labels, meta


def collate_with_metadata(
    samples: list,
    include_metadata: bool = False,
) -> Union[Tuple, Dict]:
    if not include_metadata:
        inputs = torch.stack([s[0] for s in samples])
        labels = torch.stack([s[1] for s in samples])
        return inputs, labels

    inputs = torch.stack([s[0] for s in samples])
    labels = torch.stack([s[1] for s in samples])
    if len(samples[0]) > 2:
        meta = {
            "timestamp": [s[2].get("timestamp") for s in samples],
            "brand": [s[2].get("brand") for s in samples],
            "source": [s[2].get("source") for s in samples],
        }
        return inputs, labels, meta
    return inputs, labels

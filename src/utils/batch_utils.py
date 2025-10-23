"""
Batch formatting utilities for handling tuple and dict batch formats.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import torch


def _unpack_batch(
    batch: Union[Tuple, Dict],
    batch_format: str = "tuple",
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Optional[Any]]]:
    """
    Unpack batch into (inputs, labels, meta).

    Args:
        batch: Batch data (tuple or dict)
        batch_format: 'tuple' or 'dict'

    Returns:
        inputs: Tensor[B, ...]
        labels: Tensor[B]
        meta: Dict with keys {timestamp, brand, source}, values can be None
    """
    # Default meta (all None)
    meta = {
        "timestamp": None,
        "brand": None,
        "source": None,
    }

    if batch_format == "tuple":
        if len(batch) == 2:
            # Standard (inputs, labels)
            inputs, labels = batch
        elif len(batch) == 3:
            # With metadata: (inputs, labels, meta_dict)
            inputs, labels, batch_meta = batch
            meta.update(batch_meta)
        else:
            raise ValueError(f"Expected batch of length 2 or 3, got {len(batch)}")

    elif batch_format == "dict":
        inputs = batch["inputs"]
        labels = batch["labels"]
        # Optional metadata
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
    """
    Custom collate function that can optionally include metadata.

    Args:
        samples: List of samples from dataset
        include_metadata: Whether to include metadata in batch

    Returns:
        Batch (tuple or dict format)
    """
    if not include_metadata:
        # Standard collate: just stack tensors
        inputs = torch.stack([s[0] for s in samples])
        labels = torch.stack([s[1] for s in samples])
        return inputs, labels
    else:
        # With metadata (if available)
        inputs = torch.stack([s[0] for s in samples])
        labels = torch.stack([s[1] for s in samples])

        # Check if samples have metadata (length > 2)
        if len(samples[0]) > 2:
            # Gather metadata
            meta = {
                "timestamp": [
                    s[2].get("timestamp") if len(s) > 2 else None for s in samples
                ],
                "brand": [s[2].get("brand") if len(s) > 2 else None for s in samples],
                "source": [s[2].get("source") if len(s) > 2 else None for s in samples],
            }
            return inputs, labels, meta
        else:
            # No metadata available
            return inputs, labels

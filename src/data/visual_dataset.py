"""
Visual 数据集模块
用于 ResNet-based 视觉钓鱼检测
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Tuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.utils.logging import get_logger

log = get_logger(__name__)


class VisualDataset(Dataset):
    """
    Visual modality dataset with image transforms.
    Reads image files from paths in CSV and applies torchvision transforms.

    Returns:
        Tuple of (image_tensor, label) - all torch.Tensor

    Example:
        >>> from torchvision import transforms
        >>> transform = transforms.Compose([
        ...     transforms.Resize(256),
        ...     transforms.CenterCrop(224),
        ...     transforms.ToTensor(),
        ... ])
        >>> dataset = VisualDataset(
        ...     "data/processed/master_v2.csv",
        ...     transform=transform,
        ...     split="train"
        ... )
        >>> image, label = dataset[0]
        >>> print(image.shape)  # (3, 224, 224)
    """

    def __init__(
        self,
        csv_path: str | Path,
        *,
        transform: Optional[Callable] = None,
        image_col: str = "img_path",
        label_col: str = "label",
        id_col: str = "id",
        split: Optional[str] = None,
        split_col: str = "split",
    ) -> None:
        """
        Args:
            csv_path: Path to CSV containing img_path and label columns
            transform: torchvision transforms to apply to images
            image_col: Column name for image file paths
            label_col: Column name for labels (0=benign, 1=phishing)
            id_col: Column name for sample IDs
            split: If provided, filter by split column (e.g., "train", "val", "test")
            split_col: Column name for split information
        """
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")

        self.transform = transform
        self.image_col = image_col
        self.label_col = label_col
        self.id_col = id_col

        # 加载数据
        frame = pd.read_csv(self.csv_path)

        # Filter by split if provided
        if split is not None and split_col in frame.columns:
            frame = frame[frame[split_col] == split].reset_index(drop=True)
            log.info(f"Filtered to split='{split}': {len(frame)} samples")

        required = {image_col, label_col}
        if not required.issubset(frame.columns):
            missing = ", ".join(sorted(required - set(frame.columns)))
            raise ValueError(f"Missing required columns: {missing}")

        self._image_paths = frame[image_col].fillna("").astype(str).tolist()
        self._labels = frame[label_col].astype(int).tolist()

        # Store IDs if available (for embeddings export)
        if id_col in frame.columns:
            self._ids = frame[id_col].astype(str).tolist()
        else:
            self._ids = [str(i) for i in range(len(self._labels))]

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            image: (3, 224, 224) - Transformed image tensor
            label: (,) - Binary label (0 or 1)
        """
        image_path = self._image_paths[index]
        label = self._labels[index]

        # Load image with PIL
        try:
            # Handle relative paths (relative to DATA_ROOT or project root)
            img_path = Path(image_path)
            if not img_path.is_absolute():
                # Try relative to CSV directory first
                candidate = self.csv_path.parent / image_path
                if candidate.exists():
                    img_path = candidate
                else:
                    # Try as-is (may be relative to cwd)
                    img_path = Path(image_path)

            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            log.warning(
                f"Failed to load image {image_path}: {e}. Using black placeholder."
            )
            # Create black placeholder image (224x224)
            image = Image.new("RGB", (224, 224), color=(0, 0, 0))

        # Apply transforms
        if self.transform is not None:
            image = self.transform(image)
        else:
            # Default: just convert to tensor
            from torchvision import transforms

            image = transforms.ToTensor()(image)

        return image, torch.tensor(label, dtype=torch.long)

    def get_id(self, index: int) -> str:
        """Get sample ID for a given index (used for embeddings export)."""
        return self._ids[index]

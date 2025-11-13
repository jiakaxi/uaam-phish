#!/usr/bin/env python3
"""Test if MultimodalDataModule includes image_path in batch"""
from omegaconf import OmegaConf

from src.data.multimodal_datamodule import MultimodalDataModule

# Load config
cfg = OmegaConf.load("configs/experiment/s3_iid_fixed.yaml")
base_cfg = OmegaConf.load("configs/default.yaml")
cfg = OmegaConf.merge(base_cfg, cfg)

print("=" * 70)
print("Testing MultimodalDataModule image_path")
print("=" * 70)
print()

print("[1/3] Initializing DataModule...")
dm = MultimodalDataModule(
    train_csv=str(cfg.datamodule.train_csv),
    val_csv=str(cfg.datamodule.val_csv),
    test_csv=str(cfg.datamodule.test_csv),
    image_dir=str(cfg.datamodule.image_dir),
    batch_size=4,
    num_workers=0,
    url_max_len=cfg.datamodule.url_max_len,
    url_vocab_size=cfg.datamodule.url_vocab_size,
    html_max_len=cfg.datamodule.html_max_len,
)

print("  OK")

print("\n[2/3] Setting up test dataloader...")
dm.setup("test")
test_loader = dm.test_dataloader()
print(f"  Test batches: {len(test_loader)}")

print("\n[3/3] Checking first batch...")
batch = next(iter(test_loader))

print(f"  Batch keys: {list(batch.keys())}")
print()

if "image_path" in batch:
    image_paths = batch["image_path"]
    print("  ✓ image_path found in batch!")
    print(f"  Type: {type(image_paths)}")
    print(f"  Length: {len(image_paths) if hasattr(image_paths, '__len__') else 'N/A'}")

    if isinstance(image_paths, (list, tuple)):
        print("\n  First 3 paths:")
        for i, path in enumerate(image_paths[:3]):
            print(f"    [{i}] {path}")
    else:
        print(f"  Value: {image_paths}")
else:
    print("  ✗ image_path NOT in batch!")
    print("  This is the problem - image_path is not being included")

print()
print("=" * 70)
if "image_path" not in batch:
    print("PROBLEM: image_path missing from batch")
    print("Need to check MultimodalDataset.__getitem__ return value")
else:
    print("OK: image_path is in the batch")
print("=" * 70)

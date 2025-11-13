#!/usr/bin/env python3
"""End-to-end test: DataModule -> C-Module"""
from omegaconf import OmegaConf
from src.data.multimodal_datamodule import MultimodalDataModule
from src.modules.c_module import CModule

print("=" * 70)
print("End-to-End Visual Brand Extraction Test")
print("=" * 70)
print()

# Load config
cfg = OmegaConf.merge(
    OmegaConf.load("configs/default.yaml"),
    OmegaConf.load("configs/experiment/s3_iid_fixed.yaml"),
)

# Initialize DataModule
print("[1/4] Creating DataModule...")
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
dm.setup("test")
test_loader = dm.test_dataloader()
print("  OK")

# Initialize C-Module
print("\n[2/4] Creating C-Module...")
c_module = CModule(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    use_ocr=True,
    thresh=0.60,
    brand_lexicon_path="resources/brand_lexicon.txt",
)
print(f"  OK - use_ocr={c_module.use_ocr}")

# Get first batch
print("\n[3/4] Getting first batch...")
batch = next(iter(test_loader))
sample_ids = batch["id"]
image_paths = batch["image_path"]
print(f"  Batch size: {len(sample_ids)}")
print(f"  Has image_path: {image_paths is not None}")

# Test C-Module with first 3 samples
print("\n[4/4] Testing C-Module with batch data...")
for i in range(min(3, len(sample_ids))):
    sample_id = sample_ids[i]
    img_path = image_paths[i] if i < len(image_paths) else None

    print(f"\n  Sample {i+1}:")
    print(f"    ID: {sample_id}")
    print(f"    image_path from batch: {img_path[:60] if img_path else None}...")

    # Call C-Module score_consistency
    payload = {
        "sample_id": sample_id,
        "id": sample_id,
        "image_path": img_path,  # Pass image_path from batch
    }

    result = c_module.score_consistency(payload)
    brands = result.get("meta", {}).get("brands", {})
    brand_vis = brands.get("visual", "")

    print(f"    brand_visual result: '{brand_vis}'")

    if brand_vis:
        print(f"    *** SUCCESS! Extracted: {brand_vis}")
    else:
        sources = result.get("meta", {}).get("sources", {})
        visual_src = sources.get("visual", {})
        reason = visual_src.get("reason", "N/A")
        print(f"    *** FAILED! Reason: {reason}")

print()
print("=" * 70)
print("End-to-End Test Complete")
print("=" * 70)

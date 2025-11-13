#!/usr/bin/env python3
"""
快速测试：验证 MultimodalDataset 现在返回 image_path
"""
import sys

sys.path.insert(0, ".")

from hydra import initialize, compose
from src.data.multimodal_datamodule import MultimodalDataModule

print("=" * 70)
print("测试: MultimodalDataset 是否返回 image_path")
print("=" * 70)

# 初始化 Hydra
with initialize(version_base=None, config_path="configs"):
    cfg = compose(
        config_name="config",
        overrides=[
            "experiment=s3_iid_fixed",
            "data.protocol=random",
            "data.batch_size=4",
        ],
    )

# 创建 DataModule
dm = MultimodalDataModule(cfg=cfg)
dm.setup("test")

# 获取测试数据加载器
test_loader = dm.test_dataloader()

print("\n检查前 3 个样本...")
for i, batch in enumerate(test_loader):
    if i >= 1:  # 只检查第一个batch
        break

    print(f"\nBatch {i+1}:")
    print(f"  Batch keys: {list(batch.keys())}")
    print(f"  Batch size: {len(batch['id'])}")

    if "image_path" in batch:
        image_paths = batch["image_path"]
        print("  ✓ 'image_path' 存在于batch中")
        print(f"  image_path类型: {type(image_paths)}")

        # 检查具体的路径
        if isinstance(image_paths, (list, tuple)):
            print(f"  image_path长度: {len(image_paths)}")
            for j, path in enumerate(image_paths[:3]):
                if path:
                    print(f"    样本 {j+1}: {path[:80]}...")
                else:
                    print(f"    样本 {j+1}: None")
        else:
            print(f"  image_path值: {image_paths}")
    else:
        print("  ✗ 'image_path' 不在batch中！")
        print(f"  可用的keys: {list(batch.keys())}")

print("\n" + "=" * 70)
print("测试完成")
print("=" * 70)

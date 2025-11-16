#!/usr/bin/env python
"""测试 DataModule 能否正确加载预处理缓存"""

from pathlib import Path
import sys

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.multimodal_datamodule import MultimodalDataModule


def test_iid_loading():
    """测试 IID splits 的缓存加载"""
    print("测试 IID splits 缓存加载:")
    print("=" * 60)

    datamodule = MultimodalDataModule(
        train_csv="workspace/data/splits/iid/train_cached.csv",
        val_csv="workspace/data/splits/iid/val_cached.csv",
        test_csv="workspace/data/splits/iid/test_cached.csv",
        preprocessed_train_dir="workspace/data/preprocessed/iid/train",
        preprocessed_val_dir="workspace/data/preprocessed/iid/val",
        preprocessed_test_dir="workspace/data/preprocessed/iid/test",
        batch_size=4,
        num_workers=0,  # 使用 0 避免多进程问题
        url_max_len=200,
        url_vocab_size=128,
        html_max_len=256,
    )

    datamodule.setup("fit")

    # 测试训练数据加载
    train_loader = datamodule.train_dataloader()
    batch = next(iter(train_loader))

    print("训练 batch 形状:")
    print(f"  - inputs: {type(batch)}")
    if isinstance(batch, tuple) and len(batch) >= 2:
        print(
            f"  - labels shape: {batch[1].shape if hasattr(batch[1], 'shape') else type(batch[1])}"
        )

    print("[通过] IID 训练数据加载成功")

    # 测试验证数据加载
    val_loader = datamodule.val_dataloader()
    next(iter(val_loader))
    print("[通过] IID 验证数据加载成功")

    print("=" * 60)
    return True


def test_brandood_loading():
    """测试 Brand-OOD splits 的缓存加载"""
    print("\n测试 Brand-OOD splits 缓存加载:")
    print("=" * 60)

    datamodule = MultimodalDataModule(
        train_csv="workspace/data/splits/brandood/train_cached.csv",
        val_csv="workspace/data/splits/brandood/val_cached.csv",
        test_csv="workspace/data/splits/brandood/test_id_cached.csv",
        test_ood_csv="workspace/data/splits/brandood/test_ood_cached.csv",
        preprocessed_train_dir="workspace/data/preprocessed/brandood/train",
        preprocessed_val_dir="workspace/data/preprocessed/brandood/val",
        preprocessed_test_dir="workspace/data/preprocessed/brandood/test_id",
        batch_size=4,
        num_workers=0,
        url_max_len=200,
        url_vocab_size=128,
        html_max_len=256,
    )

    datamodule.setup("fit")

    # 测试训练数据加载
    train_loader = datamodule.train_dataloader()
    next(iter(train_loader))
    print("[通过] Brand-OOD 训练数据加载成功")

    # 测试验证数据加载
    val_loader = datamodule.val_dataloader()
    next(iter(val_loader))
    print("[通过] Brand-OOD 验证数据加载成功")

    print("=" * 60)
    return True


def main():
    try:
        test_iid_loading()
        test_brandood_loading()
        print("\n[成功] 所有缓存加载测试通过")
        return True
    except Exception as e:
        print(f"\n[失败] 缓存加载测试失败: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

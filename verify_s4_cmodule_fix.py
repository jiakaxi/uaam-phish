#!/usr/bin/env python
"""
快速验证 S4 C-Module 修复是否生效
"""
import sys
import torch
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import hydra
from src.data.multimodal_datamodule import MultimodalDataModule


def test_s4_cmodule_inline_fallback():
    """测试 S4 系统的 C-Module inline 文本 fallback"""
    print("\n" + "=" * 60)
    print("验证 S4 C-Module 修复")
    print("=" * 60)

    # Load config
    with hydra.initialize(version_base="1.3", config_path="configs"):
        cfg = hydra.compose(
            config_name="config",
            overrides=[
                "experiment=s4_iid_rcaf",
                "trainer.fast_dev_run=true",
            ],
        )

    print("\n[1] 加载 DataModule...")
    dm = MultimodalDataModule(**cfg.datamodule)
    dm.setup("fit")
    train_loader = dm.train_dataloader()

    print(f"[OK] Train dataset: {len(dm.train_dataset)} samples")

    # Get one batch
    batch = next(iter(train_loader))
    batch_size = len(batch["id"])
    print(f"[OK] Batch size: {batch_size}")

    print("\n[2] 初始化 S4 系统...")
    from src.systems.s4_rcaf_system import S4RCAFSystem

    system = S4RCAFSystem(
        cfg=cfg, **{k: v for k, v in cfg.system.items() if not k.startswith("_")}
    )

    print("[OK] S4RCAFSystem 初始化完成")
    print(f"[OK] C-Module metadata sources: {len(system.c_module._registered_sources)}")

    print("\n[3] 测试 URL 解码...")
    urls = system._decode_url_tokens(batch.get("url"))
    print(f"[OK] 解码 {len(urls)} 个 URLs")
    print(f"  示例: {urls[0][:50]}..." if urls else "  (空)")

    print("\n[4] 测试一致性计算...")
    system.eval()
    with torch.no_grad():
        c_m = system._compute_consistency_batch(batch)

    print(f"[OK] c_m shape: {c_m.shape}")
    print("[OK] c_m 统计:")
    print(f"  - min: {c_m.min().item():.4f}")
    print(f"  - max: {c_m.max().item():.4f}")
    print(f"  - mean: {c_m.mean().item():.4f}")
    print(f"  - 是否包含 NaN: {torch.isnan(c_m).any().item()}")
    print(f"  - 是否包含 Inf: {torch.isinf(c_m).any().item()}")
    print(f"  - 有限值比例: {torch.isfinite(c_m).float().mean().item():.2%}")

    print("\n[5] 测试完整 forward pass...")
    outputs = system.shared_step(batch, 0)

    print("[OK] 输出检查:")
    print(f"  - probs shape: {outputs['probs'].shape}")
    print(f"  - alpha_m shape: {outputs['alpha_m'].shape}")
    print(f"  - lambda_c shape: {outputs['lambda_c'].shape}")
    print(f"  - lambda_c 是否有限: {torch.isfinite(outputs['lambda_c']).all().item()}")
    print(f"  - lambda_c mean: {outputs['lambda_c'].mean().item():.4f}")

    print("\n[6] 结果:")
    nan_count = torch.isnan(c_m).sum().item()
    total_count = c_m.numel()

    if nan_count == 0:
        print("[SUCCESS] C-Module 返回有效一致性分数（无 NaN）")
        print("[SUCCESS] Lambda_c 计算正常")
        print("[SUCCESS] 修复生效！")
        return True
    else:
        print(f"[FAIL] C-Module 仍返回 {nan_count}/{total_count} 个 NaN")
        print("[FAIL] 需要进一步调试")
        return False


if __name__ == "__main__":
    try:
        success = test_s4_cmodule_inline_fallback()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n[ERROR] 测试失败: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

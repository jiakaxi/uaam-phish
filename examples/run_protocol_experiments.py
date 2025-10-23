"""
示例：运行不同协议的实验

演示如何使用 build_splits 和协议工件生成功能。
"""

from pathlib import Path
import pandas as pd
from omegaconf import OmegaConf

from src.utils.splits import build_splits, write_split_table
from src.utils.logging import get_logger

log = get_logger(__name__)


def run_protocol_split_example():
    """
    演示如何使用不同的协议分割数据。

    注意：这是一个独立示例，不影响现有训练流程。
    """
    # 示例配置
    cfg = OmegaConf.create(
        {
            "protocol": "random",
            "data": {
                "split_ratios": {
                    "train": 0.7,
                    "val": 0.15,
                    "test": 0.15,
                }
            },
        }
    )

    # 加载数据（示例）
    master_csv = Path("data/processed/master.csv")

    if not master_csv.exists():
        log.warning(f"Master CSV not found: {master_csv}")
        log.warning("This is a demonstration script. Run data preprocessing first.")
        return

    df = pd.read_csv(master_csv)
    log.info(f"Loaded {len(df)} samples from {master_csv}")

    # 测试三种协议
    for protocol in ["random", "temporal", "brand_ood"]:
        log.info(f"\n{'='*60}")
        log.info(f"Testing protocol: {protocol}")
        log.info(f"{'='*60}")

        try:
            # 执行分割
            train_df, val_df, test_df, metadata = build_splits(
                df=df,
                cfg=OmegaConf.merge(cfg, {"protocol": protocol}),
                protocol=protocol,
            )

            # 显示结果
            log.info(
                f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}"
            )

            if metadata.get("downgraded_to"):
                log.warning(f"⚠️  Downgraded to: {metadata['downgraded_to']}")
                log.warning(f"   Reason: {metadata['downgrade_reason']}")
            else:
                log.info(f"✓ Protocol {protocol} executed successfully")

            # 保存分割表
            output_dir = Path(f"examples/output/{protocol}")
            output_dir.mkdir(parents=True, exist_ok=True)

            splits_csv = output_dir / f"splits_{protocol}.csv"
            write_split_table(metadata["split_stats"], splits_csv)
            log.info(f"Splits table saved: {splits_csv}")

            # 保存分割数据
            train_df.to_csv(output_dir / f"train_{protocol}.csv", index=False)
            val_df.to_csv(output_dir / f"val_{protocol}.csv", index=False)
            test_df.to_csv(output_dir / f"test_{protocol}.csv", index=False)

        except Exception as e:
            log.error(f"Error with protocol {protocol}: {e}")

    log.info(f"\n{'='*60}")
    log.info("Protocol split demonstration completed!")
    log.info("Check examples/output/ for results")
    log.info(f"{'='*60}")


if __name__ == "__main__":
    run_protocol_split_example()

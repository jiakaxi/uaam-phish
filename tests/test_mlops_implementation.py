"""
测试 MLOps 协议实现的所有功能
"""

import pytest
import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from src.utils.splits import build_splits, _compute_split_stats
from src.utils.metrics import compute_ece, compute_nll, get_step_metrics
from src.utils.batch_utils import _unpack_batch


class TestDataSplits:
    """测试数据分割功能"""

    def test_random_split(self):
        """测试随机分割"""
        df = pd.DataFrame(
            {
                "url_text": [f"url_{i}" for i in range(100)],
                "label": [i % 2 for i in range(100)],
            }
        )

        cfg = OmegaConf.create(
            {
                "protocol": "random",
                "data": {"split_ratios": {"train": 0.7, "val": 0.15, "test": 0.15}},
            }
        )

        train_df, val_df, test_df, metadata = build_splits(df, cfg, "random")

        assert len(train_df) + len(val_df) + len(test_df) == 100
        assert metadata["protocol"] == "random"
        assert metadata["downgraded_to"] is None

    def test_temporal_split(self):
        """测试时间序列分割"""
        df = pd.DataFrame(
            {
                "url_text": [f"url_{i}" for i in range(100)],
                "label": [i % 2 for i in range(100)],
                "timestamp": pd.date_range("2023-01-01", periods=100, freq="D"),
            }
        )

        cfg = OmegaConf.create(
            {
                "protocol": "temporal",
                "data": {"split_ratios": {"train": 0.7, "val": 0.15, "test": 0.15}},
            }
        )

        train_df, val_df, test_df, metadata = build_splits(df, cfg, "temporal")

        assert len(train_df) == 70
        assert metadata["tie_policy"] == "left-closed"

        # 验证时间顺序
        assert train_df["timestamp"].max() <= val_df["timestamp"].min()
        assert val_df["timestamp"].max() <= test_df["timestamp"].min()

    def test_temporal_downgrade_missing_column(self):
        """测试temporal协议在缺少timestamp时降级"""
        df = pd.DataFrame(
            {
                "url_text": [f"url_{i}" for i in range(100)],
                "label": [i % 2 for i in range(100)],
            }
        )

        cfg = OmegaConf.create(
            {
                "protocol": "temporal",
                "data": {"split_ratios": {"train": 0.7, "val": 0.15, "test": 0.15}},
            }
        )

        train_df, val_df, test_df, metadata = build_splits(df, cfg, "temporal")

        assert metadata["downgraded_to"] == "random"
        assert "Missing timestamp column" in metadata["downgrade_reason"]

    def test_brand_ood_split(self):
        """测试品牌OOD分割"""
        brands = ["brand_a", "brand_b", "brand_c", "brand_d", "brand_e"]
        df = pd.DataFrame(
            {
                "url_text": [f"url_{i}" for i in range(100)],
                "label": [i % 2 for i in range(100)],
                "brand": [brands[i % 5] for i in range(100)],
            }
        )

        cfg = OmegaConf.create(
            {
                "protocol": "brand_ood",
                "data": {"split_ratios": {"train": 0.7, "val": 0.15, "test": 0.15}},
            }
        )

        train_df, val_df, test_df, metadata = build_splits(df, cfg, "brand_ood")

        # 验证品牌不相交
        train_brands = set(train_df["brand"].unique())
        test_brands = set(test_df["brand"].unique())

        assert len(train_brands & test_brands) == 0
        assert metadata["brand_normalization"] == "strip+lower"
        assert metadata["split_stats"]["brand_intersection_ok"]


class TestMetrics:
    """测试指标计算功能"""

    def test_compute_ece(self):
        """测试ECE计算"""
        y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0] * 10)
        y_prob = np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4, 0.5, 0.5] * 10)

        ece_value, bins_used = compute_ece(y_true, y_prob, n_bins=None, pos_label=1)

        assert 0.0 <= ece_value <= 1.0
        assert 3 <= bins_used <= 15

    def test_compute_ece_adaptive_bins(self):
        """测试ECE自适应bins"""
        # 小样本
        y_true = np.array([1, 0, 1, 0, 1])
        y_prob = np.array([0.9, 0.1, 0.8, 0.2, 0.7])

        _, bins_used = compute_ece(y_true, y_prob, n_bins=None)
        assert bins_used == 3  # max(3, min(15, floor(sqrt(5)), 10)) = 3

        # 大样本
        y_true = np.array([1, 0] * 100)
        y_prob = np.random.rand(200)

        _, bins_used = compute_ece(y_true, y_prob, n_bins=None)
        assert bins_used == 10  # max(3, min(15, floor(sqrt(200)), 10)) = 10

    def test_compute_nll(self):
        """测试NLL计算"""
        import torch

        logits = torch.tensor([[2.0, 1.0], [1.0, 2.0], [3.0, 0.5]])
        labels = torch.tensor([0, 1, 0])

        nll = compute_nll(logits, labels)

        assert nll > 0.0
        assert isinstance(nll, float)

    def test_get_step_metrics(self):
        """测试Step级指标获取"""
        metrics = get_step_metrics(num_classes=2, average="macro", sync_dist=False)

        assert "accuracy" in metrics
        assert "auroc" in metrics
        assert "f1" in metrics


class TestBatchUtils:
    """测试Batch工具函数"""

    def test_unpack_batch_tuple_format(self):
        """测试tuple格式解包"""
        import torch

        # 标准tuple格式 (inputs, labels)
        inputs = torch.rand(4, 10)
        labels = torch.randint(0, 2, (4,))
        batch = (inputs, labels)

        unpacked_inputs, unpacked_labels, meta = _unpack_batch(batch, "tuple")

        assert torch.equal(unpacked_inputs, inputs)
        assert torch.equal(unpacked_labels, labels)
        assert meta["timestamp"] is None
        assert meta["brand"] is None
        assert meta["source"] is None

    def test_unpack_batch_with_metadata(self):
        """测试带metadata的tuple格式解包"""
        import torch

        inputs = torch.rand(4, 10)
        labels = torch.randint(0, 2, (4,))
        meta = {
            "timestamp": "2023-01-01",
            "brand": "test_brand",
            "source": "test_source",
        }
        batch = (inputs, labels, meta)

        unpacked_inputs, unpacked_labels, unpacked_meta = _unpack_batch(batch, "tuple")

        assert torch.equal(unpacked_inputs, inputs)
        assert torch.equal(unpacked_labels, labels)
        assert unpacked_meta["timestamp"] == "2023-01-01"
        assert unpacked_meta["brand"] == "test_brand"
        assert unpacked_meta["source"] == "test_source"

    def test_unpack_batch_dict_format(self):
        """测试dict格式解包"""
        import torch

        batch = {
            "inputs": torch.rand(4, 10),
            "labels": torch.randint(0, 2, (4,)),
            "timestamp": "2023-01-01",
        }

        inputs, labels, meta = _unpack_batch(batch, "dict")

        assert torch.equal(inputs, batch["inputs"])
        assert torch.equal(labels, batch["labels"])
        assert meta["timestamp"] == "2023-01-01"
        assert meta["brand"] is None  # 未提供


class TestURLEncoderProtection:
    """测试URL编码器保护机制"""

    def test_encoder_assertion(self):
        """测试URL编码器断言保护"""
        from omegaconf import OmegaConf
        from src.systems.url_only_module import UrlOnlyModule

        # 正确的配置
        cfg = OmegaConf.create(
            {
                "model": {
                    "vocab_size": 128,
                    "embedding_dim": 128,
                    "hidden_dim": 128,
                    "num_layers": 2,
                    "bidirectional": True,
                    "dropout": 0.1,
                    "pad_id": 0,
                    "proj_dim": 256,
                    "num_classes": 2,
                },
                "train": {"lr": 1e-4},
                "metrics": {"dist": {"sync_metrics": False}, "average": "macro"},
            }
        )

        # 应该成功创建
        model = UrlOnlyModule(cfg)
        assert model.encoder.bidirectional

        # 错误的配置（修改层数）
        bad_cfg = OmegaConf.create(
            {
                "model": {
                    "vocab_size": 128,
                    "embedding_dim": 128,
                    "hidden_dim": 128,
                    "num_layers": 3,  # ❌ 错误！
                    "bidirectional": True,
                    "dropout": 0.1,
                    "pad_id": 0,
                    "proj_dim": 256,
                    "num_classes": 2,
                },
                "train": {"lr": 1e-4},
                "metrics": {"dist": {"sync_metrics": False}, "average": "macro"},
            }
        )

        # 应该触发断言错误
        with pytest.raises(AssertionError, match="URL encoder must remain"):
            model = UrlOnlyModule(bad_cfg)


class TestIntegration:
    """集成测试"""

    def test_split_stats_computation(self):
        """测试分割统计计算"""
        train_df = pd.DataFrame(
            {
                "url_text": ["url_1", "url_2"],
                "label": [0, 1],
                "brand": ["brand_a", "brand_b"],
                "timestamp": ["2023-01-01", "2023-01-02"],
                "source": ["source_1", "source_1"],
            }
        )

        val_df = train_df.copy()
        test_df = train_df.copy()

        stats = _compute_split_stats(train_df, val_df, test_df)

        assert "train" in stats
        assert "val" in stats
        assert "test" in stats
        assert stats["train"]["count"] == 2
        assert stats["train"]["pos_count"] == 1
        assert stats["train"]["neg_count"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

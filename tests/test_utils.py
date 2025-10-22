"""
测试工具函数
"""

import pytest
import torch
import numpy as np


def test_set_global_seed():
    """测试随机种子设置"""
    from src.utils.seed import set_global_seed

    # 设置种子
    set_global_seed(42)

    # 生成随机数
    torch_rand1 = torch.rand(5)
    np_rand1 = np.random.rand(5)

    # 重新设置相同种子
    set_global_seed(42)

    # 应该生成相同的随机数
    torch_rand2 = torch.rand(5)
    np_rand2 = np.random.rand(5)

    assert torch.allclose(torch_rand1, torch_rand2)
    assert np.allclose(np_rand1, np_rand2)


def test_seed_reproducibility():
    """测试种子可复现性"""
    from src.utils.seed import set_global_seed

    results = []
    for seed in [42, 42, 123, 42]:
        set_global_seed(seed)
        result = torch.rand(3).tolist()
        results.append(result)

    # 相同种子应该产生相同结果
    assert results[0] == results[1]
    assert results[0] == results[3]
    # 不同种子应该产生不同结果
    assert results[0] != results[2]


def test_experiment_tracker_creation(tmp_path):
    """测试实验跟踪器创建"""
    from src.utils.experiment_tracker import ExperimentTracker
    from omegaconf import OmegaConf

    # 创建配置
    cfg = OmegaConf.create(
        {
            "run": {"name": "test_exp"},
            "model": {"pretrained_name": "test"},
            "data": {"max_length": 128},
            "train": {"bs": 16, "lr": 1e-5, "epochs": 5},
        }
    )

    # 创建跟踪器
    tracker = ExperimentTracker(cfg, base_dir=str(tmp_path))

    # 检查目录结构
    assert tracker.exp_dir.exists()
    assert tracker.results_dir.exists()
    assert tracker.logs_dir.exists()
    assert tracker.checkpoints_dir.exists()

    # 检查配置文件
    config_file = tracker.exp_dir / "config.yaml"
    assert config_file.exists()


def test_experiment_tracker_save_metrics(tmp_path):
    """测试保存指标"""
    from src.utils.experiment_tracker import ExperimentTracker
    from omegaconf import OmegaConf
    import json

    cfg = OmegaConf.create(
        {
            "run": {"name": "test"},
            "model": {"pretrained_name": "test"},
            "data": {"max_length": 128},
            "train": {"bs": 16, "lr": 1e-5, "epochs": 5},
        }
    )

    tracker = ExperimentTracker(cfg, base_dir=str(tmp_path))

    # 保存指标
    metrics = {"accuracy": 0.95, "f1": 0.92, "loss": 0.15}

    metrics_file = tracker.save_metrics(metrics, stage="test")

    # 验证文件存在
    assert metrics_file.exists()

    # 验证内容
    with open(metrics_file) as f:
        saved_metrics = json.load(f)

    assert saved_metrics["metrics"] == metrics
    assert saved_metrics["stage"] == "test"


@pytest.mark.parametrize("level", ["DEBUG", "INFO", "WARNING", "ERROR"])
def test_logger_levels(level):
    """测试不同日志级别"""
    from src.utils.logging import get_logger

    logger = get_logger(__name__)

    # 测试各种日志级别
    getattr(logger, level.lower())(f"Test {level} message")


def test_logger_formatting():
    """测试日志格式"""
    from src.utils.logging import get_logger

    logger = get_logger("test_module")

    # 应该能正常记录
    logger.info("Test message")
    logger.debug("Debug message")
    logger.warning("Warning message")

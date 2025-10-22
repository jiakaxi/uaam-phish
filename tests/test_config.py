"""
测试配置管理
"""

import pytest
from omegaconf import OmegaConf
from pathlib import Path


def test_default_config_loads():
    """测试默认配置加载"""
    config_path = Path("configs/default.yaml")
    if not config_path.exists():
        pytest.skip("默认配置文件不存在")

    cfg = OmegaConf.load(config_path)

    # 检查必需字段
    assert "model" in cfg
    assert "train" in cfg
    assert "data" in cfg
    # default.yaml 不包含 paths（paths 在 Hydra config.yaml 中）


def test_hydra_config_loads():
    """测试 Hydra 配置加载"""
    config_path = Path("configs/config.yaml")
    if not config_path.exists():
        pytest.skip("Hydra 配置文件不存在")

    cfg = OmegaConf.load(config_path)

    # 检查 defaults
    assert "defaults" in cfg
    assert "run" in cfg
    assert "paths" in cfg


def test_profile_configs():
    """测试环境配置文件"""
    profiles = ["local", "server"]

    for profile in profiles:
        config_path = Path(f"configs/profiles/{profile}.yaml")
        if config_path.exists():
            cfg = OmegaConf.load(config_path)
            # 检查硬件配置
            if "hardware" in cfg:
                assert "accelerator" in cfg.hardware
        else:
            # 尝试新的配置结构
            config_path = Path(f"configs/trainer/{profile}.yaml")
            if config_path.exists():
                cfg = OmegaConf.load(config_path)


def test_config_merge():
    """测试配置合并"""
    base_cfg = OmegaConf.create({"model": {"dropout": 0.1}, "train": {"lr": 1e-5}})

    override_cfg = OmegaConf.create({"model": {"dropout": 0.2}, "train": {"bs": 32}})

    merged = OmegaConf.merge(base_cfg, override_cfg)

    # 检查合并结果
    assert merged.model.dropout == 0.2  # 被覆盖
    assert merged.train.lr == 1e-5  # 保留
    assert merged.train.bs == 32  # 新增


def test_config_env_variables(monkeypatch):
    """测试环境变量替换"""
    # 设置环境变量
    monkeypatch.setenv("DATA_ROOT", "/custom/data/path")

    cfg = OmegaConf.create(
        {"paths": {"data_root": "${oc.env:DATA_ROOT,data/processed}"}}
    )

    # 解析环境变量
    resolved = OmegaConf.to_container(cfg, resolve=True)

    assert resolved["paths"]["data_root"] == "/custom/data/path"


def test_config_validation():
    """测试配置验证"""
    # 有效配置
    valid_cfg = OmegaConf.create(
        {"model": {"pretrained_name": "roberta-base"}, "train": {"lr": 1e-5, "bs": 16}}
    )

    assert valid_cfg.model.pretrained_name == "roberta-base"
    assert valid_cfg.train.lr == 1e-5

    # 缺失字段应该抛出异常
    invalid_cfg = OmegaConf.create({"model": {}})

    with pytest.raises(Exception):
        # 访问不存在的字段
        _ = invalid_cfg.model.pretrained_name

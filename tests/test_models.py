"""
测试模型组件
"""

import pytest
import torch
import torch.nn as nn


def test_url_encoder_forward():
    """测试 URL 编码器前向传播"""
    from src.models.url_encoder import URLEncoder

    # 测试字符级 BiLSTM 编码器
    encoder = URLEncoder(
        vocab_size=128,
        embedding_dim=64,
        hidden_dim=64,
        num_layers=2,
        bidirectional=True,
        dropout=0.1,
        pad_id=0,
        proj_dim=128,
    )

    # 输入：[batch_size, max_len]
    input_ids = torch.randint(0, 128, (2, 256), dtype=torch.long)

    output = encoder(input_ids)

    assert output.shape[0] == 2  # batch size
    assert output.shape[1] == 128  # proj_dim


def test_url_encoder_dropout():
    """测试 dropout 配置"""
    from src.models.url_encoder import URLEncoder

    encoder = URLEncoder(
        vocab_size=128,
        embedding_dim=64,
        hidden_dim=64,
        num_layers=2,
        bidirectional=True,
        dropout=0.5,
        pad_id=0,
        proj_dim=128,
    )

    # 检查 dropout 层存在
    has_dropout = any(isinstance(m, nn.Dropout) for m in encoder.modules())
    assert has_dropout, "模型应包含 Dropout 层"


def test_url_encoder_device():
    """测试设备转换"""
    from src.models.url_encoder import URLEncoder

    encoder = URLEncoder(
        vocab_size=128,
        embedding_dim=64,
        hidden_dim=64,
        num_layers=2,
        bidirectional=True,
        dropout=0.1,
        pad_id=0,
        proj_dim=128,
    )

    # CPU测试
    input_ids = torch.randint(0, 128, (2, 256), dtype=torch.long)

    output = encoder(input_ids)
    assert output.device.type == "cpu"

    # 如果有GPU，测试GPU
    if torch.cuda.is_available():
        encoder = encoder.cuda()
        input_ids = input_ids.cuda()
        output = encoder(input_ids)
        assert output.device.type == "cuda"


@pytest.mark.parametrize("batch_size", [1, 4, 8])
def test_url_encoder_batch_sizes(batch_size):
    """测试不同批次大小"""
    from src.models.url_encoder import URLEncoder

    encoder = URLEncoder(
        vocab_size=128,
        embedding_dim=64,
        hidden_dim=64,
        num_layers=2,
        bidirectional=True,
        dropout=0.1,
        pad_id=0,
        proj_dim=128,
    )

    input_ids = torch.randint(0, 128, (batch_size, 256), dtype=torch.long)

    output = encoder(input_ids)
    assert output.shape[0] == batch_size
    assert output.shape[1] == 128  # proj_dim

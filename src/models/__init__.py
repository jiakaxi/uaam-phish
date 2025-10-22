"""
模型包 - URL 编码器实现
"""

from src.models.url_encoder import URLEncoder
from src.models.url_encoder_legacy import UrlBertEncoder

__all__ = [
    "URLEncoder",  # 推荐：字符级 BiLSTM
    "UrlBertEncoder",  # Legacy：HuggingFace BERT（仅用于向后兼容）
]

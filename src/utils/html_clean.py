"""
HTML 清洗工具模块
用于从 HTML 文件中提取纯文本内容，支持多模态钓鱼检测
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Union

try:
    from bs4 import BeautifulSoup

    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False


def clean_html(html_text: str, max_chars: int = 200000) -> str:
    """
    清洗 HTML 文本，提取纯文本内容。

    Args:
        html_text: 原始 HTML 字符串
        max_chars: 最大字符数限制（防止超大文件）

    Returns:
        清洗后的纯文本（空字符串如果无效）

    清洗规则:
    - 移除 <script>, <style> 标签及其内容
    - 优先提取 <body> 内容，否则全文
    - 移除多余空白符
    - 限制长度

    Example:
        >>> html = '<html><body><h1>Hello</h1><script>alert("xss")</script></body></html>'
        >>> clean_html(html)
        'Hello'
    """
    if not html_text or not html_text.strip():
        return ""

    # 限制输入长度
    html_text = html_text[:max_chars]

    try:
        if HAS_BS4:
            # 使用 BeautifulSoup 解析（推荐）
            soup = BeautifulSoup(html_text, "lxml")

            # 移除 script 和 style 标签
            for tag in soup(["script", "style", "noscript"]):
                tag.decompose()

            # 优先提取 body，否则全文
            body = soup.find("body")
            text = body.get_text() if body else soup.get_text()
        else:
            # Fallback: 使用正则表达式（不推荐，但可用）
            text = re.sub(
                r"<script[^>]*>.*?</script>",
                "",
                html_text,
                flags=re.DOTALL | re.IGNORECASE,
            )
            text = re.sub(
                r"<style[^>]*>.*?</style>",
                "",
                html_text,
                flags=re.DOTALL | re.IGNORECASE,
            )
            text = re.sub(r"<[^>]+>", " ", text)  # 移除所有标签
            text = re.sub(r"&[a-zA-Z]+;", " ", text)  # 移除 HTML 实体

        # 清理空白符
        text = re.sub(r"\s+", " ", text).strip()
        return text

    except Exception:
        # 任何异常都返回空字符串（兜底策略）
        return ""


def load_html_from_path(html_path: Union[str, Path]) -> str:
    """
    从文件路径加载 HTML，带异常处理。

    Args:
        html_path: HTML 文件路径

    Returns:
        HTML 文本（文件不存在或读取失败返回空字符串，不报错）

    异常处理:
    - 文件不存在 → 返回 ""
    - 编码错误 → 使用 errors='ignore'
    - 读取错误 → 返回 ""

    Example:
        >>> html = load_html_from_path("data/sample.html")
        >>> text = clean_html(html)
    """
    path = Path(html_path)

    # 路径为空或不存在
    if not html_path or not path.exists() or not path.is_file():
        return ""

    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            # 限制读取大小（防止超大文件）
            return f.read(200000)
    except Exception:
        # 任何读取错误都返回空字符串
        return ""


def extract_text_from_html_file(
    html_path: Union[str, Path], max_chars: int = 200000
) -> str:
    """
    一站式函数：加载 HTML 文件并清洗提取文本。

    Args:
        html_path: HTML 文件路径
        max_chars: 最大字符数限制

    Returns:
        清洗后的纯文本

    Example:
        >>> text = extract_text_from_html_file("data/phishing.html")
        >>> print(text[:100])
    """
    html_text = load_html_from_path(html_path)
    return clean_html(html_text, max_chars=max_chars)

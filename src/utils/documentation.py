"""
Documentation management utilities for appending to project documentation files.

This module provides tools to append new content to existing documentation files
instead of regenerating them completely.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


class DocumentationAppender:
    """
    Manager for appending content to existing project documentation files.

    Supported files:
    - FINAL_SUMMARY_CN.md: Project implementation summaries
    - CHANGES_SUMMARY.md: Change summaries and file modifications
    - FILES_MANIFEST.md: File creation and modification records

    Usage:
        >>> doc = DocumentationAppender()
        >>> doc.append_to_summary(
        ...     feature_name="New Feature",
        ...     summary="Brief description",
        ...     deliverables=["Item 1", "Item 2"],
        ... )
    """

    def __init__(self, root_dir: Optional[Path] = None):
        """
        Initialize documentation appender.

        Args:
            root_dir: Project root directory (defaults to current working directory)
        """
        self.root_dir = Path(root_dir) if root_dir else Path.cwd()
        self.summary_cn_path = self.root_dir / "FINAL_SUMMARY_CN.md"
        self.changes_path = self.root_dir / "CHANGES_SUMMARY.md"
        self.manifest_path = self.root_dir / "FILES_MANIFEST.md"

    def append_to_summary(
        self,
        feature_name: str,
        date: Optional[str] = None,
        status: str = "[完成]",
        summary: Optional[str] = None,
        deliverables: Optional[List[str]] = None,
        features: Optional[List[str]] = None,
        test_results: Optional[str] = None,
        usage: Optional[str] = None,
    ) -> None:
        """
        Append a new implementation summary to FINAL_SUMMARY_CN.md.

        Args:
            feature_name: Name of the feature/implementation
            date: Implementation date (defaults to today)
            status: Status badge (e.g., "[完成]", "[进行中]")
            summary: Brief summary of the implementation
            deliverables: List of deliverables
            features: List of implemented features
            test_results: Test results summary
            usage: Usage instructions
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        # Build the new section
        section = f"\n---\n\n# {feature_name}\n\n"
        section += f"**实施日期**: {date}  \n"
        section += f"**实施状态**: {status}\n\n"

        if summary:
            section += f"## 实施摘要\n\n{summary}\n\n"

        if deliverables:
            section += "## 交付成果\n\n"
            for item in deliverables:
                section += f"- {item}\n"
            section += "\n"

        if features:
            section += "## 功能实现\n\n"
            for feature in features:
                section += f"- {feature}\n"
            section += "\n"

        if test_results:
            section += f"## 测试结果\n\n{test_results}\n\n"

        if usage:
            section += f"## 使用方法\n\n{usage}\n\n"

        # Append to file
        with open(self.summary_cn_path, "a", encoding="utf-8") as f:
            f.write(section)

        print(f"[OK] 已追加到: {self.summary_cn_path}")

    def append_to_changes(
        self,
        feature_name: str,
        date: Optional[str] = None,
        implementation_type: str = "功能增强",
        added_files: Optional[List[str]] = None,
        modified_files: Optional[List[str]] = None,
        reused_configs: Optional[List[str]] = None,
        new_features: Optional[List[str]] = None,
        stats: Optional[Dict[str, any]] = None,
    ) -> None:
        """
        Append a new change summary to CHANGES_SUMMARY.md.

        Args:
            feature_name: Name of the feature
            date: Implementation date (defaults to today)
            implementation_type: Type of implementation (e.g., "功能增强", "Bug修复")
            added_files: List of added files with descriptions
            modified_files: List of modified files with descriptions
            reused_configs: List of reused configurations
            new_features: List of new features
            stats: Statistics dict (e.g., {"新增文件": 3, "修改文件": 2})
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        section = f"\n---\n\n# {feature_name}\n\n"
        section += f"**日期**: {date}  \n"
        section += f"**类型**: {implementation_type}\n\n"

        if added_files:
            section += "## 新增文件\n\n"
            for file_info in added_files:
                section += f"- {file_info}\n"
            section += "\n"

        if modified_files:
            section += "## 修改文件\n\n"
            for file_info in modified_files:
                section += f"- {file_info}\n"
            section += "\n"

        if reused_configs:
            section += "## 复用配置\n\n"
            for config_info in reused_configs:
                section += f"- {config_info}\n"
            section += "\n"

        if new_features:
            section += "## 新增功能\n\n"
            for feature in new_features:
                section += f"- {feature}\n"
            section += "\n"

        if stats:
            section += "## 统计数据\n\n"
            section += "| 类别 | 数量 |\n"
            section += "|------|------|\n"
            for key, value in stats.items():
                section += f"| {key} | {value} |\n"
            section += "\n"

        with open(self.changes_path, "a", encoding="utf-8") as f:
            f.write(section)

        print(f"[OK] 已追加到: {self.changes_path}")

    def append_to_manifest(
        self,
        feature_name: str,
        date: Optional[str] = None,
        added_files: Optional[List[Dict[str, str]]] = None,
        modified_files: Optional[List[Dict[str, str]]] = None,
        total_stats: Optional[Dict[str, any]] = None,
    ) -> None:
        """
        Append a new file manifest entry to FILES_MANIFEST.md.

        Args:
            feature_name: Name of the feature
            date: Implementation date (defaults to today)
            added_files: List of dicts with keys: 'path', 'lines', 'description'
            modified_files: List of dicts with keys: 'path', 'changes'
            total_stats: Total statistics dict
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        section = f"\n---\n\n# {feature_name}\n\n"
        section += f"**日期**: {date}\n\n"

        if added_files:
            section += "## 新增文件\n\n"
            for file_info in added_files:
                path = file_info.get("path", "")
                lines = file_info.get("lines", "?")
                desc = file_info.get("description", "")
                section += f"### `{path}` ({lines}行)\n"
                if desc:
                    section += f"{desc}\n"
                section += "\n"

        if modified_files:
            section += "## 修改文件\n\n"
            for file_info in modified_files:
                path = file_info.get("path", "")
                changes = file_info.get("changes", "")
                section += f"### `{path}`\n"
                section += f"**修改内容**:\n{changes}\n\n"

        if total_stats:
            section += "## 统计总览\n\n"
            section += "| 类别 | 数量 |\n"
            section += "|------|------|\n"
            for key, value in total_stats.items():
                section += f"| {key} | {value} |\n"
            section += "\n"

        with open(self.manifest_path, "a", encoding="utf-8") as f:
            f.write(section)

        print(f"[OK] 已追加到: {self.manifest_path}")

    def append_all(
        self,
        feature_name: str,
        date: Optional[str] = None,
        summary_kwargs: Optional[Dict] = None,
        changes_kwargs: Optional[Dict] = None,
        manifest_kwargs: Optional[Dict] = None,
    ) -> None:
        """
        Append to all three documentation files at once.

        Args:
            feature_name: Name of the feature
            date: Implementation date (defaults to today)
            summary_kwargs: Kwargs for append_to_summary
            changes_kwargs: Kwargs for append_to_changes
            manifest_kwargs: Kwargs for append_to_manifest
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        print(f"\n{'='*60}")
        print(f"追加文档: {feature_name}")
        print(f"{'='*60}\n")

        if summary_kwargs:
            self.append_to_summary(
                feature_name=feature_name, date=date, **summary_kwargs
            )

        if changes_kwargs:
            self.append_to_changes(
                feature_name=feature_name, date=date, **changes_kwargs
            )

        if manifest_kwargs:
            self.append_to_manifest(
                feature_name=feature_name, date=date, **manifest_kwargs
            )

        print(f"\n{'='*60}")
        print("[OK] 文档追加完成")
        print(f"{'='*60}\n")


def create_new_document(
    doc_type: str,
    filename: str,
    title: str,
    description: str,
    root_dir: Optional[Path] = None,
) -> Path:
    """
    Create a completely new documentation file (only when necessary).

    This function should ONLY be used when creating a new type of documentation
    that doesn't fit into existing files.

    Args:
        doc_type: Type of document (e.g., "guide", "reference", "spec")
        filename: Name of the file (e.g., "NEW_FEATURE_GUIDE.md")
        title: Title of the document
        description: Description of the document purpose
        root_dir: Project root directory (defaults to current working directory)

    Returns:
        Path to the created file
    """
    root = Path(root_dir) if root_dir else Path.cwd()

    # Determine directory based on doc type
    if doc_type == "guide":
        doc_dir = root / "docs"
    elif doc_type == "reference":
        doc_dir = root / "docs"
    elif doc_type == "spec":
        doc_dir = root / "docs" / "specs"
    else:
        doc_dir = root

    doc_dir.mkdir(parents=True, exist_ok=True)
    filepath = doc_dir / filename

    if filepath.exists():
        print(f"[WARNING] 文件已存在: {filepath}")
        return filepath

    # Create initial content
    content = f"# {title}\n\n"
    content += f"{description}\n\n"
    content += f"**创建日期**: {datetime.now().strftime('%Y-%m-%d')}\n\n"
    content += "---\n\n"

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"[OK] 已创建新文档: {filepath}")
    return filepath

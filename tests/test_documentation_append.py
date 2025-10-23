"""
Tests for documentation appending functionality.
"""

import tempfile
from pathlib import Path

import pytest

from src.utils.documentation import DocumentationAppender


@pytest.fixture
def temp_doc_dir():
    """Create a temporary directory with empty doc files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Create initial document files
        (tmp_path / "FINAL_SUMMARY_CN.md").write_text(
            "# 项目总结\n\n初始内容\n", encoding="utf-8"
        )
        (tmp_path / "CHANGES_SUMMARY.md").write_text(
            "# 变更摘要\n\n初始内容\n", encoding="utf-8"
        )
        (tmp_path / "FILES_MANIFEST.md").write_text(
            "# 文件清单\n\n初始内容\n", encoding="utf-8"
        )

        yield tmp_path


def test_append_to_summary(temp_doc_dir):
    """Test appending to FINAL_SUMMARY_CN.md."""
    doc = DocumentationAppender(root_dir=temp_doc_dir)

    doc.append_to_summary(
        feature_name="测试功能",
        summary="这是一个测试",
        deliverables=["交付物1", "交付物2"],
        features=["✅ 功能A", "✅ 功能B"],
    )

    # Read the file
    content = (temp_doc_dir / "FINAL_SUMMARY_CN.md").read_text(encoding="utf-8")

    # Verify content was appended
    assert "初始内容" in content
    assert "测试功能" in content
    assert "这是一个测试" in content
    assert "交付物1" in content
    assert "✅ 功能A" in content


def test_append_to_changes(temp_doc_dir):
    """Test appending to CHANGES_SUMMARY.md."""
    doc = DocumentationAppender(root_dir=temp_doc_dir)

    doc.append_to_changes(
        feature_name="测试功能",
        added_files=["file1.py", "file2.py"],
        modified_files=["file3.py"],
        stats={"新增文件": 2, "修改文件": 1},
    )

    content = (temp_doc_dir / "CHANGES_SUMMARY.md").read_text(encoding="utf-8")

    assert "初始内容" in content
    assert "测试功能" in content
    assert "file1.py" in content
    assert "新增文件" in content


def test_append_to_manifest(temp_doc_dir):
    """Test appending to FILES_MANIFEST.md."""
    doc = DocumentationAppender(root_dir=temp_doc_dir)

    doc.append_to_manifest(
        feature_name="测试功能",
        added_files=[
            {"path": "src/test.py", "lines": 100, "description": "测试文件"},
        ],
        total_stats={"新增文件": 1},
    )

    content = (temp_doc_dir / "FILES_MANIFEST.md").read_text(encoding="utf-8")

    assert "初始内容" in content
    assert "测试功能" in content
    assert "src/test.py" in content
    assert "100行" in content


def test_append_all(temp_doc_dir):
    """Test appending to all documents at once."""
    doc = DocumentationAppender(root_dir=temp_doc_dir)

    doc.append_all(
        feature_name="测试功能",
        summary_kwargs={
            "summary": "测试摘要",
            "deliverables": ["交付物"],
        },
        changes_kwargs={
            "added_files": ["file.py"],
            "stats": {"新增文件": 1},
        },
        manifest_kwargs={
            "added_files": [
                {"path": "file.py", "lines": 50, "description": "测试"},
            ],
        },
    )

    # Check all three files
    summary = (temp_doc_dir / "FINAL_SUMMARY_CN.md").read_text(encoding="utf-8")
    changes = (temp_doc_dir / "CHANGES_SUMMARY.md").read_text(encoding="utf-8")
    manifest = (temp_doc_dir / "FILES_MANIFEST.md").read_text(encoding="utf-8")

    assert "测试功能" in summary
    assert "测试功能" in changes
    assert "测试功能" in manifest


def test_multiple_appends(temp_doc_dir):
    """Test multiple appends preserve history."""
    doc = DocumentationAppender(root_dir=temp_doc_dir)

    # First append
    doc.append_to_summary(
        feature_name="功能1",
        summary="第一个功能",
    )

    # Second append
    doc.append_to_summary(
        feature_name="功能2",
        summary="第二个功能",
    )

    content = (temp_doc_dir / "FINAL_SUMMARY_CN.md").read_text(encoding="utf-8")

    # Both should be present
    assert "初始内容" in content
    assert "功能1" in content
    assert "第一个功能" in content
    assert "功能2" in content
    assert "第二个功能" in content


def test_preserve_existing_content(temp_doc_dir):
    """Test that existing content is never overwritten."""
    # Add some existing content
    existing = (temp_doc_dir / "FINAL_SUMMARY_CN.md").read_text(encoding="utf-8")
    existing += "\n# 已有功能\n\n重要内容不应该被覆盖\n"
    (temp_doc_dir / "FINAL_SUMMARY_CN.md").write_text(existing, encoding="utf-8")

    # Append new content
    doc = DocumentationAppender(root_dir=temp_doc_dir)
    doc.append_to_summary(
        feature_name="新功能",
        summary="新内容",
    )

    content = (temp_doc_dir / "FINAL_SUMMARY_CN.md").read_text(encoding="utf-8")

    # Existing content should still be there
    assert "初始内容" in content
    assert "已有功能" in content
    assert "重要内容不应该被覆盖" in content
    # New content should be added
    assert "新功能" in content
    assert "新内容" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

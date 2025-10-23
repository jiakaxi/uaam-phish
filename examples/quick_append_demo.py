"""
快速演示：追加内容到现有文档

这个脚本展示了最简单的使用方式。
运行后会在现有文档末尾追加一个示例条目。
"""

from pathlib import Path
import sys

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.documentation import DocumentationAppender  # noqa: E402


def quick_demo():
    """快速演示追加功能"""
    print("\n" + "=" * 60)
    print("文档追加演示")
    print("=" * 60 + "\n")

    # 创建文档追加器
    doc = DocumentationAppender(root_dir=project_root)

    # 示例：追加一个新功能记录
    print("正在追加示例内容到文档...\n")

    doc.append_all(
        feature_name="文档追加功能演示",
        date="2025-10-24",
        summary_kwargs={
            "status": "✅ 完成并测试",
            "summary": """
实现了文档追加管理功能，允许在现有文档末尾追加新内容，而不是每次都重新生成整个文档。

**核心优势**:
- 保留完整的历史记录
- 支持增量更新
- 避免文档重复
""",
            "deliverables": [
                "`src/utils/documentation.py` - 文档追加工具类",
                "`src/utils/doc_callback.py` - Lightning 回调集成",
                "`docs/APPEND_DOCUMENTATION_GUIDE.md` - 使用指南",
            ],
            "features": [
                "✅ 追加到 FINAL_SUMMARY_CN.md",
                "✅ 追加到 CHANGES_SUMMARY.md",
                "✅ 追加到 FILES_MANIFEST.md",
                "✅ Lightning 回调自动追加",
            ],
            "test_results": "✅ 6/6 测试通过",
            "usage": """
```python
from src.utils.documentation import DocumentationAppender

doc = DocumentationAppender()
doc.append_to_summary(
    feature_name="新功能",
    summary="功能描述",
    deliverables=["交付物"],
)
```
""",
        },
        changes_kwargs={
            "implementation_type": "功能增强",
            "added_files": [
                "**`src/utils/documentation.py`** (200行) - 文档追加工具",
                "**`src/utils/doc_callback.py`** (100行) - Lightning 回调",
                "**`docs/APPEND_DOCUMENTATION_GUIDE.md`** - 使用指南",
                "**`examples/append_documentation_example.py`** - 详细示例",
                "**`examples/quick_append_demo.py`** - 快速演示",
                "**`tests/test_documentation_append.py`** - 测试文件",
            ],
            "new_features": [
                "追加式文档管理 - 保留历史不覆盖",
                "三个文档统一管理接口",
                "Lightning 训练自动集成",
                "完整的测试覆盖",
            ],
            "stats": {
                "新增文件": 6,
                "新增代码行数": "~500行",
                "测试用例": 6,
                "测试通过率": "100%",
            },
        },
        manifest_kwargs={
            "added_files": [
                {
                    "path": "src/utils/documentation.py",
                    "lines": 200,
                    "description": "**功能**: 文档追加工具类\n- `DocumentationAppender` - 统一追加接口\n- `append_to_summary()` - 追加到总结\n- `append_to_changes()` - 追加到变更\n- `append_to_manifest()` - 追加到清单",
                },
                {
                    "path": "src/utils/doc_callback.py",
                    "lines": 100,
                    "description": "**功能**: Lightning 回调集成\n- `DocumentationCallback` - 训练结束自动追加\n- 支持自定义内容\n- 可选择追加目标",
                },
                {
                    "path": "docs/APPEND_DOCUMENTATION_GUIDE.md",
                    "lines": 300,
                    "description": "**功能**: 完整的使用指南\n- API 参考\n- 使用场景\n- 最佳实践",
                },
            ],
            "total_stats": {
                "新增文件": 6,
                "总计影响文件": 6,
                "新增代码行数": "~500行",
            },
        },
    )

    print("\n" + "=" * 60)
    print("✅ 追加完成！")
    print("=" * 60 + "\n")

    print("查看追加的内容：")
    print(f"1. {project_root / 'FINAL_SUMMARY_CN.md'}")
    print(f"2. {project_root / 'CHANGES_SUMMARY.md'}")
    print(f"3. {project_root / 'FILES_MANIFEST.md'}")
    print("\n提示：打开这些文件，滚动到底部即可看到新追加的内容。\n")


if __name__ == "__main__":
    print("\n⚠️  警告：这个脚本会向实际的文档文件追加示例内容！")
    print("如果你只是想测试，请先备份文档或查看 tests/test_documentation_append.py\n")

    response = input("确定要运行吗？(y/N): ")

    if response.lower() == "y":
        quick_demo()
    else:
        print("\n已取消。")
        print("\n如果想查看示例但不追加到实际文档，请运行：")
        print("  python -m pytest tests/test_documentation_append.py -v\n")

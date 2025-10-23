"""
示例：如何使用文档管理工具记录新功能

这个脚本展示了如何在实现新功能后自动更新项目文档。
"""

from pathlib import Path
import sys

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.documentation import (  # noqa: E402
    ChangelogManager,
    ImplementationManager,
    generate_implementation_template,
)


def example_1_simple_changelog():
    """示例1：简单的 Changelog 更新"""
    print("\n=== 示例1：简单的 Changelog 更新 ===\n")

    changelog = ChangelogManager(root_dir=project_root)

    changelog.append_change(
        feature_name="添加不确定性估计模块",
        added=[
            "Monte Carlo Dropout 支持 (`src/modules/mc_dropout.py`)",
            "温度缩放校准 (`src/utils/temperature_scaling.py`)",
        ],
        modified=[
            "`URLOnlyModule` 支持不确定性估计输出",
        ],
        stats={
            "新增文件": 2,
            "修改文件": 1,
            "新增代码": "~250行",
        },
    )


def example_2_with_detailed_doc():
    """示例2：带详细文档的完整记录"""
    print("\n=== 示例2：带详细文档的完整记录 ===\n")

    # 1. 创建详细实现文档
    impl_mgr = ImplementationManager(root_dir=project_root)

    # 使用模板生成内容
    doc_content = generate_implementation_template(
        feature_name="不确定性估计模块",
        summary="""
实现了基于 Monte Carlo Dropout 的不确定性估计，用于评估模型预测的可信度。
包含温度缩放校准方法以改善概率校准。
""",
        added_files=[
            "src/modules/mc_dropout.py",
            "src/utils/temperature_scaling.py",
            "configs/uncertainty.yaml",
        ],
        modified_files=[
            "src/systems/url_only_module.py",
            "src/models/url_encoder.py",
        ],
        stats={
            "新增文件": 3,
            "修改文件": 2,
            "新增代码行数": "~250行",
            "测试用例": 8,
        },
    )

    # 创建实现文档
    doc_path = impl_mgr.create_implementation_doc(
        feature_name="不确定性估计模块",
        content=doc_content,
    )

    # 2. 更新 Changelog，引用详细文档
    changelog = ChangelogManager(root_dir=project_root)

    changelog.append_change(
        feature_name="不确定性估计模块",
        added=[
            "Monte Carlo Dropout 支持 (`src/modules/mc_dropout.py`)",
            "温度缩放校准 (`src/utils/temperature_scaling.py`)",
            "不确定性配置 (`configs/uncertainty.yaml`)",
        ],
        modified=[
            "`URLOnlyModule` - 添加不确定性估计方法",
            "`URLEncoder` - 支持 Dropout 在推理时启用",
        ],
        config_changes=[
            "新增 `uncertainty.enable: bool` 配置项",
            "新增 `uncertainty.mc_samples: int` 配置项（默认20）",
        ],
        stats={
            "新增文件": 3,
            "修改文件": 2,
            "新增代码": "~250行",
            "测试用例": 8,
        },
        doc_link=doc_path,
    )


def example_3_mlops_protocol_migration():
    """示例3：迁移现有的 MLOps 协议文档"""
    print("\n=== 示例3：迁移现有的 MLOps 协议文档 ===\n")

    # 1. 创建历史记录的实现文档
    impl_mgr = ImplementationManager(root_dir=project_root)

    # 读取现有的 FINAL_SUMMARY_CN.md 内容作为基础
    summary_path = project_root / "FINAL_SUMMARY_CN.md"
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            existing_content = f.read()

        # 创建历史文档
        impl_mgr.create_implementation_doc(
            feature_name="MLOps 协议实现",
            content=existing_content,
            date="2025-10-23",
            status="✅ 完成",
        )

    # 2. 在 Changelog 中追加历史记录
    changelog = ChangelogManager(root_dir=project_root)

    changelog.append_change(
        feature_name="MLOps 协议实现",
        added=[
            "三种数据分割协议: random/temporal/brand_ood (`src/utils/splits.py`)",
            "完整指标计算系统: ECE, NLL, AUROC, F1 (`src/utils/metrics.py`)",
            "工件自动生成回调: ProtocolArtifactsCallback (`src/utils/protocol_artifacts.py`)",
            "Batch 格式适配器 (`src/utils/batch_utils.py`)",
        ],
        modified=[
            "`src/systems/url_only_module.py` - 添加指标计算和 URL 编码器保护",
            "`src/utils/visualizer.py` - 添加 ROC 和校准曲线保存方法",
            "`scripts/train_hydra.py` - 集成协议工件回调",
        ],
        config_changes=[
            "复用 `configs/default.yaml` 中的 metrics 配置",
            "复用 `configs/data/url_only.yaml` 中的 batch_format 配置",
        ],
        stats={
            "新增文件": 9,
            "修改文件": 3,
            "复用配置": 2,
            "新增代码": "~1500行",
            "文档行数": "~1200行",
            "测试用例": 13,
            "测试通过率": "100%",
        },
        doc_link="docs/implementations/2025-10-23_mlops_协议实现.md",
        date="2025-10-23",
    )


def example_4_read_changelog():
    """示例4：读取最新的 Changelog 条目"""
    print("\n=== 示例4：读取最新的 Changelog 条目 ===\n")

    changelog = ChangelogManager(root_dir=project_root)

    # 读取最新的2个条目
    latest = changelog.read_latest(n=2)
    print("最新的2个变更记录：")
    print(latest)


def example_5_list_implementations():
    """示例5：列出所有实现文档"""
    print("\n=== 示例5：列出所有实现文档 ===\n")

    impl_mgr = ImplementationManager(root_dir=project_root)

    implementations = impl_mgr.list_implementations()

    print(f"共有 {len(implementations)} 个实现文档：\n")
    for impl in implementations:
        print(f"- [{impl['date']}] {impl['feature']} - {impl['status']}")


if __name__ == "__main__":
    print("=" * 60)
    print("文档管理工具使用示例")
    print("=" * 60)

    # 取消注释以运行不同的示例

    # example_1_simple_changelog()
    # example_2_with_detailed_doc()
    # example_3_mlops_protocol_migration()
    # example_4_read_changelog()
    # example_5_list_implementations()

    print("\n提示：取消注释上面的示例函数来运行不同的示例")
    print("\n建议的使用流程：")
    print("1. 实现新功能后，运行 example_2_with_detailed_doc() 创建文档")
    print("2. 使用 example_4_read_changelog() 查看最新变更")
    print("3. 使用 example_5_list_implementations() 查看所有实现")

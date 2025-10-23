"""
示例：如何追加内容到现有文档文件

这个脚本展示了如何在实现新功能后，将内容追加到：
- FINAL_SUMMARY_CN.md
- CHANGES_SUMMARY.md
- FILES_MANIFEST.md
"""

from pathlib import Path
import sys

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.documentation import DocumentationAppender  # noqa: E402


def example_1_append_to_summary():
    """示例1：追加到 FINAL_SUMMARY_CN.md"""
    print("\n=== 示例1：追加到 FINAL_SUMMARY_CN.md ===\n")

    doc = DocumentationAppender(root_dir=project_root)

    doc.append_to_summary(
        feature_name="不确定性估计模块",
        status="✅ 完成",
        summary="""
实现了基于 Monte Carlo Dropout 的不确定性估计功能，用于评估模型预测的可信度。
包含温度缩放校准方法以改善概率校准效果。
""",
        deliverables=[
            "`src/modules/mc_dropout.py` (150行) - Monte Carlo Dropout 实现",
            "`src/utils/temperature_scaling.py` (100行) - 温度缩放校准",
            "`configs/uncertainty.yaml` - 不确定性配置文件",
        ],
        features=[
            "✅ Monte Carlo Dropout - 采样20次获取预测分布",
            "✅ 温度缩放 - 自动校准预测概率",
            "✅ 不确定性指标 - 预测熵和方差",
        ],
        test_results="✅ 8/8 测试通过",
        usage="""
```bash
# 启用不确定性估计
python scripts/train_hydra.py uncertainty.enable=true

# 查看不确定性分析
python scripts/predict.py --uncertainty --samples 20
```
""",
    )


def example_2_append_to_changes():
    """示例2：追加到 CHANGES_SUMMARY.md"""
    print("\n=== 示例2：追加到 CHANGES_SUMMARY.md ===\n")

    doc = DocumentationAppender(root_dir=project_root)

    doc.append_to_changes(
        feature_name="不确定性估计模块",
        implementation_type="功能增强",
        added_files=[
            "**`src/modules/mc_dropout.py`** (150行) - Monte Carlo Dropout 实现",
            "**`src/utils/temperature_scaling.py`** (100行) - 温度缩放校准",
            "**`configs/uncertainty.yaml`** - 不确定性配置",
        ],
        modified_files=[
            "**`src/systems/url_only_module.py`** - 添加 `predict_with_uncertainty()` 方法",
            "**`src/models/url_encoder.py`** - 支持推理时启用 Dropout",
        ],
        reused_configs=[
            "`configs/default.yaml` - 复用现有 model 配置",
        ],
        new_features=[
            "Monte Carlo Dropout 采样",
            "温度缩放校准",
            "不确定性指标计算（熵、方差）",
        ],
        stats={
            "新增文件": 3,
            "修改文件": 2,
            "新增代码行数": "~250行",
            "测试用例": 8,
        },
    )


def example_3_append_to_manifest():
    """示例3：追加到 FILES_MANIFEST.md"""
    print("\n=== 示例3：追加到 FILES_MANIFEST.md ===\n")

    doc = DocumentationAppender(root_dir=project_root)

    doc.append_to_manifest(
        feature_name="不确定性估计模块",
        added_files=[
            {
                "path": "src/modules/mc_dropout.py",
                "lines": 150,
                "description": "**功能**: Monte Carlo Dropout 实现\n- `MCDropoutWrapper` - Dropout 包装器\n- `sample_predictions()` - 采样预测函数",
            },
            {
                "path": "src/utils/temperature_scaling.py",
                "lines": 100,
                "description": "**功能**: 温度缩放校准\n- `TemperatureScaling` - 温度缩放类\n- `calibrate()` - 校准函数",
            },
            {
                "path": "configs/uncertainty.yaml",
                "lines": 20,
                "description": "**功能**: 不确定性配置\n- `uncertainty.enable: bool`\n- `uncertainty.mc_samples: int`",
            },
        ],
        modified_files=[
            {
                "path": "src/systems/url_only_module.py",
                "changes": "- [ADDED] `predict_with_uncertainty()` 方法\n- [ADDED] 不确定性指标计算",
            },
            {
                "path": "src/models/url_encoder.py",
                "changes": "- [ADDED] `enable_mc_dropout()` 方法\n- [MODIFIED] forward() 支持 MC Dropout",
            },
        ],
        total_stats={
            "新增文件": 3,
            "修改文件": 2,
            "总计影响文件": 5,
            "新增代码行数": "~250行",
        },
    )


def example_4_append_all_at_once():
    """示例4：一次性追加到所有文档"""
    print("\n=== 示例4：一次性追加到所有文档 ===\n")

    doc = DocumentationAppender(root_dir=project_root)

    doc.append_all(
        feature_name="数据增强模块",
        summary_kwargs={
            "status": "✅ 完成",
            "summary": "实现了针对 URL 的数据增强方法",
            "deliverables": [
                "`src/data/augmentation.py` - 数据增强实现",
                "`configs/augmentation.yaml` - 增强配置",
            ],
            "features": [
                "✅ URL 变换增强",
                "✅ 混合增强策略",
            ],
        },
        changes_kwargs={
            "implementation_type": "功能增强",
            "added_files": [
                "**`src/data/augmentation.py`** (200行) - 数据增强",
            ],
            "stats": {
                "新增文件": 1,
                "新增代码": "~200行",
            },
        },
        manifest_kwargs={
            "added_files": [
                {
                    "path": "src/data/augmentation.py",
                    "lines": 200,
                    "description": "数据增强实现",
                },
            ],
            "total_stats": {
                "新增文件": 1,
            },
        },
    )


def example_5_real_world_usage():
    """示例5：实际使用场景（训练结束后自动追加）"""
    print("\n=== 示例5：实际使用场景 ===\n")

    # 模拟训练结束后的场景
    feature_name = "优化训练流程"
    test_acc = 0.8523
    test_auroc = 0.9234

    doc = DocumentationAppender(root_dir=project_root)

    doc.append_to_summary(
        feature_name=feature_name,
        status="✅ 完成并验证",
        summary=f"""
优化了训练流程，提升了训练效率和模型性能。

**测试结果**:
- 准确率: {test_acc:.4f}
- AUROC: {test_auroc:.4f}
""",
        deliverables=[
            "优化的训练脚本",
            "改进的数据加载器",
            "新的学习率调度策略",
        ],
        features=[
            f"✅ 测试准确率达到 {test_acc:.2%}",
            f"✅ AUROC 达到 {test_auroc:.2%}",
            "✅ 训练速度提升 30%",
        ],
    )

    print("\n已自动记录实验结果到文档")


if __name__ == "__main__":
    print("=" * 60)
    print("文档追加工具使用示例")
    print("=" * 60)

    # 取消注释以运行不同的示例

    # example_1_append_to_summary()
    # example_2_append_to_changes()
    # example_3_append_to_manifest()
    # example_4_append_all_at_once()
    # example_5_real_world_usage()

    print("\n💡 提示：取消注释上面的示例函数来运行")
    print("\n推荐的使用流程：")
    print("1. 实现新功能后，运行 example_4_append_all_at_once()")
    print("2. 或者单独追加到各个文档：")
    print("   - example_1_append_to_summary() - 追加到总结文档")
    print("   - example_2_append_to_changes() - 追加到变更文档")
    print("   - example_3_append_to_manifest() - 追加到文件清单")
    print("3. 在训练脚本中集成 example_5_real_world_usage()")

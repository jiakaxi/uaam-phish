# 文档结构建议

## 当前问题
- 根目录有多个总结性文档（FINAL_SUMMARY_CN.md, CHANGES_SUMMARY.md等）
- 这些文档是一次性生成的，每次新功能都需要重新生成
- 缺乏增量式、追加式的记录机制

## 推荐方案：分层 + 追加式文档管理

### 1. 根目录文档（精简）
```
项目根目录/
├── README.md                    # 项目主文档
├── CHANGELOG.md                 # 变更日志（追加式）⭐
└── QUICK_START.md               # 快速开始
```

### 2. docs/ 目录（详细文档）
```
docs/
├── ARCHITECTURE.md              # 架构文档（更新式）
├── DEVELOPMENT_GUIDE.md         # 开发指南
├── implementations/             # 功能实现记录
│   ├── 2025-10-23_mlops_protocols.md
│   ├── 2025-10-24_uncertainty_module.md
│   └── README.md                # 实现索引
├── experiments/                 # 实验文档
│   └── README.md
└── adr/                         # 架构决策记录（已有）
    └── *.md
```

### 3. 实验目录（自动生成）
```
experiments/<run_name>/
├── config.yaml                  # 配置快照
├── SUMMARY.md                   # 实验总结（自动生成）
├── logs/                        # 日志
├── checkpoints/                 # 检查点
└── results/                     # 结果
    ├── metrics_*.json
    ├── roc_*.png
    └── implementation_report.md # 协议实现报告（自动生成）
```

## 追加式文档示例

### CHANGELOG.md 格式
```markdown
# 变更日志

本文档记录项目的所有重要变更。

---

## [2025-10-24] 不确定性模块增强

### 新增
- 添加 Monte Carlo Dropout 支持 (`src/modules/uncertainty.py`)
- 新增温度缩放校准 (`src/utils/calibration.py`)

### 修改
- 更新 `URLOnlyModule` 支持不确定性估计

### 配置
- 新增 `configs/uncertainty.yaml`

### 影响
- 修改文件: 2
- 新增文件: 1
- 新增代码: ~200行

### 文档
- 详细实现见: `docs/implementations/2025-10-24_uncertainty_module.md`

---

## [2025-10-23] MLOps 协议实现

### 新增
- 三种数据分割协议: random/temporal/brand_ood
- 完整指标计算系统: ECE, NLL, AUROC, F1
- 工件自动生成回调: ProtocolArtifactsCallback

### 修改
- `src/systems/url_only_module.py` - 添加指标计算
- `src/utils/visualizer.py` - 添加可视化方法
- `scripts/train_hydra.py` - 集成协议回调

### 影响
- 新增文件: 9
- 修改文件: 3
- 新增代码: ~1500行

### 文档
- 详细实现见: `docs/implementations/2025-10-23_mlops_protocols.md`
```

### docs/implementations/README.md 格式
```markdown
# 功能实现索引

本目录包含各个功能模块的详细实现文档。

## 实现列表

| 日期 | 功能 | 文档 | 状态 |
|------|------|------|------|
| 2025-10-24 | 不确定性模块增强 | [2025-10-24_uncertainty_module.md](./2025-10-24_uncertainty_module.md) | ✅ 完成 |
| 2025-10-23 | MLOps协议实现 | [2025-10-23_mlops_protocols.md](./2025-10-23_mlops_protocols.md) | ✅ 完成 |

## 文档规范

每个实现文档应包含：
1. 实施摘要
2. 功能清单
3. 文件变更列表
4. 测试验证
5. 使用示例
```

## 自动化追加工具

### 实现追加式记录的工具类

**文件位置**: `src/utils/documentation.py`

```python
from datetime import datetime
from pathlib import Path
from typing import Dict, List

class ChangelogManager:
    """管理项目变更日志的追加和更新"""

    def __init__(self, changelog_path: str = "CHANGELOG.md"):
        self.changelog_path = Path(changelog_path)
        self._ensure_changelog_exists()

    def _ensure_changelog_exists(self):
        """确保 CHANGELOG 文件存在"""
        if not self.changelog_path.exists():
            with open(self.changelog_path, "w", encoding="utf-8") as f:
                f.write("# 变更日志\n\n")
                f.write("本文档记录项目的所有重要变更。\n\n---\n\n")

    def append_change(
        self,
        feature_name: str,
        added: List[str] = None,
        modified: List[str] = None,
        config_changes: List[str] = None,
        stats: Dict = None,
        doc_link: str = None,
    ):
        """追加新的变更记录"""
        date = datetime.now().strftime("%Y-%m-%d")

        entry = f"## [{date}] {feature_name}\n\n"

        if added:
            entry += "### 新增\n"
            for item in added:
                entry += f"- {item}\n"
            entry += "\n"

        if modified:
            entry += "### 修改\n"
            for item in modified:
                entry += f"- {item}\n"
            entry += "\n"

        if config_changes:
            entry += "### 配置\n"
            for item in config_changes:
                entry += f"- {item}\n"
            entry += "\n"

        if stats:
            entry += "### 影响\n"
            for key, value in stats.items():
                entry += f"- {key}: {value}\n"
            entry += "\n"

        if doc_link:
            entry += "### 文档\n"
            entry += f"- 详细实现见: `{doc_link}`\n\n"

        entry += "---\n\n"

        # 追加到文件
        with open(self.changelog_path, "a", encoding="utf-8") as f:
            f.write(entry)

        print(f"✅ 变更已追加到 {self.changelog_path}")

class ImplementationManager:
    """管理实现文档的创建"""

    def __init__(self, implementations_dir: str = "docs/implementations"):
        self.implementations_dir = Path(implementations_dir)
        self.implementations_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.implementations_dir / "README.md"
        self._ensure_index_exists()

    def _ensure_index_exists(self):
        """确保索引文件存在"""
        if not self.index_path.exists():
            with open(self.index_path, "w", encoding="utf-8") as f:
                f.write("# 功能实现索引\n\n")
                f.write("| 日期 | 功能 | 文档 | 状态 |\n")
                f.write("|------|------|------|------|\n")

    def create_implementation_doc(
        self,
        feature_name: str,
        content: str,
    ) -> str:
        """创建新的实现文档"""
        date = datetime.now().strftime("%Y-%m-%d")
        filename = f"{date}_{feature_name.lower().replace(' ', '_')}.md"
        filepath = self.implementations_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

        # 更新索引
        self._append_to_index(date, feature_name, filename)

        print(f"✅ 实现文档已创建: {filepath}")
        return str(filepath.relative_to(Path.cwd()))

    def _append_to_index(self, date: str, feature_name: str, filename: str):
        """追加到索引"""
        with open(self.index_path, "a", encoding="utf-8") as f:
            f.write(f"| {date} | {feature_name} | [{filename}](./{filename}) | ✅ 完成 |\n")
```

### 使用示例

```python
# 在训练脚本或回调中使用
from src.utils.documentation import ChangelogManager, ImplementationManager

# 1. 更新 CHANGELOG
changelog = ChangelogManager()
changelog.append_change(
    feature_name="不确定性模块增强",
    added=[
        "Monte Carlo Dropout 支持 (`src/modules/uncertainty.py`)",
        "温度缩放校准 (`src/utils/calibration.py`)",
    ],
    modified=[
        "`URLOnlyModule` 支持不确定性估计",
    ],
    config_changes=[
        "新增 `configs/uncertainty.yaml`",
    ],
    stats={
        "修改文件": 2,
        "新增文件": 1,
        "新增代码": "~200行",
    },
    doc_link="docs/implementations/2025-10-24_uncertainty_module.md",
)

# 2. 创建详细实现文档
impl_mgr = ImplementationManager()
doc_content = """
# 不确定性模块增强

## 实施日期
2025-10-24

## 功能概述
添加了 Monte Carlo Dropout 和温度缩放校准支持。

## 详细变更
[详细内容...]
"""
impl_mgr.create_implementation_doc("不确定性模块增强", doc_content)
```

## 迁移计划

### 步骤 1: 归档现有文档
```bash
# 创建 history 目录
mkdir -p docs/history

# 移动现有总结文档
mv FINAL_SUMMARY_CN.md docs/history/2025-10-23_mlops_implementation_summary.md
mv CHANGES_SUMMARY.md docs/history/2025-10-23_mlops_changes.md
mv FILES_MANIFEST.md docs/history/2025-10-23_files_manifest.md
mv IMPLEMENTATION_REPORT.md docs/implementations/2025-10-23_mlops_protocols.md
```

### 步骤 2: 创建新的追加式文档
```bash
# 创建 CHANGELOG.md（基于现有内容）
cat > CHANGELOG.md << 'EOF'
# 变更日志

本文档记录项目的所有重要变更。

---

## [2025-10-23] MLOps 协议实现

### 新增
- 三种数据分割协议: random/temporal/brand_ood
- 完整指标计算系统: ECE, NLL, AUROC, F1
- 工件自动生成回调: ProtocolArtifactsCallback

[更多内容...]

EOF
```

### 步骤 3: 集成到训练流程
在 `scripts/train_hydra.py` 或 callback 中添加：

```python
from src.utils.documentation import ChangelogManager

# 训练结束后
if new_feature_added:
    changelog = ChangelogManager()
    changelog.append_change(...)
```

## 优势

1. **✅ 增量式**: 每次只追加新内容，不重新生成整个文档
2. **✅ 可追溯**: 清晰的时间线和历史记录
3. **✅ 自动化**: 通过工具类自动追加，减少手动工作
4. **✅ 结构化**: 统一的格式和组织方式
5. **✅ 易维护**: 分层管理，职责清晰

## 总结

- 根目录保留精简的 `CHANGELOG.md`（追加式）
- 详细文档放在 `docs/implementations/`（独立文件）
- 实验报告保留在各自的实验目录
- 使用工具类自动化追加新变更

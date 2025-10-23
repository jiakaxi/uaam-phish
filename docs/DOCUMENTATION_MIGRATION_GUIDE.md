# 文档管理迁移指南

## 问题描述

当前项目中存在以下文档重复生成的问题：
- `FINAL_SUMMARY_CN.md`
- `CHANGES_SUMMARY.md`
- `FILES_MANIFEST.md`
- `IMPLEMENTATION_REPORT.md`

这些文档都是一次性生成的完整快照，每次有新功能时需要重新生成整个文档。

## 解决方案：追加式文档管理

采用**增量追加**的方式，而不是每次都重新生成整个文档。

---

## 📁 推荐的新文档结构

```
项目根目录/
├── README.md                           # 项目主文档
├── CHANGELOG.md                        # 变更日志（追加式）⭐ NEW
│
├── docs/
│   ├── ARCHITECTURE.md                 # 架构文档
│   ├── DOCUMENTATION_STRUCTURE.md      # 文档结构说明
│   │
│   ├── implementations/                # 详细实现文档 ⭐ NEW
│   │   ├── README.md                   # 实现索引
│   │   ├── 2025-10-23_mlops_protocols.md
│   │   └── 2025-10-24_new_feature.md
│   │
│   └── history/                        # 历史快照 ⭐ NEW
│       ├── 2025-10-23_mlops_summary.md
│       └── ...
│
└── experiments/<run>/                  # 实验文档（自动生成）
    ├── SUMMARY.md
    └── results/
        └── implementation_report.md
```

---

## 🚀 快速开始

### 1. 记录新功能变更

```python
from src.utils.documentation import ChangelogManager

# 创建 Changelog 管理器
changelog = ChangelogManager()

# 追加新变更
changelog.append_change(
    feature_name="添加不确定性估计模块",
    added=[
        "Monte Carlo Dropout 支持",
        "温度缩放校准",
    ],
    modified=[
        "URLOnlyModule - 支持不确定性输出",
    ],
    stats={
        "新增文件": 2,
        "修改文件": 1,
    },
)
```

### 2. 创建详细实现文档

```python
from src.utils.documentation import ImplementationManager, generate_implementation_template

# 创建实现管理器
impl_mgr = ImplementationManager()

# 生成文档模板
content = generate_implementation_template(
    feature_name="不确定性估计模块",
    summary="实现了基于 Monte Carlo Dropout 的不确定性估计...",
    added_files=["src/modules/mc_dropout.py"],
    modified_files=["src/systems/url_only_module.py"],
)

# 创建文档
impl_mgr.create_implementation_doc(
    feature_name="不确定性估计模块",
    content=content,
)
```

### 3. 集成到训练流程（可选）

在 `scripts/train_hydra.py` 或自定义 callback 中：

```python
from src.utils.documentation import ChangelogManager

# 训练结束后记录变更
changelog = ChangelogManager()
changelog.append_change(
    feature_name=f"实验: {exp_name}",
    added=[f"新实验配置: {protocol}"],
    stats={
        "测试准确率": f"{test_acc:.4f}",
        "测试 AUROC": f"{test_auroc:.4f}",
    },
)
```

---

## 📋 迁移步骤

### 步骤 1: 归档现有文档

```bash
# 创建 history 和 implementations 目录
mkdir -p docs/history
mkdir -p docs/implementations

# 移动现有文档到 history（归档）
mv FINAL_SUMMARY_CN.md docs/history/2025-10-23_mlops_implementation_summary.md
mv CHANGES_SUMMARY.md docs/history/2025-10-23_mlops_changes.md
mv FILES_MANIFEST.md docs/history/2025-10-23_files_manifest.md

# 移动实现报告到 implementations
mv IMPLEMENTATION_REPORT.md docs/implementations/2025-10-23_mlops_protocols.md
```

### 步骤 2: 创建新的 CHANGELOG.md

可以手动创建，或使用示例脚本：

```bash
python examples/document_change_example.py
```

### 步骤 3: 更新 README.md

在 `README.md` 中添加链接：

```markdown
## 文档

- [变更日志](CHANGELOG.md) - 项目变更记录
- [架构文档](docs/ARCHITECTURE.md) - 系统架构说明
- [实现文档](docs/implementations/) - 功能实现详情
```

---

## 🎯 使用场景

### 场景 1: 实现新功能后

```python
# 1. 更新 Changelog
changelog = ChangelogManager()
changelog.append_change(
    feature_name="新功能名称",
    added=["新增的内容"],
    modified=["修改的内容"],
    stats={"新增文件": 2},
    doc_link="docs/implementations/2025-10-XX_new_feature.md",
)

# 2. 创建详细文档
impl_mgr = ImplementationManager()
impl_mgr.create_implementation_doc(
    feature_name="新功能名称",
    content=detailed_content,
)
```

### 场景 2: 查看最近变更

```python
changelog = ChangelogManager()
recent_changes = changelog.read_latest(n=3)
print(recent_changes)
```

### 场景 3: 列出所有实现

```python
impl_mgr = ImplementationManager()
all_implementations = impl_mgr.list_implementations()
for impl in all_implementations:
    print(f"{impl['date']} - {impl['feature']}")
```

---

## 📊 对比：迁移前后

### 迁移前（当前方式）

```
❌ 每次新功能都重新生成整个 FINAL_SUMMARY_CN.md
❌ 历史记录被覆盖
❌ 难以追踪变更历史
❌ 文档冗余和重复
```

### 迁移后（推荐方式）

```
✅ 增量追加到 CHANGELOG.md
✅ 历史记录完整保留
✅ 清晰的时间线和变更轨迹
✅ 详细文档独立管理（docs/implementations/）
✅ 自动化工具支持
```

---

## 🛠️ 工具 API 参考

### ChangelogManager

```python
changelog = ChangelogManager(
    changelog_path="CHANGELOG.md",  # Changelog 文件路径
    root_dir=None,                  # 项目根目录（默认当前目录）
)

changelog.append_change(
    feature_name="功能名称",        # 必需
    added=["新增项"],               # 可选
    modified=["修改项"],            # 可选
    removed=["移除项"],             # 可选
    config_changes=["配置变更"],    # 可选
    stats={"key": "value"},         # 可选
    doc_link="path/to/doc.md",      # 可选
    date="2025-10-24",              # 可选（默认今天）
)

recent = changelog.read_latest(n=3)  # 读取最近 n 条记录
```

### ImplementationManager

```python
impl_mgr = ImplementationManager(
    implementations_dir="docs/implementations",  # 实现文档目录
    root_dir=None,                               # 项目根目录
)

doc_path = impl_mgr.create_implementation_doc(
    feature_name="功能名称",      # 必需
    content="# 文档内容",         # 必需（Markdown）
    date="2025-10-24",            # 可选
    status="✅ 完成",              # 可选
)

implementations = impl_mgr.list_implementations()  # 列出所有实现
```

### 模板生成

```python
from src.utils.documentation import generate_implementation_template

content = generate_implementation_template(
    feature_name="功能名称",
    summary="功能摘要",
    added_files=["file1.py"],
    modified_files=["file2.py"],
    stats={"新增文件": 1},
)
```

---

## ✅ 最佳实践

1. **每次实现新功能后立即记录**
   - 更新 `CHANGELOG.md`（简要记录）
   - 创建详细实现文档（完整记录）

2. **使用统一的日期格式**
   - 格式：`YYYY-MM-DD`
   - 示例：`2025-10-24`

3. **保持 Changelog 条目简洁**
   - 每个条目不超过 20 行
   - 详细内容放在实现文档中

4. **定期审查和归档**
   - 每个月审查一次实现文档
   - 归档旧的实验记录

5. **版本控制**
   - 将 `CHANGELOG.md` 和 `docs/implementations/` 纳入 Git 管理
   - 每次变更都提交

---

## 📚 示例

完整示例请参考：`examples/document_change_example.py`

运行示例：

```bash
# 查看所有示例
python examples/document_change_example.py

# 或直接在脚本中取消注释特定示例函数
```

---

## 🤔 常见问题

### Q: 现有的文档需要删除吗？

A: 不需要删除，建议移动到 `docs/history/` 目录归档保存。

### Q: 是否必须使用这些工具？

A: 不是必须的。这些工具是为了方便自动化，你也可以手动编辑 `CHANGELOG.md` 和创建实现文档。

### Q: 如何与现有的实验报告集成？

A: `ProtocolArtifactsCallback` 生成的实验报告保持不变，在实验目录下。`CHANGELOG.md` 是项目级别的变更记录，两者互补。

### Q: 能否自动生成 CHANGELOG？

A: 可以。在训练脚本或 callback 中调用 `ChangelogManager.append_change()` 即可自动追加记录。

---

## 📞 总结

**推荐行动：**

1. ✅ 将现有的 4 个总结文档移动到 `docs/history/` 归档
2. ✅ 创建新的 `CHANGELOG.md`（追加式）
3. ✅ 使用 `src/utils/documentation.py` 中的工具类管理文档
4. ✅ 以后每次新功能都追加记录，不重新生成

**好处：**
- 增量式管理，不丢失历史
- 清晰的时间线
- 易于维护和查阅
- 自动化支持

---

*更新时间: 2025-10-24*

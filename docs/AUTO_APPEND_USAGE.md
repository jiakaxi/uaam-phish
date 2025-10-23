# 自动追加文档功能使用指南

## 📋 概述

训练脚本已集成 `DocumentationCallback`，可以在训练结束后自动将实验结果追加到项目文档。

---

## 🚀 快速开始

### 方式 1：启用自动追加（推荐用于重要实验）

```bash
# 启用自动追加
python scripts/train_hydra.py logging.auto_append_docs=true

# 或使用协议 + 自动追加
python scripts/train_hydra.py protocol=temporal logging.auto_append_docs=true
```

**效果**：训练结束后，实验结果会自动追加到 `FINAL_SUMMARY_CN.md`

### 方式 2：默认行为（不自动追加）

```bash
# 默认不追加（需要手动记录重要实验）
python scripts/train_hydra.py
```

**效果**：只生成实验报告到 `experiments/<run>/results/`，不追加到项目文档

### 方式 3：配置文件中启用

编辑 `configs/default.yaml` 或创建自定义配置：

```yaml
logging:
  auto_append_docs: true  # 启用自动追加
  append_to_summary: true   # 追加到 FINAL_SUMMARY_CN.md
  append_to_changes: false  # 不追加到 CHANGES_SUMMARY.md
```

然后正常运行：

```bash
python scripts/train_hydra.py
```

---

## ⚙️ 配置选项

### 在 `configs/default.yaml` 中

```yaml
logging:
  auto_append_docs: false  # 是否启用自动追加（默认关闭）
  append_to_summary: true   # 是否追加到 FINAL_SUMMARY_CN.md
  append_to_changes: false  # 是否追加到 CHANGES_SUMMARY.md
```

### 命令行覆盖

```bash
# 启用自动追加，只追加到 FINAL_SUMMARY_CN.md
python scripts/train_hydra.py \
    logging.auto_append_docs=true \
    logging.append_to_summary=true \
    logging.append_to_changes=false

# 同时追加到两个文档
python scripts/train_hydra.py \
    logging.auto_append_docs=true \
    logging.append_to_summary=true \
    logging.append_to_changes=true
```

---

## 📊 追加的内容

当 `auto_append_docs=true` 时，训练结束后会追加：

### 追加到 `FINAL_SUMMARY_CN.md`（如果 `append_to_summary=true`）

```markdown
---

# 实验: <exp_name>

**实施日期**: 2025-10-24
**实施状态**: ✅ 完成并验证

## 📋 实施摘要

实验完成，模型在测试集上的性能如下：

**测试指标**:
- 准确率 (Accuracy): 0.8523
- AUROC: 0.9234
- F1 Score: 0.8756
- Loss: 0.3421

## 📦 交付成果

- 测试准确率: 0.8523
- 测试 AUROC: 0.9234
- 测试 F1: 0.8756

## 🎯 功能实现

- ✅ 准确率: 85.23%
- ✅ AUROC: 92.34%
- ✅ F1 Score: 87.56%

## 🧪 测试结果

✅ 测试完成 - Acc: 85.23%, AUROC: 92.34%
```

### 追加到 `CHANGES_SUMMARY.md`（如果 `append_to_changes=true`）

```markdown
---

# 实验: <exp_name>

**日期**: 2025-10-24
**类型**: 实验运行

## 🆕 新增功能

- 完成模型测试，准确率 85.23%

## 📊 统计数据

| 类别 | 数量 |
|------|------|
| 测试准确率 | 0.8523 |
| 测试 AUROC | 0.9234 |
| 测试 F1 | 0.8756 |
```

---

## 💡 使用建议

### 什么时候启用自动追加？

**推荐启用**：
- ✅ 重要的实验里程碑
- ✅ 新功能验证实验
- ✅ 性能提升的关键实验
- ✅ 准备发布的最终实验

**不推荐启用**：
- ❌ 日常调试实验
- ❌ 超参数搜索的每次尝试
- ❌ 测试性质的快速运行

### 推荐工作流

#### 方案 A：默认关闭，重要时手动启用

```bash
# 日常实验（不追加）
python scripts/train_hydra.py

# 重要实验（手动启用追加）
python scripts/train_hydra.py logging.auto_append_docs=true
```

**优点**：避免文档过度追加，只记录重要实验

#### 方案 B：始终启用（不推荐）

```yaml
# configs/default.yaml
logging:
  auto_append_docs: true
```

**缺点**：每次实验都追加，可能导致文档过长

#### 方案 C：创建专门配置（推荐）

创建 `configs/profiles/milestone.yaml`：

```yaml
logging:
  auto_append_docs: true
  append_to_summary: true
  append_to_changes: false
```

使用时：

```bash
# 重要实验
python scripts/train_hydra.py +profiles/milestone

# 日常实验
python scripts/train_hydra.py
```

---

## 🔍 验证自动追加

### 1. 运行一个测试实验

```bash
python scripts/train_hydra.py \
    logging.auto_append_docs=true \
    train.epochs=1 \
    +profiles/local
```

### 2. 检查日志输出

训练过程中会看到：

```
>> 已启用项目文档自动追加
...
[训练过程]
...
====================================================
追加实验结果到文档: 实验: <exp_name>
====================================================
✅ 已追加到: D:\uaam-phish\FINAL_SUMMARY_CN.md
✅ 文档追加完成
====================================================
```

### 3. 查看追加的内容

```bash
# 打开文档，滚动到底部
notepad FINAL_SUMMARY_CN.md

# 或使用命令查看最后50行
tail -n 50 FINAL_SUMMARY_CN.md  # Linux/Mac
Get-Content FINAL_SUMMARY_CN.md -Tail 50  # Windows PowerShell
```

---

## 🛠️ 自定义追加内容

如果需要自定义追加的内容，可以在代码中修改：

### 修改 `scripts/train_hydra.py`

```python
doc_callback = DocumentationCallback(
    feature_name=f"实验: {exp_name}",
    append_to_summary=True,
    custom_summary=f"自定义摘要: {protocol} 协议实验",
    custom_deliverables=[
        f"自定义交付物 1",
        f"自定义交付物 2",
    ],
)
```

### 或直接使用工具类

在训练脚本末尾添加：

```python
from src.utils.documentation import DocumentationAppender

if some_condition:
    doc = DocumentationAppender()
    doc.append_to_summary(
        feature_name="自定义实验记录",
        summary="自定义摘要",
        deliverables=["自定义交付物"],
    )
```

---

## 📚 相关文档

- **快速指南**: `QUICK_START_DOCS.md`
- **详细教程**: `docs/APPEND_DOCUMENTATION_GUIDE.md`
- **工具类 API**: `src/utils/documentation.py`
- **回调实现**: `src/utils/doc_callback.py`

---

## ❓ 常见问题

### Q: 自动追加会覆盖现有内容吗？

A: 不会。内容总是**追加**到文档末尾，现有内容完全保留。

### Q: 如何禁用自动追加？

A: 有三种方式：
1. 不设置 `logging.auto_append_docs` (默认 false)
2. 命令行: `logging.auto_append_docs=false`
3. 配置文件: `logging.auto_append_docs: false`

### Q: 可以选择追加到哪些文档吗？

A: 可以。通过配置控制：
```bash
# 只追加到 FINAL_SUMMARY_CN.md
logging.append_to_summary=true logging.append_to_changes=false

# 同时追加到两个文档
logging.append_to_summary=true logging.append_to_changes=true
```

### Q: 追加的内容可以编辑吗？

A: 可以。追加后可以手动编辑文档，修改或删除任何内容。

### Q: 每次实验都应该追加吗？

A: 不推荐。建议只在重要实验时启用，避免文档过度冗长。

### Q: 如何查看最近追加的内容？

A: 打开文档，滚动到底部，或使用：
```bash
# Windows PowerShell
Get-Content FINAL_SUMMARY_CN.md -Tail 50
```

---

## 🎯 总结

**核心要点**：
1. ✅ 自动追加功能已集成到训练脚本
2. ✅ 默认**关闭**，需要手动启用
3. ✅ 通过配置灵活控制
4. ✅ 只追加不覆盖，历史完整保留

**推荐使用**：
- 日常实验：不启用（默认）
- 重要实验：命令行启用 `logging.auto_append_docs=true`
- 或创建专门的 profile 配置

---

*更新时间: 2025-10-24*

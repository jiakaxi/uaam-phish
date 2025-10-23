# 文档追加使用指南

## 概述

为了避免每次运行都生成新的重复文档，我们提供了**追加式文档管理**工具，可以将新内容追加到现有文档：

- `FINAL_SUMMARY_CN.md` - 项目实施总结
- `CHANGES_SUMMARY.md` - 变更摘要
- `FILES_MANIFEST.md` - 文件清单

**原则**：保留现有文档，只追加新内容，不创建重复用途的新文档。

---

## 🚀 快速开始

### 方式 1: Python 脚本中使用

```python
from src.utils.documentation import DocumentationAppender

# 创建追加器
doc = DocumentationAppender()

# 追加到 FINAL_SUMMARY_CN.md
doc.append_to_summary(
    feature_name="新功能名称",
    summary="功能描述",
    deliverables=["交付物1", "交付物2"],
    features=["✅ 功能A", "✅ 功能B"],
)

# 追加到 CHANGES_SUMMARY.md
doc.append_to_changes(
    feature_name="新功能名称",
    added_files=["file1.py", "file2.py"],
    modified_files=["file3.py"],
    stats={"新增文件": 2, "修改文件": 1},
)

# 追加到 FILES_MANIFEST.md
doc.append_to_manifest(
    feature_name="新功能名称",
    added_files=[
        {'path': 'src/new_module.py', 'lines': 100, 'description': '新模块'},
    ],
)
```

### 方式 2: 一次性追加到所有文档

```python
doc = DocumentationAppender()

doc.append_all(
    feature_name="新功能",
    summary_kwargs={'summary': '...', 'deliverables': [...]},
    changes_kwargs={'added_files': [...], 'stats': {...}},
    manifest_kwargs={'added_files': [...]},
)
```

### 方式 3: 自动追加（训练结束后）

在 `scripts/train_hydra.py` 中添加回调：

```python
from src.utils.doc_callback import DocumentationCallback

# 添加文档回调
doc_callback = DocumentationCallback(
    feature_name=f"实验: {exp_name}",
    append_to_summary=True,  # 追加到 FINAL_SUMMARY_CN.md
    append_to_changes=False,  # 不追加到 CHANGES_SUMMARY.md
)

callbacks.append(doc_callback)

# 训练
trainer = Trainer(callbacks=callbacks)
trainer.fit(...)
trainer.test(...)  # 测试完成后自动追加
```

---

## 📋 API 参考

### DocumentationAppender

#### `append_to_summary()`
追加到 `FINAL_SUMMARY_CN.md`

**参数**:
- `feature_name` (str): 功能名称
- `date` (str, optional): 日期，默认今天
- `status` (str): 状态标记，如 "✅ 完成"
- `summary` (str, optional): 摘要说明
- `deliverables` (List[str], optional): 交付成果列表
- `features` (List[str], optional): 功能列表
- `test_results` (str, optional): 测试结果
- `usage` (str, optional): 使用说明

#### `append_to_changes()`
追加到 `CHANGES_SUMMARY.md`

**参数**:
- `feature_name` (str): 功能名称
- `date` (str, optional): 日期，默认今天
- `implementation_type` (str): 实现类型，如 "功能增强"
- `added_files` (List[str], optional): 新增文件列表
- `modified_files` (List[str], optional): 修改文件列表
- `reused_configs` (List[str], optional): 复用配置列表
- `new_features` (List[str], optional): 新功能列表
- `stats` (Dict, optional): 统计数据

#### `append_to_manifest()`
追加到 `FILES_MANIFEST.md`

**参数**:
- `feature_name` (str): 功能名称
- `date` (str, optional): 日期，默认今天
- `added_files` (List[Dict], optional): 新增文件，每个dict包含 'path', 'lines', 'description'
- `modified_files` (List[Dict], optional): 修改文件，每个dict包含 'path', 'changes'
- `total_stats` (Dict, optional): 总统计

#### `append_all()`
一次性追加到所有文档

**参数**:
- `feature_name` (str): 功能名称
- `date` (str, optional): 日期，默认今天
- `summary_kwargs` (Dict, optional): append_to_summary 的参数
- `changes_kwargs` (Dict, optional): append_to_changes 的参数
- `manifest_kwargs` (Dict, optional): append_to_manifest 的参数

---

## 💡 使用场景

### 场景 1: 实现新功能后手动记录

```python
from src.utils.documentation import DocumentationAppender

doc = DocumentationAppender()

doc.append_all(
    feature_name="数据增强模块",
    summary_kwargs={
        'summary': '实现了 URL 数据增强功能',
        'deliverables': [
            '`src/data/augmentation.py` - 数据增强实现',
        ],
        'features': [
            '✅ URL 变换增强',
            '✅ 混合增强策略',
        ],
    },
    changes_kwargs={
        'added_files': [
            '**`src/data/augmentation.py`** (200行) - 数据增强',
        ],
        'stats': {'新增文件': 1, '新增代码': '~200行'},
    },
)
```

### 场景 2: 实验运行后自动记录

```python
# 在 scripts/train_hydra.py 中
from src.utils.doc_callback import DocumentationCallback

callbacks = [
    # ... 其他回调
    DocumentationCallback(
        feature_name=f"{protocol} 协议实验",
        append_to_summary=True,
        custom_summary=f"运行 {protocol} 协议的实验",
    ),
]

trainer = Trainer(callbacks=callbacks)
```

### 场景 3: 仅记录重要的里程碑

```python
# 只在重要功能完成时记录
if is_major_milestone:
    doc = DocumentationAppender()
    doc.append_to_summary(
        feature_name="项目里程碑: v1.0",
        summary="完成了第一个稳定版本",
        features=[
            "✅ 核心功能完整",
            "✅ 测试覆盖率 >90%",
            "✅ 文档完整",
        ],
    )
```

---

## 🎯 最佳实践

### 1. 什么时候追加文档？

**推荐追加**:
- ✅ 实现重要的新功能
- ✅ 完成重大重构
- ✅ 重要的实验结果
- ✅ 版本里程碑

**不推荐追加**:
- ❌ 小的 bug 修复
- ❌ 代码格式调整
- ❌ 注释更新
- ❌ 每次训练实验（除非有特殊意义）

### 2. 保持简洁

每个追加条目应该：
- 📝 简明扼要（每个section 5-10行）
- 🎯 突出重点
- 📊 包含关键数据

### 3. 使用统一格式

```python
# 推荐：使用清晰的层次结构
doc.append_to_summary(
    feature_name="明确的功能名",  # 简短但清晰
    summary="简明的描述",         # 1-2段
    deliverables=[...],          # 列表形式
    features=[...],              # 带 ✅ 标记
)

# 不推荐：过于冗长或模糊
doc.append_to_summary(
    feature_name="做了很多改动",  # 太模糊
    summary="改了好多文件...",   # 缺乏细节
)
```

### 4. 定期审查

建议每个月审查一次文档：
- 检查是否有重复内容
- 归档旧的实验记录
- 保持文档整洁

---

## 📚 示例

完整示例请参考：`examples/append_documentation_example.py`

```bash
# 运行示例
python examples/append_documentation_example.py
```

---

## ⚠️ 注意事项

1. **不要重复追加相同内容**
   - 每次实现只追加一次
   - 如果需要更新，手动编辑文档

2. **只追加到相关文档**
   - 不是所有功能都需要追加到所有三个文档
   - 根据实际需要选择追加目标

3. **保持文档可读性**
   - 如果文档变得太长，考虑归档旧内容
   - 使用清晰的分隔符 (`---`)

4. **只在确实需要时创建新文档**
   - 优先追加到现有文档
   - 只有全新类型的文档才创建新文件

---

## 🔄 与现有系统的关系

```
训练流程
    ↓
实验结果
    ↓
DocumentationCallback (可选)
    ↓
自动追加到 FINAL_SUMMARY_CN.md
    ↓
保留历史记录，不覆盖
```

**实验报告 vs 项目文档**:
- `experiments/<run>/results/implementation_report.md` - 单次实验的详细报告（自动生成）
- `FINAL_SUMMARY_CN.md` - 项目级别的重要里程碑（手动或自动追加）

两者互补，不冲突。

---

## 📞 总结

**核心优势**:
- ✅ 保留现有文档结构
- ✅ 增量追加，不重新生成
- ✅ 完整的历史记录
- ✅ 简单易用的 API
- ✅ 可自动化集成

**使用流程**:
1. 实现新功能
2. 使用 `DocumentationAppender` 追加记录
3. 或使用 `DocumentationCallback` 自动追加

---

*更新时间: 2025-10-24*

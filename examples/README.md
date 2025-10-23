# 示例代码

本目录包含项目功能的示例代码。

## 📁 文件说明

### MLOps 协议
- `run_protocol_experiments.py` - 演示如何使用不同的数据分割协议

### 文档管理
- `append_documentation_example.py` - 文档追加功能的5个详细示例
- `quick_append_demo.py` - 快速演示文档追加（交互式）
- `document_change_example.py` - 文档管理工具使用示例（旧版，保留参考）

## 🚀 快速开始

### A. MLOps 协议示例

#### 1. 运行协议分割示例

```bash
python examples/run_protocol_experiments.py
```

这将：
- 测试所有3种协议（random, temporal, brand_ood）
- 生成分割统计表
- 保存分割后的数据到 `examples/output/`

### B. 文档管理示例

#### 1. 查看文档追加示例（不执行）

```bash
python examples/append_documentation_example.py
```

查看5个示例的代码，了解如何使用。

#### 2. 快速演示（实际追加）

```bash
python examples/quick_append_demo.py
```

⚠️ 会向实际文档追加示例内容，运行前请确认。

#### 3. 测试文档追加功能

```bash
python -m pytest tests/test_documentation_append.py -v
```

运行自动化测试，验证功能正常（不影响实际文档）。

### 2. 使用 Hydra 训练脚本

#### Random 协议（默认）
```bash
python scripts/train_hydra.py
```

#### Temporal 协议
```bash
python scripts/train_hydra.py protocol=temporal
```

#### Brand-OOD 协议
```bash
python scripts/train_hydra.py protocol=brand_ood
```

## 📊 输出结构

运行后会在 `experiments/<run_name>/results/` 生成：

```
results/
├── roc_{protocol}.png          # ROC曲线
├── calib_{protocol}.png         # 校准曲线（带ECE标注）
├── splits_{protocol}.csv        # 分割统计表
├── metrics_{protocol}.json      # 完整指标
└── implementation_report.md     # 实现报告
```

## 🔍 协议说明

### Random
- 分层随机分割（按label和brand）
- 默认协议
- 始终可用

### Temporal
- 按timestamp时间顺序分割
- 要求数据包含 `timestamp` 列
- 如果缺失，自动降级到random

### Brand-OOD
- 品牌域外泛化测试
- 要求数据包含 `brand` 列
- 确保 train/test 品牌集完全不相交
- 如果品牌数≤2或检查失败，降级到random

## 📝 自定义配置

### 修改分割比例

编辑 `configs/data/url_only.yaml`:

```yaml
data:
  split_ratios:
    train: 0.7
    val: 0.15
    test: 0.15
```

### 使用自定义配置运行

```bash
python scripts/train_hydra.py \
    protocol=temporal \
    data.split_ratios.train=0.8 \
    data.split_ratios.val=0.1 \
    data.split_ratios.test=0.1
```

## 🛡️ URL 编码器保护

URL编码器架构已被锁定：
- 2层双向LSTM (BiLSTM)
- 字符级tokenization
- Hidden size: 128
- Output dim: 256

任何尝试修改这些参数都会触发断言错误。

## 📝 文档管理使用说明

### 在代码中追加文档

```python
from src.utils.documentation import DocumentationAppender

doc = DocumentationAppender()

# 追加到 FINAL_SUMMARY_CN.md
doc.append_to_summary(
    feature_name="新功能",
    summary="功能描述",
    deliverables=["交付物"],
    features=["✅ 功能A"],
)
```

### 集成到训练流程

在 `scripts/train_hydra.py` 中：

```python
from src.utils.doc_callback import DocumentationCallback

callbacks = [
    DocumentationCallback(
        feature_name=f"实验: {exp_name}",
        append_to_summary=True,
    ),
]
```

### 更多文档管理信息

- [快速指南](../QUICK_START_DOCS.md) - 3分钟上手
- [详细教程](../docs/APPEND_DOCUMENTATION_GUIDE.md) - 完整API和场景

## 📚 更多文档

### MLOps 协议
- [快速入门指南](../docs/QUICKSTART_MLOPS_PROTOCOLS.md)
- [完整实现报告](../IMPLEMENTATION_REPORT.md)

### 项目架构
- [系统架构](../docs/PROJECT_ARCHITECTURE_CN.md)

## ❓ 常见问题

### Q: 协议自动降级到random怎么办？
A: 检查 `metrics_{protocol}.json` 中的 `warnings.downgraded_reason` 字段，了解降级原因。

### Q: 如何验证品牌集不相交？
A: 查看 `splits_{protocol}.csv` 中的 `brand_intersection_ok` 列，应该为 `True`。

### Q: ECE bins 数量如何确定？
A: 自动计算：`max(3, min(15, floor(sqrt(N)), 10))`，实际使用的bins数记录在 `metrics_{protocol}.json.ece_bins_used`。

---

*更新日期: 2025-10-24*

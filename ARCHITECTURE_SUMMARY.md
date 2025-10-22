# 架构总结 - 一目了然

> **更新:** 2025-10-22

---

## ✅ 回答你的问题

### 1. 当前代码库是什么？

**✅ Lightning + OmegaConf + 字符级 BiLSTM 的 URL-only 流水线**

```python
# 主流架构（正在使用）
URLEncoder (BiLSTM)           # src/models/url_encoder.py (10-54行)
    ↓
UrlOnlyModule (Lightning)     # src/systems/url_only_module.py
    ↓
UrlDataModule (Lightning)     # src/datamodules/url_datamodule.py
    ↓
UrlDataset (字符级编码)       # src/data/url_dataset.py
```

**⚠️ HuggingFace BERT 是 Legacy（向后兼容）**

```python
# Legacy 架构（仅用于多模态实验）
UrlBertEncoder (BERT)         # src/models/url_encoder.py (57-84行)
    ↑
标记为 "Legacy HuggingFace-based encoder kept for backward compatibility"
```

---

### 2. 是否需要重构？

**❌ 不同意！不需要重构。**

**原因：**

1. ✅ **字符级 BiLSTM 已经是主方案**
   - `URLEncoder` 是默认使用的
   - 文件结构清晰合理

2. ✅ **HuggingFace 已经在正确位置**
   - 标记为 Legacy
   - 保留用于多模态实验
   - 不影响主流程

3. ✅ **不需要创建 legacy/ 目录**
   - 两个编码器在同一文件中共存
   - 通过配置文件选择使用哪个

---

## 📊 双架构对比

| 特性 | 字符级 BiLSTM (主流) | HuggingFace BERT (Legacy) |
|------|---------------------|--------------------------|
| **类名** | `URLEncoder` | `UrlBertEncoder` |
| **位置** | url_encoder.py:10-54 | url_encoder.py:57-84 |
| **状态** | ✅ 主流使用 | ⚠️ Legacy |
| **参数** | ~1M | ~110M |
| **速度** | ⚡⚡⚡⚡⚡ | ⚡⚡ |
| **依赖** | 无 | transformers |
| **学习率** | 1e-3 | 2e-5 |
| **批次** | 32 | 16 |

---

## 🔧 配置文件

### 主流配置（默认）

```yaml
# configs/model/url_encoder.yaml
model:
  _target_: src.models.url_encoder.URLEncoder  # ✅ 字符级
  vocab_size: 128
  embedding_dim: 128
  hidden_dim: 128
  proj_dim: 256
```

### Legacy 配置

```yaml
# configs/model/url_encoder_legacy.yaml
model:
  _target_: src.models.url_encoder.UrlBertEncoder  # ⚠️ BERT
  pretrained_name: roberta-base
```

---

## 🚀 使用方式

### 默认（推荐）

```bash
# 使用字符级 BiLSTM
python scripts/train_hydra.py
```

### Legacy（特殊场景）

```bash
# 使用 HuggingFace BERT
python scripts/train_hydra.py model=url_encoder_legacy train.lr=2e-5
```

---

## 📁 当前文件结构

```
src/
├── data/
│   └── url_dataset.py              # ✅ 字符级编码
│
├── datamodules/
│   └── url_datamodule.py           # ✅ Lightning DataModule
│
├── models/
│   └── url_encoder.py              # ✅ 两个编码器共存
│       ├── URLEncoder              # 主流（字符级 BiLSTM）
│       └── UrlBertEncoder          # Legacy（HuggingFace）
│
└── systems/
    └── url_only_module.py          # ✅ 使用 URLEncoder
```

**✅ 结构完美，无需改动！**

---

## ✨ 已完成的修正

1. ✅ 更新 `configs/model/url_encoder.yaml` - 指向 URLEncoder
2. ✅ 创建 `configs/model/url_encoder_legacy.yaml` - 指向 UrlBertEncoder
3. ✅ 更新 `configs/data/url_only.yaml` - 字符级数据配置
4. ✅ 更新 `configs/trainer/default.yaml` - BiLSTM 训练参数
5. ✅ 创建 `docs/ARCHITECTURE_CLARIFICATION.md` - 详细说明

---

## 🎯 建议

### ✅ 保持现状

- 字符级 BiLSTM 为主
- HuggingFace BERT 作为 Legacy
- 通过配置文件切换

### ✅ 未来扩展

当需要多模态融合时：
```python
# 可以使用 UrlBertEncoder 保持一致的嵌入维度
url_encoder = UrlBertEncoder()      # 768-dim
html_encoder = BertEncoder()        # 768-dim
image_encoder = ViTEncoder()        # 768-dim
    ↓
RCAF Fusion (统一的 768-dim 嵌入)
```

---

**总结：你的架构已经非常好了，无需重构！** ✅

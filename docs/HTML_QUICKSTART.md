# HTML模型快速开始 ⚡

> **5分钟上手 HTML钓鱼检测**

---

## 🎯 一分钟检查清单

```bash
# ✅ 1. 依赖检查
pip install transformers beautifulsoup4 lxml

# ✅ 2. 数据检查
python -c "import pandas as pd; df=pd.read_csv('data/processed/master_v2.csv'); print(f'✅ {len(df)} samples')"

# ✅ 3. 快速测试（2分钟）
python scripts/train_hydra.py experiment=html_baseline trainer=local data.sample_fraction=0.05 train.epochs=2 model.freeze_bert=true

# ✅ 4. 查看结果
python scripts/compare_experiments.py --latest 1
```

---

## 🚀 三种训练模式

### 模式1: 快速验证（5分钟）

```bash
python scripts/train_hydra.py \
  experiment=html_baseline \
  trainer=local \
  data.sample_fraction=0.05 \
  train.epochs=2 \
  model.freeze_bert=true
```

### 模式2: DistilBERT基线（推荐，2小时）

```bash
python scripts/train_hydra.py \
  experiment=html_baseline \
  model.bert_model=distilbert-base-uncased \
  trainer=server \
  logger=wandb \
  run.name=html_distilbert_baseline
```

### 模式3: BERT-base最佳（3小时）

```bash
python scripts/train_hydra.py \
  experiment=html_baseline \
  model.bert_model=bert-base-uncased \
  trainer=server \
  logger=wandb \
  hardware.precision=16-mixed \
  run.name=html_bert_baseline
```

---

## 🎛️ 常用参数调整

| 需求 | 参数 | 示例 |
|------|------|------|
| 节省显存 | freeze_bert=true | `model.freeze_bert=true` |
| 降低batch | train.bs=16 | `train.bs=16` |
| 使用DistilBERT | bert_model | `model.bert_model=distilbert-base-uncased` |
| 减少token长度 | html_max_len | `data.html_max_len=256` |
| 梯度累积 | accumulate_grad | `trainer.accumulate_grad_batches=2` |

---

## 📊 显存需求速查

| 配置 | 显存 | 速度 | 性能 |
|------|------|------|------|
| BERT + bs=32 + fp16 | 8GB | 1x | ⭐⭐⭐⭐⭐ |
| DistilBERT + bs=32 + fp16 | 6GB | 2x | ⭐⭐⭐⭐ |
| Freeze BERT + bs=32 + fp16 | 4GB | 3x | ⭐⭐⭐ |

---

## 🔧 故障快速修复

### OOM (显存不足)
```bash
# 方案1: 冻结BERT（推荐）
model.freeze_bert=true

# 方案2: 降低batch
train.bs=16

# 方案3: DistilBERT
model.bert_model=distilbert-base-uncased
```

### 缺少依赖
```bash
pip install transformers>=4.30.0 beautifulsoup4 lxml
```

### 数据路径错误
```bash
# 检查HTML文件
ls data/processed/html/*.html | head -5

# 验证CSV
python -c "import pandas as pd; print(pd.read_csv('data/processed/master_v2.csv').columns)"
```

---

## 📁 关键文件

| 文件 | 功能 |
|------|------|
| `src/models/html_encoder.py` | BERT编码器 |
| `src/systems/html_only_module.py` | 训练模块 |
| `configs/experiment/html_baseline.yaml` | 实验配置 |

---

## 🎯 预期性能

- **AUROC**: 0.92-0.96
- **Accuracy**: 0.88-0.93
- **训练时间**: 1-4小时（取决于配置）

---

## 📚 完整文档

详细指南请参考：`docs/HTML_PROJECT_GUIDE.md`

---

**开始训练吧！** 🚀

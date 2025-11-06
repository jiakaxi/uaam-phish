# HTML模态钓鱼检测 - 完整实施指南

> **日期**: 2025-11-05
> **状态**: ✅ 代码完成，准备训练
> **作者**: UAAM-Phish Team

---

## 📋 目录

1. [项目概览](#项目概览)
2. [文件清单](#文件清单)
3. [环境准备](#环境准备)
4. [数据准备](#数据准备)
5. [训练指南](#训练指南)
6. [故障排除](#故障排除)
7. [性能基线](#性能基线)
8. [下一步](#下一步)

---

## 项目概览

### 🎯 目标

实现基于BERT的HTML内容钓鱼检测，作为多模态系统的重要组成部分。

### ✅ 已完成功能

- **HTMLEncoder**: BERT-base编码器（110M参数）
- **HtmlDataset**: 支持BERT tokenization的数据集
- **HtmlDataModule**: Lightning数据模块，支持三种协议
- **HtmlOnlyModule**: 完整的训练系统
- **配置文件**: Hydra配置，开箱即用

### 🏗️ 架构设计

```
HTML文件 → clean_html() → 纯文本
    ↓
BERT tokenizer → (input_ids, attention_mask)
    ↓
BERT-base → [CLS] token (768维)
    ↓
投影层 → 256维（与URL编码器对齐）
    ↓
分类头 → logit → BCEWithLogitsLoss
```

**关键设计原则**:
- 输出256维，与URLEncoder对齐（未来融合）
- 支持freeze_bert选项（节省显存和训练时间）
- 完整的metrics和artifacts生成
- 三种数据分割协议支持

---

## 文件清单

### 📁 核心代码文件

| 文件 | 行数 | 功能 | 状态 |
|------|------|------|------|
| `src/models/html_encoder.py` | 86 | BERT编码器 | ✅ 完成 |
| `src/data/html_dataset.py` | 111 | Dataset类 | ✅ 完成 |
| `src/datamodules/html_datamodule.py` | 152 | DataModule | ✅ 完成 |
| `src/systems/html_only_module.py` | 291 | Lightning模块 | ✅ 完成 |
| `src/utils/html_clean.py` | 76 | HTML清洗工具 | ✅ 完成 |

### 📁 配置文件

| 文件 | 功能 | 关键参数 |
|------|------|----------|
| `configs/model/html_encoder.yaml` | 模型配置 | bert_model, dropout, freeze_bert |
| `configs/data/html_only.yaml` | 数据配置 | html_max_len=512 |
| `configs/experiment/html_baseline.yaml` | 实验配置 | lr=2e-5, bs=32 |

### 📊 代码结构

```
src/
├── models/
│   └── html_encoder.py          # HTMLEncoder类
├── data/
│   └── html_dataset.py          # HtmlDataset类
├── datamodules/
│   └── html_datamodule.py       # HtmlDataModule类
├── systems/
│   └── html_only_module.py      # HtmlOnlyModule类
└── utils/
    └── html_clean.py            # clean_html(), load_html_from_path()

configs/
├── model/
│   └── html_encoder.yaml        # 模型超参数
├── data/
│   └── html_only.yaml           # 数据路径
└── experiment/
    └── html_baseline.yaml       # 完整实验配置
```

---

## 环境准备

### 1. 依赖安装

```bash
# 核心依赖
pip install transformers>=4.30.0
pip install beautifulsoup4>=4.11.0
pip install lxml>=4.9.0

# 或使用requirements.txt
pip install -r requirements.txt
```

### 2. 验证安装

```bash
# 检查transformers
python -c "from transformers import AutoModel, AutoTokenizer; print('✅ transformers OK')"

# 检查BeautifulSoup
python -c "from bs4 import BeautifulSoup; print('✅ beautifulsoup4 OK')"

# 检查BERT模型（首次会下载）
python -c "
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
print('✅ BERT model OK')
"
```

### 3. 下载BERT模型（可选，首次训练会自动下载）

```bash
# 预下载BERT-base（约440MB）
python -c "
from transformers import AutoModel, AutoTokenizer
AutoModel.from_pretrained('bert-base-uncased')
AutoTokenizer.from_pretrained('bert-base-uncased')
print('✅ BERT-base downloaded')
"

# 预下载DistilBERT（约260MB，更快）
python -c "
from transformers import AutoModel, AutoTokenizer
AutoModel.from_pretrained('distilbert-base-uncased')
AutoTokenizer.from_pretrained('distilbert-base-uncased')
print('✅ DistilBERT downloaded')
"
```

---

## 数据准备

### 1. 数据格式要求

HTML项目需要以下数据：

```csv
# master_v2.csv 必需列
url_text,html_path,label,timestamp,brand,source
https://example.com,data/html/sample1.html,0,2024-01-01,legitimate,dataset_a
https://phish.com,data/html/sample2.html,1,2024-01-02,paypal,dataset_b
```

**必需字段**:
- `html_path`: HTML文件路径（相对或绝对）
- `label`: 标签（0=benign, 1=phishing）

**可选字段** (协议需要):
- `timestamp`: 时间戳（temporal协议）
- `brand`: 品牌名称（brand_ood协议）
- `source`: 数据源（统计用）

### 2. 数据验证

```bash
# 验证CSV格式
python -c "
import pandas as pd
from pathlib import Path

df = pd.read_csv('data/processed/master_v2.csv')
print('✅ 总样本数:', len(df))
print('✅ HTML列存在:', 'html_path' in df.columns)
print('✅ 标签分布:', df['label'].value_counts().to_dict())

# 验证HTML文件
html_exists = df['html_path'].apply(lambda x: Path(x).exists()).sum()
print(f'✅ HTML文件存在: {html_exists}/{len(df)}')
"
```

### 3. 数据升级（如果需要）

如果现有数据集缺少必需字段，运行升级脚本：

```bash
python scripts/upgrade_dataset.py \
  --input data/processed/master.csv \
  --output data/processed/master_v2.csv
```

这将自动：
- 添加`brand`和`timestamp`字段
- 从HTML/URL提取品牌信息
- 生成合理的时间戳

### 4. 检查HTML文件

```bash
# 检查HTML文件可读性
python -c "
from src.utils.html_clean import load_html_from_path, clean_html
html_path = 'data/processed/html/sample.html'
html_text = load_html_from_path(html_path)
clean_text = clean_html(html_text)
print('原始长度:', len(html_text))
print('清洗后长度:', len(clean_text))
print('前100字符:', clean_text[:100])
"
```

---

## 训练指南

### 🚀 快速开始（5分钟验证）

```bash
# 最小测试 - 验证流程
python scripts/train_hydra.py \
  experiment=html_baseline \
  trainer=local \
  data.sample_fraction=0.05 \
  train.epochs=2 \
  model.freeze_bert=true \
  run.name=html_smoke_test

# 预期输出：
# Epoch 1/2: 100%|██████████| ... loss=0.xxx val/auroc=0.xxx
# Epoch 2/2: 100%|██████████| ... loss=0.xxx val/auroc=0.xxx
# ✅ Saved: experiments/html_smoke_test/results/
```

### 📊 标准训练（推荐）

#### 方案1: DistilBERT（推荐，更快）

```bash
python scripts/train_hydra.py \
  experiment=html_baseline \
  model.bert_model=distilbert-base-uncased \
  model.hidden_dim=768 \
  trainer=server \
  logger=wandb \
  run.name=html_distilbert_baseline
```

**优势**:
- 参数量66M（BERT-base的60%）
- 训练速度快2倍
- 显存需求低30%
- 性能损失<2%

#### 方案2: BERT-base（最强性能）

```bash
python scripts/train_hydra.py \
  experiment=html_baseline \
  model.bert_model=bert-base-uncased \
  trainer=server \
  logger=wandb \
  hardware.precision=16-mixed \
  run.name=html_bert_baseline
```

**优势**:
- 参数量110M
- 最佳性能
- 更好的校准

### 🔬 三种协议训练

```bash
# 1. Random协议（默认）
python scripts/train_hydra.py \
  experiment=html_baseline \
  protocol=random \
  run.name=html_random

# 2. Temporal协议（时间序列）
python scripts/train_hydra.py \
  experiment=html_baseline \
  protocol=temporal \
  run.name=html_temporal

# 3. Brand-OOD协议（品牌泛化）
python scripts/train_hydra.py \
  experiment=html_baseline \
  protocol=brand_ood \
  run.name=html_brand_ood
```

### 🎯 超参数调优

#### 学习率搜索

```bash
python scripts/train_hydra.py -m \
  experiment=html_baseline \
  train.lr=1e-5,2e-5,5e-5,1e-4 \
  run.name=html_lr_search
```

#### Batch Size调优

```bash
python scripts/train_hydra.py -m \
  experiment=html_baseline \
  train.bs=16,32,64 \
  run.name=html_bs_search
```

#### Freeze BERT对比

```bash
python scripts/train_hydra.py -m \
  experiment=html_baseline \
  model.freeze_bert=true,false \
  run.name=html_freeze_compare
```

### 📝 查看结果

```bash
# 查看最新实验
python scripts/compare_experiments.py --latest 1

# 对比多个实验
python scripts/compare_experiments.py --latest 5

# 找到最佳模型
python scripts/compare_experiments.py --find_best --metric auroc
```

---

## 故障排除

### 问题1: ModuleNotFoundError: transformers

**症状**:
```
ModuleNotFoundError: No module named 'transformers'
```

**解决**:
```bash
pip install transformers>=4.30.0
# 验证
python -c "import transformers; print(transformers.__version__)"
```

### 问题2: ModuleNotFoundError: bs4

**症状**:
```
ModuleNotFoundError: No module named 'bs4'
```

**解决**:
```bash
pip install beautifulsoup4 lxml
# 验证
python -c "from bs4 import BeautifulSoup; print('OK')"
```

### 问题3: HTML文件路径错误

**症状**:
```
FileNotFoundError: [Errno 2] No such file or directory: 'data/html/xxx.html'
```

**诊断**:
```bash
# 检查路径
python -c "
import pandas as pd
from pathlib import Path
df = pd.read_csv('data/processed/master_v2.csv')
exists = df['html_path'].apply(lambda x: Path(x).exists())
print(f'存在: {exists.sum()}/{len(df)}')
print('第一个不存在的路径:', df.loc[~exists, 'html_path'].iloc[0])
"
```

**解决**:
1. 确认HTML文件存在
2. 检查路径是否相对于项目根目录
3. 重新运行数据准备脚本

### 问题4: CUDA OOM（显存不足）

**症状**:
```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

**解决方案**:

#### 方案A: 降低batch size
```bash
python scripts/train_hydra.py experiment=html_baseline train.bs=16
```

#### 方案B: 使用DistilBERT
```bash
python scripts/train_hydra.py experiment=html_baseline \
  model.bert_model=distilbert-base-uncased
```

#### 方案C: 冻结BERT（推荐）
```bash
python scripts/train_hydra.py experiment=html_baseline \
  model.freeze_bert=true
```

这将：
- 节省50%显存
- 加速训练2-3倍
- 性能损失约3-5%

#### 方案D: 梯度累积
```bash
python scripts/train_hydra.py experiment=html_baseline \
  trainer.accumulate_grad_batches=2 \
  train.bs=16
```
等效于bs=32，但显存需求减半。

#### 方案E: CPU训练（最后手段）
```bash
python scripts/train_hydra.py experiment=html_baseline \
  hardware.accelerator=cpu \
  train.bs=8
```

### 问题5: BERT模型下载慢

**症状**:
```
Downloading: 100%|██| 440M/440M [slow...]
```

**解决**:
```bash
# 方案1: 使用镜像（中国用户）
export HF_ENDPOINT=https://hf-mirror.com
python scripts/train_hydra.py experiment=html_baseline

# 方案2: 离线下载
# 1. 手动下载模型到 ~/.cache/huggingface/
# 2. 或使用本地路径
python scripts/train_hydra.py experiment=html_baseline \
  model.bert_model=/path/to/local/bert-base-uncased
```

### 问题6: HTML清洗过慢

**症状**:
数据加载速度慢（每个样本>1秒）

**解决**:
```bash
# 方案1: 减少max_len
python scripts/train_hydra.py experiment=html_baseline \
  data.html_max_len=256  # 默认512

# 方案2: 增加workers
python scripts/train_hydra.py experiment=html_baseline \
  data.num_workers=8  # 默认4

# 方案3: 使用SSD存储HTML文件
```

---

## 性能基线

### 预期指标

基于论文和类似工作的预期性能：

| 指标 | DistilBERT | BERT-base | Freeze BERT | 说明 |
|------|-----------|-----------|-------------|------|
| **AUROC** | 0.92-0.94 | 0.94-0.96 | 0.91-0.93 | HTML语义特征强 |
| **Accuracy** | 0.88-0.91 | 0.90-0.93 | 0.87-0.90 | 依赖数据集质量 |
| **F1-macro** | 0.87-0.90 | 0.89-0.92 | 0.86-0.89 | 平衡两类 |
| **NLL** | 0.20-0.30 | 0.18-0.25 | 0.22-0.32 | BERT校准较好 |
| **ECE** | 0.03-0.06 | 0.02-0.05 | 0.04-0.07 | 需关注过拟合 |

### 训练时间

基于RTX 3090（24GB）的预估：

| 配置 | Epochs | 时间 | 显存 |
|------|--------|------|------|
| BERT-base (bs=32, fp16) | 50 | 3-4小时 | ~8GB |
| DistilBERT (bs=32, fp16) | 50 | 2小时 | ~6GB |
| Freeze BERT (bs=32, fp16) | 50 | 1小时 | ~4GB |
| BERT-base (bs=16, fp32) | 50 | 6小时 | ~12GB |

### 硬件建议

| 配置 | 最低 | 推荐 | 最佳 |
|------|------|------|------|
| **GPU** | GTX 1060 6GB | RTX 3060 12GB | RTX 3090 24GB |
| **CPU** | 4核 | 8核 | 16核 |
| **RAM** | 16GB | 32GB | 64GB |
| **存储** | HDD | SSD | NVMe SSD |

**配置建议**:
- 6GB显存: freeze_bert=true, bs=16
- 12GB显存: DistilBERT, bs=32
- 24GB显存: BERT-base, bs=64

---

## 下一步

### ✅ 验证清单

训练前请确认：

- [ ] **环境准备**
  - [ ] transformers, beautifulsoup4已安装
  - [ ] BERT模型已下载（或网络可访问）
  - [ ] GPU驱动和CUDA正常

- [ ] **数据准备**
  - [ ] master_v2.csv存在且格式正确
  - [ ] html_path列路径正确
  - [ ] HTML文件可访问
  - [ ] 标签分布合理

- [ ] **配置检查**
  - [ ] experiment=html_baseline存在
  - [ ] batch_size适配显存
  - [ ] 学习率在合理范围（1e-5到5e-5）

### 📅 实施计划

#### 今天（1小时）

```bash
# 1. 环境验证（10分钟）
pip install transformers beautifulsoup4 lxml
python -c "from transformers import AutoModel; print('OK')"

# 2. 数据验证（10分钟）
python -c "
import pandas as pd
df = pd.read_csv('data/processed/master_v2.csv')
print('Samples:', len(df))
print('HTML:', 'html_path' in df.columns)
"

# 3. 快速测试（5分钟）
python scripts/train_hydra.py \
  experiment=html_baseline \
  trainer=local \
  data.sample_fraction=0.05 \
  train.epochs=2 \
  model.freeze_bert=true

# 4. 查看结果（5分钟）
python scripts/compare_experiments.py --latest 1
```

#### 本周（10小时）

1. **Day 1-2: 基线训练**
   - Random协议完整训练
   - 记录baseline性能
   - 验证artifacts生成

2. **Day 3-4: 超参数调优**
   - 学习率搜索（1e-5, 2e-5, 5e-5）
   - Batch size优化（16, 32, 64）
   - Freeze策略对比

3. **Day 5: 协议对比**
   - Temporal协议训练
   - Brand-OOD协议训练
   - 性能对比分析

4. **Day 6-7: 结果分析**
   - 错误案例分析
   - 与URL模型对比
   - 撰写实验报告

#### 本月（40小时）

1. **Week 1: 基线建立**
   - 三种协议完整训练
   - DistilBERT vs BERT对比
   - 性能基线确立

2. **Week 2: 模型优化**
   - 超参数精细调优
   - 混合精度训练
   - 模型集成探索

3. **Week 3: 深度分析**
   - BERT attention可视化
   - 错误分析和改进
   - HTML特征重要性

4. **Week 4: 文档整理**
   - 实验报告撰写
   - 最佳实践总结
   - 论文/报告准备

### 🎯 成功标准

HTML模型达到以下标准即为成功：

✅ **基础性能**
- AUROC ≥ 0.90
- Accuracy ≥ 0.85
- F1-macro ≥ 0.84

✅ **校准质量**
- ECE ≤ 0.10
- NLL ≤ 0.40

✅ **鲁棒性**
- 三种协议均可训练
- 性能标准差 < 0.02
- 无数据泄露

✅ **可复现性**
- 配置完整保存
- 随机种子固定
- 实验可重复

✅ **工程质量**
- 无runtime错误
- 4个artifacts完整
- WandB日志完整

---

## 附录

### A. 配置参数详解

#### 模型参数 (configs/model/html_encoder.yaml)

```yaml
model:
  bert_model: bert-base-uncased  # 或 distilbert-base-uncased
  hidden_dim: 768                # BERT输出维度（固定）
  output_dim: 256                # 投影维度（必须256，融合需要）
  dropout: 0.1                   # Dropout率
  freeze_bert: false             # 是否冻结BERT参数
```

**参数说明**:
- `bert_model`: 可选bert-base-uncased, distilbert-base-uncased, roberta-base
- `freeze_bert=true`: 节省50%显存，加速2-3倍，性能损失3-5%
- `output_dim`: **不要修改**，必须保持256以便未来融合

#### 数据参数 (configs/data/html_only.yaml)

```yaml
data:
  html_max_len: 512      # BERT最大token数（建议256-512）
  num_workers: 4         # DataLoader workers（建议4-8）
  batch_format: tuple    # 不要修改
```

#### 训练参数 (configs/experiment/html_baseline.yaml)

```yaml
train:
  epochs: 50             # 训练轮数
  lr: 2.0e-5            # 学习率（BERT常用1e-5到5e-5）
  bs: 32                # Batch size（根据显存调整）
  weight_decay: 0.01    # 权重衰减（L2正则）
```

### B. 命令行参数速查

```bash
# 模型相关
model.bert_model=distilbert-base-uncased
model.freeze_bert=true
model.dropout=0.2

# 数据相关
data.sample_fraction=0.1        # 使用10%数据
data.html_max_len=256           # 减少token长度
data.num_workers=8              # 增加workers

# 训练相关
train.epochs=100
train.lr=5e-5
train.bs=64
train.weight_decay=0.01

# 硬件相关
hardware.accelerator=gpu
hardware.devices=2              # 多GPU
hardware.precision=16-mixed     # 混合精度

# 协议相关
protocol=random                 # 或 temporal, brand_ood
use_build_splits=true

# 日志相关
logger=wandb
run.name=my_experiment
run.tags=[html,baseline,v1]
```

### C. 文件路径约定

```
project_root/
├── data/
│   └── processed/
│       ├── master_v2.csv              # 主CSV
│       ├── html_train_v2.csv          # 训练集
│       ├── html_val_v2.csv            # 验证集
│       ├── html_test_v2.csv           # 测试集
│       └── html/                      # HTML文件目录
│           ├── benign_001.html
│           ├── phish_001.html
│           └── ...
├── experiments/
│   └── <run_name>/
│       ├── config.yaml                # 完整配置
│       ├── checkpoints/               # 模型检查点
│       │   └── best_model.ckpt
│       ├── results/                   # 结果artifacts
│       │   ├── roc_random.png
│       │   ├── calib_random.png
│       │   ├── splits_random.csv
│       │   └── metrics_random.json
│       └── logs/
│           └── train.log
└── configs/
    └── experiment/
        └── html_baseline.yaml
```

---

## 🔗 相关资源

- **主文档**: `FINAL_SUMMARY_CN.md`
- **论文参考**: Thesis §3.3 (HTML Encoder Architecture)
- **代码示例**: `src/systems/html_only_module.py`
- **配置示例**: `configs/experiment/html_baseline.yaml`
- **数据准备**: `scripts/upgrade_dataset.py`
- **HTML清洗**: `src/utils/html_clean.py`

---

## 📞 获取帮助

遇到问题？

1. 查看本文档的[故障排除](#故障排除)部分
2. 检查`experiments/<run>/logs/train.log`
3. 查看WandB实验页面
4. 提交Issue并附上：
   - 完整错误信息
   - 运行命令
   - 环境信息（GPU型号，Python版本等）

---

**祝HTML模型训练顺利！** 🚀

---

*最后更新: 2025-11-05*

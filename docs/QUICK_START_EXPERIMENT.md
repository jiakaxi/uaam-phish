# 实验快速启动指南

> 5 分钟内开始第一个实验

---

## 🚀 快速开始

### 步骤 1: 准备数据

```bash
# 如果已有原始数据，构建训练集
python scripts/build_master_and_splits.py \
  --benign data/raw/dataset \
  --phish data/raw/fish_dataset \
  --outdir data/processed

# 或者使用简单的数据划分
python scripts/preprocess.py \
  --src data/raw/urls.csv \
  --outdir data/processed
```

### 步骤 2: 运行第一个实验

```bash
# 本地快速测试（CPU，小批量）
python scripts/train.py --profile local --exp_name first_test

# 服务器训练（GPU，大批量）
python scripts/train.py --profile server --exp_name bert_baseline
```

### 步骤 3: 查看结果

```bash
# 查看实验目录
ls -lh experiments/

# 查看实验总结
cat experiments/first_test_*/SUMMARY.md

# 打开可视化图表
# Windows: start experiments\first_test_*\results\
# Linux:   xdg-open experiments/first_test_*/results/
# Mac:     open experiments/first_test_*/results/
```

---

## 📊 自动生成的结果

每次实验运行后，会立即生成：

### ✅ 指标文件 (`results/metrics_final.json`)
```json
{
  "experiment": "first_test_20251021_143022",
  "metrics": {
    "test/loss": 0.1234,
    "test/f1": 0.9567,
    "test/auroc": 0.9823,
    "test/fpr": 0.0234
  }
}
```

### ✅ 可视化图表 (`results/*.png`)
- **training_curves.png** - 训练曲线（Loss, F1, AUROC, FPR）
- **confusion_matrix.png** - 混淆矩阵 + 性能指标
- **roc_curve.png** - ROC 曲线 + AUC
- **threshold_analysis.png** - 最佳阈值分析

### ✅ 实验总结 (`SUMMARY.md`)
自动生成的 Markdown 格式总结

### ✅ 训练日志 (`logs/train.log`)
实时记录的训练过程

### ✅ 模型检查点 (`checkpoints/*.ckpt`)
最佳模型权重，可直接用于推理

---

## 🔄 常用实验场景

### 场景 1: 对比不同模型

```bash
# BERT 基线
python scripts/train.py --profile server --exp_name bert_baseline

# RoBERTa 对比
# 修改 configs/default.yaml 中的 pretrained_name: roberta-base
python scripts/train.py --profile server --exp_name roberta_baseline

# 对比结果
python scripts/compare_experiments.py --exp_names bert_baseline roberta_baseline
```

### 场景 2: 超参数调优

```bash
# 不同学习率
python scripts/train.py --exp_name lr_1e5  # 在配置中设置 lr=1e-5
python scripts/train.py --exp_name lr_2e5  # lr=2e-5
python scripts/train.py --exp_name lr_5e5  # lr=5e-5

# 对比所有学习率实验
python scripts/compare_experiments.py --exp_names lr_1e5 lr_2e5 lr_5e5
```

### 场景 3: 数据消融研究

```bash
# 不同数据量
python scripts/train.py --exp_name data_10pct  # sample_fraction=0.1
python scripts/train.py --exp_name data_50pct  # sample_fraction=0.5
python scripts/train.py --exp_name data_100pct # sample_fraction=1.0

# 对比
python scripts/compare_experiments.py --exp_names data_10pct data_50pct data_100pct
```

---

## 📈 实验管理技巧

### 技巧 1: 使用有意义的实验名称

```bash
# ✅ 好的命名（描述性强）
python scripts/train.py --exp_name bert_dropout02_lr2e5_bs32
python scripts/train.py --exp_name roberta_maxlen512_augmented
python scripts/train.py --exp_name ablation_url_only

# ❌ 避免的命名（无信息量）
python scripts/train.py --exp_name test1
python scripts/train.py --exp_name exp123
```

### 技巧 2: 定期查看最佳实验

```bash
# 查找 F1 最高的实验
python scripts/compare_experiments.py --find_best --metric f1

# 查找 AUROC 最高的实验
python scripts/compare_experiments.py --find_best --metric auroc
```

### 技巧 3: 导出实验报告

```bash
# 导出 CSV（Excel 兼容）
python scripts/compare_experiments.py --all --output experiments_report.csv

# 导出 Markdown（文档友好）
python scripts/compare_experiments.py --all --output experiments_report.md

# 导出 Excel
python scripts/compare_experiments.py --all --output experiments_report.xlsx
```

---

## 🔍 调试和快速迭代

### 调试模式（不保存结果）

```bash
# 快速测试代码，不保存实验结果
python scripts/train.py --profile local --no_save
```

### 小数据快速验证

```bash
# 使用 10% 数据快速验证
# 修改 configs/profiles/local.yaml: sample_fraction: 0.1
python scripts/train.py --profile local --exp_name quick_test
```

---

## 💡 实验配置速查

### 常用配置修改位置

| 参数 | 配置文件 | 位置 |
|------|----------|------|
| 模型名称 | `configs/default.yaml` | `model.pretrained_name` |
| 学习率 | `configs/default.yaml` | `train.lr` |
| 批量大小 | `configs/profiles/*.yaml` | `train.bs` |
| 训练轮数 | `configs/default.yaml` | `train.epochs` |
| 数据采样 | `configs/profiles/*.yaml` | `data.sample_fraction` |
| Dropout | `configs/default.yaml` | `model.dropout` |
| 最大长度 | `configs/default.yaml` | `data.max_length` |

### 快速配置切换

```bash
# 本地开发（CPU，小批量，快速）
python scripts/train.py --profile local

# 服务器训练（GPU，大批量，完整）
python scripts/train.py --profile server
```

---

## 📝 实验记录模板

在项目根目录创建 `EXPERIMENTS_LOG.md`:

```markdown
# 实验记录

## 实验 1: BERT 基线 (2025-10-21)

**目标:** 建立基线性能

**配置:**
- 模型: bert-base-uncased
- 学习率: 2e-5
- Batch size: 16
- Epochs: 5

**结果:**
- Test F1: 0.9234
- Test AUROC: 0.9567
- Test FPR: 0.0456

**结论:** 基线性能可接受，后续尝试 RoBERTa

---

## 实验 2: RoBERTa 对比 (2025-10-21)

**目标:** 验证 RoBERTa 是否优于 BERT

**配置:**
- 模型: roberta-base
- 其他参数同实验 1

**结果:**
- Test F1: 0.9456 (+2.2%)
- Test AUROC: 0.9723 (+1.6%)
- Test FPR: 0.0234 (-2.2%)

**结论:** ✅ RoBERTa 显著优于 BERT，采用为新基线

---
```

---

## 🎯 检查清单

开始新实验前：

- [ ] 数据已准备并验证（无重叠）
- [ ] 配置文件已检查
- [ ] 实验名称有意义
- [ ] 环境变量已设置（如 `DATA_ROOT`）
- [ ] GPU 可用（服务器模式）

实验完成后：

- [ ] 检查所有结果文件已生成
- [ ] 查看训练曲线是否正常
- [ ] 分析混淆矩阵和 ROC 曲线
- [ ] 记录实验发现到 `EXPERIMENTS_LOG.md`
- [ ] 对比与之前实验的差异

---

## ❓ 常见问题

### Q: 可视化图表没有生成？

A: 安装可视化依赖：
```bash
pip install -e ".[viz]"
```

### Q: 如何加载保存的模型进行推理？

A:
```python
from src.systems.url_only_module import UrlOnlySystem

model = UrlOnlySystem.load_from_checkpoint(
    "experiments/bert_baseline_*/checkpoints/best-*.ckpt"
)
model.eval()
```

### Q: 如何删除旧实验？

A:
```bash
# 删除特定实验
rm -rf experiments/old_experiment_*

# 只保留最近 10 个
ls -t experiments/ | tail -n +11 | xargs -I {} rm -rf experiments/{}
```

### Q: 实验目录占用空间太大？

A: 检查点文件较大，可以：
1. 只保留最佳实验的检查点
2. 压缩归档旧实验
3. 删除 `lightning_logs/`（已复制到 `experiments/`）

---

## 🔗 相关文档

- [完整实验管理指南](EXPERIMENTS.md)
- [项目结构说明](ROOT_STRUCTURE.md)
- [依赖说明](DEPENDENCIES.md)

---

**开始您的第一个实验！** 🚀

```bash
python scripts/train.py --profile local --exp_name my_first_exp
```

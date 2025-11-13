# 实验管理指南

> **Last Updated:** 2025-10-21
> **版本:** 0.1.0

本文档说明如何使用项目的实验管理系统进行有组织的实验跟踪和结果保存。

---

## 📋 目录

- [实验目录结构](#实验目录结构)
- [运行实验](#运行实验)
- [查看实验结果](#查看实验结果)
- [实验管理最佳实践](#实验管理最佳实践)
- [自定义实验跟踪](#自定义实验跟踪)

---

## 📁 实验目录结构

每次运行训练脚本时，系统会自动创建以下结构的实验目录：

```
experiments/
├── url_mvp_20251021_143022/        # 实验名称_时间戳
│   ├── config.yaml                  # ✅ 实验配置（自动保存）
│   ├── SUMMARY.md                   # ✅ 实验总结（训练后生成）
│   ├── results/                     # 📊 实验结果
│   │   ├── metrics_final.json       # ✅ 最终指标（立即保存）
│   │   ├── training_curves.png      # ✅ 训练曲线（立即生成）
│   │   ├── confusion_matrix.png     # ✅ 混淆矩阵（立即生成）
│   │   ├── roc_curve.png            # ✅ ROC曲线（立即生成）
│   │   └── threshold_analysis.png   # ✅ 阈值分析（立即生成）
│   ├── logs/                        # 📝 日志文件
│   │   ├── train.log                # ✅ 训练日志（实时记录）
│   │   └── metrics_history.csv      # ✅ 指标历史（每epoch保存）
│   └── checkpoints/                 # 💾 模型检查点
│       └── best-epoch=X-val_auroc=Y.ckpt  # ✅ 最佳模型（训练后复制）
├── url_mvp_20251021_150033/        # 另一个实验
│   └── ...
└── comparison_exp_20251022_091234/ # 对比实验
    └── ...
```

**说明:**
- ✅ 表示训练/测试完成后立即自动保存
- 📊 结果文件包含 JSON 指标和可视化图表
- 📝 日志实时记录训练过程
- 💾 检查点从 `lightning_logs/` 复制而来

---

## 🚀 运行实验

### 基本用法

```bash
# 使用默认配置
python scripts/train.py

# 使用本地配置（CPU、小批量）
python scripts/train.py --profile local

# 使用服务器配置（GPU、大批量）
python scripts/train.py --profile server
```

### 指定实验名称

```bash
# 自定义实验名称
python scripts/train.py --profile server --exp_name bert_baseline

# 生成的目录: experiments/bert_baseline_20251021_143022/
```

### 禁用实验保存（快速测试）

```bash
# 不保存实验结果（用于调试）
python scripts/train.py --profile local --no_save
```

---

## 📊 自动保存的内容

### 1. **配置文件** (`config.yaml`)
- **保存时机:** 实验开始时
- **内容:** 完整的实验配置（合并后的配置）
- **用途:** 确保实验可复现

### 2. **指标文件** (`metrics_final.json`)
- **保存时机:** 测试完成后立即保存
- **内容:** 最终测试指标
  ```json
  {
    "experiment": "url_mvp_20251021_143022",
    "timestamp": "2025-10-21T14:35:42",
    "stage": "final",
    "metrics": {
      "test/loss": 0.1234,
      "test/f1": 0.9567,
      "test/auroc": 0.9823,
      "test/fpr": 0.0234
    }
  }
  ```

### 3. **训练曲线** (`training_curves.png`)
- **保存时机:** 训练完成后立即生成
- **内容:** 4个子图
  - Loss (train & val)
  - F1 Score (train & val)
  - AUROC (train & val)
  - FPR (train & val)

### 4. **混淆矩阵** (`confusion_matrix.png`)
- **保存时机:** 测试完成后立即生成
- **内容:**
  - 2x2 混淆矩阵热力图
  - 准确率、精确率、召回率、F1

### 5. **ROC 曲线** (`roc_curve.png`)
- **保存时机:** 测试完成后立即生成
- **内容:**
  - ROC 曲线
  - AUC 值
  - 随机分类器基线

### 6. **阈值分析** (`threshold_analysis.png`)
- **保存时机:** 测试完成后立即生成
- **内容:**
  - Precision/Recall/F1 vs Threshold
  - 最佳 F1 阈值标记

### 7. **训练日志** (`train.log`)
- **保存时机:** 训练过程中实时记录
- **内容:**
  - 每个 epoch 的指标
  - 训练开始/结束时间
  ```
  [2025-10-21 14:30:22] ============================================================
  [2025-10-21 14:30:22] 训练开始
  [2025-10-21 14:30:22] 模型: roberta-base
  [2025-10-21 14:30:22] 总轮数: 5
  [2025-10-21 14:30:22] ============================================================
  [2025-10-21 14:31:05] Epoch 0: train/loss=0.3456 val/loss=0.2890 val/f1=0.8234
  ...
  ```

### 8. **实验总结** (`SUMMARY.md`)
- **保存时机:** 测试完成后立即生成
- **内容:** Markdown 格式的实验总结
  ```markdown
  # 实验总结: url_mvp_20251021_143022

  **时间:** 2025-10-21 14:35:42

  ## 配置
  - **模型:** roberta-base
  - **最大长度:** 256
  - **批量大小:** 16
  - **学习率:** 2e-05
  - **训练轮数:** 5

  ## 结果
  - **final_test_loss:** 0.1234
  - **final_test_f1:** 0.9567
  - **final_test_auroc:** 0.9823
  - **final_test_fpr:** 0.0234
  - **total_epochs:** 5
  ```

### 9. **模型检查点** (`checkpoints/`)
- **保存时机:** 训练完成后从 `lightning_logs/` 复制
- **内容:** 最佳模型权重文件

---

## 🔍 查看实验结果

### 快速查看

```bash
# 查看最近的实验
ls -lt experiments/ | head -5

# 查看特定实验的总结
cat experiments/url_mvp_20251021_143022/SUMMARY.md

# 查看指标
cat experiments/url_mvp_20251021_143022/results/metrics_final.json | jq
```

### 图表查看

在文件管理器中打开 `experiments/实验名/results/` 目录，查看所有生成的图表：
- `training_curves.png` - 训练过程
- `confusion_matrix.png` - 分类性能
- `roc_curve.png` - 判别能力
- `threshold_analysis.png` - 阈值优化

### 加载检查点进行推理

```python
import torch
from src.systems.url_only_module import UrlOnlySystem

# 加载模型
checkpoint_path = "experiments/url_mvp_20251021_143022/checkpoints/best-epoch=3-val_auroc=0.982.ckpt"
model = UrlOnlySystem.load_from_checkpoint(checkpoint_path)
model.eval()

# 使用模型进行推理
# ...
```

---

## 📈 实验管理最佳实践

### 1. **使用有意义的实验名称**

```bash
# ❌ 不好的命名
python scripts/train.py --exp_name test1

# ✅ 好的命名
python scripts/train.py --exp_name bert_baseline_lr2e5
python scripts/train.py --exp_name roberta_dropout02_bs32
python scripts/train.py --exp_name ablation_no_html
```

### 2. **实验记录表格**

在项目根目录创建 `EXPERIMENTS_LOG.md`，记录所有实验：

```markdown
| 实验名 | 日期 | 模型 | 配置变化 | Test F1 | Test AUROC | 备注 |
|--------|------|------|----------|---------|------------|------|
| bert_baseline | 2025-10-21 | bert-base | 默认 | 0.9234 | 0.9567 | 基线 |
| roberta_baseline | 2025-10-21 | roberta-base | 默认 | 0.9456 | 0.9723 | 更优 |
| roberta_dropout02 | 2025-10-21 | roberta-base | dropout=0.2 | 0.9501 | 0.9789 | 最佳 |
```

### 3. **实验对比脚本**

创建 `scripts/compare_experiments.py` 来对比多个实验：

```python
import json
from pathlib import Path
import pandas as pd

def compare_experiments(exp_names):
    results = []
    for exp in exp_names:
        exp_dir = Path(f"experiments/{exp}")
        metrics_file = exp_dir / "results/metrics_final.json"

        if metrics_file.exists():
            with open(metrics_file) as f:
                data = json.load(f)
                results.append({
                    'experiment': exp,
                    **data['metrics']
                })

    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    return df

# 使用
compare_experiments([
    'bert_baseline_20251021_143022',
    'roberta_baseline_20251021_150033',
    'roberta_dropout02_20251021_153412'
])
```

### 4. **定期清理**

```bash
# 只保留最近 10 个实验
ls -t experiments/ | tail -n +11 | xargs -I {} rm -rf experiments/{}

# 或者压缩旧实验
tar -czf experiments_archive_$(date +%Y%m%d).tar.gz \
    $(ls -t experiments/ | tail -n +11)
```

---

## 🔧 自定义实验跟踪

### 在代码中使用 ExperimentTracker

```python
from src.utils.experiment_tracker import ExperimentTracker
from omegaconf import OmegaConf

# 创建跟踪器
cfg = OmegaConf.load("configs/default.yaml")
tracker = ExperimentTracker(cfg, exp_name="my_experiment")

# 记录日志
tracker.log_text("开始预处理数据")

# 保存自定义指标
custom_metrics = {
    "train_samples": 10000,
    "val_samples": 2000,
    "test_samples": 2000,
    "avg_url_length": 85.6
}
tracker.save_metrics(custom_metrics, stage="data_stats")

# 保存图表
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot([1, 2, 3], [4, 5, 6])
tracker.save_figure(fig, name="custom_plot")

# 保存总结
tracker.save_summary({
    "best_f1": 0.9567,
    "best_threshold": 0.48,
    "notes": "使用了新的数据增强策略"
})
```

### 添加自定义回调

```python
from pytorch_lightning.callbacks import Callback

class CustomMetricsCallback(Callback):
    def __init__(self, tracker):
        super().__init__()
        self.tracker = tracker

    def on_epoch_end(self, trainer, pl_module):
        # 保存每个 epoch 的额外指标
        custom_data = {
            "learning_rate": trainer.optimizers[0].param_groups[0]['lr'],
            "epoch": trainer.current_epoch
        }
        self.tracker.log_text(f"Epoch {custom_data['epoch']}: LR={custom_data['learning_rate']}")
```

---

## 📊 可视化工具使用

### 使用 ResultVisualizer

```python
from src.utils.visualizer import ResultVisualizer
from pathlib import Path
import numpy as np

# 1. 绘制训练曲线
metrics_csv = Path("lightning_logs/version_0/metrics.csv")
ResultVisualizer.plot_training_curves(
    metrics_csv,
    save_path="my_curves.png"
)

# 2. 绘制混淆矩阵
y_true = np.array([0, 1, 1, 0, 1, 0])
y_pred = np.array([0, 1, 0, 0, 1, 0])
ResultVisualizer.plot_confusion_matrix(
    y_true, y_pred,
    class_names=['良性', '钓鱼'],
    save_path="my_cm.png"
)

# 3. 绘制 ROC 曲线
y_prob = np.array([0.1, 0.9, 0.8, 0.2, 0.95, 0.15])
ResultVisualizer.plot_roc_curve(
    y_true, y_prob,
    save_path="my_roc.png"
)

# 4. 阈值分析
fig, best_th = ResultVisualizer.plot_threshold_analysis(
    y_true, y_prob,
    save_path="my_threshold.png"
)
print(f"最佳阈值: {best_th}")

# 5. 一次性生成所有图表
ResultVisualizer.create_all_plots(
    metrics_csv=metrics_csv,
    y_true=y_true,
    y_prob=y_prob,
    output_dir=Path("results/")
)
```

---

## ⚙️ 配置选项

### 实验跟踪配置

在 `configs/default.yaml` 中添加：

```yaml
experiment:
  save_results: true          # 是否保存实验结果
  base_dir: experiments       # 实验根目录
  save_checkpoints: true      # 是否复制检查点
  generate_plots: true        # 是否生成图表
```

---

## 🐛 故障排除

### 问题1: 可视化图表未生成

**原因:** 未安装 matplotlib/seaborn

**解决:**
```bash
pip install -e ".[viz]"
```

### 问题2: 实验目录未创建

**原因:** 使用了 `--no_save` 参数

**解决:**
```bash
# 移除 --no_save 参数
python scripts/train.py --profile local
```

### 问题3: 检查点未复制

**原因:** Lightning 日志目录路径不正确

**解决:** 检查 `trainer.log_dir` 是否存在

---

## 📝 实验checklist

训练新模型前的检查清单：

- [ ] 确定实验目标和假设
- [ ] 准备并验证数据集
- [ ] 选择合适的配置 profile
- [ ] 设置有意义的实验名称
- [ ] 记录预期结果
- [ ] 运行训练
- [ ] 检查生成的所有文件
- [ ] 分析结果并记录发现
- [ ] 更新 EXPERIMENTS_LOG.md
- [ ] （可选）将最佳模型保存到独立目录

---

**维护者:** UAAM-Phish Team
**更新频率:** 每次添加新功能时更新
**最后检查:** 2025-10-21

# URL-Only 基线实验

## 🎯 实验目标

建立字符级 BiLSTM URL 编码器基线，用于后续多模态融合对比。

---

## 📊 数据切分

基于 `data/processed/url_*.csv`：

| 数据集 | 样本数 | 正负比例 | 路径 |
|--------|--------|----------|------|
| 训练集 | ~470 | ~1:1 | `data/processed/url_train.csv` |
| 验证集 | ~101 | ~1:1 | `data/processed/url_val.csv` |
| 测试集 | ~101 | ~1:1 | `data/processed/url_test.csv` |

**说明：**
- 字段：`url_text`, `label`（0=legitimate, 1=phishing）
- 切分方式：随机划分（seed=42）
- 与论文 4.6.3 节对齐

---

## 🏗️ 模型架构

**URLEncoder (字符级 BiLSTM)**

```
输入: URL 字符序列 (max_len=256, vocab_size=128)
  ↓
Embedding(128, embedding_dim=128)
  ↓
Dropout(0.1)
  ↓
BiLSTM(hidden_dim=128, num_layers=2, bidirectional=True)
  ↓
Concat[forward_last, backward_last] → (batch, 256)
  ↓
Dropout(0.1)
  ↓
Linear(256, proj_dim=256)
  ↓
Classifier: Linear(256, 2) → [legitimate_prob, phishing_prob]
```

**参数配置：** `configs/model/url_encoder.yaml`

---

## 🚀 运行实验

### 训练

```bash
# 使用默认配置
make train-url
# 或
python scripts/train_hydra.py

# 使用本地配置（快速调试）
python scripts/train_hydra.py trainer=local

# 自定义超参数
python scripts/train_hydra.py train.lr=1e-3 train.bs=32 model.dropout=0.2
```

### 预测

```bash
# 单条 URL
python scripts/predict.py \
  --config-path configs --config-name default \
  --checkpoint experiments/url_only/checkpoints/url-only-best.ckpt \
  --url "http://example.com"
# 输出: [0.998, 0.002]  # [legit_prob, phish_prob]

# 批量预测
make predict-url
# 输出: pred_url_test.csv (列: idx, label, legit_prob, phish_prob)
```

### 测试

```bash
make test-url
# 或
pytest tests/test_url_dataset.py tests/test_url_encoder.py -v
```

---

## 📈 预期基线指标

基于论文 4.6.3 节和初步实验：

| 指标 | 预期范围 | 说明 |
|------|---------|------|
| **Accuracy** | 85-90% | 整体准确率 |
| **F1-Score** | 85-90% | 平衡精确率与召回率 |
| **AUROC** | 0.90-0.95 | 判别能力 |
| **val_loss** | 0.2-0.4 | 交叉熵损失 |

**注意：** 实际结果可能因数据分布、随机种子等因素有所波动。

---

## 📝 实验记录模板

| ID | 时间戳 | Config | Seed | Val Loss | Test Acc | Test F1 | AUROC | Notes | Artifact |
|----|--------|--------|------|----------|----------|---------|-------|-------|----------|
| EXP-001 | 2025-10-22 | default.yaml | 42 | 0.35 | 0.88 | 0.87 | 0.92 | 初始基线 | url-only-best.ckpt |

---

## 🔄 复现步骤

```bash
# 1. 检出代码
git checkout <commit-hash>

# 2. 安装依赖
pip install -r requirements.txt

# 3. 验证数据
python scripts/validate_data_schema.py

# 4. 训练
python scripts/train_hydra.py

# 5. 测试
make test-url
```

---

## 🧪 消融实验建议

1. **编码维度:** `proj_dim=128/256/512`
2. **LSTM层数:** `num_layers=1/2/3`
3. **Dropout比例:** `dropout=0.1/0.2/0.3`
4. **学习率:** `lr=1e-4/1e-3/5e-3`
5. **批量大小:** `batch_size=16/32/64`

---

## 🧪 S2 Consistency 实验

S2 阶段用于验证跨模态品牌一致性信号（C-Module）。两个推荐配置：

| 实验 | 用途 | 入口 |
| --- | --- | --- |
| Brand-OOD Consistency | 针对品牌迁移场景观测一致性崩溃 | `python scripts/train_hydra.py experiment=s2_brandood_consistency` |
| IID Consistency | 对照实验，验证在 IID 场景下合法站点 ACS 更高 | `python scripts/train_hydra.py experiment=s2_iid_consistency` |

特性：
- `modules.use_cmodule=true` / `modules.use_umodule=false`，只启用 C-Module。
- `metrics.consistency_thresh` 控制 `val/consistency/*` 与 `test/consistency/*` 日志。
- `predictions_test.csv` 会多出 `c_mean` 与 `brand_url/html/vis`，方便做后续统计。

生成分布图与报告：

```bash
# 默认扫描 workspace/runs 下最新的 s0_* / s2_* 目录
python scripts/plot_s2_distributions.py --runs_dir workspace/runs

# 自定义输出位置
python scripts/plot_s2_distributions.py --runs_dir workspace/runs \
  --figures-dir figures/s2 --results-dir results/s2
```

脚本会输出：
- `figures/s0_vis_similarity_hist.png`
- `figures/s2_consistency_hist.png`
- `results/consistency_report.json`（SUMMARY.md 会读取该文件，自动对比 OVL / KS / AUC）

---

## 📚 相关文档

- [数据 Schema](DATA_SCHEMA.md)
- [实验系统功能](EXPERIMENT_SYSTEM_FEATURES.md)
- [快速开始](../QUICKSTART.md)
- [架构说明](PROJECT_ARCHITECTURE_CN.md)

---

**维护者:** UAAM-Phish Team
**最后更新:** 2025-10-22

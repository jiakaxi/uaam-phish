# 腐败数据测试指南

本文档说明如何使用 IID 训练的 checkpoint 对腐败数据进行评估。

## 📋 概述

腐败数据测试用于评估模型在不同强度腐败数据上的鲁棒性。支持两种测试类型：

1. **URL 类型**：L/M/H 三个强度（适用于 URL/HTML/IMG 模态）
2. **IID 类型**：0.1/0.3/0.5 三个强度（适用于 IID 分割的腐败数据）

## 📁 数据准备

腐败数据已准备在以下位置：

```
workspace/data/corrupt/
├── url/                    # URL 腐败数据（L/M/H）
│   ├── test_corrupt_url_L.csv
│   ├── test_corrupt_url_M.csv
│   └── test_corrupt_url_H.csv
├── html/                   # HTML 腐败数据（L/M/H）
│   ├── test_corrupt_html_L.csv
│   ├── test_corrupt_html_M.csv
│   └── test_corrupt_html_H.csv
├── img/                    # IMG 腐败数据（L/M/H）
│   ├── test_corrupt_img_L.csv
│   ├── test_corrupt_img_M.csv
│   └── test_corrupt_img_H.csv
└── iid/                    # IID 腐败数据（0.1/0.3/0.5）
    ├── url/
    │   ├── test_corrupt_url_0.1.csv
    │   ├── test_corrupt_url_0.3.csv
    │   └── test_corrupt_url_0.5.csv
    ├── html/
    │   ├── test_corrupt_html_0.1.csv
    │   ├── test_corrupt_html_0.3.csv
    │   └── test_corrupt_html_0.5.csv
    └── img/
        ├── test_corrupt_img_0.1.csv
        ├── test_corrupt_img_0.3.csv
        └── test_corrupt_img_0.5.csv
```

## 🚀 使用方法

### 方法 1：批量运行测试（推荐）

**重要**：主腐败评测运行完整的 **L/M/H × 3 模态 = 9 个测试**，包括：
- URL 模态：L, M, H
- HTML 模态：L, M, H
- IMG 模态：L, M, H

#### 使用 Python 脚本（推荐）

```bash
# 主腐败评测（L/M/H × 3 模态 = 9 个测试）
python scripts/run_corrupt_tests.py \
  --experiment-dir experiments/s0_iid_earlyconcat_20251111_025612 \
  --test-type corrupt \
  --modalities url html img \
  --levels L M H

# IID 轻噪声测试（0.1/0.3/0.5 × 3 模态 = 9 个测试）
python scripts/run_corrupt_tests.py \
  --experiment-dir experiments/s0_iid_earlyconcat_20251111_025612 \
  --test-type iid \
  --modalities url html img \
  --levels 0.1 0.3 0.5
```

#### 使用 Bash 脚本

```bash
# 主腐败评测（L/M/H × 3 模态 = 9 个测试）
bash scripts/run_corrupt_tests.sh \
  experiments/s0_iid_earlyconcat_20251111_025612

# IID 轻噪声测试（0.1/0.3/0.5 × 3 模态 = 9 个测试）
bash scripts/run_corrupt_tests_iid.sh \
  experiments/s0_iid_earlyconcat_20251111_025612
```

### 方法 2：单个测试运行

使用 Hydra 运行单个测试：

```bash
# URL 模态，L 强度
python scripts/train_hydra.py \
  experiment=s0_iid_earlyconcat \
  trainer.max_epochs=0 \
  datamodule.test_csv=workspace/data/corrupt/url/test_corrupt_url_L.csv \
  run.name=corrupt_url_L

# HTML 模态，M 强度
python scripts/train_hydra.py \
  experiment=s0_iid_earlyconcat \
  trainer.max_epochs=0 \
  datamodule.test_csv=workspace/data/corrupt/html/test_corrupt_html_M.csv \
  run.name=corrupt_html_M

# IMG 模态，H 强度
python scripts/train_hydra.py \
  experiment=s0_iid_earlyconcat \
  trainer.max_epochs=0 \
  datamodule.test_csv=workspace/data/corrupt/img/test_corrupt_img_H.csv \
  run.name=corrupt_img_H
```

### 方法 3：收集结果并生成可视化

运行测试后，使用结果收集脚本生成指标和可视化：

```bash
# 主腐败评测（L/M/H）
python scripts/test_corrupt_data.py \
  --experiment-dir experiments/s0_iid_earlyconcat_20251111_025612 \
  --test-type corrupt \
  --modalities url html img \
  --levels L M H \
  --output-dir experiments/corrupt_eval_s0

# IID 轻噪声（0.1/0.3/0.5）
python scripts/test_corrupt_data.py \
  --experiment-dir experiments/s0_iid_earlyconcat_20251111_025612 \
  --test-type iid \
  --modalities url html img \
  --levels 0.1 0.3 0.5 \
  --output-dir experiments/corrupt_eval_iid_s0
```

## 📊 输出结果

结果收集脚本会生成以下文件：

```
experiments/corrupt_eval/
├── corrupt_metrics.csv          # 所有指标的 CSV 汇总
├── corrupt_metrics.json         # 所有指标的 JSON 汇总
├── auroc_vs_intensity.png       # AUROC vs 强度柱状图（按模态分组）
└── reliability_comparison.png   # 可靠性曲线对比（IID vs H）
```

### 支持的模态和强度

脚本完全支持：

- **三模态**：
  - `url`：URL 文本模态
  - `html`：HTML 内容模态
  - `img`：图像模态

- **三级强度**（主腐败评测）：
  - `L`：低强度腐败
  - `M`：中强度腐败
  - `H`：高强度腐败

- **三级强度**（IID 轻噪声）：
  - `0.1`：10% 腐败强度
  - `0.3`：30% 腐败强度
  - `0.5`：50% 腐败强度

### CSV 文件格式

脚本能够自动识别和处理符合以下命名格式的 CSV 文件：

- `test_corrupt_{modality}_{intensity}.csv`
  - 例如：`test_corrupt_url_L.csv`、`test_corrupt_html_M.csv`、`test_corrupt_img_H.csv`
  - 或：`test_corrupt_url_0.1.csv`、`test_corrupt_html_0.3.csv`、`test_corrupt_img_0.5.csv`

### 指标说明

- **AUROC**：ROC 曲线下面积（越高越好）
- **FPR@TPR95**：当 TPR 达到 95% 时的 FPR（越低越好）
- **ECE**：期望校准误差（越低越好）
- **Brier**：Brier 分数（越低越好）

## 📈 可视化说明

### 1. AUROC vs 强度图

按模态分组显示不同强度下的 AUROC 值，用于评估模型对腐败数据的鲁棒性。

### 2. 可靠性曲线对比

对比 IID 基线（正常数据）和最高强度腐败数据的可靠性曲线，用于评估校准性能的变化。

## 🔧 参数说明

### 参数说明（统一命名）

所有脚本使用统一的参数命名：

#### `run_corrupt_tests.py` 参数

- `--experiment-dir`：IID 训练目录（必需）
  - 指向包含 `checkpoints/best.ckpt` 的实验目录
  - 脚本会自动发现实验配置和 checkpoint

- `--modalities`：要测试的模态（默认：`url html img`）
  - 支持同时测试多个模态
  - 可选值：`url`、`html`、`img`

- `--levels`：腐败强度级别（默认：`L M H`）
  - 主腐败评测：`L M H`
  - IID 轻噪声：`0.1 0.3 0.5`

- `--test-type`：测试类型（默认：`corrupt`）
  - `corrupt`：主腐败评测（L/M/H）
  - `iid`：IID 轻噪声（0.1/0.3/0.5）

- `--output-dir`：输出目录（可选）
  - 默认：`experiments/corrupt_eval_<model_name>`
  - 建议按模型分文件夹

- `--seeds`：随机种子（可选）
  - 默认：从 `experiment-dir` 自动发现
  - 否则显式传递：`--seeds 42 43 44`

#### `test_corrupt_data.py` 参数

- `--experiment-dir`：IID 训练目录（必需）
  - 脚本会在此目录下搜索所有腐败数据的预测结果
  - 支持从实验名称、路径和 CSV 文件名自动推断模态和强度

- `--modalities`：要处理的模态（默认：`url html img`）
  - 支持同时处理多个模态
  - 可选值：`url`、`html`、`img`

- `--levels`：腐败强度级别（可选）
  - 默认：根据 `test-type` 自动确定
  - 主腐败评测：`L M H`
  - IID 轻噪声：`0.1 0.3 0.5`

- `--test-type`：测试类型（默认：`corrupt`）
  - `corrupt`：主腐败评测（L/M/H）
  - `iid`：IID 轻噪声（0.1/0.3/0.5）

- `--output-dir`：输出目录（可选）
  - 默认：`experiments/corrupt_eval_<model_name>`
  - 建议按模型分文件夹

- `--collect-only`：只收集结果，不生成可视化
  - 适用于只需要指标数据，不需要图表的场景

## 📝 注意事项

1. **Checkpoint 路径**：确保 IID 训练的 checkpoint 存在且可访问
2. **预测文件**：脚本会自动搜索包含 "corrupt" 的预测文件
3. **基线数据**：可靠性曲线对比需要 IID 基线的预测结果（`predictions_test.csv`）
4. **测试顺序**：建议先运行所有测试，再收集结果生成可视化

## 🐛 故障排除

### 问题：未找到预测文件

**原因**：测试尚未运行或预测文件路径不正确

**解决**：
1. 确认已运行测试（使用 Hydra 或批量脚本）
2. 检查实验目录中是否存在 `artifacts/predictions*.csv` 文件
3. 确认预测文件路径中包含 "corrupt" 关键字

### 问题：缺少基线数据

**原因**：IID 基线的预测结果不存在

**解决**：
1. 运行 IID 测试获取基线预测结果
2. 或手动指定基线预测文件路径（修改脚本）

### 问题：可视化生成失败

**原因**：matplotlib/seaborn 未安装或数据不足

**解决**：
1. 安装依赖：`pip install matplotlib seaborn`
2. 确认有足够的测试结果数据
3. 使用 `--collect-only` 参数跳过可视化

## 📚 相关文件

- `scripts/test_corrupt_data.py`：结果收集和可视化脚本
- `scripts/run_corrupt_tests.sh`：URL 类型批量测试脚本
- `scripts/run_corrupt_tests_iid.sh`：IID 类型批量测试脚本
- `src/utils/metrics_v2.py`：指标计算函数
- `src/utils/visualizer.py`：可视化工具

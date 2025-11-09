# 工作区指南

**更新日期**: 2025-11-08
**版本**: 1.0

---

## 📋 目录

- [工作区概述](#工作区概述)
- [原来的工作区](#原来的工作区)
- [新的工作区（S0实验）](#新的工作区s0实验)
- [如何运行实验](#如何运行实验)
- [如何复现实验](#如何复现实验)
- [如何创建新工作区](#如何创建新工作区)

---

## 🔍 工作区概述

项目中有两种工作区组织方式：

1. **原来的工作区**：分散式，使用多个目录（`experiments/`, `data/processed/`, `outputs/`）
2. **新的工作区（S0实验）**：集中式，使用单一 `workspace/` 目录

---

## 📁 原来的工作区

### 目录结构

```
项目根目录/
├── experiments/                    # 实验结果（自动创建）
│   ├── <实验名>_<时间戳>/
│   │   ├── config.yaml            # 实验配置
│   │   ├── results/               # 结果文件
│   │   │   ├── metrics_*.json    # 指标
│   │   │   ├── roc_*.png         # ROC曲线
│   │   │   └── calib_*.png       # 校准图
│   │   ├── logs/                  # 日志
│   │   └── checkpoints/           # 模型检查点
│   └── ...
├── data/processed/                 # 处理后的数据
│   ├── master_v2.csv              # 主数据集
│   ├── url_train_v2.csv           # 训练集
│   ├── url_val_v2.csv             # 验证集
│   ├── url_test_v2.csv            # 测试集
│   └── screenshots/               # 图像数据
├── outputs/                        # Hydra输出（自动创建）
│   └── 2025-11-08/
│       └── 10-30-45/
│           ├── .hydra/            # Hydra配置
│           └── train.log          # 训练日志
└── lightning_logs/                 # PyTorch Lightning日志
    └── version_X/
        ├── hparams.yaml
        ├── metrics.csv
        └── checkpoints/
```

### 配置文件

**`configs/default.yaml`**:
```yaml
outputs:
  dir_root: experiments/          # 实验结果目录
data:
  csv_path: data/processed/master_v2.csv
  train_csv: data/processed/url_train_v2.csv
  val_csv: data/processed/url_val_v2.csv
  test_csv: data/processed/url_test_v2.csv
```

**`configs/experiment/multimodal_baseline.yaml`**:
```yaml
paths:
  output_dir: "${hydra:runtime.output_dir}"  # 使用Hydra默认输出
datamodule:
  master_csv: "data/processed/master_v2.csv"
  image_dir: "data/processed/screenshots"
```

### 运行方式

```bash
# 使用Hydra运行实验
python scripts/train_hydra.py experiment=multimodal_baseline

# 结果保存在:
# - outputs/2025-11-08/10-30-45/  (Hydra输出)
# - experiments/<实验名>_<时间戳>/  (实验结果)
# - lightning_logs/version_X/      (Lightning日志)
```

---

## 🆕 新的工作区（S0实验）

### 目录结构

```
项目根目录/
└── workspace/                      # 新工作区根目录
    ├── data/                       # 数据目录
    │   ├── splits/                 # 数据分割
    │   │   ├── iid/                # IID分割
    │   │   │   ├── train.csv
    │   │   │   ├── val.csv
    │   │   │   └── test.csv
    │   │   └── brandood/           # Brand-OOD分割
    │   │       ├── train.csv
    │   │       ├── val.csv
    │   │       ├── test_id.csv
    │   │       ├── test_ood.csv
    │   │       └── brand_sets.json
    │   └── corrupt/                # 腐败数据
    │       ├── html/               # HTML腐败
    │       │   ├── L/html/
    │       │   ├── M/html/
    │       │   └── H/html/
    │       ├── img/                # 图像腐败
    │       │   ├── L/shot/
    │       │   ├── M/shot/
    │       │   └── H/shot/
    │       └── url/                # URL腐败（CSV文件）
    │           ├── test_corrupt_html_L.csv
    │           ├── test_corrupt_html_M.csv
    │           └── test_corrupt_html_H.csv
    ├── runs/                       # 实验运行结果
    │   ├── s0_iid_earlyconcat/
    │   │   ├── seed_42/
    │   │   │   ├── artifacts/      # 工件
    │   │   │   │   ├── predictions_test.csv
    │   │   │   │   ├── roc_random.png
    │   │   │   │   └── calib_random.png
    │   │   │   ├── checkpoints/    # 模型检查点
    │   │   │   ├── eval_summary.json
    │   │   │   └── ...
    │   │   └── seed_43/
    │   └── s0_brandood_lateavg/
    ├── tables/                     # 汇总表格
    │   ├── s0_eval_summary.csv     # 评估汇总
    │   └── s0_eval_all_runs.csv    # 所有运行结果
    ├── figs/                       # 图表
    │   └── s0_auroc.png            # AUROC对比图
    └── reports/                    # 质量报告
        └── quality_report.json     # 质量检查报告
```

### 配置文件

**S0实验配置** (`configs/experiment/s0_iid_earlyconcat.yaml`):
```yaml
datamodule:
  train_csv: workspace/data/splits/iid/train.csv
  val_csv: workspace/data/splits/iid/val.csv
  test_csv: workspace/data/splits/iid/test.csv
  image_dir: data/processed/screenshots
  corrupt_root: workspace/data/corrupt

paths:
  output_dir: workspace/runs/${run.name}/seed_${run.seed}
```

### 运行方式

```bash
# 1. 准备数据分割
python tools/split_iid.py --in data/processed/master_v2.csv --out workspace/data/splits/iid --seed 42
python tools/split_brandood.py --in data/processed/master_v2.csv --out workspace/data/splits/brandood --seed 42 --top_k 20

# 2. 生成腐败数据（可选）
python tools/corrupt_html.py --in workspace/data/splits/iid/test.csv --out workspace/data/corrupt/html
python tools/corrupt_img.py --in workspace/data/splits/iid/test.csv --out workspace/data/corrupt/img
python tools/corrupt_url.py --in workspace/data/splits/iid/test.csv --out workspace/data/corrupt/url

# 3. 运行S0实验
python scripts/run_s0_experiments.py --scenario iid --models s0_earlyconcat s0_lateavg --seeds 42 43 44

# 4. 评估结果
python scripts/evaluate_s0.py --runs_dir workspace/runs

# 5. 汇总结果
python scripts/summarize_s0_results.py --runs_dir workspace/runs
```

---

## 🚀 如何运行实验

### 方法1: 运行原来的实验（使用experiments/工作区）

```bash
# 1. 确保数据准备完成
# 数据应该在 data/processed/ 目录下

# 2. 运行实验
python scripts/train_hydra.py experiment=multimodal_baseline

# 3. 查看结果
# - experiments/<实验名>_<时间戳>/results/
# - outputs/2025-11-08/10-30-45/
```

### 方法2: 运行S0实验（使用workspace/工作区）

```bash
# 1. 创建数据分割
python tools/split_iid.py --in data/processed/master_v2.csv --out workspace/data/splits/iid --seed 42

# 2. 运行实验
python scripts/run_s0_experiments.py --scenario iid --models s0_earlyconcat --seeds 42

# 3. 查看结果
# - workspace/runs/s0_iid_earlyconcat/seed_42/
```

### 方法3: 直接使用Hydra运行S0实验

```bash
# 运行单个实验
python scripts/train_hydra.py experiment=s0_iid_earlyconcat run.seed=42

# 结果保存在: workspace/runs/s0_iid_earlyconcat/seed_42/
```

---

## 🔄 如何复现实验

### 复现原来的实验

1. **检查配置文件**:
   ```bash
   # 查看实验配置
   cat experiments/<实验名>_<时间戳>/config.yaml
   ```

2. **恢复环境**:
   ```bash
   # 安装依赖（使用固定版本）
   pip install -r requirements.txt
   ```

3. **运行相同配置**:
   ```bash
   # 使用保存的配置运行
   python scripts/train_hydra.py experiment=<实验名> run.seed=<种子>
   ```

### 复现S0实验

1. **准备数据分割**:
   ```bash
   # 使用相同的种子和参数
   python tools/split_iid.py --in data/processed/master_v2.csv --out workspace/data/splits/iid --seed 42
   ```

2. **运行实验**:
   ```bash
   # 使用相同的配置和种子
   python scripts/run_s0_experiments.py --scenario iid --models s0_earlyconcat --seeds 42
   ```

3. **验证结果**:
   ```bash
   # 检查质量
   python scripts/validate_s0_quality.py --splits_root workspace/data/splits --runs_dir workspace/runs
   ```

---

## 🆕 如何创建新工作区

### 方法1: 创建新的S0工作区

```bash
# 1. 创建workspace目录结构
mkdir -p workspace/{data/splits,data/corrupt,runs,tables,figs,reports}

# 2. 创建数据分割
python tools/split_iid.py --in data/processed/master_v2.csv --out workspace/data/splits/iid --seed 42
python tools/split_brandood.py --in data/processed/master_v2.csv --out workspace/data/splits/brandood --seed 42 --top_k 20

# 3. 生成腐败数据（可选）
python tools/corrupt_html.py --in workspace/data/splits/iid/test.csv --out workspace/data/corrupt/html
python tools/corrupt_img.py --in workspace/data/splits/iid/test.csv --out workspace/data/corrupt/img
python tools/corrupt_url.py --in workspace/data/splits/iid/test.csv --out workspace/data/corrupt/url
```

### 方法2: 创建自定义工作区

1. **创建目录结构**:
   ```bash
   mkdir -p my_workspace/{data,runs,tables,figs,reports}
   ```

2. **修改配置文件**:
   ```yaml
   # configs/experiment/my_experiment.yaml
   datamodule:
     train_csv: my_workspace/data/train.csv
     val_csv: my_workspace/data/val.csv
     test_csv: my_workspace/data/test.csv

   paths:
     output_dir: my_workspace/runs/${run.name}/seed_${run.seed}
   ```

3. **运行实验**:
   ```bash
   python scripts/train_hydra.py experiment=my_experiment
   ```

### 方法3: 使用环境变量指定工作区

1. **设置环境变量**:
   ```bash
   # Linux/Mac
   export WORKSPACE_ROOT=/path/to/my_workspace

   # Windows
   set WORKSPACE_ROOT=D:\path\to\my_workspace
   ```

2. **修改脚本使用环境变量**:
   ```python
   # 在脚本中读取
   import os
   workspace_root = os.getenv("WORKSPACE_ROOT", "workspace")
   ```

---

## 📊 工作区对比

| 特性 | 原来的工作区 | 新的工作区（S0） |
|------|-------------|-----------------|
| **根目录** | 分散（experiments/, data/, outputs/） | 集中（workspace/） |
| **数据分割** | `data/processed/url_*_v2.csv` | `workspace/data/splits/` |
| **实验结果** | `experiments/<实验名>_<时间戳>/` | `workspace/runs/<模型>/seed_<种子>/` |
| **腐败数据** | 不支持 | `workspace/data/corrupt/` |
| **汇总表格** | 手动收集 | `workspace/tables/` |
| **质量报告** | 无 | `workspace/reports/` |
| **适用场景** | 一般实验 | S0基线实验 |

---

## 🛠️ 工具脚本

### 数据准备工具

```bash
# 创建IID分割
python tools/split_iid.py --in <输入CSV> --out <输出目录> --seed <种子>

# 创建Brand-OOD分割
python tools/split_brandood.py --in <输入CSV> --out <输出目录> --seed <种子> --top_k <品牌数>

# 生成HTML腐败数据
python tools/corrupt_html.py --in <输入CSV> --out <输出目录> --levels L M H

# 生成图像腐败数据
python tools/corrupt_img.py --in <输入CSV> --out <输出目录> --levels L M H

# 生成URL腐败数据
python tools/corrupt_url.py --in <输入CSV> --out <输出目录> --levels L M H
```

### 实验运行工具

```bash
# 运行S0实验
python scripts/run_s0_experiments.py --scenario <iid|brandood> --models <模型列表> --seeds <种子列表>

# 评估实验结果
python scripts/evaluate_s0.py --runs_dir <运行目录> --out_csv <输出CSV>

# 汇总结果
python scripts/summarize_s0_results.py --runs_dir <运行目录> --out_tables <表格目录> --out_figs <图表目录>

# 质量检查
python scripts/validate_s0_quality.py --splits_root <分割目录> --corrupt_root <腐败目录> --runs_dir <运行目录>
```

---

## ⚠️ 注意事项

### 1. 工作区隔离

- **原来的工作区**和**新的工作区**是独立的，不会相互干扰
- 可以同时使用两种工作区进行不同的实验
- 建议为不同的实验使用不同的工作区目录

### 2. 数据路径

- **原来的工作区**：数据在 `data/processed/`
- **新的工作区**：数据分割在 `workspace/data/splits/`，原始数据仍在 `data/processed/`
- 图像数据共享：两个工作区都使用 `data/processed/screenshots/`

### 3. 结果组织

- **原来的工作区**：按实验名称和时间戳组织
- **新的工作区**：按模型名称和种子组织
- 新的工作区更适合批量实验和结果对比

### 4. 版本控制

- `workspace/` 目录应该在 `.gitignore` 中（实验结果不应该提交）
- 只有配置文件和脚本应该提交到版本控制

---

## 📝 示例：完整工作流

### S0实验完整流程

```bash
# 1. 准备数据分割
python tools/split_iid.py --in data/processed/master_v2.csv --out workspace/data/splits/iid --seed 42
python tools/split_brandood.py --in data/processed/master_v2.csv --out workspace/data/splits/brandood --seed 42 --top_k 20

# 2. 生成腐败数据
python tools/corrupt_html.py --in workspace/data/splits/iid/test.csv --out workspace/data/corrupt/html --levels L M H
python tools/corrupt_img.py --in workspace/data/splits/iid/test.csv --out workspace/data/corrupt/img --levels L M H
python tools/corrupt_url.py --in workspace/data/splits/iid/test.csv --out workspace/data/corrupt/url --levels L M H

# 3. 质量检查
python scripts/validate_s0_quality.py --splits_root workspace/data/splits --corrupt_root workspace/data/corrupt

# 4. 运行实验
python scripts/run_s0_experiments.py --scenario iid --models s0_earlyconcat s0_lateavg --seeds 42 43 44
python scripts/run_s0_experiments.py --scenario brandood --models s0_earlyconcat s0_lateavg --seeds 42 43 44

# 5. 评估结果
python scripts/evaluate_s0.py --runs_dir workspace/runs

# 6. 汇总结果
python scripts/summarize_s0_results.py --runs_dir workspace/runs --out_tables workspace/tables --out_figs workspace/figs

# 7. 查看结果
# - workspace/tables/s0_eval_summary.csv
# - workspace/figs/s0_auroc.png
```

---

## 🔗 相关文档

- [实验管理指南](docs/EXPERIMENTS.md)
- [数据架构说明](docs/DATA_SCHEMA.md)
- [项目架构说明](docs/PROJECT_ARCHITECTURE_CN.md)
- [S0变更报告](S0_CHANGES_REPORT.md)

---

**最后更新**: 2025-11-08
**维护者**: 项目团队

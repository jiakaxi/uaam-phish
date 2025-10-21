# UAAM-Phish — URL-only MVP (Lightning)

这是一个最小化的第一个里程碑：基于BERT的URL分类器，使用PyTorch Lightning进行端到端训练。

---

## 🚀 快速开始

**新用户？** 查看 **[快速开始指南](QUICKSTART.md)** 5 分钟快速设置！

**详细安装？** 查看 **[完整安装指南](INSTALL.md)** 了解虚拟环境、离线模式、故障排除等。

---

## 1) 安装

### 快速安装（推荐使用 Make）
```bash
# 创建虚拟环境并安装所有依赖
python -m venv .venv
.venv\Scripts\Activate.ps1  # Windows
# source .venv/bin/activate  # Linux/macOS

make init                    # 安装所有依赖
make dvc-init               # 初始化数据管道
make install-hooks          # 安装 Git hooks（可选）
```

### 核心依赖
- PyTorch >= 2.2 + PyTorch Lightning >= 2.3
- Transformers >= 4.41
- Pandas, NumPy, scikit-learn
- OmegaConf >= 2.3

📖 **详细安装说明**: 请查看 [INSTALL.md](INSTALL.md) 或 [QUICKSTART.md](QUICKSTART.md)

## 2) 准备数据

### 使用 DVC 管道（推荐）
```bash
make dvc-init
dvc repro              # 运行数据预处理管道
make validate-data     # 验证数据schema
```

### 数据Schema要求
所有CSV文件必须包含：
- **必需列**: `url_text` (字符串), `label` (0或1)
- **可选列**: `id`, `domain`, `source`, `split`, `timestamp`

示例格式：
```csv
url_text,label
http://example.com/login,0
http://paypal.secure-update.example.cn/verify,1
```

📖 **详细说明**: 请查看 [数据Schema规范](docs/DATA_SCHEMA.md)

## 3) 配置
- 主配置：`configs/default.yaml`
- 硬件/数据配置文件：
  - 本地小数据集：`configs/profiles/local.yaml`
  - 服务器大数据集：`configs/profiles/server.yaml`

您可以通过设置环境变量 `DATA_ROOT` 来切换数据根目录而无需编辑配置：
```bash
export DATA_ROOT=/path/to/processed
```

## 4) 训练和测试
### 本地（小子集，单GPU）
```bash
export DATA_ROOT=./data/processed
python scripts/train.py --profile local
```

### 服务器（完整数据集，GPU / 多GPU）
```bash
export DATA_ROOT=/data/uaam_phish/processed
python scripts/train.py --profile server
```
（对于多GPU，在 `configs/profiles/server.yaml` 中设置 `devices` 和 `strategy: ddp`。）

指标（loss, F1, AUROC, FPR）将按epoch记录。

## 5) 实验管理

项目集成了完整的实验跟踪系统，每次训练后自动保存结果：

### 运行实验并保存结果
```bash
# 使用自定义实验名称
python scripts/train.py --profile server --exp_name roberta_baseline

# 生成的目录结构
experiments/roberta_baseline_20251021_143022/
├── config.yaml                 # 实验配置
├── SUMMARY.md                  # 实验总结
├── results/
│   ├── metrics_final.json      # ✅ 最终指标
│   ├── training_curves.png     # ✅ 训练曲线
│   ├── confusion_matrix.png    # ✅ 混淆矩阵
│   ├── roc_curve.png           # ✅ ROC曲线
│   └── threshold_analysis.png  # ✅ 阈值分析
├── logs/
│   └── train.log               # 训练日志
└── checkpoints/
    └── best-*.ckpt             # 最佳模型
```

### 对比实验结果
```bash
# 对比最近的 5 个实验
python scripts/compare_experiments.py --latest 5

# 对比特定实验
python scripts/compare_experiments.py --exp_names exp1 exp2 exp3

# 导出对比结果
python scripts/compare_experiments.py --latest 10 --output results.csv

# 查找最佳实验
python scripts/compare_experiments.py --find_best --metric auroc
```

详细说明请查看 [实验管理指南](docs/EXPERIMENTS.md)。

## 6) 下一步
- 在服务器上增加 `train.epochs` 和批量大小。
- 调整配置文件中的 `sample_fraction` 以便在本地更快迭代。
- MVP稳定后，集成HTML图和截图编码器，然后是UAAM。

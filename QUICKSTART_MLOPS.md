# 🚀 快速开始指南 - MLOps 版本

> **更新日期:** 2025-10-22
> **版本:** 2.0 (MLOps Upgrade)

---

## 📦 安装

### 1. 克隆项目

```bash
git clone https://github.com/your-username/uaam-phish.git
cd uaam-phish
```

### 2. 创建虚拟环境

```bash
# 创建环境
python -m venv .venv

# 激活环境
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Linux/Mac:
source .venv/bin/activate
```

### 3. 安装依赖

```bash
# 安装项目（可编辑模式）
pip install -e .

# 安装 MLOps 工具
pip install hydra-core wandb pre-commit

# 安装 pre-commit hooks
pre-commit install
```

---

## 🎯 快速训练

### 本地开发（CPU）

```bash
# 使用 Hydra 配置
python scripts/train_hydra.py trainer=local

# 查看结果
ls outputs/
```

### 服务器训练（GPU）

```bash
# 登录 WandB（首次）
wandb login

# 训练并跟踪实验
export WANDB_PROJECT=uaam-phish
python scripts/train_hydra.py trainer=server logger=wandb
```

---

## 🔧 常用命令

### 配置覆盖

```bash
# 修改学习率
python scripts/train_hydra.py train.lr=2e-5

# 修改批次大小和dropout
python scripts/train_hydra.py train.bs=64 model.dropout=0.3

# 使用不同模型
python scripts/train_hydra.py model.pretrained_name=bert-base-uncased
```

### 超参数搜索

```bash
# 网格搜索
python scripts/train_hydra.py -m \\
  train.lr=1e-5,2e-5,5e-5 \\
  model.dropout=0.1,0.2,0.3

# 共9次运行（3 lr × 3 dropout）
```

### 实验管理

```bash
# 使用实验配置
python scripts/train_hydra.py experiment=url_baseline

# 自定义实验名称
python scripts/train_hydra.py run.name=my_experiment logger=wandb
```

---

## 🧪 测试

```bash
# 运行所有测试
pytest tests/ -v

# 运行特定测试
pytest tests/test_data.py -v

# 查看覆盖率
pytest tests/ --cov=src --cov-report=term
```

---

## 📊 查看结果

### WandB Dashboard

1. 访问 https://wandb.ai
2. 查看项目实验
3. 对比多个运行

### 本地结果

```bash
# Hydra 输出目录
ls outputs/2025-10-22/18-45-00/

# 传统实验目录
ls experiments/
```

---

## 💡 常见任务

### 任务 1: 快速实验

```bash
# 小数据集快速验证
python scripts/train_hydra.py \\
  trainer=local \\
  data.sample_fraction=0.1 \\
  train.epochs=2
```

### 任务 2: 正式训练

```bash
# 完整数据集，GPU训练
python scripts/train_hydra.py \\
  trainer=server \\
  logger=wandb \\
  run.name=roberta_baseline_v1
```

### 任务 3: 超参数调优

```bash
# 搜索最佳学习率和dropout
python scripts/train_hydra.py -m \\
  trainer=server \\
  logger=wandb \\
  train.lr=1e-5,2e-5,5e-5 \\
  model.dropout=0.1,0.2,0.3
```

---

## 📚 文档索引

- **总体架构**: [README.md](README.md)
- **MLOps改进**: [MLOPS_IMPROVEMENTS_2025-10-22.md](docs/MLOPS_IMPROVEMENTS_2025-10-22.md)
- **WandB指南**: [WANDB_GUIDE.md](docs/WANDB_GUIDE.md)
- **CI/CD指南**: [CI_CD_GUIDE.md](docs/CI_CD_GUIDE.md)
- **项目结构**: [ROOT_STRUCTURE.md](docs/ROOT_STRUCTURE.md)

---

## ⚠️ 故障排除

### 问题 1: Hydra 找不到配置

```bash
# 确保在项目根目录
cd /path/to/uaam-phish

# 检查配置文件
ls configs/config.yaml
```

### 问题 2: WandB 登录失败

```bash
# 重新登录
wandb login --relogin

# 或使用 API key
export WANDB_API_KEY=64e15c91404e5023801580b0d943af3ebef4a033
```

### 问题 3: Pre-commit 检查失败

```bash
# 自动修复
ruff check --fix .
black .

# 重新提交
git add .
git commit -m "fix: 修复代码格式"
```

---

## 🎓 下一步

1. ✅ **阅读文档**: 查看 [MLOPS_IMPROVEMENTS_2025-10-22.md](docs/MLOPS_IMPROVEMENTS_2025-10-22.md)
2. ✅ **运行示例**: 尝试上述快速训练命令
3. ✅ **查看 WandB**: 访问你的实验Dashboard
4. ✅ **学习 Hydra**: 了解配置覆盖和组合

---

**Happy Training! 🚀**

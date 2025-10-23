# 🚀 UAAM-Phish 快速开始

> 5 分钟快速设置和运行指南

## ⚡ 最快设置（Windows PowerShell）

```powershell
# 1. 创建环境并安装依赖
python -m venv .venv
.venv\Scripts\Activate.ps1
make init

# 2. 初始化数据管道
make dvc-init
dvc repro

# 3. 验证数据schema
make validate-data

# 4. 安装 Git Hooks（可选）
make install-hooks

# 5. 运行验证
make lint
make test
make train
```

## 📋 命令说明

| 命令 | 用途 | 必需 |
|------|------|------|
| `make init` | 安装所有 Python 依赖 | ✅ |
| `make dvc-init` | 初始化 DVC 数据版本控制 | ✅ |
| `dvc repro` | 运行数据预处理管道 | ✅ |
| `make validate-data` | 验证数据schema完整性 | ✅ |
| `make install-hooks` | 安装代码质量检查 hooks | ⭕ |
| `make lint` | 检查代码风格 | ⭕ |
| `make test` | 运行测试套件 | ⭕ |
| `make train` | 开始训练模型 | ⭕ |

## 🌐 离线模式（可选）

如果需要在无网络环境下使用：

```powershell
# 1. 下载模型（需要网络，仅一次）
pip install huggingface-hub
huggingface-cli download roberta-base --local-dir models/roberta-base

# 2. 设置环境变量
$env:HF_CACHE_DIR = "$PWD/models/roberta-base"
$env:HF_LOCAL_ONLY = "1"

# 3. 正常训练
make train
```

**永久设置环境变量**（推荐）：
1. 右键"此电脑" → 属性 → 高级系统设置 → 环境变量
2. 添加用户变量：
   - `HF_CACHE_DIR` = `D:\uaam-phish\models\roberta-base`
   - `HF_LOCAL_ONLY` = `1`

## 📊 日常使用

每次工作时：

```powershell
# 进入项目并激活环境
cd D:\uaam-phish
.venv\Scripts\Activate.ps1

# 训练模型
make train

# 或运行自定义配置
python scripts/train.py --profile local

# 完成后退出
deactivate
```

## 🔧 常见任务

### 训练模型
```bash
make train
```

### 仅评估（不训练）
```bash
make eval
```

### 运行测试
```bash
make test
```

### 代码检查
```bash
make lint
```

### 数据更新
```bash
# 重新生成训练/验证/测试集
dvc repro

# 或手动运行
python scripts/build_master_and_splits.py \
  --benign data/raw/dataset \
  --phish data/raw/fish_dataset \
  --outdir data/processed
```

---

## 🚀 快速训练命令

### 方式1: 使用PowerShell脚本（推荐）
```powershell
.\start_training.ps1
```

### 方式2: 直接命令行
```powershell
python scripts/train_hydra.py logger=wandb trainer=default data=url_only model=url_encoder
```

### 方式3: 使用默认配置
```powershell
python scripts/train_hydra.py
```

---

## 📊 查看训练进度

### WandB Dashboard
访问: https://wandb.ai 查看实时图表

### 本地日志
```powershell
# 实时查看最新日志
Get-ChildItem outputs -Recurse -Filter "*.log" |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1 |
  ForEach-Object { Get-Content $_.FullName -Wait }
```

---

## 🎯 MLOps 协议快速参考

### 一行命令启动
```bash
# Random 协议（默认）
python scripts/train_hydra.py

# Temporal 协议
python scripts/train_hydra.py protocol=temporal

# Brand-OOD 协议
python scripts/train_hydra.py protocol=brand_ood
```

### 三种协议对比
| 协议 | 用途 | 要求 | 特点 |
|------|------|------|------|
| **random** | 基线 | 无 | 分层随机，始终可用 |
| **temporal** | 时序预测 | timestamp列 | 时间顺序，left-closed |
| **brand_ood** | 域泛化 | brand列，≥3品牌 | 品牌不相交 |

---

## 📚 文档管理

### 自动追加文档（已集成）
```bash
# 启用自动追加
python scripts/train_hydra.py logging.auto_append_docs=true

# 使用协议 + 自动追加
python scripts/train_hydra.py protocol=temporal logging.auto_append_docs=true
```

**效果**：训练结束后，实验结果会自动追加到 `FINAL_SUMMARY_CN.md`

### 手动追加文档
```python
from src.utils.documentation import DocumentationAppender

doc = DocumentationAppender()

# 追加到总结文档
doc.append_to_summary(
    feature_name="新功能名称",
    summary="功能描述",
    deliverables=["交付物1", "交付物2"],
    features=["✅ 功能A", "✅ 功能B"],
)
```

## 📁 重要目录

```
uaam-phish/
├── configs/          # 配置文件
│   ├── default.yaml  # 默认配置
│   ├── train.yaml    # 训练参数
│   └── profiles/     # 环境配置（local/server）
├── data/
│   ├── raw/          # 原始数据
│   └── processed/    # 预处理后的数据（CSV）
├── scripts/
│   └── train.py      # 训练脚本
├── src/
│   ├── systems/      # Lightning 模块
│   ├── models/       # 模型定义
│   ├── datamodules/  # 数据加载
│   └── utils/        # 工具函数
├── experiments/      # 实验结果（自动生成）
└── lightning_logs/   # 训练日志（自动生成）
```

## 🐛 故障排除

### 问题：PowerShell 无法运行脚本
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 问题：找不到模型
```bash
# 确保设置了环境变量或下载了模型
huggingface-cli download roberta-base --local-dir models/roberta-base
```

### 问题：CUDA 不可用
```bash
# 检查 PyTorch CUDA 支持
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# 如需安装 GPU 版本
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### 问题：测试失败
```bash
# 查看详细测试输出
pytest -v

# 运行特定测试
pytest tests/test_data.py -v
```

## 📚 更多信息

- **详细安装指南**: [INSTALL.md](INSTALL.md)
- **项目结构**: [docs/ROOT_STRUCTURE.md](docs/ROOT_STRUCTURE.md)
- **实验系统**: [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)
- **依赖说明**: [docs/DEPENDENCIES.md](docs/DEPENDENCIES.md)

## ✅ 验证清单

安装完成后，确认：

- [ ] 虚拟环境已激活（命令行显示 `(.venv)`）
- [ ] `make lint` 无错误
- [ ] `make test` 全部通过
- [ ] `make train` 成功运行
- [ ] 生成了 `lightning_logs/` 和 `experiments/` 目录
- [ ] 可以看到训练进度条和指标

---

**需要帮助？** 查看 [INSTALL.md](INSTALL.md) 了解更多详情。

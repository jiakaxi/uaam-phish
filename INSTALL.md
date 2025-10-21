# UAAM-Phish 安装指南

> **推荐方式：使用虚拟环境**

## 🔧 方法 1: 使用 venv（推荐，Python 内置）

### 步骤 1: 创建虚拟环境
```bash
# 在项目根目录执行
cd D:\uaam-phish

# 创建虚拟环境（会在项目下创建 .venv 目录）
python -m venv .venv
```

### 步骤 2: 激活虚拟环境
```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# Windows CMD
.venv\Scripts\activate.bat

# 激活成功后，命令行前会显示 (.venv) 前缀
```

### 步骤 3: 安装项目依赖
```bash
# 方式 1: 安装基础依赖
pip install -r requirements.txt

# 方式 2: 以开发模式安装项目（推荐）
pip install -e .

# 方式 3: 安装全部功能
pip install -e ".[all]"
```

### 步骤 4: 验证安装
```bash
# 检查 PyTorch 是否安装成功
python -c "import torch; print('PyTorch version:', torch.__version__)"

# 检查 Lightning 是否安装成功
python -c "import pytorch_lightning as pl; print('Lightning version:', pl.__version__)"

# 检查项目是否正确安装
python -c "from src.models.url_encoder import UrlBertEncoder; print('项目导入成功！')"
```

### 步骤 5: 退出虚拟环境（使用完毕后）
```bash
deactivate
```

---

## 🐍 方法 2: 使用 Conda（适合需要 GPU 的情况）

### 步骤 1: 创建 Conda 环境
```bash
cd D:\uaam-phish
conda env create -f environment.yml
```

### 步骤 2: 激活环境
```bash
conda activate uaam-phish
```

### 步骤 3: 验证安装
```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

### 步骤 4: 退出环境
```bash
conda deactivate
```

---

## 🔄 如果您想从全局环境迁移

### 当前情况
您已经在全局环境安装了所有依赖，可以选择：

**选项 A: 保持现状**
- ✅ 所有包已安装，可以直接使用
- ❌ 不推荐，会污染全局环境

**选项 B: 迁移到虚拟环境（推荐）**
```bash
# 1. 创建虚拟环境
python -m venv .venv

# 2. 激活虚拟环境
.venv\Scripts\Activate.ps1

# 3. 安装项目（开发模式）
pip install -e ".[all]"

# 4. 测试是否工作
python scripts/train.py --help
```

---

## ⚡ 快速开始（如果已有虚拟环境）

```bash
# 每次开始工作时
cd D:\uaam-phish
.venv\Scripts\Activate.ps1

# 训练模型
python scripts/train.py --profile local

# 工作完成后
deactivate
```

---

## 🆘 常见问题

### Q1: PowerShell 报错 "无法加载文件，因为在此系统上禁用了运行脚本"
**解决方案:**
```powershell
# 以管理员身份运行 PowerShell，执行：
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# 然后重试激活命令
.venv\Scripts\Activate.ps1
```

### Q2: 想要使用 GPU 版本的 PyTorch
**解决方案:**
```bash
# 先激活虚拟环境
.venv\Scripts\Activate.ps1

# 卸载 CPU 版本
pip uninstall torch torchvision -y

# 安装 GPU 版本（CUDA 12.1）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 或访问 https://pytorch.org/ 获取适合您 CUDA 版本的安装命令
```

### Q3: pip 安装很慢
**解决方案：使用国内镜像**
```bash
# 临时使用清华镜像
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 或永久配置（推荐）
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

### Q4: 如何删除虚拟环境重新安装？
```bash
# 1. 先退出虚拟环境
deactivate

# 2. 删除 .venv 目录
rmdir /s .venv

# 3. 重新创建
python -m venv .venv
```

---

## 📋 推荐的工作流程

### 第一次设置（仅需一次）
```bash
cd D:\uaam-phish
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[all]"
```

### 日常使用
```bash
# 1. 进入项目目录并激活环境
cd D:\uaam-phish
.venv\Scripts\Activate.ps1

# 2. 工作（训练、测试等）
python scripts/train.py --profile local

# 3. 完成后退出
deactivate
```

---

## ✅ 验证清单

安装完成后，请执行以下检查：

```bash
# 1. 确认在虚拟环境中
python -c "import sys; print('虚拟环境:', sys.prefix)"
# 应该显示: D:\uaam-phish\.venv

# 2. 检查关键包
python -c "import torch, pytorch_lightning, transformers, pandas; print('所有核心包导入成功')"

# 3. 检查项目结构
python -c "from src.systems.url_only_module import UrlOnlySystem; print('项目模块可用')"

# 4. 运行快速测试
python -c "from src.utils.seed import set_global_seed; set_global_seed(42); print('工具函数正常')"
```

---

## 🎓 IDE 配置

### VS Code
1. 打开项目文件夹
2. 按 `Ctrl + Shift + P`
3. 输入 "Python: Select Interpreter"
4. 选择 `.venv\Scripts\python.exe`

### PyCharm
1. File → Settings → Project → Python Interpreter
2. 点击齿轮图标 → Add
3. 选择 Existing Environment
4. 选择 `.venv\Scripts\python.exe`

---

**安装遇到问题？** 请查看 `docs/DEPENDENCIES.md` 获取详细的依赖说明。

---

## 🚀 完整项目设置流程（推荐）

完成基础安装后，按照以下步骤完成项目的完整设置：

### A. 离线缓存 HuggingFace 模型（可选但推荐）

这样可以在无网络环境下训练，避免每次下载模型：

```bash
# 1. 安装 HuggingFace CLI（如果还没有）
pip install huggingface-hub

# 2. 下载 RoBERTa 模型到本地
huggingface-cli download roberta-base --local-dir models/roberta-base

# 3. 设置环境变量（Windows PowerShell）
$env:HF_CACHE_DIR = "$PWD/models/roberta-base"
$env:HF_LOCAL_ONLY = "1"

# 或在 Linux/macOS:
# export HF_CACHE_DIR=$PWD/models/roberta-base
# export HF_LOCAL_ONLY=1
```

**提示**: 也可以将这些环境变量添加到系统环境变量中，避免每次手动设置。

### B. 安装测试和开发工具

确保所有开发依赖都已安装：

```bash
make init
```

这会安装 `requirements.txt` 中的所有依赖，包括：
- `ruff` - 代码检查工具
- `black` - 代码格式化工具
- `pytest` - 测试框架
- `dvc` - 数据版本控制

### C. 初始化 DVC 数据管道（首次运行）

DVC 用于管理数据处理流程和版本控制：

```bash
# 1. 初始化 DVC
make dvc-init

# 2. 运行数据预处理管道
dvc repro

# 或手动运行（如果你想看详细输出）
python scripts/build_master_and_splits.py --benign data/raw/dataset --phish data/raw/fish_dataset --outdir data/processed

# 3. 跟踪处理后的数据（可选）
git add dvc.yaml data/processed*.dvc .gitignore || true
```

**说明**: 
- `dvc repro` 会根据 `dvc.yaml` 自动运行数据处理脚本
- 生成的文件在 `data/processed/` 目录下
- DVC 会自动跟踪这些文件的变化

### D. 设置 Git Hooks（可选）

为了确保代码质量，可以设置 pre-commit hook：

**方式 1: 使用 Make 命令（推荐）**
```bash
make install-hooks
```

**方式 2: 使用安装脚本**
```bash
# Windows PowerShell
.\.github\hooks\install-hooks.ps1

# Linux/macOS/Git Bash
bash .github/hooks/install-hooks.sh
```

**方式 3: 手动复制（备选）**
```bash
# Windows PowerShell
Copy-Item .github/hooks/pre-commit .git/hooks/pre-commit

# Linux/macOS
cp .github/hooks/pre-commit .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
```

**hook 功能**:
- ✅ 自动运行 `ruff` 检查代码风格
- ✅ 自动运行 `black` 检查格式
- ✅ 自动运行 `pytest` 确保测试通过
- ⚠️ 如果检查失败,commit 会被阻止

### E. 验证设置

确认所有组件都正常工作：

```bash
# 1. 代码检查
make lint

# 2. 运行测试
make test

# 3. 快速训练测试（3个epoch）
make train
```

**预期结果**:
- `make lint`: 无错误输出
- `make test`: 所有测试通过
- `make train`: 成功启动训练并保存检查点到 `lightning_logs/`

---

## 📊 完整的一次性设置脚本

如果您想一次性完成所有设置（适合新机器或 CI 环境）：

**Windows PowerShell:**
```powershell
# 1. 创建虚拟环境
python -m venv .venv
.venv\Scripts\Activate.ps1

# 2. 安装依赖
make init

# 3. 下载模型（可选，需要网络）
# huggingface-cli download roberta-base --local-dir models/roberta-base

# 4. 设置环境变量（如果使用离线模型）
$env:HF_CACHE_DIR = "$PWD/models/roberta-base"
$env:HF_LOCAL_ONLY = "1"

# 5. 初始化 DVC 和处理数据
make dvc-init
dvc repro

# 6. 安装 Git Hooks（可选）
make install-hooks

# 7. 验证
make lint
make test
make train
```

**Linux/macOS:**
```bash
# 1. 创建虚拟环境
python -m venv .venv
source .venv/bin/activate

# 2. 安装依赖
make init

# 3. 下载模型（可选，需要网络）
# huggingface-cli download roberta-base --local-dir models/roberta-base

# 4. 设置环境变量（如果使用离线模型）
export HF_CACHE_DIR=$PWD/models/roberta-base
export HF_LOCAL_ONLY=1

# 5. 初始化 DVC 和处理数据
make dvc-init
dvc repro

# 6. 安装 Git Hooks（可选）
make install-hooks

# 7. 验证
make lint
make test
make train
```

---

## 🔄 日常开发工作流

设置完成后，日常开发流程：

```bash
# 1. 进入项目并激活环境
cd D:\uaam-phish
.venv\Scripts\Activate.ps1  # Windows
# source .venv/bin/activate  # Linux/macOS

# 2. 拉取最新代码和数据
git pull
dvc pull  # 如果使用 DVC 远程存储

# 3. 开发和测试
make lint      # 检查代码
make test      # 运行测试
make train     # 训练模型

# 4. 提交代码（pre-commit hook 会自动运行检查）
git add .
git commit -m "描述你的改动"
git push

# 5. 完成后退出环境
deactivate
```

---

## 🎯 常用 Make 命令

| 命令 | 说明 |
|------|------|
| `make init` | 安装所有依赖 |
| `make install-hooks` | 安装 Git pre-commit hooks |
| `make validate-data` | 验证数据schema完整性 |
| `make lint` | 运行代码检查（ruff + black） |
| `make test` | 运行所有测试 |
| `make train` | 使用本地配置训练模型 |
| `make eval` | 仅评估模型（不训练） |
| `make dvc-init` | 初始化 DVC |
| `make dvc-track` | 跟踪处理后的数据 |
| `make dvc-push` | 推送数据到远程存储 |

**提示**: 查看 `Makefile` 了解所有可用命令和自定义选项。

---


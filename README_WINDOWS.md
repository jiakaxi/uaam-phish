# Windows 用户指南

本项目同时支持 Linux/Unix 和 Windows 系统。本文档针对 Windows 用户提供特定说明。

## 🪟 Windows 特定文件

### 可用的 Windows 工具

| Linux/Unix 文件 | Windows 替代方案 | 用途 |
|----------------|-----------------|------|
| `Makefile` | `Makefile.ps1` | 项目任务自动化 |
| `.github/hooks/install-hooks.sh` | `.github/hooks/install-hooks.ps1` | 安装 Git hooks |
| `.github/hooks/pre-commit` | 自动适配（使用 `python -m`） | Git 提交前检查 |

## 📦 快速开始（Windows）

### 1. 安装依赖

```powershell
# 使用 PowerShell Makefile
.\Makefile.ps1 init

# 或手动安装
python -m pip install -U pip
pip install -r requirements.txt
```

### 2. 安装 Git Hooks

```powershell
# 运行 PowerShell 脚本
.\.github\hooks\install-hooks.ps1
```

### 3. 验证数据

```powershell
.\Makefile.ps1 validate-data
```

### 4. 运行测试

```powershell
.\Makefile.ps1 test
```

### 5. 代码检查

```powershell
.\Makefile.ps1 lint
```

## 🚀 训练模型

```powershell
# 本地训练
.\Makefile.ps1 train

# 或使用环境变量
$env:HF_LOCAL_ONLY = "1"
$env:HF_CACHE_DIR = ".\models\roberta-base"
$env:DATA_ROOT = ".\data\processed"
python scripts\train.py --profile local
```

## 🔧 常见 PowerShell 命令

```powershell
# 查看所有可用命令
.\Makefile.ps1 help

# 初始化 DVC
.\Makefile.ps1 dvc-init

# 追踪数据
.\Makefile.ps1 dvc-track

# 推送 DVC 数据
.\Makefile.ps1 dvc-push
```

## ⚠️ Windows 注意事项

### 1. 使用虚拟环境

推荐使用虚拟环境：

```powershell
# 创建虚拟环境
python -m venv .venv

# 激活虚拟环境
.\.venv\Scripts\Activate.ps1

# 如果遇到执行策略错误，运行：
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 2. Git Bash vs PowerShell

- **PowerShell**（推荐）：使用 `Makefile.ps1` 和 `.ps1` 脚本
- **Git Bash**：可以使用原始的 `Makefile` 和 `.sh` 脚本

### 3. 路径分隔符

Windows 使用反斜杠 `\`，但 Python 代码中已自动处理，使用 `pathlib.Path` 确保跨平台兼容。

### 4. 长路径支持

如果遇到路径过长的问题，启用 Windows 长路径支持：

```powershell
# 以管理员身份运行
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
    -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

## 🐛 故障排查

### Git hooks 无法运行

```powershell
# 重新安装 hooks
.\.github\hooks\install-hooks.ps1

# 或跳过 hooks 提交
git commit --no-verify -m "your message"
```

### 找不到命令

确保已激活虚拟环境并安装了所有依赖：

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### DVC 相关问题

```powershell
# 检查 DVC 是否安装
dvc version

# 重新初始化
.\Makefile.ps1 dvc-init
```

## 📚 更多资源

- 主要文档：`README.md`
- 快速开始：`QUICKSTART.md`
- 安装指南：`INSTALL.md`
- Linux/Unix 用户请参考 `Makefile`


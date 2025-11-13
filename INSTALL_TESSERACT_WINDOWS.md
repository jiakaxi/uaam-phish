# Windows 下安装 Tesseract OCR 指南

## 方法 1: 使用安装程序（推荐）

### 步骤 1: 下载 Tesseract 安装程序
访问官方下载页面：
- GitHub Release: https://github.com/UB-Mannheim/tesseract/wiki
- 直接下载链接（64位）: https://digi.bib.uni-mannheim.de/tesseract/tesseract-ocr-w64-setup-5.3.3.20231005.exe

### 步骤 2: 运行安装程序
1. 双击下载的 `.exe` 文件
2. 选择安装路径（建议默认）：`C:\Program Files\Tesseract-OCR\`
3. **重要**: 勾选 "Add to PATH" 选项
4. 完成安装

### 步骤 3: 配置环境变量（如果安装时未自动添加）
```powershell
# 添加到 PATH
$env:Path += ";C:\Program Files\Tesseract-OCR"
[System.Environment]::SetEnvironmentVariable("Path", $env:Path, [System.EnvironmentVariableTarget]::Machine)

# 设置 TESSDATA_PREFIX（可选）
[System.Environment]::SetEnvironmentVariable("TESSDATA_PREFIX", "C:\Program Files\Tesseract-OCR\tessdata", [System.EnvironmentVariableTarget]::Machine)
```

### 步骤 4: 验证安装
```powershell
# 重新打开 PowerShell 窗口
tesseract --version
```

预期输出：
```
tesseract 5.3.3
  leptonica-1.83.1
    libgif 5.2.1 : libjpeg 8d (libjpeg-turbo 2.1.5.1) : libpng 1.6.40 : libtiff 4.5.1 : zlib 1.2.13 : libwebp 1.3.2 : libopenjp2 2.5.0
  Found AVX2
  Found AVX
  Found FMA
  Found SSE4.1
  Found OpenMP 201511
```

---

## 方法 2: 使用 Chocolatey 包管理器

### 前置条件: 安装 Chocolatey
```powershell
# 以管理员身份运行 PowerShell
Set-ExecutionPolicy Bypass -Scope Process -Force
[System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
```

### 安装 Tesseract
```powershell
# 以管理员身份运行
choco install tesseract -y
```

### 验证安装
```powershell
tesseract --version
```

---

## 方法 3: 使用 Conda（如果使用 Conda 环境）

```bash
conda install -c conda-forge tesseract
```

---

## Python 集成测试

安装完成后，测试 pytesseract 是否能找到 Tesseract：

```python
import pytesseract
from PIL import Image

# 测试 1: 检查版本
try:
    version = pytesseract.get_tesseract_version()
    print(f"✓ Tesseract version: {version}")
except Exception as e:
    print(f"✗ Error: {e}")

# 测试 2: 如果自动检测失败，手动指定路径
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# 测试 3: 简单 OCR 测试
try:
    # 创建一个简单的测试图片
    img = Image.new('RGB', (200, 100), color='white')
    text = pytesseract.image_to_string(img)
    print("✓ OCR test successful!")
except Exception as e:
    print(f"✗ OCR test failed: {e}")
```

---

## 常见问题

### Q1: 提示 "tesseract is not installed or it's not in your PATH"
**解决**:
1. 确认 Tesseract 已安装
2. 将安装路径添加到系统 PATH
3. 重启 PowerShell/命令行窗口
4. 或在代码中手动指定路径：
   ```python
   pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
   ```

### Q2: 找到 Tesseract 但语言包缺失
**解决**:
下载中文语言包（如需要）：
1. 访问：https://github.com/tesseract-ocr/tessdata
2. 下载 `chi_sim.traineddata`（简体中文）或 `chi_tra.traineddata`（繁体中文）
3. 放入 `C:\Program Files\Tesseract-OCR\tessdata\` 目录

### Q3: C-Module 仍然提示 pytesseract_missing
**解决**:
1. 确认 pytesseract 已安装：`pip install pytesseract`
2. 检查 src/modules/c_module.py 中的 pytesseract 导入
3. 重启 Python 环境

---

## 项目中配置验证

安装完成后，验证配置：

```bash
# 检查配置
cat configs/experiment/s3_iid_fixed.yaml | grep -A 5 "c_module:"

# 应该看到：
# c_module:
#   use_ocr: true  ← 确认为 true
```

然后运行测试：
```bash
python scripts/train_hydra.py experiment=s3_iid_fixed run.seed=100 trainer.max_epochs=1 trainer.limit_test_batches=2
```

查看日志中的 `>> C-MODULE DEBUG:` 部分，确认 `brand_vis` 提取率 > 0%

---

## 快速验证脚本

创建并运行此脚本快速测试：

```python
# test_ocr.py
import sys
try:
    import pytesseract
    print("✓ pytesseract installed")
    version = pytesseract.get_tesseract_version()
    print(f"✓ Tesseract version: {version}")
    print("✓ OCR ready!")
    sys.exit(0)
except ImportError:
    print("✗ pytesseract not installed")
    print("  Run: pip install pytesseract")
    sys.exit(1)
except Exception as e:
    print(f"✗ Tesseract not found: {e}")
    print("  Install Tesseract OCR and add to PATH")
    sys.exit(1)
```

运行：
```bash
python test_ocr.py
```

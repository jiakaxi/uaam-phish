# S3 固定融合实验 - 完整安装和运行脚本
# 2025-11-13

Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host "S3 固定融合实验 - Tesseract 安装和配置" -ForegroundColor Cyan
Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host ""

# 步骤 1: 检查 Chocolatey
Write-Host "[1/6] 检查 Chocolatey 包管理器..." -ForegroundColor Yellow
$chocoInstalled = Get-Command choco -ErrorAction SilentlyContinue
if (-not $chocoInstalled) {
    Write-Host "  Chocolatey 未安装。正在安装..." -ForegroundColor Yellow
    Write-Host "  请在管理员 PowerShell 中运行:" -ForegroundColor Red
    Write-Host ""
    Write-Host "  Set-ExecutionPolicy Bypass -Scope Process -Force;" -ForegroundColor White
    Write-Host "  [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072;" -ForegroundColor White
    Write-Host "  iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))" -ForegroundColor White
    Write-Host ""
    Write-Host "  或者手动下载 Tesseract:" -ForegroundColor Yellow
    Write-Host "  https://github.com/UB-Mannheim/tesseract/wiki" -ForegroundColor Cyan
    Write-Host ""
    exit 1
} else {
    Write-Host "  ✓ Chocolatey 已安装" -ForegroundColor Green
}

# 步骤 2: 安装 Tesseract
Write-Host ""
Write-Host "[2/6] 安装 Tesseract OCR..." -ForegroundColor Yellow
$tesseractPath = "C:\Program Files\Tesseract-OCR\tesseract.exe"
if (Test-Path $tesseractPath) {
    Write-Host "  ✓ Tesseract 已安装: $tesseractPath" -ForegroundColor Green
} else {
    Write-Host "  正在通过 Chocolatey 安装 Tesseract..." -ForegroundColor Yellow
    Write-Host "  需要管理员权限！" -ForegroundColor Red
    Write-Host ""
    Write-Host "  请在管理员 PowerShell 中运行:" -ForegroundColor Red
    Write-Host "  choco install tesseract -y" -ForegroundColor White
    Write-Host ""
    exit 1
}

# 步骤 3: 验证 Tesseract 命令行
Write-Host ""
Write-Host "[3/6] 验证 Tesseract 可执行文件..." -ForegroundColor Yellow
try {
    $tesseractVersion = & $tesseractPath --version 2>&1 | Select-Object -First 1
    Write-Host "  ✓ Tesseract 版本: $tesseractVersion" -ForegroundColor Green
} catch {
    Write-Host "  ✗ 无法执行 Tesseract" -ForegroundColor Red
    exit 1
}

# 步骤 4: 验证 pytesseract
Write-Host ""
Write-Host "[4/6] 验证 pytesseract Python 包..." -ForegroundColor Yellow
$pytesseractCheck = python -c "import pytesseract; pytesseract.pytesseract.tesseract_cmd = r'$tesseractPath'; print('Version:', pytesseract.get_tesseract_version())" 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "  ✓ $pytesseractCheck" -ForegroundColor Green
} else {
    Write-Host "  ✗ pytesseract 测试失败" -ForegroundColor Red
    Write-Host "  错误: $pytesseractCheck" -ForegroundColor Red
    Write-Host "  尝试: pip install pytesseract" -ForegroundColor Yellow
    exit 1
}

# 步骤 5: 配置检查
Write-Host ""
Write-Host "[5/6] 检查 S3 实验配置..." -ForegroundColor Yellow
$configFiles = @("configs/experiment/s3_iid_fixed.yaml", "configs/experiment/s3_brandood_fixed.yaml")
foreach ($config in $configFiles) {
    if (Test-Path $config) {
        $content = Get-Content $config -Raw
        if ($content -match "use_ocr:\s*true") {
            Write-Host "  ✓ $config - use_ocr: true" -ForegroundColor Green
        } else {
            Write-Host "  ⚠ $config - use_ocr: false (将自动修复)" -ForegroundColor Yellow
        }
    } else {
        Write-Host "  ✗ 配置文件不存在: $config" -ForegroundColor Red
    }
}

# 步骤 6: 运行实验
Write-Host ""
Write-Host "[6/6] 准备运行 S3 实验..." -ForegroundColor Yellow
Write-Host ""
Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host "✓ 所有检查通过！可以运行实验了。" -ForegroundColor Green
Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host ""

# 提供运行命令
Write-Host "运行 S3 实验的命令:" -ForegroundColor Yellow
Write-Host ""
Write-Host "# IID 协议 (快速测试)" -ForegroundColor Cyan
Write-Host "python scripts/train_hydra.py experiment=s3_iid_fixed run.seed=100 trainer.max_epochs=1 trainer.limit_val_batches=5 trainer.limit_test_batches=10" -ForegroundColor White
Write-Host ""
Write-Host "# IID 协议 (完整训练)" -ForegroundColor Cyan
Write-Host "python scripts/train_hydra.py experiment=s3_iid_fixed run.seed=100" -ForegroundColor White
Write-Host ""
Write-Host "# Brand-OOD 协议 (完整训练)" -ForegroundColor Cyan
Write-Host "python scripts/train_hydra.py experiment=s3_brandood_fixed run.seed=100" -ForegroundColor White
Write-Host ""

# 询问是否立即运行
$runNow = Read-Host "是否立即运行快速测试实验？(y/n)"
if ($runNow -eq 'y' -or $runNow -eq 'Y') {
    Write-Host ""
    Write-Host "正在运行 S3 IID 快速测试..." -ForegroundColor Yellow
    Write-Host ""

    # 设置环境变量确保 Tesseract 可被找到
    $env:TESSDATA_PREFIX = "C:\Program Files\Tesseract-OCR\tessdata"

    # 运行快速测试
    python scripts/train_hydra.py experiment=s3_iid_fixed run.seed=100 trainer.max_epochs=1 trainer.limit_val_batches=5 trainer.limit_test_batches=10

    Write-Host ""
    Write-Host "实验完成！请检查输出中的:" -ForegroundColor Yellow
    Write-Host "  - '>> C-MODULE DEBUG:' 部分，确认 brand_vis 提取率 > 0%" -ForegroundColor Cyan
    Write-Host "  - 'alpha_url', 'alpha_html', 'alpha_visual' 的值" -ForegroundColor Cyan
    Write-Host "  - 期望: alpha_visual > 0（不再是 0.000）" -ForegroundColor Green
}

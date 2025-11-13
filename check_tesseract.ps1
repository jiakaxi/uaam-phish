# Tesseract OCR 安装检查脚本
# 简化版

Write-Host ""
Write-Host "======================================" -ForegroundColor Cyan
Write-Host "Tesseract OCR 安装检查" -ForegroundColor Yellow
Write-Host "======================================" -ForegroundColor Cyan
Write-Host ""

# 检查 1: Chocolatey
Write-Host "[1/5] 检查 Chocolatey..." -ForegroundColor Yellow
$choco = Get-Command choco -ErrorAction SilentlyContinue
if ($choco) {
    Write-Host "  OK: Chocolatey 已安装" -ForegroundColor Green
} else {
    Write-Host "  NOT FOUND: Chocolatey 未安装" -ForegroundColor Red
    Write-Host "  提示: 可以手动下载 Tesseract 安装" -ForegroundColor Yellow
}

# 检查 2: Tesseract 可执行文件
Write-Host ""
Write-Host "[2/5] 检查 Tesseract 可执行文件..." -ForegroundColor Yellow
$paths = @(
    "C:\Program Files\Tesseract-OCR\tesseract.exe",
    "C:\Users\$env:USERNAME\AppData\Local\Programs\Tesseract-OCR\tesseract.exe",
    "C:\Tesseract-OCR\tesseract.exe"
)

$found = $false
foreach ($path in $paths) {
    if (Test-Path $path) {
        Write-Host "  OK: 找到 Tesseract - $path" -ForegroundColor Green
        $tesseractPath = $path
        $found = $true
        break
    }
}

if (-not $found) {
    Write-Host "  NOT FOUND: Tesseract 未安装" -ForegroundColor Red
    Write-Host ""
    Write-Host "安装方法:" -ForegroundColor Yellow
    Write-Host "  1. 使用 Chocolatey (需要管理员):" -ForegroundColor White
    Write-Host "     choco install tesseract -y" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  2. 手动下载安装:" -ForegroundColor White
    Write-Host "     https://github.com/UB-Mannheim/tesseract/wiki" -ForegroundColor Cyan
    Write-Host ""
    exit 1
}

# 检查 3: Tesseract 版本
Write-Host ""
Write-Host "[3/5] 检查 Tesseract 版本..." -ForegroundColor Yellow
try {
    $version = & $tesseractPath --version 2>&1 | Select-Object -First 1
    Write-Host "  OK: $version" -ForegroundColor Green
} catch {
    Write-Host "  ERROR: 无法运行 Tesseract" -ForegroundColor Red
    exit 1
}

# 检查 4: pytesseract 包
Write-Host ""
Write-Host "[4/5] 检查 pytesseract Python 包..." -ForegroundColor Yellow
$result = python -c "import pytesseract; print('OK')" 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "  OK: pytesseract 已安装" -ForegroundColor Green
} else {
    Write-Host "  NOT FOUND: pytesseract 未安装" -ForegroundColor Red
    Write-Host "  运行: pip install pytesseract" -ForegroundColor Yellow
    exit 1
}

# 检查 5: Python 能否找到 Tesseract
Write-Host ""
Write-Host "[5/5] 测试 Python + Tesseract 集成..." -ForegroundColor Yellow
$testScript = @"
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'$tesseractPath'
try:
    version = pytesseract.get_tesseract_version()
    print(f'OK: Version {version}')
except Exception as e:
    print(f'ERROR: {e}')
    exit(1)
"@

$result = python -c $testScript 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "  $result" -ForegroundColor Green
} else {
    Write-Host "  ERROR: $result" -ForegroundColor Red
    exit 1
}

# 所有检查通过
Write-Host ""
Write-Host "======================================" -ForegroundColor Cyan
Write-Host "✓ 所有检查通过!" -ForegroundColor Green
Write-Host "======================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Tesseract 路径: $tesseractPath" -ForegroundColor Cyan
Write-Host ""
Write-Host "下一步: 运行 S3 实验" -ForegroundColor Yellow
Write-Host ""
Write-Host "快速测试 (1 epoch):" -ForegroundColor White
Write-Host "  python scripts/train_hydra.py experiment=s3_iid_fixed run.seed=100 trainer.max_epochs=1 trainer.limit_val_batches=5 trainer.limit_test_batches=10" -ForegroundColor Cyan
Write-Host ""
Write-Host "完整训练:" -ForegroundColor White
Write-Host "  python scripts/train_hydra.py experiment=s3_iid_fixed run.seed=100" -ForegroundColor Cyan
Write-Host ""

# 询问是否运行
$response = Read-Host "是否现在运行快速测试? (y/n)"
if ($response -eq 'y' -or $response -eq 'Y') {
    Write-Host ""
    Write-Host "启动 S3 IID 快速测试..." -ForegroundColor Yellow
    Write-Host ""
    python scripts/train_hydra.py experiment=s3_iid_fixed run.seed=100 trainer.max_epochs=1 trainer.limit_val_batches=5 trainer.limit_test_batches=10
}

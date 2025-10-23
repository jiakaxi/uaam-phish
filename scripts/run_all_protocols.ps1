# URL-Only 三协议一键运行脚本（Windows PowerShell）
# 依次运行 random, temporal, brand_ood 三个协议

$ErrorActionPreference = "Stop"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "URL-Only 三协议实验" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# 检查 master.csv
if (-not (Test-Path "data/processed/master.csv")) {
    Write-Host "⚠️  未找到 data/processed/master.csv" -ForegroundColor Yellow
    Write-Host "   正在创建..."
    python scripts/create_master_csv.py
    Write-Host ""
}

# 运行三个协议
$protocols = @("random", "temporal", "brand_ood")

foreach ($protocol in $protocols) {
    Write-Host "============================================================" -ForegroundColor Green
    Write-Host "Running protocol: $protocol" -ForegroundColor Green
    Write-Host "============================================================" -ForegroundColor Green

    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $runName = "url_${protocol}_${timestamp}"

    python scripts/train_hydra.py `
        protocol=$protocol `
        use_build_splits=true `
        run.name=$runName

    Write-Host ""
    Write-Host "✅ Protocol $protocol completed" -ForegroundColor Green
    Write-Host ""
}

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "All protocols completed!" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "运行验证脚本:" -ForegroundColor Yellow
Write-Host "  python tools/check_artifacts_url_only.py"

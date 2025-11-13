# S3 完整测试脚本 - 包含所有验证步骤
# 2025-11-14

Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host "S3 三模态融合 - 完整测试" -ForegroundColor Yellow
Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host ""

# 步骤 1: 验证配置
Write-Host "[1/4] 验证配置文件..." -ForegroundColor Yellow
$config = Get-Content configs\experiment\s3_iid_fixed.yaml -Raw

if ($config -match "use_umodule:\s*true") {
    Write-Host "  OK: use_umodule = true" -ForegroundColor Green
} else {
    Write-Host "  ERROR: use_umodule not enabled!" -ForegroundColor Red
    exit 1
}

if ($config -match "use_ocr:\s*true") {
    Write-Host "  OK: use_ocr = true" -ForegroundColor Green
} else {
    Write-Host "  WARNING: use_ocr not enabled" -ForegroundColor Yellow
}

Write-Host ""

# 步骤 2: 运行实验
Write-Host "[2/4] 运行 S3 IID 实验..." -ForegroundColor Yellow
Write-Host "  命令: python scripts/train_hydra.py experiment=s3_iid_fixed run.seed=600 trainer.max_epochs=1 trainer.limit_test_batches=20" -ForegroundColor Cyan
Write-Host ""

$startTime = Get-Date
python scripts/train_hydra.py experiment=s3_iid_fixed run.seed=600 trainer.max_epochs=1 trainer.limit_val_batches=5 trainer.limit_test_batches=20

$duration = (Get-Date) - $startTime
Write-Host ""
Write-Host "  实验完成，用时: $($duration.TotalMinutes.ToString('0.0')) 分钟" -ForegroundColor Green
Write-Host ""

# 步骤 3: 检查结果
Write-Host "[3/4] 分析结果..." -ForegroundColor Yellow

$latestExp = Get-ChildItem experiments\s3_iid_fixed_* | Sort-Object LastWriteTime -Descending | Select-Object -First 1
Write-Host "  最新实验: $($latestExp.Name)" -ForegroundColor Cyan

if (Test-Path "$($latestExp.FullName)\artifacts\predictions_test.csv") {
    Write-Host "  运行覆盖率分析..." -ForegroundColor Cyan
    Write-Host ""
    python check_ocr_coverage.py
} else {
    Write-Host "  ERROR: predictions_test.csv not found!" -ForegroundColor Red
}

Write-Host ""

# 步骤 4: 检查关键日志
Write-Host "[4/4] 检查关键日志..." -ForegroundColor Yellow

$logFile = "$($latestExp.FullName)\logs\train.log"
if (Test-Path $logFile) {
    Write-Host "  检查 Dropout 层检测:" -ForegroundColor Cyan
    Get-Content $logFile | Select-String "Dropout layers by modality" | Select-Object -First 1

    Write-Host ""
    Write-Host "  检查 MC Dropout 结果:" -ForegroundColor Cyan
    Get-Content $logFile | Select-String "MC DROPOUT RESULTS" -Context 0,5 | Select-Object -First 10

    Write-Host ""
    Write-Host "  检查 Visual 可靠性:" -ForegroundColor Cyan
    Get-Content $logFile | Select-String "VISUAL.*var_tensor is None|Using default variance for visual" | Select-Object -First 3
}

Write-Host ""
Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host "测试完成！" -ForegroundColor Green
Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host ""
Write-Host "查看完整日志: $logFile" -ForegroundColor Cyan
Write-Host "查看预测结果: $($latestExp.FullName)\artifacts\predictions_test.csv" -ForegroundColor Cyan

# 实时监控当前 S4 实验
# 使用方法: .\monitor_current_experiment.ps1

Write-Host "=" * 70
Write-Host "S4 实验实时监控"
Write-Host "=" * 70

# 查找最新实验
$latestExp = Get-ChildItem outputs\2025-11-14 -Directory |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

if (-not $latestExp) {
    Write-Host "[错误] 未找到实验目录" -ForegroundColor Red
    exit 1
}

Write-Host "`n[实验信息]"
Write-Host "目录: $($latestExp.FullName)"
Write-Host "开始时间: $($latestExp.CreationTime)"
Write-Host "=" * 70

$logFile = Join-Path $latestExp.FullName "train_hydra.log"

if (-not (Test-Path $logFile)) {
    Write-Host "[错误] 日志文件不存在" -ForegroundColor Red
    exit 1
}

Write-Host "`n按 Ctrl+C 停止监控`n"
Write-Host "=" * 70

# 实时监控日志（每次显示最后 15 行并自动刷新）
Get-Content $logFile -Wait -Tail 15

# 监控 Brand-OOD 实验，完成后自动启动 IID 实验

Write-Host "=" * 70
Write-Host "S4 实验监控与队列"
Write-Host "=" * 70

Write-Host "`n[1/2] 监控 Brand-OOD 实验..."

$brandoodDir = Get-ChildItem -Path "outputs\2025-11-14" -Directory |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

Write-Host "实验目录: $($brandoodDir.FullName)"

# 监控循环
$checkInterval = 30  # 每 30 秒检查一次
$maxWaitTime = 3600  # 最多等待 1 小时
$elapsed = 0

while ($elapsed -lt $maxWaitTime) {
    Start-Sleep -Seconds $checkInterval
    $elapsed += $checkInterval

    # 检查是否完成 (查找 SUMMARY.md 或 eval_summary.json)
    $summaryFile = Join-Path $brandoodDir.FullName "SUMMARY.md"
    $evalFile = Join-Path $brandoodDir.FullName "results\eval_summary.json"

    if ((Test-Path $summaryFile) -or (Test-Path $evalFile)) {
        Write-Host "`n[完成] Brand-OOD 实验已完成!" -ForegroundColor Green
        Write-Host "用时: $($elapsed)秒"
        break
    }

    # 显示进度
    $logFile = Join-Path $brandoodDir.FullName "train_hydra.log"
    if (Test-Path $logFile) {
        $epochInfo = Get-Content $logFile | Select-String "Epoch" | Select-Object -Last 1
        Write-Host "[进度] $epochInfo"
    }

    Write-Host "." -NoNewline
}

if ($elapsed -ge $maxWaitTime) {
    Write-Host "`n[超时] Brand-OOD 实验运行超过 1 小时" -ForegroundColor Yellow
    $continue = Read-Host "是否继续等待? (y/n)"
    if ($continue -ne "y") {
        Write-Host "用户取消，不启动 IID 实验"
        exit 1
    }
}

# 启动 IID 实验
Write-Host "`n"
Write-Host "=" * 70
Write-Host "[2/2] 启动 IID 实验..."
Write-Host "=" * 70

python scripts/train_hydra.py `
    experiment=s4_iid_rcaf `
    train.epochs=10 `
    logger=wandb

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n[成功] 两个实验全部完成!" -ForegroundColor Green
} else {
    Write-Host "`n[失败] IID 实验失败，退出码: $LASTEXITCODE" -ForegroundColor Red
}

Write-Host "=" * 70

# S4 实验自动化运行脚本
# 依次运行 Brand-OOD 和 IID 实验

Write-Host "=" * 70
Write-Host "S4 自适应融合实验自动化"
Write-Host "=" * 70

$experiments = @(
    @{
        name = "s4_brandood_rcaf"
        epochs = 10
        desc = "Brand-OOD (分布外品牌泛化)"
    },
    @{
        name = "s4_iid_rcaf"
        epochs = 10
        desc = "IID (独立同分布)"
    }
)

$total = $experiments.Count
$completed = 0
$failed = 0

foreach ($exp in $experiments) {
    $completed++

    Write-Host "`n"
    Write-Host "[$completed/$total] 开始实验: $($exp.desc)"
    Write-Host "配置: experiment=$($exp.name)"
    Write-Host "Epochs: $($exp.epochs)"
    Write-Host ("-" * 70)

    $startTime = Get-Date

    # 运行实验
    python scripts/train_hydra.py `
        experiment=$($exp.name) `
        train.epochs=$($exp.epochs) `
        logger=wandb

    $exitCode = $LASTEXITCODE
    $endTime = Get-Date
    $duration = $endTime - $startTime

    if ($exitCode -eq 0) {
        Write-Host "`n[成功] $($exp.name) 完成" -ForegroundColor Green
        Write-Host "用时: $($duration.ToString('hh\:mm\:ss'))"
    } else {
        Write-Host "`n[失败] $($exp.name) 退出码: $exitCode" -ForegroundColor Red
        $failed++

        # 询问是否继续
        $continue = Read-Host "是否继续下一个实验? (y/n)"
        if ($continue -ne "y") {
            Write-Host "用户取消，停止执行" -ForegroundColor Yellow
            break
        }
    }
}

# 最终总结
Write-Host "`n"
Write-Host "=" * 70
Write-Host "实验完成总结"
Write-Host "=" * 70
Write-Host "总实验数: $total"
Write-Host "已完成: $completed"
Write-Host "失败: $failed" -ForegroundColor $(if ($failed -gt 0) { "Red" } else { "Green" })

if ($failed -eq 0) {
    Write-Host "`n所有实验成功完成! ✓" -ForegroundColor Green
    Write-Host "`n下一步："
    Write-Host "1. 检查 outputs/ 目录查看结果"
    Write-Host "2. 运行分析脚本提取指标"
    Write-Host "3. 生成论文图表"
} else {
    Write-Host "`n部分实验失败，请检查日志" -ForegroundColor Yellow
}

Write-Host "=" * 70

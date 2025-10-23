# URL模型训练脚本 - 使用正确配置 + WandB同步
#
# 这个脚本使用正确的训练参数重新训练URL模型
# 并将训练过程同步到WandB进行可视化

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "URL 钓鱼检测模型训练" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "配置参数 (修复后):" -ForegroundColor Green
Write-Host "  - Dropout: 0.1" -ForegroundColor White
Write-Host "  - Epochs: 50" -ForegroundColor White
Write-Host "  - Batch Size: 64" -ForegroundColor White
Write-Host "  - Learning Rate: 0.0001" -ForegroundColor White
Write-Host "  - Logger: WandB" -ForegroundColor Yellow
Write-Host ""
Write-Host "预期结果: 准确率 95-99%, AUROC > 0.95" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# 设置实验名称
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$exp_name = "url_corrected_$timestamp"

Write-Host "实验名称: $exp_name" -ForegroundColor Yellow
Write-Host "开始训练..." -ForegroundColor Green
Write-Host ""

# 启动训练，使用WandB记录
python scripts/train_hydra.py `
    logger=wandb `
    run.name=$exp_name `
    run.tags=[url-only,corrected,fixed-config] `
    trainer=default `
    data=url_only `
    model=url_encoder

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "训练完成！" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Cyan

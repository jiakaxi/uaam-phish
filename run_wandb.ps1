# WandB 训练脚本
# 使用前请先运行: wandb login

# 设置 WandB 项目名称
$env:WANDB_PROJECT="uaam-phish"
# 可选：设置实验标签
$env:WANDB_TAGS="url-baseline,bilstm"

# 运行训练
Write-Host "开始训练 URL-only 模型并上传到 WandB..." -ForegroundColor Cyan
python scripts/train_hydra.py `
    logger=wandb `
    run.name="url_baseline_$(Get-Date -Format 'MMdd_HHmm')" `
    train.epochs=10

Write-Host "`n训练完成！" -ForegroundColor Green
Write-Host "访问 https://wandb.ai/$env:USERNAME/uaam-phish 查看结果" -ForegroundColor Yellow

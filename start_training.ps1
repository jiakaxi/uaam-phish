# 简单的训练启动脚本

Write-Host "启动URL模型训练 + WandB同步..." -ForegroundColor Green

python scripts/train_hydra.py `
    logger=wandb `
    trainer=default `
    data=url_only `
    model=url_encoder

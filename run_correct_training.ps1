# URL模型正确训练脚本
# 使用上次成功的配置参数

Write-Host "="*60
Write-Host "URL模型训练 - 使用正确配置"
Write-Host "="*60
Write-Host ""
Write-Host "配置说明:"
Write-Host "  - 学习率: 0.0001 (1e-4)"
Write-Host "  - 训练轮数: 50"
Write-Host "  - Batch size: 64"
Write-Host "  - Dropout: 0.1"
Write-Host ""
Write-Host "预期结果: 准确率 > 95%"
Write-Host "="*60
Write-Host ""

# 方式1: 使用正确的配置文件
python scripts/train_hydra.py experiment=url_baseline_correct

# 或者方式2: 直接覆盖参数
# python scripts/train_hydra.py `
#   data=url_only `
#   model=url_encoder `
#   train.epochs=50 `
#   train.bs=64 `
#   train.lr=0.0001 `
#   model.dropout=0.1 `
#   run.name=url_baseline_fixed

# HTML Baseline Training Script - Simple Version
# 10 epochs, freeze_bert=false

Write-Host "========================================"
Write-Host "HTML Baseline Training - 10 epochs"
Write-Host "========================================"

# Run training
python scripts/train_hydra.py `
    experiment=html_baseline `
    train.epochs=10 `
    model.freeze_bert=false `
    protocol=random `
    train.bs=32 `
    hardware.precision=16-mixed `
    run.name=html_baseline_10ep

Write-Host ""
Write-Host "Training completed with exit code: $LASTEXITCODE"

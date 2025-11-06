# HTML Baseline Training Script
# 10 epochs, freeze_bert=false, 全量数据

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "HTML 钓鱼检测基线训练" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 配置参数
$experiment = "html_baseline"
$epochs = 10
$freeze_bert = "false"
$protocol = "random"
$batch_size = 32
$precision = "16-mixed"

Write-Host "训练参数:" -ForegroundColor Yellow
Write-Host "  - Experiment: $experiment"
Write-Host "  - Epochs: $epochs"
Write-Host "  - Freeze BERT: $freeze_bert"
Write-Host "  - Protocol: $protocol"
Write-Host "  - Batch Size: $batch_size"
Write-Host "  - Precision: $precision"
Write-Host ""

# 检查数据
Write-Host "检查数据..." -ForegroundColor Yellow
if (Test-Path "data/processed/master_v2.csv") {
    $dataInfo = python -c "import pandas as pd; df=pd.read_csv('data/processed/master_v2.csv'); print(f'{len(df)} samples, Label={df[''label''].value_counts().to_dict()}')"
    Write-Host "  ✓ $dataInfo" -ForegroundColor Green
} else {
    Write-Host "  ✗ 数据文件不存在！" -ForegroundColor Red
    exit 1
}

# 检查依赖
Write-Host "检查依赖..." -ForegroundColor Yellow
Write-Host "  (跳过依赖检查)" -ForegroundColor Gray

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "开始训练..." -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 运行训练
python scripts/train_hydra.py `
    experiment=$experiment `
    train.epochs=$epochs `
    model.freeze_bert=$freeze_bert `
    protocol=$protocol `
    train.bs=$batch_size `
    hardware.precision=$precision `
    run.name="html_baseline_10ep"

# 检查训练结果
if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "训练完成！" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host ""

    # 查找最新的实验目录
    $latestExp = Get-ChildItem "experiments" -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1

    if ($latestExp) {
        Write-Host "实验目录: $($latestExp.FullName)" -ForegroundColor Cyan
        Write-Host ""

        # 检查工件
        $resultsDir = Join-Path $latestExp.FullName "results"
        if (Test-Path $resultsDir) {
            Write-Host "工件检查:" -ForegroundColor Yellow

            $artifacts = @(
                "metrics_$protocol.json",
                "roc_$protocol.png",
                "calib_$protocol.png",
                "splits_$protocol.csv"
            )

            foreach ($artifact in $artifacts) {
                $path = Join-Path $resultsDir $artifact
                if (Test-Path $path) {
                    Write-Host "  ✓ $artifact" -ForegroundColor Green
                } else {
                    Write-Host "  ✗ $artifact 缺失！" -ForegroundColor Red
                }
            }

            # 显示指标
            $metricsPath = Join-Path $resultsDir "metrics_$protocol.json"
            if (Test-Path $metricsPath) {
                Write-Host ""
                Write-Host "性能指标:" -ForegroundColor Yellow
                $metrics = Get-Content $metricsPath | ConvertFrom-Json
                Write-Host "  - AUROC: $($metrics.auroc)" -ForegroundColor Cyan
                Write-Host "  - Accuracy: $($metrics.accuracy)" -ForegroundColor Cyan
                Write-Host "  - F1-Macro: $($metrics.f1_macro)" -ForegroundColor Cyan
                Write-Host "  - NLL: $($metrics.nll)" -ForegroundColor Cyan
                Write-Host "  - ECE: $($metrics.ece)" -ForegroundColor Cyan

                # 检查指标是否符合要求
                $aurocValue = [double]$metrics.auroc
                if ($aurocValue -ge 0.70) {
                    Write-Host ""
                    Write-Host "✓ AUROC >= 0.70 达标！" -ForegroundColor Green
                } else {
                    Write-Host ""
                    Write-Host "✗ AUROC 未达标！" -ForegroundColor Red
                }
            }
        }
    }
} else {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Red
    Write-Host "训练失败！退出码: $LASTEXITCODE" -ForegroundColor Red
    Write-Host "========================================" -ForegroundColor Red
    exit 1
}

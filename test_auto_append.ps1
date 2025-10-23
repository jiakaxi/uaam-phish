# 测试自动追加功能的脚本
# 运行一个快速实验并启用文档自动追加

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "测试文档自动追加功能" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "📝 即将运行一个快速实验（1 epoch），并启用文档自动追加" -ForegroundColor Yellow
Write-Host ""
Write-Host "实验完成后，结果会追加到：" -ForegroundColor Yellow
Write-Host "  - FINAL_SUMMARY_CN.md（文档末尾）" -ForegroundColor Green
Write-Host ""

$confirm = Read-Host "是否继续？(y/N)"

if ($confirm -ne 'y') {
    Write-Host "已取消" -ForegroundColor Red
    exit
}

Write-Host ""
Write-Host "🚀 启动训练..." -ForegroundColor Cyan
Write-Host ""

# 运行训练
python scripts/train_hydra.py `
    logging.auto_append_docs=true `
    train.epochs=1 `
    +profiles/local

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "✅ 完成！" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "查看追加的内容：" -ForegroundColor Yellow
Write-Host "1. 打开 FINAL_SUMMARY_CN.md" -ForegroundColor White
Write-Host "2. 滚动到文档末尾" -ForegroundColor White
Write-Host "3. 查看最新追加的实验记录" -ForegroundColor White
Write-Host ""

$viewDoc = Read-Host "是否打开 FINAL_SUMMARY_CN.md？(y/N)"

if ($viewDoc -eq 'y') {
    notepad FINAL_SUMMARY_CN.md
}

Write-Host ""
Write-Host "💡 提示：" -ForegroundColor Cyan
Write-Host "  - 日常实验：不启用自动追加（默认）" -ForegroundColor White
Write-Host "  - 重要实验：启用 logging.auto_append_docs=true" -ForegroundColor White
Write-Host ""

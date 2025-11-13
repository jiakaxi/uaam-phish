# 腐败数据测试快速开始

## 📋 主腐败评测（L/M/H × 3 模态 = 9 个测试）

### 步骤 1：运行测试

```bash
python scripts/run_corrupt_tests.py \
  --experiment-dir experiments/s0_iid_earlyconcat_20251111_025612 \
  --test-type corrupt \
  --modalities url html img \
  --levels L M H
```

### 步骤 2：收集结果

```bash
python scripts/test_corrupt_data.py \
  --experiment-dir experiments/s0_iid_earlyconcat_20251111_025612 \
  --test-type corrupt \
  --modalities url html img \
  --levels L M H
```

## 📊 输出

结果将保存在 `experiments/corrupt_eval_<model_name>/` 目录下：

- `corrupt_metrics.csv`：所有指标的 CSV 汇总
- `corrupt_metrics.json`：所有指标的 JSON 汇总
- `auroc_vs_intensity.png`：AUROC vs 强度柱状图
- `reliability_comparison.png`：可靠性曲线对比

## 🔍 参数说明

- `--experiment-dir`：IID 训练目录（包含 checkpoints/best.ckpt）
- `--test-type`：`corrupt`（主腐败评测）或 `iid`（轻噪声）
- `--modalities`：`url html img`（默认：所有三模态）
- `--levels`：`L M H`（主腐败评测）或 `0.1 0.3 0.5`（IID 轻噪声）
- `--output-dir`：输出目录（可选，默认按模型分文件夹）

## 📝 注意事项

1. 确保 IID 训练的 checkpoint 存在于 `experiment-dir/checkpoints/` 中
2. 测试会自动跳过训练（`max_epochs=0`），直接加载 checkpoint 进行测试
3. 所有 9 个测试（3 模态 × 3 强度）都会自动运行

# 腐败数据测试完整示例

## 🎯 主腐败评测（L/M/H × 3 模态 = 9 个测试）

### 完整工作流程

#### 步骤 1：运行所有测试

```bash
python scripts/run_corrupt_tests.py \
  --experiment-dir experiments/s0_iid_earlyconcat_20251111_025612 \
  --test-type corrupt \
  --modalities url html img \
  --levels L M H
```

**说明**：
- 自动从 `experiment-dir` 查找 checkpoint
- 运行 9 个测试：URL/HTML/IMG × L/M/H
- 每个测试使用对应的 `test_corrupt_{mod}_{level}.csv`

**预期输出**：
```
==========================================
腐败数据批量测试 - 完整套件
==========================================
>> 实验目录: experiments/s0_iid_earlyconcat_20251111_025612
>> 实验配置: s0_iid_earlyconcat
>> Checkpoint: experiments/.../checkpoints/best-*.ckpt
>> 测试类型: corrupt
>> 模态: url, html, img
>> 强度级别: L, M, H
==========================================

>> 测试计划: 3 模态 × 3 强度 = 9 个测试
==========================================

[1/9] URL-L
  CSV: workspace/data/corrupt/url/test_corrupt_url_L.csv
  Checkpoint: experiments/.../checkpoints/best-*.ckpt
  ✓ 完成

[2/9] URL-M
  ...

[9/9] IMG-H
  ✓ 完成

==========================================
>> 所有 9 个测试完成！
==========================================
```

#### 步骤 2：收集结果并生成可视化

```bash
python scripts/test_corrupt_data.py \
  --experiment-dir experiments/s0_iid_earlyconcat_20251111_025612 \
  --test-type corrupt \
  --modalities url html img \
  --levels L M H
```

**说明**：
- 自动搜索所有腐败数据的预测结果
- 计算指标：AUROC、FPR@TPR95、ECE、Brier
- 生成可视化图表

**预期输出**：
```
==========================================
腐败数据测试结果收集
==========================================
>> 实验目录: experiments/s0_iid_earlyconcat_20251111_025612
>> 输出目录: experiments/corrupt_eval_s0
>> 测试类型: corrupt
>> 模态: url, html, img
>> 强度级别: L, M, H
==========================================

>> 搜索腐败数据预测结果...
>> 找到预测文件: ... (模态=url, 强度=L)
>> 找到预测文件: ... (模态=url, 强度=M)
...
>> 收集结果: URL-L - AUROC=0.xxxx, ECE=0.xxxx, FPR@TPR95=0.xxxx, Brier=0.xxxx
...

>> 保存指标结果: experiments/corrupt_eval_s0/corrupt_metrics.csv
>> 保存指标结果: experiments/corrupt_eval_s0/corrupt_metrics.json
>> 生成可视化...
>> 保存 AUROC vs 强度图: experiments/corrupt_eval_s0/auroc_vs_intensity.png
>> 保存可靠性曲线对比图: experiments/corrupt_eval_s0/reliability_comparison.png

>> 腐败数据测试结果已保存到: experiments/corrupt_eval_s0
```

## 📊 输出文件

```
experiments/corrupt_eval_s0/
├── corrupt_metrics.csv          # 所有指标的 CSV 汇总
├── corrupt_metrics.json         # 所有指标的 JSON 汇总
├── auroc_vs_intensity.png       # AUROC vs 强度柱状图（按模态分组）
└── reliability_comparison.png   # 可靠性曲线对比（IID vs H）
```

## 🔍 参数说明

### `run_corrupt_tests.py`

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--experiment-dir` | IID 训练目录（必需） | - |
| `--test-type` | `corrupt`（主腐败评测）或 `iid`（轻噪声） | `corrupt` |
| `--modalities` | 要测试的模态 | `url html img` |
| `--levels` | 强度级别 | `L M H`（corrupt）或 `0.1 0.3 0.5`（iid） |
| `--output-dir` | 输出目录（可选） | `experiments/corrupt_eval_<model_name>` |
| `--dry-run` | 只打印命令，不执行 | `False` |
| `--continue-on-error` | 遇到错误时继续运行 | `False` |

### `test_corrupt_data.py`

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--experiment-dir` | IID 训练目录（必需） | - |
| `--test-type` | `corrupt`（主腐败评测）或 `iid`（轻噪声） | `corrupt` |
| `--modalities` | 要处理的模态 | `url html img` |
| `--levels` | 强度级别（可选） | 根据 `test-type` 自动确定 |
| `--output-dir` | 输出目录（可选） | `experiments/corrupt_eval_<model_name>` |
| `--collect-only` | 只收集结果，不生成可视化 | `False` |

## ✅ 验证清单

运行前请确认：

- [ ] IID 训练目录存在且包含 `checkpoints/best-*.ckpt`
- [ ] 腐败数据 CSV 文件存在于 `workspace/data/corrupt/`
- [ ] 所有 9 个 CSV 文件都存在：
  - `workspace/data/corrupt/url/test_corrupt_url_{L,M,H}.csv`
  - `workspace/data/corrupt/html/test_corrupt_html_{L,M,H}.csv`
  - `workspace/data/corrupt/img/test_corrupt_img_{L,M,H}.csv`

## 🐛 常见问题

### 问题：找不到 checkpoint

**解决**：确保 `experiment-dir/checkpoints/` 或 `experiment-dir/lightning_logs/version_*/checkpoints/` 中存在 `.ckpt` 文件

### 问题：找不到预测结果

**解决**：
1. 确认已运行步骤 1 的测试
2. 检查实验目录中是否存在 `artifacts/predictions*.csv` 文件
3. 确认预测文件路径中包含 "corrupt" 关键字

### 问题：CSV 文件不存在

**解决**：检查 `workspace/data/corrupt/` 目录结构，确保所有 CSV 文件都已生成

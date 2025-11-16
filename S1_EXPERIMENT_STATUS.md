# S1实验训练状态跟踪

**更新时间**: 2025-11-12 16:22

## 📊 整体进度

- **已完成**: 0/6
- **进行中**: 1/6 (S1 IID seed=42)
- **待运行**: 5/6
- **预计总时长**: ~12小时

---

## 🔄 当前运行状态

### ✅ Phase 1-2: 配置验证与Smoke Test (已完成)

**已修复的问题**:
1. U-Module温度优化数值稳定性 (`src/modules/u_module.py`)
2. train_hydra.py max_epochs处理

**Smoke test结果** (1 epoch):
- AUROC: 0.9999
- ECE_post: 0.0820
- 所有artifacts正常生成 ✅

---

### 🟢 实验 1/6: S1 IID seed=42 (运行中)

**状态**: 🏃 运行中
**开始时间**: 2025-11-12 15:53
**实验目录**: `experiments/s1_iid_lateavg_20251112_155335/`

**训练进度**:
- 当前进度: Epoch 5/20 (截至 16:17)
- 每epoch耗时: ~3.8分钟
- **预计剩余时间**: ~53分钟 (14 epochs)
- **预计完成时间**: ~17:10

**当前指标** (Epoch 5):
- val/acc: 0.9983
- val/auroc: 1.0000
- val/loss: 0.1251

**日志文件**: `experiments/s1_iid_lateavg_20251112_155335/logs/train.log`

---

### ⏳ 待运行实验 (2-6/6)

#### 实验 2/6: S1 IID seed=43
- **命令**: `python scripts/train_hydra.py experiment=s1_iid_lateavg run.seed=43`
- **预计耗时**: ~2小时
- **状态**: 待运行

#### 实验 3/6: S1 IID seed=44
- **命令**: `python scripts/train_hydra.py experiment=s1_iid_lateavg run.seed=44`
- **预计耗时**: ~2小时
- **状态**: 待运行

#### 实验 4/6: S1 Brand-OOD seed=42
- **命令**: `python scripts/train_hydra.py experiment=s1_brandood_lateavg run.seed=42`
- **预计耗时**: ~2小时
- **状态**: 待运行

#### 实验 5/6: S1 Brand-OOD seed=43
- **命令**: `python scripts/train_hydra.py experiment=s1_brandood_lateavg run.seed=43`
- **预计耗时**: ~2小时
- **状态**: 待运行

#### 实验 6/6: S1 Brand-OOD seed=44
- **命令**: `python scripts/train_hydra.py experiment=s1_brandood_lateavg run.seed=44`
- **预计耗时**: ~2小时
- **状态**: 待运行

---

## 🚀 自动运行方案

### 方案A: 使用批处理脚本 (推荐)

**步骤**:
1. 等待第一个实验完成 (~17:10)
2. 验证完成:
   ```powershell
   Test-Path experiments\s1_iid_lateavg_20251112_155335\SUMMARY.md
   ```
3. 运行批处理脚本:
   ```powershell
   .\run_remaining_s1_experiments.bat
   ```

**优点**:
- 简单可靠
- 无编码问题
- 清晰的进度显示

### 方案B: 手动逐个运行

如果需要更精细的控制，可以逐个运行:

```bash
# 实验2
python scripts/train_hydra.py experiment=s1_iid_lateavg run.seed=43

# 实验3
python scripts/train_hydra.py experiment=s1_iid_lateavg run.seed=44

# 实验4
python scripts/train_hydra.py experiment=s1_brandood_lateavg run.seed=42

# 实验5
python scripts/train_hydra.py experiment=s1_brandood_lateavg run.seed=43

# 实验6
python scripts/train_hydra.py experiment=s1_brandood_lateavg run.seed=44
```

---

## 📁 实验输出结构

每个实验会生成以下artifacts:

```
experiments/s1_{protocol}_lateavg_YYYYMMDD_HHMMSS/
├── artifacts/
│   ├── calibration.json          # 温度校准参数
│   ├── predictions_test.csv       # 预测结果（包含r_url/r_html/r_img）
│   ├── reliability_before_ts_val.png
│   ├── reliability_post_test.png  # 校准前后对比
│   └── roc_curve_test.png
├── results/
│   ├── eval_summary.json          # 完整评估摘要
│   └── metrics_final.json         # 最终指标
├── checkpoints/
│   └── best-*.ckpt
├── SUMMARY.md                     # 包含RO1洞察
└── config.yaml
```

---

## 📈 Phase 4: 结果评估 (待所有训练完成后)

**任务**:
1. 提取所有6个实验的评估结果
2. 生成S0/S1组合总结表格
3. 对比分析

**脚本**:
```bash
# 提取结果
python scripts/evaluate_s0.py --runs_dir workspace/runs

# 生成汇总
python scripts/summarize_s0_results.py
```

**预期输出**:
- `workspace/runs/evaluation_results.json`
- `workspace/runs/evaluation_results.csv`
- `workspace/runs/s0_s1_summary.csv`
- `workspace/runs/s0_s1_summary.md`

---

## 🔍 监控命令

### 检查当前训练状态
```powershell
Get-Content experiments\s1_iid_lateavg_20251112_155335\logs\train.log -Tail 10
```

### 检查是否完成
```powershell
Test-Path experiments\s1_iid_lateavg_20251112_155335\SUMMARY.md
```

### 查看最终指标
```powershell
Get-Content experiments\s1_iid_lateavg_20251112_155335\results\metrics_final.json | ConvertFrom-Json | ConvertTo-Json -Depth 10
```

### 列出所有S1实验
```powershell
Get-ChildItem experiments\ -Directory -Filter "s1_*" | Sort-Object LastWriteTime
```

---

## ⚠️ 注意事项

1. **GPU内存**: 确保有足够的GPU内存（约8GB）
2. **磁盘空间**: 每个实验约需要5-10GB
3. **训练时长**: 如果GPU性能较低，单个实验可能需要更长时间
4. **WandB日志**: 确保WandB配置正确（或设置offline模式）

---

## 📝 更新日志

- **2025-11-12 15:53**: 启动实验1 (S1 IID seed=42)
- **2025-11-12 16:22**: 创建状态跟踪文档和批处理脚本



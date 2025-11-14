# S4 实验运行中

**启动时间**: 2025-11-14
**状态**: 🟢 **自动化运行中**

---

## 🚀 运行的实验

### 顺序执行计划

1. **S4 Brand-OOD** (10 epochs)
   - 配置: `s4_brandood_rcaf`
   - 测试: 分布外品牌泛化能力
   - 预计时间: ~30-40 分钟

2. **S4 IID** (10 epochs)
   - 配置: `s4_iid_rcaf`
   - 测试: 独立同分布性能
   - 预计时间: ~40-50 分钟

**总预计时间**: 70-90 分钟

---

## 📊 修复内容回顾

### 已解决的问题

1. ✅ **Metadata 注册**: C-Module 成功加载 16,000 条记录
2. ✅ **NaN 处理**: r_m 和 c_m 的 fallback 机制
3. ✅ **配置修复**: Brand-OOD 使用正确的 test_id_cached.csv

### 修复效果

| 指标 | 修复前 | 修复后 |
|------|--------|--------|
| 警告次数 | ~300次/epoch | **0次** |
| 有效模态 | 0/3 | ≥2/3 |
| C-Module | 失败 | 正常工作 |

---

## 📂 输出目录

实验结果将保存在:
```
outputs/2025-11-14/HH-MM-SS/
├── train_hydra.log          # 训练日志
├── s4_iid_rcaf/
│   └── version_0/
│       ├── metrics.csv       # 训练曲线
│       ├── hparams.yaml      # 超参数
│       └── checkpoints/      # 模型检查点
├── s4_lambda_stats.json     # Lambda_c 统计 (按场景)
├── s4_per_sample.csv        # 每样本的权重
├── SUMMARY.md               # 实验总结
└── results/
    ├── eval_summary.json    # 评估指标
    ├── roc_*.png            # ROC 曲线
    └── calib_*.png          # 校准图
```

---

## 🔍 监控命令

### 检查当前进度

```powershell
# 查看最新实验目录
Get-ChildItem outputs\2025-11-14 -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1

# 查看训练日志 (最后 20 行)
Get-Content outputs\2025-11-14\<timestamp>\train_hydra.log -Tail 20

# 查看指标
Get-Content outputs\2025-11-14\<timestamp>\s4_*\version_0\metrics.csv
```

### 检查进程状态

```powershell
# 查看 Python 进程
Get-Process python | Where-Object {$_.WS -gt 500MB} | Select-Object Id, CPU, @{Name='Memory(GB)';Expression={[math]::Round($_.WS/1GB,2)}}
```

### 实时监控日志

```powershell
# Windows PowerShell
Get-Content outputs\2025-11-14\<timestamp>\train_hydra.log -Wait -Tail 10
```

---

## ⚙️ 关键配置

### S4 Brand-OOD

```yaml
protocol: presplit
system:
  _target_: src.systems.s4_rcaf_system.S4RCAFSystem

fusion:
  hidden_dim: 16           # Lambda gate 隐藏层
  temperature: 2.0         # Softmax temperature (gamma)
  warmup_epochs: 5         # 前5个epoch固定权重 (可选)
  lambda_regularization: 0.01  # L2正则化

optimizer:
  encoder_lr: 1.0e-4       # 编码器学习率
  fusion_lr: 1.0e-3        # 融合模块学习率 (更高)

modules:
  use_umodule: true        # 启用 U-Module (可靠性)
  use_cmodule: true        # 启用 C-Module (一致性)

train:
  epochs: 10
  bs: 32
```

### S4 IID

相同配置，但使用 IID 数据划分。

---

## 📈 预期结果

### Lambda_c 统计 (训练结束时)

**成功标准**:
- `lambda_c_mean`: [0.2, 0.8]
- `lambda_c_std`: > 0.05 (证明自适应性)
- 不同场景有差异

**示例**:
```json
{
  "clean": {
    "lambda_c": {"mean": 0.45, "std": 0.12},
    "alpha_m": {
      "url": {"mean": 0.35},
      "html": {"mean": 0.40},
      "visual": {"mean": 0.25}
    }
  }
}
```

### 性能指标 (vs S0 Baseline)

**目标提升**:
- IID AUROC: ≥ +1.5%
- Brand-OOD F1: ≥ +45 pp
- Corruption AUROC: ≥ +8%

---

## 🎯 完成后的工作

### 1. 结果验证 (5 分钟)

```bash
# 检查输出文件
ls outputs/2025-11-14/*/s4_lambda_stats.json
ls outputs/2025-11-14/*/s4_per_sample.csv

# 查看 lambda_c 统计
cat outputs/2025-11-14/*/s4_lambda_stats.json | jq .

# 检查警告次数 (应该 = 0)
grep "Some samples have no valid modalities" outputs/2025-11-14/*/train_hydra.log | wc -l
```

### 2. 提取关键指标 (10 分钟)

创建分析脚本:
```python
# scripts/analyze_s4_results.py
import json
import pandas as pd

# 读取 lambda_stats
with open("outputs/.../s4_lambda_stats.json") as f:
    stats = json.load(f)

# 提取指标
for scenario, data in stats.items():
    print(f"{scenario}:")
    print(f"  lambda_c_mean: {data['lambda_c']['mean']:.3f}")
    print(f"  lambda_c_std: {data['lambda_c']['std']:.3f}")

# 读取 per_sample 数据
df = pd.read_csv("outputs/.../s4_per_sample.csv")
print(f"\nLambda_c range: [{df['lambda_c_url'].min():.3f}, {df['lambda_c_url'].max():.3f}]")
```

### 3. 对比 S3 vs S4 (20 分钟)

- S3: 固定 lambda_c (超参数)
- S4: 学习 lambda_c (自适应)

**关键对比点**:
1. Lambda_c 方差 (S4 应该 > S3)
2. 场景适应能力 (S4 在 OOD/Corruption 下更好)
3. 性能提升

### 4. 生成论文图表 (30 分钟)

需要的图表:
1. **Lambda_c 分布图** (boxplot by scenario)
2. **融合权重变化** (heatmap: scenario × modality)
3. **性能对比** (bar chart: S0 vs S3 vs S4)
4. **视觉模态抑制曲线** (line: corruption_level → alpha_visual)

---

## 🚨 故障排除

### 如果实验失败

**检查点**:
1. 查看最后的错误日志
2. 检查 GPU 内存是否耗尽
3. 验证数据文件存在

**重新运行**:
```bash
# 单独运行失败的实验
python scripts/train_hydra.py experiment=s4_brandood_rcaf train.epochs=10

# 或使用 CPU (如果 GPU OOM)
python scripts/train_hydra.py experiment=s4_brandood_rcaf train.epochs=10 hardware.accelerator=cpu
```

### 如果出现 NaN 警告

虽然已修复，但如果再次出现:
1. 检查 r_m 和 c_m 的值
2. 验证 NaN fallback 是否生效
3. 查看 C-Module 的 metadata 加载

---

## 📞 实时支持

**查看状态**:
- 进程是否运行？`Get-Process python`
- 日志是否更新？`Get-Item outputs/.../train_hydra.log | Select-Object LastWriteTime`
- GPU 是否被占用？`nvidia-smi`

**中断实验**:
```powershell
# 停止所有 Python 训练进程
Get-Process python | Where-Object {$_.WS -gt 500MB} | Stop-Process -Force
```

---

**当前状态**: 🟢 **运行中**
**预计完成**: ~90 分钟后
**监控**: 使用上述命令实时查看

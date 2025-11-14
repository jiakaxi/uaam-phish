# S4 RCAF Full - 使用指南

## 概述

S4 RCAF Full 系统实现了自适应融合机制，使用学习型 λ_c（而非 S3 的固定权重）来动态平衡可靠性和一致性分数。

**关键特性**:
- **自适应 λ_c**: 每个样本有不同的 λ_c 权重（通过 LambdaGate 学习）
- **端到端训练**: 全流程使用 adaptive fusion，确保梯度流向 lambda gate
- **场景感知**: 支持按 scenario（clean/light/medium/heavy/brandood）分析

## 快速开始

### 1. 运行单个实验

```bash
# IID 实验
python scripts/train_hydra.py experiment=s4_iid_rcaf

# Brand-OOD 实验
python scripts/train_hydra.py experiment=s4_brandood_rcaf

# Corruption 实验（指定 level）
python scripts/train_hydra.py experiment=s4_corruption_rcaf corruption_level=light
```

### 2. 超参数扫描（gamma）

```bash
# 扫描 temperature (gamma) 参数
bash scripts/run_s4_sweep.sh

# 或手动运行
python scripts/train_hydra.py \
    experiment=s4_iid_rcaf \
    -m \
    system.fusion.temperature=1.0,2.0,3.0,5.0
```

### 3. 运行测试

```bash
# Scenario 标签测试
pytest tests/test_datamodule_scenario.py -v

# S4 自适应融合测试
pytest tests/test_s4_adaptive.py -v
```

## 配置说明

### 系统配置 (`configs/system/s4_rcaf.yaml`)

```yaml
# Fusion 配置
fusion:
  hidden_dim: 16                  # Lambda gate 隐藏层维度
  temperature: 2.0                # Temperature scaling (gamma)
  warmup_epochs: 5                # 可选：前N个epoch固定λ_c=0.5
  lambda_regularization: 0.01     # L2 正则化（仅针对 lambda_gate）

# Optimizer 配置
optimizer:
  encoder_lr: 1.0e-4   # 编码器学习率
  fusion_lr: 1.0e-3    # Fusion 学习率（可以设置更高）
```

### 实验配置

- `s4_iid_rcaf.yaml`: IID 场景
- `s4_brandood_rcaf.yaml`: Brand-OOD 场景
- `s4_corruption_rcaf.yaml`: Corruption 鲁棒性测试

## 输出文件

训练完成后，会在日志目录生成：

1. **`s4_lambda_stats.json`**: 按 scenario 分组的统计量
   ```json
   {
     "clean": {
       "lambda_c": {"mean": 0.52, "std": 0.08, ...},
       "alpha_m": {"url": {...}, "html": {...}, "visual": {...}}
     },
     "heavy": {...},
     "brandood": {...}
   }
   ```

2. **`s4_per_sample.csv`**: 每个样本的详细数据
   ```csv
   sample_id,scenario,lambda_c_url,lambda_c_html,lambda_c_visual,alpha_url,alpha_html,alpha_visual,pred,label
   ```

## 分析脚本（TODO）

以下脚本需要在运行实验后创建：

```bash
# λ_c 分布和方差分析
python scripts/analyze_s4_adaptivity.py --run_dir workspace/runs/s4_iid_rcaf/

# 视觉模态抑制率分析
python scripts/plot_s4_suppression.py --run_dir workspace/runs/s4_iid_rcaf/

# S3 vs S4 性能对比
python scripts/compare_s3_s4.py \
    --s3_run workspace/runs/s3_iid_fixed/ \
    --s4_run workspace/runs/s4_iid_rcaf/
```

## 成功标准

### 性能指标（vs S0 基线）
1. IID AUROC: ≥ +1.5%
2. Brand-OOD F1: ≥ +45 pp
3. Heavy Corruption AUROC: ≥ +8%

### 自适应行为（证明不是 S3）
1. **λ_c 跨场景变化显著**:
   - `lambda_c/std > 0.15`
   - Clean vs Heavy 差异 > 0.2
   - Clean vs OOD 差异 > 0.15

2. **视觉模态抑制**:
   - Heavy corruption: ≥ 40% 权重下降（vs clean）
   - URL/HTML: < 15% 权重下降

3. **与 S3 协同优势**:
   - S4 OOD F1 > S3 OOD F1 + 5 pp
   - S4 Corruption AUROC > S3 Corruption AUROC + 1%

## 训练监控

### 关键指标

```python
# 训练时监控
train/lambda_c_mean    # 应在 [0.2, 0.8] 范围内
train/lambda_c_std     # 应 > 0.05（证明非常量）
train/cls_loss         # 分类损失
train/reg_loss         # 正则化损失
train/auroc            # 训练 AUROC

# 验证时监控
val/lambda_c_mean
val/lambda_c_std
val/auroc              # 用于选择最佳 gamma
```

### Sanity Checks

如果出现以下警告，说明训练可能有问题：

```
⚠️ Lambda_c collapsed! std=0.03 < 0.05
⚠️ Lambda_c mean=0.15 out of range [0.2, 0.8]
```

**解决方案**:
- 增加 `lambda_regularization`
- 降低 `fusion_lr`
- 检查数据是否有问题

## S3 vs S4 对比

| 特性 | S3 (Fixed Fusion) | S4 (Adaptive Fusion) |
|------|------------------|---------------------|
| λ_c | 固定超参数 (e.g., 0.5) | 学习网络输出 |
| 每样本不同? | ✗ 否（所有样本相同）| ✓ 是（每样本自适应）|
| 训练方式 | LateAvg（仅编码器）| Adaptive fusion（全流程）|
| 超参数调优 | 网格搜索 λ_c + γ | 仅搜索 γ |
| 场景适应性 | 无 | 自动适应 |

**λ_c 的方差是证明 S4 "自适应"的关键证据！**

## 故障排查

### 问题: Lambda_c collapse（std 很小）

**原因**: Lambda gate 学习到了常量函数，所有样本输出相同

**解决**:
1. 增加正则化: `fusion.lambda_regularization: 0.05`
2. 使用 warmup: `fusion.warmup_epochs: 10`
3. 降低 fusion learning rate: `optimizer.fusion_lr: 5.0e-4`

### 问题: 训练不收敛

**原因**: Learning rate 太高或数据问题

**解决**:
1. 降低学习率: `train.lr: 5.0e-5`
2. 添加 gradient clipping（已默认启用）
3. 检查数据质量

### 问题: 性能不如 S3

**原因**: Gamma 参数未调优或训练不足

**解决**:
1. 运行完整的 gamma sweep
2. 增加训练 epochs: `train.epochs: 80`
3. 检查 lambda_c 是否真的在变化（看 std）

## 参考

- 计划文档: `s4-adaptive-fusion.plan.md`
- 变更记录: `CHANGES_SUMMARY.md`
- 测试: `tests/test_s4_adaptive.py`

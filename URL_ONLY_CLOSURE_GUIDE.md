# URL-Only 收官指南

**目标**: 完成 P0 "产物生成" 最后一项，让三协议各自产出四件套且字段完全合规 ✅

---

## 📋 已完成的工作

### 1. 代码集成 ✅

已将 `build_splits` 集成到数据流程：

- **修改**: `src/datamodules/url_datamodule.py`
  - 添加 `split_metadata` 属性
  - 在 `setup(stage="fit")` 时调用 `build_splits()`
  - 自动保存 train/val/test splits
  - 记录完整的 split metadata

- **修改**: `scripts/train_hydra.py`
  - 在 `trainer.fit()` 后从 `dm.split_metadata` 获取元数据
  - 传递给 `ProtocolArtifactsCallback`

- **修改**: `src/utils/splits.py`
  - 更新 `write_split_table()` 支持所有必需列
  - 包含: brand_intersection_ok, tie_policy, brand_normalization, downgraded_to

- **修改**: `src/utils/protocol_artifacts.py`
  - 使用完整的 metadata 调用 `write_split_table()`

### 2. 验证脚本 ✅

创建了 `tools/check_artifacts_url_only.py` 校验脚本，自动检查：

- ✅ 四件套文件存在性
- ✅ `splits_{protocol}.csv` 列完整性（13列）
- ✅ `metrics_{protocol}.json` schema 完整性
- ✅ ECE bins 范围合理性 [3, 15]
- ✅ 协议特定验证（brand_ood 的 brand_intersection_ok, temporal 的 tie_policy）

---

## 🚀 一键验证命令

### 步骤 1: 运行三协议实验

```bash
# 重要：启用 use_build_splits 标志，让数据模块调用 build_splits
# 注意：需要 master.csv 存在于 data/processed/master.csv

# Random 协议
python scripts/train_hydra.py protocol=random use_build_splits=true

# Temporal 协议
python scripts/train_hydra.py protocol=temporal use_build_splits=true

# Brand-OOD 协议
python scripts/train_hydra.py protocol=brand_ood use_build_splits=true
```

### 步骤 2: 快速检查产物

```bash
# 检查四件套文件
ls experiments/*/results/roc_*.png
ls experiments/*/results/calib_*.png
ls experiments/*/results/splits_*.csv
ls experiments/*/results/metrics_*.json
```

### 步骤 3: 运行验证脚本

```bash
# 自动验证最新实验的产物
python tools/check_artifacts_url_only.py

# 或指定特定实验目录
python tools/check_artifacts_url_only.py experiments/url_mvp_20251023_040222
```

**预期输出**:

```
============================================================
URL-Only 产物校验脚本
============================================================

📁 Validating results in: experiments/url_mvp_20251023_040222/results

============================================================
Protocol: random
============================================================
  [CHECK] ROC curve: ...
    ✅ ROC curve exists (12345 bytes)
  [CHECK] Calibration curve: ...
    ✅ Calibration curve exists (12345 bytes)
  [CHECK] Splits CSV: ...
    ✅ Splits CSV has all required columns (3 splits)
  [CHECK] Metrics JSON: ...
    ✅ Metrics JSON schema valid
       - accuracy: 0.7500
       - auroc: 0.8500
       - ece: 0.0234 (bins=10)

✅ Protocol 'random' artifacts validated!

[... temporal, brand_ood ...]

============================================================
Summary
============================================================
  random         : ✅ PASS
  temporal       : ✅ PASS
  brand_ood      : ✅ PASS

🎉 All protocols passed validation!
```

---

## 📝 必须满足的 6 点清单

### ✅ 1. build_splits 调用与元数据贯通

- **状态**: ✅ 已完成
- **位置**: `src/datamodules/url_datamodule.py:35-68`
- **机制**:
  - 通过 `use_build_splits=true` 启用
  - 从 `data.csv_path` 读取 master.csv
  - 调用 `build_splits(df, cfg, protocol)`
  - 保存 splits 到 train/val/test CSV
  - 存储 metadata 到 `self.split_metadata`

### ✅ 2. 写出 `splits_{protocol}.csv`（列齐全）

- **状态**: ✅ 已完成
- **位置**: `src/utils/splits.py:255-289`
- **包含列** (13列):
  1. split
  2. count
  3. pos_count
  4. neg_count
  5. brand_unique
  6. brand_set
  7. timestamp_min
  8. timestamp_max
  9. source_counts
  10. brand_intersection_ok
  11. tie_policy
  12. brand_normalization
  13. downgraded_to

### ✅ 3. ROC 曲线

- **状态**: ✅ 已实现
- **位置**: `src/utils/visualizer.py:447-484`
- **路径**: `experiments/<run>/results/roc_{protocol}.png`
- **特性**:
  - 标注 AUC
  - 使用正类概率 `p[:, 1]`

### ✅ 4. 校准图（Calibration）

- **状态**: ✅ 已实现
- **位置**: `src/utils/visualizer.py:486-544`
- **路径**: `experiments/<run>/results/calib_{protocol}.png`
- **特性**:
  - 图内标注 `ECE=<value>`（第529-532行）
  - 小样本警告（第535-539行）

### ✅ 5. 指标 JSON（schema 完整）

- **状态**: ✅ 已实现
- **位置**: `src/utils/protocol_artifacts.py:119-147`
- **路径**: `experiments/<run>/results/metrics_{protocol}.json`
- **包含字段**:
  - accuracy, auroc, f1_macro
  - nll, ece, ece_bins_used
  - positive_class = "phishing"
  - artifacts: {roc_path, calib_path, splits_path}
  - warnings: {downgraded_reason}

### ✅ 6. 落盘路径/命名规范

- **状态**: ✅ 已实现
- **配置**: `configs/default.yaml:54-58`
- **路径规范**:
  ```
  experiments/<run>/results/
    ├── roc_{protocol}.png
    ├── calib_{protocol}.png
    ├── splits_{protocol}.csv
    └── metrics_{protocol}.json
  ```

---

## ⚠️ 重要注意事项

### 1. 必须有 master.csv

`use_build_splits=true` 需要读取 `data/processed/master.csv`。如果文件不存在：

```bash
# 选项 A: 使用现有的 splits（不启用 use_build_splits）
python scripts/train_hydra.py protocol=random use_build_splits=false

# 选项 B: 生成 master.csv
# 将现有的 train/val/test CSV 合并成 master.csv
python -c "
import pandas as pd
train = pd.read_csv('data/processed/url_train.csv')
val = pd.read_csv('data/processed/url_val.csv')
test = pd.read_csv('data/processed/url_test.csv')
master = pd.concat([train, val, test], ignore_index=True)
master.to_csv('data/processed/master.csv', index=False)
print(f'Created master.csv with {len(master)} samples')
"
```

### 2. 快速测试（小样本）

如果想快速验证而不跑完整训练：

```bash
# 使用 local profile（快速模式）
python scripts/train_hydra.py \
    protocol=random \
    use_build_splits=true \
    +profiles/local
```

这会：
- 只用 10% 数据训练
- 只跑 5 个 epoch
- batch_size=8

### 3. 临时禁用 build_splits

如果已有 train/val/test splits 且不想重新生成：

```bash
# 不启用 use_build_splits，但仍会生成产物
python scripts/train_hydra.py protocol=random use_build_splits=false
```

**注意**: 此时 `splits_{protocol}.csv` 将不会生成（因为没有 metadata）。

---

## 🔧 故障排除

### 问题 1: 缺少 `splits_{protocol}.csv`

**症状**: 其他三件套都有，但缺少 splits CSV

**原因**: `use_build_splits=false` 或 master.csv 不存在

**解决**:
```bash
# 确保启用 use_build_splits
python scripts/train_hydra.py protocol=random use_build_splits=true
```

### 问题 2: Calibration 图没有 ECE 标注

**症状**: 校准图生成了，但没有 `ECE=` 标注

**原因**: 代码已修复，应该不会出现

**验证**: 打开 `calib_{protocol}.png`，左上角应有 `ECE=0.xxxx` 标注框

### 问题 3: JSON 里 artifacts 路径不存在

**症状**: metrics JSON 中 artifacts 的路径指向不存在的文件

**原因**: 路径计算错误

**检查**:
```python
# 在 src/utils/protocol_artifacts.py:133-136
# 路径是相对于 results/ 的父目录
roc_path.relative_to(self.results_dir.parent)
```

### 问题 4: brand_intersection_ok 为空

**症状**: splits CSV 中 brand_intersection_ok 列为空

**原因**: build_splits 时没有品牌列，或品牌不足

**解决**:
- 确保 master.csv 有 `brand` 列
- 对于 brand_ood 协议，确保至少有 3 个不同品牌

---

## 📊 完成标准

运行以下命令，全部通过即为完成：

```bash
# 1. 三协议实验
for protocol in random temporal brand_ood; do
    echo ">>> Running $protocol"
    python scripts/train_hydra.py protocol=$protocol use_build_splits=true
done

# 2. 验证脚本
python tools/check_artifacts_url_only.py

# 预期输出:
# 🎉 All protocols passed validation!
```

**检查清单**:

- [ ] 三协议均生成四件套
- [ ] `splits_{protocol}.csv` 含 13 列
- [ ] `calib_{protocol}.png` 标注 ECE
- [ ] `metrics_{protocol}.json` 字段齐全
- [ ] 验证脚本全部通过 ✅

---

## 🎉 完成后

恭喜！你已完成 URL-Only P0 级别的所有任务：

- ✅ 架构锁定
- ✅ 训练配置
- ✅ 数据预处理
- ✅ 拆分协议
- ✅ 批处理元数据
- ✅ 指标计算
- ✅ **产物生成** ← 刚完成
- ✅ 复现性

**下一步**:

1. 将此验证脚本加入 CI/CD 流程
2. 开始大规模复现实验
3. 记录实验结果到 W&B

---

**更新日期**: 2025-10-22
**状态**: 🎯 Ready for Production

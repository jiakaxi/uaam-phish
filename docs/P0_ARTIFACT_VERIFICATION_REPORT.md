# P0 工件生成验证报告

**日期**: 2025-11-05
**测试人员**: AI Assistant
**测试环境**: Windows 10, Python 3.x, CPU
**测试实验**: `experiments/p0_smoke_20251105_232726`

---

## 📋 执行摘要

**目标**: 验证训练结束后自动生成所有必需的工件（四件套）

**结果**: ✅ **通过**

所有四件套工件成功生成，格式符合规范，无异常报错。

---

## 🎯 测试目标

验证训练结束会自动在 `experiments/<run>/results/` 下生成：

1. `roc_*.png` - ROC 曲线图
2. `calib_*.png` - 校准曲线图
3. `splits_*.csv` - 数据分割统计表（13 列）
4. `metrics_*.json` - 指标 JSON

### 合格标准（DoD）

- [x] `experiments/p0_smoke/` 下能看到四件套工件
- [x] `metrics_*.json` 里至少有 Accuracy / AUROC
- [x] 无异常报错
- [x] 日志记录了 splits 元数据

---

## 🧪 测试执行

### 测试配置

创建了 P0 烟雾测试配置 `configs/experiment/p0_smoke.yaml`:

```yaml
run:
  name: p0_smoke
  seed: 42

protocol: random
use_build_splits: true

train:
  epochs: 2  # 仅2轮，快速完成
  bs: 32
  lr: 0.0001

hardware:
  accelerator: cpu
  devices: 1
  precision: 32
```

### 运行命令

```bash
python scripts/train_hydra.py +experiment=p0_smoke
```

### 验证命令

```bash
python tools/check_artifacts_url_only.py experiments/p0_smoke_20251105_232726
```

---

## 📊 测试结果

### 工件生成情况

| 工件文件 | 状态 | 文件大小 | 备注 |
|---------|------|----------|------|
| `roc_random.png` | ✅ 通过 | 124,347 bytes | 包含 AUC 标注 |
| `calib_random.png` | ✅ 通过 | 133,530 bytes | 包含 ECE 标注 |
| `splits_random.csv` | ✅ 通过 | 1,423 bytes | 13 列，3 个分割 |
| `metrics_random.json` | ✅ 通过 | 334 bytes | 完整 schema |
| `implementation_report.md` | ✅ 通过 | 3,456 bytes | 自动生成的实现报告 |

### Metrics JSON 内容

```json
{
  "accuracy": 0.5098039507865906,
  "auroc": 0.6133645176887512,
  "f1_macro": 0.6685003638267517,
  "nll": 0.6922833919525146,
  "ece": 0.011598973535001278,
  "ece_bins_used": 10,
  "positive_class": "phishing",
  "artifacts": {
    "roc_path": "results\\roc_random.png",
    "calib_path": "results\\calib_random.png",
    "splits_path": "results\\splits_random.csv"
  },
  "warnings": {
    "downgraded_reason": null
  }
}
```

✅ **验证点**:
- ✅ 包含所有必需的顶层字段（9个）
- ✅ `accuracy` 和 `auroc` 正确记录
- ✅ `ece_bins_used` 在合理范围 [3, 15]
- ✅ `positive_class` 为 "phishing"
- ✅ `artifacts` 路径正确

### Splits CSV 内容

**列（13 列）**:
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

**数据行（3 行）**:

| split | count | pos_count | neg_count | brand_unique |
|-------|-------|-----------|-----------|--------------|
| train | 469   | 241       | 228       | 265          |
| val   | 100   | 61        | 39        | 78           |
| test  | 102   | 52        | 50        | 85           |

✅ **验证点**:
- ✅ 所有 13 列都存在
- ✅ 数据统计合理
- ✅ `brand_intersection_ok` 为 "false"（符合 random 协议）

### 训练日志摘要

关键日志输出：

```
>> Building splits from data\processed\master_v2.csv using protocol 'random'
>> Splits saved: train=469, val=100, test=102

>> Updated protocol_callback with split_metadata:
   ['protocol', 'downgraded_to', 'downgrade_reason', 'tie_policy',
    'brand_normalization', 'split_stats', 'brand_intersection_ok']

>> Generating artifacts for protocol 'random'...
[SUCCESS] ROC curve saved: experiments\p0_smoke_...\results\roc_random.png
[SUCCESS] Calibration curve saved: experiments\p0_smoke_...\results\calib_random.png
[SUCCESS] Split table saved: experiments\p0_smoke_...\results\splits_random.csv
>> Metrics saved: experiments\p0_smoke_...\results\metrics_random.json
>> Implementation report saved: experiments\p0_smoke_...\results\implementation_report.md
>> All artifacts saved to: experiments\p0_smoke_...\results
```

✅ 无异常报错，所有工件成功生成。

---

## 🐛 发现并修复的问题

### 问题 1: `'bool' object is not subscriptable`

**现象**: 第一次测试时，`splits_random.csv` 生成失败，报错：

```
[WARNING] Failed to save split table: 'bool' object is not subscriptable
```

**原因**:
- `build_splits()` 返回的 `metadata["brand_intersection_ok"]` 是 `bool` 类型
- `write_split_table()` 期望所有字段都是字符串

**修复**:
在 `src/utils/protocol_artifacts.py` 中添加类型转换：

```python
# Convert bool to str for brand_intersection_ok
brand_inter = self.split_metadata.get("brand_intersection_ok", "")
if isinstance(brand_inter, bool):
    brand_inter = "true" if brand_inter else "false"

metadata_for_csv = {
    # ...
    "brand_intersection_ok": brand_inter,
}
```

**验证**: 修复后重新运行，`splits_random.csv` 成功生成 ✅

### 问题 2: metadata 结构不一致

**现象**: `brand_intersection_ok` 被放在 `split_stats` 字典中

**修复**:
将其移至 `metadata` 顶层，在 `src/utils/splits.py` 中：

```python
# Store as bool in metadata (not in split_stats)
metadata["brand_intersection_ok"] = (
    len(train_brands & test_brands) == 0
)
```

**验证**: 修复后 metadata 传递正确 ✅

---

## 🎓 结论

### ✅ 测试通过

P0 工件生成功能**完全符合预期**：

1. ✅ 训练结束后自动生成所有四件套工件
2. ✅ `metrics_*.json` 包含完整的指标数据
3. ✅ `splits_*.csv` 包含所有 13 列元数据
4. ✅ 图像工件（ROC、Calibration）正确生成
5. ✅ 无异常报错
6. ✅ 日志清晰记录了整个过程

### 📝 建议

1. **字体警告**: 图表生成时有中文字体警告（Glyph missing），建议配置中文字体（可选）
2. **协议扩展**: 当前仅测试 `random` 协议，建议后续验证 `temporal` 和 `brand_ood`
3. **自动化**: 可以将此验证流程集成到 CI/CD

### 🚀 后续步骤

- [ ] 验证 `temporal` 协议的工件生成
- [ ] 验证 `brand_ood` 协议的工件生成
- [ ] 验证协议降级场景（downgrade）
- [ ] 编写单元测试覆盖工件生成逻辑

---

## 📎 附件

- 测试实验目录: `experiments/p0_smoke_20251105_232726/`
- 配置文件: `configs/experiment/p0_smoke.yaml`
- 验证脚本: `tools/check_artifacts_url_only.py`
- 日志输出: （已包含在本报告中）

---

**报告生成时间**: 2025-11-05 23:30:00
**签名**: AI Assistant

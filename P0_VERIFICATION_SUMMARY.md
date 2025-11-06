# P0 工件生成验证摘要

**日期**: 2025-11-05
**状态**: ✅ **通过**

---

## 验证目标

确认训练结束会自动在 `experiments/<run>/` 下生成：

1. `roc_*.png` - ROC 曲线图
2. `calib_*.png` - 校准曲线图
3. `splits_*.csv` - 数据分割统计表
4. `metrics_*.json` - 指标 JSON

## DoD 检查清单

- [x] `experiments/p0_smoke/` 下能看到四件套工件
- [x] `metrics_*.json` 里至少有 Accuracy / AUROC
- [x] 无异常报错
- [x] 日志记录了 splits 元数据

## 测试结果

**测试实验**: `experiments/p0_smoke_20251105_232726`

### 工件生成情况

| 工件 | 状态 | 详情 |
|------|------|------|
| `roc_random.png` | ✅ 通过 | 124KB, AUC 标注正确 |
| `calib_random.png` | ✅ 通过 | 133KB, ECE 标注正确 |
| `splits_random.csv` | ✅ 通过 | 13 列完整，3 个分割 |
| `metrics_random.json` | ✅ 通过 | 完整 schema |

### 指标数据

```json
{
  "accuracy": 0.5098,
  "auroc": 0.6134,
  "f1_macro": 0.6685,
  "nll": 0.6923,
  "ece": 0.0116,
  "ece_bins_used": 10,
  "positive_class": "phishing"
}
```

### 数据分割统计

| split | count | pos_count | neg_count | brand_unique |
|-------|-------|-----------|-----------|--------------|
| train | 469   | 241       | 228       | 265          |
| val   | 100   | 61        | 39        | 78           |
| test  | 102   | 52        | 50        | 85           |

## 修复内容

### 1. 修复 `brand_intersection_ok` 类型错误

**问题**: `'bool' object is not subscriptable`

**原因**: `build_splits()` 返回的 `metadata["brand_intersection_ok"]` 是 bool 类型

**修复**: 在 `protocol_artifacts.py` 中添加类型转换

```python
brand_inter = self.split_metadata.get("brand_intersection_ok", "")
if isinstance(brand_inter, bool):
    brand_inter = "true" if brand_inter else "false"
```

### 2. 修正 metadata 结构

将 `brand_intersection_ok` 从 `split_stats` 移至 `metadata` 顶层

## 运行命令

```bash
# 创建 P0 烟雾测试
python scripts/train_hydra.py +experiment=p0_smoke

# 验证工件
python tools/check_artifacts_url_only.py experiments/p0_smoke_20251105_232726

# 结果
[SUCCESS] Protocol 'random' artifacts validated!
```

## 结论

✅ **P0 工件生成功能验证通过！**

所有必需工件成功生成，格式符合规范，无异常报错。

详细报告请参见: `docs/P0_ARTIFACT_VERIFICATION_REPORT.md`

---

**签名**: AI Assistant
**时间**: 2025-11-05 23:30:00

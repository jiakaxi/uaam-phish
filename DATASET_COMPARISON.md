# 数据集文件对比分析

**分析日期**: 2025-11-08
**对比文件**: `master_v2.csv` vs `master_v2_backup.csv`

---

## 📊 文件基本信息对比

| 特性 | master_v2.csv | master_v2_backup.csv |
|------|--------------|---------------------|
| **行数** | 16,000 行 | 16,656 行 |
| **列数** | 18 列 | 17 列 |
| **差异** | - | 多 656 行，少 1 列 |

---

## 🔍 详细对比

### 1. 列差异

**master_v2.csv 独有的列**:
- `timestamp_original` - 原始时间戳字段

**master_v2_backup.csv**:
- 不包含 `timestamp_original` 列

**共同列** (17个):
- `id`, `stem`, `label`, `url_text`, `html_path`, `img_path`
- `domain`, `source`, `split`, `brand_raw`, `brand`
- `timestamp`, `domain_source`, `timestamp_source`
- `folder`, `html_sha1`, `img_sha1`

---

### 2. 标签分布对比

| 文件 | 正样本 (label=1) | 负样本 (label=0) | 总计 |
|------|----------------|----------------|------|
| **master_v2.csv** | 8,000 (50.0%) | 8,000 (50.0%) | 16,000 |
| **master_v2_backup.csv** | 8,352 (50.1%) | 8,304 (49.9%) | 16,656 |

**结论**:
- `master_v2.csv` 完全平衡（1:1）
- `master_v2_backup.csv` 基本平衡（轻微不平衡）

---

### 3. split列状态对比

**master_v2.csv**:
```
split
unsplit    16000  (100%)
```
- 所有数据都是 `unsplit` 状态
- 需要重新分割用于实验

**master_v2_backup.csv**:
```
split
unsplit    15985  (95.9%)
train        469   (2.8%)
test         101   (0.6%)
val          101   (0.6%)
```
- 大部分是 `unsplit`
- 已有部分数据被分割（train/test/val）
- 但这些分割可能不是S0实验需要的格式

---

### 4. 数据源对比

**master_v2.csv**:
```
source
phish     8000  (50.0%)
benign    8000  (50.0%)
```

**master_v2_backup.csv**:
```
source
phish                                   7998  (48.0%)
benign                                  7987  (47.9%)
D:\uaam-phish\data\raw\fish_dataset     354   (2.1%)
D:\uaam-phish\data\raw\dataset          317   (1.9%)
```

**结论**:
- `master_v2.csv` 数据源更统一（只有 phish 和 benign）
- `master_v2_backup.csv` 包含文件路径作为source（不一致）

---

### 5. 时间戳字段

**master_v2.csv**:
- ✅ 有 `timestamp` 列: 16,000 个非空值 (100%)
- ✅ 有 `timestamp_original` 列: 15,985 个非空值 (99.9%)

**master_v2_backup.csv**:
- ✅ 有 `timestamp` 列: 16,654 个非空值 (100%)
- ❌ 无 `timestamp_original` 列

**结论**: `master_v2.csv` 有更完整的时间戳信息

---

### 6. 品牌信息

| 文件 | 唯一品牌数 |
|------|-----------|
| **master_v2.csv** | 7,915 |
| **master_v2_backup.csv** | 8,250 |

**结论**: `master_v2_backup.csv` 有更多品牌，但可能包含重复或不规范的数据

---

## ✅ 结论和建议

### S0实验应该使用: `master_v2.csv`

**原因**:

1. ✅ **完整的时间戳信息**
   - 包含 `timestamp_original` 字段
   - 100% 的数据有时间戳
   - 支持时间序列分析和temporal分割协议

2. ✅ **数据平衡**
   - 完美的1:1正负样本平衡
   - 适合S0基线实验

3. ✅ **数据源统一**
   - source字段一致（只有phish和benign）
   - 数据质量更高

4. ✅ **配置文件已指向此文件**
   - 所有配置文件默认使用 `master_v2.csv`
   - `configs/default.yaml`: `csv_path: data/processed/master_v2.csv`
   - `configs/experiment/multimodal_baseline.yaml`: `master_csv: "data/processed/master_v2.csv"`

5. ✅ **split状态适合S0**
   - 所有数据都是 `unsplit` 状态
   - 可以使用S0工具重新分割为IID或Brand-OOD
   - 分割过程可控、可复现

---

### master_v2_backup.csv 的用途

**建议用途**:
- 作为备份文件保留
- 如果需要恢复历史数据，可以参考
- 不应用于S0实验

**不推荐用于S0的原因**:
- ❌ 缺少 `timestamp_original` 字段
- ❌ 数据源不一致（包含文件路径）
- ❌ 已有部分分割，但不符合S0需求
- ❌ 数据不平衡（轻微）

---

## 🚀 S0实验数据准备

### 使用 master_v2.csv 进行S0实验

```bash
# 1. 创建IID分割
python tools/split_iid.py \
  --in data/processed/master_v2.csv \
  --out workspace/data/splits/iid \
  --seed 42

# 2. 创建Brand-OOD分割
python tools/split_brandood.py \
  --in data/processed/master_v2.csv \
  --out workspace/data/splits/brandood \
  --seed 42 \
  --top_k 20

# 3. 生成腐败数据（可选）
python tools/corrupt_html.py \
  --in workspace/data/splits/iid/test.csv \
  --out workspace/data/corrupt/html

python tools/corrupt_img.py \
  --in workspace/data/splits/iid/test.csv \
  --out workspace/data/corrupt/img

python tools/corrupt_url.py \
  --in workspace/data/splits/iid/test.csv \
  --out workspace/data/corrupt/url
```

---

## 📝 配置文件说明

### S0实验配置

所有S0实验配置文件都指向 `master_v2.csv`:

```yaml
# configs/experiment/s0_iid_earlyconcat.yaml
datamodule:
  train_csv: workspace/data/splits/iid/train.csv
  val_csv: workspace/data/splits/iid/val.csv
  test_csv: workspace/data/splits/iid/test.csv
```

**数据流程**:
1. 使用 `master_v2.csv` 作为输入
2. 通过 `split_iid.py` 或 `split_brandood.py` 创建分割
3. 分割后的CSV保存在 `workspace/data/splits/`
4. 实验配置指向分割后的CSV文件

---

## 🔄 数据版本管理建议

### 当前状态

- ✅ `master_v2.csv` - **当前使用**（16,000行，18列）
- 📦 `master_v2_backup.csv` - **备份文件**（16,656行，17列）

### 建议

1. **S0实验**: 使用 `master_v2.csv`
2. **备份保留**: 保留 `master_v2_backup.csv` 作为历史备份
3. **版本控制**: 在 `.gitignore` 中排除数据文件，使用DVC管理

---

## 📊 数据质量对比总结

| 指标 | master_v2.csv | master_v2_backup.csv | 胜者 |
|------|--------------|---------------------|------|
| 数据平衡 | ✅ 完美 (50:50) | ⚠️ 基本平衡 | master_v2.csv |
| 时间戳完整性 | ✅ 100% | ✅ 100% | 平局 |
| timestamp_original | ✅ 有 | ❌ 无 | master_v2.csv |
| 数据源一致性 | ✅ 统一 | ❌ 不一致 | master_v2.csv |
| split状态 | ✅ 统一 (unsplit) | ⚠️ 混合 | master_v2.csv |
| 配置文件支持 | ✅ 已配置 | ❌ 未配置 | master_v2.csv |

**总体评价**: `master_v2.csv` 更适合S0实验 ✅

---

**最后更新**: 2025-11-08
**分析工具**: `compare_csv_files.py`



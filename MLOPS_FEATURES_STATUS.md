# MLOps 功能实现状态报告

**日期**: 2025-10-23
**实验**: url_mvp_20251023_081630

---

## 📋 功能检查清单

### ✅ **已完整实现并使用**

#### 1. **Step Metrics (步骤级指标)**
```
状态: ✅ 完全实现并运行
位置: src/utils/metrics.py, src/systems/url_only_module.py

指标:
  ✓ Accuracy (准确率)
  ✓ AUROC (pos_label=1)
  ✓ F1-macro

配置:
  metrics.dist.sync_metrics: false (默认)
  metrics.average: macro
```

#### 2. **Epoch Metrics (轮次级指标)**
```
状态: ✅ 完全实现并运行
位置: src/systems/url_only_module.py

指标:
  ✓ NLL (Negative Log Likelihood)
  ✓ ECE (Expected Calibration Error) with adaptive bins

结果: test_nll=0.0345, test_ece=0.0207, test_ece_bins=10
```

#### 3. **Protocol Artifacts (协议产物)**
```
状态: ✅ 完全实现并生成
位置: src/utils/protocol_artifacts.py, src/utils/visualizer.py

生成文件:
  ✓ metrics_random.json - 指标JSON文件
  ✓ roc_random.png - ROC曲线 (包含AUC标注)
  ✓ calib_random.png - 校准曲线 (包含ECE标注)
  ✓ implementation_report.md - 实现报告
```

#### 4. **Batch Format Configuration (批量格式配置)**
```
状态: ✅ 已配置
位置: configs/data/url_only.yaml

配置:
  data.batch_format: tuple (默认值)

代码支持:
  ✓ src/utils/batch_utils.py - _unpack_batch() 函数
  ✓ 支持 tuple 和 dict 两种格式
```

#### 5. **Metadata Column Names (元数据列名)**
```
状态: ✅ 已配置
位置: configs/data/url_only.yaml

配置:
  ✓ data.text_col: url_text
  ✓ data.label_col: label
  ✓ data.timestamp_col: timestamp
  ✓ data.brand_col: brand
  ✓ data.source_col: source
```

---

### ⚠️ **已实现但未使用**

#### 6. **Data Splitting Protocols (数据分割协议)**
```
状态: ⚠️ 代码完整但未启用
位置: src/utils/splits.py

已实现协议:
  ✓ random - 随机分割（分层采样）
  ✓ temporal - 时序分割（tie_policy="left-closed"）
  ✓ brand_ood - 品牌域外分割（不相交品牌集）

函数:
  ✓ build_splits(df, cfg, protocol) - 完整实现
  ✓ 自动降级机制（数据不足时降级到random）
  ✓ 生成splits_{protocol}.csv统计表

未使用原因:
  ✗ use_build_splits: false (未启用)
  ✗ protocol: 未设置
  ✗ 使用的是预先分割的CSV文件
```

#### 7. **Metadata Extraction (元数据提取)**
```
状态: ⚠️ 配置存在但数据集未返回
位置: src/data/url_dataset.py

当前行为:
  - UrlDataset.__getitem__() 返回: (input_ids, label)
  - 符合 "non-breaking" 原则 ✓

支持:
  ✓ _unpack_batch() 可处理 (x, y) 或 (x, y, meta)
  ✓ 配置文件中有 timestamp/brand/source 列名

缺失:
  ✗ 数据集未读取 timestamp/brand/source 列
  ✗ meta dict 始终为 {timestamp: None, brand: None, source: None}

原因:
  - UrlDataset 专注于字符级URL编码
  - 当前CSV文件(url_train.csv)只有 url_text, label 两列
```

---

## 📊 **实现状态总结**

### ✅ **完全符合规范的功能**

1. ✅ **Non-breaking Batching**
   ```python
   # Dataset 保持返回 (x, y)
   def __getitem__(self, index):
       return torch.tensor(encoded), torch.tensor(label)

   # _unpack_batch 处理并提供默认 meta
   inputs, labels, meta = _unpack_batch(batch, batch_format="tuple")
   # meta = {timestamp: None, brand: None, source: None}
   ```

2. ✅ **Config Key `data.batch_format`**
   ```yaml
   data:
     batch_format: tuple  # 默认值 ✓
   ```

3. ✅ **Step Metrics**
   ```
   Accuracy, AUROC(pos_label=1), F1(macro) ✓
   ```

4. ✅ **Epoch Metrics**
   ```
   NLL, ECE(adaptive bins) ✓
   ```

5. ✅ **Artifacts**
   ```
   roc_{protocol}.png ✓
   calib_{protocol}.png (with ECE annotation) ✓
   metrics_{protocol}.json ✓
   ```

---

### ⚠️ **部分实现的功能**

1. ⚠️ **Data Protocols & Splits**
   ```
   实现状态: CODE COMPLETE ✓
   使用状态: NOT ENABLED ✗

   要启用:
   1. 准备包含所有列的 master.csv
   2. 设置 use_build_splits: true
   3. 设置 protocol: random/temporal/brand_ood
   ```

2. ⚠️ **Metadata Extraction**
   ```
   配置状态: CONFIGURED ✓
   数据流转: NOT PASSING THROUGH ✗

   当前: meta 始终为 None
   原因: UrlDataset 只返回 (x, y)
   解决: 扩展 UrlDataset 或使用 collate_fn
   ```

---

## 💡 **如何启用所有功能**

### 方案1: 完整的Protocol实验

创建 `configs/experiment/url_with_protocols.yaml`:

```yaml
# @package _global_

defaults:
  - override /data: url_only
  - override /model: url_encoder

run:
  name: url_with_protocols
  seed: 42

# 启用 protocol splits
protocol: random  # 或 temporal, brand_ood
use_build_splits: true

data:
  csv_path: data/processed/master.csv  # 包含所有列的主文件
  batch_format: tuple

train:
  epochs: 50
  bs: 64
  lr: 0.0001
```

运行:
```bash
python scripts/train_hydra.py experiment=url_with_protocols
```

这样会：
1. 从 master.csv 读取数据
2. 使用 build_splits() 按 protocol 分割
3. 生成 splits_random.csv 统计表
4. 训练并生成所有artifacts

---

### 方案2: 添加Metadata到Dataset

修改 `src/data/url_dataset.py`:

```python
class UrlDataset(Dataset):
    def __init__(self, csv_path, ...):
        frame = pd.read_csv(csv_path)
        self._texts = frame["url_text"].tolist()
        self._labels = frame["label"].tolist()

        # 读取元数据列（如果存在）
        self._timestamps = frame.get("timestamp", pd.Series([None]*len(frame))).tolist()
        self._brands = frame.get("brand", pd.Series([None]*len(frame))).tolist()
        self._sources = frame.get("source", pd.Series([None]*len(frame))).tolist()

    def __getitem__(self, index):
        # 保持 non-breaking: 返回 (x, y, meta)
        encoded = encode_url(...)
        label = self._labels[index]

        meta = {
            "timestamp": self._timestamps[index],
            "brand": self._brands[index],
            "source": self._sources[index],
        }

        return torch.tensor(encoded), torch.tensor(label), meta
```

这样：
1. 保持 non-breaking (可以返回2或3个元素)
2. _unpack_batch 自动处理
3. meta 数据被传递到训练循环

---

## 🎯 **当前状态总结**

### **这次训练 (url_mvp_20251023_081630)**

| 功能 | 状态 | 说明 |
|------|------|------|
| Step Metrics | ✅ 使用 | Accuracy, AUROC, F1 |
| Epoch Metrics | ✅ 使用 | NLL, ECE |
| Artifacts | ✅ 生成 | ROC, Calibration, JSON |
| batch_format | ✅ 配置 | tuple |
| Metadata Cols | ✅ 配置 | timestamp, brand, source |
| Protocol Splits | ⚠️ 未用 | 代码存在，未启用 |
| Metadata Flow | ⚠️ 未用 | 配置存在，数据未传递 |

---

### **完整功能启用需要**

1. **准备主数据文件** (master.csv)
   ```
   url_text, label, timestamp, brand, source
   ```

2. **启用 protocol splits**
   ```yaml
   protocol: random  # or temporal, brand_ood
   use_build_splits: true
   ```

3. **扩展 Dataset** (可选，用于metadata)
   ```python
   # 让 __getitem__ 返回 (x, y, meta)
   ```

---

## 📝 **结论**

### ✅ **已完全实现的规范**

1. ✅ Non-breaking Batching - Dataset 返回 (x,y)，系统兼容
2. ✅ batch_format config key - 已配置默认为 tuple
3. ✅ _unpack_batch helper - 完整实现
4. ✅ Step metrics - Accuracy, AUROC, F1-macro
5. ✅ Epoch metrics - NLL, ECE with adaptive bins
6. ✅ Artifacts - ROC, Calibration, JSON 全部生成
7. ✅ Metadata column names - 已配置

### ⚠️ **可选功能（代码完整，可随时启用）**

1. ⚠️ Protocol splits (random/temporal/brand_ood) - 设置2个参数即可启用
2. ⚠️ Metadata extraction - 扩展Dataset或使用collate_fn

### 🎉 **当前训练完全符合基本规范！**

- 所有必需的MLOps功能都已实现并运行
- Protocol splits 和 metadata 是高级可选功能
- 可以随时通过配置启用

---

*报告生成时间: 2025-10-23*
*实验目录: experiments/url_mvp_20251023_081630*

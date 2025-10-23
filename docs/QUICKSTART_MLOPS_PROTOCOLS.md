# MLOps Quickstart: Data Split Protocols

本指南展示如何使用不同的数据分割协议运行实验。

## 📋 支持的协议

1. **random** - 随机分层分割（默认）
2. **temporal** - 时间序列分割（按timestamp排序）
3. **brand_ood** - 品牌域外泛化（品牌集不相交）

---

## 🚀 零代码快速启动

### 1. Random 基线（默认）

```bash
python scripts/train_hydra.py
```

或显式指定：

```bash
python scripts/train_hydra.py protocol=random
```

**预期输出：**
- 工件保存在 `experiments/<run_name>/results/`
- 文件：
  - `roc_random.png`
  - `calib_random.png`
  - `splits_random.csv`
  - `metrics_random.json`

---

### 2. Temporal 分割

```bash
python scripts/train_hydra.py protocol=temporal
```

**要求：**
- 数据必须包含 `timestamp` 列
- 如果缺失，自动降级到 `random` 并记录原因

**特性：**
- 按时间升序排序
- Tie policy = "left-closed"（相同时间戳归入较早的分割）

**输出文件：**
- `roc_temporal.png`
- `calib_temporal.png`
- `splits_temporal.csv`（包含 `timestamp_min/max`）
- `metrics_temporal.json`

---

### 3. Brand-OOD 分割

```bash
python scripts/train_hydra.py protocol=brand_ood
```

**要求：**
- 数据必须包含 `brand` 列
- 至少 3 个不同品牌（否则降级到 `random`）

**特性：**
- 品牌归一化：`strip().lower()`
- 严格的品牌不相交：`train_brands ∩ test_brands = ∅`
- 如果相交检查失败，自动降级到 `random`

**输出文件：**
- `roc_brand_ood.png`
- `calib_brand_ood.png`
- `splits_brand_ood.csv`（包含 `brand_set`, `brand_unique`）
- `metrics_brand_ood.json`

---

## 📊 输出说明

### Artifacts 目录结构

```
experiments/<run_name>/
├── config.yaml
├── checkpoints/
│   └── best.ckpt
├── logs/
│   └── metrics_history.csv
└── results/
    ├── roc_{protocol}.png          # ROC曲线
    ├── calib_{protocol}.png         # 校准曲线（带ECE标注）
    ├── splits_{protocol}.csv        # 分割统计
    └── metrics_{protocol}.json      # 完整指标
```

### Metrics JSON 格式

```json
{
  "accuracy": 0.95,
  "auroc": 0.98,
  "f1_macro": 0.94,
  "nll": 0.12,
  "ece": 0.03,
  "ece_bins_used": 10,
  "positive_class": "phishing",
  "artifacts": {
    "roc_path": "results/roc_random.png",
    "calib_path": "results/calib_random.png",
    "splits_path": "results/splits_random.csv"
  },
  "warnings": {
    "downgraded_reason": null
  }
}
```

### Splits CSV 格式

| split | count | pos_count | neg_count | brand_unique | timestamp_min | timestamp_max |
|-------|-------|-----------|-----------|--------------|---------------|---------------|
| train | 7000  | 3500      | 3500      | 15           | 2023-01-01    | 2023-06-30    |
| val   | 1500  | 750       | 750       | 8            | 2023-07-01    | 2023-09-15    |
| test  | 1500  | 750       | 750       | 7            | 2023-09-16    | 2023-12-31    |

---

## ⚙️ 配置选项

### 修改分割比例

编辑 `configs/data/url_only.yaml`:

```yaml
data:
  split_ratios:
    train: 0.7
    val: 0.15
    test: 0.15
```

### 启用 WandB 日志

```bash
python scripts/train_hydra.py protocol=temporal logger=wandb
```

### 使用本地配置（快速测试）

```bash
python scripts/train_hydra.py +profiles/local protocol=random
```

---

## 🔍 降级机制

协议会在以下情况下自动降级到 `random`:

| 协议 | 降级条件 | 记录位置 |
|------|----------|----------|
| temporal | 缺少 `timestamp` 列 | `metrics_{protocol}.json.warnings.downgraded_reason` |
| brand_ood | 缺少 `brand` 列 | 同上 |
| brand_ood | 品牌数 ≤ 2 | 同上 |
| brand_ood | 品牌集相交（验证失败） | 同上 + `splits_{protocol}.csv.brand_intersection_ok=False` |

降级后：
- 只生成 `*_random.*` 文件
- `splits_random.csv` 包含 `downgraded_to` 列

---

## 🧪 指标说明

### Step 级指标（每个batch）
- **Accuracy**: 准确率
- **AUROC**: ROC曲线下面积（pos_label=1）
- **F1**: F1分数（macro平均）

### Epoch 级指标（整个epoch）
- **NLL**: 负对数似然（CrossEntropyLoss均值）
- **ECE**: 期望校准误差
  - 自适应bins: `max(3, min(15, floor(sqrt(N)), 10))`
  - 记录实际使用的bins数量

---

## 📝 实现报告

每次运行后，在 `experiments/<run>/results/implementation_report.md` 查看：
- 详细变更日志
- 工件路径
- Metrics JSON 前20行
- Splits CSV 前10行
- 所有警告和降级信息

---

## 🛡️ URL 编码器锁定

URL编码器架构已锁定，不可修改：
- 2层双向LSTM（BiLSTM）
- 字符级tokenization
- Hidden size: 128
- Output dim: 256

任何尝试修改将触发断言错误。

---

## ❓ 故障排除

### 问题：协议降级到 random
**解决：**
1. 检查数据是否包含必需列
2. 查看 `metrics_{protocol}.json.warnings.downgraded_reason`

### 问题：缺少 ECE bins 警告
**原因：** 样本量太小，bins自动减少
**解决：** 在 `calib_{protocol}.png` 上会显示警告标记

### 问题：Brand-OOD 品牌集相交
**解决：**
1. 检查 `splits_{protocol}.csv.brand_intersection_ok`
2. 如果为 `False`，说明品牌分割失败
3. 查看日志了解具体品牌重叠情况

---

## 📚 相关文档

- [ARCHITECTURE_CLARIFICATION.md](ARCHITECTURE_CLARIFICATION.md) - 系统架构
- [EXPERIMENTS.md](EXPERIMENTS.md) - 实验设计
- [WANDB_GUIDE.md](WANDB_GUIDE.md) - WandB集成
- [TESTING_GUIDE.md](TESTING_GUIDE.md) - 测试指南

# URL-Only 快速参考卡

## 🚀 一键运行（推荐）

```bash
# Linux/Mac
bash scripts/run_all_protocols.sh

# Windows PowerShell
.\scripts\run_all_protocols.ps1
```

---

## 📦 单协议运行

```bash
# Random
python scripts/train_hydra.py protocol=random use_build_splits=true

# Temporal
python scripts/train_hydra.py protocol=temporal use_build_splits=true

# Brand-OOD
python scripts/train_hydra.py protocol=brand_ood use_build_splits=true
```

---

## ✅ 验证产物

```bash
# 自动验证最新实验
python tools/check_artifacts_url_only.py

# 验证特定实验
python tools/check_artifacts_url_only.py experiments/url_random_20251022_120000
```

---

## 🛠️ 准备工作

```bash
# 如果没有 master.csv，先创建
python scripts/create_master_csv.py

# 检查数据
ls -lh data/processed/*.csv
```

---

## 🎯 预期产物（四件套 × 3协议 = 12文件）

```
experiments/<run>/results/
├── roc_random.png           ← ROC 曲线
├── calib_random.png         ← 校准图（含ECE标注）
├── splits_random.csv        ← 数据分割统计（13列）
├── metrics_random.json      ← 指标JSON（9个key）
├── roc_temporal.png
├── calib_temporal.png
├── splits_temporal.csv
├── metrics_temporal.json
├── roc_brand_ood.png
├── calib_brand_ood.png
├── splits_brand_ood.csv
└── metrics_brand_ood.json
```

---

## 📋 splits_*.csv 必需列（13列）

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

---

## 📊 metrics_*.json 必需字段

```json
{
  "accuracy": 0.xx,
  "auroc": 0.xx,
  "f1_macro": 0.xx,
  "nll": 0.xx,
  "ece": 0.xx,
  "ece_bins_used": 10,
  "positive_class": "phishing",
  "artifacts": {
    "roc_path": "...",
    "calib_path": "...",
    "splits_path": "..."
  },
  "warnings": {
    "downgraded_reason": null
  }
}
```

---

## 🔧 常用参数覆盖

```bash
# 快速测试（10% 数据，5 epochs）
python scripts/train_hydra.py protocol=random use_build_splits=true +profiles/local

# 自定义 batch size
python scripts/train_hydra.py protocol=random use_build_splits=true train.batch_size=128

# 自定义 epochs
python scripts/train_hydra.py protocol=random use_build_splits=true train.epochs=100

# 禁用 early stopping
python scripts/train_hydra.py protocol=random use_build_splits=true eval.patience=999

# 使用 WandB logger
python scripts/train_hydra.py protocol=random use_build_splits=true logger=wandb
```

---

## 🐛 故障排除

### 问题: 缺少 master.csv

```bash
python scripts/create_master_csv.py
```

### 问题: 缺少 splits_*.csv

```bash
# 确保启用 use_build_splits
python scripts/train_hydra.py protocol=random use_build_splits=true
```

### 问题: 校准图没有 ECE 标注

- 检查 `src/utils/visualizer.py:529-532`
- 应该有 `ax.text(... "ECE = ...")`

### 问题: brand_intersection_ok 为空

- 确保 master.csv 有 `brand` 列
- 对于 brand_ood，确保至少 3 个品牌

---

## 📚 文档链接

- **完整指南**: `URL_ONLY_CLOSURE_GUIDE.md`
- **自检报告**: `CHANGES_SUMMARY.md` （末尾）
- **实现报告**: `IMPLEMENTATION_REPORT.md`
- **快速开始**: `docs/QUICKSTART_MLOPS_PROTOCOLS.md`

---

**更新**: 2025-10-22
**状态**: ✅ Ready

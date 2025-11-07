# 多模态 Baseline 烟雾测试指南

## 测试失败原因总结

### 问题 1: Hydra Struct 模式错误
**错误信息**:
```
Could not override 'trainer.fast_dev_run'.
Key 'fast_dev_run' is not in struct
```

**原因**: 配置文件缺少调试参数的默认定义。

**解决**: 已在 `configs/trainer/default.yaml` 添加 `trainer` 调试参数。

---

### 问题 2: fast_dev_run 与 checkpoint 冲突
**错误信息**:
```
ValueError: You cannot execute .test(ckpt_path="best") with fast_dev_run=True
```

**原因**: 烟雾测试模式不保存检查点，但代码尝试加载 "best" 检查点。

**解决**: 已修改 `scripts/train_hydra.py`，在 fast_dev_run 模式下使用当前权重。

---

### 问题 3: 缺少依赖库
**错误信息**:
```
无法从源码解析导入 "bs4"
```

**原因**: `requirements.txt` 缺少 beautifulsoup4 等必需库。

**解决**: 已更新 `requirements.txt`，补全所有依赖。

---

## 快速修复步骤

### 1. 确保在正确的虚拟环境中

**重要**: 确保你的终端提示符显示 `(.venv)` 或类似的虚拟环境标识。

如果没有激活虚拟环境：
```bash
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# Windows CMD
.\.venv\Scripts\activate.bat

# Linux/Mac
source .venv/bin/activate
```

### 2. 安装所有依赖

**方式一：安装完整依赖列表（推荐）**
```bash
python -m pip install -r requirements.txt
```

**方式二：仅安装核心依赖（快速测试）**
```bash
python -m pip install hydra-core omegaconf pytorch-lightning torch transformers torchmetrics torchvision pandas scikit-learn Pillow beautifulsoup4 lxml tldextract matplotlib seaborn
```

**验证安装**:
```bash
python -c "import hydra; import torch; import pytorch_lightning; from bs4 import BeautifulSoup; print('Dependencies OK')"
```

---

## 运行烟雾测试

### 测试 1: Dry-run 烟雾测试（验证产物五件套）

```bash
python scripts/train_hydra.py experiment=multimodal_baseline trainer.fast_dev_run=true
```

**预期行为**:
- ✅ 配置加载成功
- ✅ 训练 1 个 batch
- ✅ 验证 1 个 batch
- ✅ 测试 1 个 batch（使用当前权重，不加载 checkpoint）
- ✅ 生成验证集产物：
  - `predictions_val.csv`
  - `metrics_val.json`
  - `roc_curve_val.png`
  - `reliability_before_ts_val.png`

**预计耗时**: 2-5 分钟（首次运行需下载 BERT 模型）

---

### 测试 2: 随机分割回归测试（验证 70/15/15 + data_splits.json）

```bash
python scripts/train_hydra.py experiment=multimodal_baseline datamodule.split_protocol=random trainer.fast_dev_run=true
```

**预期行为**:
- ✅ 使用 `build_splits` 进行随机分割（70/15/15）
- ✅ 生成 `splits_random.csv`（包含分割统计信息）
- ✅ 其他行为同测试 1

**预计耗时**: 2-5 分钟

---

## 预期输出产物

### 实验目录结构

```
experiments/url_mvp_YYYYMMDD_HHMMSS/
├── config.yaml                          # 完整配置快照
├── artifacts/
│   ├── predictions_val.csv              # 验证集预测结果
│   ├── metrics_val.json                 # 验证集指标
│   ├── roc_curve_val.png               # ROC 曲线
│   └── reliability_before_ts_val.png   # 校准图
└── results/
    └── splits_presplit.csv (or splits_random.csv)  # 数据分割元数据
```

### metrics_val.json 示例

```json
{
  "accuracy": 0.465,
  "auroc": 0.723,
  "f1_macro": 0.000,
  "nll": 3.160,
  "ece": 0.XX,
  "positive_class": "phishing",
  "split_protocol": "presplit",
  "artifacts": {
    "roc_path": "experiments/.../roc_curve_val.png",
    "calib_path": "experiments/.../reliability_before_ts_val.png"
  }
}
```

---

## 故障排查

### 错误: `ModuleNotFoundError: No module named 'bs4'`

**解决**:
```bash
pip install beautifulsoup4 lxml
```

---

### 错误: `ModuleNotFoundError: No module named 'PIL'`

**解决**:
```bash
pip install Pillow
```

---

### 错误: `ModuleNotFoundError: No module named 'torchvision'`

**解决**:
```bash
pip install torchvision
```

---

### 警告: `Could not override 'trainer.fast_dev_run'`

**原因**: 你可能在使用旧版本的配置文件。

**解决**: 确保 `configs/trainer/default.yaml` 包含以下内容：

```yaml
trainer:
  fast_dev_run: false
  limit_train_batches: null
  limit_val_batches: null
  limit_test_batches: null
  overfit_batches: 0
```

---

### 错误: `ValueError: You cannot execute .test(ckpt_path="best")`

**原因**: 你可能在使用旧版本的 `train_hydra.py`。

**解决**: 确保 `scripts/train_hydra.py` 第 173 行包含：

```python
ckpt_path = "best" if not getattr(cfg.trainer, "fast_dev_run", False) else None
```

---

## 技术说明

### fast_dev_run 模式特性

- **目的**: 快速验证代码语法、数据管道、模型前向传播
- **限制**:
  - 仅运行 1 个 batch（train/val/test）
  - 不保存检查点（ModelCheckpoint 被禁用）
  - 不记录到外部 logger（WandB、TensorBoard 等）
  - EarlyStopping 被禁用
- **适用场景**:
  - CI/CD 管道
  - 代码重构后的快速验证
  - 新功能的烟雾测试

### 为什么测试阶段不加载 checkpoint？

在 `fast_dev_run` 模式下：
1. PyTorch Lightning 自动禁用 `ModelCheckpoint` callback
2. 因此不会保存任何 `.ckpt` 文件
3. 如果强制加载 `ckpt_path="best"`，会抛出 `ValueError`
4. **解决方案**: 使用当前训练的权重（`ckpt_path=None`）

这在烟雾测试中是合理的，因为我们只关心代码能否运行，不关心模型性能。

---

## 下一步

### 运行完整训练（非 fast_dev_run）

```bash
# 使用预分割数据（默认）
python scripts/train_hydra.py experiment=multimodal_baseline

# 使用随机分割
python scripts/train_hydra.py experiment=multimodal_baseline datamodule.split_protocol=random

# 使用时序分割
python scripts/train_hydra.py experiment=multimodal_baseline datamodule.split_protocol=temporal

# 使用品牌 OOD 分割
python scripts/train_hydra.py experiment=multimodal_baseline datamodule.split_protocol=brand_ood
```

**预计耗时**: 15-30 分钟（取决于 GPU 和数据集大小）

---

## 验证论文合规性

所有变更遵循论文定义的设计原则：

✅ **Add-only & Idempotent**: 未删除任何现有代码或配置
✅ **Non-breaking**: 现有实验配置无需修改
✅ **Reproducibility**: 添加的参数不影响随机种子或模型行为

详见 `CHANGES_SUMMARY.md`。

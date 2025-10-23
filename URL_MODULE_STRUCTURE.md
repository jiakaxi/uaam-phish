# URL 模块项目结构（完整逻辑流程）

> **更新时间**: 2025-10-22
> **状态**: ✅ 生产就绪

---

## 📋 目录

1. [数据流程](#1-数据流程)
2. [配置系统](#2-配置系统)
3. [核心模块](#3-核心模块)
4. [训练系统](#4-训练系统)
5. [推理预测](#5-推理预测)
6. [工具验证](#6-工具验证)
7. [实验产出](#7-实验产出)
8. [文档系统](#8-文档系统)

---

## 1. 数据流程

### 1.1 原始数据源

```
data/raw/
├── dataset/              ← 合法网站数据（benign）
│   └── *.csv
└── fish_dataset/         ← 钓鱼网站数据（phishing）
    └── *.csv
```

**说明**:
- 两个数据源分别存放合法和钓鱼网站的原始数据
- 必须包含: `url_text`, `label`, `timestamp`(可选), `brand`(可选), `source`(可选)

---

### 1.2 数据处理脚本

#### 生成主数据集
```bash
# 脚本位置
scripts/build_master_and_splits.py
scripts/create_master_csv.py        # 简化版，只生成 master.csv

# 用法
python scripts/create_master_csv.py

# 产出
data/processed/master.csv           # 合并后的主数据集
```

**master.csv 必需列**:
- `url_text`: URL 文本（必需）
- `label`: 标签 0=合法, 1=钓鱼（必需）
- `timestamp`: 时间戳（temporal协议需要）
- `brand`: 品牌名称（brand_ood协议需要）
- `source`: 数据来源（可选）

---

### 1.3 数据分割（三协议）

#### 分割策略实现
```python
# 位置: src/utils/splits.py

def build_splits(df, cfg, protocol) -> (train_df, val_df, test_df, metadata):
    """
    三种协议:
    - random: 随机分割 (默认 70/15/15)
    - temporal: 时间序列分割 (按timestamp排序)
    - brand_ood: 品牌域外分割 (train/test品牌不重叠)
    """
```

#### 自动分割机制
```python
# 位置: src/datamodules/url_datamodule.py

class UrlDataModule:
    def setup(self, stage="fit"):
        if self.cfg.get("use_build_splits", False):
            # 自动调用 build_splits 生成分割
            train_df, val_df, test_df, metadata = build_splits(...)
            # 保存到 CSV
            train_df.to_csv(data/processed/url_train.csv)
            val_df.to_csv(data/processed/url_val.csv)
            test_df.to_csv(data/processed/url_test.csv)
            # 保存元数据供后续使用
            self.split_metadata = metadata
```

#### 分割产出
```
data/processed/
├── master.csv            # 主数据集（输入）
├── url_train.csv         # 训练集（输出）
├── url_val.csv           # 验证集（输出）
└── url_test.csv          # 测试集（输出）
```

---

### 1.4 数据集类（字符级编码）

```python
# 位置: src/data/url_dataset.py

class UrlDataset(Dataset):
    """
    字符级 URL 数据集
    - 输入: CSV 文件 (url_text, label)
    - 编码: 字符 → ASCII码 (0-127)
    - 输出: (input_ids: Tensor[L], label: int)
    """

def encode_url(text, max_len, vocab_size, pad_id):
    """
    字符级编码函数:
    1. 每个字符 → ord(char)
    2. 超出vocab_size → vocab_size-1
    3. 填充到 max_len
    """
```

**默认参数**:
- `max_len`: 256 字符
- `vocab_size`: 128 (ASCII标准)
- `pad_id`: 0

---

## 2. 配置系统

### 2.1 配置结构（Hydra）

```
configs/
├── config.yaml           # 主配置（组合所有部分）
├── default.yaml          # 默认基础配置
├── base.yaml             # 基础设置
├── hparams.yaml          # 超参数
├── encoders.yaml         # 编码器配置
│
├── data/
│   └── url_only.yaml     # URL数据配置 ⭐
│
├── model/
│   └── url_encoder.yaml  # URL编码器模型配置 ⭐
│
├── trainer/
│   ├── local.yaml        # 本地快速测试 (10%数据, 5 epochs)
│   ├── server.yaml       # 服务器完整训练
│   └── default.yaml      # 默认训练器
│
├── profiles/
│   ├── local.yaml        # 本地环境配置
│   └── server.yaml       # 服务器环境配置
│
├── experiment/
│   └── url_baseline.yaml # URL基线实验配置 ⭐
│
└── logger/
    ├── csv.yaml          # CSV日志
    ├── tensorboard.yaml  # TensorBoard
    └── wandb.yaml        # Weights & Biases
```

---

### 2.2 核心配置文件

#### A. URL 数据配置
```yaml
# configs/data/url_only.yaml
data:
  csv_path: data/processed/master.csv      # 主数据集
  train_csv: data/processed/url_train.csv  # 训练集
  val_csv: data/processed/url_val.csv      # 验证集
  test_csv: data/processed/url_test.csv    # 测试集

  text_col: url_text                       # URL文本列名
  label_col: label                         # 标签列名
  timestamp_col: timestamp                 # 时间戳列名
  brand_col: brand                         # 品牌列名
  source_col: source                       # 来源列名

  num_workers: 4                           # DataLoader工作进程数
  batch_format: tuple                      # 批次格式: (input_ids, labels)

  split_ratios:
    train: 0.7                             # 训练集比例
    val: 0.15                              # 验证集比例
    test: 0.15                             # 测试集比例
```

#### B. URL 编码器配置
```yaml
# configs/model/url_encoder.yaml
model:
  vocab_size: 128           # ASCII字符集大小
  embedding_dim: 128        # 字符嵌入维度
  hidden_dim: 128           # LSTM隐藏层维度
  num_layers: 2             # LSTM层数 (固定)
  bidirectional: true       # 双向LSTM (固定)
  dropout: 0.1              # Dropout率
  pad_id: 0                 # 填充符号ID
  proj_dim: 256             # 投影层维度 (固定)
  max_len: 256              # URL最大长度
  num_classes: 2            # 分类数（二分类）
```

**🔒 架构锁定**:
- 2层双向LSTM
- 字符级编码
- 256维输出
- 代码中有断言保护，不可修改

---

## 3. 核心模块

### 3.1 URL 编码器（BiLSTM）

```python
# 位置: src/models/url_encoder.py

class URLEncoder(nn.Module):
    """
    字符级双向LSTM编码器

    架构（固定）:
    ┌─────────────────────────┐
    │  Input: URL字符序列     │
    │  [char_1, ..., char_n]  │
    └────────────┬────────────┘
                 │
    ┌────────────▼────────────┐
    │  Embedding (128-dim)    │
    └────────────┬────────────┘
                 │
    ┌────────────▼────────────┐
    │  Dropout (0.1)          │
    └────────────┬────────────┘
                 │
    ┌────────────▼────────────┐
    │  BiLSTM (2 layers)      │
    │  Hidden: 128            │
    │  Output: 256 (2×128)    │
    └────────────┬────────────┘
                 │
    ┌────────────▼────────────┐
    │  Projection (256-dim)   │
    └────────────┬────────────┘
                 │
    ┌────────────▼────────────┐
    │  Output: z_url ∈ R^256  │
    └─────────────────────────┘
    """

    def forward(self, input_ids):
        # 1. 字符嵌入
        embeddings = self.embedding(input_ids)

        # 2. BiLSTM编码
        _, (hidden, _) = self.lstm(embeddings)
        forward_h = hidden[-2]   # 前向最后层
        backward_h = hidden[-1]  # 后向最后层
        features = torch.cat([forward_h, backward_h], dim=1)

        # 3. 投影到256维
        return self.project(features)
```

**参数量**:
- Embedding: 128 × 128 = 16,384
- LSTM: ~200K
- Projection: 256 × 256 = 65,536
- **总计**: ~282K 参数

---

### 3.2 URL-Only 系统模块

```python
# 位置: src/systems/url_only_module.py

class UrlOnlyModule(pl.LightningModule):
    """
    完整的训练/评估系统

    组件:
    1. URLEncoder (编码器)
    2. Linear Classifier (分类器)
    3. Metrics (指标计算)
    4. Loss (损失函数)
    """

    def __init__(self, cfg):
        # 编码器
        self.encoder = URLEncoder(...)

        # 分类器（线性层）
        self.classifier = nn.Linear(proj_dim, num_classes)

        # 损失函数
        self.criterion = nn.CrossEntropyLoss()

        # 步级指标 (每个batch计算)
        self.train_metrics = {
            "accuracy": Accuracy(),
            "auroc": AUROC(pos_label=1),
            "f1": F1Score(average="macro")
        }

        # 轮次级指标 (整个epoch计算)
        # NLL: 负对数似然
        # ECE: 期望校准误差（自适应bins）

    def forward(self, input_ids):
        """编码: URL → 256维向量"""
        return self.encoder(input_ids)

    def predict_logits(self, input_ids):
        """预测: URL → logits (2维)"""
        z = self.forward(input_ids)
        return self.classifier(z)

    def training_step(self, batch, batch_idx):
        """训练步骤"""
        input_ids, labels = batch
        logits = self.predict_logits(input_ids)
        loss = self.criterion(logits, labels)
        return loss

    def validation_step(self, batch, batch_idx):
        """验证步骤（计算步级+轮次级指标）"""
        # ... 同上，并收集输出供 on_validation_epoch_end 使用

    def test_step(self, batch, batch_idx):
        """测试步骤（收集预测用于可视化）"""
        # ... 同上

    def on_validation_epoch_end(self):
        """验证轮次结束（计算NLL和ECE）"""
        all_logits = torch.cat([...])
        all_labels = torch.cat([...])
        all_probs = torch.softmax(all_logits, dim=1)

        # 计算 NLL
        nll = compute_nll(all_logits, all_labels)

        # 计算 ECE（自适应bins）
        ece, bins_used = compute_ece(y_true, y_prob, n_bins=None)

        self.log("val_nll", nll)
        self.log("val_ece", ece)

    def configure_optimizers(self):
        """优化器: AdamW"""
        return torch.optim.AdamW(self.parameters(), lr=cfg.train.lr)
```

**指标体系**:

| 指标类型 | 指标名称 | 计算时机 | 说明 |
|---------|---------|---------|------|
| 步级 | Accuracy | 每个batch | 准确率 |
| 步级 | AUROC | 每个batch | ROC曲线下面积（pos_label=1） |
| 步级 | F1-macro | 每个batch | 宏平均F1分数 |
| 轮次级 | NLL | 每个epoch | 负对数似然 |
| 轮次级 | ECE | 每个epoch | 期望校准误差（自适应bins: 3-15） |

---

### 3.3 数据模块（DataModule）

```python
# 位置: src/datamodules/url_datamodule.py

class UrlDataModule(pl.LightningDataModule):
    """
    Lightning 数据模块

    功能:
    1. 数据加载: 从CSV读取
    2. 自动分割: use_build_splits=true时调用build_splits
    3. DataLoader: 提供train/val/test加载器
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.split_metadata = {}  # 分割元数据（供callbacks使用）

    def setup(self, stage):
        # 如果启用 use_build_splits，自动生成分割
        if stage == "fit" and cfg.get("use_build_splits", False):
            train_df, val_df, test_df, metadata = build_splits(...)
            # 保存分割
            train_df.to_csv(train_csv)
            val_df.to_csv(val_csv)
            test_df.to_csv(test_csv)
            # 保存元数据
            self.split_metadata = metadata

        # 创建数据集
        self.train_dataset = UrlDataset(train_csv, ...)
        self.val_dataset = UrlDataset(val_csv, ...)
        self.test_dataset = UrlDataset(test_csv, ...)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=..., shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=..., shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=..., shuffle=False)
```

---

## 4. 训练系统

### 4.1 训练脚本（Hydra版）

```python
# 位置: scripts/train_hydra.py

@hydra.main(config_path="../configs", config_name="config")
def train(cfg):
    """
    Hydra训练主函数

    流程:
    1. 设置随机种子
    2. 初始化数据模块和模型
    3. 配置callbacks
    4. 配置trainer
    5. 训练: trainer.fit(model, datamodule)
    6. 测试: trainer.test(model, datamodule)
    7. 生成可视化和产物
    """

    # 1. 初始化
    pl.seed_everything(cfg.run.seed)
    dm = UrlDataModule(cfg)
    model = UrlOnlySystem(cfg)

    # 2. 配置callbacks
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=3),
        ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1),
        ExperimentResultsCallback(exp_tracker),      # 保存实验配置
        TestPredictionCollector(),                   # 收集测试预测
        ProtocolArtifactsCallback(protocol, ...),    # 生成协议产物
        DocumentationCallback(...),                   # 自动文档追加（可选）
    ]

    # 3. 配置trainer
    trainer = pl.Trainer(
        max_epochs=cfg.train.epochs,
        accelerator=cfg.hardware.accelerator,
        devices=cfg.hardware.devices,
        callbacks=callbacks,
        logger=logger,
    )

    # 4. 训练
    trainer.fit(model, dm)

    # 5. 更新 split_metadata（从 dm 传递给 callback）
    protocol_callback.split_metadata = dm.split_metadata

    # 6. 测试
    trainer.test(model, dm, ckpt_path="best")

    # 7. 生成可视化
    ResultVisualizer.create_all_plots(
        metrics_csv=...,
        y_true=...,
        y_prob=...,
        output_dir=exp_tracker.results_dir,
    )
```

---

### 4.2 运行命令

#### 单协议运行
```bash
# Random 协议
python scripts/train_hydra.py protocol=random use_build_splits=true

# Temporal 协议
python scripts/train_hydra.py protocol=temporal use_build_splits=true

# Brand-OOD 协议
python scripts/train_hydra.py protocol=brand_ood use_build_splits=true
```

#### 一键运行（三协议）
```bash
# Linux/Mac
bash scripts/run_all_protocols.sh

# Windows PowerShell
.\scripts\run_all_protocols.ps1
```

#### 自定义参数
```bash
# 使用本地配置（快速测试）
python scripts/train_hydra.py protocol=random use_build_splits=true +profiles/local

# 自定义batch size和epochs
python scripts/train_hydra.py protocol=random use_build_splits=true train.bs=128 train.epochs=50

# 使用WandB日志
python scripts/train_hydra.py protocol=random use_build_splits=true logger=wandb
```

#### 超参数搜索（多运行）
```bash
# 学习率和dropout网格搜索
python scripts/train_hydra.py -m \
  protocol=random \
  use_build_splits=true \
  train.lr=1e-5,2e-5,5e-5 \
  model.dropout=0.1,0.2,0.3
```

---

### 4.3 Callbacks（回调系统）

```python
# 1. ExperimentResultsCallback
# 位置: src/utils/callbacks.py
# 功能: 保存实验配置和元数据到 results/ 目录

# 2. TestPredictionCollector
# 位置: src/utils/callbacks.py
# 功能: 收集测试集预测（y_true, y_prob）供可视化使用

# 3. ProtocolArtifactsCallback
# 位置: src/utils/protocol_artifacts.py
# 功能: 生成协议四件套产物
#   - roc_{protocol}.png
#   - calib_{protocol}.png
#   - splits_{protocol}.csv
#   - metrics_{protocol}.json

# 4. DocumentationCallback
# 位置: src/utils/doc_callback.py
# 功能: 自动追加实验记录到项目文档
```

---

## 5. 推理预测

### 5.1 预测脚本

```python
# 位置: scripts/predict.py

# 单URL预测
python scripts/predict.py \
  --checkpoint experiments/url_only/checkpoints/url-only-best.ckpt \
  --url "https://example.com/login"

# 批量预测
python scripts/predict.py \
  --checkpoint experiments/url_only/checkpoints/url-only-best.ckpt \
  --test data/processed/url_test.csv \
  --out predictions.csv
```

### 5.2 预测流程

```python
# 1. 加载模型
model = UrlOnlyModule.load_from_checkpoint(checkpoint_path)
model.eval()

# 2. 编码URL
input_ids = encode_url(url, max_len=256, vocab_size=128, pad_id=0)
input_tensor = torch.tensor([input_ids])

# 3. 预测
with torch.no_grad():
    logits = model.predict_logits(input_tensor)
    probs = torch.softmax(logits, dim=1)
    pred_class = logits.argmax(dim=1).item()
    confidence = probs[0, pred_class].item()

# 4. 输出
print(f"预测类别: {pred_class} (0=合法, 1=钓鱼)")
print(f"置信度: {confidence:.4f}")
```

---

## 6. 工具验证

### 6.1 产物验证工具

```python
# 位置: tools/check_artifacts_url_only.py

# 自动验证最新实验
python tools/check_artifacts_url_only.py

# 验证特定实验
python tools/check_artifacts_url_only.py experiments/url_random_20251022_120000
```

**验证项**:
1. ✅ 四件套文件存在性
2. ✅ `splits_{protocol}.csv` 13列完整性
3. ✅ `metrics_{protocol}.json` schema完整性
4. ✅ ECE bins范围合理性 [3, 15]
5. ✅ 协议特定验证
   - brand_ood: `brand_intersection_ok == true`
   - temporal: `tie_policy == "left-closed"`

---

### 6.2 数据一致性检查

```bash
# 检查数据重叠
python check_overlap.py

# 验证数据schema
python scripts/validate_data_schema.py
```

---

## 7. 实验产出

### 7.1 实验目录结构

```
experiments/
└── url_{protocol}_{timestamp}/          # 单次实验目录
    ├── config/                          # 配置备份
    │   └── config.yaml
    │
    ├── checkpoints/                     # 模型检查点
    │   └── best-epoch=X-val_loss=Y.ckpt
    │
    ├── results/                         # 实验结果 ⭐
    │   ├── roc_{protocol}.png           # ROC曲线图
    │   ├── calib_{protocol}.png         # 校准曲线图（含ECE标注）
    │   ├── splits_{protocol}.csv        # 数据分割统计表（13列）
    │   ├── metrics_{protocol}.json      # 指标JSON（9个key）
    │   └── implementation_report.md     # 实现报告
    │
    └── lightning_logs/                  # PyTorch Lightning日志
        └── version_0/
            └── metrics.csv              # 训练过程指标
```

---

### 7.2 四件套详解

#### A. ROC曲线图 (`roc_{protocol}.png`)

```python
# 生成: src/utils/visualizer.py :: save_roc_curve()

内容:
- X轴: False Positive Rate (FPR)
- Y轴: True Positive Rate (TPR)
- 曲线: ROC curve
- 标注: AUC = 0.xxxx
- 基线: 对角线（随机分类器）
```

#### B. 校准曲线图 (`calib_{protocol}.png`)

```python
# 生成: src/utils/visualizer.py :: save_calibration_curve()

内容:
- X轴: Mean Predicted Probability
- Y轴: Fraction of Positives
- 曲线: Calibration curve (bins)
- 标注: ECE = 0.xxxx
- 警告: "⚠️ Small sample, bins reduced to N" (如果 bins < 10)
- 基线: 对角线（完美校准）
```

#### C. 数据分割统计表 (`splits_{protocol}.csv`)

**必需13列**:

| 列名 | 类型 | 说明 | 示例 |
|-----|------|------|------|
| split | str | 分割名称 | train, val, test |
| count | int | 样本数 | 7000 |
| pos_count | int | 正样本数（钓鱼） | 3500 |
| neg_count | int | 负样本数（合法） | 3500 |
| brand_unique | int | 唯一品牌数 | 25 |
| brand_set | str | 品牌列表（前10个） | ['google', 'paypal', ...] |
| timestamp_min | str | 最早时间戳 | 2023-01-01 00:00:00 |
| timestamp_max | str | 最晚时间戳 | 2023-12-31 23:59:59 |
| source_counts | str | 数据源统计 | {'source_a': 1000, ...} |
| brand_intersection_ok | bool | 品牌不重叠（brand_ood） | true/false |
| tie_policy | str | 时间戳并列策略（temporal） | left-closed |
| brand_normalization | str | 品牌归一化方法（brand_ood） | strip+lower |
| downgraded_to | str | 降级协议（如有） | random / "" |

**生成**: `src/utils/splits.py :: write_split_table()`

#### D. 指标JSON (`metrics_{protocol}.json`)

**必需字段**:

```json
{
  "accuracy": 0.9234,           // 准确率
  "auroc": 0.9567,              // AUROC (pos_label=1)
  "f1_macro": 0.9201,           // 宏平均F1
  "nll": 0.1823,                // 负对数似然
  "ece": 0.0234,                // 期望校准误差
  "ece_bins_used": 10,          // ECE计算使用的bins数
  "positive_class": "phishing", // 正类名称

  "artifacts": {
    "roc_path": "results/roc_random.png",
    "calib_path": "results/calib_random.png",
    "splits_path": "results/splits_random.csv"
  },

  "warnings": {
    "downgraded_reason": null   // 降级原因（如有）
  }
}
```

**生成**: `src/utils/protocol_artifacts.py :: ProtocolArtifactsCallback`

---

### 7.3 实验跟踪

```python
# 位置: src/utils/experiment_tracker.py

class ExperimentTracker:
    """
    实验管理工具

    功能:
    1. 创建唯一实验目录: experiments/{name}_{timestamp}
    2. 保存配置: config/config.yaml
    3. 创建子目录: checkpoints/, results/
    4. 记录实验元数据
    """

    def __init__(self, cfg, exp_name):
        self.exp_dir = f"experiments/{exp_name}_{timestamp}"
        self.config_dir = self.exp_dir / "config"
        self.results_dir = self.exp_dir / "results"
        self.checkpoints_dir = self.exp_dir / "checkpoints"

        # 创建目录
        self.exp_dir.mkdir(parents=True)
        self.config_dir.mkdir()
        self.results_dir.mkdir()

        # 保存配置
        OmegaConf.save(cfg, self.config_dir / "config.yaml")
```

---

## 8. 文档系统

### 8.1 URL模块文档

```
# 快速参考
URL_ONLY_QUICKREF.md         # 快速命令参考卡
URL_ONLY_CLOSURE_GUIDE.md    # 收官指南

# 详细指南
docs/QUICKSTART_MLOPS_PROTOCOLS.md  # 三协议快速开始
docs/DATA_README.md                 # 数据说明
docs/WANDB_GUIDE.md                 # WandB集成指南
QUICK_START_DOCS.md                 # 项目快速开始

# 实现报告
IMPLEMENTATION_REPORT.md     # MLOps实现报告
CHANGES_SUMMARY.md           # 变更总结
FINAL_SUMMARY_CN.md          # 项目总结（中文）
```

### 8.2 自动文档追加

```python
# 位置: src/utils/documentation.py, src/utils/doc_callback.py

# 功能: 训练完成后自动追加实验记录到项目文档

# 启用方式
python scripts/train_hydra.py \
  protocol=random \
  use_build_splits=true \
  logging.auto_append_docs=true \
  logging.append_to_summary=true
```

---

## 9. 完整数据流图

```
┌──────────────────────────────────────────────────────────────────┐
│                        1. 数据准备                               │
└──────────────────────────────────────────────────────────────────┘
                               ↓
        data/raw/{dataset, fish_dataset}/*.csv
                               ↓
         [scripts/create_master_csv.py]
                               ↓
              data/processed/master.csv
                               ↓
┌──────────────────────────────────────────────────────────────────┐
│                        2. 数据分割                               │
└──────────────────────────────────────────────────────────────────┘
                               ↓
              [UrlDataModule.setup() + build_splits()]
                   ↙         ↓         ↘
        url_train.csv   url_val.csv   url_test.csv
                   ↘         ↓         ↙
┌──────────────────────────────────────────────────────────────────┐
│                        3. 数据加载                               │
└──────────────────────────────────────────────────────────────────┘
                               ↓
                    [UrlDataset: 字符级编码]
                               ↓
                  DataLoader (batch: tuple格式)
                               ↓
┌──────────────────────────────────────────────────────────────────┐
│                        4. 模型训练                               │
└──────────────────────────────────────────────────────────────────┘
                               ↓
              [UrlOnlyModule: Encoder + Classifier]
                               ↓
           ┌─────────────────┼─────────────────┐
           ↓                 ↓                 ↓
   training_step()   validation_step()   test_step()
           ↓                 ↓                 ↓
      train_loss        val_loss/acc       test_metrics
                               ↓
┌──────────────────────────────────────────────────────────────────┐
│                        5. 产物生成                               │
└──────────────────────────────────────────────────────────────────┘
                               ↓
         [ProtocolArtifactsCallback.on_test_end()]
                               ↓
           ┌──────────┬────────┼────────┬──────────┐
           ↓          ↓        ↓        ↓          ↓
    roc_{p}.png  calib_{p}.png  splits_{p}.csv  metrics_{p}.json
                               ↓
           [ResultVisualizer.create_all_plots()]
                               ↓
              experiments/{name}_{timestamp}/results/
                               ↓
┌──────────────────────────────────────────────────────────────────┐
│                        6. 验证检查                               │
└──────────────────────────────────────────────────────────────────┘
                               ↓
              [tools/check_artifacts_url_only.py]
                               ↓
                    ✅ 验证通过 / ❌ 发现问题
```

---

## 10. 常用命令速查

### 10.1 数据准备

```bash
# 生成主数据集
python scripts/create_master_csv.py

# 检查数据
ls -lh data/processed/*.csv
head -n 5 data/processed/master.csv
```

### 10.2 训练运行

```bash
# 单协议（完整训练）
python scripts/train_hydra.py protocol=random use_build_splits=true

# 快速测试（10%数据，5 epochs）
python scripts/train_hydra.py protocol=random use_build_splits=true +profiles/local

# 三协议一键运行
bash scripts/run_all_protocols.sh  # Linux/Mac
.\scripts\run_all_protocols.ps1    # Windows
```

### 10.3 验证检查

```bash
# 验证最新实验
python tools/check_artifacts_url_only.py

# 验证特定实验
python tools/check_artifacts_url_only.py experiments/url_random_20251022_120000

# 查看实验列表
ls -lt experiments/
```

### 10.4 推理预测

```bash
# 单URL预测
python scripts/predict.py \
  --checkpoint experiments/url_only/checkpoints/url-only-best.ckpt \
  --url "https://suspicious-site.com/login"

# 批量预测
python scripts/predict.py \
  --checkpoint experiments/url_only/checkpoints/url-only-best.ckpt \
  --test data/new_urls.csv \
  --out predictions.csv
```

---

## 11. 故障排除

### 问题1: 缺少 master.csv
```bash
# 解决: 运行数据准备脚本
python scripts/create_master_csv.py
```

### 问题2: 缺少 splits_*.csv
```bash
# 解决: 确保启用 use_build_splits
python scripts/train_hydra.py protocol=random use_build_splits=true
```

### 问题3: 校准图没有ECE标注
```bash
# 检查: src/utils/visualizer.py 第529-532行
# 应该有: ax.text(0.05, 0.95, f"ECE = {ece_value:.4f}", ...)
```

### 问题4: brand_intersection_ok 为空
```bash
# 原因: master.csv 缺少 brand 列
# 解决: 确保原始数据包含品牌信息
# 对于 brand_ood，至少需要3个不同品牌
```

### 问题5: 训练速度慢
```bash
# 减少数据量（快速测试）
python scripts/train_hydra.py protocol=random use_build_splits=true +profiles/local

# 减少workers
python scripts/train_hydra.py protocol=random use_build_splits=true data.num_workers=0

# 使用更小的batch size
python scripts/train_hydra.py protocol=random use_build_splits=true train.bs=16
```

---

## 12. 性能指标参考

### 12.1 模型规模

| 组件 | 参数量 | 说明 |
|-----|--------|------|
| Embedding | 16K | 128×128 |
| BiLSTM | ~200K | 2层×双向×128 |
| Projection | 65K | 256×256 |
| Classifier | 512 | 256×2 |
| **总计** | **~282K** | 轻量级模型 |

### 12.2 训练时间参考

| 配置 | 数据量 | Epochs | 硬件 | 时间 |
|-----|--------|--------|------|------|
| Local | 10% | 5 | CPU | ~2分钟 |
| Server | 100% | 50 | CPU | ~30分钟 |
| Server | 100% | 50 | GPU | ~10分钟 |

### 12.3 性能基线

| 协议 | Accuracy | AUROC | F1-macro | ECE |
|-----|----------|-------|----------|-----|
| Random | ~0.92 | ~0.95 | ~0.91 | <0.05 |
| Temporal | ~0.89 | ~0.93 | ~0.88 | <0.06 |
| Brand-OOD | ~0.85 | ~0.90 | ~0.84 | <0.08 |

*注: 实际性能取决于数据质量和分布*

---

## 13. 扩展与定制

### 13.1 添加新协议

```python
# 1. 在 src/utils/splits.py 添加新分割函数
def _custom_split(df, train_ratio, val_ratio, test_ratio):
    # 实现自定义分割逻辑
    ...
    return train_df, val_df, test_df

# 2. 在 build_splits() 添加协议分支
if protocol == "custom":
    train_df, val_df, test_df = _custom_split(...)
    metadata["custom_field"] = "..."
```

### 13.2 修改模型架构

**⚠️ 警告**: URL编码器架构已锁定，修改需要:

1. 移除断言保护: `src/systems/url_only_module.py` 第38-43行
2. 修改配置: `configs/model/url_encoder.yaml`
3. 重新训练所有协议
4. 更新文档说明修改原因

### 13.3 添加新指标

```python
# 1. 在 src/utils/metrics.py 实现新指标
def compute_custom_metric(y_true, y_pred):
    ...
    return metric_value

# 2. 在 UrlOnlyModule 添加指标计算
def on_test_epoch_end(self):
    custom_metric = compute_custom_metric(...)
    self.log("test_custom", custom_metric)

# 3. 在 ProtocolArtifactsCallback 添加到metrics.json
metrics_dict["custom_metric"] = float(logged_metrics.get("test_custom", 0.0))
```

---

## 14. 依赖环境

### 14.1 核心依赖

```txt
# requirements.txt (核心)
torch>=1.13.0
pytorch-lightning>=2.0.0
pandas>=1.5.0
numpy>=1.23.0
omegaconf>=2.3.0
hydra-core>=1.3.0

# 可视化（可选）
matplotlib>=3.6.0
seaborn>=0.12.0

# 日志（可选）
wandb>=0.13.0
tensorboard>=2.11.0
```

### 14.2 环境配置

```bash
# 使用 conda（推荐）
conda env create -f environment.yml
conda activate uaam-phish

# 或使用 pip
pip install -r requirements.txt

# 或开发模式安装
pip install -e .
```

---

## 15. 项目文件索引

### 核心代码 (src/)
- `src/data/url_dataset.py` - URL数据集类
- `src/datamodules/url_datamodule.py` - Lightning数据模块
- `src/models/url_encoder.py` - BiLSTM编码器
- `src/systems/url_only_module.py` - 训练系统模块
- `src/utils/splits.py` - 数据分割工具
- `src/utils/metrics.py` - 指标计算（ECE, NLL）
- `src/utils/visualizer.py` - 可视化工具
- `src/utils/protocol_artifacts.py` - 产物生成回调
- `src/utils/callbacks.py` - 其他回调
- `src/utils/experiment_tracker.py` - 实验跟踪

### 脚本 (scripts/)
- `scripts/train_hydra.py` - Hydra训练脚本（主入口）
- `scripts/train.py` - 简单训练脚本（旧版）
- `scripts/predict.py` - 预测脚本
- `scripts/create_master_csv.py` - 生成主数据集
- `scripts/build_master_and_splits.py` - 数据构建（DVC版）
- `scripts/run_all_protocols.sh` - 一键运行脚本（Linux/Mac）
- `scripts/run_all_protocols.ps1` - 一键运行脚本（Windows）

### 配置 (configs/)
- `configs/config.yaml` - 主配置
- `configs/data/url_only.yaml` - URL数据配置
- `configs/model/url_encoder.yaml` - 编码器配置
- `configs/experiment/url_baseline.yaml` - 基线实验配置
- `configs/trainer/*.yaml` - 训练器配置
- `configs/logger/*.yaml` - 日志配置

### 工具 (tools/)
- `tools/check_artifacts_url_only.py` - 产物验证工具

### 文档 (docs/ & root)
- `URL_ONLY_QUICKREF.md` - 快速参考
- `URL_ONLY_CLOSURE_GUIDE.md` - 收官指南
- `docs/QUICKSTART_MLOPS_PROTOCOLS.md` - 协议快速开始
- `IMPLEMENTATION_REPORT.md` - 实现报告
- `CHANGES_SUMMARY.md` - 变更总结
- `FINAL_SUMMARY_CN.md` - 项目总结

---

## 16. 许可证与引用

### 项目信息
- **项目名称**: UAAM-Phish (URL-Aware Anti-phishing Model)
- **模块**: URL-Only Baseline
- **架构**: 字符级 2层BiLSTM (256维输出)

### 引用
如果使用本项目，请引用相关论文（待补充）

---

---

## 🚀 快速运行命令

### 一键运行（推荐）
```bash
# Linux/Mac
bash scripts/run_all_protocols.sh

# Windows PowerShell
.\scripts\run_all_protocols.ps1
```

### 单协议运行
```bash
# Random
python scripts/train_hydra.py protocol=random use_build_splits=true

# Temporal
python scripts/train_hydra.py protocol=temporal use_build_splits=true

# Brand-OOD
python scripts/train_hydra.py protocol=brand_ood use_build_splits=true
```

### 验证产物
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

## ✅ 验证清单

### 四件套文件存在性
- ✅ `roc_{protocol}.png` - ROC曲线
- ✅ `calib_{protocol}.png` - 校准曲线（含ECE）
- ✅ `splits_{protocol}.csv` - 分割统计
- ✅ `metrics_{protocol}.json` - 完整指标

### splits_{protocol}.csv 列完整性（13列）
- split, count, pos_count, neg_count, brand_unique, brand_set, timestamp_min, timestamp_max, source_counts, brand_intersection_ok, tie_policy, brand_normalization, downgraded_to

### metrics_{protocol}.json schema 完整性
- accuracy, auroc, f1_macro, nll, ece, ece_bins_used, positive_class, artifacts, warnings

### ECE bins 范围合理性 [3, 15]
- 自适应计算：`max(3, min(15, floor(sqrt(N)), 10))`

### 协议特定验证
- brand_ood 的 brand_intersection_ok
- temporal 的 tie_policy

---

**文档版本**: 1.0
**最后更新**: 2025-10-22
**维护者**: AI Assistant

---

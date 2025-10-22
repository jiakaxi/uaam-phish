# UAAM-Phish 项目根目录结构说明

> **Last Updated:** 2025-10-21
> **版本:** 0.1 (MVP - URL-only 阶段)

本文档详细描述了 UAAM-Phish 项目的完整目录结构、每个文件/目录的用途及其在项目中的角色。

---

## 📁 项目概览

UAAM-Phish 是一个基于深度学习的钓鱼网站检测系统，使用 URL、HTML 和截图等多模态数据进行检测。当前版本是 MVP（最小可行产品），实现了基于 BERT/RoBERTa 的 URL 分类器。

**技术栈:**
- PyTorch + PyTorch Lightning (深度学习框架)
- Transformers (预训练语言模型)
- OmegaConf (配置管理)
- scikit-learn (数据划分和评估)

---

## 📂 根目录文件

### `.gitignore`
**用途:** Git 版本控制忽略规则
**内容:** 定义不需要纳入版本控制的文件和目录（缓存文件、数据文件、日志、模型检查点等）

### `README.md`
**用途:** 项目主文档
**内容:**
- 项目简介和技术栈说明
- 安装步骤
- 数据准备指南
- 训练和测试命令
- 下一步计划

### `requirements.txt`
**用途:** Python 依赖包列表
**内容:** 指定项目所需的 Python 包及其版本
```
torch>=2.2
pytorch-lightning>=2.3
transformers>=4.41
torchmetrics>=1.0
scikit-learn>=1.4
pandas>=2.1
numpy>=1.26
omegaconf>=2.3
```

### `environment.yml`
**用途:** Conda 环境配置文件（待填充）
**状态:** 占位符，用于未来的 Conda 环境管理

### `setup.py`
**用途:** Python 包安装配置
**功能:**
- 定义包名称 `uaam-phish`
- 自动发现 src/ 下的所有 Python 包
- 声明核心依赖

### `Makefile`
**用途:** 项目自动化命令（待填充）
**计划功能:**
- 环境设置
- 数据预处理
- 模型训练
- 测试和评估
- 代码质量检查

### `check_overlap.py`
**用途:** 数据集重叠检查工具
**功能:** 检查 train/val/test 数据集之间是否存在 URL 重叠，确保数据划分的有效性

---

## 📂 `configs/` - 配置文件目录

项目使用 OmegaConf 进行分层配置管理，支持配置合并和环境变量替换。

### `configs/default.yaml`
**用途:** 主配置文件，定义所有默认参数
**包含:**
- **model:** 模型配置（预训练模型名称、dropout）
- **paths:** 数据路径（支持 `DATA_ROOT` 环境变量）
- **hardware:** 硬件配置（accelerator, devices, precision, num_workers）
- **data:** 数据配置（列名、最大长度、采样比例）
- **train:** 训练参数（epochs, lr, bs, pos_weight, seed）
- **eval:** 评估参数（监控指标、早停耐心值）

### `configs/profiles/local.yaml`
**用途:** 本地开发环境配置
**特点:**
- CPU 模式
- 小批量大小 (bs=4)
- 无并行数据加载 (num_workers=0)
- 适合快速迭代和调试

### `configs/profiles/server.yaml`
**用途:** 服务器/GPU 环境配置
**特点:**
- GPU 加速
- 混合精度训练 (16-mixed)
- 大批量大小 (bs=32)
- 多进程数据加载 (num_workers=8)
- 支持多 GPU 分布式训练 (DDP)

### `configs/base.yaml` *(新增空文件)*
**计划用途:** 未来的基础配置，可能用于不同实验的公共配置

### `configs/encoders.yaml` *(新增空文件)*
**计划用途:** 编码器配置（URL、HTML、图像编码器的专用配置）

### `configs/train.yaml` *(新增空文件)*
**计划用途:** 训练专用配置

### `configs/eval.yaml` *(新增空文件)*
**计划用途:** 评估专用配置

### `configs/hparams.yaml` *(新增空文件)*
**计划用途:** 超参数搜索配置

---

## 📂 `data/` - 数据目录

### `data/raw/`
**用途:** 原始数据存储
**内容:**
- `dataset/` - 良性网站数据集目录
- `fish_dataset/` - 钓鱼网站数据集目录
- `collection.log` - 数据收集日志
- `metadata.json` - 数据集元数据
- `duplicate_urls.txt` - 重复 URL 列表
- `failed_urls.json/txt` - 采集失败的 URL 记录

**数据结构:** 每个样本可能包含:
- `url.txt` - URL 文本
- `html.html` - 网页 HTML 源码
- `shot.png` - 网页截图

### `data/processed/`
**用途:** 处理后的训练数据
**文件:**
- `master.csv` - 主数据表，包含所有样本的完整信息
  - 列: `id`, `stem`, `label`, `url_text`, `html_path`, `img_path`, `domain`, `source`, `split`
- `train.csv` - 训练集 (仅包含 `url_text` 和 `label`)
- `val.csv` - 验证集
- `test.csv` - 测试集

---

## 📂 `scripts/` - 可执行脚本

### `scripts/build_master_and_splits.py`
**用途:** 从原始数据构建主数据表和训练/验证/测试划分
**功能:**
- 扫描 `data/raw/` 下的数据集目录
- 提取 URL、HTML 路径、图片路径
- 解析域名 (domain-aware splitting)
- 使用 `GroupShuffleSplit` 进行分组划分（避免同域名样本泄漏）
- 生成 `master.csv` 和三个划分文件

**用法:**
```bash
python scripts/build_master_and_splits.py \
  --benign data/raw/dataset \
  --phish data/raw/fish_dataset \
  --outdir data/processed \
  --val_size 0.15 --test_size 0.15
```

### `scripts/preprocess.py`
**用途:** 简单的数据预处理脚本
**功能:** 从单个 CSV 文件进行随机划分（不考虑域名分组）
**适用场景:** 快速测试和简单数据集

### `scripts/train.py`
**用途:** 模型训练主脚本
**功能:**
1. 加载配置（支持 `--profile` 参数选择环境配置）
2. 设置随机种子
3. 初始化数据模块和模型系统
4. 配置回调函数（EarlyStopping, ModelCheckpoint）
5. 启动训练和测试

**用法:**
```bash
# 本地开发
python scripts/train.py --profile local

# 服务器训练
python scripts/train.py --profile server
```

---

## 📂 `src/` - 源代码目录

项目核心代码，采用模块化设计。

### `src/__init__.py`
**用途:** 包初始化文件

### `src/datamodules/` - 数据模块

#### `src/datamodules/url_datamodule.py`
**内容:**
- `UrlDataset`: PyTorch Dataset 类
  - 从 CSV 加载数据
  - 使用 Transformers tokenizer 进行文本编码
  - 支持采样（`sample_fraction`）用于快速迭代
- `UrlDataModule`: PyTorch Lightning DataModule
  - 管理 train/val/test 三个数据集
  - 提供 DataLoader 配置

### `src/models/` - 模型定义

#### `src/models/url_encoder.py`
**内容:**
- `UrlBertEncoder`: URL 编码器
  - 基于 BERT/RoBERTa 的预训练模型
  - 提取 [CLS] token 表示作为 URL 嵌入
  - 可配置 dropout 和投影层

### `src/systems/` - Lightning 系统模块

#### `src/systems/url_only_module.py`
**内容:**
- `UrlOnlySystem`: PyTorch Lightning Module
  - 完整的训练/验证/测试流程
  - 二分类任务（良性 vs 钓鱼）
  - 损失函数: BCEWithLogitsLoss（支持 pos_weight 处理类别不平衡）
  - 评估指标: F1, AUROC, FPR
  - 自动阈值优化（在验证集上扫描最佳 F1 阈值）
  - 优化器: AdamW

### `src/utils/` - 工具函数

#### `src/utils/seed.py`
**用途:** 随机种子设置工具
**功能:** 确保 Python、NumPy、PyTorch 等库的随机性可复现

### `src/data/` *(新增目录)*
**计划用途:** 数据处理工具（特征提取、数据增强等）

### `src/modules/` *(新增目录)*
**计划用途:** 核心模块（Uncertainty、Consistency、Fusion 等）

### `src/evaluation/` *(新增目录)*
**计划用途:** 评估工具和指标计算

### `src/cli/` *(新增目录)*
**计划用途:** 命令行接口工具

---

## 📂 `tests/` - 测试目录

所有测试文件（当前为占位符）。

### `tests/test_data.py` *(待实现)*
**计划测试:** 数据加载、预处理、DataModule 功能

### `tests/test_uncertainty.py` *(待实现)*
**计划测试:** 不确定性估计模块

### `tests/test_consistency.py` *(待实现)*
**计划测试:** 一致性检查模块

### `tests/test_fusion.py` *(待实现)*
**计划测试:** 多模态融合模块

---

## 📂 `docs/` - 文档目录

### `docs/ROOT_STRUCTURE.md`
**本文档:** 项目结构说明

### `docs/RULES.md` *(待填充)*
**计划内容:** 项目开发规则和最佳实践

### `docs/DATA_README.md` *(待填充)*
**计划内容:** 数据集详细说明、格式、统计信息

### `docs/EXPERIMENTS.md` *(待填充)*
**计划内容:** 实验记录和结果追踪

### `docs/TESTING_GUIDE.md` *(待填充)*
**计划内容:** 测试指南和测试用例编写规范

### `docs/DEBUG_LOGGING.md` *(待填充)*
**计划内容:** 调试和日志记录指南

### `docs/CODE_REVIEW_SUB_AGENT_PROMPT.md` *(待填充)*
**计划内容:** 代码审查提示和检查清单

### `docs/adr/` - 架构决策记录 (Architecture Decision Records)

#### `docs/adr/0001-choose-uncertainty-method.md` *(待填充)*
**计划内容:** 不确定性估计方法选择的决策记录

### `docs/specs/` - 技术规格说明

#### `docs/specs/uncertainty.md` *(待填充)*
**计划内容:** 不确定性模块的技术规格

#### `docs/specs/consistency.md` *(待填充)*
**计划内容:** 一致性检查模块的技术规格

#### `docs/specs/fusion_rcaf.md` *(待填充)*
**计划内容:** RCAF 融合方法的技术规格

### `docs/impl/` - 实现文档

#### `docs/impl/uncertainty_impl.md` *(待填充)*
**计划内容:** 不确定性模块实现细节

#### `docs/impl/consistency_impl.md` *(待填充)*
**计划内容:** 一致性模块实现细节

#### `docs/impl/fusion_rcaf_impl.md` *(待填充)*
**计划内容:** RCAF 融合实现细节

### `docs/AI_CONVERSATIONS/` - AI 对话记录

#### `docs/AI_CONVERSATIONS/2025-10-21_u_module_implementation.md` *(待填充)*
**用途:** 记录与 AI 助手的重要对话和决策过程

---

## 📂 `.github/` - GitHub 配置

### `.github/workflows/ci.yml` *(待实现)*
**计划内容:** GitHub Actions CI/CD 配置
- 自动化测试
- 代码质量检查 (linting)
- 依赖安全检查

---

## 📂 `experiments/` - 实验结果目录 ⭐

**重要:** 项目集成了完整的实验管理系统，每次训练后自动保存所有结果！

### 目录结构
```
experiments/
├── url_mvp_20251021_143022/        # 实验名称_时间戳
│   ├── config.yaml                  # ✅ 实验配置（自动保存）
│   ├── SUMMARY.md                   # ✅ 实验总结（训练后生成）
│   ├── results/                     # 📊 实验结果
│   │   ├── metrics_final.json       # ✅ 最终指标（立即保存）
│   │   ├── training_curves.png      # ✅ 训练曲线（立即生成）
│   │   ├── confusion_matrix.png     # ✅ 混淆矩阵（立即生成）
│   │   ├── roc_curve.png            # ✅ ROC曲线（立即生成）
│   │   └── threshold_analysis.png   # ✅ 阈值分析（立即生成）
│   ├── logs/                        # 📝 日志文件
│   │   ├── train.log                # ✅ 训练日志（实时记录）
│   │   └── metrics_history.csv      # ✅ 指标历史
│   └── checkpoints/                 # 💾 模型检查点
│       └── best-epoch=X-val_auroc=Y.ckpt  # ✅ 最佳模型
└── roberta_baseline_20251021_150033/
    └── ...
```

### 自动保存内容

#### 1. **配置文件** (`config.yaml`)
- 完整的实验配置，确保可复现

#### 2. **指标文件** (`results/metrics_final.json`)
```json
{
  "experiment": "url_mvp_20251021_143022",
  "timestamp": "2025-10-21T14:35:42",
  "metrics": {
    "test/loss": 0.1234,
    "test/f1": 0.9567,
    "test/auroc": 0.9823,
    "test/fpr": 0.0234
  }
}
```

#### 3. **可视化图表** (`results/*.png`)
- **training_curves.png**: 训练曲线（Loss, F1, AUROC, FPR）
- **confusion_matrix.png**: 混淆矩阵 + 性能指标
- **roc_curve.png**: ROC 曲线 + AUC
- **threshold_analysis.png**: 最佳阈值分析

#### 4. **训练日志** (`logs/train.log`)
实时记录训练过程，包含每个 epoch 的指标

#### 5. **实验总结** (`SUMMARY.md`)
Markdown 格式的实验总结，包含配置和结果

#### 6. **模型检查点** (`checkpoints/`)
从 `lightning_logs/` 复制的最佳模型权重

### 使用方法

```bash
# 运行实验（自动保存所有结果）
python scripts/train.py --profile server --exp_name my_experiment

# 对比实验结果
python scripts/compare_experiments.py --latest 5

# 查找最佳实验
python scripts/compare_experiments.py --find_best --metric auroc
```

**详细说明:** 请参阅 [实验管理指南](EXPERIMENTS.md) 和 [快速启动指南](QUICK_START_EXPERIMENT.md)

**注意:** 此目录已在 `.gitignore` 中，不会被纳入版本控制

---

## 📂 `lightning_logs/` - PyTorch Lightning 日志

PyTorch Lightning 自动生成的训练日志目录（原始日志）。

**结构:**
```
lightning_logs/
├── version_0/
│   ├── checkpoints/        # 模型检查点
│   │   └── epoch=2-step=57.ckpt
│   ├── hparams.yaml        # 超参数记录
│   └── metrics.csv         # 训练指标
├── version_1/
...
```

**说明:**
- 每次训练运行创建一个新的 `version_X/` 目录
- `checkpoints/` 存储模型权重（根据配置保存最佳模型）
- `hparams.yaml` 记录训练时的超参数配置
- `metrics.csv` 记录每个 epoch 的指标（loss, F1, AUROC, FPR 等）

**注意:**
- 此目录的内容会被自动复制到 `experiments/` 目录
- 已在 `.gitignore` 中，不会被纳入版本控制
- 可以定期清理，重要内容已保存在 `experiments/`

---

## 📂 `uaam_phish.egg-info/` - 包元数据

`pip install -e .` 生成的包安装元数据，不需要手动修改。

**注意:** 已在 `.gitignore` 中忽略

---

## 🔄 典型工作流程

### 1. 数据准备
```bash
# 从原始数据构建训练集
python scripts/build_master_and_splits.py \
  --benign data/raw/dataset \
  --phish data/raw/fish_dataset \
  --outdir data/processed

# 检查数据集重叠
python check_overlap.py
```

### 2. 本地开发和调试
```bash
# 使用本地配置（CPU，小批量）
export DATA_ROOT=./data/processed
python scripts/train.py --profile local
```

### 3. 服务器训练
```bash
# 使用服务器配置（GPU，大批量）
export DATA_ROOT=/data/uaam_phish/processed
python scripts/train.py --profile server
```

### 4. 查看训练结果
- 检查 `lightning_logs/version_X/metrics.csv`
- 加载最佳检查点进行推理

---

## 📝 开发规范

### 代码组织原则
1. **模块化**: 每个功能模块独立，接口清晰
2. **配置驱动**: 所有参数通过配置文件管理，避免硬编码
3. **可复现性**: 使用固定随机种子，记录所有超参数
4. **可扩展性**: 设计时考虑未来多模态扩展

### 配置管理
- 使用 OmegaConf 分层配置
- 环境特定配置使用 profiles
- 敏感路径使用环境变量

### 测试策略
- 单元测试: `tests/test_*.py`
- 集成测试: 端到端训练流程
- 数据验证: 检查数据集质量和划分

---

## 🚀 下一步计划

### MVP 后续扩展
1. **HTML 编码器**: 集成 HTML 内容分析
2. **图像编码器**: 集成网页截图分析
3. **多模态融合**: RCAF (Reliability-Constrained Attention Fusion)
4. **不确定性估计**: Monte Carlo Dropout / Deep Ensembles
5. **一致性检查**: 跨模态一致性验证

### 工程改进
- [ ] 完善单元测试
- [ ] 实现 CI/CD 流程
- [ ] 添加模型推理脚本
- [ ] 优化数据加载性能
- [ ] 实现模型服务化 (API)

---

## 📊 项目状态

**当前版本:** MVP (Milestone 1)
**状态:** ✅ 可运行
**最后更新:** 2025-10-21

### 已完成功能
- ✅ 基于 BERT/RoBERTa 的 URL 分类器
- ✅ 完整的训练/验证/测试流程
- ✅ 多环境配置管理
- ✅ 域名感知的数据划分
- ✅ 评估指标 (F1, AUROC, FPR)
- ✅ 早停和模型检查点保存

### 进行中
- 🔄 文档完善
- 🔄 测试覆盖
- 🔄 多模态编码器实现

---

**维护者:** UAAM-Phish Team
**联系方式:** [项目 GitHub Issues]

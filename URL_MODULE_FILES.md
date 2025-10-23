# URL 模块文件清单

> 按逻辑流程组织，每个文件后面是它的功能说明

---

## 📦 数据处理

### 原始数据
- `data/raw/dataset/` - 合法网站原始数据目录
- `data/raw/fish_dataset/` - 钓鱼网站原始数据目录

### 数据处理脚本
- `scripts/create_master_csv.py` - 合并原始数据生成主数据集 master.csv
- `scripts/build_master_and_splits.py` - DVC版数据构建脚本（合并+分割）
- `scripts/validate_data_schema.py` - 验证数据schema是否符合要求
- `check_overlap.py` - 检查训练/测试数据是否有重叠

### 处理后数据
- `data/processed/master.csv` - 主数据集（所有数据合并后）
- `data/processed/url_train.csv` - 训练集（由 build_splits 自动生成）
- `data/processed/url_val.csv` - 验证集（由 build_splits 自动生成）
- `data/processed/url_test.csv` - 测试集（由 build_splits 自动生成）

---

## 🔧 核心源码

### 数据层 (src/data/)
- `src/data/url_dataset.py` - URL数据集类，实现字符级编码和PyTorch Dataset接口

### 数据模块层 (src/datamodules/)
- `src/datamodules/url_datamodule.py` - Lightning数据模块，封装train/val/test DataLoader，集成build_splits

### 模型层 (src/models/)
- `src/models/url_encoder.py` - 2层双向LSTM编码器，输入URL字符序列，输出256维向量

### 系统层 (src/systems/)
- `src/systems/url_only_module.py` - Lightning训练系统，包含编码器+分类器+指标计算+训练循环

### 工具层 (src/utils/)
- `src/utils/splits.py` - 数据分割工具，实现random/temporal/brand_ood三种协议
- `src/utils/metrics.py` - 指标计算函数（ECE自适应bins、NLL、Accuracy、AUROC、F1）
- `src/utils/visualizer.py` - 可视化工具，生成ROC曲线图和校准曲线图
- `src/utils/protocol_artifacts.py` - 协议产物生成Callback，生成四件套（roc/calib/splits/metrics）
- `src/utils/callbacks.py` - 其他训练回调（实验结果保存、预测收集）
- `src/utils/doc_callback.py` - 自动文档追加Callback
- `src/utils/documentation.py` - 文档工具函数，支持自动追加到SUMMARY和CHANGES
- `src/utils/experiment_tracker.py` - 实验跟踪器，创建实验目录和保存配置
- `src/utils/batch_utils.py` - 批次格式转换工具
- `src/utils/logging.py` - 日志工具
- `src/utils/seed.py` - 随机种子设置工具

---

## ⚙️ 配置文件

### 主配置
- `configs/config.yaml` - Hydra主配置文件，组合所有配置组
- `configs/default.yaml` - 默认配置，包含所有基础设置
- `configs/base.yaml` - 基础配置
- `configs/hparams.yaml` - 超参数配置
- `configs/encoders.yaml` - 编码器选择配置

### 数据配置
- `configs/data/url_only.yaml` - URL数据配置（CSV路径、列名、batch_format、split_ratios）

### 模型配置
- `configs/model/url_encoder.yaml` - URL编码器模型配置（vocab_size、hidden_dim、proj_dim等）

### 训练器配置
- `configs/trainer/default.yaml` - 默认训练器配置
- `configs/trainer/local.yaml` - 本地快速测试配置（10%数据，5 epochs）
- `configs/trainer/server.yaml` - 服务器完整训练配置

### 环境配置
- `configs/profiles/local.yaml` - 本地环境配置（CPU，小数据）
- `configs/profiles/server.yaml` - 服务器环境配置（GPU，完整数据）

### 实验配置
- `configs/experiment/url_baseline.yaml` - URL基线实验配置

### 日志配置
- `configs/logger/csv.yaml` - CSV日志配置
- `configs/logger/tensorboard.yaml` - TensorBoard日志配置
- `configs/logger/wandb.yaml` - Weights & Biases日志配置

---

## 🚀 训练脚本

- `scripts/train_hydra.py` - **主训练脚本**（Hydra配置管理，支持三协议，生成四件套）
- `scripts/train.py` - 简单训练脚本（旧版，不推荐）

### 运行脚本
- `scripts/run_all_protocols.sh` - 一键运行三协议训练（Linux/Mac）
- `scripts/run_all_protocols.ps1` - 一键运行三协议训练（Windows PowerShell）

---

## 🔮 推理预测

- `scripts/predict.py` - 预测脚本，支持单URL预测和批量预测
- `pred_url_test.csv` - 示例预测结果文件

---

## ✅ 验证工具

- `tools/check_artifacts_url_only.py` - 验证实验产物（四件套）是否完整且符合规范

---

## 📊 实验产出

### 实验目录结构
```
experiments/url_{protocol}_{timestamp}/
├── config/
│   └── config.yaml              - 实验配置备份
├── checkpoints/
│   └── best-epoch=X-val_loss=Y.ckpt  - 最佳模型检查点
├── results/
│   ├── roc_{protocol}.png       - ROC曲线图（AUC标注）
│   ├── calib_{protocol}.png     - 校准曲线图（ECE标注）
│   ├── splits_{protocol}.csv    - 数据分割统计表（13列）
│   ├── metrics_{protocol}.json  - 指标JSON（9个key）
│   └── implementation_report.md - 实现报告
└── lightning_logs/
    └── version_0/
        └── metrics.csv          - 训练过程指标（loss、acc等）
```

### 历史实验
- `experiments/url_baseline_test_20251023_014450/` - 历史实验1
- `experiments/url_full_baseline_20251023_014800/` - 历史实验2
- `experiments/url_mvp_20251023_035337/` - 历史实验3
- `experiments/wandb-test_20251022_235012/` - WandB连接测试
- `lightning_logs/version_X/` - Lightning默认日志目录

---

## 📝 文档

### URL模块专属文档
- `URL_MODULE_STRUCTURE.md` - **URL模块完整架构文档**（详细版，刚才生成的）
- `URL_MODULE_FILES.md` - **URL模块文件清单**（本文档，简洁版）
- `URL_ONLY_QUICKREF.md` - URL模块快速参考卡（命令速查）
- `URL_ONLY_CLOSURE_GUIDE.md` - URL模块收官指南（P0任务清单）

### 通用文档
- `README.md` - 项目主README
- `README_WINDOWS.md` - Windows环境特殊说明
- `QUICKSTART.md` - 项目快速开始
- `QUICK_START_DOCS.md` - 快速开始文档汇总
- `QUICK_REFERENCE.md` - 快速参考

### 实现报告
- `IMPLEMENTATION_REPORT.md` - MLOps实现报告
- `CHANGES_SUMMARY.md` - 变更总结（追加式）
- `FINAL_SUMMARY_CN.md` - 项目最终总结（中文）
- `SOLUTION_SUMMARY.md` - 解决方案总结
- `MLOPS_STATUS_REPORT.md` - MLOps状态报告
- `AUTO_APPEND_INTEGRATION_COMPLETE.md` - 自动追加集成完成报告

### 技术文档 (docs/)
- `docs/QUICKSTART_MLOPS_PROTOCOLS.md` - **三协议快速开始指南**
- `docs/DATA_README.md` - 数据说明
- `docs/DATA_SCHEMA.md` - 数据schema定义
- `docs/WANDB_GUIDE.md` - WandB集成指南
- `docs/EXPERIMENTS.md` - 实验管理指南
- `docs/TESTING_GUIDE.md` - 测试指南
- `docs/DEPENDENCIES.md` - 依赖说明
- `docs/DEBUG_LOGGING.md` - 调试日志说明
- `docs/APPEND_DOCUMENTATION_GUIDE.md` - 文档追加指南
- `docs/AUTO_APPEND_USAGE.md` - 自动追加使用说明
- `docs/DOCUMENTATION_STRUCTURE.md` - 文档结构说明
- `docs/DOCUMENTATION_MIGRATION_GUIDE.md` - 文档迁移指南
- `docs/PROJECT_ARCHITECTURE_CN.md` - 项目架构（中文）
- `docs/ARCHITECTURE_CLARIFICATION.md` - 架构说明
- `docs/ROOT_STRUCTURE.md` - 根目录结构说明
- `docs/RULES.md` - 项目规则
- `docs/VALIDATION_REPORT.md` - 验证报告
- `docs/MLOPS_IMPROVEMENTS_2025-10-22.md` - MLOps改进记录
- `docs/EXPERIMENT_SYSTEM_FEATURES.md` - 实验系统特性

### 示例代码 (examples/)
- `examples/append_documentation_example.py` - 文档追加示例
- `examples/document_change_example.py` - 变更文档示例
- `examples/quick_append_demo.py` - 快速追加演示
- `examples/run_protocol_experiments.py` - 协议实验运行示例
- `examples/README.md` - 示例说明

---

## 🧪 测试

- `tests/test_url_dataset.py` - URL数据集测试
- `tests/test_url_encoder.py` - URL编码器测试
- `tests/test_models.py` - 模型测试
- `tests/test_data.py` - 数据处理测试
- `tests/test_config.py` - 配置测试
- `tests/test_consistency.py` - 一致性测试
- `tests/test_fusion.py` - 融合模块测试
- `tests/test_uncertainty.py` - 不确定性测试
- `tests/test_utils.py` - 工具函数测试
- `tests/test_documentation_append.py` - 文档追加功能测试
- `tests/test_mlops_implementation.py` - MLOps实现测试

### 测试脚本
- `test_auto_append.ps1` - 自动追加功能测试（PowerShell）
- `test_mlops_configs.py` - MLOps配置测试
- `test_wandb.py` - WandB集成测试

---

## 📦 项目管理

### 依赖管理
- `requirements.txt` - Python依赖列表（pip）
- `environment.yml` - Conda环境配置
- `setup.py` - Python包安装配置
- `uaam_phish.egg-info/` - 包元数据目录

### 构建工具
- `Makefile` - Make命令（Linux/Mac）
- `Makefile.ps1` - Make命令（Windows PowerShell）

### 版本控制
- `.gitignore` - Git忽略文件配置
- `dvc.yaml` - DVC数据版本控制配置

### 输出目录
- `outputs/2025-10-22/` - Hydra输出目录（按日期）
- `outputs/2025-10-23/` - Hydra输出目录（按日期）

### 文件清单
- `FILES_MANIFEST.md` - 项目文件清单（自动生成）
- `ARCHITECTURE_SUMMARY.md` - 架构总结

---

## 🔑 关键文件速查

### 数据流
```
原始数据 → scripts/create_master_csv.py → data/processed/master.csv
       → src/utils/splits.py (build_splits) → train/val/test.csv
       → src/data/url_dataset.py (字符编码) → DataLoader
```

### 训练流
```
scripts/train_hydra.py (入口)
  ↓
src/datamodules/url_datamodule.py (数据加载)
  ↓
src/systems/url_only_module.py (训练系统)
  ├─ src/models/url_encoder.py (编码器)
  ├─ src/utils/metrics.py (指标)
  └─ src/utils/callbacks.py (回调)
  ↓
src/utils/protocol_artifacts.py (产物生成)
  ├─ src/utils/visualizer.py (ROC/Calib图)
  └─ src/utils/splits.py (splits表)
  ↓
experiments/{name}_{timestamp}/results/ (四件套)
```

### 配置流
```
configs/config.yaml (主配置)
  ├─ configs/data/url_only.yaml (数据)
  ├─ configs/model/url_encoder.yaml (模型)
  ├─ configs/trainer/local.yaml (训练器)
  └─ configs/logger/csv.yaml (日志)
```

---

## 🎯 最常用的文件

### 开发时常看
1. `src/systems/url_only_module.py` - 训练逻辑主入口
2. `src/models/url_encoder.py` - 模型架构
3. `configs/data/url_only.yaml` - 数据配置
4. `configs/model/url_encoder.yaml` - 模型配置

### 运行时常用
1. `scripts/train_hydra.py` - 训练入口
2. `scripts/predict.py` - 预测入口
3. `scripts/run_all_protocols.sh/.ps1` - 批量运行

### 调试时常用
1. `tools/check_artifacts_url_only.py` - 验证产物
2. `check_overlap.py` - 检查数据重叠
3. `scripts/validate_data_schema.py` - 验证数据格式

### 文档时常看
1. `URL_ONLY_QUICKREF.md` - 快速命令参考
2. `docs/QUICKSTART_MLOPS_PROTOCOLS.md` - 协议快速开始
3. `URL_MODULE_STRUCTURE.md` - 完整架构文档（详细）
4. `URL_MODULE_FILES.md` - 本文档（简洁）

---

**更新时间**: 2025-10-22
**总文件数**: 100+ 个与URL模块相关的文件

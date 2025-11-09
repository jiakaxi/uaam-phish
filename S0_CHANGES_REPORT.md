# Git 变更报告

**对比基准**: 上次 Git 提交 (d3089ea)
**报告日期**: 2025-11-08
**工作区**: 新工作区（与原来不同）

---

## 📊 变更概览

### 统计信息
- **修改的文件**: 15 个
- **新增的文件**: 15 个
- **删除的文件**: 0 个
- **总变更行数**: +352, -312

---

## 🆕 新增文件列表（15个）

### 1. 配置和文档文件（5个）

#### 1.1 `WANDB_CONFIG_CHECK.md`
**类型**: 配置检查文档
**描述**: WandB配置检查报告，详细说明：
- 正确的WandB配置格式
- 常见配置错误及修正方法
- 环境变量设置说明（WANDB_API_KEY, WANDB_PROJECT, WANDB_ENTITY）
- 项目实际配置与推荐配置的对比

**关键内容**:
- 检查WANDB_ENTITY格式（不应包含项目路径）
- 说明WANDB_MODE环境变量的限制
- 提供Linux/Mac和Windows的配置示例

---

#### 1.2 `configs/experiment/s0_brandood_earlyconcat.yaml`
**类型**: 实验配置文件
**描述**: S0 Brand-OOD场景下的EarlyConcat模型配置
- 使用`MultimodalBaselineSystem`（早期融合）
- 数据集分割协议：`brand_ood`
- 训练/验证/测试数据路径：`workspace/data/splits/brandood/`
- 学习率：3.0e-4，批次大小：64，训练轮数：50

---

#### 1.3 `configs/experiment/s0_brandood_lateavg.yaml`
**类型**: 实验配置文件
**描述**: S0 Brand-OOD场景下的Late Average模型配置
- 使用`S0LateAverageSystem`（晚期融合）
- 数据集分割协议：`brand_ood`
- 其他配置与earlyconcat版本相同

---

#### 1.4 `configs/experiment/s0_iid_earlyconcat.yaml`
**类型**: 实验配置文件
**描述**: S0 IID场景下的EarlyConcat模型配置
- 使用`MultimodalBaselineSystem`
- 数据集分割协议：`presplit`
- 训练/验证/测试数据路径：`workspace/data/splits/iid/`

---

#### 1.5 `configs/experiment/s0_iid_lateavg.yaml`
**类型**: 实验配置文件
**描述**: S0 IID场景下的Late Average模型配置
- 使用`S0LateAverageSystem`
- 数据集分割协议：`presplit`
- 其他配置与earlyconcat版本相同

---

### 2. 实验脚本文件（4个）

#### 2.1 `scripts/evaluate_s0.py`
**类型**: 评估脚本
**描述**: 从保存的预测CSV文件中聚合评估指标
- **功能**:
  - 读取`workspace/runs`目录下的预测结果
  - 计算准确率、F1、AUROC、Brier分数、ECE、FPR@TPR95等指标
  - 为每个运行生成`eval_summary.json`
  - 生成聚合的CSV文件：`workspace/tables/s0_eval_summary.csv`

**关键特性**:
- 自动扫描所有模型目录和种子运行
- 使用`metrics_v2.py`中的指标计算函数
- 支持批量处理多个实验运行

---

#### 2.2 `scripts/run_s0_experiments.py`
**类型**: 实验编排脚本
**描述**: 编排S0实验（多个模型 × 多个种子）
- **功能**:
  - 支持`iid`和`brandood`两种场景
  - 支持`s0_earlyconcat`和`s0_lateavg`两种模型
  - 支持多个随机种子的批量运行
  - 自动构建Hydra训练命令

**使用示例**:
```bash
python scripts/run_s0_experiments.py --scenario iid --models s0_earlyconcat s0_lateavg --seeds 42 43 44
```

**关键特性**:
- 场景到配置的映射（`SCENARIO_TO_CONFIG`）
- 支持dry-run模式（仅打印命令，不执行）
- 支持Hydra配置覆盖

---

#### 2.3 `scripts/summarize_s0_results.py`
**类型**: 结果汇总脚本
**描述**: 汇总S0评估结果（跨种子）
- **功能**:
  - 收集所有运行的`eval_summary.json`文件
  - 计算每个模型的平均值和标准差
  - 生成汇总表格：`workspace/tables/s0_eval_summary.csv`
  - 生成AUROC柱状图：`workspace/figs/s0_auroc.png`

**关键特性**:
- 按模型分组统计
- 自动生成可视化图表
- 支持多种评估指标（accuracy, f1, auroc, brier, ece, fpr_at_tpr95）

---

#### 2.4 `scripts/validate_s0_quality.py`
**类型**: 质量检查脚本
**描述**: S0资产的质量门检查（数据分割、腐败数据、运行工件）
- **功能**:
  - 检查数据分割文件的列完整性
  - 验证图像腐败数据的SHA256校验和
  - 检查运行目录的完整性（eval_summary.json）
  - 生成质量报告：`workspace/reports/quality_report.json`

**检查项**:
- 数据分割：检查必需列（id, label, url_text, html_path, img_path, brand, timestamp等）
- 腐败数据：验证文件存在性和SHA256匹配
- 运行完整性：检查eval_summary.json文件

---

### 3. 核心系统文件（2个）

#### 3.1 `src/systems/s0_late_avg_system.py`
**类型**: PyTorch Lightning模块
**描述**: S0晚期融合基线系统（统一平均）
- **架构**:
  - URL编码器：2层BiLSTM
  - HTML编码器：BERT-base
  - 视觉编码器：ResNet50
  - 三个独立的分类头（URL/HTML/Visual）
  - 概率平均融合

**关键特性**:
- 三个模态独立编码和分类
- 训练时使用三个损失的均值
- 推理时对三个概率进行平均
- 支持工件生成（ROC曲线、校准图等）
- 支持分离的学习率（BERT: 2e-5, 其他: 3e-4）

**与EarlyConcat的区别**:
- EarlyConcat：早期融合（连接嵌入后分类）
- Late Average：晚期融合（平均概率）

---

#### 3.2 `src/utils/metrics_v2.py`
**类型**: 指标计算工具
**描述**: S0特定的指标辅助函数
- **功能**:
  - `compute_fpr_at_tpr95()`: 计算TPR=95%时的FPR
  - `compute_brier_score()`: 计算Brier分数
  - `compute_ece()`: 计算期望校准误差（固定15个bin）

**关键特性**:
- 使用完整的ROC曲线（`drop_intermediate=False`）
- 固定15个bin以确保比较的一致性
- 返回低样本警告标志

---

### 4. 数据工具文件（4个）

#### 4.1 `tools/corrupt_html.py`
**类型**: HTML腐败工具
**描述**: 为S0实验生成腐败的HTML文件
- **腐败级别**:
  - **L (Low)**: 移除所有`<script>`标签
  - **M (Medium)**: 移除所有HTML标签，截取一半内容
  - **H (High)**: 仅保留前1/3内容，添加噪声注释

**输出**:
- 腐败的HTML文件：`{output_dir}/{LEVEL}/html/{sample_id}.html`
- CSV文件：`test_corrupt_html_{LEVEL}.csv`（包含`html_path_corrupt`和`html_sha256_corrupt`列）

**关键特性**:
- 支持多种腐败级别
- 自动计算SHA256校验和
- 处理缺失HTML路径的情况

---

#### 4.2 `tools/corrupt_img.py`
**类型**: 图像腐败工具
**描述**: 为S0实验生成腐败的截图（L/M/H级别）
- **腐败级别**:
  - **L (Low)**: 高斯模糊（radius=0.8）
  - **M (Medium)**: 颜色抖动（0.6倍）+ 高斯模糊（radius=1.2）
  - **H (High)**: 下采样（1/2）+ 上采样 + 对比度降低（0.5倍）+ 高斯模糊（radius=1.5）

**输出**:
- 腐败的图像文件：`{output_dir}/{LEVEL}/shot/{sample_id}.jpg`
- CSV文件：`test_corrupt_img_{LEVEL}.csv`（包含`img_path_corrupt`和`img_sha256_corrupt`列）

**关键特性**:
- 使用PIL进行图像处理
- 自动处理缺失图像路径（生成灰色占位图）
- JPEG质量：95%

---

#### 4.3 `tools/corrupt_url.py`
**类型**: URL腐败工具
**描述**: 为S0实验生成腐败的URL文本（仅文本，不修改文件）
- **腐败级别**:
  - **L (Low)**: 添加查询参数（`?ref=secure-update`或`&utm_ref=s0`）
  - **M (Medium)**: 替换字符（`.`→`-`，`/`→`//`）
  - **H (High)**: 同形字符替换（a→@/4, o→0, e→3等）+ 插入特殊字符

**输出**:
- CSV文件：`test_corrupt_url_{LEVEL}.csv`（包含`url_text_corrupt`和`corruption_level`列）

**关键特性**:
- 使用同形字符字典进行替换
- 随机插入特殊字符（-, _, ~）
- 仅修改CSV中的文本，不修改实际文件

---

#### 4.4 `tools/split_brandood.py`
**类型**: 数据分割工具
**描述**: 创建Brand-OOD分割（train/val/test_id/test_ood + brand_sets.json）
- **功能**:
  - 选择top-k品牌作为in-domain品牌
  - 其余品牌作为out-of-domain品牌
  - 对in-domain数据进行分层分割（train/val/test_id）
  - 从out-of-domain数据中采样test_ood（比例可配置）

**输出**:
- `train.csv`, `val.csv`, `test_id.csv`, `test_ood.csv`
- `brand_sets.json`（包含`b_ind`和`b_ood`品牌列表）

**关键特性**:
- 品牌归一化（`.strip().lower()`）
- 自动计算`etld_plus_one`（如果缺失）
- 支持可配置的OOD比例（默认25%）
- 分层分割确保标签分布一致

---

#### 4.5 `tools/split_iid.py`
**类型**: 数据分割工具
**描述**: 创建IID分割（70/15/15）
- **功能**:
  - 分层随机分割为train/val/test
  - 默认比例：70%训练，15%验证，15%测试
  - 支持自定义比例

**输出**:
- `train.csv`, `val.csv`, `test.csv`

**关键特性**:
- 品牌归一化
- 自动计算`etld_plus_one`（如果缺失）
- 支持`--check-only`模式（仅验证架构，不写入文件）
- 分层分割确保标签分布一致

---

## 🔄 修改文件列表（15个）

### 1. 核心功能修改

#### 1.1 `src/utils/metrics.py`
**修改内容**:
- **ECE计算改进**:
  - 从自适应bin改为固定15个bin
  - 第一个bin改为左闭区间（`>=`而不是`>`），确保概率为0.0的样本不被跳过
  - 返回值增加`low_sample_warning`标志（样本数<150时警告）
- **新增函数**:
  - `compute_fpr_at_tpr95()`: 计算TPR=95%时的FPR（使用完整ROC曲线）

**影响**:
- 提高了ECE计算的稳定性（固定bin数量）
- 支持S0实验的FPR@TPR95指标

---

#### 1.2 `src/utils/protocol_artifacts.py`
**修改内容**:
- 小幅修改以支持新的指标计算
- 可能与ECE返回值的更改相关（从2元组改为3元组）

---

#### 1.3 `src/data/multimodal_datamodule.py`
**修改内容**:
- 大幅扩展（+102行）
- 可能增加了对腐败数据的支持
- 可能增加了对S0数据分割的支持

---

#### 1.4 `src/systems/multimodal_baseline.py`
**修改内容**:
- 中等修改（+43行）
- 可能增加了对S0实验的支持
- 可能增加了工件生成功能

---

#### 1.5 `src/systems/html_only_module.py`, `url_only_module.py`, `visual_only_module.py`
**修改内容**:
- 小幅修改（+20行，+20行，+16行）
- 可能统一了接口或增加了工件生成支持

---

### 2. 配置和文档修改

#### 2.1 `requirements.txt`
**修改内容**:
- **从范围版本改为固定版本**:
  - `torch==2.5.1+cu121`（之前：`torch>=2.2`）
  - `pytorch-lightning==2.4.0`（之前：`>=2.3`）
  - `transformers==4.45.2`（之前：`>=4.41`）
  - 其他依赖也改为固定版本
- **版本锁定原因**: 确保实验的可重现性

---

#### 2.2 `configs/experiment/html_baseline.yaml`, `visual_baseline.yaml`
**修改内容**:
- 小幅修改（+3行）
- 可能更新了配置参数

---

#### 2.3 `configs/logger/wandb.yaml`
**修改内容**:
- 可能更新了WandB配置以支持S0实验

---

#### 2.4 `docs/DATA_SCHEMA.md`
**修改内容**:
- 大幅精简（-264行，大幅重构）
- 可能简化了文档结构，移除了冗余内容

---

#### 2.5 `FINAL_SUMMARY_CN.md`
**修改内容**:
- 更新了项目总结文档
- 可能添加了S0实验的相关内容

---

#### 2.6 `THESIS_COMPLIANCE_CHECK.md`
**修改内容**:
- 小幅修改（+3行）
- 可能更新了论文合规性检查

---

### 3. 测试文件修改

#### 3.1 `tests/test_mlops_implementation.py`
**修改内容**:
- 中等修改（+37行）
- 可能增加了对S0功能的测试
- 可能更新了测试以匹配新的指标计算

---

## 📁 新增文件结构

```
项目根目录/
├── WANDB_CONFIG_CHECK.md                    # WandB配置检查文档
├── configs/experiment/
│   ├── s0_brandood_earlyconcat.yaml         # Brand-OOD EarlyConcat配置
│   ├── s0_brandood_lateavg.yaml             # Brand-OOD Late Average配置
│   ├── s0_iid_earlyconcat.yaml              # IID EarlyConcat配置
│   └── s0_iid_lateavg.yaml                  # IID Late Average配置
├── scripts/
│   ├── evaluate_s0.py                       # S0评估脚本
│   ├── run_s0_experiments.py                # S0实验编排脚本
│   ├── summarize_s0_results.py              # S0结果汇总脚本
│   └── validate_s0_quality.py               # S0质量检查脚本
├── src/
│   ├── systems/
│   │   └── s0_late_avg_system.py            # S0晚期融合系统
│   └── utils/
│       └── metrics_v2.py                    # S0指标计算工具
└── tools/
    ├── corrupt_html.py                      # HTML腐败工具
    ├── corrupt_img.py                       # 图像腐败工具
    ├── corrupt_url.py                       # URL腐败工具
    ├── split_brandood.py                    # Brand-OOD分割工具
    └── split_iid.py                         # IID分割工具
```

---

## 🎯 主要功能总结

### 1. S0实验框架
- **目的**: 实现S0基线实验（EarlyConcat vs Late Average）
- **场景**: IID和Brand-OOD两种数据分割场景
- **模型**: 两种融合策略（早期融合 vs 晚期融合）

### 2. 数据腐败工具
- **目的**: 生成不同级别的腐败数据以测试模型鲁棒性
- **模态**: URL文本、HTML文件、图像截图
- **级别**: L (Low), M (Medium), H (High)

### 3. 数据分割工具
- **目的**: 创建IID和Brand-OOD数据分割
- **输出**: 标准化的CSV文件和品牌集合JSON文件

### 4. 评估和汇总工具
- **目的**: 自动化S0实验的评估和结果汇总
- **功能**: 指标计算、结果汇总、质量检查、可视化

### 5. 配置管理
- **目的**: 统一管理S0实验的配置
- **格式**: Hydra YAML配置文件

---

## 🔍 关键变更说明

### 1. 指标计算改进
- **ECE计算**: 从自适应bin改为固定15个bin，提高稳定性
- **新增指标**: FPR@TPR95、Brier分数
- **低样本警告**: 当样本数<150时发出警告

### 2. 版本锁定
- **依赖版本**: 所有依赖从范围版本改为固定版本
- **原因**: 确保实验的可重现性

### 3. 数据分割协议
- **IID分割**: 标准的70/15/15分层分割
- **Brand-OOD分割**: 基于品牌的域外分割（train/val/test_id/test_ood）

### 4. 腐败数据生成
- **多模态支持**: URL、HTML、图像三种模态
- **多级别支持**: L/M/H三个腐败级别
- **完整性检查**: SHA256校验和验证

---

## 📝 使用建议

### 1. 运行S0实验
```bash
# 运行IID场景的EarlyConcat和Late Average模型
python scripts/run_s0_experiments.py --scenario iid --models s0_earlyconcat s0_lateavg --seeds 42 43 44

# 运行Brand-OOD场景
python scripts/run_s0_experiments.py --scenario brandood --models s0_earlyconcat s0_lateavg --seeds 42 43 44
```

### 2. 评估实验结果
```bash
# 评估所有运行
python scripts/evaluate_s0.py --runs_dir workspace/runs

# 汇总结果
python scripts/summarize_s0_results.py --runs_dir workspace/runs
```

### 3. 生成腐败数据
```bash
# 生成HTML腐败数据
python tools/corrupt_html.py --in workspace/data/splits/iid/test.csv --out workspace/data/corrupt/html

# 生成图像腐败数据
python tools/corrupt_img.py --in workspace/data/splits/iid/test.csv --out workspace/data/corrupt/img

# 生成URL腐败数据
python tools/corrupt_url.py --in workspace/data/splits/iid/test.csv --out workspace/data/corrupt/url
```

### 4. 创建数据分割
```bash
# 创建IID分割
python tools/split_iid.py --in data/processed/master.csv --out workspace/data/splits/iid --seed 42

# 创建Brand-OOD分割
python tools/split_brandood.py --in data/processed/master.csv --out workspace/data/splits/brandood --seed 42 --top_k 20
```

---

## ⚠️ 注意事项

1. **工作区差异**: 新文件在新工作区，与原来的工作区不同
2. **版本锁定**: `requirements.txt`已锁定版本，安装时需使用CUDA 12.1
3. **数据路径**: 所有数据路径使用`workspace/`前缀，确保目录结构正确
4. **WandB配置**: 参考`WANDB_CONFIG_CHECK.md`正确配置WandB环境变量
5. **质量检查**: 运行实验前建议先运行`validate_s0_quality.py`检查数据质量

---

## 📊 变更统计

| 类别 | 新增 | 修改 | 删除 |
|------|------|------|------|
| 配置文件 | 4 | 3 | 0 |
| 脚本文件 | 4 | 0 | 0 |
| 系统文件 | 1 | 5 | 0 |
| 工具文件 | 5 | 0 | 0 |
| 工具模块 | 1 | 2 | 0 |
| 文档文件 | 1 | 3 | 0 |
| **总计** | **15** | **15** | **0** |

---

**报告生成时间**: 2025-11-08
**报告生成工具**: Git diff + 文件分析

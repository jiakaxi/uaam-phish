# 项目依赖说明

> **Last Updated:** 2025-10-21
> **版本:** 0.1.0

本文档详细说明了 UAAM-Phish 项目的所有依赖包及其用途。

---

## 📦 依赖管理文件

项目提供三种依赖管理方式：

| 文件 | 用途 | 适用场景 |
|------|------|----------|
| `requirements.txt` | pip 依赖列表 | 快速安装、CI/CD、Docker |
| `environment.yml` | Conda 环境配置 | 完整环境管理、科研复现 |
| `setup.py` | Python 包安装配置 | 开发模式安装、包分发 |

---

## 🔧 核心依赖

### 深度学习框架

#### PyTorch >= 2.2
- **用途:** 深度学习框架，提供张量计算和自动微分
- **为什么:** 业界标准，性能优秀，生态完善
- **安装:** `pip install torch` 或 `conda install pytorch`
- **文档:** https://pytorch.org/

#### PyTorch Lightning >= 2.3
- **用途:** PyTorch 高级封装，简化训练流程
- **为什么:**
  - 减少样板代码
  - 自动化分布式训练
  - 统一的训练/验证/测试接口
  - 内置日志和检查点管理
- **项目中使用:**
  - `src/systems/url_only_module.py` - LightningModule
  - `src/datamodules/url_datamodule.py` - LightningDataModule
- **文档:** https://lightning.ai/docs/pytorch/

---

### 预训练模型和评估

#### Transformers >= 4.41
- **用途:** Hugging Face 预训练模型库
- **为什么:**
  - 提供 BERT、RoBERTa 等预训练模型
  - 统一的 tokenizer 接口
  - 简化模型加载和微调
- **项目中使用:**
  - `src/models/url_encoder.py` - AutoModel, AutoConfig
  - `src/datamodules/url_datamodule.py` - AutoTokenizer
- **当前模型:** `roberta-base` (可配置)
- **文档:** https://huggingface.co/docs/transformers/

#### TorchMetrics >= 1.0
- **用途:** PyTorch 评估指标库
- **为什么:**
  - 自动处理分布式计算
  - 与 Lightning 无缝集成
  - 准确性经过验证
- **项目中使用:**
  - `BinaryF1Score` - F1 分数计算
  - `BinaryAUROC` - ROC 曲线下面积
- **文档:** https://lightning.ai/docs/torchmetrics/

---

### 数据处理

#### Pandas >= 2.1
- **用途:** 数据处理和分析
- **项目中使用:**
  - CSV 文件读写 (`train.csv`, `val.csv`, `test.csv`)
  - 数据清洗和转换
  - 数据集划分
- **文档:** https://pandas.pydata.org/

#### NumPy >= 1.26
- **用途:** 数值计算基础库
- **为什么:** Pandas 和 PyTorch 的底层依赖
- **文档:** https://numpy.org/

#### scikit-learn >= 1.4
- **用途:** 机器学习工具库
- **项目中使用:**
  - `train_test_split` - 数据划分
  - `GroupShuffleSplit` - 域名分组划分（避免数据泄漏）
- **文件:** `scripts/build_master_and_splits.py`, `scripts/preprocess.py`
- **文档:** https://scikit-learn.org/

---

### URL 解析

#### tldextract >= 3.4
- **用途:** 提取 URL 的域名、子域名和顶级域名
- **为什么:**
  - 比 `urlparse` 更智能
  - 处理复杂的国际域名
  - 用于域名分组（domain-aware splitting）
- **项目中使用:**
  - `scripts/build_master_and_splits.py` - 解析域名用于分组
- **示例:**
  ```python
  import tldextract
  ext = tldextract.extract('http://forums.news.cnn.com/')
  # ext.domain = 'cnn'
  # ext.suffix = 'com'
  # ext.subdomain = 'forums.news'
  ```
- **文档:** https://github.com/john-kurkowski/tldextract

---

### 配置管理

#### OmegaConf >= 2.3
- **用途:** 层次化配置管理
- **为什么:**
  - 支持 YAML 配置文件
  - 配置合并和覆盖
  - 环境变量插值
  - 类型检查
- **项目中使用:**
  - 加载 `configs/default.yaml`
  - 合并 `configs/profiles/*.yaml`
  - 环境变量替换（如 `${oc.env:DATA_ROOT}`）
- **文件:** `scripts/train.py`
- **文档:** https://omegaconf.readthedocs.io/

---

## 🎨 可选依赖

### 数据可视化

#### Matplotlib >= 3.7
- **用途:** 基础绘图库
- **安装:** `pip install -e ".[viz]"`
- **计划用途:**
  - 训练曲线可视化
  - 混淆矩阵绘制
  - ROC 曲线绘制
- **文档:** https://matplotlib.org/

#### Seaborn >= 0.12
- **用途:** 高级统计可视化
- **安装:** `pip install -e ".[viz]"`
- **计划用途:**
  - 数据分布分析
  - 相关性热力图
  - 美化图表
- **文档:** https://seaborn.pydata.org/

---

### 用户体验

#### tqdm >= 4.65
- **用途:** 进度条显示
- **安装:** `pip install -e ".[viz]"`
- **计划用途:** 数据预处理进度、推理进度
- **文档:** https://tqdm.github.io/

---

## 🛠️ 开发依赖

### 测试

#### pytest >= 7.0
- **用途:** 单元测试框架
- **安装:** `pip install -e ".[dev]"`
- **使用:** `pytest tests/`
- **文档:** https://docs.pytest.org/

#### pytest-cov >= 4.0
- **用途:** 测试覆盖率报告
- **安装:** `pip install -e ".[dev]"`
- **使用:** `pytest --cov=src tests/`
- **文档:** https://pytest-cov.readthedocs.io/

---

### 代码质量

#### Black >= 23.0
- **用途:** Python 代码格式化工具
- **安装:** `pip install -e ".[dev]"`
- **使用:** `black src/ scripts/ tests/`
- **为什么:** 统一代码风格，避免格式争论
- **文档:** https://black.readthedocs.io/

#### Flake8 >= 6.0
- **用途:** 代码风格和错误检查
- **安装:** `pip install -e ".[dev]"`
- **使用:** `flake8 src/ scripts/`
- **检查内容:** PEP 8 规范、语法错误、未使用变量等
- **文档:** https://flake8.pycqa.org/

#### isort >= 5.12
- **用途:** Python import 语句排序
- **安装:** `pip install -e ".[dev]"`
- **使用:** `isort src/ scripts/ tests/`
- **为什么:** 统一 import 顺序，提高可读性
- **文档:** https://pycqa.github.io/isort/

---

## 📥 安装指南

### 最小安装（仅核心功能）
```bash
pip install -r requirements.txt
```

### 开发模式安装（推荐）
```bash
# 基础安装
pip install -e .

# 包含可视化工具
pip install -e ".[viz]"

# 包含开发工具
pip install -e ".[dev]"

# 完整安装（全部功能）
pip install -e ".[all]"
```

### Conda 环境安装
```bash
# 创建新环境
conda env create -f environment.yml
conda activate uaam-phish

# 更新现有环境
conda env update -f environment.yml --prune
```

---

## 🔄 依赖更新策略

### 版本固定原则
- **核心依赖:** 使用 `>=` 指定最低版本，允许向后兼容的更新
- **重大版本:** 锁定主版本号（如 `torch>=2.2` 但不会自动升级到 3.x）
- **安全更新:** 定期检查安全漏洞，及时更新

### 更新检查
```bash
# 检查过期的包
pip list --outdated

# 使用 pip-audit 检查安全漏洞
pip install pip-audit
pip-audit
```

### 更新依赖
1. 更新 `requirements.txt`
2. 更新 `environment.yml`
3. 更新 `setup.py` 中的 `install_requires`
4. 测试兼容性
5. 更新本文档

---

## ⚠️ 已知问题和兼容性

### PyTorch 版本选择
- **CUDA 12.1:** `pytorch-cuda=12.1`
- **CUDA 11.8:** `pytorch-cuda=11.8`
- **CPU only:** 删除 `environment.yml` 中的 `pytorch-cuda` 行

### Windows 系统注意事项
- PyTorch 安装可能需要特定的 CUDA 版本
- 建议使用 Anaconda 进行环境管理
- `num_workers` 在 Windows 上可能需要设置为 0

### M1/M2 Mac 注意事项
- 使用 `device: mps` 启用 GPU 加速
- 某些包可能需要从源码编译
- 建议使用 Conda 进行安装

---

## 📊 依赖关系图

```
uaam-phish/
├── torch (核心)
│   └── numpy
├── pytorch-lightning (训练框架)
│   └── torch
├── transformers (预训练模型)
│   └── torch
├── torchmetrics (评估)
│   └── torch
├── pandas (数据处理)
│   └── numpy
├── scikit-learn (工具)
│   └── numpy
├── tldextract (URL 解析)
├── omegaconf (配置)
│   └── PyYAML
└── matplotlib, seaborn, tqdm (可选)
```

---

## 🆘 故障排除

### 安装失败

**问题:** `pip install torch` 很慢或失败
**解决:** 使用清华镜像源
```bash
pip install torch -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**问题:** Conda 解析环境很慢
**解决:** 使用 mamba
```bash
conda install mamba -c conda-forge
mamba env create -f environment.yml
```

**问题:** 找不到 CUDA
**解决:** 确认 CUDA 版本与 PyTorch 版本匹配
```bash
nvcc --version  # 查看 CUDA 版本
python -c "import torch; print(torch.cuda.is_available())"
```

### 版本冲突

**问题:** 依赖版本冲突
**解决:** 使用虚拟环境隔离
```bash
# 删除旧环境
conda env remove -n uaam-phish
# 重新创建
conda env create -f environment.yml
```

---

## 📚 参考资源

- [PyTorch 官方文档](https://pytorch.org/docs/)
- [Lightning 文档](https://lightning.ai/docs/)
- [Hugging Face 文档](https://huggingface.co/docs)
- [Python 打包指南](https://packaging.python.org/)

---

**维护者:** UAAM-Phish Team
**更新频率:** 每月检查更新
**最后检查:** 2025-10-21

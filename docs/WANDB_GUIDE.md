# WandB 云端实验跟踪指南

## 📊 什么是 WandB？

**Weights & Biases (WandB)** 是一个强大的机器学习实验跟踪平台，提供：
- ✅ 实时指标可视化
- ✅ 超参数对比
- ✅ 模型版本管理
- ✅ 团队协作
- ✅ 报告生成

---

## 🚀 快速开始

### 1. 安装 WandB

```bash
pip install wandb
```

### 2. 登录 WandB

```bash
# 首次使用需要登录
wandb login

# 或者设置 API key
export WANDB_API_KEY=64e15c91404e5023801580b0d943af3ebef4a033
```

### 3. 使用 WandB Logger

```bash
# 使用 WandB logger
python scripts/train_hydra.py logger=wandb

# 指定项目名称
export WANDB_PROJECT=my-phish-detection
python scripts/train_hydra.py logger=wandb

# 离线模式（稍后同步）
python scripts/train_hydra.py logger=wandb logger.offline=true
```

---

## 🎯 配置选项

### 环境变量

在 `.env` 文件或命令行中设置：

```bash
# WandB 项目名称
export WANDB_PROJECT=uaam-phish

# WandB team/username
export WANDB_ENTITY=your-team-name

# 实验标签
export WANDB_TAGS=url-only,baseline

# API Key
export WANDB_API_KEY=64e15c91404e5023801580b0d943af3ebef4a033

# 离线模式
export WANDB_MODE=offline
```

### Hydra 配置文件

编辑 `configs/logger/wandb.yaml`:

```yaml
logger:
  _target_: pytorch_lightning.loggers.WandbLogger
  project: uaam-phish
  name: ${run.name}
  offline: false
  log_model: true  # 上传模型检查点
  tags: [url-only, roberta]
  notes: "URL-only baseline experiment"
```

### 命令行覆盖

```bash
# 更改项目名称
python scripts/train_hydra.py logger=wandb logger.project=my-project

# 启用模型上传
python scripts/train_hydra.py logger=wandb logger.log_model=true

# 添加标签
python scripts/train_hydra.py logger=wandb logger.tags=[bert,baseline]
```

---

## 📈 使用示例

### 基本训练

```bash
# 本地开发（使用 CSV logger）
python scripts/train_hydra.py trainer=local

# 服务器训练（使用 WandB）
python scripts/train_hydra.py trainer=server logger=wandb
```

### 实验对比

```bash
# 运行多个实验
python scripts/train_hydra.py logger=wandb run.name=exp1 model.dropout=0.1
python scripts/train_hydra.py logger=wandb run.name=exp2 model.dropout=0.2
python scripts/train_hydra.py logger=wandb run.name=exp3 model.dropout=0.3

# 在 WandB Dashboard 中对比结果
```

### 超参数搜索

```bash
# 使用 Hydra multirun
python scripts/train_hydra.py -m logger=wandb \\
  train.lr=1e-5,2e-5,5e-5 \\
  model.dropout=0.1,0.2,0.3 \\
  train.bs=16,32

# WandB 会自动跟踪所有运行
```

---

## 🎨 WandB Dashboard 功能

### 1. 实时指标监控

在训练过程中，实时查看：
- Loss curves
- F1, AUROC, FPR
- Learning rate
- Gradient norms
- Consistency (S2): `val/consistency/acs`, `val/consistency/mr@τ_s`, `test/consistency/acs`

### 2. 超参数对比

自动记录和对比：
- 模型配置
- 训练参数
- 数据配置
- 硬件设置

### 3. 系统监控

跟踪：
- GPU/CPU 使用率
- 内存使用
- 网络流量
- 训练时间

### 4. 生成报告

创建交互式报告：
- 实验总结
- 可视化图表
- 团队分享

---

## 🔧 高级功能

### 1. 自定义日志

在代码中添加自定义日志：

```python
# 在 LightningModule 中
import wandb

def training_step(self, batch, batch_idx):
    loss = ...

    # 记录自定义指标
    self.log("custom/metric", value)

    # 记录图像
    if self.logger and isinstance(self.logger, WandbLogger):
        self.logger.experiment.log({
            "predictions": wandb.Image(img)
        })

    return loss
```

### 2. 保存 Artifacts

保存重要文件：

```python
# 保存模型
wandb.save("model.ckpt")

# 保存数据集
artifact = wandb.Artifact("dataset", type="dataset")
artifact.add_file("data/train.csv")
wandb.log_artifact(artifact)
```

### 3. 团队协作

```bash
# 设置 team
export WANDB_ENTITY=your-team

# 所有成员可以查看实验
python scripts/train_hydra.py logger=wandb
```

---

## 📋 最佳实践

### 1. 命名规范

```bash
# 使用描述性的实验名称
python scripts/train_hydra.py logger=wandb \\
  run.name="roberta-base_lr2e5_bs32_dropout02"
```

### 2. 使用标签

```bash
# 添加有意义的标签
python scripts/train_hydra.py logger=wandb \\
  logger.tags=[url-only,baseline,roberta,v1]
```

### 3. 添加备注

编辑 `configs/logger/wandb.yaml`:

```yaml
logger:
  notes: |
    实验目标：
    - 测试 RoBERTa-base 作为 URL 编码器
    - 基线性能评估
    变更：
    - 增加 dropout 到 0.2
    - 使用 cosine scheduler
```

### 4. 组织项目

```bash
# 按功能组织项目
WANDB_PROJECT=uaam-phish-url python scripts/train_hydra.py logger=wandb
WANDB_PROJECT=uaam-phish-multimodal python scripts/train_hydra.py logger=wandb
```

---

## 🐛 故障排除

### 问题 1: WandB 登录失败

```bash
# 重新登录
wandb login --relogin

# 或设置 API key
export WANDB_API_KEY=64e15c91404e5023801580b0d943af3ebef4a033
```

### 问题 2: 网络问题

```bash
# 使用离线模式
python scripts/train_hydra.py logger=wandb logger.offline=true

# 稍后同步
wandb sync outputs/2025-10-22/18-45-00/wandb/latest-run
```

### 问题 3: 日志过多

```bash
# 减少日志频率
python scripts/train_hydra.py logger=wandb train.log_every=100
```

---

## 📚 其他 Logger 选项

### TensorBoard

```bash
python scripts/train_hydra.py logger=tensorboard

# 查看
tensorboard --logdir outputs/
```

### CSV Logger（默认）

```bash
python scripts/train_hydra.py logger=csv

# 结果在 outputs/*/metrics.csv
```

### MLflow

```bash
# 需要先安装
pip install mlflow

# 创建 configs/logger/mlflow.yaml
# 然后使用
python scripts/train_hydra.py logger=mlflow
```

---

## 🔗 资源

- [WandB 官方文档](https://docs.wandb.ai/)
- [PyTorch Lightning + WandB](https://docs.wandb.ai/guides/integrations/lightning)
- [Hydra + WandB](https://hydra.cc/docs/plugins/wandb_sweeper/)

---

**维护者:** UAAM-Phish Team
**最后更新:** 2025-10-22

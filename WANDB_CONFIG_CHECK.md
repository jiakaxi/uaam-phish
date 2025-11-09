# WandB 配置检查报告

## 📋 你的配置 vs 项目实际配置

### ✅ 正确的配置项

```bash
export WANDB_API_KEY="64e15c91404e5023801580b0d943af3ebef4a033"
export WANDB_PROJECT="uaam-s0"
```
**状态**: ✅ 正确

---

### ⚠️ 需要修正的配置

#### 1. **WANDB_ENTITY** 格式错误

**你提供的**:
```bash
export WANDB_ENTITY="jiakaxilove-jiakaxi/uaam-phish/"
```

**问题**:
- ❌ 包含了项目路径 `/uaam-phish/`
- ❌ 末尾有多余的斜杠 `/`
- `WANDB_ENTITY` 应该只是**用户名或团队名**，不包含项目名

**正确的配置应该是**:
```bash
# 如果用户名是 jiakaxilove-jiakaxi
export WANDB_ENTITY="jiakaxilove-jiakaxi"

# 或者如果用户名是 jiakaxi
export WANDB_ENTITY="jiakaxi"
```

**验证方法**:
访问 https://wandb.ai/settings，查看你的用户名或团队名

---

#### 2. **WANDB_MODE** 可能无效

**你提供的**:
```bash
export WANDB_MODE="online"
```

**问题**:
- ⚠️ 项目使用 PyTorch Lightning 的 `WandbLogger`
- ⚠️ `WandbLogger` 使用 `offline` 参数，不是 `WANDB_MODE` 环境变量
- ⚠️ 项目配置中已经设置了 `offline: false`（在 `configs/logger/wandb.yaml`）

**说明**:
- 虽然 WandB SDK 原生支持 `WANDB_MODE`，但 PyTorch Lightning 的 `WandbLogger` 主要通过 `offline` 参数控制
- 如果项目已经运行成功，说明当前的 `offline: false` 配置是有效的
- `WANDB_MODE` 环境变量可能**不会生效**，因为 Lightning 使用自己的配置

**如果需要离线模式，应该使用**:
```bash
# 方法 1: 通过 Hydra 配置覆盖
python scripts/train_hydra.py logger=wandb logger.offline=true

# 方法 2: 修改 configs/logger/wandb.yaml 中的 offline: true
```

**建议**: 可以移除 `WANDB_MODE` 环境变量，因为它可能不会生效

---

## ✅ 推荐的完整配置

### Linux/Mac (bash)

```bash
# WandB 认证
export WANDB_API_KEY="64e15c91404e5023801580b0d943af3ebef4a033"

# WandB 项目名称
export WANDB_PROJECT="uaam-s0"

# WandB 实体（用户名或团队名）- 请根据实际情况修改
export WANDB_ENTITY="jiakaxilove-jiakaxi"  # 或 "jiakaxi"

# 可选：实验标签
export WANDB_TAGS="s0,baseline,experiment"

# 注意：WANDB_MODE 不需要设置，因为项目使用 logger.offline 参数
```

### Windows (PowerShell)

```powershell
# WandB 认证
$env:WANDB_API_KEY="64e15c91404e5023801580b0d943af3ebef4a033"

# WandB 项目名称
$env:WANDB_PROJECT="uaam-s0"

# WandB 实体（用户名或团队名）- 请根据实际情况修改
$env:WANDB_ENTITY="jiakaxilove-jiakaxi"  # 或 "jiakaxi"

# 可选：实验标签
$env:WANDB_TAGS="s0,baseline,experiment"
```

---

## 🔍 项目中的实际配置

查看 `configs/logger/wandb.yaml`:

```yaml
logger:
  _target_: pytorch_lightning.loggers.WandbLogger
  project: ${oc.env:WANDB_PROJECT,uaam-phish}  # 从环境变量读取，默认 uaam-phish
  name: ${run.name}
  save_dir: ${hydra:runtime.output_dir}
  offline: false  # 在线模式（不是通过 WANDB_MODE）
  log_model: false
  tags: ${oc.env:WANDB_TAGS,null}
  notes: null
  entity: ${oc.env:WANDB_ENTITY,null}  # 从环境变量读取
```

**关键点**:
1. `project` 从 `WANDB_PROJECT` 环境变量读取 ✅
2. `entity` 从 `WANDB_ENTITY` 环境变量读取 ✅
3. `offline` 是硬编码的 `false`，不是从 `WANDB_MODE` 读取 ❌

---

## ✅ 验证配置

### 1. 检查 WandB 登录状态

```bash
wandb login
# 或者
wandb status
```

### 2. 测试配置

```bash
# 设置环境变量
export WANDB_API_KEY="64e15c91404e5023801580b0d943af3ebef4a033"
export WANDB_PROJECT="uaam-s0"
export WANDB_ENTITY="jiakaxilove-jiakaxi"  # 请确认正确的用户名

# 运行一个快速测试
python scripts/train_hydra.py logger=wandb trainer.fast_dev_run=1
```

### 3. 检查 WandB Dashboard

访问: https://wandb.ai/jiakaxilove-jiakaxi/uaam-s0 (根据你的实际用户名调整)

---

## 📝 总结

### 需要修正

1. ✅ **WANDB_ENTITY**: 移除项目路径，只保留用户名
   - 错误: `"jiakaxilove-jiakaxi/uaam-phish/"`
   - 正确: `"jiakaxilove-jiakaxi"` 或 `"jiakaxi"`

2. ⚠️ **WANDB_MODE**: 可以移除，因为项目不使用它
   - 项目使用 `logger.offline=false` 控制在线/离线模式
   - `WANDB_MODE` 环境变量不会被 PyTorch Lightning 的 WandbLogger 读取

### 保持不变

1. ✅ **WANDB_API_KEY**: 正确
2. ✅ **WANDB_PROJECT**: 正确

---

## 🚀 快速修复

```bash
# 修正后的配置
export WANDB_API_KEY="64e15c91404e5023801580b0d943af3ebef4a033"
export WANDB_PROJECT="uaam-s0"
export WANDB_ENTITY="jiakaxilove-jiakaxi"  # 请确认正确的用户名

# 移除 WANDB_MODE（不需要）
# export WANDB_MODE="online"  # ❌ 删除这行
```

---

**最后更新**: 2025-11-08
**检查者**: AI Assistant

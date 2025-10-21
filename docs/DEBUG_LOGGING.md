# Debug Logging 指南

## 📝 日志系统架构

### 1. 日志模块：`src/utils/logging.py`

```python
from src.utils.logging import get_logger

logger = get_logger(__name__)
logger.info("开始处理...")
logger.warning("注意：...")
logger.error("错误：...")
logger.debug("调试信息...")
```

#### 特点：
- ✅ 使用 `rich.logging.RichHandler` 提供彩色输出
- ✅ 支持富文本追踪（rich_tracebacks）
- ✅ 统一格式：`时间 | 级别 | 模块名 | 消息`
- ✅ 默认日志级别：INFO

---

## 🔍 日志记录点

### 训练脚本 (`scripts/train.py`)

```python
log = get_logger(__name__)
log.info("Training start")  # ✅ 已有

# 建议补充：
log.info(f"配置加载完成: profile={args.profile}")
log.info(f"数据加载完成: train={len(train)}, val={len(val)}")
log.info(f"模型初始化完成: {cfg.model.pretrained_name}")
log.warning(f"早停触发: epoch={epoch}")
log.info(f"训练完成: 最佳 {monitor}={best_val:.4f}")
```

### 实验回调 (`src/utils/callbacks.py`)

```python
# ✅ 已实现的日志点：
- 训练开始时记录配置
- 每个 epoch 结束记录指标
- 训练结束标记
- 测试完成标记

# 写入到: experiments/<exp_name>/logs/train.log
```

---

## 📊 日志输出层级

### 1. **控制台日志**（实时）
- 使用 `get_logger()` → stdout
- 彩色输出，便于查看
- 包含时间戳和级别

### 2. **实验日志文件**（持久化）
- 路径：`experiments/<exp_name>/logs/train.log`
- 通过 `ExperimentTracker.log_text()` 写入
- 包含完整训练过程

### 3. **Lightning 日志**
- 路径：`lightning_logs/version_X/metrics.csv`
- 自动记录所有指标
- 用于绘制训练曲线

---

## 🐛 Debug 最佳实践

### 启用 DEBUG 级别日志

```python
# 方法1: 修改 logging.py 中的默认级别
logger.setLevel(logging.DEBUG)  # 改为 DEBUG

# 方法2: 环境变量控制
import os
log_level = os.getenv("LOG_LEVEL", "INFO")
logger.setLevel(getattr(logging, log_level))
```

使用：
```bash
LOG_LEVEL=DEBUG python scripts/train.py
```

### 关键调试点

```python
# 数据加载
logger.debug(f"Batch shape: {batch['input_ids'].shape}")
logger.debug(f"Label distribution: {batch['label'].mean():.2f}")

# 模型前向
logger.debug(f"Logits range: [{logits.min():.3f}, {logits.max():.3f}]")

# 损失计算
logger.debug(f"Loss components: bce={bce:.4f}, reg={reg:.4f}")

# 梯度
logger.debug(f"Grad norm: {grad_norm:.4f}")
```

---

## 📈 实验跟踪日志

### ExperimentTracker 自动记录

```python
tracker = ExperimentTracker(cfg)

# 自动创建的日志：
tracker.log_text("训练开始")           # ✅ 自动
tracker.log_text(f"Epoch {i}: ...")  # ✅ 每轮自动
tracker.log_text("训练完成")           # ✅ 自动

# 手动补充：
tracker.log_text("数据预处理完成", filename="preprocessing.log")
tracker.log_text("模型评估中...", filename="eval.log")
```

---

## 🔧 当前状态检查

### ✅ 已有的日志功能：

1. **基础日志模块** (`src/utils/logging.py`)
   - Rich 彩色输出
   - 统一格式

2. **实验日志** (`ExperimentTracker`)
   - 自动记录训练过程
   - 持久化到文件

3. **Lightning 集成**
   - 自动记录指标到 CSV
   - 支持 TensorBoard

4. **回调日志** (`ExperimentResultsCallback`)
   - 训练开始/结束
   - 每轮指标

### ⚠️ 建议补充：

1. **更细粒度的日志点**
   ```python
   # 在关键步骤添加日志
   logger.info("数据加载开始...")
   logger.info(f"✅ 数据加载完成: {len(dataset)} 样本")
   logger.info(f"✅ 模型初始化: {model_name}")
   ```

2. **异常日志**
   ```python
   try:
       result = process()
   except Exception as e:
       logger.error(f"处理失败: {e}", exc_info=True)
       raise
   ```

3. **性能日志**
   ```python
   import time
   start = time.time()
   result = expensive_operation()
   logger.info(f"⏱️ 耗时: {time.time() - start:.2f}s")
   ```

4. **数据验证日志**
   ```python
   logger.info(f"数据分布检查:")
   logger.info(f"  - Train: {train_count} 样本")
   logger.info(f"  - Val: {val_count} 样本")
   logger.info(f"  - 正负样本比: {pos/neg:.2f}")
   ```

---

## 📋 日志检查清单

运行实验后，应该有以下日志：

- [ ] **控制台输出**: 训练开始/进度/完成
- [ ] **实验日志**: `experiments/<exp>/logs/train.log`
- [ ] **指标历史**: `lightning_logs/version_X/metrics.csv`
- [ ] **实验总结**: `experiments/<exp>/SUMMARY.md`

---

## 💡 调试技巧

### 1. 快速查看最近的实验日志
```bash
# 查看最新实验的日志
ls -t experiments/ | head -1 | xargs -I {} cat experiments/{}/logs/train.log
```

### 2. 监控训练日志（实时）
```bash
# 实时查看日志文件
tail -f experiments/latest_exp/logs/train.log
```

### 3. 搜索错误日志
```bash
# 查找所有错误
grep -r "ERROR" experiments/*/logs/
```

### 4. 对比不同实验的日志
```bash
# 对比两个实验的训练日志
diff experiments/exp1/logs/train.log experiments/exp2/logs/train.log
```

---

## 🎯 日志规范

### DO ✅

```python
# 好的日志实践
logger.info("开始训练 - 配置: lr=2e-5, bs=32, epochs=10")
logger.info(f"✅ Epoch {epoch}: loss={loss:.4f}, f1={f1:.4f}")
logger.warning(f"⚠️ 验证集性能下降: {val_f1:.4f} -> {new_val_f1:.4f}")
logger.error(f"❌ 数据加载失败: {file_path}", exc_info=True)
```

### DON'T ❌

```python
# 不好的日志实践
logger.info("start")  # 太简短
logger.info(f"loss: {loss}")  # 缺少上下文
print("training...")  # 使用 print 而不是 logger
logger.info(large_tensor)  # 打印大对象
```

---

## 📚 相关资源

- [Python logging 文档](https://docs.python.org/3/library/logging.html)
- [Rich logging](https://rich.readthedocs.io/en/latest/logging.html)
- [PyTorch Lightning logging](https://lightning.ai/docs/pytorch/stable/extensions/logging.html)

---

**更新日期:** 2025-10-21  
**维护者:** UAAM-Phish Team


# MLOps 协议快速参考卡

## 🚀 一行命令启动

```bash
# Random 协议（默认）
python scripts/train_hydra.py

# Temporal 协议
python scripts/train_hydra.py protocol=temporal

# Brand-OOD 协议
python scripts/train_hydra.py protocol=brand_ood
```

---

## 📁 输出文件位置

```
experiments/<run_name>/results/
├── roc_{protocol}.png          # ROC曲线
├── calib_{protocol}.png         # 校准曲线（含ECE）
├── splits_{protocol}.csv        # 分割统计
└── metrics_{protocol}.json      # 完整指标
```

---

## 📊 三种协议对比

| 协议 | 用途 | 要求 | 特点 |
|------|------|------|------|
| **random** | 基线 | 无 | 分层随机，始终可用 |
| **temporal** | 时序预测 | timestamp列 | 时间顺序，left-closed |
| **brand_ood** | 域泛化 | brand列，≥3品牌 | 品牌不相交 |

---

## 🔍 检查实验结果

```bash
# 进入实验目录
cd experiments/<run_name>/results/

# 查看指标
cat metrics_{protocol}.json

# 查看分割统计
cat splits_{protocol}.csv

# 查看实现报告
cat implementation_report.md
```

---

## ⚙️ 常用配置

### 修改分割比例
```bash
python scripts/train_hydra.py \
    protocol=temporal \
    data.split_ratios.train=0.8 \
    data.split_ratios.val=0.1 \
    data.split_ratios.test=0.1
```

### 启用 WandB
```bash
python scripts/train_hydra.py \
    protocol=brand_ood \
    logger=wandb
```

### 本地快速测试
```bash
python scripts/train_hydra.py \
    +profiles/local \
    protocol=random
```

---

## 📋 指标说明

### Step 级（每个batch）
- **Accuracy** - 准确率
- **AUROC** - ROC曲线下面积（phishing类）
- **F1** - F1分数（macro平均）

### Epoch 级（整个epoch）
- **NLL** - 负对数似然
- **ECE** - 期望校准误差（自适应bins）

---

## ⚠️ 自动降级

协议会在以下情况自动降级到 random：

| 协议 | 降级条件 |
|------|----------|
| temporal | 缺少 timestamp 列 |
| brand_ood | 缺少 brand 列 |
| brand_ood | 品牌数 ≤ 2 |
| brand_ood | 品牌集相交检查失败 |

**查看降级原因**:
```bash
cat metrics_{protocol}.json | grep downgraded_reason
```

---

## 🛡️ URL 编码器锁定

架构已锁定，不可修改：
- 2层双向LSTM
- 字符级tokenization
- Hidden: 128
- Output: 256

修改将触发 `AssertionError`

---

## 🧪 运行测试

```bash
# 运行所有MLOps测试
python -m pytest tests/test_mlops_implementation.py -v

# 运行特定测试
python -m pytest tests/test_mlops_implementation.py::TestDataSplits -v
```

---

## 📚 完整文档

- **快速入门**: `docs/QUICKSTART_MLOPS_PROTOCOLS.md`
- **实现报告**: `IMPLEMENTATION_REPORT.md`
- **变更摘要**: `CHANGES_SUMMARY.md`
- **最终总结**: `FINAL_SUMMARY_CN.md`

---

## 🆘 常见问题

**Q: 为什么降级到 random？**
A: 查看 `metrics_{protocol}.json.warnings.downgraded_reason`

**Q: 如何验证品牌不相交？**
A: 查看 `splits_{protocol}.csv.brand_intersection_ok`

**Q: ECE bins 怎么确定？**
A: 自适应计算：`max(3, min(15, √N, 10))`

**Q: 可以修改URL编码器吗？**
A: ❌ 不可以！已被断言锁定

---

## ✅ 验证安装

```bash
# 验证所有依赖
python -c "from src.utils.splits import build_splits; print('✅ 安装成功')"

# 运行测试
python -m pytest tests/test_mlops_implementation.py --tb=short

# 预期：13/13 测试通过
```

---

**版本**: 1.0.0
**更新**: 2025-10-23
**状态**: ✅ 生产就绪

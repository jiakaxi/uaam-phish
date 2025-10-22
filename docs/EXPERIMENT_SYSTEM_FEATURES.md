# 实验管理系统功能清单

> **系统版本:** 1.0
> **更新日期:** 2025-10-21

本文档总结了项目实验管理系统的所有功能和特性。

---

## ✨ 核心功能

### 1. **自动目录创建**
- ✅ 每次实验自动创建独立目录
- ✅ 目录名称包含实验名和时间戳
- ✅ 标准化的子目录结构（results/, logs/, checkpoints/）

### 2. **配置自动保存**
- ✅ 实验开始时立即保存完整配置
- ✅ YAML 格式，易于阅读和复现
- ✅ 包含所有合并后的配置参数

### 3. **指标自动保存**
- ✅ 训练结束后立即保存 JSON 格式指标
- ✅ 包含时间戳和实验元数据
- ✅ 支持多阶段指标（train/val/test）

### 4. **训练日志实时记录**
- ✅ 每个 epoch 的指标实时写入
- ✅ 包含时间戳的日志条目
- ✅ 训练开始/结束标记

### 5. **可视化自动生成** 📊
- ✅ **训练曲线**: Loss, F1, AUROC, FPR
- ✅ **混淆矩阵**: 热力图 + 性能指标
- ✅ **ROC 曲线**: ROC + AUC 值
- ✅ **阈值分析**: 最佳 F1 阈值扫描
- ✅ 高分辨率 PNG (300 DPI)

### 6. **检查点管理**
- ✅ 自动复制最佳模型检查点
- ✅ 从 lightning_logs/ 到 experiments/
- ✅ 独立存储，避免被覆盖

### 7. **实验总结生成**
- ✅ Markdown 格式总结文档
- ✅ 包含配置和最终结果
- ✅ 便于快速查看和分享

### 8. **实验对比工具**
- ✅ 对比多个实验的指标
- ✅ 表格化展示结果
- ✅ 自动排序和筛选
- ✅ 导出 CSV/Excel/Markdown

### 9. **最佳实验查找**
- ✅ 按指定指标查找最佳实验
- ✅ 支持 F1, AUROC, Loss 等指标
- ✅ 快速定位最优配置

### 10. **灵活的实验命名**
- ✅ 自定义实验名称
- ✅ 自动添加时间戳
- ✅ 避免命名冲突

---

## 🛠️ 核心组件

### Python 模块

#### `src/utils/experiment_tracker.py`
**ExperimentTracker 类**
- 实验目录管理
- 配置保存
- 指标保存
- 日志记录
- 总结生成

**关键方法:**
```python
tracker = ExperimentTracker(cfg, exp_name="my_exp")
tracker.save_metrics(metrics, stage="final")
tracker.save_figure(fig, name="plot")
tracker.log_text("Training started")
tracker.save_summary(summary_dict)
tracker.copy_checkpoints(lightning_log_dir)
```

#### `src/utils/visualizer.py`
**ResultVisualizer 类**
- 训练曲线绘制
- 混淆矩阵生成
- ROC 曲线绘制
- 阈值分析
- 批量图表生成

**关键方法:**
```python
ResultVisualizer.plot_training_curves(metrics_csv, save_path)
ResultVisualizer.plot_confusion_matrix(y_true, y_pred, save_path)
ResultVisualizer.plot_roc_curve(y_true, y_prob, save_path)
ResultVisualizer.plot_threshold_analysis(y_true, y_prob, save_path)
ResultVisualizer.create_all_plots(metrics_csv, y_true, y_prob, output_dir)
```

#### `src/utils/callbacks.py`
**Lightning 回调**
- `ExperimentResultsCallback`: 自动保存实验结果
- `TestPredictionCollector`: 收集测试预测用于可视化

**集成方式:**
```python
callbacks = [
    ExperimentResultsCallback(experiment_tracker),
    TestPredictionCollector()
]
```

#### `scripts/compare_experiments.py`
**实验对比工具**
- 加载多个实验的指标
- 表格化对比
- 导出多种格式
- 查找最佳实验

**用法:**
```bash
python scripts/compare_experiments.py --latest 5
python scripts/compare_experiments.py --find_best --metric f1
python scripts/compare_experiments.py --all --output report.csv
```

---

## 📁 文件输出清单

每次实验自动生成以下文件：

| 文件 | 格式 | 时机 | 用途 |
|------|------|------|------|
| `config.yaml` | YAML | 实验开始 | 配置保存 |
| `results/metrics_final.json` | JSON | 测试结束 | 最终指标 |
| `results/training_curves.png` | PNG | 训练结束 | 训练可视化 |
| `results/confusion_matrix.png` | PNG | 测试结束 | 分类性能 |
| `results/roc_curve.png` | PNG | 测试结束 | 判别能力 |
| `results/threshold_analysis.png` | PNG | 测试结束 | 阈值优化 |
| `logs/train.log` | TXT | 实时 | 训练日志 |
| `logs/metrics_history.csv` | CSV | 训练中 | 指标历史 |
| `checkpoints/*.ckpt` | PyTorch | 训练结束 | 模型权重 |
| `SUMMARY.md` | Markdown | 测试结束 | 实验总结 |

---

## 🎯 使用场景

### 场景 1: 快速原型开发
```bash
# 快速测试，不保存结果
python scripts/train.py --profile local --no_save
```

### 场景 2: 正式实验
```bash
# 完整实验，保存所有结果
python scripts/train.py --profile server --exp_name bert_baseline
```

### 场景 3: 超参数搜索
```bash
# 运行多组实验
for lr in 1e-5 2e-5 5e-5; do
    python scripts/train.py --exp_name lr_${lr}
done

# 对比结果
python scripts/compare_experiments.py --exp_names lr_1e-5 lr_2e-5 lr_5e-5
```

### 场景 4: 模型对比
```bash
# BERT vs RoBERTa
python scripts/train.py --exp_name bert_baseline
python scripts/train.py --exp_name roberta_baseline

# 查看对比
python scripts/compare_experiments.py --exp_names bert roberta
```

### 场景 5: 结果分享
```bash
# 生成报告
python scripts/compare_experiments.py --all --output experiments_report.md

# 分享 Markdown 文件和图表
# experiments/exp_name/results/*.png
```

---

## 📊 可视化示例

### 训练曲线图
- 4 个子图：Loss, F1, AUROC, FPR
- Train & Val 对比
- 每个 epoch 的变化趋势

### 混淆矩阵
- 2x2 热力图
- 真阳性/假阳性/真阴性/假阴性
- 附加：Accuracy, Precision, Recall, F1

### ROC 曲线
- TPR vs FPR 曲线
- AUC 值标注
- 随机分类器基线

### 阈值分析
- Precision/Recall/F1 vs Threshold
- 最佳阈值标记（红色虚线）
- F1 最大值点

---

## 🔧 配置选项

### 训练脚本参数

```bash
python scripts/train.py \
    --profile [local|server]  # 环境配置
    --exp_name NAME           # 实验名称
    --no_save                 # 不保存结果（调试）
```

### 对比脚本参数

```bash
python scripts/compare_experiments.py \
    --base_dir DIR            # 实验根目录
    --exp_names EXP1 EXP2     # 指定实验
    --latest N                # 最近 N 个
    --all                     # 所有实验
    --output FILE             # 导出文件
    --metric METRIC           # 排序指标
    --find_best               # 查找最佳
```

---

## 💡 最佳实践

### ✅ 推荐做法

1. **有意义的命名**
   ```bash
   python scripts/train.py --exp_name bert_dropout02_lr2e5
   ```

2. **定期对比**
   ```bash
   python scripts/compare_experiments.py --latest 10
   ```

3. **记录发现**
   - 维护 `EXPERIMENTS_LOG.md`
   - 记录每个实验的目的和结论

4. **保留最佳模型**
   ```bash
   # 找到最佳实验
   python scripts/compare_experiments.py --find_best

   # 复制到专门目录
   cp -r experiments/best_exp/ saved_models/production_v1/
   ```

5. **清理旧实验**
   ```bash
   # 只保留最近 20 个
   ls -t experiments/ | tail -n +21 | xargs -I {} rm -rf experiments/{}
   ```

### ❌ 避免的做法

1. **无意义命名**: `test1`, `exp123`
2. **不查看结果**: 训练完不分析图表
3. **不记录**: 忘记实验的配置和发现
4. **重复命名**: 手动删除实验导致混淆

---

## 🚀 未来扩展

### 计划中的功能

- [ ] TensorBoard 集成
- [ ] MLflow 集成
- [ ] 自动超参数搜索（Optuna）
- [ ] 实验版本控制（Git commit hash）
- [ ] 云存储同步（S3/OSS）
- [ ] 实验报告自动生成（PDF）
- [ ] Slack/Email 通知
- [ ] 实验依赖关系图
- [ ] 模型性能对比雷达图
- [ ] 交互式可视化（Plotly）

### 可能的改进

- [ ] 分布式训练时的多进程日志
- [ ] 实验标签系统
- [ ] 实验搜索功能
- [ ] Web UI 查看实验
- [ ] 实验回滚和恢复
- [ ] 增量实验（基于已有检查点）

---

## 📚 相关文档

- [实验管理指南](EXPERIMENTS.md) - 完整使用文档
- [快速启动指南](QUICK_START_EXPERIMENT.md) - 5分钟入门
- [项目结构说明](ROOT_STRUCTURE.md) - 目录结构详解

---

## 📝 更新日志

### v1.0 (2025-10-21)
- ✅ 实验目录自动创建
- ✅ 配置和指标自动保存
- ✅ 4种可视化图表自动生成
- ✅ 训练日志实时记录
- ✅ 实验对比工具
- ✅ 最佳实验查找
- ✅ 多种导出格式支持

---

**系统状态:** ✅ 稳定运行
**测试覆盖:** 手动测试通过
**文档完整性:** ✅ 完整

开始使用实验管理系统：
```bash
python scripts/train.py --profile local --exp_name my_first_experiment
```

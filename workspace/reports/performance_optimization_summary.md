# S0 训练性能优化总结

## 已实施的优化

### 1. ✅ 修复超大图像问题
**修改文件**：`src/data/multimodal_datamodule.py`

**变更**：
- 增加PIL图像大小限制：`Image.MAX_IMAGE_PIXELS = 500_000_000`
- 在`_load_image`中添加图像大小检查和快速resize
- 超过10MP的图像立即缩放到合理大小（保持宽高比）
- 使用`thumbnail`方法进行高效resize

**预期效果**：消除DecompressionBombWarning，大幅减少图像加载时间（10-50x提升）

### 2. ✅ HTML文件预加载
**修改文件**：`src/data/multimodal_datamodule.py`

**变更**：
- 在`MultimodalDataset.__init__`中预加载所有HTML文件到内存
- 添加`html_cache`字典缓存HTML内容
- `_load_html`方法优先使用缓存

**预期效果**：消除文件IO瓶颈，提升数据加载速度（2-5x提升）

### 3. ✅ Windows多进程优化
**修改文件**：
- `configs/experiment/s0_iid_earlyconcat.yaml`
- `configs/experiment/s0_iid_lateavg.yaml`
- `configs/experiment/s0_brandood_earlyconcat.yaml`
- `configs/experiment/s0_brandood_lateavg.yaml`

**变更**：
- 将`num_workers`从4改为0（单进程模式）
- 避免Windows上多进程启动开销和进程间通信开销

**预期效果**：在Windows上提升1.5-2x速度

## 性能预期

### 优化前
- **训练速度**：0.04 it/s
- **每个batch时间**：~25秒
- **每个epoch时间**：~73分钟
- **50个epoch预计时间**：~61小时（2.5天）

### 优化后（预期）
- **训练速度**：1-2 it/s（提升25-50倍）
- **每个batch时间**：0.5-1秒
- **每个epoch时间**：~2-3分钟
- **50个epoch预计时间**：~2-3小时

## 内存影响

### HTML预加载
- 训练集：11,200样本
- 假设平均HTML大小：50KB
- 总内存占用：~560MB（可接受）

### 图像处理
- 大图像立即resize，避免加载超大图像到内存
- 实际内存占用减少

## 下一步行动

1. **重新启动训练**，验证性能提升
2. **监控训练速度**，确认达到预期
3. **如果仍有瓶颈**，考虑：
   - 增加batch size（如果显存允许）
   - 使用更快的图像库（如opencv）
   - 预处理图像到固定尺寸

## 测试建议

在重新训练前，可以运行一个小规模测试：
```bash
python scripts/train_hydra.py \
  experiment=s0_iid_earlyconcat \
  trainer.max_epochs=1 \
  trainer.limit_train_batches=10 \
  run.seed=42
```

检查：
- HTML预加载是否正常工作
- 图像加载是否快速
- 训练速度是否提升

## 注意事项

1. **HTML预加载**：首次初始化数据集时会需要一些时间（预加载HTML文件），但之后训练会更快
2. **内存使用**：HTML预加载会增加内存使用，但通常在可接受范围内
3. **Windows单进程**：`num_workers=0`意味着数据加载在主进程中进行，CPU利用率可能较低，但避免了多进程开销

## 文件修改清单

1. `src/data/multimodal_datamodule.py`
   - 增加PIL图像大小限制
   - 添加HTML预加载功能
   - 优化图像加载（大图像resize）
   - 修改`_load_html`使用缓存

2. `configs/experiment/s0_*.yaml` (4个文件)
   - 将`num_workers`改为0

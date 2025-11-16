# S0 训练速度分析报告

## 问题概述

训练速度极慢：**0.04 it/s**（每个batch约25秒），远低于正常速度（预期应>1 it/s）。

## 发现的性能瓶颈

### 1. 🔴 严重：超大图像文件
**问题**：PIL DecompressionBombWarning 显示有图像超过1.13亿像素（113,753,650 pixels）
```
DecompressionBombWarning: Image size (113753650 pixels) exceeds limit of 89478485 pixels
```

**影响**：
- 单张图像加载和处理可能需要数秒
- 内存占用巨大
- 图像resize操作极慢

**解决方案**：
- 在`_load_image`中添加图像大小限制和快速resize
- 使用`Image.MAX_IMAGE_PIXELS`增加限制或提前resize
- 考虑预处理图像到固定尺寸

### 2. 🔴 严重：HTML文件实时读取
**问题**：每次`__getitem__`都要从磁盘读取HTML文件
```python
def _load_html(self, row: pd.Series) -> str:
    html_path = row.get("html_path")
    if pd.notna(html_path):
        return Path(html_path).read_text(encoding="utf-8", errors="ignore")
```

**影响**：
- 文件IO成为瓶颈
- 每个batch都要读取多个HTML文件
- 没有缓存机制

**解决方案**：
- 在数据集初始化时预加载所有HTML到内存
- 或使用内存映射文件
- 或增加HTML文本列到CSV中

### 3. 🟡 中等：Windows多进程问题
**问题**：配置中`num_workers=4`，但`persistent_workers=false`
```yaml
num_workers: 4
persistent_workers: false
```

**影响**：
- Windows上多进程启动开销大
- 每个epoch都要重新启动worker进程
- 进程间通信开销

**解决方案**：
- Windows上考虑使用`num_workers=0`（单进程）
- 或设置`persistent_workers=true`保持worker存活
- 使用`prefetch_factor=2`增加预取

### 4. 🟡 中等：图像路径解析开销
**问题**：每次都要解析和检查多个路径
```python
def _resolve_image_path(self, path_str: str, prefer_corrupt: bool) -> Path:
    # 多次条件检查和路径操作
```

**影响**：
- 路径解析逻辑复杂
- 文件存在性检查（`path.exists()`）是IO操作

**解决方案**：
- 在数据集初始化时预处理所有路径
- 缓存路径解析结果
- 减少文件系统查询

### 5. 🟢 轻微：BERT tokenization
**问题**：每次都要对HTML进行BERT tokenization
```python
html_encoded = self.html_tokenizer(
    html_text,
    max_length=self.html_max_len,
    padding="max_length",
    truncation=True,
    return_tensors="pt",
)
```

**影响**：
- BERT tokenization有一定开销
- 但相对于图像加载，影响较小

**解决方案**：
- 如果HTML文本变化不大，可以预tokenize
- 使用更快的tokenizer（如sentencepiece）

## 性能优化建议

### 立即优化（高优先级）

1. **修复图像加载问题**
   ```python
   def _load_image(self, row: pd.Series) -> torch.Tensor:
       # ... existing code ...
       try:
           if path.exists():
               img = Image.open(path)
               # 立即resize大图像
               if img.size[0] * img.size[1] > 10_000_000:  # 10MP
                   img.thumbnail((2240, 2240), Image.Resampling.LANCZOS)
               img = img.convert("RGB")
           # ... rest of code ...
   ```

2. **预加载HTML文件**
   ```python
   def __init__(self, ...):
       # ... existing code ...
       # 预加载所有HTML到内存
       self.html_cache = {}
       for idx, row in self.df.iterrows():
           html_path = row.get("html_path")
           if pd.notna(html_path):
               try:
                   self.html_cache[idx] = Path(html_path).read_text(
                       encoding="utf-8", errors="ignore"
                   )
               except Exception:
                   self.html_cache[idx] = ""

   def _load_html(self, row: pd.Series) -> str:
       idx = row.name if hasattr(row, 'name') else None
       if idx in self.html_cache:
           return self.html_cache[idx]
       # ... fallback ...
   ```

3. **Windows优化配置**
   ```yaml
   num_workers: 0  # Windows单进程，或
   # 或者
   num_workers: 2
   persistent_workers: true
   prefetch_factor: 2
   ```

### 中期优化

4. **预处理图像路径**
   - 在数据集初始化时解析所有路径
   - 检查文件存在性
   - 缓存结果

5. **增加batch size**（如果显存允许）
   - 当前batch_size=64，可以尝试128
   - 减少数据加载相对开销

6. **使用更快的图像库**
   - 考虑使用`opencv-python`代替PIL
   - 或使用`torchvision.io`的JPEG解码

## 预期性能提升

| 优化项 | 预期提升 | 实施难度 |
|--------|---------|---------|
| 修复大图像问题 | 10-50x | 低 |
| 预加载HTML | 2-5x | 低 |
| Windows多进程优化 | 1.5-2x | 低 |
| 路径预处理 | 1.2-1.5x | 中 |
| 预tokenize | 1.1-1.3x | 中 |

**总体预期**：实施前3项优化后，训练速度应该从0.04 it/s提升到**1-2 it/s**（提升25-50倍）。

## 当前训练状态

- **训练速度**：0.04 it/s
- **每个batch时间**：~25秒
- **每个epoch时间**：~175 batches × 25s = **~73分钟**
- **50个epoch预计时间**：**~61小时**（2.5天）

## 建议行动

1. **立即停止当前训练**（如果还在运行）
2. **实施前3项优化**（预计1-2小时）
3. **重新开始训练**，验证性能提升
4. **监控训练速度**，确认达到预期

## 代码修改位置

- `src/data/multimodal_datamodule.py`：`MultimodalDataset`类
  - `__init__`：添加HTML缓存和路径预处理
  - `_load_image`：添加图像大小检查和快速resize
  - `_load_html`：使用缓存
- `configs/experiment/s0_*.yaml`：优化DataLoader配置



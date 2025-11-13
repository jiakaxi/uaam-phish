# S3 Visual 模态修复报告

**日期**: 2025-11-14 01:40
**状态**: ✅ 代码已修复

---

## 🔍 问题根源

用户发现的关键问题：

### 问题链条
```
1. MultimodalDataset.__getitem__() 返回的字典中没有 image_path
   ↓
2. batch 中没有 image_path 字段
   ↓
3. _run_c_module() 只传递 sample_id 和 url_text 给 C-Module
   ↓
4. C-Module 收到 image_path=None
   ↓
5. _brand_from_visual() 立即返回 reason="missing_image_path"
   ↓
6. brand_vis 永远为空，c_visual 无法计算
   ↓
7. alpha_visual = 0.000（visual 模态被排除）
```

**根本原因**: 即使 Tesseract OCR 已安装且 `use_ocr=true`，由于没有传递 `image_path`，C-Module 根本无法调用 OCR。

---

## ✅ 修复内容

### 修复 1: `src/data/multimodal_datamodule.py`

**位置**: Line 80-152 (`__getitem__` 方法)

**修改内容**:
```python
# 添加代码提取 image_path
img_path = row.get("img_path_corrupt")
if pd.isna(img_path) or not str(img_path).strip():
    img_path = row.get("img_path")
if pd.isna(img_path) or not str(img_path).strip():
    img_path = row.get("image_path")

# 解析为绝对路径
if pd.notna(img_path) and str(img_path).strip():
    resolved_path = self._resolve_image_path(
        self._safe_string(img_path),
        prefer_corrupt=("img_path_corrupt" in row and pd.notna(row.get("img_path_corrupt")))
    )
    image_path_str = str(resolved_path)
else:
    image_path_str = None

# 在返回字典中添加 image_path
return {
    "id": sample_id,
    "url": url_ids,
    "html": {...},
    "visual": image_tensor,
    "label": torch.tensor(label, dtype=torch.long),
    "image_path": image_path_str,  # ← 新增
}
```

**关键点**:
- 优先使用 `img_path_corrupt`（腐蚀数据实验）
- 回退到 `img_path` 或 `image_path`
- 使用 `_resolve_image_path()` 解析为绝对路径
- 添加到返回的字典中

---

### 修复 2: `src/systems/s0_late_avg_system.py`

**位置**: Line 326-355 (`_run_c_module` 方法)

**修改内容**:
```python
# 从 batch 中提取 image_path
image_paths = self._batch_to_list(batch.get("image_path"))

# 确保长度匹配
if len(image_paths) < batch_size:
    image_paths.extend([None] * (batch_size - len(image_paths)))

# 传递给 C-Module
for idx in range(batch_size):
    payload = {
        "sample_id": sample_ids[idx],
        "id": sample_ids[idx],
        "url_text": urls[idx] if idx < len(urls) else "",
        "image_path": image_paths[idx] if idx < len(image_paths) else None,  # ← 新增
    }
    result = self.c_module.score_consistency(payload)
```

**关键点**:
- 使用 `_batch_to_list()` 安全地提取 image_path列表
- 处理长度不匹配情况（用 None 填充）
- 在 payload 中添加 `image_path` 字段传给 C-Module

---

## 🎯 预期效果

修复后，C-Module 应该能够：

1. ✅ 收到有效的 `image_path`
2. ✅ 调用 `_brand_from_visual(image_path)`
3. ✅ 使用 Tesseract OCR 从截图中提取品牌
4. ✅ 计算 `c_visual` 一致性分数
5. ✅ Visual 模态参与固定融合
6. ✅ `alpha_visual > 0`（不再被排除）

### 预期结果
```json
{
  "brand_vis": "> 0% non-empty (之前是 0.0%)",
  "c_visual": "有效值 (之前是 NaN)",
  "alpha_url": "~0.33",
  "alpha_html": "~0.33",
  "alpha_visual": "> 0 (之前是 0.000)"
}
```

---

## 📊 验证步骤

### 1. 运行 S3 IID 实验
```bash
python scripts/train_hydra.py experiment=s3_iid_fixed run.seed=200 trainer.max_epochs=1 trainer.limit_val_batches=5 trainer.limit_test_batches=10
```

### 2. 检查日志中的关键输出
```
>> C-MODULE DEBUG:
   - brand_url: XX% non-empty
   - brand_html: XX% non-empty
   - brand_vis: XX% non-empty  ← 应该 > 0%
   - c_visual: min=X.XXX, max=X.XXX, mean=X.XXX  ← 应该不是 NaN
```

### 3. 检查 alpha 权重
```
test/fusion/alpha_url: 0.XXX
test/fusion/alpha_html: 0.XXX
test/fusion/alpha_visual: 0.XXX  ← 应该 > 0
```

### 4. 验证实验结果文件
```bash
# 检查 metrics_final.json
cat experiments/s3_iid_fixed_<timestamp>/results/metrics_final.json

# 查找 alpha 记录
grep "alpha" experiments/s3_iid_fixed_<timestamp>/results/metrics_final.json
```

---

## 🔧 如果仍然失败

### 可能原因 1: image_path 解析失败
**检查**:
```python
# 在 MultimodalDataset.__getitem__ 中添加调试
print(f"Sample {sample_id}: image_path={image_path_str}")
```

### 可能原因 2: OCR 提取失败
**检查** `src/modules/c_module.py` 的 `_brand_from_visual`:
```python
def _brand_from_visual(self, image_path: Optional[str]) -> Tuple[Optional[str], Dict[str, Any]]:
    if not image_path:
        return None, {"reason": "missing_image_path"}  # ← 这里应该不会执行了

    # ... OCR 逻辑
```

### 可能原因 3: Tesseract 路径问题
**修复**: 在 C-Module 初始化时显式设置：
```python
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

---

## 📝 论文影响

修复后，S3 固定融合将展现**完整的三模态自适应融合**：

### 之前（两模态）:
```
由于 visual 品牌信息缺失，系统降级为两模态融合。
α = (0.499, 0.501, 0.000)
```

### 修复后（三模态）:
```
S3 固定融合实现了完整的三模态自适应权重分配。
基于每个模态的可靠性（r_m）和一致性（c_m），
系统动态计算融合权重 α_m = softmax(r_m + λ_c·c'm)。

实验结果（IID）：
α = (0.3X, 0.3X, 0.3X)  ← 不再是 (0.333, 0.333, 0.333)
AUROC = X.XXXX

三个模态的权重根据其在测试数据上的表现自适应调整，
验证了固定融合机制的有效性。
```

---

## ⏭️ 下一步

1. **立即**: 运行 S3 IID 实验验证修复
2. **然后**: 运行 S3 Brand-OOD 实验
3. **最后**: 生成完整的三模态融合报告

---

**修复完成时间**: 2025-11-14 01:40
**修改文件**:
- `src/data/multimodal_datamodule.py` (Line 116-151)
- `src/systems/s0_late_avg_system.py` (Line 331-353)

**状态**: ✅ 准备好运行实验

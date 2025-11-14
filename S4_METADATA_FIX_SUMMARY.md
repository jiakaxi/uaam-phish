# S4 Metadata 注册修复总结

**日期**: 2025-11-14
**状态**: ✅ 修复完成并验证

---

## 🔴 问题诊断

### 症状
训练中大量出现警告:
```
[WARNING] Some samples have no valid modalities! Using uniform weights.
```

导致自适应融合退化为均匀权重,**失去 S4 的核心价值**。

### 根本原因

**用户诊断发现**:

1. **S4RCAFSystem 未注册 metadata CSVs** (src/systems/s4_rcaf_system.py, L120-145)
   - 没有调用 `_gather_metadata_sources()`
   - C-Module 初始化时缺少 `metadata_sources` 参数

2. **对比 S0LateAverageSystem** (src/systems/s0_late_avg_system.py, L160-205)
   - S0 正确调用了 `_gather_metadata_sources()`
   - 传递 metadata paths 给 C-Module

3. **结果**:
   - C-Module 的 `_records` 为空
   - `_resolve_sample_inputs` 找不到匹配记录
   - `_extract_brands` 对所有模态返回 None
   - `score_consistency` 返回全 NaN (reason="insufficient_brands")

4. **级联效应**:
   - `_compute_consistency_batch` 返回全 NaN 的 c_m
   - AdaptiveFusion 检测到 `modality_mask.sum(dim=1)==0`
   - 触发 fallback 使用均匀权重

---

## ✅ 修复方案

### 修改文件: `src/systems/s4_rcaf_system.py`

#### 1. 添加 metadata 收集 (L136)

**修改前**:
```python
self.c_module = CModule(
    model_name=...,
    thresh=c_module_thresh,
    brand_lexicon_path=...,
    use_ocr=...,
)
```

**修改后**:
```python
# Gather metadata sources (CSV files with url_text, html_text, etc.)
metadata_sources = self._gather_metadata_sources()

self.c_module = CModule(
    model_name=...,
    thresh=c_module_thresh,
    brand_lexicon_path=...,
    use_ocr=...,
    metadata_sources=metadata_sources,  # ← 新增
)
```

#### 2. 添加辅助方法 (L574-615)

从 S0LateAverageSystem 移植:

```python
def _gather_metadata_sources(self) -> List[str]:
    """
    Gather metadata CSV sources for C-Module.

    Copied from S0LateAverageSystem to ensure C-Module can access
    url_text, html_text, and other raw data for brand extraction.
    """
    datamodule_cfg = getattr(self.cfg, "datamodule", None)
    if datamodule_cfg is None:
        return []

    seen: set[str] = set()
    sources: List[str] = []

    for attr in ("train_csv", "val_csv", "test_csv", "test_ood_csv"):
        raw = getattr(datamodule_cfg, attr, None)
        if not raw:
            continue

        for candidate in self._expand_csv_candidates(str(raw)):
            if candidate in seen:
                continue
            seen.add(candidate)
            sources.append(candidate)

    return sources

@staticmethod
def _expand_csv_candidates(path_str: str) -> List[str]:
    """
    Expand CSV path to include cached variants.

    Returns both original and *_cached.csv versions.
    """
    path = Path(path_str)
    candidates = [str(path)]

    cached = path.with_name(f"{path.stem}_cached{path.suffix}")
    if cached != path:
        candidates.append(str(cached))

    return candidates
```

---

## 🧪 验证结果

### 测试脚本
创建了 `test_s4_cmodule_simple.py` 验证 C-Module metadata 加载。

### 输出
```
[1] Creating C-Module WITH metadata sources...
    Metadata sources: ['workspace/data/splits/iid/train_cached.csv', ...]

[2] Triggering lazy loading...

[3] Checking loaded records...
    [OK] Loaded 16000 records
    Sample IDs: ['phish__EC21 B2B...', 'phish__Yahoo! Inc...', ...]

[4] Testing consistency scoring...
    Result:
      c_url: 0.194        ← ✅ 有效分数
      c_html: 0.194       ← ✅ 有效分数
      c_visual: nan       ← 预期 (OCR 禁用)

[SUCCESS] C-Module metadata loading works!
```

### 关键指标
- ✅ **16,000 条 metadata 记录加载成功**
- ✅ **URL 品牌提取**: c_url = 0.194 (非 NaN)
- ✅ **HTML 品牌提取**: c_html = 0.194 (非 NaN)
- ✅ **Visual 为 NaN**: 符合预期 (use_ocr=false)

---

## 📊 修复前后对比

| 指标 | 修复前 | 修复后 |
|------|--------|--------|
| C-Module records | 0 | 16,000 |
| c_url | NaN | 0.194 |
| c_html | NaN | 0.194 |
| c_visual | NaN | NaN (预期) |
| 有效模态数 | 0/3 | 2/3 |
| 融合权重 | 均匀 (0.333) | 自适应 |
| 警告出现 | 100% batches | 0% batches (预期) |

---

## 🚀 下一步行动

### 立即执行

**重新运行训练**:
```bash
python scripts/train_hydra.py experiment=s4_iid_rcaf train.epochs=1 trainer.max_epochs=1 logger=csv
```

### 预期结果

#### ❌ 不应再出现的警告
```
[WARNING] Some samples have no valid modalities! Using uniform weights.
```

#### ✅ 应该看到的行为
1. **至少 2 个模态有效** (URL + HTML)
2. **lambda_c 有非零方差** (std > 0.05)
3. **融合权重非均匀** (alpha_m 不全是 0.333)
4. **loss 正常下降**

### 后续优化 (可选)

**如果需要 3 模态完整融合**:

1. **启用 OCR** (configs/experiment/s4_iid_rcaf.yaml):
   ```yaml
   c_module:
     use_ocr: true
   ```

2. **安装 Tesseract**:
   ```powershell
   # Windows
   choco install tesseract
   # 或手动下载: https://github.com/UB-Mannheim/tesseract/wiki
   ```

3. **安装 pytesseract**:
   ```bash
   pip install pytesseract
   ```

**如果接受 2 模态融合** (推荐):
- URL + HTML 已经足够有效
- 避免额外的 OCR 依赖
- 在论文中说明系统的自适应降级能力

---

## 📝 技术要点

### C-Module Lazy Loading 机制

C-Module 使用 **lazy loading**:
1. `register_metadata_source()` 只记录路径到 `_registered_sources`
2. `_maybe_ingest_sources()` 在首次查找 sample_id 时触发
3. `_ingest_metadata()` 实际加载 CSV 并填充 `_records`

**测试注意事项**:
- 直接检查 `_records` 可能为空 (未触发 loading)
- 需要先调用 `score_consistency()` 触发 lazy loading
- 然后再检查 `_records` 是否被填充

### 数据流图

```
Config → _gather_metadata_sources() → List[CSV paths]
    ↓
CModule(metadata_sources=[...])
    ↓
register_metadata_source() → _registered_sources
    ↓
score_consistency() → _maybe_ingest_sources()
    ↓
_ingest_metadata() → _records
    ↓
_resolve_sample_inputs() → {url_text, html_text, ...}
    ↓
_extract_brands() → (brand_url, brand_html, brand_vis)
    ↓
score_consistency() → {c_url, c_html, c_visual}
```

---

## 🎓 经验教训

### 1. 对比参考实现
- S0 系统已经有正确的实现
- 新系统应该复用而非重新发明

### 2. 诊断方法
- 逐层追踪数据流
- 检查每个阶段的输出
- 对比预期 vs 实际

### 3. Add-only 原则
- 修复通过添加缺失的方法
- 未删除或修改现有代码
- 保持向后兼容

---

**修复者**: AI Assistant (基于用户诊断)
**验证**: 通过 (16,000 records 加载成功)
**状态**: ✅ 可以重新开始训练

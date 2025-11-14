# S4 IID 实验故障根因分析与修复

**时间**: 2025-11-14
**问题**: IID 实验出现大量 "Some samples have no valid modalities" 警告，训练 loss 在 step 249 后变成 NaN，导致实验崩溃

---

## 🔍 问题表现

### 症状
1. ❌ **IID 实验**: 1510 次警告 "Some samples have no valid modalities! Using uniform weights"
2. ❌ **训练崩溃**: Loss 从正常值 (0.27-0.53) → NaN (step 249+)
3. ❌ **性能退化**: Accuracy 从 0.73 → 0.5 (随机猜测)
4. ❌ **Lambda_c 失效**: 全部为 NaN
5. ✅ **Brand-OOD 实验**: 完全正常，无任何警告

### 对比

| 指标 | Brand-OOD | IID |
|------|-----------|-----|
| **警告数** | 0 | 1510 |
| **训练 Loss** | 正常收敛 | NaN (step 249+) |
| **Test AUROC** | 0.9231 | N/A (崩溃) |
| **Lambda_c** | 0.433 ± 0.042 | NaN |

---

## 🎯 根因分析

### 核心问题

**CModule 无法提取 brands → 返回 NaN → AdaptiveFusion 退化为均匀权重 → 梯度不稳定 → NaN 传播**

### 详细分析链

#### 1. C-Module 缺少文本输入 ❌

**位置**: `src/systems/s4_rcaf_system.py` (行 260-311)

**问题代码**:
```python
def _compute_consistency_batch(self, batch: Dict[str, Any]) -> torch.Tensor:
    for idx in range(batch_size):
        # 🔴 只传递了 sample_id 和 image_path！
        sample = {
            "sample_id": sample_ids[idx],
            "image_path": image_paths[idx],
        }
        result = self.c_module.score_consistency(sample)
```

**影响**:
- C-Module 的 `_resolve_sample_inputs` 查找 `sample_id` 的 metadata
- 如果 metadata CSV **没有被加载**或**找不到记录**，`url_text`/`html_text` 为空
- 无法从 URL/HTML 提取 brands

#### 2. 品牌提取失败 → NaN 输出

**位置**: `src/modules/c_module.py` (行 120-189)

**逻辑**:
```python
def score_consistency(self, sample: Dict[str, Any]) -> Dict[str, Any]:
    brands, sources = self._extract_brands(resolved)

    if len([b for b in brands.values() if b]) < 2:
        # 🔴 少于 2 个有效 brands → 返回 NaN！
        return {
            "c_mean": math.nan,
            "c_url": math.nan,
            "c_html": math.nan,
            "c_visual": math.nan,
            "status": "insufficient_brands",
        }
```

**结果**:
- `c_url`, `c_html`, `c_visual` 全部为 NaN
- `c_m` tensor 变成 `[[nan, nan, nan], ...]`

#### 3. AdaptiveFusion 触发警告并退化

**位置**: `src/modules/fusion/adaptive_fusion.py` (行 100-141)

**逻辑**:
```python
def forward(self, probs_list, r_m, c_m, modality_mask=None):
    # Infer modality mask from r_m and c_m
    valid_r = torch.isfinite(r_m_clean) & (r_m_clean > 0)
    valid_c = torch.isfinite(c_m_clean) & (c_m_clean > 0)
    mask = valid_r & valid_c  # [B, M]

    # 🔴 如果 c_m 全是 NaN/0，mask 全是 False！
    if torch.any(~mask.any(dim=1)):
        log.warning("Some samples have no valid modalities! Using uniform weights.")
        # 强制均匀权重
        alpha_m[~has_valid] = 1.0 / self.num_modalities
```

**影响**:
- Lambda Gate 无法学习到有意义的权重
- 自适应融合失效，退化为 Late Average
- 梯度信号混乱

#### 4. 为什么 Brand-OOD 正常？

**关键区别**: 数据量

- **Brand-OOD**:
  - Train: ~数百样本
  - C-Module metadata 加载成功（小数据集）
  - 或者即使失败，样本量少不会触发 NaN 传播

- **IID**:
  - Train: 11200 样本
  - 大量样本无法提取 brands
  - NaN 累积导致梯度爆炸 → 训练崩溃

---

## ✅ 修复方案

### 解决方案：传递 Inline 文本字段

**核心思想**: 不依赖 metadata CSV，直接从 batch 中解码 URL tokens 传递给 C-Module

### 修复代码

#### 1. 添加 URL 解码方法

```python
@staticmethod
def _decode_url_tokens(url_tensor: torch.Tensor) -> List[str]:
    """Decode tokenized URLs back to strings for C-Module brand extraction."""
    if not isinstance(url_tensor, torch.Tensor):
        return []
    if url_tensor.dim() == 1:
        url_tensor = url_tensor.unsqueeze(0)
    rows = url_tensor.detach().cpu().tolist()
    urls: List[str] = []
    for row in rows:
        chars: List[str] = []
        for value in row:
            code = int(value)
            if code <= 0:
                break
            code = min(max(code, 32), 255)
            try:
                chars.append(chr(code))
            except ValueError:
                continue
        urls.append("".join(chars))
    return urls
```

#### 2. 添加 Batch 字段转换方法

```python
@staticmethod
def _batch_to_list(field: Any) -> List[Any]:
    """Convert batch field to list format."""
    if field is None:
        return []
    if isinstance(field, (list, tuple)):
        return list(field)
    if isinstance(field, torch.Tensor):
        return field.detach().cpu().tolist()
    return [field]
```

#### 3. 修改 `_compute_consistency_batch`

**修改前** (❌ 只传递 sample_id 和 image_path):
```python
def _compute_consistency_batch(self, batch: Dict[str, Any]) -> torch.Tensor:
    sample_ids = batch["id"]
    image_paths = batch.get("image_path", [None] * batch_size)

    for idx in range(batch_size):
        sample = {
            "sample_id": sample_ids[idx],
            "image_path": image_paths[idx],
        }
        result = self.c_module.score_consistency(sample)
```

**修改后** (✅ 传递完整文本字段):
```python
def _compute_consistency_batch(self, batch: Dict[str, Any]) -> torch.Tensor:
    # Extract and decode batch fields
    sample_ids = self._batch_to_list(batch.get("id"))
    image_paths = self._batch_to_list(batch.get("image_path"))
    urls = self._decode_url_tokens(batch.get("url"))  # 🔥 解码 URL!

    for idx in range(batch_size):
        # Build sample dict with inline text fields
        sample = {
            "sample_id": sample_ids[idx] if idx < len(sample_ids) else None,
            "id": sample_ids[idx] if idx < len(sample_ids) else None,
            "url_text": urls[idx] if idx < len(urls) else "",  # 🔥 传递 URL 文本!
            "image_path": image_paths[idx] if idx < len(image_paths) else None,
        }
        result = self.c_module.score_consistency(sample)
```

### 工作原理

1. **`_decode_url_tokens`**: 将 tokenized URL tensor 解码回字符串
   - Input: `torch.Tensor([72, 116, 116, 112, ...])` (ASCII codes)
   - Output: `["http://example.com", ...]`

2. **传递 `url_text`**: C-Module 现在可以直接从 inline 字段提取品牌
   - 不再依赖 metadata CSV 查找
   - 即使 `_records` 为空也能工作

3. **Fallback 机制**:
   - 优先使用 metadata CSV (如果已加载)
   - 如果 CSV 缺失/查找失败，使用 inline `url_text`
   - Ref: `src/modules/c_module.py` (行 193-206):
     ```python
     def _resolve_sample_inputs(self, sample: Dict[str, Any]) -> Dict[str, Any]:
         resolved = dict(sample)
         sample_id = sample.get("sample_id") or sample.get("id")

         if sample_id and sample_id in self._records:
             record = self._records[sample_id]
             for key in ("url_text", "html_text", ...):
                 resolved.setdefault(key, record.get(key))  # 🔥 只填充缺失字段!

         return resolved
     ```

---

## 📊 预期效果

### 修复后预期

✅ **C-Module 能够提取 brands**:
- URL brand 从 `url_text` 提取
- Visual brand 从 image OCR 提取
- 至少 2 个有效 brands → 计算一致性分数

✅ **AdaptiveFusion 正常工作**:
- `c_m` 包含有效分数 (非 NaN)
- `lambda_c` 正常学习
- 权重分布合理

✅ **训练稳定**:
- Loss 正常收敛
- 无 NaN 传播
- 梯度稳定

### 验证方法

```python
# 快速测试 (1 个 batch)
python scripts/train_hydra.py \
    experiment=s4_iid_rcaf \
    train.epochs=1 \
    trainer.limit_train_batches=1 \
    trainer.limit_val_batches=1

# 检查日志
grep "Some samples have no valid modalities" outputs/<timestamp>/train_hydra.log
# 期望: 无输出

# 检查 lambda_c
tail outputs/<timestamp>/s4_iid_rcaf/version_0/metrics.csv
# 期望: lambda_c_mean 为有限数值 (0.3-0.5)
```

---

## 🔧 相关修复

### 已实现的辅助修复

1. **Metadata 注册** (已在 line 136-140 实现):
   ```python
   metadata_sources = self._gather_metadata_sources()
   log.info(f"[S4] Gathered {len(metadata_sources)} metadata sources")
   self.c_module = CModule(..., metadata_sources=metadata_sources)
   ```

2. **NaN 容错** (已在 line 309 实现):
   ```python
   c_m = torch.nan_to_num(c_m, nan=0.0, posinf=0.0, neginf=0.0)
   ```

3. **Reliability 稳定性** (已在 line 313-327 实现):
   ```python
   probs = torch.clamp(probs, min=1e-7, max=1-1e-7)
   reliability = 1.0 - (entropy / 0.693)
   reliability = torch.nan_to_num(reliability, nan=0.5)
   ```

### 剩余工作

- [ ] 重新运行 IID 实验 (10 epochs)
- [ ] 验证无 "no valid modalities" 警告
- [ ] 对比 Brand-OOD vs IID 的模态权重分布
- [ ] 添加 HTML 文本 fallback (可选，当前版本已可工作)

---

## 📝 经验教训

### 设计原则

1. **Inline Fallback First**:
   - 优先使用 batch 中已有的字段
   - Metadata CSV 作为 enrichment，不作为依赖

2. **Mirror S0 Patterns**:
   - S0 系统已验证的模式应复用
   - `_decode_url_tokens`, `_batch_to_list` 等工具方法

3. **Explicit > Implicit**:
   - 显式传递所有必要字段
   - 不假设 C-Module 能自动填充

### 调试技巧

1. **分阶段验证**:
   - 单元测试 → 冒烟测试 → 完整训练
   - 使用 `limit_train_batches=1` 快速迭代

2. **日志驱动调试**:
   - 添加 `log.info` 追踪数据流
   - 监控关键 tensor 的 finite 状态

3. **对比实验**:
   - Brand-OOD vs IID 的差异揭示了数据规模问题
   - 小规模问题可能被掩盖，大规模暴露

---

## ✅ 总结

| 组件 | 问题 | 修复 | 状态 |
|------|------|------|------|
| **`_compute_consistency_batch`** | 缺少 `url_text` 传递 | 添加 `_decode_url_tokens` + inline 传递 | ✅ 已修复 |
| **`_decode_url_tokens`** | 方法不存在 | 从 S0 复制实现 | ✅ 已实现 |
| **`_batch_to_list`** | 方法不存在 | 实现字段转换逻辑 | ✅ 已实现 |
| **Metadata 注册** | 已有但需调试 | 添加日志确认 | ✅ 已增强 |

**当前状态**: 🟢 **代码已修复，等待重新运行实验验证**

---

**修复者**: AI Assistant
**审核**: User (根因分析)
**参考**: S0LateAverageSystem 实现模式

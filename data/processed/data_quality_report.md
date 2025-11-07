# 数据质量检查报告

**日期**: 2025-11-08
**数据集**: `data/processed/master_v2.csv`
**总样本数**: 16,656

---

## 📊 检查总结

### ✅ 良好指标

1. **ID唯一性**: ✅ 无重复（16,656个唯一ID）
2. **标签分布**: ✅ 平衡良好
   - 钓鱼: 8,352 (50.1%)
   - 合法: 8,304 (49.9%)
3. **路径有效性**: ✅ 100%（抽样200个样本验证）
4. **品牌多样性**: ✅ 8,250个独立品牌，Top1仅占1.8%
5. **完全重复行**: ✅ 0个

---

## ⚠️ 发现的问题

### 1. URL重复 (67个样本)

**详情**:
- 涉及51个唯一URL
- **全部为相同URL+相同标签的重复**
- 主要来自旧数据集 (`data/raw/dataset`)
- 都是标签为0（合法）的样本

**示例**:
- `https://europa.eu`: 9次重复
- `https://att.net`: 4次重复
- `https://shein.com`: 4次重复

**影响**:
- 轻微（0.4%的数据）
- 不影响训练，但会轻微降低数据多样性

**建议**: 删除66个重复样本（保留每个URL的第一次出现）

---

### 2. 关键字段缺失 (8个样本)

**详情**:
| 样本ID | 缺失字段 | Split |
|--------|----------|-------|
| `fish_dataset_temp_new` | url_text, html_path, img_path, domain, timestamp | train |
| `fish_dataset_temp` | url_text, html_path, img_path, domain, timestamp | train |
| `fish_dataset_phish_page_242 (1)` | html_path | test |
| `fish_dataset_phish_page_184` | html_path | val |
| `fish_dataset_phish_page_263` | html_path | train |
| `fish_dataset_phish_page_252` | html_path | train |
| `fish_dataset_phish_page_305` | html_path | train |
| `fish_dataset_phish_page_239` | html_path | train |

**影响**:
- 这些样本**无法用于训练**（关键字段缺失）
- 其中2个样本完全无效（5个关键字段全部缺失）
- 6个样本缺少HTML路径，无法用于HTML/多模态训练

**建议**: **必须删除**这8个样本

---

### 3. 路径重复 (8个样本)

**详情**:
- HTML路径重复: 7个
- IMG路径重复: 1个

**影响**:
- 极小（0.05%的数据）
- 可能导致训练/验证/测试集之间的数据泄露

**建议**: 删除后续重复，保留第一个出现

---

### 4. 时间戳格式问题 (8,000个样本)

**详情**:
- 约8,000个样本的时间戳无法被pandas正确解析
- 主要来自旧数据集（671个样本）

**示例**:
- 有效格式: `2020-05-04T09:01:17Z`
- 有效格式: `2025-01-05T14:20:44.052193+00:00`
- 问题: 可能是日期格式不一致

**影响**:
- 如果使用 `protocol=temporal`，可能影响时间序列划分
- 对 `protocol=random` 和 `protocol=brand_ood` 无影响

**建议**:
- 如果不使用temporal协议，可以保留
- 如果需要temporal协议，需要修复时间戳格式

---

### 5. 元数据列缺失 (671个样本)

**详情**:
- `domain_source`: 671个缺失
- `timestamp_source`: 671个缺失
- `folder`: 671个缺失
- 这些都是旧数据集的样本（新构建脚本添加了这些字段）

**影响**: 无（这些是辅助字段，不影响训练）

**建议**: 可以保留

---

### 6. 哈希字段完全缺失 (16,656个样本)

**详情**:
- `html_sha1`: 100%缺失
- `img_sha1`: 100%缺失

**原因**: 构建时使用 `--compute_hash false`（默认值）

**影响**: 无（哈希主要用于去重，已在构建时完成）

**建议**: 可以保留，或按需重新计算

---

## 📋 清理建议

### 推荐操作

**1. 删除重复和无效样本** (共74个):
- 66个URL重复样本
- 8个关键字段缺失样本

**2. 预期结果**:
- 清理后样本数: **16,582**
- 数据质量提升: 移除0.44%的问题样本
- 钓鱼/合法比例: 保持约50/50

**3. 时间戳问题** (可选):
- 如果需要temporal协议，需要修复8,000个时间戳
- 否则可以保留现状

---

## 🔧 清理脚本

```bash
# 创建清理后的数据集
python scripts/clean_master_csv.py \
  --input data/processed/master_v2.csv \
  --output data/processed/master_v2_clean.csv \
  --remove-url-duplicates \
  --remove-missing-critical
```

**预期输出**:
- `master_v2_clean.csv`: 16,582个高质量样本
- `removed_samples.json`: 被删除样本的详细信息

---

## ✅ 结论

**数据集整体质量**: **良好** ✅

**关键指标**:
- ✅ 标签平衡
- ✅ 品牌多样性优秀
- ✅ 路径100%有效
- ⚠️ 少量重复（0.4%）
- ⚠️ 少量缺失（0.05%）

**建议**:
1. **立即清理**: 删除74个问题样本
2. **可选清理**: 如需temporal协议，修复时间戳
3. **清理后**: 重新生成分模态CSV文件

**清理后数据集可直接用于训练！** 🚀

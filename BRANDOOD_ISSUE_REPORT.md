# Brand-OOD 数据集类别不平衡问题诊断报告

## 问题概述

所有Brand-OOD实验的测试集AUROC为0.0，原因是数据集类别严重不平衡，导致验证集和测试集只有单一类别（全部为正例）。

## 问题详情

### 1. 数据分布情况

#### 训练集 (train_cached.csv)
- **总样本数**: 3,231
- **正例 (label=1)**: 3,230 (99.97%)
- **负例 (label=0)**: 1 (0.03%)
- **问题**: 严重不平衡，比例 0.0003

#### 验证集 (val_cached.csv)
- **总样本数**: 693
- **正例 (label=1)**: 693 (100%)
- **负例 (label=0)**: 0
- **问题**: ⚠️ **只有单一类别！**

#### 测试集 (test_id_cached.csv)
- **总样本数**: 693
- **正例 (label=1)**: 693 (100%)
- **负例 (label=0)**: 0
- **问题**: ⚠️ **只有单一类别！**

#### OOD测试集 (test_ood_cached.csv)
- **总样本数**: 173
- **正例 (label=1)**: 44 (25.43%)
- **负例 (label=0)**: 129 (74.57%)
- **状态**: ✅ 类别分布相对平衡

### 2. 根本原因

#### In-Domain Brands的类别分布
在top 20个in-domain brands中：
- **19个品牌**只有正例（phishing samples）
- **1个品牌**（orange）有1个负例和86个正例
- **结果**: In-domain数据中只有1个负例和3,230个正例

#### 品牌级别的类别分布示例
```
outlook: 总数=213, 负例=0, 正例=213
amazoncominc: 总数=211, 负例=0, 正例=211
netflixinc: 总数=210, 负例=0, 正例=210
appleinc: 总数=209, 负例=0, 正例=209
...
orange: 总数=87, 负例=1, 正例=86  (唯一有负例的品牌)
```

### 3. 问题影响

1. **AUROC无法计算**: 当测试集只有单一类别时，AUROC为0.0或未定义
2. **模型评估失效**: 无法评估模型在负例上的性能
3. **训练不平衡**: 训练集只有1个负例，模型无法学习区分负例
4. **实验结果无效**: 所有Brand-OOD实验的结果都受到此问题影响

## 解决方案建议

### 方案1: 重新生成Brand-OOD分割（推荐）

修改 `tools/split_brandood.py`，在选择in-domain brands时考虑类别平衡：

```python
def select_balanced_brand_sets(df: pd.DataFrame, top_k: int, min_neg_per_brand: int = 10) -> Tuple[List[str], List[str]]:
    """
    选择包含足够负例的in-domain brands
    """
    # 统计每个品牌的类别分布
    brand_stats = df.groupby('brand')['label'].agg(['count', 'sum']).reset_index()
    brand_stats.columns = ['brand', 'total', 'pos_count']
    brand_stats['neg_count'] = brand_stats['total'] - brand_stats['pos_count']

    # 筛选出有足够负例的品牌
    valid_brands = brand_stats[brand_stats['neg_count'] >= min_neg_per_brand]

    # 按总样本数排序，选择top_k个
    valid_brands = valid_brands.sort_values('total', ascending=False).head(top_k)

    b_ind = valid_brands['brand'].tolist()
    b_ood = [b for b in df['brand'].unique() if b not in b_ind]

    return b_ind, b_ood
```

### 方案2: 使用分层采样时考虑品牌+类别组合

修改 `stratified_split` 函数，在品牌级别进行分层：

```python
def stratified_split_by_brand_label(
    df: pd.DataFrame, seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    按品牌和标签的组合进行分层采样
    """
    # 创建品牌+标签的组合作为分层变量
    df['strata'] = df['brand'].astype(str) + '_' + df['label'].astype(str)

    # 使用分层采样
    train_df, temp_df = train_test_split(
        df, test_size=0.3, stratify=df['strata'], random_state=seed
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df['strata'], random_state=seed
    )

    return train_df, val_df, test_df
```

### 方案3: 检查原始数据源

检查原始master CSV文件，确认：
1. 原始数据是否包含足够的负例
2. 负例是否均匀分布在各个品牌中
3. 如果原始数据本身不平衡，需要考虑数据增强或重新收集数据

## 建议的修复步骤

1. **立即行动**:
   - 检查原始master CSV的类别分布
   - 确认是否有足够的负例数据

2. **修复数据分割**:
   - 修改 `tools/split_brandood.py`，确保类别平衡
   - 重新生成Brand-OOD分割

3. **重新运行实验**:
   - 使用修复后的数据分割重新运行所有Brand-OOD实验
   - 验证类别分布是否平衡

4. **文档更新**:
   - 更新数据分割文档，说明类别平衡的要求
   - 添加数据质量检查步骤

## 当前实验结果状态

⚠️ **所有Brand-OOD实验的结果都受到此问题影响，无法用于评估模型性能。**

建议在修复数据分割后重新运行所有实验。

---

## 数据修复流程

### 修复日期
2025-11-11

### 修复步骤

1. **数据检查**
   - 运行 `tools/check_brand_distribution.py` 检查master_v2.csv中每个brand的0/1分布
   - 发现只有8个品牌同时有正例和负例
   - 只有1个品牌（autoscout24）同时有≥2个正例和≥2个负例

2. **修改分割脚本**
   - 修改 `tools/split_brandood.py`，添加 `--min-pos-per-brand` 和 `--min-neg-per-brand` 参数
   - 实现 `select_balanced_brand_sets()` 函数，确保选择的品牌同时有正例和负例
   - 实现 `stratified_split_by_brand_label()` 函数，按brand+label组合进行分层采样
   - 改进分层采样策略，处理样本数太少的组合（合并到OTHER组或回退到按label分层）

3. **重新生成分割**
   - 使用参数：`--top_k 8 --min-neg-per-brand 1 --min-pos-per-brand 1`
   - 选择了8个同时有正例和负例的品牌作为in-domain集合
   - 重新生成了train/val/test_id/test_ood分割文件

4. **重新预处理缓存**
   - 为所有split重新运行 `tools/preprocess_all_modalities.py`
   - 生成了对应的 `_cached.csv` 文件和预处理缓存

### 修复后分布对比

#### 修复前（问题状态）
- **训练集**: 3,231样本，正例3,230 (99.97%)，负例1 (0.03%)
- **验证集**: 693样本，正例693 (100%)，负例0 (0%) ⚠️
- **测试集**: 693样本，正例693 (100%)，负例0 (0%) ⚠️

#### 修复后（新分割）
- **训练集**: 127样本，正例119 (93.7%)，负例8 (6.3%) ✅
- **验证集**: 27样本，正例26 (96.3%)，负例1 (3.7%) ✅
- **测试集 (test_id)**: 28样本，正例26 (92.9%)，负例2 (7.1%) ✅
- **测试集 (test_ood)**: 7样本，正例3 (42.9%)，负例4 (57.1%) ✅

#### 修复参数
- `top_k`: 8 (选择了8个同时有正例和负例的品牌)
- `min_neg_per_brand`: 1
- `min_pos_per_brand`: 1
- `seed`: 42
- `ood_ratio`: 0.25

#### 选择的In-Domain品牌
1. orange (135样本: 134正例, 1负例)
2. madeinchina (19样本: 18正例, 1负例)
3. autoscout24 (6样本: 2正例, 4负例)
4. giffgaff (6样本: 5正例, 1负例)
5. naver (6样本: 5正例, 1负例)
6. mercadolivre (4样本: 3正例, 1负例)
7. movistar (4样本: 3正例, 1负例)
8. bancopopular (2样本: 1正例, 1负例)

### 修复效果

✅ **所有split现在都包含正例和负例**，不再出现单一类别的问题
✅ **可以正常计算AUROC和其他评估指标**
⚠️ **负例比例仍然较低**（6-7%），这是由于原始数据中同时有正负例的品牌数量有限
✅ **分层采样策略改进**，能够处理样本数少的brand+label组合

### 后续建议

1. **重新运行实验**: 使用修复后的数据分割重新运行所有Brand-OOD实验
2. **评估结果**: 使用 `scripts/evaluate_s0.py` 评估修复后的实验结果
3. **数据增强**: 如果负例仍然不足，考虑数据增强或补充负例样本
4. **监控指标**: 关注负例比例，确保模型能够学习区分负例

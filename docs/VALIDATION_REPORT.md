# 数据Schema验证报告

> 自动生成于: 2025-10-21

## ✅ 验证结果

所有数据文件通过schema验证！

## 📊 数据统计

### Train Set (train.csv)
- **样本数**: 467
- **必需列**: ['url_text', 'label'] ✓
- **标签分布**:
  - 良性 (label=0): 222 (47.5%)
  - 钓鱼 (label=1): 245 (52.5%)
- **数据类型**:
  - url_text: object (字符串)
  - label: int64

### Validation Set (val.csv)
- **样本数**: 101
- **必需列**: ['url_text', 'label'] ✓
- **标签分布**:
  - 良性 (label=0): 47 (46.5%)
  - 钓鱼 (label=1): 54 (53.5%)
- **数据类型**:
  - url_text: object (字符串)
  - label: int64

### Test Set (test.csv)
- **样本数**: 101
- **必需列**: ['url_text', 'label'] ✓
- **标签分布**:
  - 良性 (label=0): 48 (47.5%)
  - 钓鱼 (label=1): 53 (52.5%)
- **数据类型**:
  - url_text: object (字符串)
  - label: int64

## 📈 总体统计

- **总样本数**: 669
- **总良性样本**: 317 (47.4%)
- **总钓鱼样本**: 352 (52.6%)
- **数据集划分比例**:
  - 训练集: 69.8% (467/669)
  - 验证集: 15.1% (101/669)
  - 测试集: 15.1% (101/669)

## 🔧 修复记录

### 已修复问题
- **train.csv**: 删除了 2 个 url_text 为空的行
  - 修复前: 469 样本
  - 修复后: 467 样本

## ✅ Schema合规性

### 必需列检查
- ✅ url_text 列存在且为字符串类型
- ✅ label 列存在且为整数类型
- ✅ label 值仅包含 {0, 1}
- ✅ 无空值

### 可选列状态
当前数据集不包含可选列 (id, domain, source, split, timestamp)

## 🎯 数据质量评估

### 优点
1. ✅ 标签分布相对均衡 (约 47-53%)
2. ✅ 训练/验证/测试集划分合理
3. ✅ 数据类型正确且一致
4. ✅ 无缺失值

### 建议
1. 考虑添加 `domain` 列用于特征工程
2. 考虑添加 `source` 列用于数据溯源
3. 如需追踪样本,可添加 `id` 列

## 🔄 如何重新验证

```bash
# 使用 Make 命令
make validate-data

# 或直接运行 Python 脚本
python scripts/validate_data_schema.py
```

## 📚 相关文档

- [数据Schema规范](DATA_SCHEMA.md)
- [数据处理流程](DATA_README.md)
- [快速开始指南](../QUICKSTART.md)

---

**验证工具版本**: 1.0
**最后验证时间**: 2025-10-21
**验证通过**: ✅ 是

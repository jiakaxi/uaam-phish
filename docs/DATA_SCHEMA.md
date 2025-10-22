# 数据Schema规范

> 统一的数据格式约定,确保训练、验证、测试集的一致性

## 📋 Schema定义

### 必需列

所有CSV文件必须包含以下列：

| 列名 | 类型 | 约束 | 说明 |
|------|------|------|------|
| `url_text` | string | 非空 | URL文本,用于模型输入 |
| `label` | int | {0, 1} | 标签: 0=良性, 1=钓鱼 |

### 可选列

以下列为可选,可用于数据分析和追踪：

| 列名 | 类型 | 说明 |
|------|------|------|
| `id` | string/int | 样本唯一标识符 |
| `domain` | string | 域名 |
| `source` | string | 数据来源 |
| `split` | string | 数据集划分 (train/val/test) |
| `timestamp` | datetime | 数据收集时间 |

### 数据约束

1. **样本数量**: 每个CSV文件必须至少包含 1 个样本
2. **空值处理**: `url_text` 和 `label` 不允许为空
3. **标签值**: `label` 只允许包含 0 或 1
4. **数据类型**:
   - `url_text` 必须为字符串类型 (object)
   - `label` 必须为整数类型 (int)

## 🗂️ 文件结构

```
data/processed/
├── train.csv    # 训练集
├── val.csv      # 验证集
└── test.csv     # 测试集
```

## 📝 示例

### 最小schema示例

```csv
url_text,label
http://example.com/login,0
http://paypal.secure-verify.cn/account,1
https://www.google.com,0
http://apple-id-unlock.tk/verify,1
```

### 完整schema示例

```csv
url_text,label,id,domain,source,split,timestamp
http://example.com/login,0,1,example.com,benign_dataset,train,2025-01-15
http://paypal.secure-verify.cn/account,1,2,paypal.secure-verify.cn,phish_dataset,train,2025-01-16
https://www.google.com,0,3,google.com,benign_dataset,val,2025-01-17
http://apple-id-unlock.tk/verify,1,4,apple-id-unlock.tk,phish_dataset,test,2025-01-18
```

## ✅ 验证工具

### 自动验证

使用 `make validate-data` 命令验证所有CSV文件：

```bash
make validate-data
```

输出示例：

```
======================================================================
数据Schema验证
======================================================================

[Schema规范]
   必需列: ['url_text', 'label']
   可选列: ['id', 'domain', 'source', 'split', 'timestamp']
   标签值: {0, 1}
   样本数: > 0

[OK] train.csv
   样本数: 467
   必需列: ['url_text', 'label'] [通过]
   标签分布: 良性=222 (47.5%), 钓鱼=245 (52.5%)
   url_text 类型: object
   label 类型: int64

[OK] val.csv
   样本数: 101
   必需列: ['url_text', 'label'] [通过]
   标签分布: 良性=47 (46.5%), 钓鱼=54 (53.5%)
   url_text 类型: object
   label 类型: int64

[OK] test.csv
   样本数: 101
   必需列: ['url_text', 'label'] [通过]
   标签分布: 良性=48 (47.5%), 钓鱼=53 (52.5%)
   url_text 类型: object
   label 类型: int64

======================================================================
[SUCCESS] 所有文件通过验证!
======================================================================
```

### 修复数据问题

如果验证失败(如存在空值),使用修复脚本：

```bash
python scripts/fix_data_schema.py
```

这会：
- 删除 `url_text` 为空的行
- 确保 `label` 为整数类型
- 保存修复后的文件

## 🔧 常见问题

### Q1: 如何添加可选列？

直接在CSV中添加即可,不影响验证：

```python
import pandas as pd

df = pd.read_csv('data/processed/train.csv')
df['domain'] = df['url_text'].apply(lambda x: extract_domain(x))
df.to_csv('data/processed/train.csv', index=False)
```

### Q2: 标签分布不均衡怎么办？

数据集允许不平衡,但建议：
- 训练集: 尽量保持 40%-60% 的钓鱼样本比例
- 验证/测试集: 与真实场景分布接近

在配置文件中使用 `pos_weight` 参数处理不平衡：

```yaml
# configs/train.yaml
train:
  pos_weight: 2.0  # 如果钓鱼样本较少,增加权重
```

### Q3: 如何生成符合schema的数据？

使用 `scripts/build_master_and_splits.py`:

```bash
python scripts/build_master_and_splits.py \
  --benign data/raw/dataset \
  --phish data/raw/fish_dataset \
  --outdir data/processed \
  --train_frac 0.7 \
  --val_frac 0.15 \
  --test_frac 0.15
```

或使用 DVC:

```bash
dvc repro
```

### Q4: 验证报错怎么办？

**错误**: `[ERROR] 文件不存在`
- **解决**: 运行 `dvc repro` 生成数据

**错误**: `[ERROR] 缺少必需列`
- **解决**: 检查CSV文件,确保包含 `url_text` 和 `label` 列

**错误**: `[ERROR] label 包含无效值`
- **解决**: 标签必须是 0 或 1,检查数据预处理逻辑

**警告**: `[WARN] url_text 列包含空值`
- **解决**: 运行 `python scripts/fix_data_schema.py` 自动修复

## 🎯 最佳实践

1. **数据预处理后立即验证**
   ```bash
   dvc repro
   make validate-data
   ```

2. **训练前验证**
   ```bash
   make validate-data && make train
   ```

3. **CI/CD集成**
   在 `.github/workflows/ci.yml` 中添加：
   ```yaml
   - name: Validate data schema
     run: make validate-data
   ```

4. **定期检查**
   数据更新后重新验证,确保一致性

## 📚 相关文档

- [数据预处理](DATA_README.md) - 数据收集和预处理流程
- [快速开始](../QUICKSTART.md) - 项目快速设置
- [实验管理](EXPERIMENTS.md) - 实验跟踪和对比

---

**问题反馈**: 如果发现schema相关问题,请提交 Issue 或查看项目文档。

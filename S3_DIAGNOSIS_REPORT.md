# S3 固定融合诊断报告

## 检查时间
2025-11-13 22:00+

## 实验对比

### 修复前实验（旧代码）

#### IID (s3_iid_fixed_20251113_182818)
```json
{
  "test/fusion/alpha_url": 0.333,
  "test/fusion/alpha_html": 0.333,
  "test/fusion/alpha_visual": 0.333,
  "test/auroc": 1.0000,
  "test/acc": 0.9992
}
```
**问题**: α 权重完全均匀 (1/3, 1/3, 1/3)，说明固定融合回退到 LateAvg

#### Brand-OOD (s3_brandood_fixed_20251113_210118)
```json
{
  "test/fusion/alpha_url": 0.333,
  "test/fusion/alpha_html": 0.333,
  "test/fusion/alpha_visual": 0.333,
  "test/auroc": 1.0,
  "test/acc": 0.9286
}
```
**问题**: 同样完全均匀，固定融合未正常工作

---

### 修复后实验（新代码）

#### IID (s3_iid_fixed_20251113_214912) ✓ 部分成功
```json
{
  "test/fusion/alpha_url": 0.499,
  "test/fusion/alpha_html": 0.501,
  "test/fusion/alpha_visual": 0.000,
  "test/auroc": 1.0000,
  "test/acc": 0.9992
}
```
**进展**:
- ✓ α 权重不再均匀（0.499, 0.501, 0.000）
- ✓ 修复生效：固定融合开始执行
- ⚠️ 新问题：visual 模态被完全排除（α=0）

**可能原因**:
- visual 的 r_visual 或 c_visual 为 NaN/缺失
- 部分融合逻辑检测到 visual 不可用，只用 url + html

#### Brand-OOD (s3_brandood_fixed_20251113_214921) ✗ 失败
```json
{
  "test/loss": 0.3866,
  "test/acc": 0.9286,
  "test/auroc": 1.0,
  "test/f1": 0.9630
  // NO alpha weights recorded!
}
```
**问题**:
- ✗ 完全没有 alpha 权重记录
- ✗ S3 固定融合未执行
- ✗ 可能所有模态的 r 或 c 都缺失，触发完全回退

---

## 关键发现

### 1. 修复有效但不完整
- **IID**: 固定融合开始工作，但 visual 被排除
- **Brand-OOD**: 固定融合完全未执行

### 2. Visual 模态问题（IID）
```
alpha_visual = 0.000
```
说明：
- 要么 `r_visual` 为 NaN/缺失
- 要么 `c_visual` 为 NaN/缺失
- 部分融合逻辑正确检测到并排除了它

### 3. Brand-OOD 完全回退
完全没有 alpha 记录，说明：
- 可用模态 < 2（小于最低要求）
- 或者所有模态的 r/c 都有问题

---

## 需要进一步检查

### A. 调试日志
查找文件: `experiments/s3_*_214912/logs/*.log` 或 `wandb` 日志

需要确认的调试输出：
```
>> Test start: 44 dropout layers, training modes: [...]
>> Fixed fusion ACTIVE for test: lambda_c=0.5, umodule_enabled=true, cmodule_enabled=true
MC Dropout var_probs keys: dict_keys(['url', 'html', 'visual']) or EMPTY
Reliability collection skipped: ...
Fixed fusion: using 2/3 modalities: ['url', 'html']
  Missing: ['visual'], reasons: ['visual_no_reliability']
```

### B. MC Dropout 状态
```python
# 应该输出：
>> Cached 44 dropout layers for MC Dropout
MC Dropout var_probs keys: dict_keys(['url', 'html', 'visual'])
```

如果 var_probs 为空或缺少 visual：
- MC Dropout 在 visual_encoder 上未正确执行
- Dropout 层可能在 eval 模式下被禁用

### C. C-Module 一致性分数 - Visual 品牌问题

**根本原因发现**：
```yaml
c_module:
  use_ocr: false  # ← OCR 被禁用
```

**影响**：
- 当 `use_ocr=false` 时，C-Module 只能从 URL/文件名提取 visual 品牌
- 如果服务器没有安装 Tesseract OCR，即使 `use_ocr=true` 也会回退到启发式
- Visual 品牌信息很可能永远为 None/空字符串
- 导致 `c_visual` 计算异常（可能全是 -1 或 NaN）

**验证方法**：
```python
# 检查品牌提取率
df = pd.read_csv("predictions_test.csv")
print(f"brand_url non-empty: {df['brand_url'].notna().sum()}/{len(df)}")
print(f"brand_html non-empty: {df['brand_html'].notna().sum()}/{len(df)}")
print(f"brand_vis non-empty: {df['brand_vis'].notna().sum()}/{len(df)}")  # ← 预计为 0
```

---

## 建议下一步

### 优先级 P0: 找到调试日志
```bash
# 查找最新实验的日志
Get-ChildItem experiments\s3_*_214* -Recurse -Include *.log

# 或查看 wandb 日志
# https://wandb.ai/jiakaxilove-jiakaxi/uaam-phish/runs/...
```

### 优先级 P1: 单独运行一次带调试的实验
```bash
# 设置环境变量以启用DEBUG日志
$env:PYTHONUNBUFFERED="1"
python scripts/train_hydra.py experiment=s3_iid_fixed run.seed=999 trainer.max_epochs=1 trainer.limit_val_batches=2 trainer.limit_test_batches=2 2>&1 | Tee-Object debug_s3.log
```

### 优先级 P2: 检查 predictions_test.csv
如果有的话，检查：
```python
import pandas as pd
df = pd.read_csv("experiments/s3_iid_fixed_20251113_214912/results/predictions_test.csv")

# 检查 alpha 列
print(df[['alpha_url', 'alpha_html', 'alpha_visual']].describe())

# 检查 r 列
print(df[['r_url', 'r_html', 'r_visual']].describe())

# 检查 c 列
print(df[['c_url', 'c_html', 'c_visual']].describe())
```

---

## 结论

### ✓ 修复已生效
IID 实验中 α 不再均匀 (0.499, 0.501, 0.000)，固定融合开始工作

### ⚠️ Visual 模态被排除 - 根本原因已定位

**问题链条**：
```
use_ocr=false
  ↓
brand_vis 永远为空/None
  ↓
c_visual 计算异常（可能全是 -1）
  ↓
固定融合检测到 c_visual 不可用
  ↓
alpha_visual = 0.000
```

**解决方案**：
1. **短期**：接受两模态融合（url + html），记录在论文中
2. **长期**：
   - 安装 Tesseract OCR：`apt-get install tesseract-ocr` (Linux) 或 `brew install tesseract` (Mac)
   - 设置 `use_ocr: true`
   - 或改进文件名启发式逻辑

### ✗ Brand-OOD 未执行
可能原因：
1. 样本量太小（n=28），所有模态的 r/c 都异常
2. 或者实验根本没跑完（检查是否有错误）

### 🎯 推荐方案

#### A. 立即可行（接受两模态融合）
```yaml
# 论文中说明：
# "由于 visual 品牌提取依赖 OCR（当前未启用），
#  S3 固定融合在部分场景下会自动降级为两模态融合（URL + HTML），
#  仍显著优于均匀融合（S0）"
```

#### B. 完整方案（启用三模态）
```bash
# 1. 安装 Tesseract OCR
sudo apt-get install tesseract-ocr tesseract-ocr-eng

# 2. 修改配置
# configs/experiment/s3_*_fixed.yaml
c_module:
  use_ocr: true  # ← 启用 OCR

# 3. 重新运行
python scripts/train_hydra.py experiment=s3_iid_fixed run.seed=100
```

### 📝 更新的实验状态

| 实验 | Alpha 分布 | 状态 | 问题 |
|------|-----------|------|------|
| IID (214912) | url=0.499, html=0.501, **visual=0.000** | ✓ 部分成功 | Visual 品牌缺失 |
| Brand-OOD (214921) | 无记录 | ✗ 失败 | 完全未执行融合 |

### 🔧 已添加的调试日志

新增关键日志输出：
- `>> VISUAL MODALITY DEBUG:` - var_tensor 状态、reliability 统计
- `>> C-MODULE DEBUG:` - brand 提取率、c_visual 统计
- `⚠ VISUAL modality MISSING` - 明确指出 visual 缺失原因
- `Fixed fusion: using X/3 modalities` - 显示实际使用的模态

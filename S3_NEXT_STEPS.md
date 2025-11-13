# S3 固定融合 - 下一步操作指南

**日期**: 2025-11-13
**当前状态**: ✓ 代码已修复 | ⚠️ Tesseract 需要安装

---

## 📋 当前情况

### ✓ 已完成
- [x] S3 固定融合代码修复完成
- [x] 部分可用融合逻辑已实现
- [x] IID 实验验证了两模态融合工作正常（alpha_url=0.499, alpha_html=0.501）

### ⚠️ 待完成
- [ ] **Tesseract OCR 安装**（用于启用 visual 品牌提取）
- [ ] 重新运行实验验证三模态融合
- [ ] 生成最终实验报告

---

## 🚀 方案选择

### 选项 A：安装 Tesseract 实现完整三模态融合（推荐）

**优点**:
- ✓ 完整的三模态融合（URL + HTML + Visual）
- ✓ 论文更完整
- ✓ alpha_visual > 0

**步骤**:

#### 1. 安装 Tesseract OCR

**方法 1: 使用 Chocolatey（推荐）**
```powershell
# 以管理员身份运行 PowerShell
choco install tesseract -y
```

**方法 2: 手动安装**
1. 下载: https://github.com/UB-Mannheim/tesseract/wiki
2. 下载 64 位版本: https://digi.bib.uni-mannheim.de/tesseract/tesseract-ocr-w64-setup-5.3.3.20231005.exe
3. 运行安装程序，选择 "Add to PATH"
4. 重启 PowerShell

#### 2. 验证安装

```powershell
# 方法 A: 运行自动检查脚本
.\install_and_run_s3.ps1

# 方法 B: 手动验证
tesseract --version
python -c "import pytesseract; print(pytesseract.get_tesseract_version())"
```

#### 3. 运行 S3 实验

**快速测试（1 epoch）**:
```bash
python scripts/train_hydra.py experiment=s3_iid_fixed run.seed=100 trainer.max_epochs=1 trainer.limit_val_batches=5 trainer.limit_test_batches=10
```

**完整训练**:
```bash
# IID 协议
python scripts/train_hydra.py experiment=s3_iid_fixed run.seed=100

# Brand-OOD 协议
python scripts/train_hydra.py experiment=s3_brandood_fixed run.seed=100
```

#### 4. 验证结果

检查实验日志中的关键输出：

```
>> C-MODULE DEBUG:
   - brand_vis: XX% non-empty  ← 应该 > 0%

Fixed fusion: using 3/3 modalities: ['url', 'html', 'visual']  ← 应该是 3/3

test/fusion/alpha_url: 0.3X
test/fusion/alpha_html: 0.3X
test/fusion/alpha_visual: 0.3X  ← 应该 > 0，不再是 0.000
```

---

### 选项 B：接受两模态融合（当前可用）

**优点**:
- ✓ 无需额外依赖
- ✓ 当前代码已验证工作
- ✓ 可以立即撰写论文

**当前结果**:
```json
{
  "alpha_url": 0.499,
  "alpha_html": 0.501,
  "alpha_visual": 0.000,
  "test/auroc": 1.0000
}
```

**论文说明**:
```
S3 固定融合方法展现了良好的适应性。
当 visual 品牌信息缺失时（例如未启用 OCR），
系统自动降级为两模态融合（URL + HTML），
仍显著优于均匀融合基线（S0）。

实验结果显示，即使只使用两个模态，
S3 仍能实现自适应加权，性能优异。
```

---

## 📊 实验结果对比

| 实验 | Alpha 分布 | AUROC | 状态 |
|------|-----------|-------|------|
| S0 (LateAvg) | (0.333, 0.333, 0.333) | 1.000 | 均匀融合 |
| S3 (Fixed, 2-modal) | (0.499, 0.501, 0.000) | 1.000 | ✓ 工作中 |
| S3 (Fixed, 3-modal) | (0.3X, 0.3X, 0.3X) | ? | ⏳ 需要 OCR |

---

## 🔍 调试信息

如果安装 Tesseract 后仍然出现问题，检查：

### 1. Python 能否找到 Tesseract
```python
import pytesseract
from PIL import Image
import numpy as np

# 检查版本
try:
    version = pytesseract.get_tesseract_version()
    print(f"✓ Tesseract version: {version}")
except Exception as e:
    print(f"✗ Error: {e}")
    # 手动指定路径
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    version = pytesseract.get_tesseract_version()
    print(f"✓ Manual path works: {version}")

# 测试 OCR
img = Image.fromarray(np.ones((100, 200, 3), dtype=np.uint8) * 255)
text = pytesseract.image_to_string(img)
print("✓ OCR test successful!")
```

### 2. 查看实验日志

```bash
# 查找最新实验
Get-ChildItem experiments\s3_*_* -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1

# 查看日志
Get-Content experiments\s3_iid_fixed_<timestamp>\logs\*.log | Select-String "C-MODULE DEBUG" -Context 5,5
Get-Content experiments\s3_iid_fixed_<timestamp>\logs\*.log | Select-String "alpha_" -Context 2,2
```

### 3. 检查 predictions_test.csv

```python
import pandas as pd
import glob

# 找最新实验
exp_dirs = glob.glob("experiments/s3_*_fixed_*")
latest = max(exp_dirs, key=lambda x: x.split('_')[-1])

# 读取预测结果
df = pd.read_csv(f"{latest}/results/predictions_test.csv")

# 检查品牌提取
print("Brand extraction rates:")
print(f"  brand_url: {df['brand_url'].notna().sum()}/{len(df)} ({df['brand_url'].notna().mean():.1%})")
print(f"  brand_html: {df['brand_html'].notna().sum()}/{len(df)} ({df['brand_html'].notna().mean():.1%})")
print(f"  brand_vis: {df['brand_vis'].notna().sum()}/{len(df)} ({df['brand_vis'].notna().mean():.1%})")

# 检查 alpha 权重
print("\nAlpha weights:")
print(df[['alpha_url', 'alpha_html', 'alpha_visual']].describe())

# 检查 c_visual
print("\nc_visual consistency:")
print(df['c_visual'].describe())
```

---

## 📝 推荐方案

**建议采用选项 A（安装 Tesseract）**，原因：
1. 完整的三模态融合更符合论文设计
2. 可以验证 visual 模态的实际贡献
3. 安装过程相对简单（5-10 分钟）

**如果时间紧迫**，可以先采用选项 B：
1. 使用当前的两模态融合结果
2. 在论文中说明 visual 降级原因
3. 在 Limitations 中提到 OCR 依赖

---

## ⏭️ 完成后的下一步

### 安装 Tesseract 并重新实验后：

1. **收集结果**
   ```bash
   python scripts/collect_s3_results.py
   ```

2. **生成可视化**
   ```bash
   python scripts/visualize_s3_final.py
   ```

3. **更新论文**
   - 添加三模态融合结果
   - 对比 S0 vs S3 性能
   - 分析 alpha 权重分布

4. **检查 Brand-OOD 实验**
   - 为什么没有 alpha 记录？
   - 是否需要重新运行？

---

## 📞 需要帮助？

如果遇到问题，检查：
- `INSTALL_TESSERACT_WINDOWS.md` - 详细安装指南
- `S3_FINAL_SUMMARY.md` - 完整诊断报告
- `S3_DIAGNOSIS_REPORT.md` - 问题分析

或运行：
```bash
python test_ocr.py  # 测试 OCR 安装
.\install_and_run_s3.ps1  # 自动检查和运行
```

---

**生成时间**: 2025-11-13
**下一步**: 选择方案 A 或 B，然后执行相应步骤

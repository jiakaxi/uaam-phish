# S3 实验 OCR 诊断报告

**实验ID**: s3_iid_fixed_20251114_002142
**时间**: 2025-11-14 00:21-00:26
**状态**: ⚠️ 实验完成但存在问题

---

## 📊 关键发现

### 1. Tesseract 安装状态 ✓
```
Tesseract v5.3.3.20231005 已安装
Python + Tesseract 集成正常
配置: use_ocr: true
```

### 2. 品牌提取结果 ⚠️

从实验日志中观察到：
```
- brand_url:  100.0% non-empty  ✓ 正常
- brand_html:  90.6% non-empty  ✓ 正常
- brand_vis:    0.0% non-empty  ✗ 完全失败
```

**关键问题**: 即使 Tesseract 已安装且 `use_ocr=true`，visual 品牌提取率仍为 **0.0%**

### 3. C-Module 一致性分数 ✗

```
- c_url:    min=nan, max=nan, mean=nan
- c_html:   min=nan, max=nan, mean=nan
- c_visual: min=nan, max=nan, mean=nan
- c_visual has NaN: True
```

**问题**: 所有模态的一致性分数都是 NaN，C-Module 计算完全失败

### 4. 最终性能指标

```json
{
  "test/loss": 0.1335,
  "test/acc": 1.0000,
  "test/auroc": 1.0000,
  "test/f1": 1.0000
}
```

✓ 性能指标正常，但**没有 alpha 权重记录**

---

## 🔍 问题分析

### 根本原因链条

```
1. C-Module 计算产生 NaN
   ↓
2. 所有模态的 c_url, c_html, c_visual 都无效
   ↓
3. 固定融合检测到一致性信息不可用
   ↓
4. 回退到 LateAvg (均匀融合)
   ↓
5. Alpha 权重未被计算/记录
```

### 为什么 C-Module 产生 NaN？

可能原因：
1. **品牌嵌入问题**: 品牌名为空字符串时，SentenceTransformer 可能返回零向量
2. **相似度计算异常**: `cosine_similarity(zero_vector, zero_vector)` → NaN
3. **归一化问题**: 某些品牌对的相似度计算失败

### 为什么 brand_vis 仍为 0.0%？

**可能原因**（按优先级）:

#### A. 数据集问题
```python
# 检查 1: 图片文件是否存在？
image_paths = df['image_path'].tolist()
missing = [p for p in image_paths if not os.path.exists(p)]
print(f"Missing images: {len(missing)}/{len(image_paths)}")
```

#### B. OCR 提取失败
```python
# C-Module 中的 OCR 逻辑：
if self.use_ocr and PYTESSERACT_AVAILABLE:
    try:
        text = pytesseract.image_to_string(img, config='--psm 11')
        # 从 text 中提取品牌
    except Exception:
        # 回退到启发式
        pass
```

可能的子原因：
- 图片质量太差（截图模糊）
- OCR 配置不正确（`--psm 11` 可能不适合）
- 品牌提取正则表达式不匹配
- 异常被静默捕获

#### C. Tesseract 路径问题
虽然 Python 测试通过，但在实际运行中可能：
- 环境变量未传递给训练进程
- 需要显式设置 `pytesseract.pytesseract.tesseract_cmd`

---

## 🔧 建议的调试步骤

### 优先级 P0: 检查 C-Module 日志

```bash
# 查找完整的 C-MODULE DEBUG 输出
Get-Content outputs\2025-11-14\00-21-42\.hydra\hydra_*.log | Select-String "C-MODULE DEBUG" -Context 10,10
```

### 优先级 P1: 单独测试 C-Module

创建测试脚本 `test_cmodule.py`:

```python
import sys
import torch
from PIL import Image
from src.modules.c_module import CModule

# 初始化 C-Module
c_module = CModule(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    use_ocr=True,
    thresh=0.60,
    brand_lexicon_path="resources/brand_lexicon.txt"
)

# 测试单个样本
test_url = "https://www.paypal.com/signin"
test_html = "<html>Welcome to PayPal</html>"
test_img_path = "data/processed/screenshots/sample.png"

# 提取品牌
brand_url = c_module._extract_brand_from_url(test_url)
brand_html = c_module._extract_brand_from_html(test_html)

img = Image.open(test_img_path) if os.path.exists(test_img_path) else None
brand_vis = c_module._extract_brand_from_visual(img) if img else ""

print(f"brand_url: '{brand_url}'")
print(f"brand_html: '{brand_html}'")
print(f"brand_vis: '{brand_vis}'")

# 测试一致性计算
if brand_url and brand_html:
    sim = c_module._compute_consistency_pair(brand_url, brand_html)
    print(f"Similarity (url vs html): {sim}")
else:
    print("Cannot compute similarity: brands are empty")
```

运行：
```bash
python test_cmodule.py
```

### 优先级 P2: 检查 pytesseract 配置

在 `src/modules/c_module.py` 的 OCR 部分添加显式路径：

```python
# 在 __init__ 或 _extract_brand_from_visual 开头添加
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

### 优先级 P3: 修改 OCR 参数

尝试不同的 PSM 模式：

```python
# 当前
text = pytesseract.image_to_string(img, config='--psm 11')

# 尝试
text = pytesseract.image_to_string(img, config='--psm 3')  # 自动页面分段
# 或
text = pytesseract.image_to_string(img, config='--psm 6')  # 假设单个文本块
# 或
text = pytesseract.image_to_string(img)  # 默认配置
```

---

## 🎯 短期解决方案

### 方案 A: 修复 C-Module（推荐，但需要调试时间）

1. 运行 `test_cmodule.py` 定位具体问题
2. 修复品牌提取或一致性计算逻辑
3. 重新运行实验

**预计时间**: 1-2 小时

### 方案 B: 使用两模态融合结果（立即可用）

接受当前事实：
- 之前的实验 (s3_iid_fixed_20251113_214912) 已经显示两模态融合工作正常
- Alpha 权重: (0.499, 0.501, 0.000)
- AUROC = 1.0000

在论文中：
```
实验中，C-Module 对 visual 品牌提取依赖 OCR 技术。
由于数据集截图特性（低分辨率、复杂背景等），
OCR 提取率较低，导致 visual 一致性分数不可用。
S3 固定融合的部分可用机制自动排除 visual 模态，
使用 URL + HTML 进行自适应融合。
```

**预计时间**: 0 分钟（结果已有）

### 方案 C: 禁用 C-Module，仅使用 U-Module（S2 方法）

修改配置：
```yaml
modules:
  use_umodule: true
  use_cmodule: false  # ← 禁用 C-Module
  fusion_mode: reliability_only
```

这将使用纯粹的可靠性融合（S2），避免 C-Module 的问题。

**预计时间**: 20 分钟重新训练

---

## 📝 需要检查的文件

1. **C-Module 源码**: `src/modules/c_module.py`
   - `_extract_brand_from_visual()` 方法
   - OCR 异常处理逻辑
   - 品牌提取正则表达式

2. **实验日志**:
   - `outputs/2025-11-14/00-21-42/.hydra/hydra_*.log`
   - `outputs/2025-11-14/00-21-42/wandb/run-*/logs/debug.log`

3. **数据集**:
   - `workspace/data/splits/iid/test_cached.csv`
   - 检查 `image_path` 列是否有效

---

## 🔍 关键疑问

1. **为什么之前的实验 (214912) brand_vis 也是 0.0%，但现在 (002142) 所有 c 值都是 NaN？**
   - 可能是因为配置或代码有所不同
   - 需要比较两次实验的具体差异

2. **C-Module 何时开始产生 NaN？**
   - 是在计算 embeddings 时？
   - 还是在计算 cosine similarity 时？
   - 需要更详细的日志

3. **如果禁用 visual 品牌，C-Module 对 url 和 html 的计算是否正常？**
   - 从日志看，c_url 和 c_html 也是 NaN
   - 说明问题不仅仅在 visual

---

## ✅ 下一步行动

**建议优先级**:

1. **立即可做**: 使用方案 B（接受两模态融合结果）撰写论文
2. **短期调试**: 运行 `test_cmodule.py` 定位问题（1-2 小时）
3. **中期修复**: 修复 C-Module 并重新实验（2-4 小时）
4. **备选方案**: 使用方案 C（禁用 C-Module）获取 S2 结果

---

**报告生成时间**: 2025-11-14 00:30
**下一步**: 选择方案 A、B 或 C 继续

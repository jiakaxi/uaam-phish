#!/usr/bin/env python3
"""
C-Module 诊断测试脚本
测试品牌提取和一致性计算功能
"""
import os
import sys
import torch
import pytesseract
from PIL import Image
import pandas as pd

print("=" * 70)
print("C-Module 诊断测试")
print("=" * 70)
print()

# 测试 1: 导入检查
print("[1/6] 检查导入...")
try:
    from src.modules.c_module import CModule

    print("  [OK] CModule import success")
except Exception as e:
    print(f"  [FAIL] CModule import failed: {e}")
    sys.exit(1)

# 测试 2: pytesseract 配置
print("\n[2/6] 检查 pytesseract 配置...")
try:
    # 显式设置 Tesseract 路径
    tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(tesseract_path):
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
        print(f"  [OK] Tesseract path set: {tesseract_path}")
    else:
        print(f"  [WARN] Tesseract path not found: {tesseract_path}")

    version = pytesseract.get_tesseract_version()
    print(f"  [OK] Tesseract version: {version}")
except Exception as e:
    print(f"  [FAIL] pytesseract test failed: {e}")

# 测试 3: 初始化 C-Module
print("\n[3/6] 初始化 C-Module...")
try:
    c_module = CModule(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        use_ocr=True,
        thresh=0.60,
        brand_lexicon_path="resources/brand_lexicon.txt",
    )
    print("  [OK] C-Module initialized")
    print(f"  - use_ocr: {c_module.use_ocr}")
    print(f"  - threshold: {c_module.thresh}")
except Exception as e:
    print(f"  [FAIL] C-Module init failed: {e}")
    sys.exit(1)

# 测试 4: 品牌提取（文本）
print("\n[4/6] 测试品牌提取（URL 和 HTML）...")
test_cases = [
    ("https://www.paypal.com/signin", "PayPal"),
    ("https://login.facebook.com/", "Facebook"),
    ("https://www.google.com/accounts", "Google"),
]

for test_url, expected_brand in test_cases:
    brand = c_module._extract_brand_from_url(test_url)
    status = "[OK]" if brand else "[FAIL]"
    print(f"  {status} URL: {test_url[:40]:40s} -> '{brand}'")

test_html = "<html><head><title>PayPal Login</title></head></html>"
brand_html = c_module._extract_brand_from_html(test_html)
print(
    f"  {'[OK]' if brand_html else '[FAIL]'} HTML: {test_html[:40]:40s} -> '{brand_html}'"
)

# 测试 5: 品牌提取（Visual - OCR）
print("\n[5/6] 测试品牌提取（Visual - OCR）...")

# 查找测试图片
test_data_csv = "workspace/data/splits/iid/test_cached.csv"
if os.path.exists(test_data_csv):
    df = pd.read_csv(test_data_csv)
    if "image_path" in df.columns:
        # 取前3个样本测试
        sample_paths = df["image_path"].head(3).tolist()
        print(f"  测试 {len(sample_paths)} 个图片样本:")

        for i, img_path in enumerate(sample_paths):
            full_path = img_path if os.path.isabs(img_path) else img_path
            if os.path.exists(full_path):
                try:
                    img = Image.open(full_path)
                    brand_vis = c_module._extract_brand_from_visual(img)
                    status = "[OK]" if brand_vis else "[FAIL]"
                    print(
                        f"    {status} [{i+1}] {os.path.basename(full_path):30s} -> '{brand_vis}'"
                    )
                except Exception as e:
                    print(f"    [FAIL] [{i+1}] OCR failed: {e}")
            else:
                print(f"    [FAIL] [{i+1}] Image not found: {full_path}")
    else:
        print("  [WARN] CSV has no image_path column")
else:
    print(f"  [WARN] Test data file not found: {test_data_csv}")
    print("  Trying manual OCR test...")

    # Create simple test image
    from PIL import ImageDraw

    test_img = Image.new("RGB", (400, 100), color="white")
    draw = ImageDraw.Draw(test_img)
    try:
        draw.text((10, 30), "PayPal", fill="black")
    except Exception:
        draw.text((10, 30), "PayPal", fill="black")

    brand_vis = c_module._extract_brand_from_visual(test_img)
    print(f"  {'[OK]' if brand_vis else '[FAIL]'} Test image -> '{brand_vis}'")

# 测试 6: 一致性计算
print("\n[6/6] 测试一致性计算...")
test_pairs = [
    ("paypal", "paypal", "应该高相似度"),
    ("paypal", "facebook", "应该低相似度"),
    ("", "paypal", "空字符串"),
    ("", "", "双空字符串"),
]

for brand1, brand2, desc in test_pairs:
    try:
        sim = c_module._compute_consistency_pair(brand1, brand2)
        status = "[OK]" if not torch.isnan(torch.tensor(sim)) else "[NAN]"
        print(f"  {status} '{brand1:10s}' vs '{brand2:10s}': {sim:6.3f} ({desc})")
    except Exception as e:
        print(f"  [ERROR] '{brand1:10s}' vs '{brand2:10s}': {e}")

# Summary
print()
print("=" * 70)
print("[SUCCESS] Diagnostic test completed!")
print("=" * 70)
print()
print("If Visual brand extraction rate is 0, possible reasons:")
print("  1. Image paths are incorrect")
print("  2. OCR extraction failed (image quality, config issues)")
print("  3. Brand lexicon incomplete")
print()
print("If consistency computation produces NaN, possible reasons:")
print("  1. Empty brand string -> zero embedding vector")
print("  2. cosine_similarity(zero, zero) -> NaN")
print("  3. Need to add empty brand check in C-Module")
print()

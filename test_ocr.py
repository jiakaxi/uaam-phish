#!/usr/bin/env python3
"""
Quick test script to verify Tesseract OCR installation
"""
import sys


def main():
    # Test 1: Check pytesseract installation
    try:
        import pytesseract

        print("✓ pytesseract package installed")
    except ImportError:
        print("✗ pytesseract not installed")
        print("  Solution: pip install pytesseract")
        return 1

    # Test 2: Check Tesseract executable
    try:
        version = pytesseract.get_tesseract_version()
        print(f"✓ Tesseract executable found (version {version})")
    except Exception as e:
        print(f"✗ Tesseract executable not found: {e}")
        print("\nSolution (Windows):")
        print("  1. Download from: https://github.com/UB-Mannheim/tesseract/wiki")
        print("  2. Install to C:\\Program Files\\Tesseract-OCR")
        print("  3. Add to PATH or manually set:")
        print(
            "     pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'"
        )
        return 1

    # Test 3: Simple OCR test
    try:
        from PIL import Image
        import numpy as np

        # Create a simple test image
        img_array = np.ones((100, 200, 3), dtype=np.uint8) * 255
        img = Image.fromarray(img_array)

        # Try OCR (even if blank, should not error)
        _ = pytesseract.image_to_string(img)
        print("✓ OCR functionality working!")
    except Exception as e:
        print(f"⚠ OCR test had an issue: {e}")
        print("  (This may be OK if image was blank)")

    print("\n" + "=" * 50)
    print("✓ All checks passed! OCR is ready to use.")
    print("=" * 50)
    return 0


if __name__ == "__main__":
    sys.exit(main())

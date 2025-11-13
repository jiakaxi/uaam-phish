#!/usr/bin/env python
"""检查GPU状态和可见性"""

import os
import sys

print("=" * 60)
print("GPU状态检查")
print("=" * 60)

# 检查环境变量
print("\n环境变量:")
cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "未设置")
nvidia_visible = os.environ.get("NVIDIA_VISIBLE_DEVICES", "未设置")
print(f"  CUDA_VISIBLE_DEVICES: {cuda_visible}")
print(f"  NVIDIA_VISIBLE_DEVICES: {nvidia_visible}")

# 检查PyTorch
try:
    import torch

    print("\nPyTorch信息:")
    print(f"  版本: {torch.__version__}")
    print(f"  CUDA可用: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"  CUDA版本: {torch.version.cuda}")
        print(f"  cuDNN版本: {torch.backends.cudnn.version()}")
        print(f"  GPU数量: {torch.cuda.device_count()}")

        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"\n  GPU {i}: {props.name}")
            print(f"    总内存: {props.total_memory / 1024**3:.2f} GB")
            print(f"    计算能力: {props.major}.{props.minor}")

            # 检查内存使用
            torch.cuda.set_device(i)
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            free = (props.total_memory - torch.cuda.memory_reserved(i)) / 1024**3

            print(f"    已分配内存: {allocated:.2f} GB")
            print(f"    已保留内存: {reserved:.2f} GB")
            print(f"    空闲内存: {free:.2f} GB")
            print(
                f"    内存使用率: {reserved / (props.total_memory / 1024**3) * 100:.1f}%"
            )

            # 检查是否有PyTorch tensor在GPU上
            if allocated > 0:
                print(f"    [WARNING] GPU {i} 有已分配的内存，可能正在使用中")
            else:
                print(f"    [OK] GPU {i} 内存空闲")
    else:
        print("  [ERROR] CUDA不可用")
        print("  可能的原因:")
        print("    1. PyTorch未安装CUDA版本")
        print("    2. CUDA驱动未正确安装")
        print("    3. CUDA_VISIBLE_DEVICES环境变量限制了GPU可见性")

except ImportError:
    print("\n[ERROR] PyTorch未安装")
    sys.exit(1)
except Exception as e:
    print(f"\n[ERROR] 检查GPU时出错: {e}")
    import traceback

    traceback.print_exc()

# 检查是否有训练进程
print("\n" + "=" * 60)
print("GPU使用建议:")
print("=" * 60)

if torch.cuda.is_available():
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    reserved = torch.cuda.memory_reserved(0) / 1024**3
    free = total_memory - reserved

    if free > total_memory * 0.5:
        print(f"[OK] GPU内存充足 ({free:.2f} GB 空闲 / {total_memory:.2f} GB 总计)")
        print("  可以开始训练")
    elif free > total_memory * 0.2:
        print(
            f"[WARNING] GPU内存部分使用 ({free:.2f} GB 空闲 / {total_memory:.2f} GB 总计)"
        )
        print("  可以开始训练，但可能需要减小batch size")
    else:
        print(f"[ERROR] GPU内存不足 ({free:.2f} GB 空闲 / {total_memory:.2f} GB 总计)")
        print("  建议:")
        print("    1. 关闭其他使用GPU的程序")
        print("    2. 减小batch size")
        print("    3. 使用CPU训练（不推荐）")
else:
    print("[ERROR] 无法使用GPU，建议检查CUDA安装")

print("\n" + "=" * 60)

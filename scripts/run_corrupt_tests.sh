#!/bin/bash
# 批量运行腐败数据测试脚本（主腐败评测）
# 运行完整的 L/M/H × 3 模态 = 9 个测试
# 使用方法: bash scripts/run_corrupt_tests.sh <experiment_dir>

set -e

EXPERIMENT_DIR=${1:-"experiments/s0_iid_earlyconcat_20251111_025612"}

if [ ! -d "$EXPERIMENT_DIR" ]; then
    echo "错误: 实验目录不存在: $EXPERIMENT_DIR"
    exit 1
fi

CORRUPT_ROOT="workspace/data/corrupt"
MODALITIES=("url" "html" "img")
LEVELS=("L" "M" "H")
TEST_TYPE="corrupt"

echo "=========================================="
echo "腐败数据批量测试 - 主腐败评测"
echo "=========================================="
echo "实验目录: $EXPERIMENT_DIR"
echo "腐败数据根目录: $CORRUPT_ROOT"
echo "测试类型: $TEST_TYPE (L/M/H)"
echo ""
echo "测试计划: 3 模态 × 3 强度 = 9 个测试"
echo "  模态: URL, HTML, IMG"
echo "  强度: L, M, H"
echo "=========================================="

# 定义所有测试组合
TOTAL_TESTS=$((${#MODALITIES[@]} * ${#LEVELS[@]}))
CURRENT_TEST=0

# 运行所有测试组合
for modality in "${MODALITIES[@]}"; do
    echo ""
    echo ">> 运行 ${modality^^} 模态测试 (L/M/H)..."
    for level in "${LEVELS[@]}"; do
        CURRENT_TEST=$((CURRENT_TEST + 1))
        echo ""
        echo "  [${CURRENT_TEST}/${TOTAL_TESTS}] ${modality^^} - ${level}"
        CSV_PATH="$CORRUPT_ROOT/${modality}/test_corrupt_${modality}_${level}.csv"
        echo "  CSV: $CSV_PATH"

        python scripts/train_hydra.py \
            experiment=s0_iid_earlyconcat \
            trainer.max_epochs=0 \
            datamodule.test_csv="$CSV_PATH" \
            run.name="corrupt_${modality}_${level}" \
            --config-path configs \
            --config-name config

        if [ $? -eq 0 ]; then
            echo "  ✓ 完成"
        else
            echo "  ✗ 失败"
            exit 1
        fi
    done
done

echo ""
echo "=========================================="
echo "所有 ${TOTAL_TESTS} 个测试完成！"
echo "=========================================="
echo "测试覆盖："
echo "  - URL: L, M, H"
echo "  - HTML: L, M, H"
echo "  - IMG: L, M, H"
echo ""
echo "现在可以运行结果收集脚本："
echo "  python scripts/test_corrupt_data.py \\"
echo "    --experiment-dir $EXPERIMENT_DIR \\"
echo "    --test-type corrupt \\"
echo "    --modalities url html img \\"
echo "    --levels L M H"
echo "=========================================="

#!/bin/bash
#
# URL-Only 三协议一键运行脚本
# 依次运行 random, temporal, brand_ood 三个协议
#

set -e  # 遇到错误立即退出

echo "============================================================"
echo "URL-Only 三协议实验"
echo "============================================================"
echo ""

# 检查 master.csv
if [ ! -f "data/processed/master.csv" ]; then
    echo "⚠️  未找到 data/processed/master.csv"
    echo "   正在创建..."
    python scripts/create_master_csv.py
    echo ""
fi

# 运行三个协议
protocols=("random" "temporal" "brand_ood")

for protocol in "${protocols[@]}"; do
    echo "============================================================"
    echo "Running protocol: $protocol"
    echo "============================================================"
    python scripts/train_hydra.py \
        protocol=$protocol \
        use_build_splits=true \
        run.name="url_${protocol}_$(date +%Y%m%d_%H%M%S)"
    echo ""
    echo "✅ Protocol $protocol completed"
    echo ""
done

echo "============================================================"
echo "All protocols completed!"
echo "============================================================"
echo ""
echo "运行验证脚本:"
echo "  python tools/check_artifacts_url_only.py"

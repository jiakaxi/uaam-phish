#!/bin/bash
# S4 Hyperparameter Sweep Script
#
# Sweeps only gamma (temperature) parameter.
# Lambda_c is learned during training, NOT a hyperparameter.

set -e  # Exit on error

echo "=================================================="
echo "S4 RCAF Full - Gamma (Temperature) Sweep"
echo "=================================================="
echo ""
echo "Note: lambda_c is LEARNED (not swept)"
echo "Only sweeping gamma (temperature scaling)"
echo ""

# ===== Configuration =====
EXPERIMENT="s4_iid_rcaf"
GAMMA_VALUES="1.0,2.0,3.0,5.0"
SCRIPT="scripts/train_hydra.py"

# Check if train script exists
if [ ! -f "$SCRIPT" ]; then
    echo "Error: Training script not found: $SCRIPT"
    exit 1
fi

# ===== Run Sweep =====
echo "Starting gamma sweep with values: $GAMMA_VALUES"
echo ""

python "$SCRIPT" \
    experiment="$EXPERIMENT" \
    -m \
    system.fusion.temperature="$GAMMA_VALUES"

echo ""
echo "=================================================="
echo "Sweep completed!"
echo "=================================================="
echo ""
echo "Next steps:"
echo "1. Check wandb/tensorboard logs to find best gamma (based on val/auroc)"
echo "2. Run final experiments with best gamma on all protocols:"
echo "   - python $SCRIPT experiment=s4_iid_rcaf system.fusion.temperature=<best>"
echo "   - python $SCRIPT experiment=s4_brandood_rcaf system.fusion.temperature=<best>"
echo "   - python $SCRIPT experiment=s4_corruption_rcaf system.fusion.temperature=<best> corruption_level=clean"
echo "   - python $SCRIPT experiment=s4_corruption_rcaf system.fusion.temperature=<best> corruption_level=light"
echo "   - python $SCRIPT experiment=s4_corruption_rcaf system.fusion.temperature=<best> corruption_level=medium"
echo "   - python $SCRIPT experiment=s4_corruption_rcaf system.fusion.temperature=<best> corruption_level=heavy"
echo ""
echo "3. Run analysis scripts:"
echo "   - python scripts/analyze_s4_adaptivity.py"
echo "   - python scripts/plot_s4_suppression.py"
echo "   - python scripts/compare_s3_s4.py"

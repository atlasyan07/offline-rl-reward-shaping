#!/bin/bash
# Section 2: Complete workflow script
# Trains IQL baseline, IQL+DR3, evaluates both, and analyzes collapse

set -e  # Exit on error

# Configuration
DATASET="outputs/behavior_collection/dataset.npz"
METADATA="outputs/behavior_collection/metadata.json"
ENV_CONFIG="configs/behavior_collection.yaml"

# Check dataset exists
if [ ! -f "$DATASET" ]; then
    echo "Error: Dataset not found at $DATASET"
    echo "Please run Section 1 data collection first:"
    echo "  python scripts/generate_dataset.py --config configs/behavior_collection.yaml"
    exit 1
fi

echo "========================================"
echo "Section 2: IQL Baseline + DR3"
echo "========================================"

# Step 1: Train IQL Baseline
echo ""
echo "Step 1/5: Training IQL Baseline (no DR3)..."
echo "----------------------------------------"
python scripts/section2/train_iql.py \
    --config configs/section2/iql_baseline.yaml \
    --dataset "$DATASET" \
    --metadata "$METADATA" \
    --output-dir outputs/section2_iql_baseline

# Step 2: Train IQL + DR3
echo ""
echo "Step 2/5: Training IQL + DR3..."
echo "----------------------------------------"
python scripts/section2/train_iql.py \
    --config configs/section2/iql_dr3.yaml \
    --dataset "$DATASET" \
    --metadata "$METADATA" \
    --output-dir outputs/section2_iql_dr3 \
    --use-dr3

# Step 3: Evaluate IQL Baseline
echo ""
echo "Step 3/5: Evaluating IQL Baseline..."
echo "----------------------------------------"
python scripts/section2/eval_policy.py \
    --model outputs/section2_iql_baseline/final_model.pt \
    --env-config "$ENV_CONFIG" \
    --layouts train_easy train_medium train_hard \
    --num-episodes 100 \
    --seed-offset 10000 \
    --output outputs/section2_iql_baseline/eval_results.json \
    --video-dir outputs/section2_iql_baseline/videos

# Step 4: Evaluate IQL + DR3
echo ""
echo "Step 4/5: Evaluating IQL + DR3..."
echo "----------------------------------------"
python scripts/section2/eval_policy.py \
    --model outputs/section2_iql_dr3/final_model.pt \
    --env-config "$ENV_CONFIG" \
    --layouts train_easy train_medium train_hard \
    --num-episodes 100 \
    --seed-offset 10000 \
    --output outputs/section2_iql_dr3/eval_results.json \
    --video-dir outputs/section2_iql_dr3/videos

# Step 5: Analyze representation collapse
echo ""
echo "Step 5/5: Analyzing representation collapse..."
echo "----------------------------------------"
python scripts/section2/analyze_collapse.py \
    --dataset "$DATASET" \
    --models \
        outputs/section2_iql_baseline/final_model.pt \
        outputs/section2_iql_dr3/final_model.pt \
    --labels "IQL Baseline" "IQL + DR3" \
    --output-dir outputs/section2_analysis \
    --num-samples 10000

# Summary
echo ""
echo "========================================"
echo "Section 2 Complete!"
echo "========================================"
echo ""
echo "Results:"
echo "  - IQL Baseline:"
echo "      Model: outputs/section2_iql_baseline/final_model.pt"
echo "      Eval:  outputs/section2_iql_baseline/eval_results.json"
echo ""
echo "  - IQL + DR3:"
echo "      Model: outputs/section2_iql_dr3/final_model.pt"
echo "      Eval:  outputs/section2_iql_dr3/eval_results.json"
echo ""
echo "  - Collapse Analysis:"
echo "      JSON:  outputs/section2_analysis/collapse_analysis.json"
echo "      Plots: outputs/section2_analysis/*.png"
echo ""
echo "Next: Review results and proceed to Section 3 (Offline IRL)"

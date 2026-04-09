#!/bin/bash
# Full pipeline execution script
# 1. Run hybrid feature selection (CALB=Pearson/0.98, Others=Spearman/0.95, Seeds=10)
# 2. Run analysis report generation

echo "=== Starting Full Pipeline ==="
echo "Timestamp: $(date)"

# Step 1: Feature Engineering Pipeline
echo ">> Phase 1: Running Hybrid Robustness Pipeline..."
bash pipelines/run_hybrid_robustness_v2.sh

if [ $? -ne 0 ]; then
    echo "Error: Hybrid pipeline failed."
    exit 1
fi

# Step 2: Generate Analysis Reports
echo ">> Phase 2: Generating Analysis Reports..."

echo "1. Updating NaN Report..."
python pipelines/detect_nan.py

echo "2. Analyzing Robust Features (Global)..."
python pipelines/analyze_robust_features.py

echo "3. Analyzing Li-Battery Features..."
python pipelines/analyze_li_features.py

echo "=== Pipeline Completed Successfully ==="
echo "Timestamp: $(date)"

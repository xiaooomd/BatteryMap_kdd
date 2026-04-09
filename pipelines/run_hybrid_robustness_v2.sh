#!/bin/bash
# Hybrid mode execution script (v2)
# CALB: Pearson, Threshold 0.98
# Others: Spearman, Threshold 0.95

# 1. Run CALB (Pearson, 0.98)
echo "Running CALB with Pearson (Threshold: 0.98)..."
python run_feature_selection.py --dataset_id CALB --filter_mode 0 --filter_method p --filter_threshold 0.98 --selector_method shap --top_k 20 --n_seeds 10

# 2. Run other datasets (Spearman, 0.95)
DATASETS=("CALCE" "HNEI" "HUST" "ISU_ILCC" "MATR" "MICH_EXP" "MICH" "NA" "RWTH" "SNL" "Stanford" "Tongji" "XJTU" "ZNion")

echo "Running others with Spearman (Threshold: 0.95)..."
for id in "${DATASETS[@]}"; do
    echo "Processing $id..."
    python run_feature_selection.py --dataset_id "$id" --filter_mode 0 --filter_method s --filter_threshold 0.95 --selector_method shap --top_k 20 --n_seeds 10
done

# 3. Aggregation
echo "Aggregating results..."
python pipelines/aggregate_seed_robustness.py

echo "Hybrid pipeline finished."

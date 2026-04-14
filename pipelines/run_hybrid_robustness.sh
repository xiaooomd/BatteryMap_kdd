#!/bin/bash
# Hybrid mode execution script
# CALB: Pearson (keep Ambient_Temperature)
# Others: Spearman (preserve nonlinear features)

# 1. Run CALB (Pearson)
echo "Running CALB with Pearson..."
python run.py --dataset_id CALB --filter_mode 0 --filter_method p --selector_method shap --top_k 20 --n_seeds 10

# 2. Run other datasets (Spearman)
# Exclude CALB, and those handled automatically by subtasks (SNL_*, Tongji_*)
DATASETS=("CALCE" "HNEI" "HUST" "ISU_ILCC" "MATR" "MICH_EXP" "MICH" "NA" "RWTH" "SNL" "Stanford" "Tongji" "XJTU" "ZNion")

echo "Running others with Spearman..."
for id in "${DATASETS[@]}"; do
    echo "Processing $id..."
    python run.py --dataset_id "$id" --filter_mode 0 --filter_method s --selector_method shap --top_k 20 --n_seeds 10
done

# 3. Aggregate
echo "Aggregating results..."
python scripts/aggregate_seed_robustness.py

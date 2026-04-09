#!/bin/bash
# 混合模式执行脚本
# CALB: Pearson (保留 Ambient_Temperature)
# 其他: Spearman (保留非线性特征)

# 1. 运行 CALB (Pearson)
echo "Running CALB with Pearson..."
python run.py --dataset_id CALB --filter_mode 0 --filter_method p --selector_method shap --top_k 20 --n_seeds 10

# 2. 运行其他数据集 (Spearman)
# 排除 CALB, 以及子任务自动处理的 (SNL_*, Tongji_*)
DATASETS=("CALCE" "HNEI" "HUST" "ISU_ILCC" "MATR" "MICH_EXP" "MICH" "NA" "RWTH" "SNL" "Stanford" "Tongji" "XJTU" "ZNion")

echo "Running others with Spearman..."
for id in "${DATASETS[@]}"; do
    echo "Processing $id..."
    python run.py --dataset_id "$id" --filter_mode 0 --filter_method s --selector_method shap --top_k 20 --n_seeds 10
done

# 3. 聚合
echo "Aggregating results..."
python scripts/aggregate_seed_robustness.py

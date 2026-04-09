#!/bin/bash
# Sensitivity analysis script - DLinear
# Task: Feature number sensitivity (n_selected) & early cycle sensitivity (early_cycle_threshold)
# Datasets: CALB1, CALB2
# Parallel: Using GPUs 0-5

# --datasets NAion2024,ZN-coin2024 (Commented out to run ALL datasets)
python run.py sensitivity \
    --model DLinear \
    --gpus 0,1,2,3,4,5 \
    --output_csv hyperparam_results/DLinear/sensitivity_DLinear.csv


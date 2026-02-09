#!/bin/bash
# Feature selection experiment script - Transformer
# Task: Evaluate performance of li_selected_results and robust_selected_results feature sets
# Datasets: All (default)
# Parallel: Using GPUs 0-5

python run_feature_selection_experiments.py \
    --model Transformer \
    --gpus 0,1,2,3,4,5 \
    --output_dir hyperparam_results/Transformer/ \
    --feature_dirs "Group_All_Replaced,Group_No_Centroid,Group_No_ICHV,Group_No_UVP"

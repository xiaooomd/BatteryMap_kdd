#!/bin/bash
# DLinear multi-dataset parallel hyperparameter optimization script
# Purpose: optimize all recommended datasets in parallel and keep per-dataset best parameters

echo "========================================"
echo "DLinear Multi-dataset Parallel Hyperparameter Optimization"
echo "Optimizing all recommended datasets by default"
echo "========================================"

python run.py multi-dataset-opt \
    --method pso \
    --model DLinear \
    --feature_type extracted_features \
    --task_type early_prediction \
    --root_path dataset/selected_result \
    --n_selected 20 \
    --n_particles 20 \
    --n_iterations 30 \
    --train_epochs 100 \
    --patience 5 \
    --gpus 0 1 2 3 4 5 \
    --max_workers_per_gpu 4 \
    --output_dir ./hyperparam_results/multi_dataset/DLinear

echo ""
echo "All optimizations completed! Results summarized in: ./hyperparam_results/multi_dataset/DLinear"

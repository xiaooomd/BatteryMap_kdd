#!/bin/bash
# CNN multi-dataset parallel hyperparameter optimization test script (minimum parameter configuration)
# Purpose: Quickly test if the script runs normally

echo "========================================"
echo "CNN Multi-dataset Parallel Hyperparameter Optimization - Test Mode"
echo "Use minimum parameters to quickly verify script functionality"
echo "========================================"

python run.py multi-dataset-opt \
    --method pso \
    --model CNN \
    --feature_type extracted_features \
    --task_type early_prediction \
    --root_path dataset/selected_result \
    --n_selected 5 \
    --n_particles 2 \
    --n_iterations 2 \
    --train_epochs 5 \
    --patience 3 \
    --gpus 0 \
    --max_workers_per_gpu 1 \
    --output_dir ./hyperparam_results/test/CNN

echo ""
echo "Test completed! Results in: ./hyperparam_results/test/CNN"


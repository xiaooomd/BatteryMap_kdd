#!/bin/bash
# iTransformer multi-dataset parallel hyperparameter optimization script (Recommended)
# Purpose: Parallel optimization on all 17 datasets, obtaining independent optimal parameters for each dataset

echo "========================================"
echo "iTransformer Multi-dataset Parallel Hyperparameter Optimization"
echo "Optimizing all 17 recommended datasets by default"
echo "========================================"

python run.py multi-dataset-opt \
    --method pso \
    --model iTransformer \
    --feature_type extracted_features \
    --task_type early_prediction \
    --root_path dataset/selected_result \
    --n_selected 20 \
    --n_particles 15 \
    --n_iterations 20 \
    --train_epochs 100 \
    --patience 5 \
    --gpus 0 1 2 3 4 5 \
    --max_workers_per_gpu 4 \
    --output_dir ./hyperparam_results/multi_dataset/iTransformer

echo ""
echo "All optimizations completed! Results summarized in: ./hyperparam_results/multi_dataset/iTransformer"


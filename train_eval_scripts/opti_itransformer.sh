#!/bin/bash
# iTransformer single dataset hyperparameter optimization script
# Purpose: Quickly test optimization effect on a single dataset

# Default parameters
DATASET=${1:-"li_selected"}
GPU="5"  # Fixed use of GPU 5

echo "========================================"
echo "iTransformer Single Dataset Hyperparameter Optimization"
echo "Dataset: $DATASET"
echo "GPU: $GPU"
echo "========================================"

python run.py hyperopt \
    --method pso \
    --model iTransformer \
    --dataset $DATASET \
    --feature_type extracted_features \
    --task_type early_prediction \
    --root_path dataset/li_results \
    --n_selected -1 \
    --n_particles 20 \
    --n_iterations 10 \
    --train_epochs 100 \
    --patience 5 \
    --gpu $GPU \
    --output_dir ./hyperparam_results/single_dataset/iTransformer/$DATASET

echo ""
echo "Optimization completed! Results saved in: ./hyperparam_results/single_dataset/iTransformer/$DATASET"


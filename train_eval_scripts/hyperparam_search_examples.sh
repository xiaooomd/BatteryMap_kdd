#!/bin/bash

# Hyperparameter optimization example script
# Usage: bash hyperparam_search_examples.sh

echo "=========================================="
echo "Hyperparameter Optimization Example Script"
echo "=========================================="

# Activate conda environment
conda activate batterylife

# ==========================================
# Example 1: Grid Search - MLP model + HUST dataset
# ==========================================
echo ""
echo "Example 1: Grid Search - MLP model"
echo "------------------------------------------"

# Note: Grid search tries all parameter combinations, which can take a long time if the search space is large
# Suggest testing with a small search space first

python hyperparameter_optimization.py \
    --method grid \
    --model MLP \
    --dataset HUST \
    --feature_type curve \
    --train_epochs 10 \
    --batch_size 32 \
    --metric val_mae \
    --gpu 0 \
    --output_dir ./hyperparam_search_results


# ==========================================
# Example 2: Particle Swarm Optimization - CPMLP model + HUST dataset
# ==========================================
echo ""
echo "Example 2: Particle Swarm Optimization - CPMLP model"
echo "------------------------------------------"

python hyperparameter_optimization.py \
    --method pso \
    --model CPMLP \
    --dataset HUST \
    --feature_type curve \
    --n_particles 20 \
    --n_iterations 30 \
    --w 0.7 \
    --c1 1.5 \
    --c2 1.5 \
    --train_epochs 10 \
    --metric val_rmse \
    --gpu 0 \
    --output_dir ./hyperparam_search_results


# ==========================================
# Example 3: PSO optimization - Transformer model (Feature mode)
# ==========================================
echo ""
echo "Example 3: PSO optimization - Transformer (Feature mode)"
echo "------------------------------------------"

python hyperparameter_optimization.py \
    --method pso \
    --model Transformer \
    --dataset HUST \
    --feature_type extracted_features \
    --n_particles 15 \
    --n_iterations 40 \
    --train_epochs 15 \
    --metric val_mape \
    --gpu 0 \
    --output_dir ./hyperparam_search_results


# ==========================================
# Example 4: Grid Search - BiLSTM model (early_prediction task)
# ==========================================
echo ""
echo "Example 4: Grid Search - BiLSTM (Early prediction)"
echo "------------------------------------------"

python hyperparameter_optimization.py \
    --method grid \
    --model BiLSTM \
    --dataset HUST \
    --task_type early_prediction \
    --feature_type curve \
    --train_epochs 10 \
    --metric val_mae \
    --gpu 0 \
    --output_dir ./hyperparam_search_results


# ==========================================
# Example 5: PSO optimization - DLinear model (Fast test)
# ==========================================
echo ""
echo "Example 5: PSO Fast Test - DLinear model"
echo "------------------------------------------"

python hyperparameter_optimization.py \
    --method pso \
    --model DLinear \
    --dataset HUST \
    --feature_type curve \
    --n_particles 10 \
    --n_iterations 20 \
    --train_epochs 5 \
    --metric val_mae \
    --gpu 0 \
    --output_dir ./hyperparam_search_results


echo ""
echo "=========================================="
echo "All examples executed successfully!"
echo "Results saved in ./hyperparam_search_results/"
echo "=========================================="

# Multi-Dataset Hyperparameter Optimization Quick Start Guide

## Quick Selection Between Two Schemes

### Scheme A: Multi-Dataset Aggregated Optimization (Universal Parameters)
Find **one** set of hyperparameters that optimizes average performance across all datasets

```bash
# Requires applying patch first
python optimization/apply_multi_dataset_patch.py

# Then run
python run.py hyperopt \
    --method pso \
    --model MLP \
    --datasets HUST CALB CALCE \
    --aggregation mean \
    --gpu 0
```

**Output**: 1 set of parameters, applicable to all datasets

---

### Scheme C: Parallel Independent Optimization (Customized Parameters) - Recommended
Find optimal parameters **separately** for each dataset

```bash
# Run directly
python run.py multi-dataset-opt \
    --datasets HUST CALB CALCE MIT MATR SNL ISU_ILCC NA RWTH Stanford XJTU HNEI MICH MICH_EXP UL_PUR Tongji ZNion \
    --method pso \
    --model MLP \
    --n_particles 20 \
    --n_iterations 50 \
    --gpus 0 1 2 3 4 5 \
    --max_workers_per_gpu 4
```

**Output**: 17 sets of parameters, each dataset independently configured

---

## Quick Test (3 Datasets)

```bash
# Scheme C - Quick Test
python run.py multi-dataset-opt \
    --datasets HUST CALB CALCE \
    --method pso \
    --model MLP \
    --n_particles 10 \
    --n_iterations 20 \
    --train_epochs 5 \
    --gpus 0 1 \
    --max_workers_per_gpu 2
```

Estimated time: ~1-2 hours (depends on GPU performance)

---

## Viewing Results

### Scheme A Results
```
hyperparam_search_results/
└── pso_MLP_HUST_CALB_CALCE_20260119_220045/
    ├── all_trials.csv           # All trial records
    ├── best_params.json         # Best universal parameters
    └── search_config.json       # Search configuration
```

### Scheme C Results
```
hyperparam_search_results/
└── multi_dataset_optimization_20260119_220045/
    ├── summary.json             # Summary results
    ├── summary.csv              # Summary table
    ├── config.json              # Runtime configuration
    ├── HUST/
    │   ├── all_trials.csv
    │   └── best_params.json
    ├── CALB/
    │   ├── all_trials.csv
    │   └── best_params.json
    └── ... (other datasets)
```

---

## Common Parameter Adjustments

### Quick Test (Reduce Computation)
```bash
--n_particles 10        # Reduce from 20 to 10
--n_iterations 20       # Reduce from 50 to 20
--train_epochs 5        # Reduce from 10 to 5
```

### Deep Optimization (Increase Precision)
```bash
--n_particles 30        # Increase from 20 to 30
--n_iterations 100      # Increase from 50 to 100
--train_epochs 20       # Increase from 10 to 20
```

### GPU Configuration
```bash
# 6 GPUs, up to 4 tasks each = 24 parallel
--gpus 0 1 2 3 4 5
--max_workers_per_gpu 4

# When GPU memory is insufficient, reduce parallel tasks
--max_workers_per_gpu 2
```

---

## Recommended Workflow

### Step 1: Quick Exploration (Scheme C, few iterations)
```bash
python run.py multi-dataset-opt \
    --datasets HUST CALB CALCE \
    --method pso \
    --model MLP \
    --n_particles 10 \
    --n_iterations 20 \
    --train_epochs 5 \
    --gpus 0 1 2
```

### Step 2: Deep Optimization (Scheme C, full iteration)
```bash
python run.py multi-dataset-opt \
    --datasets HUST CALB CALCE \
    --method pso \
    --model MLP \
    --n_particles 30 \
    --n_iterations 100 \
    --train_epochs 20 \
    --gpus 0 1 2
```

### Step 3 (Optional): Test Universality (Scheme A)
```bash
# Apply patch first
python optimization/apply_multi_dataset_patch.py

# Run Scheme A
python run.py hyperopt \
    --method pso \
    --model MLP \
    --datasets HUST CALB CALCE \
    --aggregation mean \
    --n_particles 20 \
    --n_iterations 50 \
    --train_epochs 10 \
    --gpu 0
```

---

## Troubleshooting

### Problem 1: A dataset always fails
- Scheme C automatically skips failed datasets, not affecting others
- Check the independent log for that dataset
- Possible causes: missing data, too few samples, insufficient GPU memory

### Problem 2: Insufficient GPU memory
```bash
--max_workers_per_gpu 2  # Reduce parallel tasks
--batch_size 16          # Reduce batch size
```

### Problem 3: Process hangs
- In Scheme C, a single dataset hanging does not affect others
- Can manually kill that process, other datasets continue running

---

## Detailed Documentation

- **Complete Usage Documentation**: [docs/MULTI_DATASET_OPTIMIZATION.md](docs/MULTI_DATASET_OPTIMIZATION.md)
- **Single-Dataset Optimization**: [docs/HYPERPARAMETER_OPTIMIZATION.md](docs/HYPERPARAMETER_OPTIMIZATION.md)
- **Code Details**: See docstrings in various scripts

---

## Time Estimation

### Scheme C (17 datasets, 6 GPUs)
```
Single dataset time = 20 particles × 50 iterations × 5 minutes/run = 83 hours
Parallel speedup = 6 GPUs × 4 tasks/GPU = 24 parallel
Actual time ≈ 83 hours (datasets don't finish exactly at the same time)

Suggestion: Test with few iterations first (takes ~10 hours)
```

### Scheme A (3 datasets)
```
Single trial = 3 datasets × 5 minutes = 15 minutes
Total time = 20 particles × 50 iterations × 15 minutes = 250 hours

Suggestion: Use grid search + small search space
```

---

## Core Concepts

- **Scheme A**: One parameter set → Universal for all datasets → Simplified deployment
- **Scheme C**: N parameter sets → Customized per dataset → Best performance
- **GPU Configuration**: You have 6 GPUs, each can run 4 tasks → Up to 24 parallel
- **Aggregation Methods**: mean (average), min (worst), weighted_mean (weighted)
- **Failure Tolerance**: One dataset failing does not affect other datasets

---

**Start optimizing!**

For questions, see: [docs/MULTI_DATASET_OPTIMIZATION.md](docs/MULTI_DATASET_OPTIMIZATION.md)

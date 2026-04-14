# Optimization Folder

This folder contains auxiliary files and documentation related to hyperparameter optimization.

## File Descriptions

### Auxiliary Tools
- **`apply_multi_dataset_patch.py`**: Scheme A patch tool
  - Adds multi-dataset aggregation support to `scripts/hyperparameter_optimization.py`
  - Automatically backs up original files
  - Usage: `python optimization/apply_multi_dataset_patch.py`

### Documentation
- **`QUICK_START_MULTI_DATASET.md`**: Quick start guide
  - Quick selection between two schemes and example commands
  - Common parameter adjustments and troubleshooting

- **`MULTI_DATASET_OPTIMIZATION_SUMMARY.md`**: Completion summary
  - Feature overview and key takeaways
  - Recommended workflow
  - Documentation navigation

## Main Entry Points

### Single-dataset or Multi-dataset Aggregation (Scheme A)
```bash
python run.py hyperopt --method pso --model MLP --dataset HUST
```

### Multi-dataset Parallel Independent Optimization (Scheme C) - Recommended
```bash
python run.py multi-dataset-opt \
    --datasets HUST CALB CALCE \
    --method pso --model MLP \
    --gpus 0 1 2
```

### Interactive Launch
```bash
python run.py hyperopt-batch
```

## Full Documentation

Detailed documentation is in the `docs/` folder:
- `docs/HYPERPARAMETER_OPTIMIZATION.md` - Single-dataset optimization detailed documentation
- `docs/MULTI_DATASET_OPTIMIZATION.md` - Multi-dataset optimization complete guide

## Related File Locations

- **Core Algorithm**: `utils/optimization.py`
  - GridSearchOptimizer
  - PSOOptimizer
  - RandomSearchOptimizer

- **Search Space Configuration**: `configs/hyperparam_search_config.py`
  - Predefined search spaces
  - Parameter ranges for all models

- **Example Scripts**: `train_eval_scripts/hyperparam_search_examples.sh`
  - Bash script examples

## Quick Start

1. **Test single-dataset optimization**
   ```bash
   python run.py hyperopt \
       --method pso --model MLP --dataset HUST \
       --n_particles 10 --n_iterations 20
   ```

2. **Test multi-dataset parallel optimization** - Recommended
   ```bash
   python run.py multi-dataset-opt \
       --datasets HUST CALB CALCE \
       --method pso --model MLP \
       --n_particles 10 --n_iterations 20 \
       --gpus 0 1 2
   ```

3. **View quick guide**
   ```bash
   cat optimization/QUICK_START_MULTI_DATASET.md
   ```

## Related Links

- [Quick Start Guide](QUICK_START_MULTI_DATASET.md)
- [Feature Summary](MULTI_DATASET_OPTIMIZATION_SUMMARY.md)
- [Full Documentation](../docs/MULTI_DATASET_OPTIMIZATION.md)

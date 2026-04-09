# BatteryMap: Battery Feature Engineering and Life Prediction

BatteryMap is a modular platform for battery lifecycle analysis, combining physical feature engineering and deep learning based life prediction.

## Current Capabilities

- Multi-dataset feature extraction (CALB, CALCE, HUST, MATR, SNL, Stanford, XJTU, ZN-coin, and more)
- Physics-informed feature computation (capacity, energy, efficiency, IC/DV, statistical descriptors)
- End-to-end feature engineering pipeline (cleaning, imputation, filter selection, wrapper selection)
- Deep learning training and evaluation for battery life prediction
- Hyperparameter optimization with Grid Search and PSO, including multi-dataset optimization scripts

## Project Structure

```text
BatteryMap_kdd/
├─ run.py                          # Unified command entry
├─ run_feature_extraction.py       # Feature extraction entry
├─ run_feature_selection.py        # Feature engineering entry (PPCF-related code and descriptions are here)
├─ run_main.py                     # Training and evaluation entry
├─ features/                       # Dataset-specific extraction scripts and base extractor
├─ modules/
│  ├─ data_processor/              # Cleaning, imputation, loading
│  ├─ feature_selector/            # Filter and wrapper selectors
│  └─ predictor/                   # Trainer, evaluator, model factory
├─ data_provider/                  # Data loading and split records
├─ models/                         # Model implementations
├─ scripts/                        # Optimization, evaluation, and batch scripts
├─ configs/                        # Best hyperparameters and search spaces
└─ docs/                           # Detailed documentation
```

## Quick Start

Run feature extraction:

```bash
python run_feature_extraction.py
```

Run feature engineering:

```bash
python run_feature_selection.py --dataset_id HUST
```

Run model training:

```bash
python run_main.py --model Autoformer --dataset HUST
```

Or use the unified entry:

```bash
python run.py extraction
python run.py selection --dataset_id HUST
python run.py predict --model Autoformer --dataset HUST
```

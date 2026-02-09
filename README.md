# BatteryMap: Full-Stack Battery Feature Engineering & Life Prediction Platform

**BatteryMap** is a modular, scalable battery data analysis platform that deeply integrates a **Physical Feature Extraction Pipeline** with a **Deep Learning Life Prediction Engine**.

## ✨ Key Features

*   **Multi-Source Data Support**: Standardized ETL pipelines for major battery datasets (CALB, HUST, MIT, etc.).
*   **Physics-Informed Feature Engineering**: Automated extraction of voltage, current, temperature, and differential features (IC/DV curves).
*   **Advanced Feature Selection**: Hybrid strategy combining correlation analysis (Pearson/Spearman) and model-based importance (SHAP/RFE).
*   **SOTA Prediction Models**: Integrated 18+ deep learning models including Autoformer, iTransformer, PatchTST, and DLinear.
*   **Automated Optimization**: Built-in hyperparameter tuning using Particle Swarm Optimization (PSO).
*   **Modular Architecture**: Decoupled design for data loading, feature processing, and model training.

## 📁 Project Structure

```text
BatteryMap/
├── features/               # [Feature Eng] Feature Extraction Scripts (ETL)
│   ├── features_CALB.py    # Example: CALB dataset extraction
│   └── ...
├── modules/                # [Core] Core Business Logic
│   ├── feature_selector/   # Feature Selectors (Filter/Wrapper)
│   ├── data_processor/     # Data Cleaning & Loading
│   ├── predictor/          # Deep Learning Prediction Module
│   └── ...
├── src/                    # [Utils] Low-level Utility Library
│   ├── utils/
│   │   ├── math_tools.py   # Math Utilities
│   │   └── feature_tools.py # Physical Feature Extraction Tools
│   └── ...
├── pipelines/              # [Orchestration] Orchestration Scripts
│   ├── run_all_and_report.sh    # Full Pipeline Entry
│   ├── run_hybrid_robustness.sh # Hybrid Selection Pipeline
│   └── ...
├── models/                 # [Deep Learning] Prediction Model Library
│   ├── Autoformer/
│   ├── PatchTST/
│   └── ...
├── run_feature_selection.py # [Entry] Feature Engineering Entry Point
├── run_main.py             # [Entry] Deep Learning Training Entry Point
└── requirements.txt        # Project Dependencies
```

## 🚀 Quick Start

### 1. Environment Setup
```bash
pip install -r requirements.txt
```

### 2. Feature Engineering

**Phase 1: Feature Extraction**
Convert raw battery data (`.pkl` ) into standardized feature CSV files.

1.  Modify `processed_data_dir` in `features/features_*.py` to point to your raw data path.
2.  Run the extraction script:
    ```bash
    python features/features_CALB.py
    ```

**Phase 2: Analysis & Selection**
Run selection and analysis using the unified entry point:
```bash
# Run default feature selection pipeline
python run_feature_selection.py
```

### 3. Life Prediction (RUL)

Train and evaluate using deep learning models.

**Data Preparation**:
Place processed datasets into the `dataset/` directory.

**Model Training**:
```bash
# Example: Train on HUST dataset using Autoformer
python run_main.py --model Autoformer --dataset HUST
```

**Hyperparameter Optimization (PSO/Grid Search)**:
```bash
python hyperparameter_optimization.py --method pso --model MLP --dataset HUST
```

## ⚙️ Core Modules

### Feature Engineering Module
*   **Direct Features**: Capacity, Energy, Coulombic Efficiency, Temperature, etc.
*   **IC/DV Features**: Incremental Capacity (IC) and Differential Voltage (DV) curves based on `src.utils.feature_tools`.
*   **Hybrid Selection Strategy**: Combines Pearson/Spearman correlation filtering with SHAP value importance ranking.

### Deep Learning Module
*   **Supported Models**: Transformer (Autoformer, iTransformer), MLP (DLinear), RNN (LSTM, GRU), CNN, and 18+ SOTA models.
*   **Capabilities**: Supports full supervised training, Fine-tuning, and Domain Adaptation.

---
*Powered by BatteryMap Team*

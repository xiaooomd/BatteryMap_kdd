import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, List, Dict
import pandas as pd
import numpy as np

# Add project root to sys.path
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

from modules.data_processor.loader import DataLoader
from modules.feature_selector.filter_methods import (
    PearsonFilter, SpearmanFilter, KendallFilter
)
from modules.feature_selector.wrapper_methods import (
    ShapSelector, RFESelector
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("RunFeatureSelection")


def get_filter_selector(method_code: str, threshold: float, mode: int = 0):
    """Factory to get the appropriate filter selector."""
    method_code = method_code.lower()
    if method_code == 'p':
        return PearsonFilter(mode=mode, threshold=threshold)
    elif method_code == 's':
        return SpearmanFilter(mode=mode, threshold=threshold)
    elif method_code == 'k':
        return KendallFilter(mode=mode, threshold=threshold)
    else:
        raise ValueError(f"Unknown filter method code: {method_code}")


def get_wrapper_selector(method_code: str, top_k: int, n_seeds: int):
    """Factory to get the appropriate wrapper selector."""
    method_code = method_code.lower()
    if method_code == 'shap':
        return ShapSelector(top_k=top_k, robust_mode=True, random_state=42)
    elif method_code == 'rfe':
        return RFESelector(top_k=top_k, robust_criterion=True, random_state=42)
    else:
        raise ValueError(f"Unknown wrapper method code: {method_code}")


def main():
    parser = argparse.ArgumentParser(description="Feature Selection Pipeline")

    # Dataset arguments
    parser.add_argument('--dataset_id', type=str, required=True, help='Dataset ID (e.g., CALB)')
    parser.add_argument('--input_dir', type=str, default='./results/features', help='Directory containing feature CSVs')
    parser.add_argument('--label_dir', type=str, default='./data/labels', help='Directory containing label JSONs')
    parser.add_argument('--output_dir', type=str, default='./results/selection', help='Directory to save results')

    # Filter method arguments
    parser.add_argument('--filter_mode', type=int, default=0, choices=[0, 1], help='0: Filter then Wrapper, 1: Wrapper only')
    parser.add_argument('--filter_method', type=str, default='p', choices=['p', 's', 'k'], help='p: Pearson, s: Spearman')
    parser.add_argument('--filter_threshold', type=float, default=0.95, help='Correlation threshold for filter method')

    # Wrapper method arguments (Selector)
    parser.add_argument('--selector_method', type=str, default='shap', choices=['shap', 'rfe'], help='Wrapper method')
    parser.add_argument('--top_k', type=int, default=20, help='Number of top features to select')
    parser.add_argument('--n_seeds', type=int, default=1, help='Number of random seeds for robustness check')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir) / args.dataset_id
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting feature selection for {args.dataset_id}")
    logger.info(f"Params: Filter={args.filter_method} (Th={args.filter_threshold}), Wrapper={args.selector_method}, Seeds={args.n_seeds}")

    # 1. Load Data
    loader = DataLoader(args.input_dir, args.label_dir)
    # We load all batteries for the given dataset_id
    # DataLoader yields (dataset_id, {battery_id: {'X': df, 'y': label}})
    # We need to construct a single aggregated X and y for feature selection

    # Manually fetch generator to filter specific dataset
    data_gen = loader.load_dataset_generator([args.dataset_id])

    dataset_data = None
    try:
        _, dataset_data = next(data_gen)
    except StopIteration:
        logger.error(f"No data found for dataset {args.dataset_id}")
        sys.exit(1)

    # Aggregating data from all batteries
    X_list = []
    y_list = []

    for battery_id, data in dataset_data.items():
        # Flatten time series features into a single row per battery (Mean aggregation as simple baseline)
        # OR take the last cycle? OR specific cycle?
        # Based on features_CALB.py, it produces one CSV per battery with multiple cycles.
        # Feature selection typically operates on "battery-level" or "cycle-level".
        # Assuming we want to predict life based on early cycles (e.g. cycle 100).

        df = data['X']
        label = data['y']

        # Strategy: Use the last available cycle in the 'early life' window (e.g. 100th cycle)
        # The DataLoader already limits to nrows=100 (early cycles).
        # We take the mean of the features across these early cycles to represent the "early life behavior".
        features_mean = df.mean(numeric_only=True).to_frame().T
        features_mean['battery_id'] = battery_id # Keep ID for reference

        X_list.append(features_mean)
        y_list.append(label)

    if not X_list:
        logger.error("No valid samples extracted.")
        sys.exit(1)

    X_agg = pd.concat(X_list, ignore_index=True)
    y_agg = pd.Series(y_list, name='Cycle_Life')

    # Drop non-feature columns
    if 'battery_id' in X_agg.columns:
        X_agg = X_agg.drop(columns=['battery_id'])

    # Remove constant columns
    X_agg = X_agg.loc[:, (X_agg != X_agg.iloc[0]).any()]

    logger.info(f"Data Loaded: {X_agg.shape[0]} samples, {X_agg.shape[1]} features.")

    # 2. Filter Method
    current_features = X_agg.columns.tolist()

    if args.filter_mode == 0:
        filter_selector = get_filter_selector(args.filter_method, args.filter_threshold, mode=0)
        selected_by_filter = filter_selector.select(X_agg, y_agg)

        # Save drop report
        filter_selector.save_report(str(output_dir), f"{args.filter_method}_drop_report.csv")

        X_filtered = X_agg[selected_by_filter]
        logger.info(f"Filter Step ({args.filter_method}): Reduced to {len(selected_by_filter)} features.")
    else:
        X_filtered = X_agg
        logger.info("Skipping Filter Step (Mode 1).")

    # 3. Wrapper Method (Robustness Check with Seeds)
    wrapper_selector = get_wrapper_selector(args.selector_method, args.top_k, args.n_seeds)

    # Run selection
    # Note: BaseSelector doesn't natively support n_seeds loop in select(),
    # but run_hybrid_robustness_v2.sh implies the script runs once.
    # The shell script loops for *datasets*, but passes --n_seeds to python.
    # If the Python script needs to aggregate over seeds, we should loop here.

    feature_counts = {}

    for seed in range(args.n_seeds):
        logger.info(f"Wrapper Round {seed+1}/{args.n_seeds}...")
        # Update random state
        wrapper_selector.random_state = 42 + seed

        try:
            selected_feats, _ = wrapper_selector.select(X_filtered, y_agg)
            for f in selected_feats:
                feature_counts[f] = feature_counts.get(f, 0) + 1
        except Exception as e:
            logger.error(f"Error in seed {seed}: {e}")

    # 4. Final Aggregation & Saving
    # Save frequency of each feature being selected
    final_df = pd.DataFrame(list(feature_counts.items()), columns=['feature', 'selection_count'])
    final_df = final_df.sort_values(by='selection_count', ascending=False)

    output_file = output_dir / f"robust_selection_{args.selector_method}.csv"
    final_df.to_csv(output_file, index=False)

    logger.info(f"Selection complete. Results saved to {output_file}")


if __name__ == "__main__":
    main()

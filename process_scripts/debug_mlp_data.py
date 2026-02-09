import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from data_provider.data_loader import Dataset_CustomFeatures
from models import MLP

def debug_mlp():
    print("=== Starting MLP data loading and model run debugging ===")

    # Mock command line arguments
    class Args:
        root_path = 'dataset/selected_result'
        n_selected = 20
        early_cycle_threshold = 100
        seq_len = 1
        seed = 2021
        enc_in = 20
        dec_in = 20
        c_out = 20
        d_model = 16
        d_ff = 32
        e_layers = 2
        dropout = 0.1
        charge_discharge_length = 100 # MLP.py needs this, though feature mode might not
        feature_type = 'extracted_features'
        task_type = 'early_prediction'
        dataset = 'HUST'
        weighted_loss = False
        batch_size = 2
        num_workers = 0

    args = Args()

    # 1. Check if directory exists
    if not os.path.exists(args.root_path):
        print(f"Error: Root directory not found: {args.root_path}")
        return

    # 2. Try to initialize Dataset
    print(f"Initializing Dataset_CustomFeatures (root={args.root_path})...")
    try:
        dataset = Dataset_CustomFeatures(args, flag='train')
    except Exception as e:
        print(f"Dataset initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return

    print(f"Dataset length: {len(dataset)}")

    if len(dataset) == 0:
        print("Error: Dataset is empty! CSV files might not be found or all files were filtered.")
        # Recursively list files to see
        print(f"Files in directory {args.root_path}:")
        for root, dirs, files in os.walk(args.root_path):
             for file in files:
                 if file.endswith('.csv'):
                     print(os.path.join(root, file))
        return

    # 3. Check Scaler
    print("Checking Scaler...")
    if dataset.feature_scaler:
        print(f"Feature Scaler Mean: {dataset.feature_scaler.mean_}")
        print(f"Feature Scaler Var: {dataset.feature_scaler.var_}")
        if np.any(dataset.feature_scaler.var_ == 0):
            print("Warning: Some features have a variance of 0!")
    else:
        print("Feature Scaler not initialized (None)")

    if dataset.label_scaler:
        print(f"Label Scaler Mean: {dataset.label_scaler.mean_}")
        print(f"Label Scaler Var: {dataset.label_scaler.var_}")
        if np.any(dataset.label_scaler.var_ == 0):
            print("Warning: Label variance is 0! There might be only one sample or all labels are the same.")
            # This would cause std=0 in run_main.py
    else:
        print("Label Scaler not initialized (None)")

    # 4. Try DataLoader
    print("Trying DataLoader...")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    try:
        batch = next(iter(loader))
        print("Successfully loaded one Batch")
        # Unpack batch (dict)
        cycle_curve_data = batch['cycle_curve_data']
        labels = batch['labels']
        curve_attn_mask = batch['curve_attn_mask']

        print(f"Input Shape: {cycle_curve_data.shape}")
        print(f"Label Shape: {labels.shape}")

        if torch.isnan(cycle_curve_data).any():
            print("Error: Input data contains NaN")
        if torch.isinf(cycle_curve_data).any():
            print("Error: Input data contains Inf")

    except Exception as e:
        print(f"DataLoader iteration failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # 5. Try model forward pass
    print("Initializing MLP model...")
    try:
        model = MLP.Model(args)
        print(f"Model Input Dim: {model.input_dim}")

        output = model(cycle_curve_data, curve_attn_mask)
        print(f"Model output Shape: {output.shape}")
        print(f"Model output: {output}")

        if torch.isnan(output).any():
            print("Error: Model output contains NaN")

        # Simulate Loss calculation
        criterion = torch.nn.MSELoss()

        # Squeeze logic from run_main.py
        if output.ndim == 2 and output.shape[1] == 1:
            output = output.squeeze()
        if labels.ndim == 2 and labels.shape[1] == 1:
            labels = labels.squeeze()

        loss = criterion(output, labels)
        print(f"Loss: {loss.item()}")

    except Exception as e:
        print(f"Model run failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    debug_mlp()

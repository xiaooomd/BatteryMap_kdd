#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Example of loading the best PSO model.

Demonstrates how to load and use the best model saved in the best_pso_model directory.
"""

import torch
import joblib
import json
from pathlib import Path
import argparse


def load_best_model(model_dir: str):
    """Load the best PSO model.

    Args:
        model_dir: Model directory path, e.g., 'best_pso_model/CALB42_MLP_20260131_143022'

    Returns:
        dict: A dictionary containing model weights, scalers, and configuration.
    """
    model_path = Path(model_dir)

    if not model_path.exists():
        raise FileNotFoundError(f"Model directory does not exist: {model_path}")

    print(f"\n{'='*80}")
    print(f"Loading best PSO model...")
    print(f"Model directory: {model_path}")
    print(f"{'='*80}\n")

    # 1. Load model weights
    checkpoint_path = model_path / 'checkpoint'
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print(f"✓ Model weights loaded")
    else:
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    # 2. Load label scaler
    label_scaler_path = model_path / 'label_scaler'
    if label_scaler_path.exists():
        label_scaler = joblib.load(label_scaler_path)
        print(f"✓ Label scaler loaded")
    else:
        print(f"⚠ Label scaler not found")
        label_scaler = None

    # 3. Load life cycle scaler
    life_class_scaler_path = model_path / 'life_class_scaler'
    if life_class_scaler_path.exists():
        life_class_scaler = joblib.load(life_class_scaler_path)
        print(f"✓ Life cycle scaler loaded")
    else:
        print(f"⚠ Life cycle scaler not found")
        life_class_scaler = None

    # 4. Load training arguments
    args_path = model_path / 'args.json'
    if args_path.exists():
        with open(args_path, 'r', encoding='utf-8') as f:
            training_args = json.load(f)
        print(f"✓ Training arguments loaded")
    else:
        print(f"⚠ Training arguments not found")
        training_args = None

    # 5. Load best model information
    best_info_path = model_path / 'best_model_info.json'
    if best_info_path.exists():
        with open(best_info_path, 'r', encoding='utf-8') as f:
            best_info = json.load(f)
        print(f"✓ Best model information loaded")

        print(f"\nBest model information:")
        print(f"  - Dataset: {best_info['dataset']}")
        print(f"  - Model: {best_info['model']}")
        print(f"  - Trial ID: {best_info['trial_id']}")
        print(f"  - Timestamp: {best_info['timestamp']}")
        print(f"\nBest hyperparameters:")
        for key, value in best_info['best_params'].items():
            print(f"  - {key}: {value}")
    else:
        print(f"⚠ Best model information not found")
        best_info = None

    print(f"\n{'='*80}")
    print(f"Model loading complete!")
    print(f"{'='*80}\n")

    return {
        'checkpoint': checkpoint,
        'label_scaler': label_scaler,
        'life_class_scaler': life_class_scaler,
        'training_args': training_args,
        'best_info': best_info
    }


def list_available_models(base_dir: str = 'best_pso_model'):
    """List all available best models.

    Args:
        base_dir: best_pso_model directory path
    """
    base_path = Path(base_dir)

    if not base_path.exists():
        print(f"Directory not found: {base_path}")
        return []

    model_dirs = sorted([d for d in base_path.iterdir() if d.is_dir()],
                       key=lambda x: x.name, reverse=True)

    if not model_dirs:
        print(f"No models found")
        return []

    print(f"\n{'='*80}")
    print(f"Available Best PSO Models (Total: {len(model_dirs)})")
    print(f"{'='*80}\n")

    models_info = []

    for i, model_dir in enumerate(model_dirs, 1):
        print(f"{i}. {model_dir.name}")

        # Try to read model info
        info_path = model_dir / 'best_model_info.json'
        if info_path.exists():
            with open(info_path, 'r', encoding='utf-8') as f:
                info = json.load(f)
            print(f"   Dataset: {info['dataset']}")
            print(f"   Model: {info['model']}")
            print(f"   Trial ID: {info['trial_id']}")
            print(f"   Timestamp: {info['timestamp']}")
            models_info.append(info)
        print()

    print(f"{'='*80}\n")

    return models_info


def demo_inference(model_dir: str):
    """Demonstrate how to use the loaded model for inference.

    Args:
        model_dir: Model directory path
    """
    # 1. Load model
    model_data = load_best_model(model_dir)

    # 2. Get model configuration
    training_args = model_data['training_args']
    best_info = model_data['best_info']

    if training_args is None:
        print("Warning: Missing training arguments, cannot demonstrate fully")
        return

    # 3. Reconstruct model structure (needs to be adapted based on actual model type)
    model_name = training_args.get('model', 'MLP')
    print(f"\nReconstructing model structure: {model_name}")

    print(f"Note: You need to import the corresponding model class and initialize it based on the model type")
    print(f"      e.g.: from models import {model_name}")

    # 4. Perform inference
    print(f"\nInference steps:")
    print(f"1. Prepare input data")
    print(f"2. Use the model for prediction")
    print(f"3. Use label_scaler to inverse transform the output")

    # Example code framework
    print(f"\nExample code:")
    print(f"""
    # Prepare input
    input_data = prepare_input(...)

    # Inference
    model.eval()
    with torch.no_grad():
        output = model(input_data)

    # Inverse transform
    if model_data['label_scaler'] is not None:
        output_real = model_data['label_scaler'].inverse_transform(output)

    print(f"Prediction result: {{output_real}}")
    """)


def main():
    parser = argparse.ArgumentParser(description='Load and use the best PSO model')
    parser.add_argument('--action', type=str, default='list',
                       choices=['list', 'load', 'demo'],
                       help='Action: list=list models, load=load model, demo=demonstrate inference')
    parser.add_argument('--model_dir', type=str, default=None,
                       help='Model directory path (for load and demo actions)')
    parser.add_argument('--base_dir', type=str, default='best_pso_model',
                       help='Base directory for best_pso_model')

    args = parser.parse_args()

    if args.action == 'list':
        list_available_models(args.base_dir)

    elif args.action == 'load':
        if args.model_dir is None:
            print("Error: Please specify the --model_dir parameter")
            return
        model_data = load_best_model(args.model_dir)
        print(f"Model data contains the following keys: {list(model_data.keys())}")

    elif args.action == 'demo':
        if args.model_dir is None:
            print("Error: Please specify the --model_dir parameter")
            return
        demo_inference(args.model_dir)


if __name__ == '__main__':
    main()

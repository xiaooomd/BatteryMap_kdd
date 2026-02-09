"""Hyperparameter optimization main script.

Supports two hyperparameter search methods: Grid Search and Particle Swarm Optimization.
Evaluates the performance of each set of hyperparameters by calling the training logic of run_main.py.

Supports two modes:
    - Single dataset optimization: Find optimal parameters for a single dataset
    - Multi-dataset aggregated optimization (Scheme A): Find a universal set of parameters that performs best on average across all datasets

Example:
    Single dataset:
        python hyperparameter_optimization.py --method grid --model MLP --dataset HUST

    Multi-dataset aggregation:
        python hyperparameter_optimization.py --method pso --model MLP --datasets HUST CALB CALCE --aggregation mean
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np
import pandas as pd

from utils.optimization import GridSearchOptimizer, PSOOptimizer
from configs.hyperparam_search_config import get_search_space


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Hyperparameter Optimization')

    # Optimization method
    parser.add_argument('--method', type=str, required=True, choices=['grid', 'pso'],
                        help='Optimization method: grid=Grid Search, pso=Particle Swarm Optimization')

    # Basic configuration
    parser.add_argument('--model', type=str, required=True,
                        help='Model name (e.g., MLP, CPMLP, Transformer)')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset name (e.g., HUST, CALB)')
    parser.add_argument('--task_type', type=str, default='forecasting',
                        choices=['forecasting', 'early_prediction'],
                        help='Task type')
    parser.add_argument('--feature_type', type=str, default='curve',
                        choices=['curve', 'extracted_features'],
                        help='Feature type')

    # PSO specific parameters
    parser.add_argument('--n_particles', type=int, default=20,
                        help='Number of PSO particles')
    parser.add_argument('--n_iterations', type=int, default=50,
                        help='Number of PSO iterations')
    parser.add_argument('--w', type=float, default=0.7,
                        help='PSO inertia weight')
    parser.add_argument('--c1', type=float, default=1.5,
                        help='PSO individual learning factor')
    parser.add_argument('--c2', type=float, default=1.5,
                        help='PSO social learning factor')

    # Training configuration
    parser.add_argument('--train_epochs', type=int, default=10,
                        help='Number of training epochs for each trial')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--seed', type=int, default=2021,
                        help='Random seed')

    # Output configuration
    parser.add_argument('--output_dir', type=str, default='./hyperparam_search_results',
                        help='Directory to save results')
    parser.add_argument('--metric', type=str, default='val_mae',
                        choices=['val_mae', 'val_rmse', 'val_mape'],
                        help='Optimization objective metric')

    # GPU configuration
    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU ID to use')

    # CSV feature mode configuration (only effective when --feature_type is extracted_features)
    parser.add_argument('--root_path', type=str, default='dataset/selected_result',
                        help='Root directory for CSV feature data (dataset/selected_result)')
    parser.add_argument('--n_selected', type=int, default=20,
                        help='Number of feature columns to select in extracted_features mode, -1 for all')

    return parser.parse_args()


def run_single_experiment(params: Dict[str, Any],
                         args: argparse.Namespace,
                         trial_id: int) -> Tuple[float, Dict[str, float]]:
    """Run a single experiment.

    Args:
        params: Hyperparameter dictionary
        args: Global configuration arguments
        trial_id: Trial number

    Returns:
        Optimization objective value and a dictionary of all evaluation metrics
    """
    # Build command line arguments
    cmd = [
        'python', 'run_main.py',
        '--model', args.model,
        '--dataset', args.dataset,
        '--task_type', args.task_type,
        '--feature_type', args.feature_type,
        '--root_path', args.root_path,
        '--n_selected', str(args.n_selected),
        '--train_epochs', str(args.train_epochs),
        '--patience', str(args.patience),
        '--batch_size', str(args.batch_size),
        '--seed', str(args.seed),
        '--model_comment', f'hypersearch_trial_{trial_id}',
        '--is_training', '1',
    ]

    # Add hyperparameters
    for key, value in params.items():
        if isinstance(value, bool):
            if value:
                cmd.append(f'--{key}')
        else:
            cmd.extend([f'--{key}', str(value)])

    # Set environment variables
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = args.gpu
    env['WANDB_MODE'] = 'disabled'  # Disable wandb to avoid login issues

    print(f"\n{'='*80}")
    print(f"Trial {trial_id}: Running experiment...")
    print(f"Parameters: {json.dumps(params, indent=2)}")
    print(f"{'='*80}\n")

    try:
        # Run training
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace'
        )

        # Parse output to get validation set performance
        metrics = parse_training_output(result.stdout)

        if metrics is None:
            print(f"Warning: Trial {trial_id} failed to parse performance metrics")
            print(f"Tail of output (last 500 characters):")
            print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
            print(f"Error output:")
            print(result.stderr[-500:] if len(result.stderr) > 500 else result.stderr)
            return float('inf'), {}

        # Return optimization objective
        target_metric = metrics.get(args.metric, float('inf'))
        print(f"Trial {trial_id} completed. {args.metric}: {target_metric:.4f}")

        return target_metric, metrics

    except Exception as e:
        print(f"Error: Trial {trial_id} execution failed: {str(e)}")
        return float('inf'), {}


def parse_training_output(output: str) -> Dict[str, float]:
    """Parse performance metrics from training output.

    Args:
        output: Standard output of the training script

    Returns:
        A dictionary containing various metrics, or None if parsing fails
    """
    metrics = {}

    # Scheme 1: Look for "Best model performance:" line (standard format)
    for line in output.split('\n'):
        if 'Best model performance:' in line and 'Test MAE:' in line:
            # Parse format: "Best model performance: Test MAE: 123.45 | Test RMSE: ..."
            parts = line.split('|')
            for part in parts:
                part = part.strip()
                if ':' in part:
                    # Extract metric name and value
                    metric_name = part.split(':')[0].strip().lower().replace(' ', '_')
                    try:
                        metric_value = float(part.split(':')[1].strip())
                        metrics[metric_name] = metric_value
                    except (ValueError, IndexError):
                        continue

    if metrics:
        return metrics

    # Scheme 2: Look for "Validation performance:" or "vali loss" line (fallback)
    for line in output.split('\n'):
        if 'vali loss' in line.lower() or 'val_mae' in line.lower():
            try:
                # Try to extract numbers
                import re
                numbers = re.findall(r'[-+]?\d*\.\d+|\d+', line)
                if numbers:
                    # Assume the first number is loss or MAE
                    metrics['val_mae'] = float(numbers[0])
                    return metrics
            except:
                continue

    return None


def save_results(results: List[Dict[str, Any]],
                args: argparse.Namespace,
                best_params: Dict[str, Any],
                best_score: float) -> None:
    """Save optimization results.

    Args:
        results: List of results from all trials
        args: Global configuration arguments
        best_params: Best hyperparameters
        best_score: Best performance score
    """
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir) / f'{args.method}_{args.model}_{args.dataset}_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save detailed results
    df = pd.DataFrame(results)
    df.to_csv(output_dir / 'all_trials.csv', index=False, encoding='utf-8')

    # Save best parameters
    best_result = {
        'method': args.method,
        'model': args.model,
        'dataset': args.dataset,
        'best_params': best_params,
        'best_score': float(best_score),
        'metric': args.metric,
        'timestamp': timestamp
    }

    with open(output_dir / 'best_params.json', 'w', encoding='utf-8') as f:
        json.dump(best_result, f, indent=2, ensure_ascii=False)

    # Save configuration
    config = vars(args)
    with open(output_dir / 'search_config.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*80}")
    print(f"Optimization complete!")
    print(f"Best parameters: {json.dumps(best_params, indent=2, ensure_ascii=False)}")
    print(f"Best {args.metric}: {best_score:.4f}")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*80}\n")

    return output_dir


def copy_best_checkpoint(best_trial_id: int, args: argparse.Namespace,
                        best_params: Dict[str, Any], output_dir: Path) -> None:
    """Copy the checkpoint of the best trial to the best_pso_model directory.

    Args:
        best_trial_id: ID of the best trial
        args: Global configuration arguments
        best_params: Best hyperparameters
        output_dir: Directory where optimization results are saved
    """
    # Build the path to the best trial's checkpoint
    # Path format: ./checkpoints/<setting>-hypersearch_trial_<trial_id>/
    setting = f'{args.model}_sl{args.seq_len}_lr{best_params.get("learning_rate", 0.0001)}_'
    setting += f'dm{best_params.get("d_model", 64)}_nh{best_params.get("n_heads", 4)}_'
    setting += f'el{best_params.get("e_layers", 2)}_dl{best_params.get("d_layers", 1)}_'
    setting += f'df{best_params.get("d_ff", 128)}_lradj{best_params.get("lradj", "constant")}_'
    setting += f'dataset{args.dataset}_loss{best_params.get("loss", "MSE")}_'
    setting += f'wd{best_params.get("weight_decay", 0.0)}_wl{best_params.get("weighted_loss", False)}_'
    setting += f'bs{args.batch_size}_s{args.seed}'

    source_checkpoint = Path('./checkpoints') / f'{setting}-hypersearch_trial_{best_trial_id}'

    # Create target directory: best_pso_model/<dataset>_<model>_<timestamp>/
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    target_dir = Path('./best_pso_model') / f'{args.dataset}_{args.model}_{timestamp}'
    target_dir.mkdir(parents=True, exist_ok=True)

    # Check if the source checkpoint exists
    if not source_checkpoint.exists():
        print(f"\nWarning: Best checkpoint directory not found: {source_checkpoint}")
        print(f"Possible reasons: checkpoint path mismatch or file has been deleted")
        print(f"\nAttempting to find checkpoint containing 'hypersearch_trial_{best_trial_id}'...")

        # Try to search for a matching folder in the checkpoints directory
        checkpoints_dir = Path('./checkpoints')
        if checkpoints_dir.exists():
            matching_dirs = list(checkpoints_dir.glob(f'*hypersearch_trial_{best_trial_id}'))
            if matching_dirs:
                source_checkpoint = matching_dirs[0]
                print(f"Found matching checkpoint: {source_checkpoint}")
            else:
                print(f"No checkpoint containing 'hypersearch_trial_{best_trial_id}' found")
                return
        else:
            print(f"Checkpoints directory does not exist")
            return

    try:
        # Copy the entire checkpoint folder
        print(f"\n{'='*80}")
        print(f"Copying best model checkpoint...")
        print(f"Source path: {source_checkpoint}")
        print(f"Target path: {target_dir}")

        # Copy checkpoint file
        if (source_checkpoint / 'checkpoint').exists():
            shutil.copy2(source_checkpoint / 'checkpoint', target_dir / 'checkpoint')
            print(f"✓ Copied model weights file")

        # Copy other relevant files
        for file_name in ['label_scaler', 'life_class_scaler', 'args.json']:
            src_file = source_checkpoint / file_name
            if src_file.exists():
                shutil.copy2(src_file, target_dir / file_name)
                print(f"✓ Copied {file_name}")

        # Save best parameters to the target directory
        best_info = {
            'trial_id': best_trial_id,
            'model': args.model,
            'dataset': args.dataset,
            'best_params': best_params,
            'source_checkpoint': str(source_checkpoint),
            'optimization_results': str(output_dir),
            'timestamp': timestamp
        }

        with open(target_dir / 'best_model_info.json', 'w', encoding='utf-8') as f:
            json.dump(best_info, f, indent=2, ensure_ascii=False)
        print(f"✓ Saved best model information")

        print(f"\nBest model successfully saved to: {target_dir}")
        print(f"{'='*80}\n")

    except Exception as e:
        print(f"\nError: Failed to copy checkpoint: {str(e)}")
        import traceback
        traceback.print_exc()


def main():
    """Main function."""
    args = parse_args()

    # Get search space
    search_space = get_search_space(args.model, args.feature_type)

    print(f"\n{'='*80}")
    print(f"Hyperparameter Optimization Configuration")
    print(f"Method: {args.method}")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Optimization Metric: {args.metric}")
    print(f"Search Space: {json.dumps(search_space, indent=2)}")
    print(f"{'='*80}\n")

    # Create optimizer
    if args.method == 'grid':
        optimizer = GridSearchOptimizer(search_space)
        print(f"Grid Search: Total {optimizer.get_total_combinations()} parameter combinations\n")
    else:  # PSO
        optimizer = PSOOptimizer(
            search_space=search_space,
            n_particles=args.n_particles,
            n_iterations=args.n_iterations,
            w=args.w,
            c1=args.c1,
            c2=args.c2
        )
        print(f"Particle Swarm Optimization: {args.n_particles} particles, {args.n_iterations} iterations\n")

    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    current_output_dir = Path(args.output_dir) / f'{args.method}_{args.model}_{args.dataset}_{timestamp}'
    current_output_dir.mkdir(parents=True, exist_ok=True)

    # Save initial configuration
    config = vars(args)
    with open(current_output_dir / 'search_config.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    # Define evaluation function
    def evaluate_fn(params: Dict[str, Any]) -> float:
        """Evaluation function wrapper."""
        trial_id = evaluate_fn.counter
        evaluate_fn.counter += 1

        # Run experiment
        score, metrics = run_single_experiment(params, args, trial_id)

        # Record result
        result = {'trial_id': trial_id, **params, 'score': score, **metrics}
        evaluate_fn.results.append(result)

        # Save/append results to CSV in real-time
        df_result = pd.DataFrame([result])
        csv_path = current_output_dir / 'trials_log.csv'
        # Write header if file doesn't exist, otherwise append without header
        df_result.to_csv(csv_path, mode='a', header=not csv_path.exists(), index=False, encoding='utf-8')

        return score

    evaluate_fn.counter = 0
    evaluate_fn.results = []

    # Run optimization
    best_params, best_score = optimizer.optimize(evaluate_fn)

    # Save final summary results (keep original logic as backup)
    output_dir = save_results(evaluate_fn.results, args, best_params, best_score)

    # Find the ID of the best trial
    best_trial_id = None
    best_trial_score = float('inf')
    for result in evaluate_fn.results:
        if result['score'] < best_trial_score:
            best_trial_score = result['score']
            best_trial_id = result['trial_id']

    # Copy the best checkpoint
    if best_trial_id is not None:
        copy_best_checkpoint(best_trial_id, args, best_params, output_dir)
    else:
        print("\nWarning: No valid best trial ID found, skipping checkpoint copy")


if __name__ == '__main__':
    main()

"""Run PSO hyperparameter optimization on li_results_ALL.

This script supports two modes:
    - curve: merged split over the raw curve dataset tree (default)
    - extracted_features: merged split over CSV feature files

The merged split semantics are fixed by data_split_recorder and are not
per-dataset aggregation.
"""

import subprocess
import sys
import argparse
from pathlib import Path


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='PSO optimization for li_results_ALL merged split')

    # Model configuration
    parser.add_argument('--model', type=str, default='Autoformer',
                        help='Model name (Autoformer, Transformer, MLP, CPMLP, etc.)')

    parser.add_argument('--feature_type', type=str, default='curve',
                        choices=['curve', 'extracted_features'],
                        help='Feature type to optimize on. Default: curve')
    parser.add_argument('--root_path', type=str, default=None,
                        help='Optional data root override. Defaults to ./dataset for curve and ./dataset/li_results_all for extracted_features')

    # PSO parameters
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
    parser.add_argument('--train_epochs', type=int, default=50,
                        help='Number of training epochs for each trial')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')
    parser.add_argument('--n_selected', type=int, default=-1,
                        help='Number of selected features, -1 for all features')

    # GPU configuration
    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU ID to use')

    # Output configuration
    parser.add_argument('--output_dir', type=str, default='./hyperparam_results_li_results_all',
                        help='Directory to save results')
    parser.add_argument('--metric', type=str, default='val_mae',
                        choices=['val_mae', 'val_rmse', 'val_mape'],
                        help='Optimization objective metric')

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    root_path = args.root_path
    if root_path is None:
        root_path = './dataset/li_results_all' if args.feature_type == 'extracted_features' else './dataset'

    # Ensure output directory exists
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Build PSO optimization configuration
    config = {
        'method': 'pso',
        'model': args.model,
        'dataset': 'li_results_ALL',  # Key: use li_results_ALL dataset
        'feature_type': args.feature_type,
        'root_path': root_path,
        'n_selected': args.n_selected,
        'n_particles': args.n_particles,
        'n_iterations': args.n_iterations,
        'w': args.w,
        'c1': args.c1,
        'c2': args.c2,
        'train_epochs': args.train_epochs,
        'batch_size': args.batch_size,
        'patience': args.patience,
        'metric': args.metric,
        'gpu': args.gpu,
        'output_dir': args.output_dir,
        'seed': 2021,
    }

    # Build command
    cmd = [sys.executable, str(Path(__file__).resolve().parent / 'hyperparameter_optimization.py')]

    for key, value in config.items():
        if isinstance(value, bool):
            if value:
                cmd.append(f'--{key}')
        else:
            cmd.extend([f'--{key}', str(value)])

    print(f"\n{'='*80}")
    print(f"PSO Hyperparameter Optimization for all li_results datasets")
    print(f"{'='*80}")
    print(f"Model: {config['model']}")
    print(f"Dataset: {config['dataset']} (all datasets merged)")
    print(f"Feature Type: {config['feature_type']}")
    print(f"Data Path: {config['root_path']}")
    print(f"PSO Config: {config['n_particles']} particles, {config['n_iterations']} iterations")
    print(f"Training Config: {config['train_epochs']} epochs, batch_size={config['batch_size']}")
    print(f"Number of features: {'All' if config['n_selected'] == -1 else config['n_selected']}")
    print(f"Optimization Metric: {config['metric']}")
    print(f"GPU: {config['gpu']}")
    print(f"Output Directory: {config['output_dir']}")
    print(f"{'='*80}\n")

    print("Starting PSO optimization...")
    print(f"Command: {' '.join(cmd)}\n")

    try:
        # Run optimization
        subprocess.run(cmd, check=True)
        print(f"\n{'='*80}")
        print(f"✓ Optimization completed!")
        print(f"Results saved in: {config['output_dir']}")
        print(f"{'='*80}\n")
    except subprocess.CalledProcessError as e:
        print(f"\n{'='*80}")
        print(f"✗ Optimization failed: {e}")
        print(f"{'='*80}\n")
        sys.exit(1)
    except KeyboardInterrupt:
        print(f"\n\nProgram interrupted by user")
        sys.exit(1)


if __name__ == '__main__':
    main()

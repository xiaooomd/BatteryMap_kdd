"""Multi-dataset parallel hyperparameter optimization script (Scheme C).

Optimizes hyperparameters for each dataset independently, leveraging multi-GPU for parallel acceleration.
Each dataset gets a customized set of optimal parameters.

Example:
    # Example 1: Optimize all recommended datasets (default behavior, no --datasets needed)
    python run_multi_dataset_optimization.py \
        --method pso \
        --model MLP \
        --n_particles 20 \
        --n_iterations 50 \
        --gpus 0 1 2 3 4 5 \
        --max_workers_per_gpu 4

    # Example 2: Optimize only specified datasets
    python run_multi_dataset_optimization.py \
        --datasets HUST CALB2024 CALCE MIT \
        --method pso \
        --model MLP \
        --n_particles 20 \
        --n_iterations 50 \
        --gpus 0 1 2 3 4 5 \
        --max_workers_per_gpu 4

    # Example 3: Use grid search to optimize some datasets
    python run_multi_dataset_optimization.py \
        --datasets HUST MIT MATR \
        --method grid \
        --model CPMLP \
        --gpus 0 1 \
        --max_workers_per_gpu 2
"""

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple
import time

import pandas as pd


# Special optimization task dataset list (only for CALB dataset testing)
SPECIAL_PSO_DATASETS = [
    'CALB1', 'CALB2',                  # Run only CALB1 and CALB2 (0°C and 25/35°C groups)
    # 'CALB42', 'CALB2024',              # [Replaced] Old CALB groupings
    # 'NAion42', 'NAion',                # [Temporarily Disabled] NAion
    # 'NAion2024',                       # [Temporarily Disabled] NAion 2024 version
    # 'ZN-coin2024',                     # [Temporarily Disabled] ZN-coin
]

# List of all recommended datasets (uses 2024 version by default)
ALL_DATASETS = [
    'HUST', 'MIT', 'MATR', 'SNL', 'RWTH',
    'MICH', 'MICH_EXP', 'UL_PUR',
    'CALCE', 'HNEI',
    'Tongji', 'Stanford', 'ISU_ILCC', 'XJTU',
    'NAion2024',    # Recommended 2024 version for NA-ion
    'CALB2024',     # Recommended 2024 version for CALB
    'ZN-coin2024',  # Recommended 2024 version for ZN-coin
]

# Dataset order for specific models (for custom GPU load balancing)
MODEL_SPECIFIC_DATASET_ORDER = {
    # MICN: Move GPU5 datasets to the front, others shift back (circular shift)
    'MICN': [
        'MICH', 'Stanford',  # Original GPU5 -> GPU0
        'HUST', 'MICH_EXP', 'ISU_ILCC',  # Original GPU0 -> GPU1
        'MIT', 'UL_PUR', 'XJTU',  # Original GPU1 -> GPU2
        'MATR', 'CALCE', 'NAion2024',  # Original GPU2 -> GPU3
        'SNL', 'HNEI', 'CALB2024',  # Original GPU3 -> GPU4
        'RWTH', 'Tongji', 'ZN-coin2024',  # Original GPU4 -> GPU5
    ],
    # iTransformer: Swap GPU4 and GPU5
    'iTransformer': [
        'HUST', 'MICH_EXP', 'ISU_ILCC',  # GPU0 unchanged
        'MIT', 'UL_PUR', 'XJTU',  # GPU1 unchanged
        'MATR', 'CALCE', 'NAion2024',  # GPU2 unchanged
        'SNL', 'HNEI', 'CALB2024',  # GPU3 unchanged
        'MICH', 'Stanford',  # Original GPU5 -> GPU4
        'RWTH', 'Tongji', 'ZN-coin2024',  # Original GPU4 -> GPU5
    ],
    # PatchTST: Swap GPU1 and GPU2, swap GPU4 and GPU5
    'PatchTST': [
        'HUST', 'MICH_EXP', 'ISU_ILCC',  # GPU0 unchanged
        'MATR', 'CALCE', 'NAion2024',  # Original GPU2 -> GPU1
        'MIT', 'UL_PUR', 'XJTU',  # Original GPU1 -> GPU2
        'SNL', 'HNEI', 'CALB2024',  # GPU3 unchanged
        'MICH', 'Stanford',  # Original GPU5 -> GPU4
        'RWTH', 'Tongji', 'ZN-coin2024',  # Original GPU4 -> GPU5
    ],
    # Transformer: Swap GPU1 and GPU2, swap GPU4 and GPU5 (same as PatchTST)
    'Transformer': [
        'HUST', 'MICH_EXP', 'ISU_ILCC',  # GPU0 unchanged
        'MATR', 'CALCE', 'NAion2024',  # Original GPU2 -> GPU1
        'MIT', 'UL_PUR', 'XJTU',  # Original GPU1 -> GPU2
        'SNL', 'HNEI', 'CALB2024',  # GPU3 unchanged
        'MICH', 'Stanford',  # Original GPU5 -> GPU4
        'RWTH', 'Tongji', 'ZN-coin2024',  # Original GPU4 -> GPU5
    ],
}


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Multi-dataset parallel hyperparameter optimization - Scheme C (independent optimization)')

    # Basic configuration
    parser.add_argument('--datasets', type=str, nargs='+', default=None,
                        help='List of datasets to optimize (optional). '
                             'If not specified, all recommended datasets are optimized by default. '
                             'Available datasets include: HUST, MIT, MATR, SNL, RWTH, MICH, MICH_EXP, UL_PUR, '
                             'CALCE, HNEI, Tongji, Stanford, ISU_ILCC, XJTU, '
                             'NAion2024, CALB2024, ZN-coin2024. '
                             'Note: Multi-version datasets default to the 2024 version (more reasonable split).')
    parser.add_argument('--method', type=str, required=True, choices=['grid', 'pso'],
                        help='Optimization method')
    parser.add_argument('--model', type=str, required=True,
                        help='Model name')
    parser.add_argument('--task_type', type=str, default='forecasting',
                        choices=['forecasting', 'early_prediction'])
    parser.add_argument('--feature_type', type=str, default='curve',
                        choices=['curve', 'extracted_features'])

    # PSO parameters
    parser.add_argument('--n_particles', type=int, default=20)
    parser.add_argument('--n_iterations', type=int, default=50)
    parser.add_argument('--w', type=float, default=0.7)
    parser.add_argument('--c1', type=float, default=1.5)
    parser.add_argument('--c2', type=float, default=1.5)

    # Data configuration
    parser.add_argument('--root_path', type=str, default='dataset/selected_result',
                        help='Data root directory')
    parser.add_argument('--n_selected', type=int, default=20,
                        help='Number of selected features')

    # Training configuration
    parser.add_argument('--train_epochs', type=int, default=10)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=2021)
    parser.add_argument('--metric', type=str, default='val_mae',
                        choices=['val_mae', 'val_rmse', 'val_mape'])

    # Parallel configuration
    parser.add_argument('--gpus', type=str, nargs='+', default=['0'],
                        help='List of available GPU IDs')
    parser.add_argument('--max_workers_per_gpu', type=int, default=4,
                        help='Maximum number of concurrent tasks per GPU')

    # Output configuration
    parser.add_argument('--output_dir', type=str, default='./hyperparam_search_results')

    return parser.parse_args()


def optimize_single_dataset(dataset: str,
                            gpu: str,
                            args: argparse.Namespace,
                            output_dir: Path) -> Tuple[str, bool, Dict[str, Any]]:
    """Execute optimization for a single dataset.

    Args:
        dataset: Dataset name
        gpu: GPU ID
        args: Global configuration
        output_dir: Output directory

    Returns:
        (Dataset name, whether successful, result dictionary)
    """
    start_time = time.time()
    print(f"\n[{dataset}] Starting optimization (GPU {gpu})...")

    # Build command
    cmd = [
        sys.executable,
        'hyperparameter_optimization.py',
        '--method', args.method,
        '--model', args.model,
        '--dataset', dataset,
        '--task_type', args.task_type,
        '--feature_type', args.feature_type,
        '--root_path', args.root_path,
        '--n_selected', str(args.n_selected),
        '--train_epochs', str(args.train_epochs),
        '--patience', str(args.patience),
        '--batch_size', str(args.batch_size),
        '--seed', str(args.seed),
        '--metric', args.metric,
        '--gpu', gpu,
        '--output_dir', str(output_dir / dataset)
    ]

    # PSO parameters
    if args.method == 'pso':
        cmd.extend([
            '--n_particles', str(args.n_particles),
            '--n_iterations', str(args.n_iterations),
            '--w', str(args.w),
            '--c1', str(args.c1),
            '--c2', str(args.c2)
        ])

    try:
        # Ensure output directory exists
        (output_dir / dataset).mkdir(parents=True, exist_ok=True)
        log_file_out = output_dir / dataset / "stdout.log"
        log_file_err = output_dir / dataset / "stderr.log"

        # Run optimization
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=None  # No timeout
        )

        # Write log files
        with open(log_file_out, 'w', encoding='utf-8') as f:
            f.write(result.stdout)
        with open(log_file_err, 'w', encoding='utf-8') as f:
            f.write(result.stderr)

        # Check for CRITICAL ERROR (Inf/NaN)
        if "[CRITICAL ERROR]" in result.stdout:
            print(f"\n⚠️  [{dataset}] Detected Inf/NaN error! Relevant log snippet:")
            # Extract Critical Error context
            lines = result.stdout.split('\n')
            for i, line in enumerate(lines):
                if "[CRITICAL ERROR]" in line:
                    # Print the line and the next 5 lines (containing Input/Output stats)
                    context = '\n'.join(lines[i:i+6])
                    print(f"--------------------------------------------------\n{context}\n--------------------------------------------------")


        # Check for result file
        best_params_file = None
        for item in (output_dir / dataset).rglob('best_params.json'):
            best_params_file = item
            break

        elapsed = time.time() - start_time

        if best_params_file and best_params_file.exists():
            with open(best_params_file, 'r', encoding='utf-8') as f:
                result_data = json.load(f)
            print(f"[{dataset}] Optimization complete ✓ - Best score: {result_data['best_score']:.4f} - Elapsed: {elapsed/60:.1f}min")
            result_data['elapsed_time'] = elapsed
            return dataset, True, result_data
        else:
            print(f"[{dataset}] Optimization failed ✗ - Result file not found - Elapsed: {elapsed/60:.1f}min")
            return dataset, False, {'elapsed_time': elapsed}

    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        print(f"[{dataset}] Optimization timed out ✗ - Elapsed: {elapsed/60:.1f}min")
        return dataset, False, {'elapsed_time': elapsed, 'error': 'timeout'}
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"[{dataset}] Optimization failed ✗ - {str(e)} - Elapsed: {elapsed/60:.1f}min")
        return dataset, False, {'elapsed_time': elapsed, 'error': str(e)}


def save_summary(results: Dict[str, Any], args: argparse.Namespace, output_dir: Path) -> None:
    """Save summary results.

    Args:
        results: Results for all datasets
        args: Global configuration
        output_dir: Output directory
    """
    # Statistics
    successful = [k for k, v in results.items() if v.get('success', False)]
    failed = [k for k in results.keys() if k not in successful]

    # Summary JSON
    summary = {
        'method': args.method,
        'model': args.model,
        'task_type': args.task_type,
        'feature_type': args.feature_type,
        'total_datasets': len(args.datasets),
        'successful_datasets': len(successful),
        'failed_datasets': failed,
        'best_params_per_dataset': {},
        'best_scores': {},
        'elapsed_times': {},
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S')
    }

    for dataset in successful:
        data = results[dataset].get('data', {})
        summary['best_params_per_dataset'][dataset] = data.get('best_params', {})
        summary['best_scores'][dataset] = data.get('best_score', None)
        summary['elapsed_times'][dataset] = data.get('elapsed_time', 0)

    with open(output_dir / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Summary CSV
    rows = []
    for dataset in successful:
        data = results[dataset].get('data', {})
        row = {
            'dataset': dataset,
            'best_score': data.get('best_score', None),
            'elapsed_time_minutes': data.get('elapsed_time', 0) / 60
        }
        row.update(data.get('best_params', {}))
        rows.append(row)

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(output_dir / 'summary.csv', index=False, encoding='utf-8')

    # Print summary report
    print(f"\n{'='*80}")
    print(f"All optimizations completed!")
    print(f"Success: {len(successful)}/{len(args.datasets)}")
    if successful:
        print(f"\nSuccessful datasets and their best scores:")
        for dataset in successful:
            score = summary['best_scores'][dataset]
            elapsed = summary['elapsed_times'][dataset] / 60
            print(f"  {dataset}: {score:.4f} ({elapsed:.1f}min)")
    if failed:
        print(f"\nFailed datasets: {failed}")
    print(f"\nResults saved to: {output_dir}")
    print(f"For detailed results, see: {output_dir}/summary.json")
    print(f"{'='*80}\n")


def main():
    """Main function."""
    args = parse_args()

    # If no datasets are specified, use all recommended datasets
    if args.datasets is None:
        # Check for model-specific dataset order
        if args.model in MODEL_SPECIFIC_DATASET_ORDER:
            args.datasets = MODEL_SPECIFIC_DATASET_ORDER[args.model].copy()
            print(f"\nNo datasets specified, using model-specific dataset order for {args.model} (total {len(args.datasets)})")
            print(f"Hint: This order is optimized for GPU load balancing")
        else:
            args.datasets = ALL_DATASETS.copy()
            print(f"\nNo datasets specified, optimizing all recommended datasets (total {len(args.datasets)})")

    # Version compatibility hint and auto-conversion
    version_mapping = {
        'NAion': 'NAion2024',
        'NAion42': 'NAion2024',
        'CALB': 'CALB2024',
        'CALB42': 'CALB2024',
        'ZN-coin': 'ZN-coin2024',
        'ZN-coin42': 'ZN-coin2024'
    }
    for i, dataset in enumerate(args.datasets):
        if dataset in version_mapping:
            recommended = version_mapping[dataset]
            print(f"⚠️  Hint: {dataset} -> {recommended} (automatically using 2024 version)")
            args.datasets[i] = recommended

    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir) / f'multi_dataset_optimization_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"Multi-dataset Parallel Hyperparameter Optimization (Scheme C - Independent Optimization)")
    print(f"Method: {args.method}")
    print(f"Model: {args.model}")
    print(f"Number of datasets: {len(args.datasets)}")
    print(f"Datasets: {args.datasets}")
    print(f"Number of GPUs: {len(args.gpus)}")
    print(f"Max parallel tasks per GPU: {args.max_workers_per_gpu}")
    print(f"Estimated max parallelism: {len(args.gpus) * args.max_workers_per_gpu}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*80}\n")

    # Create GPU task queue
    gpu_queue = []
    for i, dataset in enumerate(args.datasets):
        gpu_id = args.gpus[i % len(args.gpus)]
        gpu_queue.append((dataset, gpu_id))

    print("GPU Task Allocation:")
    for gpu in args.gpus:
        datasets_on_gpu = [d for d, g in gpu_queue if g == gpu]
        print(f"  GPU {gpu}: {datasets_on_gpu}")
    print()

    # Save configuration
    config = vars(args)
    with open(output_dir / 'config.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    # Parallel execution
    max_workers = len(args.gpus) * args.max_workers_per_gpu
    results = {}

    print(f"Starting parallel optimization (up to {max_workers} parallel tasks)...\n")
    start_time = time.time()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {}
        for dataset, gpu in gpu_queue:
            future = executor.submit(optimize_single_dataset, dataset, gpu, args, output_dir)
            futures[future] = dataset

        # Collect results
        completed = 0
        for future in as_completed(futures):
            dataset, success, data = future.result()
            results[dataset] = {'success': success, 'data': data}
            completed += 1
            print(f"\nProgress: {completed}/{len(args.datasets)} completed\n")

    total_elapsed = time.time() - start_time
    print(f"\nTotal elapsed time: {total_elapsed/60:.1f} minutes ({total_elapsed/3600:.2f} hours)")

    # Save summary
    save_summary(results, args, output_dir)


if __name__ == '__main__':
    main()

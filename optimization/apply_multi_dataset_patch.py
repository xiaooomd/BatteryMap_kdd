"""
Patch script to add multi-dataset aggregation support to scripts/hyperparameter_optimization.py.

Usage:
    python apply_multi_dataset_patch.py

This script will automatically modify scripts/hyperparameter_optimization.py to support Scheme A (multi-dataset aggregated optimization).
"""

import re
from pathlib import Path


def apply_patch():
    """Apply the patch."""
    file_path = Path(__file__).resolve().parents[1] / 'scripts' / 'hyperparameter_optimization.py'

    if not file_path.exists():
        print("Error: Cannot find scripts/hyperparameter_optimization.py")
        return False

    # Read original file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Backup
    backup_path = Path(__file__).resolve().parents[1] / 'scripts' / 'hyperparameter_optimization_before_patch.py'
    with open(backup_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Backup created: {backup_path}")

    # Apply modifications
    modified = content
    changes = []

    # 1. Modify argument parsing (parse_args)
    # Target: add --datasets, --aggregation, --dataset_weights; change --dataset to optional
    if "'--datasets'" not in modified:
        # Change --dataset from required=True to default=None
        modified = re.sub(
            r"parser\.add_argument\('--dataset', type=str, required=True,",
            "parser.add_argument('--dataset', type=str, default=None,\n                        help='Dataset name (e.g., HUST, CALB) - mutually exclusive with --datasets'),\n    parser.add_argument('--datasets', type=str, nargs='+', default=None,\n                        help='Multiple dataset names (e.g., HUST CALB CALCE) - mutually exclusive with --dataset'),\n    parser.add_argument('--aggregation', type=str, default='mean',\n                        choices=['mean', 'weighted_mean', 'min', 'median'],\n                        help='Multi-dataset aggregation method'),\n    parser.add_argument('--dataset_weights', type=float, nargs='+', default=None,\n                        help='Dataset weights (only used for weighted_mean aggregation'),",
            modified
        )
        changes.append("Added multi-dataset arguments")

    # 2. Add aggregate_scores function after parse_args()
    if 'def aggregate_scores' not in modified:
        aggregate_fn = '''

def aggregate_scores(scores: Dict[str, float],
                     method: str,
                     weights: List[float] = None) -> float:
    """Aggregate performance scores from multiple datasets.

    Args:
        scores: Mapping from dataset name to score
        method: Aggregation method
        weights: Weight list (only used for weighted_mean)

    Returns:
        Aggregated score (lower is better)
    """
    score_values = list(scores.values())

    if method == 'mean':
        return float(np.mean(score_values))
    elif method == 'weighted_mean':
        if weights is None:
            weights = [1.0] * len(score_values)
        return float(np.average(score_values, weights=weights))
    elif method == 'min':  # worst performance
        return float(np.max(score_values))
    elif method == 'median':
        return float(np.median(score_values))
    else:
        return float(np.mean(score_values))

'''
        # Insert after parse_args() function (after "return parser.parse_args()")
        parse_args_end = modified.find('    return parser.parse_args()')
        if parse_args_end != -1:
            next_newline = modified.find('\n', parse_args_end)
            if next_newline != -1:
                modified = modified[:next_newline+1] + aggregate_fn + modified[next_newline+1:]
                changes.append("Added aggregate_scores function")

    # 3. Modify run_single_experiment signature to support dataset parameter
    if 'dataset: str = None' not in modified:
        old_sig = 'def run_single_experiment(params: Dict[str, Any],\n                         args: argparse.Namespace,\n                         trial_id: int) -> Tuple[float, Dict[str, float]]:'
        new_sig = 'def run_single_experiment(params: Dict[str, Any],\n                         args: argparse.Namespace,\n                         trial_id: int,\n                         dataset: str = None) -> Tuple[float, Dict[str, float]]:'
        if old_sig in modified:
            modified = modified.replace(old_sig, new_sig)
            changes.append("Modified run_single_experiment signature")
        else:
            # Try alternative indentation pattern
            old_sig2 = 'def run_single_experiment(params: Dict[str, Any], args: argparse.Namespace, trial_id: int) -> Tuple[float, Dict[str, float]]:'
            new_sig2 = 'def run_single_experiment(params: Dict[str, Any], args: argparse.Namespace, trial_id: int, dataset: str = None) -> Tuple[float, Dict[str, float]]:'
            if old_sig2 in modified:
                modified = modified.replace(old_sig2, new_sig2)
                changes.append("Modified run_single_experiment signature")

    # 4. In run_single_experiment body, use dataset parameter if provided
    if "dataset is not None" not in modified and "'--dataset', args.dataset" in modified:
        # Replace the hardcoded --dataset arg with conditional logic
        old_cmd_dataset = "['--dataset', args.dataset,"
        new_cmd_dataset = "['--dataset', dataset if dataset is not None else args.dataset,"
        modified = modified.replace(old_cmd_dataset, new_cmd_dataset)
        changes.append("Modified run_single_experiment to use dataset parameter")

    # 5. Add run_multi_dataset_experiment function before parse_training_output
    if 'def run_multi_dataset_experiment' not in modified:
        multi_dataset_fn = '''

def run_multi_dataset_experiment(params: Dict[str, Any],
                                 args: argparse.Namespace,
                                 trial_id: int) -> Tuple[float, Dict[str, Any]]:
    """Run experiments on multiple datasets and aggregate results.

    Args:
        params: Hyperparameter dictionary
        args: Global configuration parameters
        trial_id: Trial number

    Returns:
        Aggregated score and detailed results dictionary
    """
    dataset_scores = {}
    dataset_metrics = {}

    print(f"\\n{'='*80}")
    print(f"Trial {trial_id} [multi-dataset]: Evaluating on {len(args.datasets)} datasets")
    print(f"Datasets: {args.datasets}")
    print(f"Parameters: {json.dumps(params, indent=2)}")
    print(f"{'='*80}\\n")

    # Evaluate on each dataset
    for dataset in args.datasets:
        try:
            score, metrics = run_single_experiment(params, args, trial_id, dataset=dataset)
            dataset_scores[dataset] = score
            dataset_metrics[dataset] = metrics
        except Exception as e:
            print(f"Warning: Dataset {dataset} training failed, skipping - {str(e)}")
            dataset_scores[dataset] = float('inf')
            dataset_metrics[dataset] = {}

    # Filter out failed datasets
    valid_scores = {k: v for k, v in dataset_scores.items() if v != float('inf')}

    if not valid_scores:
        print(f"Error: Trial {trial_id} all datasets failed")
        return float('inf'), {}

    # Aggregate scores
    aggregated_score = aggregate_scores(valid_scores, args.aggregation, args.dataset_weights)

    print(f"\\nTrial {trial_id} aggregated results:")
    for dataset, score in valid_scores.items():
        print(f"  {dataset}: {score:.4f}")
    print(f"  Aggregated score ({args.aggregation}): {aggregated_score:.4f}")

    # Build return result
    result = {
        'aggregated_score': aggregated_score,
        'dataset_scores': dataset_scores,
        'dataset_metrics': dataset_metrics
    }

    return aggregated_score, result

'''
        # Insert before parse_training_output
        parse_output_start = modified.find('def parse_training_output')
        if parse_output_start != -1:
            modified = modified[:parse_output_start] + multi_dataset_fn + modified[parse_output_start:]
            changes.append("Added run_multi_dataset_experiment function")

    # 6. Modify main() to add parameter validation
    if '"--dataset or --datasets"' not in modified:
        main_start = modified.find('def main():')
        if main_start != -1:
            # Find after args = parse_args()
            args_line = modified.find('args = parse_args()', main_start)
            if args_line != -1:
                # Find the line after args = parse_args()
                next_newline = modified.find('\n', args_line)
                if next_newline != -1:
                    validation_code = '''
    # Validate parameters
    if args.dataset is None and args.datasets is None:
        raise ValueError("Must specify either --dataset or --datasets")
    if args.dataset is not None and args.datasets is not None:
        raise ValueError("--dataset and --datasets cannot be specified simultaneously")

    # Normalize to list
    if args.dataset:
        args.datasets = [args.dataset]

    # Validate weights
    if args.aggregation == 'weighted_mean' and args.dataset_weights:
        if len(args.dataset_weights) != len(args.datasets):
            raise ValueError(f"Weight count ({len(args.dataset_weights)}) must match dataset count ({len(args.datasets)})")

'''
                    modified = modified[:next_newline+1] + validation_code + modified[next_newline+1:]
                    changes.append("Added parameter validation logic")

    # Write back file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(modified)

    if changes:
        print("\nSuccessfully applied the following changes:")
        for change in changes:
            print(f"  - {change}")
        print(f"\nFile updated: {file_path}")
        print("\nManual modifications still required:")
        print("  1. In run_single_experiment body, use the dataset parameter for command building")
        print("  2. Add multi-dataset logic in evaluate_fn to switch between single/multi mode")
        print("  3. Handle multi-dataset results in save_results")
        print("\nFor details, see: docs/MULTI_DATASET_OPTIMIZATION.md")
        return True
    else:
        print("File already contains all modifications, no need to reapply")
        return True


if __name__ == '__main__':
    print("="*80)
    print("Applying multi-dataset support patch to scripts/hyperparameter_optimization.py")
    print("="*80 + "\n")

    success = apply_patch()

    if success:
        print("\nPatch application complete!")
        print("\nNext steps:")
        print("1. See docs/MULTI_DATASET_OPTIMIZATION.md for detailed usage")
        print("2. Test single-dataset mode still works:")
        print("   python run.py hyperopt --method pso --model MLP --dataset HUST")
        print("3. Test multi-dataset mode:")
        print("   python run.py hyperopt --method pso --model MLP --datasets HUST CALB --aggregation mean")
    else:
        print("\nPatch application failed")

"""Hyperparameter optimization quick start script (Windows PowerShell).

Directly run the Python script for hyperparameter optimization without a bash environment.
"""

import subprocess
import sys
from pathlib import Path


def run_optimization(config: dict) -> None:
    """Run a single optimization task.

    Args:
        config: Optimization configuration dictionary
    """
    cmd = [
        sys.executable,  # Use the current Python interpreter
        str(Path(__file__).resolve().parent / 'hyperparameter_optimization.py'),
    ]

    # Add all parameters
    for key, value in config.items():
        if isinstance(value, bool):
            if value:
                cmd.append(f'--{key}')
        else:
            cmd.extend([f'--{key}', str(value)])

    print(f"\n{'='*80}")
    print(f"Running configuration: {config['model']} - {config['method']}")
    print(f"{'='*80}\n")

    # Run optimization
    subprocess.run(cmd, check=True)


def main():
    """Main function - defines multiple optimization tasks."""

    # Ensure output directory exists
    Path('./hyperparam_search_results').mkdir(exist_ok=True)

    # Define list of optimization tasks
    tasks = [
        # Task 1: Grid Search - MLP model (Quick test)
        {
            'method': 'grid',
            'model': 'MLP',
            'dataset': 'HUST',
            'feature_type': 'curve',
            'train_epochs': 5,
            'batch_size': 32,
            'metric': 'val_mae',
            'gpu': '0',
            'output_dir': './hyperparam_search_results',
        },

        # Task 2: PSO optimization - CPMLP model
        {
            'method': 'pso',
            'model': 'CPMLP',
            'dataset': 'HUST',
            'feature_type': 'curve',
            'n_particles': 15,
            'n_iterations': 30,
            'w': 0.7,
            'c1': 1.5,
            'c2': 1.5,
            'train_epochs': 10,
            'metric': 'val_rmse',
            'gpu': '0',
            'output_dir': './hyperparam_search_results',
        },

        # Task 3: PSO optimization - Transformer model (Feature mode)
        {
            'method': 'pso',
            'model': 'Transformer',
            'dataset': 'HUST',
            'feature_type': 'extracted_features',
            'n_particles': 20,
            'n_iterations': 40,
            'train_epochs': 10,
            'metric': 'val_mae',
            'gpu': '0',
            'output_dir': './hyperparam_search_results',
        },
    ]

    # Ask user which task to run
    print("Available optimization tasks:")
    for i, task in enumerate(tasks, 1):
        print(f"{i}. {task['method'].upper()} - {task['model']} "
              f"({task['feature_type']}, {task.get('n_particles', 'N/A')} particles)")

    print(f"{len(tasks) + 1}. Run all tasks")
    print("0. Exit")

    try:
        choice = int(input("\nPlease select task number: "))

        if choice == 0:
            print("Exiting program")
            return
        elif choice == len(tasks) + 1:
            # Run all tasks
            for task in tasks:
                run_optimization(task)
        elif 1 <= choice <= len(tasks):
            # Run selected task
            run_optimization(tasks[choice - 1])
        else:
            print("Invalid selection")

    except ValueError:
        print("Invalid input")
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user")
    except Exception as e:
        print(f"\nError: {str(e)}")

    print(f"\n{'='*80}")
    print("Optimization completed! Results saved in ./hyperparam_search_results/")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()

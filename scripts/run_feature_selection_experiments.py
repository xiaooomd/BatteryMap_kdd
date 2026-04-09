#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Feature selection experiment script - Evaluate the impact of different feature sets on model performance

Based on scripts/run_sensitivity_analysis.py, used to evaluate:
1. li_selected_results (27 features)
2. robust_selected_results (24 features)

Uses the optimal hyperparameters obtained from PSO optimization to evaluate model performance on the two feature sets.
n_selected=-1 is used to indicate using all features.

Example:
    # Run all models and datasets
    python scripts/run_feature_selection_experiments.py \
        --model MLP \
        --gpus 0,1,2,3,4,5

    # Run specified datasets
    python scripts/run_feature_selection_experiments.py \
        --model Transformer \
        --datasets HNEI,CALB1,MIT \
        --gpus 0,1,2,3
"""

import os
import argparse
import subprocess
import json
import pandas as pd
from tqdm import tqdm
import multiprocessing
import time
import queue


def run_experiment_worker(task_info):
    """
    Worker function to run a single experiment on a specific GPU.
    """
    gpu_id, args_dict = task_info

    # Set the visible device for this process
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    cmd = ["python", "run_main.py"]
    for key, value in args_dict.items():
        cmd.append(f"--{key}")
        cmd.append(str(value))

    try:
        # Run process
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)

        # Print stdout for debugging
        if result.stdout:
            print(f"[GPU {gpu_id}] STDOUT:\n{result.stdout}")
        
        if result.returncode != 0:
            print(f"[GPU {gpu_id}] Error running experiment: {result.stderr}")
            return None
        
        # Print stderr even on success (warnings might be there)
        if result.stderr:
            print(f"[GPU {gpu_id}] STDERR:\n{result.stderr}")

        # Find the output directory to read profiling_stats.json
        checkpoints_dir = args_dict.get('checkpoints', './checkpoints/')
        model_comment = args_dict.get('model_comment', '')

        # Filter dirs by model_comment and find the most recent one
        target_dir = None
        latest_time = 0
        matched_dirs = []

        try:
            for d in os.listdir(checkpoints_dir):
                if model_comment in d:
                    full_path = os.path.join(checkpoints_dir, d)
                    if os.path.isdir(full_path):
                        mtime = os.path.getmtime(full_path)
                        matched_dirs.append((full_path, mtime))
                        if mtime > latest_time:
                            latest_time = mtime
                            target_dir = full_path
        except Exception as e:
            print(f"[GPU {gpu_id}] Error listing checkpoint directory: {e}")
            return None

        if target_dir:
            print(f"[GPU {gpu_id}] Found checkpoint directory: {target_dir}")
            if len(matched_dirs) > 1:
                print(f"[GPU {gpu_id}] Warning: Multiple directories matched '{model_comment}': {len(matched_dirs)} dirs")
            
            stats_path = os.path.join(target_dir, "profiling_stats.json")
            
            # Retry logic with small delay for filesystem sync
            max_retries = 3
            for attempt in range(max_retries):
                if os.path.exists(stats_path):
                    try:
                        with open(stats_path, 'r') as f:
                            stats = json.load(f)
                        print(f"[GPU {gpu_id}] Successfully loaded profiling_stats.json")
                        return stats
                    except json.JSONDecodeError as e:
                        print(f"[GPU {gpu_id}] JSON decode error (attempt {attempt+1}/{max_retries}): {e}")
                        if attempt < max_retries - 1:
                            time.sleep(0.5)
                    except Exception as e:
                        print(f"[GPU {gpu_id}] Error reading profiling_stats.json: {e}")
                        return None
                else:
                    if attempt < max_retries - 1:
                        print(f"[GPU {gpu_id}] profiling_stats.json not found yet, retrying... (attempt {attempt+1}/{max_retries})")
                        time.sleep(1.0)
                    else:
                        print(f"[GPU {gpu_id}] Warning: profiling_stats.json not found in {target_dir} after {max_retries} attempts")
                        try:
                            files = os.listdir(target_dir)
                            print(f"[GPU {gpu_id}] Directory contents: {files}")
                        except:
                            pass
            return None
        else:
            print(f"[GPU {gpu_id}] Error: No checkpoint directory found for '{model_comment}'")
            print(f"[GPU {gpu_id}] Checkpoint base directory: {checkpoints_dir}")
            try:
                all_dirs = [d for d in os.listdir(checkpoints_dir) if os.path.isdir(os.path.join(checkpoints_dir, d))]
                print(f"[GPU {gpu_id}] Available directories: {all_dirs[:5]}...")
            except:
                pass
            return None

    except Exception as e:
        print(f"[GPU {gpu_id}] Exception during execution: {e}")
        return None


def worker_process(gpu_id, task_queue, result_queue):
    """Worker process that processes tasks from the queue."""
    while True:
        try:
            task = task_queue.get(block=False)
        except queue.Empty:
            break

        task_data = task['args']
        meta_data = task['meta']

        print(f"[GPU {gpu_id}] Starting task: {task_data['model_comment']}")
        stats = run_experiment_worker((gpu_id, task_data))

        if stats:
            res = {
                "dataset": meta_data['dataset'],
                "feature_set": meta_data['feature_set'],
                "n_features": meta_data['n_features'],
                "status": "success",
                **stats
            }
            result_queue.put(res)
        else:
            # Record failure with NaN values for metrics
            print(f"[GPU {gpu_id}] Task failed: {task_data['model_comment']}")
            res = {
                "dataset": meta_data['dataset'],
                "feature_set": meta_data['feature_set'],
                "n_features": meta_data['n_features'],
                "status": "failed",
                "flops_M": float('nan'),
                "params_M": float('nan'),
                "latency_ms": float('nan'),
                "test_mape": float('nan'),
                "test_rmse": float('nan'),
                "test_mae": float('nan')
            }
            result_queue.put(res)

        task_queue.task_done()


def detect_feature_count(csv_dir):
    """Detect the number of features in a CSV directory."""
    try:
        csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
        if csv_files:
            sample_csv = os.path.join(csv_dir, csv_files[0])
            df = pd.read_csv(sample_csv, nrows=1)
            return len(df.columns)
    except Exception as e:
        print(f"Warning: Could not detect feature count in {csv_dir}: {e}")
    return None


def main():
    parser = argparse.ArgumentParser(description='Feature Selection Experiment Script (Multi-GPU)')
    parser.add_argument('--model', type=str, required=True,
                        help='Model name (e.g. MLP, CNN, Transformer, Autoformer, etc.)')
    parser.add_argument('--datasets', type=str, default=None,
                        help='Comma-separated dataset names (e.g. CALB1,CALB2). Runs all datasets if not provided')
    parser.add_argument('--output_dir', type=str, default='feature_selection_results',
                        help='Directory to save results')
    parser.add_argument('--gpus', type=str, default='0,1,2,3,4,5',
                        help='Comma-separated GPU IDs')
    parser.add_argument('--feature_dirs', type=str,
                        default='li_selected_results,robust_selected_results',
                        help='Comma-separated feature directory names')
    args = parser.parse_args()

    gpu_list = [int(g.strip()) for g in args.gpus.split(',')]
    print(f"Using GPUs: {gpu_list}")

    # Feature directories to evaluate
    feature_dirs = [d.strip() for d in args.feature_dirs.split(',')]
    print(f"Feature directories: {feature_dirs}")

    # Full list of datasets (same as in scripts/run_sensitivity_analysis.py)
    ALL_DATASETS = [
        "CALCE", "HNEI", "HUST", "ISU_ILCC", "MATR", "MICH", "MICH_EXP", "MIT",
        "RWTH", "SNL", "Stanford", "Tongji", "XJTU", "NAion2024", "CALB1", "CALB2", "ZN-coin2024"
    ]

    if args.datasets:
        dataset_list = [d.strip() for d in args.datasets.split(',')]
    else:
        dataset_list = ALL_DATASETS
        print(f"No datasets specified. Running all {len(dataset_list)} datasets: {dataset_list}")

    # Prepare Task Queue
    task_queue = multiprocessing.JoinableQueue()
    result_queue = multiprocessing.Queue()

    total_tasks = 0

    # Ensure output directory exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created output directory: {args.output_dir}")

    # Map dataset names to directory names if they differ
    dataset_dir_map = {
        "NAion2024": "NA",
        "ZN-coin2024": "ZNion",
    }

    # Process each dataset and feature directory combination
    for feature_dir_name in feature_dirs:
        print(f"\n{'='*80}")
        print(f"Processing feature directory: {feature_dir_name}")
        print(f"{'='*80}")
        
        for dataset_name in dataset_list:
            # Dynamic config loading logic
            agg_config_path = f"configs/{args.model}_best_hparams.json"
            temp_config_path = None
            base_hparams = {}

            if os.path.exists(agg_config_path):
                print(f"Loading aggregated config from: {agg_config_path}")
                try:
                    with open(agg_config_path, 'r') as f:
                        all_configs = json.load(f)
                    if dataset_name in all_configs:
                        base_hparams = all_configs[dataset_name]
                        temp_config_path = f"configs/temp_{args.model}_{dataset_name}.json"
                        with open(temp_config_path, 'w') as f:
                            json.dump(base_hparams, f, indent=4)
                        config_path = temp_config_path
                    else:
                        # Fallback
                        config_path = f"configs/{args.model}_{dataset_name}_hparams.json"
                        if os.path.exists(config_path):
                            with open(config_path, 'r') as f:
                                base_hparams = json.load(f)
                        else:
                            print(f"Skipping {dataset_name} (No config found)")
                            continue
                except Exception as e:
                    print(f"Error parsing config: {e}")
                    continue
            else:
                config_path = f"configs/{args.model}_{dataset_name}_hparams.json"
                if not os.path.exists(config_path):
                    print(f"Skipping {dataset_name} (No config: {config_path})")
                    continue
                with open(config_path, 'r') as f:
                    base_hparams = json.load(f)

            # Handle directory mapping
            dir_name = dataset_dir_map.get(dataset_name, dataset_name)
            root_path = f"dataset/{feature_dir_name}/{dir_name}"

            if not os.path.exists(root_path):
                print(f"Warning: Dataset directory {root_path} does not exist. Skipping {dataset_name}.")
                continue

            # Detect feature count for this dataset
            n_features = detect_feature_count(root_path)
            if n_features is None:
                print(f"Warning: Could not detect features for {dataset_name} in {root_path}. Skipping.")
                continue
            
            print(f"Dataset: {dataset_name}, Feature set: {feature_dir_name}, Features: {n_features}")

            common_args = {
                "model": args.model,
                "dataset": dataset_name,
                "config": config_path,
                "task_type": "early_prediction",
                "feature_type": "extracted_features",
                "train_epochs": base_hparams.get("train_epochs", 100),
                "patience": base_hparams.get("patience", 5),
                "seed": 2021,
                "root_path": root_path,
                "n_selected": -1,  # Use all features
                "early_cycle_threshold": 100,  # Fixed at 100 cycles
                "pred_len": 1,
                "dec_in": 1
            }

            # Create unique model comment
            exp_args = common_args.copy()
            exp_args["model_comment"] = f"{args.model}_{dataset_name}_{feature_dir_name}_all_features"

            task = {
                "args": exp_args,
                "meta": {
                    "dataset": dataset_name,
                    "feature_set": feature_dir_name,
                    "n_features": n_features
                }
            }
            task_queue.put(task)
            total_tasks += 1

    print(f"\n{'='*80}")
    print(f"Total tasks scheduled: {total_tasks}")
    print(f"{'='*80}\n")

    if total_tasks == 0:
        print("No tasks to run. Exiting.")
        return

    # Start Workers
    processes = []
    for gpu_id in gpu_list:
        p = multiprocessing.Process(target=worker_process, args=(gpu_id, task_queue, result_queue))
        p.start()
        processes.append(p)

    # Progress Monitor
    results = []
    generated_files = set()

    with tqdm(total=total_tasks, desc="Total Progress") as pbar:
        completed_count = 0
        while completed_count < total_tasks:
            try:
                # Wait for result with timeout to update progress
                res = result_queue.get(timeout=1)
                results.append(res)

                # Save to CSV file per feature set
                feature_set = res['feature_set']
                filename = f"{args.model}_{feature_set}_results.csv"
                filepath = os.path.join(args.output_dir, filename)

                # Append to specific file
                df_single = pd.DataFrame([res])
                header = not os.path.exists(filepath)
                df_single.to_csv(filepath, mode='a', header=header, index=False)
                generated_files.add(filepath)

                pbar.update(1)
                completed_count += 1
            except queue.Empty:
                # Check if workers are still alive
                if not any(p.is_alive() for p in processes) and task_queue.empty():
                    break
                continue

    # Cleanup
    for p in processes:
        p.join()

    print(f"\n{'='*80}")
    print(f"All experiments completed!")
    print(f"{'='*80}")
    print(f"Results saved to directory: {args.output_dir}")
    print(f"\nGenerated files:")
    for f in sorted(list(generated_files)):
        print(f" - {f}")
    
    # Print summary statistics
    if results:
        df_all = pd.DataFrame(results)
        print(f"\n{'='*80}")
        print("Summary Statistics:")
        print(f"{'='*80}")
        
        for feature_set in feature_dirs:
            df_subset = df_all[df_all['feature_set'] == feature_set]
            if len(df_subset) > 0:
                successful = df_subset[df_subset['status'] == 'success']
                print(f"\n{feature_set}:")
                print(f"  Total: {len(df_subset)}, Success: {len(successful)}, Failed: {len(df_subset) - len(successful)}")
                if len(successful) > 0:
                    print(f"  Avg Test MAE: {successful['test_mae'].mean():.4f}")
                    print(f"  Avg Test RMSE: {successful['test_rmse'].mean():.4f}")
                    print(f"  Avg Test MAPE: {successful['test_mape'].mean():.4f}")


if __name__ == "__main__":
    main()

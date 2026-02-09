import os
import argparse
import subprocess
import json
import pandas as pd
import datetime
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

    # print(f"[GPU {gpu_id}] Executing: {' '.join(cmd)}")

    try:
        # Run process
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)

        # Print stdout for debugging (especially for FLOPs calculation)
        if result.stdout:
            print(f"[GPU {gpu_id}] STDOUT:\n{result.stdout}")
        
        if result.returncode != 0:
            print(f"[GPU {gpu_id}] Error running experiment: {result.stderr}")
            return None
        
        # Print stderr even on success (warnings might be there)
        if result.stderr:
            print(f"[GPU {gpu_id}] STDERR:\n{result.stderr}")

        # Find the output directory to read profiling_stats.json
        # Heuristic: Scan checkpoints/ for the most recent directory containing profiling_stats.json
        # IMPORTANT: Since multiple processes run in parallel, simple timestamp check might be race-prone
        # But run_main creates unique dirs with timestamp.
        # Ideally run_main should output the path, but capturing stdout is messy.
        # We will use the 'model_comment' to find the specific directory if possible, or scan carefully.

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
                        time.sleep(1.0)  # Wait for filesystem sync
                    else:
                        print(f"[GPU {gpu_id}] Warning: profiling_stats.json not found in {target_dir} after {max_retries} attempts")
                        # List directory contents for debugging
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
                print(f"[GPU {gpu_id}] Available directories: {all_dirs[:5]}...")  # Show first 5
            except:
                pass
            return None

    except Exception as e:
        print(f"[GPU {gpu_id}] Exception during execution: {e}")
        return None

def worker_process(gpu_id, task_queue, result_queue):
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
                "type": meta_data['type'],
                "n_selected": meta_data['n_selected'],
                "early_cycle": meta_data['early_cycle'],
                "status": "success",
                **stats
            }
            result_queue.put(res)
        else:
            # Record failure with NaN values for metrics
            print(f"[GPU {gpu_id}] Task failed: {task_data['model_comment']}")
            res = {
                "dataset": meta_data['dataset'],
                "type": meta_data['type'],
                "n_selected": meta_data['n_selected'],
                "early_cycle": meta_data['early_cycle'],
                "status": "failed",
                "flops_M": float('nan'),
                "params_M": float('nan'),
                "latency_ms": float('nan'),
                "test_mape": float('nan'),
                "test_rmse": float('nan')
            }
            result_queue.put(res)

        task_queue.task_done()

def main():
    parser = argparse.ArgumentParser(description='Sensitivity Analysis Script (Multi-GPU)')
    parser.add_argument('--model', type=str, required=True, help='Model name (e.g. MLP, CNN, Transformer)')
    parser.add_argument('--datasets', type=str, default=None, help='Comma-separated dataset names (e.g. CALB1,CALB2). If not provided, runs all datasets.')
    parser.add_argument('--output_csv', type=str, default='sensitivity_results.csv', help='Output CSV file (Path used for directory)')
    parser.add_argument('--gpus', type=str, default='0,1,2,3,4,5', help='Comma-separated GPU IDs to use')
    args = parser.parse_args()

    gpu_list = [int(g.strip()) for g in args.gpus.split(',')]
    print(f"Using GPUs: {gpu_list}")

    # Define Experiment Grid
    n_selected_list = [35, 15, 10, 5]
    # Assuming n_selected=20 for cycle sensitivity task unless specified otherwise
    early_cycle_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    # Full list of datasets
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
    output_dir = os.path.dirname(args.output_csv)
    if not output_dir:
        output_dir = "." # Current directory if no path
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Map dataset names to directory names if they differ
    dataset_dir_map = {
        "NAion2024": "NA",
        "ZN-coin2024": "ZNion",
    }

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
                        with open(config_path, 'r') as f: base_hparams = json.load(f)
                    else:
                        print(f"Skipping {dataset_name} (No config)")
                        continue
            except Exception as e:
                print(f"Error parsing config: {e}")
                continue
        else:
            config_path = f"configs/{args.model}_{dataset_name}_hparams.json"
            if not os.path.exists(config_path):
                 print(f"Skipping {dataset_name} (No config)")
                 continue
            with open(config_path, 'r') as f: base_hparams = json.load(f)

        # Handle directory mapping
        dir_name = dataset_dir_map.get(dataset_name, dataset_name)
        root_path = f"dataset/selected_result/{dir_name}"

        if not os.path.exists(root_path):
            print(f"Warning: Dataset directory {root_path} does not exist. Skipping {dataset_name}.")
            continue

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
            "pred_len": 1,
            "dec_in": 1  # Will be updated to match enc_in dynamically in run_main, but providing a default is safe
        }

        # Task 1
        fixed_cycle = 100
        for n_sel in n_selected_list:
            exp_args = common_args.copy()
            exp_args["n_selected"] = n_sel
            exp_args["early_cycle_threshold"] = fixed_cycle
            exp_args["model_comment"] = f"{args.model}_{dataset_name}_sens_feat_{n_sel}"

            task = {
                "args": exp_args,
                "meta": {
                    "dataset": dataset_name,
                    "type": "feature_sensitivity",
                    "n_selected": n_sel,
                    "early_cycle": fixed_cycle
                }
            }
            task_queue.put(task)
            total_tasks += 1

        # Task 2
        fixed_n_selected = 20
        for cycle in early_cycle_list:
            exp_args = common_args.copy()
            exp_args["n_selected"] = fixed_n_selected
            exp_args["early_cycle_threshold"] = cycle
            exp_args["model_comment"] = f"{args.model}_{dataset_name}_sens_cycle_{cycle}"

            task = {
                "args": exp_args,
                "meta": {
                    "dataset": dataset_name,
                    "type": "cycle_sensitivity",
                    "n_selected": fixed_n_selected,
                    "early_cycle": cycle
                }
            }
            task_queue.put(task)
            total_tasks += 1

    print(f"Total tasks scheduled: {total_tasks}")

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

                # Split Logic: Save to individual CSV files
                dataset_name = res['dataset']
                task_type_raw = res['type']

                # Simplify task name
                if task_type_raw == 'feature_sensitivity':
                    task_suffix = 'feature'
                elif task_type_raw == 'cycle_sensitivity':
                    task_suffix = 'cycle'
                else:
                    task_suffix = task_type_raw

                filename = f"{dataset_name}_{task_suffix}.csv"
                filepath = os.path.join(output_dir, filename)

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

    print(f"\nAll experiments completed.")
    print(f"Results saved to directory: {output_dir}")
    print("Generated files:")
    for f in sorted(list(generated_files)):
        print(f" - {f}")

if __name__ == "__main__":
    main()

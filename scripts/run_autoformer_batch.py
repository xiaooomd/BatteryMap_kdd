"""
Batch train Autoformer on multiple small datasets
Based on datasets with complete train/val/test in data_split_recorder.py
NA, ZN, CALB uniformly use the 2024 version
"""
import subprocess
import os
from datetime import datetime

# Define all datasets (based on datasets with complete train/val/test in data_split_recorder.py)
DATASETS = [
    "HUST", "MATR", "SNL", "RWTH", "MICH", "MICH_EXP",
    "CALCE", "HNEI", "Tongji", "Stanford", "ISU_ILCC", "XJTU",
    "ZN_2024", "CALB_2024", "NAion_2024"
]

# Training configuration
CONFIG = {
    "model_name": "Autoformer",
    "train_epochs": 100,
    "early_cycle_threshold": 100,
    "learning_rate": 0.00005,
    "master_port": 25529,
    "num_process": 2,
    "batch_size": 4,
    "n_heads": 4,
    "seq_len": 1,
    "accumulation_steps": 4,
    "lstm_layers": 6,
    "e_layers": 2,
    "d_layers": 2,
    "d_model": 128,
    "d_ff": 256,
    "dropout": 0.1,
    "charge_discharge_length": 300,
    "patience": 5,
    "lradj": "constant",
    "loss": "MSE",
    "patch_len": 50,
    "stride": 50,
    "seed": 2021,
    "data": "Dataset_original",
    "root_path": "./dataset",
    "comment": "Autoformer",
    "task_name": "classification",
}


def train_on_dataset(dataset_name):
    """Train model on a specified dataset"""
    print("=" * 50)
    print(f"Start training dataset: {dataset_name}")
    print("=" * 50)

    # Create checkpoint path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoints = f"./checkpoints/Autoformer_{dataset_name}_{timestamp}"

    # Build command
    cmd = [
        "accelerate", "launch",
        "--multi_gpu",
        "--num_processes", str(CONFIG["num_process"]),
        "--main_process_port", str(CONFIG["master_port"]),
        "run_main.py",
        "--task_name", CONFIG["task_name"],
        "--data", CONFIG["data"],
        "--is_training", "1",
        "--root_path", CONFIG["root_path"],
        "--model_id", f"Autoformer_{dataset_name}",
        "--model", CONFIG["model_name"],
        "--features", "MS",
        "--seed", str(CONFIG["seed"]),
        "--seq_len", str(CONFIG["seq_len"]),
        "--label_len", "50",
        "--factor", "3",
        "--enc_in", "3",
        "--dec_in", "1",
        "--c_out", "1",
        "--des", "Exp",
        "--itr", "1",
        "--class_num", "1",
        "--d_model", str(CONFIG["d_model"]),
        "--d_ff", str(CONFIG["d_ff"]),
        "--batch_size", str(CONFIG["batch_size"]),
        "--learning_rate", str(CONFIG["learning_rate"]),
        "--train_epochs", str(CONFIG["train_epochs"]),
        "--model_comment", CONFIG["comment"],
        "--accumulation_steps", str(CONFIG["accumulation_steps"]),
        "--charge_discharge_length", str(CONFIG["charge_discharge_length"]),
        "--dataset", dataset_name,
        "--num_workers", "32",
        "--e_layers", str(CONFIG["e_layers"]),
        "--lstm_layers", str(CONFIG["lstm_layers"]),
        "--d_layers", str(CONFIG["d_layers"]),
        "--patience", str(CONFIG["patience"]),
        "--n_heads", str(CONFIG["n_heads"]),
        "--early_cycle_threshold", str(CONFIG["early_cycle_threshold"]),
        "--dropout", str(CONFIG["dropout"]),
        "--lradj", CONFIG["lradj"],
        "--loss", CONFIG["loss"],
        "--checkpoints", checkpoints,
    ]

    # Set CUDA devices
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0,1"

    # Execute training
    try:
        subprocess.run(cmd, env=env, check=True)
        print(f"\nDataset {dataset_name} training completed!\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\nDataset {dataset_name} training failed: {e}\n")
        return False


def main():
    """Main function: batch train all datasets"""
    results = {}

    for dataset in DATASETS:
        success = train_on_dataset(dataset)
        results[dataset] = "Success" if success else "Failed"

    # Print summary results
    print("=" * 50)
    print("All datasets training completed! Summary:")
    print("=" * 50)
    for dataset, status in results.items():
        print(f"{dataset}: {status}")


if __name__ == "__main__":
    main()

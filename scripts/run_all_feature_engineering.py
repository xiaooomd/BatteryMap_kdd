import os
import sys
import subprocess
import logging
from pathlib import Path

# Project Context
project_root = Path(__file__).resolve().parent.parent

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("RunAllFeatureEngineering")

def run_extraction():
    features_dir = project_root / "features"
    feature_scripts = sorted(features_dir.glob("features_*.py"))
    
    logger.info(f"Found {len(feature_scripts)} feature extraction scripts.")
    for script_path in feature_scripts:
        logger.info(f"Running extraction script: {script_path.name}")
        cmd = [sys.executable, str(script_path)]
        try:
            subprocess.run(cmd, check=True, cwd=str(project_root))
            logger.info(f"Successfully finished extraction for {script_path.name}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Extraction failed for {script_path.name}: {e}")

def run_cleaning_and_selection():
    results_features_dir = project_root / "results" / "features"
    if not results_features_dir.exists():
        logger.warning(f"Feature output directory {results_features_dir} does not exist. Nothing to clean.")
        return
        
    dataset_folders = [d.name for d in results_features_dir.iterdir() if d.is_dir()]
    logger.info(f"Found {len(dataset_folders)} datasets for feature cleaning: {dataset_folders}")
    
    for dataset_id in dataset_folders:
        logger.info(f"Running feature selection/cleaning for dataset: {dataset_id}")
        cmd = [
            sys.executable, 
            str(project_root / "run_feature_selection.py"),
            "--dataset_id", dataset_id,
            "--nrows", "-1"
        ]
        try:
            # Check if directory exist or has valid files
            subprocess.run(cmd, check=True, cwd=str(project_root))
            logger.info(f"Successfully finished cleaning for {dataset_id}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Cleaning failed for {dataset_id}: {e}")

def main():
    # Set the root directory for battery raw data reading
    data_root = r"F:\datasets\battery"
    os.environ["BATTERY_DATA_ROOT"] = data_root
    logger.info(f"Set BATTERY_DATA_ROOT to {data_root}")
    
    logger.info("=== STEP 1: Feature Extraction (All Cycles) ===")
    run_extraction()
    
    logger.info("=== STEP 2: Feature Cleaning & Standardizing (All Cycles) ===")
    run_cleaning_and_selection()
    
    logger.info("All feature engineering tasks completed.")

if __name__ == "__main__":
    main()

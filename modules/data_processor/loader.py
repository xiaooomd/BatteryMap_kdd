import json
import logging
from pathlib import Path
from typing import Iterator, Tuple, Dict, Any, List, Optional, Union

import pandas as pd
from tqdm import tqdm


class DataLoader:
    """Data loading utility following the generator pattern.

    Loads feature CSVs and label JSONs, returning structured data per battery.
    Updated to use pathlib.Path for robust cross-platform path handling.
    """

    def __init__(self,
                 feature_dir: Union[str, Path],
                 label_dir: Union[str, Path],
                 nrows: int = 100,
                 early_cycle_threshold: int = 100):
        """Initializes DataLoader.

        Args:
            feature_dir: Directory containing feature CSV files (organized by dataset folders).
            label_dir: Directory containing label JSON files.
            nrows: Number of rows (cycles) to read from each feature CSV.
            early_cycle_threshold: Minimum cycle life threshold; batteries with lower life are skipped.
        """
        self.feature_dir = Path(feature_dir)
        self.label_dir = Path(label_dir)
        self.nrows = nrows
        self.early_cycle_threshold = early_cycle_threshold
        self.logger = logging.getLogger("FeatureSelection.DataLoader")

    def get_available_datasets(self) -> List[str]:
        """Scans the feature directory and returns available dataset IDs (folder names)."""
        if not self.feature_dir.is_dir():
            self.logger.error(f"Feature directory not found or not a directory: {self.feature_dir}")
            return []
        try:
            # List only dataset directories that contain feature CSV files.
            # This avoids picking helper folders like "*_curves".
            dataset_folders = sorted([
                d.name
                for d in self.feature_dir.iterdir()
                if d.is_dir() and not d.name.lower().endswith("_curves") and any(d.glob("*.csv"))
            ])
            return dataset_folders
        except Exception as e:
            self.logger.error(f"Error scanning dataset folders: {e}")
            return []

    def _load_labels(self) -> Dict[str, Any]:
        """Loads and merges all label JSON files into a normalized dictionary.

        Keys are normalized (extension stripped) to match feature filenames.
        Filters out batteries with life <= early_cycle_threshold.
        """
        label_map = {}
        if not self.label_dir.is_dir():
            self.logger.error(f"Label directory not found: {self.label_dir}")
            return label_map

        for file_path in self.label_dir.glob('*.json'):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for key, value in data.items():
                        # Validate value is numeric and above threshold
                        if isinstance(value, (int, float)) and value <= self.early_cycle_threshold:
                            self.logger.debug(f"Skipping battery {key}: Life {value} <= {self.early_cycle_threshold}")
                            continue

                        # key might be filename like 'b1c0.csv', strip extension just in case
                        # But typically keys in JSON are battery IDs.
                        # We normalize by stripping likely extensions if present in key.
                        base_key = Path(key).stem
                        label_map[base_key] = value
            except Exception as e:
                self.logger.error(f"Failed to load label file {file_path}: {e}")

        self.logger.info(f"Loaded {len(label_map)} valid labels from {self.label_dir} "
                         f"(Filtered life <= {self.early_cycle_threshold}).")
        return label_map

    def load_dataset_generator(self, dataset_ids_to_run: Optional[List[str]] = None) -> Iterator[Tuple[str, Dict[str, Dict[str, Any]]]]:
        """Generator that yields data for each dataset sequentially.

        Args:
            dataset_ids_to_run: Optional list of dataset IDs to process. If None, processes all found.

        Yields:
            Tuple containing:
                - dataset_id (str): Name of the dataset.
                - battery_data (dict): Dictionary mapping battery_id to {'X': DataFrame, 'y': label}.
        """
        label_map = self._load_labels()
        if not label_map:
            self.logger.error("No label data loaded. Aborting data loading.")
            return

        all_datasets = self.get_available_datasets()
        if not all_datasets:
            self.logger.warning(f"No dataset folders found in {self.feature_dir}.")
            return

        datasets_to_process = dataset_ids_to_run if dataset_ids_to_run is not None else all_datasets

        for dataset_id in tqdm(datasets_to_process, desc="Processing Datasets"):
            folder_path = self.feature_dir / dataset_id
            if not folder_path.exists():
                self.logger.warning(f"Dataset folder '{dataset_id}' does not exist. Skipped.")
                continue

            csv_files = sorted(list(folder_path.glob('*.csv')))
            if not csv_files:
                self.logger.warning(f"No CSV files found in '{dataset_id}'. Skipped.")
                continue

            battery_data_dict = {}

            for file_path in csv_files:
                battery_id = file_path.stem
                label_value = label_map.get(battery_id)

                # Skip if no label found (or filtered out)
                if label_value is None:
                    continue

                try:
                    # Read only first nrows (early cycles)
                    # Support nrows=-1 or None for all cycles
                    actual_nrows = self.nrows if self.nrows is not None and self.nrows > 0 else None
                    df_part = pd.read_csv(file_path, header=0, nrows=actual_nrows)

                    if not df_part.empty:
                        battery_data_dict[battery_id] = {
                            'X': df_part,
                            'y': label_value
                        }
                    else:
                        self.logger.warning(f"Battery '{battery_id}' CSV is empty. Skipped.")

                except Exception as e:
                    self.logger.error(f"Error reading CSV {file_path}: {e}")

            if not battery_data_dict:
                self.logger.warning(f"Dataset '{dataset_id}' yielded no valid data. Skipped.")
                continue

            self.logger.info(f"Loaded dataset '{dataset_id}' with {len(battery_data_dict)} samples.")
            yield dataset_id, battery_data_dict


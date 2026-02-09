import numpy as np
import pandas as pd
import logging
import re
from typing import Dict, List, Optional

class MissingValueImputer:
    """
    Missing Value Imputer.
    Responsible for filling NaNs in DataFrames based on missing counts and dataset-specific rules.
    """
    def __init__(self):
        self.logger = logging.getLogger("BatteryFeatureProject.Imputer")
        self.battery_data_dict = {} # Reference to all currently processed battery data
        self.dataset_name = ""
        self.global_means = {}      # Global mean cache {col: mean}

    def process(self, battery_data_dict: Dict[str, pd.DataFrame], dataset_name: str) -> Dict[str, pd.DataFrame]:
        """
        Execute the main imputation workflow.
        """
        self.logger.info(f"Starting missing value imputation (Dataset: {dataset_name})...")
        self.battery_data_dict = battery_data_dict
        self.dataset_name = dataset_name

        # 1. Compute global means (as a final fallback strategy)
        self._compute_global_stats()

        imputed_dict = {}
        processed_count = 0

        # 2. Iterate through each battery
        for bat_id, df in battery_data_dict.items():
            # Copy to avoid modifying the original reference
            df_imputed = df.copy()
            has_change = False

            # 3. Iterate through each column
            for col in df_imputed.columns:
                # Process numeric columns only
                if not pd.api.types.is_numeric_dtype(df_imputed[col]):
                    continue

                # Check if all values are empty (if so, nan_count == len(df))
                nan_count = df_imputed[col].isna().sum()
                if nan_count == 0:
                    continue

                # Rule (1): Missing values < 10 -> Linear interpolation
                if nan_count < 10:
                    # limit_direction='both' ensures NaNs at the start and end are also filled
                    df_imputed[col] = df_imputed[col].interpolate(method='linear', limit_direction='both')
                    has_change = True

                # Rule (2): Missing values >= 10 -> Population mean imputation
                else:
                    self._apply_population_imputation(df_imputed, col, bat_id)
                    has_change = True

            imputed_dict[bat_id] = df_imputed
            if has_change:
                processed_count += 1

        self.logger.info(f"Missing value imputation completed, {processed_count} batteries processed.")
        return imputed_dict

    def _compute_global_stats(self):
        """Compute column means (Global Mean) for all batteries in the current dataset."""
        try:
            all_dfs = list(self.battery_data_dict.values())
            if not all_dfs:
                return

            # Select numeric columns only
            sample_df = pd.concat(all_dfs, ignore_index=True, sort=False)
            self.global_means = sample_df.mean(numeric_only=True).to_dict()
        except Exception as e:
            self.logger.warning(f"Failed to compute global means: {e}, skipping global fallback imputation.")
            self.global_means = {}

    def _apply_population_imputation(self, df: pd.DataFrame, col: str, bat_id: str):
        """
        Apply population imputation logic.
        Routing order: SNL specific rules -> ISU specific rules -> Global mean fallback.
        """
        fill_value = None

        ds_upper = self.dataset_name.upper()

        if ds_upper.startswith('SNL'):
            fill_value = self._get_snl_fill_value(col, bat_id)
        elif ds_upper.startswith('ISU') or 'ILCC' in ds_upper:
            fill_value = self._get_isu_fill_value(col, bat_id)

        # Fallback: If specific rules do not return a value (None or NaN), use global mean
        if fill_value is None or np.isnan(fill_value):
            fill_value = self.global_means.get(col)

        # Execute filling
        if fill_value is not None and not np.isnan(fill_value):
            df[col] = df[col].fillna(fill_value)
        else:
            # Rare case: Entire column is NaN globally, cannot impute, fill with 0 to prevent downstream crashes
            df[col] = df[col].fillna(0)

    def _get_snl_fill_value(self, col: str, bat_id: str) -> Optional[float]:
        """
        Handle specific imputation logic for the SNL dataset.
        Rule source: need_fixed.md (3), (4)
        """
        col_lower = col.lower()

        # Rule (3): Missing temperature
        # Broadly match all temperature-related features
        is_temp_col = any(k in col_lower for k in ['temp', 't_rise', 'mint', 'maxt', 'mat_', 'met_'])

        if is_temp_col:
            # Logic: Look for other batteries in the same group (a/b/c/d)
            # Typical filename format: SNL_18650_LFP_35C_0-100_0.5-1C_c
            # Attempt to remove the last underscore suffix
            prefix_match = re.match(r'(.+)_[a-zA-Z0-9]+$', bat_id)
            if prefix_match:
                group_prefix = prefix_match.group(1)

                # Search for other batteries in the same group (startswith prefix)
                neighbor_values = []
                for other_id, other_df in self.battery_data_dict.items():
                    if other_id != bat_id and other_id.startswith(group_prefix):
                        # Get the mean of this column for the other battery (nan if entirely empty)
                        if col in other_df.columns:
                            val = other_df[col].mean()
                            if not np.isnan(val):
                                neighbor_values.append(val)

                if neighbor_values:
                    return np.mean(neighbor_values)

        # Rule (4): NCA 25C DVP missing -> Impute with 0-100 group
        is_dvp_col = 'dvp' in col_lower

        if is_dvp_col and 'NCA' in bat_id and '25C' in bat_id and '20-80' in bat_id:
            # Construct target pattern: replace '20-80' with '0-100'
            target_pattern = bat_id.replace('20-80', '0-100')
            # Remove suffix (to find the mean of the group a/b/c/d)
            prefix_match = re.match(r'(.+)_[a-zA-Z0-9]+$', target_pattern)
            if prefix_match:
                target_group_prefix = prefix_match.group(1)

                neighbor_values = []
                for other_id, other_df in self.battery_data_dict.items():
                    # If it contains the target prefix (belongs to the 0-100 group)
                    if target_group_prefix in other_id:
                        if col in other_df.columns:
                            val = other_df[col].mean()
                            if not np.isnan(val):
                                neighbor_values.append(val)

                if neighbor_values:
                    return np.mean(neighbor_values)

        return None

    def _get_isu_fill_value(self, col: str, bat_id: str) -> Optional[float]:
        """
        Handle specific imputation logic for the ISU-ILCC dataset.
        Rule source: need_fixed.md (5) + new G49C3 requirement
        """
        # Extract Group number from ID
        # Assume formats like "G57C1", "I1_G57_C1", searching for pattern "G(\d+)"
        match = re.search(r'G(\d+)', bat_id)
        if not match:
            return None

        current_num = int(match.group(1))

        # --- Strategy A: Intra-Group Imputation [for G49C3] ---
        # Prioritize looking for other batteries in the same group (same G number)
        intra_group_vals = []
        g_pattern = f"G{current_num}"

        for other_id, other_df in self.battery_data_dict.items():
            if other_id == bat_id:
                continue

            # Check if it contains the same Gxx marker
            if g_pattern in other_id:
                if col in other_df.columns:
                    val = other_df[col].mean()
                    if not np.isnan(val):
                        intra_group_vals.append(val)

        if intra_group_vals:
            # Return only if intra-group has valid values
            return np.mean(intra_group_vals)

        # --- Strategy B: Inter-Group Neighbor Imputation [for G57] ---
        # If the group is entirely empty, search G ± 5
        search_range = range(current_num - 5, current_num + 6) # [current-5, current+5]

        neighbor_values = []
        for other_id, other_df in self.battery_data_dict.items():
            if other_id == bat_id:
                continue

            # Extract Group number of other batteries
            other_match = re.search(r'G(\d+)', other_id)
            if other_match:
                other_num = int(other_match.group(1))
                # Exclude own group (already checked in Strategy A, but this is a range check)
                if other_num != current_num and other_num in search_range:
                    if col in other_df.columns:
                        val = other_df[col].mean()
                        if not np.isnan(val):
                            neighbor_values.append(val)

        if neighbor_values:
            return np.mean(neighbor_values)

        return None

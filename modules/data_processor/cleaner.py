import logging
import pandas as pd
import numpy as np
from typing import Tuple, List, Optional, Dict, Any

class SingleBatteryCleaner:
    """
    Responsible for cleaning and preprocessing data for a single battery.
    Includes: Median filter outlier detection and repair, relative value calculation.
    """
    def __init__(self):
        self.logger = logging.getLogger("BatteryFeatureProject.SingleBatteryCleaner")
        # These columns do not have relative values calculated
        self.skip_relative_cols = [
            'rate', 'charge_rate', 'discharge_rate',
            'uvp', 'lvp',
            'peak_mode', 'dvp_type',
            'soc', 'depth_of_discharge', 'dod',
            'workload_type', 'icp_is_missing', 'ichv', 'idv',
            'ambient_temperature'
        ]

    def _apply_median_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Outlier detection and repair: median filtering.
        If abs(val - median) > 0.2 * median, it is considered an anomaly and repaired with linear interpolation.
        If the column contains 0 or consecutive 0s, skip detection.
        """
        df_cleaned = df.copy()

        # Ensure data types are numeric
        cols_to_process = df_cleaned.select_dtypes(include=[np.number]).columns

        window = 5

        for col in cols_to_process:
            series = df_cleaned[col]

            # Check if 0 exists (physical zero protection, or the data itself contains 0)
            if (series == 0).any():
                continue

            # Calculate rolling median
            rolling_median = series.rolling(window=window, center=True, min_periods=1).median()

            # Detect anomalies
            threshold = rolling_median.abs() * 0.2
            diff = (series - rolling_median).abs()

            is_outlier = diff > threshold

            if is_outlier.any():
                # Set outliers to NaN
                df_cleaned.loc[is_outlier, col] = np.nan
                # Linear interpolation
                df_cleaned[col] = df_cleaned[col].interpolate(method='linear', limit_direction='both')

        return df_cleaned

    def _calculate_relative_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate relative values: subtract data from the 10th cycle.
        """
        if len(df) < 10:
            # Log warning but continue, as some batteries may have very short lives or data segments
            # self.logger.warning(f"Battery data has fewer than 10 rows ({len(df)}), skipping relative value calculation.")
            return df

        # Index 9 is the 10th row
        baseline = df.iloc[9]

        df_rel = df.copy()

        for col in df.columns:
            # Check if in skip list
            col_lower = col.lower()
            if any(skip in col_lower for skip in self.skip_relative_cols):
                continue

            if pd.api.types.is_numeric_dtype(df[col]):
                df_rel[col] = df[col] - baseline[col]

        return df_rel

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Execute single battery cleaning workflow."""
        # 1. Outlier cleaning
        df = self._apply_median_filter(df)

        # 2. Relative value calculation
        df = self._calculate_relative_values(df)

        return df


class DatasetCleaner:
    """
    Responsible for cleaning the entire dataset (after aggregating all batteries).
    Includes: constant removal, all-empty removal, high missing rate removal.
    """
    def __init__(self):
        self.logger = logging.getLogger("BatteryFeatureProject.DatasetCleaner")
        self.dropped_info = {}

    def process(self, df: pd.DataFrame, manual_drop_cols: List[str] = None) -> pd.DataFrame:
        self.logger.info(f"Starting dataset aggregation cleaning. Initial shape: {df.shape}")
        self.dropped_info = {}

        # 0. Manual column cleaning
        if manual_drop_cols:
            existing_cols = [c for c in manual_drop_cols if c in df.columns]
            if existing_cols:
                df = df.drop(columns=existing_cols)
                self.dropped_info['manual_drop'] = existing_cols
                self.logger.info(f"Manually removed {len(existing_cols)} specified columns: {existing_cols}")

            missing_cols = set(manual_drop_cols) - set(existing_cols)
            if missing_cols:
                self.logger.warning(f"The following columns were not found in the manual removal list: {missing_cols}")

        # 1. Delete columns with standard deviation (std) of 0
        stds = df.std(numeric_only=True)
        constant_cols = stds[stds == 0].index.tolist()

        if constant_cols:
            df = df.drop(columns=constant_cols)
            self.dropped_info['constant'] = constant_cols
            self.logger.info(f"Removed {len(constant_cols)} constant columns (std=0).")

        # 2. Delete columns that are all NaN
        initial_cols = df.columns
        df = df.dropna(axis=1, how='all')
        all_nan_cols = list(set(initial_cols) - set(df.columns))
        if all_nan_cols:
            self.dropped_info['all_nan'] = all_nan_cols
            self.logger.info(f"Removed {len(all_nan_cols)} all-NaN columns.")

        # 3. High missing rate cleaning (> 20%)
        missing_rates = df.isnull().mean()
        high_missing_cols = missing_rates[missing_rates > 0.2].index.tolist()

        if high_missing_cols:
            df = df.drop(columns=high_missing_cols)
            self.dropped_info['high_missing'] = high_missing_cols
            self.logger.info(f"Removed {len(high_missing_cols)} high missing rate columns (>20%).")

        self.logger.info(f"Dataset cleaning completed. Final shape: {df.shape}")
        return df

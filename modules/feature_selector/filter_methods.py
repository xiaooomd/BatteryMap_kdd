"""Filter method selectors for feature engineering.

This module provides filter-based feature selection methods, including:
- Correlation-based filters (Pearson, Spearman, Kendall)
- Mutual Information filter

It adheres to the Strategy Pattern, allowing different selection algorithms
to be interchangeable.
"""

import logging
import os
from abc import ABC, abstractmethod
from typing import List, Set, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression


class BaseFilter(ABC):
    """Abstract base class for filter-based feature selectors.

    Attributes:
        mode (int): Selection mode.
            0: Feature vs Feature (Collinearity removal).
            1: Feature vs Target (Relevance selection).
        threshold (float): Selection threshold value.
        selected_features_ (List[str]): List of selected feature names.
        drop_report_ (pd.DataFrame): Detailed report of dropped features (only for mode 0).
        logger (logging.Logger): Logger instance.
    """

    def __init__(self, mode: int = 0, threshold: float = 0.95):
        """Initialize the BaseFilter.

        Args:
            mode: Selection mode (0 or 1).
            threshold: Threshold for selection.

        Raises:
            ValueError: If mode is not 0 or 1.
        """
        if mode not in [0, 1]:
            raise ValueError("Mode must be 0 (feature_vs_feature) or 1 (feature_vs_target).")

        self.mode = mode
        self.threshold = threshold
        self.logger = logging.getLogger(f"FeatureSelection.{self.__class__.__name__}")
        self.selected_features_: List[str] = []
        self.drop_report_: pd.DataFrame = pd.DataFrame()

    @abstractmethod
    def select(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """Execute feature selection logic.

        Args:
            X: Feature DataFrame.
            y: Target Series.

        Returns:
            List of selected feature names.
        """
        raise NotImplementedError("Subclasses must implement the select method.")

    def save_report(self, output_dir: str, file_name: str = "filter_drop_report.csv") -> None:
        """Save the dropped features report to a CSV file.

        Args:
            output_dir: Directory to save the report.
            file_name: Name of the CSV file.
        """
        if self.drop_report_.empty:
            return

        try:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, file_name)
            self.drop_report_.to_csv(output_path, index=False)
            self.logger.info(f"Feature drop report saved to: {output_path}")
        except OSError as e:
            self.logger.error(f"Failed to save report: {e}")


class CorrelationFilter(BaseFilter):
    """Unified filter based on correlation, supporting physics-based priority retention."""

    CORR_METHOD = ''  # Subclasses must define 'pearson', 'spearman', or 'kendall'

    def select(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """Execute correlation-based selection.

        Args:
            X: Feature DataFrame.
            y: Target Series.

        Returns:
            List of selected feature names.

        Raises:
            ValueError: If CORR_METHOD is not defined.
        """
        if not self.CORR_METHOD:
            raise ValueError("CorrelationFilter cannot be used directly. Use a subclass (e.g., PearsonFilter).")

        if self.mode == 0:
            return self._select_feature_vs_feature(X, y)
        else:
            return self._select_feature_vs_target(X, y)

    def _get_priority(self, feature_name: str) -> int:
        """Get the physical priority of a feature (Level 1-5).

        Lower number indicates higher priority. Based on code_rules.md 1.3.2.

        Args:
            feature_name: Name of the feature.

        Returns:
            Priority level (1-5).
        """
        f_lower = feature_name.lower()

        # Level 2 (Morphological Features) - Prioritized to avoid ICP matching ICP_Area
        l2_keywords = [
            'icp_area', 'dvp_area', 'icp_fwhm', 'centroid_voltage',
            't_rise_discharge', 'dtp', 'heatrate', 'ichv', 'idv', 'v_rest_end',
            'uvp_time', 'lvp_time', 'charge_slope_', 'discharge_slope_', 'tevi_', 'tevd_'
        ]
        if any(x in f_lower for x in l2_keywords):
            return 2

        # Level 1 (Core Mechanism)
        l1_keywords = [
            'icpl_v', 'dvpl_v', 'icvl_v', 'dvvl_v',
            'icp', 'dvp', 'icv', 'dvv',
            'internal_resistance', 'cv_current_tau', 'rcv', 'coulombic_efficiency',
            'ambient_temperature'
        ]
        if any(x in f_lower for x in l1_keywords):
            return 1

        # Level 3 (Statistical Features)
        l3_keywords = ['skew_', 'kurtosis_', 'var_', 'median_']
        if any(x in f_lower for x in l3_keywords):
            return 3

        # Level 4 (Result Features)
        l4_keywords = [
            'discharge_capacity', 'charge_capacity',
            'discharge_energy', 'charge_energy'
        ]
        if any(x in f_lower for x in l4_keywords):
            return 4

        # Default lowest priority (Level 5)
        return 5

    def _select_feature_vs_feature(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """Mode 0: Remove collinear features based on physical priority."""
        self.logger.info(f"Applying {self.CORR_METHOD} collinearity filter (mode 0), threshold={self.threshold}...")

        # Calculate correlation matrix
        corr_matrix = X.corr(method=self.CORR_METHOD).abs()
        # Get upper triangle (excluding diagonal)
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        to_drop: Set[str] = set()
        drop_records = []

        # Pre-calculate correlation with target as tie-breaker
        corr_with_target = X.corrwith(y).abs().fillna(0)

        cols = upper.columns
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                col_i = cols[i]
                col_j = cols[j]

                # Check if correlation exceeds threshold
                corr_val = upper.iloc[i, j]
                if corr_val > self.threshold:
                    # 1. Compare physical priority
                    prio_i = self._get_priority(str(col_i))
                    prio_j = self._get_priority(str(col_j))

                    reason = ""
                    loser = ""
                    winner = ""

                    if prio_i < prio_j:
                        # i has higher priority (lower level), keep i
                        loser, winner = str(col_j), str(col_i)
                        reason = f"Priority Level {prio_i} vs {prio_j}"
                    elif prio_j < prio_i:
                        # j has higher priority, keep j
                        loser, winner = str(col_i), str(col_j)
                        reason = f"Priority Level {prio_j} vs {prio_i}"
                    else:
                        # Priority tie, compare correlation with target
                        score_i = corr_with_target[col_i]
                        score_j = corr_with_target[col_j]

                        if score_i < score_j:
                            loser, winner = str(col_i), str(col_j)
                            reason = "Target Correlation"
                        else:
                            loser, winner = str(col_j), str(col_i)
                            reason = "Target Correlation"

                    # Record the drop
                    if loser not in to_drop:
                        to_drop.add(loser)
                        drop_records.append({
                            'dropped_feature': loser,
                            'kept_substitute': winner,
                            'correlation_between_features': corr_val,
                            'drop_reason': reason,
                            'method': self.CORR_METHOD
                        })

        # Generate report
        self.drop_report_ = pd.DataFrame(drop_records)
        if not self.drop_report_.empty:
            self.drop_report_ = self.drop_report_.sort_values(
                by='correlation_between_features', ascending=False
            )

        if to_drop:
            self.logger.info(f"Found {len(to_drop)} highly collinear features to drop.")

        self.selected_features_ = X.columns.drop(list(to_drop)).tolist()
        return self.selected_features_

    def _select_feature_vs_target(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """Mode 1: Keep features strongly correlated with target."""
        self.logger.info(f"Applying {self.CORR_METHOD} correlation filter (mode 1), threshold={self.threshold}...")
        correlations = X.corrwith(y, method=self.CORR_METHOD).abs()
        selected = correlations[correlations >= self.threshold]

        self.selected_features_ = selected.index.tolist()
        self.logger.info(f"Found {len(self.selected_features_)} features with target correlation >= {self.threshold}.")
        return self.selected_features_


class PearsonFilter(CorrelationFilter):
    """Pearson correlation filter."""
    CORR_METHOD = 'pearson'


class SpearmanFilter(CorrelationFilter):
    """Spearman rank correlation filter."""
    CORR_METHOD = 'spearman'


class KendallFilter(CorrelationFilter):
    """Kendall rank correlation filter."""
    CORR_METHOD = 'kendall'


class MutualInfoFilter(BaseFilter):
    """Mutual Information filter. Only supports 'Feature vs Target' mode (mode=1)."""

    def __init__(self, mode: int = 1, threshold: float = 0.1):
        """Initialize MutualInfoFilter.

        Forces mode to 1 as Mutual Information is irrelevant for collinearity removal in this context.
        """
        super().__init__(mode=mode, threshold=threshold)
        if self.mode == 0:
            self.logger.warning("MutualInfoFilter does not support mode=0. Automatically switched to mode=1.")
            self.mode = 1

    def select(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """Execute Mutual Information selection."""
        self.logger.info(f"Applying Mutual Information filter (mode 1), threshold={self.threshold}...")
        # Fill NaN to prevent errors
        X_clean = X.fillna(0)
        mi_scores = mutual_info_regression(X_clean, y)
        mi_series = pd.Series(mi_scores, index=X.columns)

        selected = mi_series[mi_series >= self.threshold]
        self.selected_features_ = selected.index.tolist()
        self.logger.info(f"Found {len(self.selected_features_)} features with MI >= {self.threshold}.")
        return self.selected_features_


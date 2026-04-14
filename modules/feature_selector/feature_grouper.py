import logging
import re
from typing import Dict, List

class FeatureGrouper:
    """
    Responsible for grouping features into physical categories.
    Follows the principle of physical meaning priority.
    """
    def __init__(self):
        self.logger = logging.getLogger("FeatureSelection.FeatureGrouper")

        # Define deterministic grouping lists (Code Rules 1.3)
        # Note: Case-insensitive, convert to lowercase during comparison
        self.exact_rules = {
            'Energy': {
                'discharge_capacity', 'charge_capacity', 'discharge_energy',
                'charge_energy', 'coulombic_efficiency', 'energy_efficiency'
            },
            'Kinetics': {
                'internal_resistance', 'cv_current_tau', 'tccc', 'tcvc', 'rcv',
                'charge_c_rate', 'discharge_c_rate', 'rest_time', 'total_discharge_time',
                'uvp_time', 'lvp_time', 'v_rest_end', 'charge_time_ratio_1_2',
                'charge_time_1', 'charge_time_2', 'charge_time_3',
                'discharge_time_1', 'discharge_time_2', 'discharge_time_3', 'discharge_time_4'
            },
            'Thermodynamics': {
                'dtp', 'dtpl_v', 'mat_charge', 'mat_discharge', 'met_charge', 'met_discharge',
                'mint_charge', 'mint_discharge', 't_rise_charge', 't_rise_discharge',
                'thermal_load_charge', 'thermal_load_discharge', 'skew_t_discharge',
                'temperature', 'heatrate', 'ambient_temperature'
            },
            'Curve': {
                'icp', 'icpl_v', 'icp_area', 'icp_fwhm', 'icv', 'icvl_v',
                'dvp', 'dvpl_v', 'dvp_q', 'dvv', 'dvvl_v', 'dvv_q',
                'dvp_fwhm', 'dvp_area', 'centroid_voltage',
                'ratio_peak1_peak3', 'v_diff_peak3_peak1'
            },
            'Geometric': {
                'skew_v_discharge', 'var_i_charge', 'max_i_charge', 'var_i_discharge',
                'var_v_discharge', 'median_v_discharge',
                'charge_slope_1', 'charge_slope_2', 'charge_slope_3',
                'discharge_slope_1', 'discharge_slope_2', 'discharge_slope_3',
                'tevi_1', 'tevi_2', 'tevi_3', 'tevd_1', 'tevd_2', 'tevd_3',
                'charge_current_1', 'charge_current_2', 'charge_current_3',
                'discharge_current_1', 'discharge_current_2', 'discharge_current_3', 'discharge_current_4'
            },
            'Metadata': {
                'workload_type', 'icp_is_missing', 'dvp_type', 'peak_mode',
                'ichv', 'idv', 'uvp', 'lvp', 'soc'
            }
        }

        # Fuzzy matching rules (Fallback)
        self.fuzzy_rules = {
            'Energy': ['capacity', 'energy'],
            'Kinetics': ['resistance', 'tau', 'time', 'rate', 'current_tau'],
            'Thermodynamics': ['temperature', 'heat', 't_rise', 'temp'],
            'Curve': ['ic', 'dv', 'peak', 'area'],
            'Geometric': ['slope', 'skewness', 'kurtosis', 'var_', 'mean_', 'median_']
        }

    def _normalize_feature_name(self, feature_name: str) -> str:
        normalized = str(feature_name).strip().lower()
        # Remove unit-like suffixes, e.g. RCV(V) -> rcv.
        normalized = re.sub(r"\([^)]*\)", "", normalized)
        normalized = re.sub(r"[\s\-/]+", "_", normalized)
        normalized = re.sub(r"_+", "_", normalized).strip("_")
        return normalized

    def group_features(self, features: List[str]) -> Dict[str, List[str]]:
        """
        Group features by feature names.
        Logic:
        1. Exclude Cycle_Number
        2. Exact match
        3. Fuzzy match (by priority)
        """
        self.logger.info("Starting feature grouping...")

        # Initialize groups
        self.groups = {
            'Energy': [],
            'Kinetics': [],
            'Thermodynamics': [],
            'Curve': [],
            'Geometric': [], # Geometric & Statistical
            'Metadata': []
        }

        # Track ungrouped features
        ungrouped = []

        for f in features:
            f_lower = f.lower()
            f_normalized = self._normalize_feature_name(f)

            # 1. Exclude Cycle_Number
            if f_normalized == 'cycle_number':
                continue

            assigned = False

            # 2. Exact match
            for group_name, exact_set in self.exact_rules.items():
                if f_normalized in exact_set:
                    self.groups[group_name].append(f)
                    assigned = True
                    break

            if assigned:
                continue

            # 3. Fuzzy match (Fallback)
            # Priority order: Thermodynamics -> Energy -> Kinetics -> Curve -> Geometric -> Metadata

            # Thermodynamics
            if any(k in f_normalized for k in self.fuzzy_rules['Thermodynamics']):
                self.groups['Thermodynamics'].append(f)
                assigned = True

            # Energy
            elif not assigned and any(k in f_normalized for k in self.fuzzy_rules['Energy']):
                self.groups['Energy'].append(f)
                assigned = True

            # Kinetics
            elif not assigned and any(k in f_normalized for k in self.fuzzy_rules['Kinetics']):
                self.groups['Kinetics'].append(f)
                assigned = True

            # Curve
            elif not assigned and any(k in f_normalized for k in self.fuzzy_rules['Curve']):
                self.groups['Curve'].append(f)
                assigned = True

            # Geometric
            elif not assigned and any(k in f_normalized for k in self.fuzzy_rules['Geometric']):
                self.groups['Geometric'].append(f)
                assigned = True

            # Fallback: if no match, tentatively assign to Curve (shape features) or Geometric?
            # Based on past experience, unmatched features are mostly shape parameters
            elif not assigned:
                self.logger.warning(f"Feature '{f}' did not match any rule, defaulting to Geometric")
                self.groups['Geometric'].append(f)

        # Remove empty groups
        self.groups = {k: v for k, v in self.groups.items() if v}

        self.logger.info(f"Feature grouping completed: { {k: len(v) for k, v in self.groups.items()} }")
        return self.groups

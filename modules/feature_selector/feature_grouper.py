import logging
import re
from typing import Dict, List

class FeatureGrouper:
    """
    负责将特征归类到物理组别中。
    遵循物理意义优先的原则。
    """
    def __init__(self):
        self.logger = logging.getLogger("FeatureSelection.FeatureGrouper")

        # 定义确定的分组列表 (Code Rules 1.3)
        # 注意：不区分大小写，后续比较时统一转小写
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

        # 模糊匹配规则 (Fallback)
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
        根据特征名称进行分组。
        逻辑：
        1. 排除 Cycle_Number
        2. 精确匹配
        3. 模糊匹配 (按优先级)
        """
        self.logger.info("开始特征分组...")

        # 初始化分组
        self.groups = {
            'Energy': [],
            'Kinetics': [],
            'Thermodynamics': [],
            'Curve': [],
            'Geometric': [], # Geometric & Statistical
            'Metadata': []
        }

        # 记录未分类特征
        ungrouped = []

        for f in features:
            f_lower = f.lower()
            f_normalized = self._normalize_feature_name(f)

            # 1. 排除 Cycle_Number
            if f_normalized == 'cycle_number':
                continue

            assigned = False

            # 2. 精确匹配
            for group_name, exact_set in self.exact_rules.items():
                if f_normalized in exact_set:
                    self.groups[group_name].append(f)
                    assigned = True
                    break

            if assigned:
                continue

            # 3. 模糊匹配 (Fallback)
            # 优先级顺序：Thermodynamics -> Energy -> Kinetics -> Curve -> Geometric -> Metadata

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

            # 兜底：如果没有匹配上，先暂存到 Curve (形状特征) 或者 Geometric?
            # 根据过往经验，未匹配的大多是形状参数
            elif not assigned:
                self.logger.warning(f"特征 '{f}' 未匹配任何规则，默认归类为 Geometric")
                self.groups['Geometric'].append(f)

        # 移除空组
        self.groups = {k: v for k, v in self.groups.items() if v}

        self.logger.info(f"特征分组完成: { {k: len(v) for k, v in self.groups.items()} }")
        return self.groups

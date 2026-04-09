"""
Feature Extraction Script for XJTU Battery Dataset.
"""

import argparse
import warnings
import sys
from pathlib import Path
from typing import Any, Dict

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from features.base_extractor import BaseFeatureExtractor, resolve_dataset_input_dir
from src.physics.ic_dv_extractor import extract_ic_dv_features

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

CV_VOLTAGE_TOLERANCE_V = 0.05
CC_CURRENT_DROP_RATIO = 0.95


class XJTUFeatureExtractor(BaseFeatureExtractor):
    CHARGE_SLOPES = [(0.1, 0.3), (0.1, 0.5), (0.1, 0.8)]
    DISCHARGE_SLOPES = [(0.1, 0.3), (0.1, 0.5), (0.1, 0.8)]
    TEVI_INTERVALS = [(3.6, 3.7), (3.7, 3.9), (3.9, 4.1)]
    TEVD_INTERVALS = [(4.1, 3.9), (3.9, 3.7), (3.7, 3.5)]

    def extract_cycle_features(
        self,
        cycle_data: Dict[str, Any],
        battery_data: Dict[str, Any],
        output_dir: Path,
    ) -> Dict[str, Any]:
        cycle_df = self.build_cycle_frame(cycle_data)
        cycle_num = cycle_data.get("cycle_number", 0)
        charge_df, discharge_df, _ = self.split_phase_frames(cycle_df)
        discharge_df = self.trim_discharge_to_voltage_limit(
            discharge_df,
            battery_data.get("min_voltage_limit_in_V"),
        )
        rest_df = self._extract_rest_segment(cycle_df, charge_df, discharge_df)

        direct_features = self.calculate_capacity_energy_features(cycle_num, charge_df, discharge_df)
        direct_features["Rest_Time"] = self._calculate_rest_time(cycle_df, charge_df, discharge_df)
        direct_features["charge_c_rate"] = self.get_protocol_rate(battery_data.get("charge_protocol"))
        direct_features["discharge_c_rate"] = self.get_protocol_rate(battery_data.get("discharge_protocol"))
        direct_features.update(
            self.calculate_charge_phase_features(
                charge_df,
                uvp=battery_data.get("max_voltage_limit_in_V", 0.0),
                time_mode="absolute",
                cv_voltage_tolerance=CV_VOLTAGE_TOLERANCE_V,
                cv_current_drop_ratio=CC_CURRENT_DROP_RATIO,
            )
        )
        direct_features.update(
            self.calculate_discharge_phase_features(
                discharge_df,
                lvp=battery_data.get("min_voltage_limit_in_V", 0.0),
                time_mode="absolute",
            )
        )

        nominal_capacity = battery_data.get("nominal_capacity_in_Ah", 2.0)
        derivative_features = extract_ic_dv_features(
            discharge_df,
            config={
                "peak_mode": 1,
                "nominal_capacity": nominal_capacity,
                "window_length_ic": 21,
                "window_length_dv": 21,
                "peak_height_ic": 0.01,
                "voltage_range_ic": (3.3, 4.2),
                "prominence_ic": 0.02,
                "ic_step_size": 0.01,
                "dv_step_size": nominal_capacity * 0.005,
                "search_window_dvv": 0.1,
                "search_window_dvp": 0.1,
                "initial_capacity_cut_fraction": 0.02,
                "icv_search_offset_lower": 0.05,
                "icv_search_offset_upper": 0.5,
            },
            plot_params=self.build_plot_params(battery_data, cycle_num, output_dir),
        )
        advanced_features = self.calculate_advanced_features_common(
            cycle_df,
            charge_df,
            discharge_df,
            direct_features,
            rest_df=rest_df,
            compute_cv_tau=True,
            cv_voltage_tolerance=CV_VOLTAGE_TOLERANCE_V,
            cv_current_threshold=0.01,
        )
        anchor_features = self.calculate_anchor_features_common(
            charge_df,
            discharge_df,
            self.CHARGE_SLOPES,
            self.DISCHARGE_SLOPES,
            self.TEVI_INTERVALS,
            self.TEVD_INTERVALS,
        )
        return {**direct_features, **derivative_features, **advanced_features, **anchor_features}

    def _calculate_rest_time(
        self,
        cycle_df,
        charge_df,
        discharge_df,
    ) -> float:
        if charge_df.empty or discharge_df.empty:
            return 0.0
        last_charge_idx = charge_df.index[-1]
        first_discharge_idx = discharge_df.index[0]
        if first_discharge_idx <= last_charge_idx + 1:
            return 0.0
        rest_segment = cycle_df.loc[last_charge_idx + 1:first_discharge_idx - 1]
        if rest_segment.empty:
            return 0.0
        return max(0.0, rest_segment["Time(s)"].max() - rest_segment["Time(s)"].min())

    def _extract_rest_segment(self, cycle_df, charge_df, discharge_df):
        if charge_df.empty or discharge_df.empty:
            return cycle_df.iloc[0:0].copy()
        last_charge_idx = charge_df.index[-1]
        first_discharge_idx = discharge_df.index[0]
        if first_discharge_idx <= last_charge_idx + 1:
            return cycle_df.iloc[0:0].copy()
        return cycle_df.loc[last_charge_idx + 1:first_discharge_idx - 1].copy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_cycles", type=int, default=None, help="Number of cycles to process (default: all)")
    args = parser.parse_args()
    input_dir = resolve_dataset_input_dir(project_root, "XJTU", "data/XJTU")
    output_dir = project_root / "results" / "features" / "XJTU"
    XJTUFeatureExtractor().run(input_dir, output_dir, num_cycles=args.num_cycles)



if __name__ == "__main__":
    main()

"""
Feature Extraction Script for RWTH Battery Dataset.
"""

import argparse
import warnings
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from features.base_extractor import BaseFeatureExtractor, resolve_dataset_input_dir
from src.physics.ic_dv_extractor import extract_ic_dv_features

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class RWTHFeatureExtractor(BaseFeatureExtractor):
    CHARGE_SLOPES = [(0.1, 0.3), (0.1, 0.5), (0.1, 0.8)]
    DISCHARGE_SLOPES = [(0.1, 0.3), (0.1, 0.5), (0.1, 0.8)]
    TEVI_INTERVALS = [(3.55, 3.7), (3.7, 3.85), (3.55, 3.85)]
    TEVD_INTERVALS = [(3.85, 3.7), (3.7, 3.55), (3.85, 3.55)]

    def build_cycle_frame(self, cycle_data: Dict[str, Any]) -> pd.DataFrame:
        cycle_df = super().build_cycle_frame(cycle_data)
        cycle_df["Time(s)"] = cycle_df["Time(s)"] / 1000.0
        return cycle_df

    def extract_cycle_features(
        self,
        cycle_data: Dict[str, Any],
        battery_data: Dict[str, Any],
        output_dir: Path,
    ) -> Dict[str, Any]:
        cycle_df = self.build_cycle_frame(cycle_data)
        cycle_num = cycle_data.get("cycle_number", 0)
        charge_df, discharge_df, _ = self.split_phase_frames(cycle_df)

        direct_features = self.calculate_capacity_energy_features(cycle_num, charge_df, discharge_df)
        direct_features["charge_current"] = battery_data.get("charge_protocol", [{}])[0].get("current_in_A", 0)
        direct_features["discharge_current"] = battery_data.get("discharge_protocol", [{}])[0].get("current_in_A", 0)
        direct_features.update(
            self.calculate_charge_phase_features(
                charge_df,
                uvp=battery_data.get("max_voltage_limit_in_V", 0.0),
                time_mode="duration",
                cv_voltage_tolerance=0.01,
            )
        )
        direct_features.update(
            self.calculate_discharge_phase_features(
                discharge_df,
                lvp=battery_data.get("min_voltage_limit_in_V", 0.0),
                time_mode="duration",
            )
        )

        derivative_features = extract_ic_dv_features(
            self._build_charge_derivative_frame(charge_df, battery_data),
            config=self._build_derivative_config(battery_data),
            plot_params=self.build_plot_params(battery_data, cycle_num, output_dir),
        )
        advanced_features = self.calculate_advanced_features_common(
            cycle_df,
            charge_df,
            discharge_df,
            direct_features,
            compute_cv_tau=False,
        )
        if not charge_df.empty and not discharge_df.empty:
            v_dis_end = discharge_df["Voltage(V)"].iloc[-1]
            v_chg_start = charge_df["Voltage(V)"].iloc[0]
            i_chg_start = charge_df["Current(A)"].iloc[0]
            advanced_features["Internal_Resistance"] = (
                (v_chg_start - v_dis_end) / i_chg_start if i_chg_start > 0.001 else 0.0
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

    def _build_charge_derivative_frame(
        self,
        charge_df: pd.DataFrame,
        battery_data: Dict[str, Any],
    ) -> pd.DataFrame:
        if charge_df.empty:
            return pd.DataFrame()

        target_current = battery_data.get("charge_protocol", [{}])[0].get("current_in_A", 0)
        if target_current <= 0.001:
            target_current = charge_df["Current(A)"].quantile(0.90)

        voltage_threshold = charge_df["Voltage(V)"].max() - 0.05
        current_threshold = target_current * 0.98
        cv_indices = (
            (charge_df["Voltage(V)"].values >= voltage_threshold) &
            (charge_df["Current(A)"].values < current_threshold)
        ).nonzero()[0]

        cc_charge_df = charge_df.iloc[:cv_indices[0]].copy() if len(cv_indices) > 0 and cv_indices[0] > 10 else charge_df.copy()
        if not cc_charge_df.empty:
            voltage_cut = cc_charge_df["Voltage(V)"].min() + 0.02
            filtered_df = cc_charge_df[cc_charge_df["Voltage(V)"] > voltage_cut]
            if len(filtered_df) > 10:
                cc_charge_df = filtered_df.copy()

        derivative_df = cc_charge_df.copy()
        if "Charge_Capacity(Ah)" in derivative_df.columns:
            derivative_df["Discharge_Capacity(Ah)"] = derivative_df["Charge_Capacity(Ah)"]
        return derivative_df

    def _build_derivative_config(self, battery_data: Dict[str, Any]) -> Dict[str, Any]:
        nominal_capacity = battery_data.get("nominal_capacity_in_Ah", 2.0)
        return {
            "peak_mode": 1,
            "nominal_capacity": nominal_capacity,
            "window_length_ic": 51,
            "window_length_dv": 21,
            "peak_height_ic": 0.01,
            "voltage_range_ic": (3.7, 3.88),
            "prominence_ic": 0.01,
            "ic_step_size": 0.002,
            "dv_step_size": nominal_capacity * 0.005,
            "search_window_dvv": 0.1,
            "search_window_dvp": 0.1,
            "initial_capacity_cut_fraction": 0.02,
            "icv_search_offset_lower": 0.05,
            "icv_search_offset_upper": 0.1,
            "plot_interval": 50,
            "ic_area_voltage_range": (3.75, 3.85),
            "disable_dvv": True,
            "dvp_capacity_range": (0.1, 0.4),
            "dvpl_v_capacity_fraction": 0.5,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_cycles", type=int, default=None, help="Number of cycles to process (default: all)")
    args = parser.parse_args()
    input_dir = resolve_dataset_input_dir(project_root, "RWTH", "data/RWTH")
    output_dir = project_root / "results" / "features" / "RWTH"
    RWTHFeatureExtractor().run(input_dir, output_dir, num_cycles=args.num_cycles)



if __name__ == "__main__":
    main()

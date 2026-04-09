"""
Feature Extraction Script for ZN-ion Battery Dataset.
"""

import argparse
import warnings
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from tqdm import tqdm

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from features.base_extractor import BaseFeatureExtractor, resolve_dataset_input_dir
from src.physics.ic_dv_extractor import extract_ic_dv_features

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class ZNionFeatureExtractor(BaseFeatureExtractor):
    CHARGE_SLOPES = [(0.1, 0.3), (0.1, 0.5), (0.1, 0.8)]
    DISCHARGE_SLOPES = [(0.1, 0.3), (0.1, 0.5), (0.1, 0.8)]
    TEVI_INTERVALS = [(1.0, 1.2), (1.2, 1.4), (1.4, 1.6)]
    TEVD_INTERVALS = [(1.6, 1.4), (1.4, 1.2), (1.2, 1.0)]

    def extract_cycle_features(
        self,
        cycle_data: Dict[str, Any],
        battery_data: Dict[str, Any],
        output_dir: Path,
    ) -> Optional[Dict[str, Any]]:
        cycle_df = self.build_cycle_frame(cycle_data)
        cycle_num = cycle_data.get("cycle_number", 0)
        charge_df, discharge_df, _ = self.split_phase_frames(cycle_df)
        discharge_df = self.trim_discharge_to_voltage_limit(
            discharge_df,
            battery_data.get("min_voltage_limit_in_V"),
        )

        direct_features = self.calculate_capacity_energy_features(cycle_num, charge_df, discharge_df)
        direct_features["Workload_Type"] = self._get_workload_type(charge_df, discharge_df)
        direct_features["charge_c_rate"] = self.get_protocol_rate(battery_data.get("charge_protocol"))
        direct_features["discharge_c_rate"] = self.get_protocol_rate(battery_data.get("discharge_protocol"))
        direct_features.update(
            self.calculate_charge_phase_features(
                charge_df,
                uvp=battery_data.get("max_voltage_limit_in_V", 0.0),
                time_mode="duration",
                force_no_cv=True,
            )
        )
        direct_features.update(
            self.calculate_discharge_phase_features(
                discharge_df,
                lvp=battery_data.get("min_voltage_limit_in_V", 0.0),
                time_mode="duration",
            )
        )

        total_time = (
            direct_features.get("TCCC", 0.0) +
            direct_features.get("TCVC", 0.0) +
            direct_features.get("total_discharge_time", 0.0)
        )
        if total_time > 10000:
            return None

        nominal_capacity = battery_data.get("nominal_capacity_in_Ah", 0.5)
        derivative_features = extract_ic_dv_features(
            discharge_df,
            config={
                "peak_mode": 2,
                "nominal_capacity": nominal_capacity,
                "window_length_ic": 21,
                "window_length_dv": 21,
                "peak_height_ic": 0.0001,
                "voltage_range_ic": (1.0, 1.6),
                "prominence_ic": 0.0001,
                "ic_step_size": 0.005,
                "dv_step_size": nominal_capacity * 0.005,
                "search_window_dvv": 0.3,
                "search_window_dvp": 0.3,
                "initial_capacity_cut_fraction": 0.02,
                "icv_search_offset_lower": 0.01,
                "icv_search_offset_upper": 0.25,
                "icv_search_direction": "left",
            },
            plot_params=self.build_plot_params(battery_data, cycle_num, output_dir),
        )
        advanced_features = self.calculate_advanced_features_common(
            cycle_df,
            charge_df,
            discharge_df,
            direct_features,
            compute_cv_tau=True,
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

    def process_battery(
        self,
        file_path: Path,
        output_dir: Path,
        num_cycles: Optional[int] = None,
    ) -> None:
        try:
            battery_data = self.load_battery_data(file_path)
        except Exception as exc:
            print(f"Error loading {file_path}: {exc}")
            return

        cell_id = self.get_cell_id(battery_data, file_path)
        cycles_to_process = self.get_cycles_to_process(battery_data, None)
        valid_cycle_count = 0
        all_cycle_features = []

        for cycle_data in tqdm(cycles_to_process, desc=f"Processing {cell_id}", leave=False):
            if not cycle_data.get("time_in_s"):
                continue
            try:
                features = self.extract_cycle_features(cycle_data, battery_data, output_dir)
            except Exception as exc:
                self.handle_cycle_error(cell_id, exc)
                continue
            if not features:
                continue
            valid_cycle_count += 1
            features["Cycle_Number"] = valid_cycle_count
            all_cycle_features.append(features)
            if num_cycles is not None and num_cycles > 0 and valid_cycle_count >= num_cycles:
                break

        if not all_cycle_features:
            print(f"Warning: No features extracted for {cell_id}")
            return

        output_file = output_dir / f"{cell_id}.csv"
        self.order_columns(self.build_feature_frame(all_cycle_features)).to_csv(output_file, index=False)
        print(f"Features for {cell_id} saved to {output_file}")

    def _get_workload_type(self, charge_df: pd.DataFrame, discharge_df: pd.DataFrame) -> str:
        if not charge_df.empty and not discharge_df.empty:
            return "0" if charge_df["Time(s)"].iloc[0] < discharge_df["Time(s)"].iloc[0] else "1"
        if not charge_df.empty:
            return "2"
        if not discharge_df.empty:
            return "3"
        return "-1"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_cycles", type=int, default=None, help="Number of cycles to process (default: all)")
    args = parser.parse_args()
    input_dir = resolve_dataset_input_dir(project_root, "ZN-coin", "data/ZN-coin")
    output_dir = project_root / "results" / "features" / "ZN-coin"
    ZNionFeatureExtractor().run(input_dir, output_dir, num_cycles=args.num_cycles)



if __name__ == "__main__":
    main()

"""
Feature Extraction Script for HNEI Battery Dataset.
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

BATTERY_NOMINAL_CAPACITY = 2.8
CHARGE_C_RATE = 2.0
DISCHARGE_C_RATE = 1.0
VOLTAGE_UPPER_LIMIT = 4.3
VOLTAGE_LOWER_LIMIT = 3.0


class HNEIFeatureExtractor(BaseFeatureExtractor):
    CHARGE_SLOPES = [(0.10, 0.30), (0.10, 0.50), (0.10, 0.80)]
    DISCHARGE_SLOPES = [(0.10, 0.30), (0.10, 0.50), (0.10, 0.80)]
    TEVI_INTERVALS = [(3.5, 3.8), (3.8, 4.1), (4.1, 4.25)]
    TEVD_INTERVALS = [(4.2, 3.9), (3.9, 3.6), (3.6, 3.3)]

    def extract_cycle_features(
        self,
        cycle_data: Dict[str, Any],
        battery_data: Dict[str, Any],
        output_dir: Path,
    ) -> Dict[str, Any]:
        cycle_df = self.build_cycle_frame(cycle_data)
        cycle_num = cycle_data.get("cycle_number", 0)
        charge_df, discharge_df, _ = self.split_phase_frames(cycle_df)
        discharge_df = self.trim_discharge_to_voltage_limit(discharge_df, VOLTAGE_LOWER_LIMIT)

        direct_features = self.calculate_capacity_energy_features(cycle_num, charge_df, discharge_df)
        direct_features["charge_c_rate"] = CHARGE_C_RATE
        direct_features["discharge_c_rate"] = DISCHARGE_C_RATE
        direct_features.update(
            self.calculate_charge_phase_features(
                charge_df,
                uvp=VOLTAGE_UPPER_LIMIT,
                time_mode="duration",
                cv_voltage_tolerance=0.02,
            )
        )
        direct_features.update(
            self.calculate_discharge_phase_features(
                discharge_df,
                lvp=VOLTAGE_LOWER_LIMIT,
                time_mode="duration",
            )
        )

        derivative_features = extract_ic_dv_features(
            discharge_df,
            config={
                "peak_mode": 1,
                "nominal_capacity": BATTERY_NOMINAL_CAPACITY,
                "window_length_ic": 31,
                "window_length_dv": 21,
                "peak_height_ic": 0.01,
                "voltage_range_ic": (3.2, 4.2),
                "prominence_ic": 0.01,
                "ic_step_size": 0.002,
                "dv_step_size": BATTERY_NOMINAL_CAPACITY * 0.005,
                "search_window_dvv": 0.2,
                "search_window_dvp": 0.2,
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
            compute_cv_tau=False,
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_cycles", type=int, default=None, help="Number of cycles to process (default: all)")
    args = parser.parse_args()
    input_dir = resolve_dataset_input_dir(project_root, "HNEI", "data/HNEI")
    output_dir = project_root / "results" / "features" / "HNEI"
    HNEIFeatureExtractor().run(input_dir, output_dir, num_cycles=args.num_cycles)



if __name__ == "__main__":
    main()

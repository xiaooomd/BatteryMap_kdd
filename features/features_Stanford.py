"""
Feature Extraction Script for Stanford battery datasets.
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


class StanfordFeatureExtractor(BaseFeatureExtractor):
    CHARGE_SLOPES = [(0.10, 0.30), (0.10, 0.50), (0.10, 0.80)]
    DISCHARGE_SLOPES = [(0.10, 0.30), (0.10, 0.50), (0.10, 0.80)]
    TEVI_INTERVALS = [(3.5, 3.7), (3.7, 3.9), (3.9, 4.1)]
    TEVD_INTERVALS = [(4.1, 3.9), (3.9, 3.7), (3.7, 3.5)]

    def extract_cycle_features(
        self,
        cycle_data: Dict[str, Any],
        battery_data: Dict[str, Any],
        output_dir: Path,
    ) -> Dict[str, Any]:
        cycle_df = self.build_cycle_frame(cycle_data)
        cycle_num = cycle_data.get("cycle_number", 0)
        charge_df, discharge_df, rest_df = self.split_phase_frames(cycle_df)
        discharge_df = self.trim_discharge_to_voltage_limit(
            discharge_df,
            battery_data.get("min_voltage_limit_in_V", 0.0),
        )

        direct_features = self.calculate_capacity_energy_features(cycle_num, charge_df, discharge_df)
        direct_features["Rest_Time"] = 0.0
        direct_features["charge_c_rate"] = self.get_protocol_rate(battery_data.get("charge_protocol"))
        direct_features["discharge_c_rate"] = self.get_protocol_rate(battery_data.get("discharge_protocol"))
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

        nominal_capacity = battery_data.get("nominal_capacity_in_Ah", 2.0)
        derivative_features = extract_ic_dv_features(
            discharge_df,
            config={
                "peak_mode": 2,
                "nominal_capacity": nominal_capacity,
                "window_length_ic": 21,
                "window_length_dv": 21,
                "peak_height_ic": 0.01,
                "voltage_range_ic": (3.4, 4.1),
                "prominence_ic": 0.01,
                "ic_step_size": 0.002,
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
            cv_current_threshold=0.001,
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
    parser.add_argument(
        "--dataset_ids",
        type=str,
        default=None,
        help="Comma-separated Stanford datasets to process (choices: Stanford,Stanford_2). Default: both.",
    )
    args = parser.parse_args()

    dataset_candidates = ["Stanford_2", "Stanford"]
    if args.dataset_ids:
        requested = [item.strip() for item in str(args.dataset_ids).split(",") if item.strip()]
        invalid = [item for item in requested if item not in dataset_candidates]
        if invalid:
            raise ValueError(f"Unsupported dataset_ids for Stanford extractor: {invalid}")
        dataset_candidates = list(dict.fromkeys(requested))

    for dataset_name in dataset_candidates:
        input_dir = resolve_dataset_input_dir(project_root, dataset_name, f"data/{dataset_name}")
        output_dir = project_root / "results" / "features" / dataset_name
        print(f"\n{'=' * 40}")
        print(f"Processing Dataset: {dataset_name}")
        print(f"Input: {input_dir}")
        print(f"{'=' * 40}")
        StanfordFeatureExtractor().run(input_dir, output_dir, num_cycles=args.num_cycles)



if __name__ == "__main__":
    main()

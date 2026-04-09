"""
Feature Extraction Script for CALB Battery Dataset.

CALB is NCM chemistry with CCCV charging. Temperature is parsed from
filename (format: CALB_Temp_ID.pkl). Ambient temperature is appended
to each cycle row.
"""

import argparse
import warnings
import sys
from pathlib import Path
from typing import Any, Dict, Optional

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from features.base_extractor import BaseFeatureExtractor, resolve_dataset_input_dir
from src.physics.ic_dv_extractor import extract_ic_dv_features

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class CALBFeatureExtractor(BaseFeatureExtractor):
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

        lvp = battery_data.get("min_voltage_limit_in_V", 2.5)
        uvp = battery_data.get("max_voltage_limit_in_V", 4.2)
        discharge_df = self.trim_discharge_to_voltage_limit(discharge_df, lvp)
        nominal_cap = battery_data.get("nominal_capacity_in_Ah", 2.0)

        direct_features = self.calculate_capacity_energy_features(
            cycle_num, charge_df, discharge_df
        )
        direct_features.update(
            self.calculate_charge_phase_features(
                charge_df,
                uvp=uvp,
                time_mode="duration",
                cv_voltage_tolerance=0.01,
            )
        )
        direct_features.update(
            self.calculate_discharge_phase_features(
                discharge_df,
                lvp=lvp,
                time_mode="duration",
            )
        )
        direct_features["discharge_c_rate"] = self.get_protocol_rate(
            battery_data.get("discharge_protocol")
        )
        direct_features["avg_charge_c_rate"] = self.get_protocol_rate(
            battery_data.get("charge_protocol")
        )

        derivative_features = extract_ic_dv_features(
            discharge_df,
            config={
                "peak_mode": 2,
                "nominal_capacity": nominal_cap,
                "window_length_ic": 41,
                "window_length_dv": 25,
                "peak_height_ic": 0.5,
                "voltage_range_ic": (3.2, 3.6),
                "prominence_ic": 0.1,
                "ic_step_size": 0.002,
                "dv_step_size": nominal_cap * 0.005,
                "search_window_dvv": 0.1,
                "search_window_dvp": 0.04,
                "initial_capacity_cut_fraction": 0.02,
                "icv_search_offset_lower": 0.05,
                "icv_search_offset_upper": 0.5,
            },
            plot_params=self.build_plot_params(battery_data, cycle_num, output_dir),
        )
        advanced_features = self.calculate_advanced_features_common(
            cycle_df, charge_df, discharge_df, direct_features,
            rest_df=rest_df, compute_cv_tau=True, cv_voltage_tolerance=0.01,
        )
        anchor_features = self.calculate_anchor_features_common(
            charge_df, discharge_df,
            self.CHARGE_SLOPES, self.DISCHARGE_SLOPES,
            self.TEVI_INTERVALS, self.TEVD_INTERVALS,
        )
        return {**direct_features, **derivative_features, **advanced_features, **anchor_features}

    def process_battery(
        self,
        file_path: Path,
        output_dir: Path,
        num_cycles: Optional[int] = None,
    ) -> None:
        """Override to append Ambient_Temperature parsed from filename."""
        # Extract ambient temperature from filename format: CALB_Temp_ID.pkl
        try:
            ambient_temp = float(file_path.stem.split("_")[1])
        except (IndexError, ValueError):
            ambient_temp = None

        # Call parent; it saves the CSV
        super().process_battery(file_path, output_dir, num_cycles)

        # Post-process: re-open the saved CSV and append ambient temp
        if ambient_temp is not None:
            try:
                battery_data = self.load_battery_data(file_path)
                cell_id = self.get_cell_id(battery_data, file_path)
                csv_path = output_dir / f"{cell_id}.csv"
                if csv_path.exists():
                    import pandas as pd
                    df = pd.read_csv(csv_path)
                    df["Ambient_Temperature"] = ambient_temp
                    df.to_csv(csv_path, index=False)
            except Exception:
                pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_cycles", type=int, default=None, help="Number of cycles to process (default: all)")
    args = parser.parse_args()
    input_dir = resolve_dataset_input_dir(project_root, "CALB", "data/CALB")
    output_dir = project_root / "results" / "features" / "CALB"
    CALBFeatureExtractor().run(input_dir, output_dir, num_cycles=args.num_cycles)



if __name__ == "__main__":
    main()

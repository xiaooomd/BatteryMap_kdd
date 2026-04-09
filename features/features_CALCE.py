"""
Feature Extraction Script for CALCE Battery Dataset.

Refactored to use BaseFeatureExtractor. CALCE is an LCO/NMC dataset
with simple CC-CV charging. Cell conditions (C-rate, nominal capacity)
are looked up from the BATTERY_CONDITIONS table by cell model name.
"""

import argparse
import warnings
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from features.base_extractor import BaseFeatureExtractor, resolve_dataset_input_dir
from src.physics.ic_dv_extractor import extract_ic_dv_features

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# CALCE Dataset Battery Conditions
BATTERY_CONDITIONS = {
    # 0.5C discharge rate
    "CS2_33": {"discharge_rate": 0.5, "nominal_capacity": 1.1},
    "CS2_34": {"discharge_rate": 0.5, "nominal_capacity": 1.1},
    "CX2_16": {"discharge_rate": 0.5, "nominal_capacity": 1.35},
    "CX2_33": {"discharge_rate": 0.5, "nominal_capacity": 1.35},
    "CX2_34": {"discharge_rate": 0.5, "nominal_capacity": 1.35},
    "CX2_35": {"discharge_rate": 0.5, "nominal_capacity": 1.35},
    "CX2_36": {"discharge_rate": 0.5, "nominal_capacity": 1.35},
    "CX2_37": {"discharge_rate": 0.5, "nominal_capacity": 1.35},
    "CX2_38": {"discharge_rate": 0.5, "nominal_capacity": 1.35},
    # 1C discharge rate
    "CS2_35": {"discharge_rate": 1.0, "nominal_capacity": 1.1},
    "CS2_36": {"discharge_rate": 1.0, "nominal_capacity": 1.1},
    "CS2_37": {"discharge_rate": 1.0, "nominal_capacity": 1.1},
    "CS2_38": {"discharge_rate": 1.0, "nominal_capacity": 1.1},
}


class CalceFeatureExtractor(BaseFeatureExtractor):
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

        lvp = battery_data.get("min_voltage_limit_in_V", 2.7)
        uvp = battery_data.get("max_voltage_limit_in_V", 4.2)
        discharge_df = self.trim_discharge_to_voltage_limit(discharge_df, lvp)

        cell_id = battery_data.get("cell_id", "unknown")
        cell_model = cell_id.replace("CALCE_", "")
        conditions = BATTERY_CONDITIONS.get(cell_model, {})
        nominal_cap = conditions.get("nominal_capacity", 1.1)

        direct_features = self.calculate_capacity_energy_features(
            cycle_num, charge_df, discharge_df
        )
        direct_features["charge_c_rate"] = 0.5
        direct_features["discharge_c_rate"] = conditions.get("discharge_rate", 1.0)
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

        derivative_features = extract_ic_dv_features(
            discharge_df,
            config={
                "peak_mode": 1,
                "nominal_capacity": nominal_cap,
                "window_length_ic": 25,
                "window_length_dv": 31,
                "peak_height_ic": 0.05,
                "voltage_range_ic": (3.4, 4.2),
                "prominence_ic": 0.01,
                "ic_step_size": 0.001,
                "dv_step_size": nominal_cap * 0.005,
                "search_window_dvv": 0.05,
                "search_window_dvp": 0.05,
                "initial_capacity_cut_fraction": 0.01,
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
            cv_voltage_tolerance=0.01,
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

    def handle_cycle_error(self, cell_id: str, exc: Exception) -> None:
        return None

    def order_columns(self, features_df) -> object:
        base_cols = [
            "Cycle_Number", "Discharge_Capacity", "Charge_Capacity",
            "Coulombic_Efficiency", "Discharge_Energy", "Charge_Energy",
            "Energy_Efficiency", "charge_c_rate", "discharge_c_rate",
            "ICHV", "UVP_time", "TCCC", "TCVC", "CV_Current_Tau", "UVP",
            "IDV", "LVP_time", "var_I_discharge", "var_V_discharge",
            "median_V_discharge", "total_discharge_time", "LVP",
            "ICP", "ICPL_V", "ICP_FWHM", "ICP_Area", "DVP", "DVPL_V", "DVV", "DVVL_V",
            "Internal_Resistance", "RCV", "skew_V_discharge",
            "charge_slope_1", "charge_slope_2", "charge_slope_3",
            "discharge_slope_1", "discharge_slope_2", "discharge_slope_3",
            "TEVI_1", "TEVI_2", "TEVI_3", "TEVD_1", "TEVD_2", "TEVD_3",
        ]
        existing = [c for c in base_cols if c in features_df.columns]
        remaining = sorted([c for c in features_df.columns if c not in existing])
        return features_df[existing + remaining]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_cycles", type=int, default=None, help="Number of cycles to process (default: all)")
    args = parser.parse_args()
    input_dir = resolve_dataset_input_dir(project_root, "CALCE", "data/CALCE")
    output_dir = project_root / "results" / "features" / "CALCE"
    CalceFeatureExtractor().run(input_dir, output_dir, num_cycles=args.num_cycles)



if __name__ == "__main__":
    main()

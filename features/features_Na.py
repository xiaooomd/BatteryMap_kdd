"""
Feature Extraction Script for Na-ion Battery Dataset.

Na-ion cells use CC-only charging (no CV phase) and different voltage
ranges (typically 2.0–4.0V). AttrDict pkl format is used.
Special cells require modified IC/DV search parameters.
"""

import argparse
import warnings
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from features.base_extractor import BaseFeatureExtractor, resolve_dataset_input_dir
from src.physics.ic_dv_extractor import extract_ic_dv_features

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Na-ion cells requiring wider IC/DV search
_SPECIAL_CELLS = {
    "NA-ion_270040-6-5-27",
    "NA-ion_270040-1-1-64",
    "NA-ion_270040-8-3-18",
}


class _AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


class NaFeatureExtractor(BaseFeatureExtractor):
    CHARGE_SLOPES = [(0.10, 0.30), (0.10, 0.50), (0.10, 0.80)]
    DISCHARGE_SLOPES = [(0.10, 0.30), (0.10, 0.50), (0.10, 0.80)]
    TEVI_INTERVALS = [(3.0, 3.2), (3.3, 3.5), (3.6, 3.8)]
    TEVD_INTERVALS = [(3.9, 3.6), (3.5, 3.2), (3.1, 2.8)]

    def load_battery_data(self, file_path: Path) -> Dict[str, Any]:
        import pickle
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        battery = _AttrDict(data)
        if battery.get("cycle_data"):
            battery["cycle_data"] = [_AttrDict(c) for c in battery["cycle_data"]]
        if battery.get("charge_protocol"):
            battery["charge_protocol"] = [_AttrDict(p) for p in battery["charge_protocol"]]
        if battery.get("discharge_protocol"):
            battery["discharge_protocol"] = [_AttrDict(p) for p in battery["discharge_protocol"]]
        return battery

    def build_cycle_frame(self, cycle_data):
        import pandas as pd
        return pd.DataFrame({
            "Time(s)": cycle_data.time_in_s,
            "Current(A)": cycle_data.current_in_A,
            "Voltage(V)": cycle_data.voltage_in_V,
            "Charge_Capacity(Ah)": cycle_data.charge_capacity_in_Ah,
            "Discharge_Capacity(Ah)": cycle_data.discharge_capacity_in_Ah,
        })

    def get_cycles_to_process(self, battery_data, num_cycles):
        cycles = list(battery_data.get("cycle_data") or [])
        if num_cycles and num_cycles > 0:
            cycles = cycles[:num_cycles]
        return cycles

    def extract_cycle_features(
        self,
        cycle_data,
        battery_data: Dict[str, Any],
        output_dir: Path,
    ) -> Optional[Dict[str, Any]]:
        cycle_df = self.build_cycle_frame(cycle_data)
        cycle_num = getattr(cycle_data, "cycle_number", 0)
        charge_df, discharge_df, _ = self.split_phase_frames(cycle_df)

        lvp = getattr(battery_data, "min_voltage_limit_in_V", 2.0)
        uvp = getattr(battery_data, "max_voltage_limit_in_V", 4.0)
        discharge_df = self.trim_discharge_to_voltage_limit(discharge_df, lvp)
        nominal_cap = battery_data.get("nominal_capacity_in_Ah", 1.0)
        cell_id = getattr(battery_data, "cell_id", "")

        direct_features = self.calculate_capacity_energy_features(
            cycle_num, charge_df, discharge_df
        )
        try:
            direct_features["charge_c_rate"] = battery_data.charge_protocol[0].rate_in_C
            direct_features["discharge_c_rate"] = battery_data.discharge_protocol[0].rate_in_C
        except (AttributeError, IndexError, TypeError):
            direct_features["charge_c_rate"] = np.nan
            direct_features["discharge_c_rate"] = np.nan

        # Na-ion: no CV phase → force_no_cv=True
        direct_features.update(
            self.calculate_charge_phase_features(
                charge_df, uvp=uvp, time_mode="duration", force_no_cv=True,
            )
        )
        direct_features.update(
            self.calculate_discharge_phase_features(
                discharge_df, lvp=lvp, time_mode="duration",
            )
        )

        ic_config = {
            "peak_mode": 2,
            "nominal_capacity": nominal_cap,
            "window_length_ic": 21,
            "window_length_dv": 21,
            "peak_height_ic": 0.01,
            "voltage_range_ic": (2.4, 3.0),
            "prominence_ic": 0.05,
            "ic_step_size": 0.01,
            "dv_step_size": nominal_cap * 0.005,
            "search_window_dvv": 0.1,
            "search_window_dvp": 0.1,
            "initial_capacity_cut_fraction": 0.02,
            "icv_search_offset_lower": 0.05,
            "icv_search_offset_upper": 0.5,
            "icv_search_direction": "right",
        }
        if cell_id in _SPECIAL_CELLS:
            ic_config["voltage_range_ic"] = (2.4, 3.5)
            ic_config["icv_search_direction"] = "left"
            ic_config["icv_search_offset_upper"] = 1.0

        derivative_features = extract_ic_dv_features(
            discharge_df,
            config=ic_config,
            plot_params=self.build_plot_params(battery_data, cycle_num, output_dir),
        )
        advanced_features = self.calculate_advanced_features_common(
            cycle_df, charge_df, discharge_df, direct_features, compute_cv_tau=False,
        )
        anchor_features = self.calculate_anchor_features_common(
            charge_df, discharge_df,
            self.CHARGE_SLOPES, self.DISCHARGE_SLOPES,
            self.TEVI_INTERVALS, self.TEVD_INTERVALS,
        )
        return {**direct_features, **derivative_features, **advanced_features, **anchor_features}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_cycles", type=int, default=None, help="Number of cycles to process (default: all)")
    args = parser.parse_args()
    input_dir = resolve_dataset_input_dir(project_root, "NA-ion", "data/NA-ion")
    output_dir = project_root / "results" / "features" / "NA-ion"
    NaFeatureExtractor().run(input_dir, output_dir, num_cycles=args.num_cycles)



if __name__ == "__main__":
    main()

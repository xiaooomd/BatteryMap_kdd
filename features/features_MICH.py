"""
Feature Extraction Script for MICH Battery Dataset (LG HG2 NCM).

MICH pkl files store battery_data and cycle_data as AttrDict-style objects
(dot-notation access). The load_battery_data override wraps them in a plain
dict so the base class can consume them normally.
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


class _AttrDict(dict):
    """Dict with dot-notation access, used by raw MICH pkl files."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


class MICHFeatureExtractor(BaseFeatureExtractor):
    CHARGE_SLOPES = [(0.10, 0.30), (0.10, 0.50), (0.10, 0.80)]
    DISCHARGE_SLOPES = [(0.10, 0.30), (0.10, 0.50), (0.10, 0.80)]
    TEVI_INTERVALS = [(3.5, 3.7), (3.7, 3.9), (3.9, 4.1)]
    TEVD_INTERVALS = [(4.1, 3.9), (3.9, 3.7), (3.7, 3.5)]

    def load_battery_data(self, file_path: Path) -> Dict[str, Any]:
        """Load and convert AttrDict pkl format to plain dict."""
        import pickle
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        # Wrap in AttrDict so dot-notation and dict-notation both work
        battery_data = _AttrDict(data)
        if battery_data.get("cycle_data"):
            battery_data["cycle_data"] = [
                _AttrDict(c) for c in battery_data["cycle_data"]
            ]
        if battery_data.get("charge_protocol"):
            battery_data["charge_protocol"] = [
                _AttrDict(p) for p in battery_data["charge_protocol"]
            ]
        if battery_data.get("discharge_protocol"):
            battery_data["discharge_protocol"] = [
                _AttrDict(p) for p in battery_data["discharge_protocol"]
            ]
        return battery_data

    def build_cycle_frame(self, cycle_data):
        """Build DataFrame from AttrDict cycle_data (dot-notation)."""
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
        charge_df, discharge_df, rest_df = self.split_phase_frames(cycle_df)

        lvp = getattr(battery_data, "min_voltage_limit_in_V", 2.5)
        uvp = getattr(battery_data, "max_voltage_limit_in_V", 4.2)
        discharge_df = self.trim_discharge_to_voltage_limit(discharge_df, lvp)

        nominal_cap = battery_data.get("nominal_capacity_in_Ah", 1.1)

        direct_features = self.calculate_capacity_energy_features(
            cycle_num, charge_df, discharge_df
        )
        # C-rates from AttrDict protocol objects
        try:
            direct_features["charge_c_rate"] = battery_data.charge_protocol[0].rate_in_C
            direct_features["discharge_c_rate"] = battery_data.discharge_protocol[0].rate_in_C
        except (AttributeError, IndexError, TypeError):
            direct_features["charge_c_rate"] = np.nan
            direct_features["discharge_c_rate"] = np.nan

        direct_features.update(
            self.calculate_charge_phase_features(
                charge_df, uvp=uvp, time_mode="duration", cv_voltage_tolerance=0.01,
            )
        )
        direct_features.update(
            self.calculate_discharge_phase_features(
                discharge_df, lvp=lvp, time_mode="duration",
            )
        )

        derivative_features = extract_ic_dv_features(
            discharge_df,
            config={
                "peak_mode": 1,
                "nominal_capacity": nominal_cap,
                "window_length_ic": 21,
                "window_length_dv": 21,
                "peak_height_ic": 0.01,
                "voltage_range_ic": (3.3, 4.2),
                "prominence_ic": 0.02,
                "ic_step_size": 0.01,
                "dv_step_size": nominal_cap * 0.005,
                "search_window_dvv": 0.1,
                "search_window_dvp": 0.1,
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_cycles", type=int, default=None, help="Number of cycles to process (default: all)")
    args = parser.parse_args()
    input_dir = resolve_dataset_input_dir(project_root, "MICH", "data/MICH")
    output_dir = project_root / "results" / "features" / "MICH"
    MICHFeatureExtractor().run(input_dir, output_dir, num_cycles=args.num_cycles)



if __name__ == "__main__":
    main()

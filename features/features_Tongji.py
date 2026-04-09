"""
Feature Extraction Script for Tongji University Battery Dataset.

Tongji NCM cells use C-rate values inferred from current data (not
stored in metadata). An auxiliary peak feature (V_diff_Peak3_Peak1,
Ratio_Peak1_Peak3) is computed from a secondary IC peak. AttrDict pkl format.
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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


class TongjiFeatureExtractor(BaseFeatureExtractor):
    CHARGE_SLOPES = [(0.1, 0.3), (0.1, 0.5), (0.1, 0.8)]
    DISCHARGE_SLOPES = [(0.1, 0.3), (0.1, 0.5), (0.1, 0.8)]
    TEVI_INTERVALS = [(3.5, 3.7), (3.7, 3.9), (3.9, 4.1)]
    TEVD_INTERVALS = [(4.1, 3.9), (3.9, 3.7), (3.7, 3.5)]

    def load_battery_data(self, file_path: Path) -> Dict[str, Any]:
        import pickle
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        battery = _AttrDict(data)
        if battery.get("cycle_data"):
            battery["cycle_data"] = [_AttrDict(c) for c in battery["cycle_data"]]
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
        charge_df, discharge_df, rest_df = self.split_phase_frames(cycle_df)

        v_lower = getattr(battery_data, "min_voltage_limit_in_V", 2.5)
        v_upper = getattr(battery_data, "max_voltage_limit_in_V", 4.2)
        discharge_df = self.trim_discharge_to_voltage_limit(discharge_df, v_lower)
        nominal_cap = getattr(battery_data, "nominal_capacity_in_Ah", 2.0)
        cell_id = getattr(battery_data, "cell_id", "")

        direct_features = self.calculate_capacity_energy_features(
            cycle_num, charge_df, discharge_df
        )
        # Tongji: infer C-rates from current data since metadata may be absent
        direct_features["charge_c_rate"] = self._infer_c_rate(charge_df, nominal_cap, use_cc_half=True)
        direct_features["discharge_c_rate"] = self._infer_c_rate(discharge_df, nominal_cap)
        direct_features.update(
            self.calculate_charge_phase_features(
                charge_df, uvp=v_upper, time_mode="duration", cv_voltage_tolerance=0.01,
            )
        )
        direct_features.update(
            self.calculate_discharge_phase_features(
                discharge_df, lvp=v_lower, time_mode="duration",
            )
        )

        ic_config = {
            "peak_mode": 2,
            "nominal_capacity": nominal_cap,
            "window_length_ic": 21,
            "window_length_dv": 21,
            "peak_height_ic": 0.01,
            "voltage_range_ic": (3.8, 4.0),
            "prominence_ic": 0.01,
            "ic_step_size": 0.005,
            "dv_step_size": nominal_cap * 0.005,
            "search_window_dvv": 0.1,
            "search_window_dvp": 0.1,
            "initial_capacity_cut_fraction": 0.02,
            "icv_search_offset_lower": 0.05,
            "icv_search_offset_upper": 0.2,
            "icv_search_direction": "left",
            "fwhm_method": "valley_limited",
            "ic_area_config": {"method": "fixed_width", "width_v": 0.05},
            "aux_peak_config": {"voltage_range": (3.1, 3.75), "selection": "first"},
        }
        # High-rate (3C) cells: ICP peak shifts down due to polarization
        if "CY25-05_4--" in cell_id:
            ic_config["voltage_range_ic"] = (3.6, 4.05)

        derivative_features = extract_ic_dv_features(
            discharge_df,
            config=ic_config,
            plot_params=self.build_plot_params(battery_data, cycle_num, output_dir),
        )
        # Compute auxiliary peak relationship features
        derivative_features.update(self._compute_aux_peak_ratios(derivative_features))

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

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _infer_c_rate(df, nominal_cap: float, use_cc_half: bool = False) -> float:
        if df.empty or nominal_cap <= 0:
            return 0.0
        if use_cc_half:
            cc_half = df.iloc[: len(df) // 2]
            current = cc_half["Current(A)"].median() if not cc_half.empty else 0.0
        else:
            current = df["Current(A)"].abs().median()
        return round(float(current) / nominal_cap, 2)

    @staticmethod
    def _compute_aux_peak_ratios(derivative_features: Dict[str, Any]) -> Dict[str, Any]:
        icpl_v_main = derivative_features.get("ICPL_V", np.nan)
        icpl_v_aux = derivative_features.get("ICPL_V_Aux", np.nan)
        icp_main = derivative_features.get("ICP", 0.0)
        icp_aux = derivative_features.get("ICP_Aux", 0.0)
        extras = {}
        if not (np.isnan(icpl_v_main) or np.isnan(icpl_v_aux)):
            extras["V_diff_Peak3_Peak1"] = icpl_v_main - icpl_v_aux
        else:
            extras["V_diff_Peak3_Peak1"] = 0.0
        extras["Ratio_Peak1_Peak3"] = (
            icp_aux / icp_main if icp_main > 1e-6 and not np.isnan(icp_aux) else 0.0
        )
        return extras


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_cycles", type=int, default=None, help="Number of cycles to process (default: all)")
    args = parser.parse_args()
    input_dir = resolve_dataset_input_dir(project_root, "Tongji", "data/Tongji")
    output_dir = project_root / "results" / "features" / "Tongji"
    TongjiFeatureExtractor().run(input_dir, output_dir, num_cycles=args.num_cycles)



if __name__ == "__main__":
    main()

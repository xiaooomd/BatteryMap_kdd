"""
Feature Extraction Script for ISU-ILCC Battery Dataset.

ISU-ILCC is the most complex dataset:
- C-rates come from a hard-coded CYCLING_RATES lookup keyed by cell group.
- Time arrays may be datetime objects → normalized to relative seconds.
- Per-cell regex-based IC/DV configuration (peak voltage ranges).
- Dynamic TEVI/TEVD voltage intervals based on cell group.
"""

import argparse
import re
import warnings
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from features.base_extractor import BaseFeatureExtractor, resolve_dataset_input_dir
from src.physics.ic_dv_extractor import extract_ic_dv_features

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Rates map: group_key → [Charge Rate, Discharge Rate]
CYCLING_RATES = {
    "ISU-ILCC_G1": [0.5, 0.5], "ISU-ILCC_G2": [0.5, 0.5], "ISU-ILCC_G3": [0.5, 0.5],
    "ISU-ILCC_G4": [1, 0.5], "ISU-ILCC_G5": [1, 0.5], "ISU-ILCC_G6": [2, 0.5],
    "ISU-ILCC_G7": [2, 0.5], "ISU-ILCC_G8": [2, 0.5], "ISU-ILCC_G9": [2, 0.5],
    "ISU-ILCC_G10": [2.5, 0.5], "ISU-ILCC_G12": [3, 0.5], "ISU-ILCC_G13": [3, 0.5],
    "ISU-ILCC_G14": [3, 0.5], "ISU-ILCC_G15": [3, 0.5], "ISU-ILCC_G16": [0.5, 0.5],
    "ISU-ILCC_G17": [1, 0.5], "ISU-ILCC_G18": [2.5, 0.5], "ISU-ILCC_G19": [2.5, 0.5],
    "ISU-ILCC_G20": [0.8, 0.5], "ISU-ILCC_G21": [1.2, 0.5], "ISU-ILCC_G22": [1.4, 0.5],
    "ISU-ILCC_G23": [1.6, 0.5], "ISU-ILCC_G24": [1.8, 0.5], "ISU-ILCC_G25": [1.8, 0.6],
    "ISU-ILCC_G26": [1.4, 2.2], "ISU-ILCC_G27": [0.6, 2.4], "ISU-ILCC_G28": [2.4, 1.6],
    "ISU-ILCC_G29": [1.6, 1.8], "ISU-ILCC_G30": [0.8, 0.8], "ISU-ILCC_G31": [1.2, 1],
    "ISU-ILCC_G32": [1, 1.4], "ISU-ILCC_G33": [2, 1.2], "ISU-ILCC_G34": [2.2, 2],
    "ISU-ILCC_G35": [1.825, 0.5], "ISU-ILCC_G36": [2.075, 0.5], "ISU-ILCC_G37": [0.725, 0.5],
    "ISU-ILCC_G38": [1.875, 0.5], "ISU-ILCC_G39": [1.475, 0.5], "ISU-ILCC_G40": [1.825, 1.025],
    "ISU-ILCC_G41": [2.075, 1.775], "ISU-ILCC_G42": [0.725, 2.375], "ISU-ILCC_G43": [1.875, 2.325],
    "ISU-ILCC_G44": [0.775, 1.275], "ISU-ILCC_G45": [1.125, 1.725], "ISU-ILCC_G46": [1.225, 2.025],
    "ISU-ILCC_G47": [2.325, 1.925], "ISU-ILCC_G48": [2.375, 2.225], "ISU-ILCC_G49": [0.975, 0.675],
    "ISU-ILCC_G50": [2.425, 1.625], "ISU-ILCC_G51": [2.275, 1.875], "ISU-ILCC_G52": [1.425, 0.875],
    "ISU-ILCC_G53": [2.025, 0.825], "ISU-ILCC_G54": [0.925, 1.125], "ISU-ILCC_G55": [1.025, 2.475],
    "ISU-ILCC_G56": [2.175, 0.975], "ISU-ILCC_G57": [1.775, 1.175], "ISU-ILCC_G58": [2.475, 0.575],
    "ISU-ILCC_G59": [1.325, 1.825], "ISU-ILCC_G60": [0.675, 1.325], "ISU-ILCC_G61": [2.125, 1.975],
    "ISU-ILCC_G62": [1.575, 2.425], "ISU-ILCC_G63": [1.975, 1.675], "ISU-ILCC_G64": [1.175, 1.425],
}

CHARGE_CUTOFF_V = 4.2
DISCHARGE_CUTOFF_V = 3.0


class ISUILCCFeatureExtractor(BaseFeatureExtractor):
    CHARGE_SLOPES = [(0.10, 0.30), (0.10, 0.50), (0.10, 0.80)]
    DISCHARGE_SLOPES = [(0.10, 0.30), (0.10, 0.50), (0.10, 0.80)]
    # Default TEVI/TEVD — overridden per-cell by _get_voltage_intervals()
    TEVI_INTERVALS = [(3.6, 3.75), (3.8, 3.95), (4.0, 4.15)]
    TEVD_INTERVALS = [(4.1, 3.9), (3.8, 3.6), (3.5, 3.3)]

    def build_cycle_frame(self, cycle_data: Dict[str, Any]) -> pd.DataFrame:
        """Normalize time (may be datetime objects) and build DataFrame."""
        times = self._normalize_time_array(cycle_data.get("time_in_s"))
        return pd.DataFrame({
            "Time(s)": times,
            "Current(A)": cycle_data["current_in_A"],
            "Voltage(V)": cycle_data["voltage_in_V"],
            "Charge_Capacity(Ah)": cycle_data["charge_capacity_in_Ah"],
            "Discharge_Capacity(Ah)": cycle_data["discharge_capacity_in_Ah"],
        })

    def extract_cycle_features(
        self,
        cycle_data: Dict[str, Any],
        battery_data: Dict[str, Any],
        output_dir: Path,
    ) -> Optional[Dict[str, Any]]:
        cycle_df = self.build_cycle_frame(cycle_data)
        if cycle_df.empty or cycle_df["Time(s)"].size == 0:
            return {}
        cycle_num = cycle_data.get("cycle_number", 0)
        cell_id = battery_data.get("cell_id", "")

        charge_df, discharge_df, rest_df = self.split_phase_frames(cycle_df)
        discharge_df = self.trim_discharge_to_voltage_limit(discharge_df, DISCHARGE_CUTOFF_V)

        group_key = self._get_group_key(cell_id)
        rates = CYCLING_RATES.get(group_key, [np.nan, np.nan])

        direct_features = self.calculate_capacity_energy_features(
            cycle_num, charge_df, discharge_df
        )
        direct_features["charge_c_rate"] = rates[0]
        direct_features["discharge_c_rate"] = rates[1]
        direct_features.update(
            self.calculate_charge_phase_features(
                charge_df, uvp=CHARGE_CUTOFF_V, time_mode="duration",
                cv_voltage_tolerance=0.01,
            )
        )
        direct_features.update(
            self.calculate_discharge_phase_features(
                discharge_df, lvp=DISCHARGE_CUTOFF_V, time_mode="duration",
            )
        )

        ic_config = self._get_isu_ilcc_config(cell_id)
        derivative_features = extract_ic_dv_features(
            discharge_df,
            config=ic_config,
            plot_params=self.build_plot_params(battery_data, cycle_num, output_dir),
        )
        advanced_features = self.calculate_advanced_features_common(
            cycle_df, charge_df, discharge_df, direct_features,
            rest_df=rest_df, compute_cv_tau=True, cv_voltage_tolerance=0.01,
        )
        tevi_ints, tevd_ints = self._get_voltage_intervals(cell_id)
        anchor_features = self.calculate_anchor_features_common(
            charge_df, discharge_df,
            self.CHARGE_SLOPES, self.DISCHARGE_SLOPES,
            tevi_ints, tevd_ints,
        )
        return {**direct_features, **derivative_features, **advanced_features, **anchor_features}

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _normalize_time_array(times) -> np.ndarray:
        if times is None:
            return np.array([])
        arr = np.array(times)
        if arr.size == 0:
            return arr
        if np.issubdtype(arr.dtype, np.datetime64):
            ns = arr.astype("datetime64[ns]").astype("int64")
            s = ns.astype("float64") / 1e9
            return s - s[0]
        if arr.dtype == "O":
            try:
                ts = pd.to_datetime(arr, errors="coerce")
                if not ts.isna().all():
                    ns = ts.values.astype("int64")
                    s = ns.astype("float64") / 1e9
                    return s - s[0]
            except Exception:
                pass
        try:
            arrf = arr.astype("float64")
            if arrf.size > 0 and np.nanmax(np.abs(arrf)) > 1e11:
                arrf /= 1e9
            if arrf.size > 0:
                arrf -= arrf[0]
            return arrf
        except ValueError:
            return np.array([])

    @staticmethod
    def _get_group_key(cell_id: str) -> str:
        m = re.search(r"(ISU-ILCC_G\d+)", cell_id)
        return m.group(1) if m else cell_id

    @staticmethod
    def _get_voltage_intervals(
        cell_id: str,
    ) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        if re.search(r"(ISU-ILCC_)?(G5C|G16C|G2[0-4]C|G3[5-9]C)", cell_id):
            return [(3.25, 3.35), (3.35, 3.45), (3.45, 3.55)], [(3.5, 3.4), (3.3, 3.2), (3.2, 3.05)]
        if re.search(r"(ISU-ILCC_)?(G1C|G2C|G4C|G6C|G7C|G10C|G12C|G13C|G18C|G25C|G50C|G54C|G5[6-9]C)", cell_id):
            return [(4.0, 4.05), (4.05, 4.1), (4.1, 4.15)], [(4.15, 4.1), (4.0, 3.9), (3.9, 3.8)]
        return [(3.6, 3.75), (3.8, 3.95), (4.0, 4.15)], [(4.1, 3.9), (3.8, 3.6), (3.5, 3.3)]

    @staticmethod
    def _get_isu_ilcc_config(cell_id: str) -> Dict[str, Any]:
        nominal_cap = 2.0
        config: Dict[str, Any] = {
            "peak_mode": 2, "nominal_capacity": nominal_cap,
            "window_length_ic": 21, "window_length_dv": 21,
            "peak_height_ic": 0.01, "voltage_range_ic": (3.85, 4.2),
            "prominence_ic": 0.02, "ic_step_size": 0.01,
            "dv_step_size": nominal_cap * 0.005,
            "search_window_dvv": 0.1, "search_window_dvp": 0.1,
            "initial_capacity_cut_fraction": 0.02,
            "icv_search_offset_lower": 0.02, "icv_search_offset_upper": 0.2,
            "ic_area_config": {"method": "fixed_width", "width_v": 0.03},
            "icv_method": "first_valley_left",
            "force_icp_fwhm_zero": True,
            "aux_peak_config": {"voltage_range": (3.75, 3.85), "selection": "max", "default_value": 0.0},
        }
        # Per-group voltage range overrides
        if re.search(r"G40C3|G41C[34]|G42C[1-3]|G43C[12]|G46C[13]|G47C1|G48C[1-4]|G51C4|G59C4|G60C2|G61C[24]|G62C4|G63C[12]|G64C[1-4]", cell_id):
            config["voltage_range_ic"] = (3.8, 4.0)
            config["aux_peak_config"]["voltage_range"] = (3.7, 3.8)
        elif re.search(r"G27C1", cell_id):
            config["voltage_range_ic"] = (3.7, 3.9)
            config["aux_peak_config"]["voltage_range"] = (3.6, 3.7)
        elif re.search(r"G27C[2-4]", cell_id):
            config["voltage_range_ic"] = (3.75, 3.9)
            config["aux_peak_config"]["voltage_range"] = (3.5, 3.75)
        elif re.search(r"G34C[2-4]|G43C[34]|G51C2|G55C[1-4]|G62C[1-3]", cell_id):
            config["voltage_range_ic"] = (3.75, 3.9)
            config["aux_peak_config"]["voltage_range"] = (3.65, 3.75)
        if re.search(r"G57", cell_id):
            config["force_icv_zero"] = True
        if re.search(r"G40C3", cell_id):
            config["window_length_ic"] = 51
            config["window_length_dv"] = 51
        return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_cycles", type=int, default=None, help="Number of cycles to process (default: all)")
    args = parser.parse_args()
    input_dir = resolve_dataset_input_dir(project_root, "ISU-ILCC", "data/ISU-ILCC")
    output_dir = project_root / "results" / "features" / "ISU-ILCC"
    ISUILCCFeatureExtractor().run(input_dir, output_dir, num_cycles=args.num_cycles)



if __name__ == "__main__":
    main()

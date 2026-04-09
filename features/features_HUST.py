"""
Feature Extraction Script for HUST Battery Dataset.

HUST uses LFP A123 cells with multi-stage CC-CV charging (5C then 1C)
and multi-stage CC discharging. IC/DV is computed on the low-rate (C2)
charge phase rather than discharge.
"""

import argparse
import traceback
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

# HUST Dataset Specifics
CHARGE_CUTOFF_V = 3.6
DISCHARGE_CUTOFF_V = 2.0

# Reference Table for Discharge Protocols (from dataset paper)
TABLE_S1 = {
    "#1":  {"Cycle life": 1504, "C1": "5C", "C2": "1C", "C3": "1C", "C4": "1C"},
    "#2":  {"Cycle life": 2678, "C1": "5C", "C2": "1C", "C3": "2C", "C4": "1C"},
    "#3":  {"Cycle life": 1858, "C1": "5C", "C2": "1C", "C3": "3C", "C4": "1C"},
    "#4":  {"Cycle life": 1500, "C1": "5C", "C2": "1C", "C3": "4C", "C4": "1C"},
    "#5":  {"Cycle life": 1971, "C1": "5C", "C2": "1C", "C3": "5C", "C4": "1C"},
    "#6":  {"Cycle life": 1143, "C1": "5C", "C2": "2C", "C3": "1C", "C4": "1C"},
    "#7":  {"Cycle life": 1678, "C1": "5C", "C2": "2C", "C3": "2C", "C4": "1C"},
    "#8":  {"Cycle life": 2285, "C1": "5C", "C2": "2C", "C3": "3C", "C4": "1C"},
    "#9":  {"Cycle life": 2651, "C1": "5C", "C2": "2C", "C3": "5C", "C4": "1C"},
    "#10": {"Cycle life": 1751, "C1": "5C", "C2": "3C", "C3": "1C", "C4": "1C"},
    "#11": {"Cycle life": 1499, "C1": "5C", "C2": "3C", "C3": "2C", "C4": "1C"},
    "#12": {"Cycle life": 1386, "C1": "5C", "C2": "3C", "C3": "3C", "C4": "1C"},
    "#13": {"Cycle life": 1572, "C1": "5C", "C2": "3C", "C3": "4C", "C4": "1C"},
    "#14": {"Cycle life": 2202, "C1": "5C", "C2": "3C", "C3": "5C", "C4": "1C"},
    "#15": {"Cycle life": 1481, "C1": "5C", "C2": "4C", "C3": "1C", "C4": "1C"},
    "#16": {"Cycle life": 1938, "C1": "5C", "C2": "4C", "C3": "2C", "C4": "1C"},
    "#17": {"Cycle life": 2283, "C1": "5C", "C2": "4C", "C3": "3C", "C4": "1C"},
    "#18": {"Cycle life": 1649, "C1": "5C", "C2": "4C", "C3": "4C", "C4": "1C"},
    "#19": {"Cycle life": 1766, "C1": "5C", "C2": "4C", "C3": "5C", "C4": "1C"},
    "#20": {"Cycle life": 2657, "C1": "5C", "C2": "5C", "C3": "1C", "C4": "1C"},
    "#21": {"Cycle life": 2491, "C1": "5C", "C2": "5C", "C3": "2C", "C4": "1C"},
    "#22": {"Cycle life": 2479, "C1": "5C", "C2": "5C", "C3": "3C", "C4": "1C"},
    "#23": {"Cycle life": 2342, "C1": "5C", "C2": "5C", "C3": "4C", "C4": "1C"},
    "#24": {"Cycle life": 2217, "C1": "5C", "C2": "5C", "C3": "5C", "C4": "1C"},
    "#25": {"Cycle life": 1782, "C1": "4C", "C2": "1C", "C3": "1C", "C4": "1C"},
    "#26": {"Cycle life": 1142, "C1": "4C", "C2": "1C", "C3": "2C", "C4": "1C"},
    "#27": {"Cycle life": 1491, "C1": "4C", "C2": "1C", "C3": "3C", "C4": "1C"},
    "#28": {"Cycle life": 1561, "C1": "4C", "C2": "1C", "C3": "4C", "C4": "1C"},
    "#29": {"Cycle life": 1380, "C1": "4C", "C2": "1C", "C3": "5C", "C4": "1C"},
    "#30": {"Cycle life": 2216, "C1": "4C", "C2": "2C", "C3": "1C", "C4": "1C"},
    "#31": {"Cycle life": 1706, "C1": "4C", "C2": "2C", "C3": "2C", "C4": "1C"},
    "#32": {"Cycle life": 2507, "C1": "4C", "C2": "2C", "C3": "3C", "C4": "1C"},
    "#33": {"Cycle life": 1926, "C1": "4C", "C2": "2C", "C3": "4C", "C4": "1C"},
    "#34": {"Cycle life": 2689, "C1": "4C", "C2": "2C", "C3": "5C", "C4": "1C"},
    "#35": {"Cycle life": 1962, "C1": "4C", "C2": "3C", "C3": "1C", "C4": "1C"},
    "#36": {"Cycle life": 1583, "C1": "4C", "C2": "3C", "C3": "2C", "C4": "1C"},
    "#37": {"Cycle life": 2460, "C1": "4C", "C2": "3C", "C3": "3C", "C4": "1C"},
    "#38": {"Cycle life": 1448, "C1": "4C", "C2": "3C", "C3": "4C", "C4": "1C"},
    "#39": {"Cycle life": 1609, "C1": "4C", "C2": "4C", "C3": "1C", "C4": "1C"},
    "#40": {"Cycle life": 1908, "C1": "4C", "C2": "4C", "C3": "2C", "C4": "1C"},
    "#41": {"Cycle life": 1804, "C1": "4C", "C2": "4C", "C3": "3C", "C4": "1C"},
    "#42": {"Cycle life": 1717, "C1": "4C", "C2": "4C", "C3": "4C", "C4": "1C"},
    "#43": {"Cycle life": 2178, "C1": "4C", "C2": "4C", "C3": "5C", "C4": "1C"},
    "#44": {"Cycle life": 2468, "C1": "4C", "C2": "5C", "C3": "1C", "C4": "1C"},
    "#45": {"Cycle life": 2450, "C1": "4C", "C2": "5C", "C3": "3C", "C4": "1C"},
    "#46": {"Cycle life": 1690, "C1": "4C", "C2": "5C", "C3": "4C", "C4": "1C"},
    "#47": {"Cycle life": 2030, "C1": "4C", "C2": "5C", "C3": "5C", "C4": "1C"},
    "#48": {"Cycle life": 1295, "C1": "3C", "C2": "1C", "C3": "1C", "C4": "1C"},
    "#49": {"Cycle life": 1393, "C1": "3C", "C2": "1C", "C3": "2C", "C4": "1C"},
    "#50": {"Cycle life": 1875, "C1": "3C", "C2": "1C", "C3": "3C", "C4": "1C"},
    "#51": {"Cycle life": 1419, "C1": "3C", "C2": "1C", "C3": "4C", "C4": "1C"},
    "#52": {"Cycle life": 1685, "C1": "3C", "C2": "1C", "C3": "5C", "C4": "1C"},
    "#53": {"Cycle life": 1938, "C1": "3C", "C2": "2C", "C3": "1C", "C4": "1C"},
    "#54": {"Cycle life": 1308, "C1": "3C", "C2": "2C", "C3": "2C", "C4": "1C"},
    "#55": {"Cycle life": 2041, "C1": "3C", "C2": "2C", "C3": "3C", "C4": "1C"},
    "#56": {"Cycle life": 2290, "C1": "3C", "C2": "2C", "C3": "4C", "C4": "1C"},
    "#57": {"Cycle life": 1885, "C1": "3C", "C2": "2C", "C3": "5C", "C4": "1C"},
    "#58": {"Cycle life": 1348, "C1": "3C", "C2": "3C", "C3": "1C", "C4": "1C"},
    "#59": {"Cycle life": 2365, "C1": "3C", "C2": "3C", "C3": "2C", "C4": "1C"},
    "#60": {"Cycle life": 2047, "C1": "3C", "C2": "3C", "C3": "3C", "C4": "1C"},
    "#61": {"Cycle life": 1679, "C1": "3C", "C2": "3C", "C3": "4C", "C4": "1C"},
    "#62": {"Cycle life": 2057, "C1": "3C", "C2": "3C", "C3": "5C", "C4": "1C"},
    "#63": {"Cycle life": 2143, "C1": "3C", "C2": "4C", "C3": "1C", "C4": "1C"},
    "#64": {"Cycle life": 1905, "C1": "3C", "C2": "4C", "C3": "2C", "C4": "1C"},
    "#65": {"Cycle life": 1975, "C1": "3C", "C2": "4C", "C3": "3C", "C4": "1C"},
    "#66": {"Cycle life": 2168, "C1": "3C", "C2": "4C", "C3": "4C", "C4": "1C"},
    "#67": {"Cycle life": 1742, "C1": "3C", "C2": "4C", "C3": "5C", "C4": "1C"},
    "#68": {"Cycle life": 2012, "C1": "3C", "C2": "5C", "C3": "1C", "C4": "1C"},
    "#69": {"Cycle life": 2308, "C1": "3C", "C2": "5C", "C3": "2C", "C4": "1C"},
    "#70": {"Cycle life": 1702, "C1": "3C", "C2": "5C", "C3": "3C", "C4": "1C"},
    "#71": {"Cycle life": 1697, "C1": "3C", "C2": "5C", "C3": "4C", "C4": "1C"},
    "#72": {"Cycle life": 1848, "C1": "3C", "C2": "5C", "C3": "5C", "C4": "1C"},
    "#73": {"Cycle life": 1811, "C1": "2C", "C2": "4C", "C3": "1C", "C4": "1C"},
    "#74": {"Cycle life": 2030, "C1": "2C", "C2": "5C", "C3": "2C", "C4": "1C"},
    "#75": {"Cycle life": 2285, "C1": "2C", "C2": "3C", "C3": "3C", "C4": "1C"},
    "#76": {"Cycle life": 1783, "C1": "2C", "C2": "2C", "C3": "4C", "C4": "1C"},
    "#77": {"Cycle life": 1400, "C1": "2C", "C2": "1C", "C3": "5C", "C4": "1C"},
}


class HustFeatureExtractor(BaseFeatureExtractor):
    CHARGE_SLOPES = [(0.10, 0.30), (0.10, 0.50), (0.10, 0.80)]
    DISCHARGE_SLOPES = [(0.10, 0.30), (0.10, 0.50), (0.10, 0.80)]
    TEVI_INTERVALS = [(3.0, 3.2), (3.2, 3.4), (3.4, 3.5)]
    TEVD_INTERVALS = [(3.0, 2.8), (2.8, 2.5), (2.5, 2.2)]

    # LFP (A123) IC/DV config
    _LFP_IC_CONFIG = {
        "peak_mode": 1,
        "nominal_capacity": 1.1,
        "window_length_ic": 51,
        "window_length_dv": 21,
        "peak_height_ic": 0.01,
        "voltage_range_ic": (3.0, 3.6),
        "prominence_ic": 0.02,
        "ic_step_size": 0.001,
        "dv_step_size": 1.1 * 0.005,
        "search_window_dvv": 0.1,
        "search_window_dvp": 0.1,
        "initial_capacity_cut_fraction": 0.05,
        "icv_search_offset_lower": 0.05,
        "icv_search_offset_upper": 0.5,
        "disable_dvv": True,
        "dvp_capacity_range": (1.05, 1.15),
        "ic_area_config": {"method": "fixed_width", "width_v": 0.05},
    }

    def get_cycles_to_process(
        self,
        battery_data: Dict[str, Any],
        num_cycles: Optional[int],
    ) -> List[Dict[str, Any]]:
        cycles = list(battery_data.get("cycle_data") or [])
        # Skip corrupt first two cycles for cell HUST_7-5
        if battery_data.get("cell_id") == "HUST_7-5":
            cycles = cycles[2:]
        if num_cycles is not None and num_cycles > 0:
            cycles = cycles[:num_cycles]
        return cycles

    def extract_cycle_features(
        self,
        cycle_data: Dict[str, Any],
        battery_data: Dict[str, Any],
        output_dir: Path,
    ) -> Dict[str, Any]:
        cycle_df = self.build_cycle_frame(cycle_data)
        cycle_num = cycle_data.get("cycle_number", 0)
        charge_df, discharge_df, rest_df = self.split_phase_frames(cycle_df)
        discharge_df = self.trim_discharge_to_voltage_limit(discharge_df, DISCHARGE_CUTOFF_V)

        # --- Direct features ---
        direct_features = self.calculate_capacity_energy_features(
            cycle_num, charge_df, discharge_df
        )
        direct_features.update(
            self.calculate_charge_phase_features(
                charge_df,
                uvp=CHARGE_CUTOFF_V,
                time_mode="duration",
                cv_voltage_tolerance=0.01,
            )
        )
        direct_features.update(
            self.calculate_discharge_phase_features(
                discharge_df,
                lvp=DISCHARGE_CUTOFF_V,
                time_mode="duration",
            )
        )

        # --- HUST multi-stage features ---
        chg_stages = self._detect_stages(charge_df)
        direct_features.update(self._stage_features(chg_stages, "charge", n_stages=2))
        if len(chg_stages) >= 2:
            # Voltage jump at C1→C2 transition (internal resistance indicator)
            idx = chg_stages[1]["start_iloc"]
            v_before = charge_df["Voltage(V)"].iloc[max(0, idx - 1)]
            v_after = charge_df["Voltage(V)"].iloc[idx]
            direct_features["resistance_jump_dV"] = abs(v_before - v_after)
            t1 = direct_features.get("charge_time_1", 0)
            t2 = direct_features.get("charge_time_2", 1)
            direct_features["charge_time_ratio_1_2"] = t1 / t2 if t2 > 0 else 0.0
        else:
            direct_features.setdefault("resistance_jump_dV", 0.0)
            direct_features.setdefault("charge_time_ratio_1_2", 0.0)

        dis_stages = self._detect_stages(discharge_df)
        direct_features.update(self._stage_features(dis_stages, "discharge", n_stages=4))

        # --- IC/DV on low-rate charge phase (C2) ---
        ic_df = self._build_ic_input_df(charge_df, chg_stages)
        derivative_features = extract_ic_dv_features(
            ic_df,
            config=self._LFP_IC_CONFIG,
            plot_params=self.build_plot_params(battery_data, cycle_num, output_dir),
        )

        # --- Advanced features ---
        advanced_features = self.calculate_advanced_features_common(
            cycle_df, charge_df, discharge_df, direct_features,
            rest_df=rest_df, compute_cv_tau=True, cv_voltage_tolerance=0.01,
        )

        # --- Anchor features (charge slopes + TEVI only; discharge slopes disabled) ---
        anchor_features = self.calculate_anchor_features_common(
            charge_df, discharge_df,
            self.CHARGE_SLOPES, [],
            self.TEVI_INTERVALS, [],
        )

        return {**direct_features, **derivative_features, **advanced_features, **anchor_features}

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _detect_stages(
        df: pd.DataFrame,
        i_thresh: float = 0.5,
        v_thresh: float = 0.01,
        min_len: int = 5,
    ) -> List[Dict[str, Any]]:
        """Detect operational stages from current/voltage step changes."""
        if df.empty:
            return []
        curr = df["Current(A)"].values
        volt = df["Voltage(V)"].values
        time = df["Time(s)"].values
        dI = np.abs(np.diff(curr, prepend=curr[0]))
        dV = np.abs(np.diff(volt, prepend=volt[0]))
        is_step = (dI > i_thresh) | ((dI < 0.1) & (dV > v_thresh))
        step_indices = np.where(is_step)[0]
        step_indices = step_indices[step_indices > 0]
        boundaries = [0] + sorted(set(step_indices.tolist())) + [len(df)]
        stages = []
        for i in range(len(boundaries) - 1):
            s, e = boundaries[i], boundaries[i + 1]
            if e - s < min_len:
                continue
            stages.append({
                "start_iloc": s,
                "end_iloc": e,
                "duration": time[e - 1] - time[s],
                "current": float(np.mean(curr[s:e])),
            })
        return stages

    @staticmethod
    def _stage_features(
        stages: List[Dict[str, Any]],
        phase: str,
        n_stages: int,
    ) -> Dict[str, Any]:
        """Build per-stage current/time features."""
        feats: Dict[str, Any] = {}
        for i in range(1, n_stages + 1):
            if i <= len(stages):
                feats[f"{phase}_current_{i}"] = abs(stages[i - 1]["current"])
                feats[f"{phase}_time_{i}"] = stages[i - 1]["duration"]
            else:
                feats[f"{phase}_current_{i}"] = 0.0
                feats[f"{phase}_time_{i}"] = 0.0
        return feats

    @staticmethod
    def _build_ic_input_df(
        charge_df: pd.DataFrame,
        stages: List[Dict[str, Any]],
    ) -> pd.DataFrame:
        """Select the low-rate C2 segment and remap capacity column."""
        if len(stages) >= 2:
            s, e = stages[1]["start_iloc"], stages[1]["end_iloc"]
            target = charge_df.iloc[s:e].copy()
        else:
            target = charge_df.copy()
        if not target.empty and "Charge_Capacity(Ah)" in target.columns:
            target["Discharge_Capacity(Ah)"] = target["Charge_Capacity(Ah)"]
        return target

    # ------------------------------------------------------------------ overrides

    def handle_cycle_error(self, cell_id: str, exc: Exception) -> None:
        traceback.print_exc()

    def order_columns(self, features_df: pd.DataFrame) -> pd.DataFrame:
        priority = [
            "Cycle_Number",
            "Discharge_Capacity", "Charge_Capacity",
            "Discharge_Energy", "Charge_Energy",
            "Coulombic_Efficiency", "Energy_Efficiency",
            "charge_current_1", "charge_time_1",
            "charge_current_2", "charge_time_2",
            "charge_time_ratio_1_2", "resistance_jump_dV",
            "discharge_current_1", "discharge_time_1",
            "discharge_current_2", "discharge_time_2",
            "discharge_current_3", "discharge_time_3",
            "discharge_current_4", "discharge_time_4",
            "ICHV", "UVP_time", "TCCC", "TCVC", "CV_Current_Tau", "UVP",
            "IDV", "LVP_time", "total_discharge_time", "LVP",
            "ICP", "ICPL_V", "ICP_FWHM", "ICP_Area",
            "ICV", "ICVL_V",
            "DVP", "DVPL_V", "DVP_FWHM", "DVP_Area",
            "Internal_Resistance", "RCV",
            "charge_slope_1", "charge_slope_2", "charge_slope_3",
            "TEVI_1", "TEVI_2", "TEVI_3",
        ]
        existing = [c for c in priority if c in features_df.columns]
        remaining = sorted(c for c in features_df.columns if c not in existing)
        return features_df[existing + remaining]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_cycles", type=int, default=None, help="Number of cycles to process (default: all)")
    args = parser.parse_args()
    input_dir = resolve_dataset_input_dir(project_root, "HUST", "data/HUST")
    output_dir = project_root / "results" / "features" / "HUST"
    HustFeatureExtractor().run(input_dir, output_dir, num_cycles=args.num_cycles)



if __name__ == "__main__":
    main()

"""
Feature Extraction Script for UL-PUR Battery Dataset.

UL-PUR has two unique features vs. other datasets:
1. Discharge voltage glitch stitching (_preprocess_discharge_data) to handle
   relay interruptions in the data.
2. Enhanced thermal features (smoothed dT/dt, thermal load) from Temperature column.
"""

import argparse
import warnings
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from scipy.signal import savgol_filter

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from features.base_extractor import BaseFeatureExtractor, resolve_dataset_input_dir
from src.physics.ic_dv_extractor import extract_ic_dv_features

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class ULPURFeatureExtractor(BaseFeatureExtractor):
    CHARGE_SLOPES = [(0.10, 0.30), (0.10, 0.50), (0.10, 0.80)]
    DISCHARGE_SLOPES = [(0.10, 0.30), (0.10, 0.50), (0.10, 0.80)]
    TEVI_INTERVALS = [(3.5, 3.7), (3.7, 3.9), (3.9, 4.1)]
    TEVD_INTERVALS = [(4.1, 3.9), (3.9, 3.7), (3.7, 3.5)]

    def build_cycle_frame(self, cycle_data: Dict[str, Any]) -> pd.DataFrame:
        """Include Temperature column if present."""
        df = pd.DataFrame({
            "Time(s)": cycle_data["time_in_s"],
            "Current(A)": cycle_data["current_in_A"],
            "Voltage(V)": cycle_data["voltage_in_V"],
            "Charge_Capacity(Ah)": cycle_data["charge_capacity_in_Ah"],
            "Discharge_Capacity(Ah)": cycle_data["discharge_capacity_in_Ah"],
        })
        if "temperature_in_C" in cycle_data and cycle_data["temperature_in_C"] is not None:
            df["Temperature(C)"] = cycle_data["temperature_in_C"]
        return df

    def split_phase_frames(self, cycle_df: pd.DataFrame):
        """Override to use 0.01 A threshold (UL-PUR noise) and preprocess discharge."""
        charge_df = cycle_df[cycle_df["Current(A)"] > 0.01].copy()
        discharge_df = cycle_df[cycle_df["Current(A)"] < -0.01].copy()
        rest_df = pd.DataFrame()
        # Preprocess discharge to stitch voltage rebound glitches
        if not discharge_df.empty:
            discharge_df = self._preprocess_discharge_data(discharge_df)
        return charge_df, discharge_df, rest_df

    def extract_cycle_features(
        self,
        cycle_data: Dict[str, Any],
        battery_data: Dict[str, Any],
        output_dir: Path,
    ) -> Optional[Dict[str, Any]]:
        cycle_df = self.build_cycle_frame(cycle_data)
        cycle_num = cycle_data.get("cycle_number", 0)
        charge_df, discharge_df, _ = self.split_phase_frames(cycle_df)

        lvp = battery_data.get("min_voltage_limit_in_V", 2.5)
        uvp = battery_data.get("max_voltage_limit_in_V", 4.2)
        discharge_df = self.trim_discharge_to_voltage_limit(discharge_df, lvp)
        nominal_cap = battery_data.get("nominal_capacity_in_Ah", 2.0)

        direct_features = self.calculate_capacity_energy_features(
            cycle_num, charge_df, discharge_df
        )
        direct_features["charge_c_rate"] = battery_data.get("charge_protocol", [{}])[0].get(
            "rate_in_C", 0.0
        )
        direct_features["discharge_c_rate"] = battery_data.get("discharge_protocol", [{}])[0].get(
            "rate_in_C", 0.0
        )
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
        # Thermal features (unique to UL-PUR)
        direct_features.update(self._calculate_enhanced_thermal_features(charge_df, "charge"))
        direct_features.update(self._calculate_enhanced_thermal_features(discharge_df, "discharge"))

        derivative_features = extract_ic_dv_features(
            discharge_df,
            config={
                "peak_mode": 2,
                "nominal_capacity": nominal_cap,
                "window_length_ic": 21,
                "window_length_dv": 31,
                "peak_height_ic": 0.01,
                "voltage_range_ic": (3.1, 3.7),
                "prominence_ic": 0.02,
                "ic_step_size": 0.002,
                "dv_step_size": nominal_cap * 0.005,
                "search_window_dvv": 0.1,
                "search_window_dvp": 0.1,
                "initial_capacity_cut_fraction": 0.02,
                "icv_search_offset_lower": 0.02,
                "icv_search_offset_upper": 0.1,
                "ic_area_config": {"method": "fixed_width", "width_v": 0.05},
            },
            plot_params=self.build_plot_params(battery_data, cycle_num, output_dir),
        )
        # Disable unreliable FWHM and DVP features for this dataset
        derivative_features.pop("ICP_FWHM", None)
        derivative_features["ICP_FWHM"] = 0.0
        for k in list(derivative_features):
            if "DVP" in k:
                derivative_features[k] = 0.0

        advanced_features = self.calculate_advanced_features_common(
            cycle_df, charge_df, discharge_df, direct_features, compute_cv_tau=True,
        )
        anchor_features = self.calculate_anchor_features_common(
            charge_df, discharge_df,
            self.CHARGE_SLOPES, self.DISCHARGE_SLOPES,
            self.TEVI_INTERVALS, self.TEVD_INTERVALS,
        )
        return {**direct_features, **derivative_features, **advanced_features, **anchor_features}

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _preprocess_discharge_data(df: pd.DataFrame) -> pd.DataFrame:
        """Stitch voltage-rebound glitches and keep the longest continuous segment."""
        if df.empty:
            return df
        idx_series = df.index.to_series()
        gaps = idx_series.diff() > 1
        if not gaps.any():
            return df
        gap_ilocs = [df.index.get_loc(i) for i in df.index[gaps]]
        boundaries = [0] + gap_ilocs + [len(df)]
        segments = [df.iloc[boundaries[i]: boundaries[i + 1]].copy()
                    for i in range(len(boundaries) - 1)]
        chains: List[List[pd.DataFrame]] = []
        current_chain = [segments[0]]
        for i in range(len(segments) - 1):
            pre, post = segments[i], segments[i + 1]
            dt = post["Time(s)"].iloc[0] - pre["Time(s)"].iloc[-1]
            if dt < 10.0:
                dv = post["Voltage(V)"].iloc[0] - pre["Voltage(V)"].iloc[-1]
                if dv > 0:
                    post["Voltage(V)"] -= dv
                current_chain.append(post)
            else:
                chains.append(current_chain)
                current_chain = [post]
        chains.append(current_chain)
        best = max(chains, key=lambda c: sum(len(s) for s in c))
        return pd.concat(best)

    @staticmethod
    def _calculate_enhanced_thermal_features(
        df: pd.DataFrame, phase: str
    ) -> Dict[str, float]:
        keys = [
            f"MAT_{phase}", f"MET_{phase}", f"MinT_{phase}",
            f"T_rise_{phase}", f"Max_HeatRate_{phase}",
            f"Mean_HeatRate_{phase}", f"Thermal_Load_{phase}",
        ]
        default: Dict[str, float] = {k: 0.0 for k in keys}
        if "Temperature(C)" not in df.columns or df.empty or len(df) < 5:
            return default
        temp_raw = df["Temperature(C)"].values
        time_arr = df["Time(s)"].values
        wl = min(len(temp_raw), 51)
        if wl % 2 == 0:
            wl -= 1
        try:
            temp_sm = savgol_filter(temp_raw, window_length=max(wl, 5), polyorder=3) if wl >= 5 else temp_raw
        except ValueError:
            temp_sm = temp_raw
        feats = {
            f"MAT_{phase}": float(np.max(temp_sm)),
            f"MET_{phase}": float(np.mean(temp_sm)),
            f"MinT_{phase}": float(np.min(temp_sm)),
            f"T_rise_{phase}": float(temp_sm[-1] - temp_sm[0]),
            f"Max_HeatRate_{phase}": 0.0,
            f"Mean_HeatRate_{phase}": 0.0,
            f"Thermal_Load_{phase}": 0.0,
        }
        if len(time_arr) > 5:
            dT_dt = np.gradient(temp_sm, time_arr)
            feats[f"Max_HeatRate_{phase}"] = float(np.max(dT_dt))
            feats[f"Mean_HeatRate_{phase}"] = float(np.mean(dT_dt))
        try:
            feats[f"Thermal_Load_{phase}"] = float(trapezoid(y=temp_sm, x=time_arr))
        except Exception:
            pass
        return feats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_cycles", type=int, default=None, help="Number of cycles to process (default: all)")
    args = parser.parse_args()
    input_dir = resolve_dataset_input_dir(project_root, "UL_PUR", "data/UL_PUR")
    output_dir = project_root / "results" / "features" / "UL_PUR"
    ULPURFeatureExtractor().run(input_dir, output_dir, num_cycles=args.num_cycles)



if __name__ == "__main__":
    main()

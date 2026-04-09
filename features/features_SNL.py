"""
Feature Extraction Script for SNL Battery Dataset.

SNL data requires special cleaning:
1. _get_shape_stream_charge: voltage-threshold filter + time re-zeroing.
2. _get_shape_stream_discharge: longest-block selection + head/tail truncation
   (ghost points at start, voltage rise artifact at end).

SNL uses both discharge IC/DV and charge IC/DV (LFP phase detection).
"""

import argparse
import warnings
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from features.base_extractor import BaseFeatureExtractor, resolve_dataset_input_dir
from src.physics.ic_dv_extractor import extract_ic_dv_features

try:
    from src.physics.ic_dv_extractor import extract_charge_ic_dv_features  # type: ignore
    _HAS_CHARGE_IC = True
except ImportError:
    _HAS_CHARGE_IC = False

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Default SNL IC/DV config (LFP / NCM mixed dataset)
_DEFAULT_IC_CONFIG = {
    "peak_mode": 1,
    "nominal_capacity": 3.0,
    "window_length_ic": 31,
    "window_length_dv": 21,
    "peak_height_ic": 0.01,
    "voltage_range_ic": (3.0, 3.8),
    "prominence_ic": 0.02,
    "ic_step_size": 0.001,
    "dv_step_size": 3.0 * 0.005,
    "search_window_dvv": 0.1,
    "search_window_dvp": 0.1,
    "initial_capacity_cut_fraction": 0.02,
    "icv_search_offset_lower": 0.05,
    "icv_search_offset_upper": 0.5,
    "ic_area_config": {"method": "fixed_width", "width_v": 0.05},
}


class SNLFeatureExtractor(BaseFeatureExtractor):
    CHARGE_SLOPES = [(0.10, 0.30), (0.10, 0.50), (0.10, 0.80)]
    DISCHARGE_SLOPES = [(0.10, 0.30), (0.10, 0.50), (0.10, 0.80)]
    TEVI_INTERVALS = [(3.0, 3.2), (3.2, 3.4), (3.4, 3.6)]
    TEVD_INTERVALS = [(3.6, 3.4), (3.4, 3.2), (3.2, 3.0)]

    def split_phase_frames(self, cycle_df: pd.DataFrame):
        """Override to apply SNL-specific cleaning to charge and discharge."""
        raw_charge = cycle_df[cycle_df["Current(A)"] > 0].copy()
        raw_discharge = cycle_df[cycle_df["Current(A)"] < 0].copy()
        charge_df = self._get_shape_stream_charge(raw_charge)
        discharge_df = self._get_shape_stream_discharge(raw_discharge)
        rest_df = pd.DataFrame()
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

        lvp = battery_data.get("min_voltage_limit_in_V", 2.0)
        uvp = battery_data.get("max_voltage_limit_in_V", 3.6)
        discharge_df = self.trim_discharge_to_voltage_limit(discharge_df, lvp)
        nominal_cap = battery_data.get("nominal_capacity_in_Ah", 3.0)

        direct_features = self.calculate_capacity_energy_features(
            cycle_num, charge_df, discharge_df
        )
        direct_features["charge_c_rate"] = self.get_protocol_rate(
            battery_data.get("charge_protocol")
        )
        direct_features["discharge_c_rate"] = self.get_protocol_rate(
            battery_data.get("discharge_protocol")
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

        ic_config = dict(_DEFAULT_IC_CONFIG)
        ic_config["nominal_capacity"] = nominal_cap
        ic_config["dv_step_size"] = nominal_cap * 0.005
        plot_params = self.build_plot_params(battery_data, cycle_num, output_dir)

        derivative_features = extract_ic_dv_features(
            discharge_df, config=ic_config, plot_params=plot_params,
        )
        # Supplementary charge IC/DV if available
        if _HAS_CHARGE_IC:
            try:
                chg_ic = extract_charge_ic_dv_features(
                    charge_df, config=ic_config, plot_params=plot_params
                )
                for k, v in chg_ic.items():
                    derivative_features[f"chg_{k}"] = v
            except Exception:
                pass

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
    def _get_shape_stream_charge(
        raw_charge_df: pd.DataFrame,
        v_thresh: float = 2.5,
    ) -> pd.DataFrame:
        if raw_charge_df.empty:
            return pd.DataFrame()
        mask = (raw_charge_df["Voltage(V)"] >= v_thresh) & (raw_charge_df["Current(A)"] > 0.01)
        clean_df = raw_charge_df[mask].copy()
        if clean_df.empty:
            return pd.DataFrame()
        clean_df["Time(s)"] = clean_df["Time(s)"] - clean_df["Time(s)"].iloc[0]
        return clean_df

    @staticmethod
    def _get_shape_stream_discharge(raw_discharge_df: pd.DataFrame) -> pd.DataFrame:
        if raw_discharge_df.empty:
            return pd.DataFrame()
        mask = raw_discharge_df["Current(A)"] < -0.005
        if not mask.any():
            return pd.DataFrame()
        df_temp = raw_discharge_df.copy()
        df_temp["block_id"] = (mask != mask.shift()).cumsum()
        valid_blocks = df_temp[mask]
        if valid_blocks.empty:
            return pd.DataFrame()
        best_block_id = valid_blocks["block_id"].value_counts().idxmax()
        clean_df = valid_blocks[valid_blocks["block_id"] == best_block_id].copy()
        # Head truncation: clear ghost low-voltage points before real discharge
        if not clean_df.empty:
            max_v_loc = clean_df["Voltage(V)"].argmax()
            if max_v_loc < len(clean_df) - 1:
                clean_df = clean_df.iloc[max_v_loc:].copy()
        # Tail truncation: remove voltage rise at end of discharge
        if not clean_df.empty:
            min_v_loc = clean_df["Voltage(V)"].argmin()
            if min_v_loc > 10:
                clean_df = clean_df.iloc[: min_v_loc + 1].copy()
        if clean_df.empty:
            return pd.DataFrame()
        clean_df["Time(s)"] = clean_df["Time(s)"] - clean_df["Time(s)"].iloc[0]
        return clean_df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_cycles", type=int, default=None, help="Number of cycles to process (default: all)")
    args = parser.parse_args()
    input_dir = resolve_dataset_input_dir(project_root, "SNL", "data/SNL")
    output_dir = project_root / "results" / "features" / "SNL"
    SNLFeatureExtractor().run(input_dir, output_dir, num_cycles=args.num_cycles)



if __name__ == "__main__":
    main()

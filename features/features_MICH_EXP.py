"""
Feature Extraction Script for MICH_EXP Battery Dataset.

MICH_EXP uses a complex per-cell IC/DV configuration (based on cell group
inferred from cell_id), a robust rolling-window charge/discharge phase
detection algorithm, and RPT cycle filtering.

AttrDict pkl format.
"""

import argparse
import re
import warnings
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from features.base_extractor import BaseFeatureExtractor, resolve_dataset_input_dir
from src.physics.ic_dv_extractor import extract_ic_dv_features

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

_CURRENT_THRESHOLD = 1e-3
_VOLTAGE_MIN = 2.5


class _AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


class MICHEXPFeatureExtractor(BaseFeatureExtractor):
    CHARGE_SLOPES = [(0.10, 0.30), (0.10, 0.50), (0.10, 0.80)]
    DISCHARGE_SLOPES = [(0.10, 0.30), (0.10, 0.50), (0.10, 0.80)]
    TEVI_INTERVALS = [(3.5, 3.7), (3.7, 3.9), (3.9, 4.1)]
    TEVD_INTERVALS = [(4.1, 3.9), (3.9, 3.7), (3.7, 3.5)]

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

    def build_cycle_frame(self, cycle_data) -> pd.DataFrame:
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

    def process_battery(
        self,
        file_path: Path,
        output_dir: Path,
        num_cycles: Optional[int] = None,
    ) -> None:
        """Override to apply RPT filtering after extract_cycle_features."""
        try:
            battery_data = self.load_battery_data(file_path)
        except Exception as exc:
            print(f"Error loading {file_path}: {exc}")
            return

        cell_id = self.get_cell_id(battery_data, file_path)
        cycles_to_process = self.get_cycles_to_process(battery_data, num_cycles)
        all_cycle_features = []
        valid_count = 0

        for cycle_data in tqdm(cycles_to_process, desc=f"Processing {cell_id}", leave=False):
            if not getattr(cycle_data, "time_in_s", None):
                continue
            try:
                features = self.extract_cycle_features(cycle_data, battery_data, output_dir)
            except Exception as exc:
                self.handle_cycle_error(cell_id, exc)
                continue
            if not features:
                continue
            # RPT filtering: skip 0–100% calibration cycles in 50–100% files
            if features.get("soc") == 50 and features.get("ICHV", 100) < 3.55:
                print(f"  [Skipping Cycle {getattr(cycle_data, 'cycle_number', '?')}] "
                    f"RPT Detected (ICHV={features.get('ICHV'):.3f}V < 3.55V)")
                continue
            valid_count += 1
            features["Cycle_Number"] = valid_count
            all_cycle_features.append(features)

        if not all_cycle_features:
            print(f"Warning: No features extracted for {cell_id}")
            return
        output_file = output_dir / f"{cell_id}.csv"
        self.order_columns(self.build_feature_frame(all_cycle_features)).to_csv(
            output_file, index=False
        )
        print(f"Features for {cell_id} saved to {output_file}")

    def extract_cycle_features(
        self,
        cycle_data,
        battery_data: Dict[str, Any],
        output_dir: Path,
    ) -> Optional[Dict[str, Any]]:
        cycle_df = self.build_cycle_frame(cycle_data)
        cycle_num = getattr(cycle_data, "cycle_number", 0)
        cell_id = battery_data.get("cell_id", "Unknown")

        # Robust phase detection (rolling-window stable current)
        charge_df = self._detect_charge_phase(cycle_df)
        discharge_df = self._detect_discharge_phase(cycle_df)
        if not discharge_df.empty:
            lvp = getattr(battery_data, "min_voltage_limit_in_V", 2.5)
            cutoff = discharge_df.index[discharge_df["Voltage(V)"] <= lvp]
            if not cutoff.empty:
                discharge_df = discharge_df.loc[:cutoff[0]]

        # Rest between charge and discharge
        rest_between = pd.DataFrame()
        if not charge_df.empty and not discharge_df.empty:
            t_c_end = charge_df["Time(s)"].iloc[-1]
            t_d_start = discharge_df["Time(s)"].iloc[0]
            if t_d_start > t_c_end:
                rest_between = cycle_df[
                    (cycle_df["Time(s)"] > t_c_end)
                    & (cycle_df["Time(s)"] < t_d_start)
                    & (cycle_df["Current(A)"].abs() <= _CURRENT_THRESHOLD)
                ].copy()

        uvp = getattr(battery_data, "max_voltage_limit_in_V", 4.2)
        direct_features = self.calculate_capacity_energy_features(
            cycle_num, charge_df, discharge_df
        )
        try:
            direct_features["charge_c_rate"] = battery_data.charge_protocol[0].rate_in_C
            direct_features["discharge_c_rate"] = battery_data.discharge_protocol[0].rate_in_C
        except (AttributeError, IndexError, TypeError):
            direct_features["charge_c_rate"] = 0.0
            direct_features["discharge_c_rate"] = 0.0

        direct_features.update(
            self.calculate_charge_phase_features(
                charge_df, uvp=uvp, time_mode="duration", cv_voltage_tolerance=0.01,
            )
        )
        direct_features.update(
            self.calculate_discharge_phase_features(
                discharge_df, lvp=getattr(battery_data, "min_voltage_limit_in_V", 2.5),
                time_mode="duration",
            )
        )

        # Cell-specific IC config
        tevi_overrides, tevd_overrides = self._get_tevi_tevd_overrides(cell_id)
        ic_config = self._build_ic_config(cell_id, battery_data)
        derivative_features = extract_ic_dv_features(
            discharge_df,
            config=ic_config,
            plot_params=self.build_plot_params(battery_data, cycle_num, output_dir),
        )

        # Zero out IC features for 50% SOC cells
        soc_val = self._parse_soc(cell_id)
        if soc_val == 50:
            for k in ["ICP", "ICPL_V", "ICV", "ICVL_V", "DVP", "DVPL_V", "DVV", "DVVL_V"]:
                derivative_features[k] = 0.0

        advanced_features = self.calculate_advanced_features_common(
            cycle_df, charge_df, discharge_df, direct_features,
            rest_df=rest_between, compute_cv_tau=True, cv_voltage_tolerance=0.01,
        )
        anchor_features = self.calculate_anchor_features_common(
            charge_df, discharge_df,
            self.CHARGE_SLOPES, self.DISCHARGE_SLOPES,
            tevi_overrides or self.TEVI_INTERVALS,
            tevd_overrides or self.TEVD_INTERVALS,
        )
        final = {**direct_features, **derivative_features, **advanced_features, **anchor_features}
        final["temperature"] = self._parse_temperature(cell_id)
        final["soc"] = soc_val
        return final

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _detect_charge_phase(cycle_df: pd.DataFrame) -> pd.DataFrame:
        """Rolling-window stable-current charge detection."""
        prelim = cycle_df[
            (cycle_df["Current(A)"] > _CURRENT_THRESHOLD) &
            (cycle_df["Voltage(V)"] > _VOLTAGE_MIN)
        ].copy()
        if len(prelim) < 50:
            return pd.DataFrame()
        nom_current = prelim["Current(A)"].max()
        if nom_current <= 0.1:
            return pd.DataFrame()
        threshold = nom_current * 0.90
        is_stable = (prelim["Current(A)"] >= threshold)
        rolling_sum = is_stable.rolling(window=50, min_periods=50).sum()
        stable_ends = (rolling_sum == 50).to_numpy().nonzero()[0]
        voltages = prelim["Voltage(V)"].values
        times = prelim["Time(s)"].values
        for end_iloc in stable_ends:
            start_iloc = end_iloc - 49
            if (times[end_iloc] - times[start_iloc]) <= 1e-3:
                continue
            if (voltages[end_iloc] - voltages[start_iloc]) / (times[end_iloc] - times[start_iloc]) <= 1e-5:
                continue
            subset = prelim.iloc[start_iloc:].copy()
            # Truncate at large time gaps (>30 s)
            td = subset["Time(s)"].diff()
            gaps = td[td > 30.0].index
            if not gaps.empty:
                subset = subset.iloc[: subset.index.get_loc(gaps[0])]
            # Truncate at voltage drop > 50 mV below running maximum
            if not subset.empty:
                v_vals = subset["Voltage(V)"].values
                run_max = np.maximum.accumulate(v_vals)
                drop = np.where((run_max - v_vals) > 0.05)[0]
                if len(drop) > 0:
                    subset = subset.iloc[: drop[0]]
            return subset
        return pd.DataFrame()

    @staticmethod
    def _detect_discharge_phase(cycle_df: pd.DataFrame) -> pd.DataFrame:
        """Rolling-window stable-current discharge detection."""
        prelim = cycle_df[
            (cycle_df["Current(A)"] < -_CURRENT_THRESHOLD) &
            (cycle_df["Voltage(V)"] > _VOLTAGE_MIN)
        ].copy()
        if len(prelim) < 10:
            return pd.DataFrame()
        nom_current = prelim["Current(A)"].min()
        if nom_current >= -0.1:
            return pd.DataFrame()
        threshold = nom_current * 0.90
        is_stable = (prelim["Current(A)"] <= threshold)
        rolling_sum = is_stable.rolling(window=50, min_periods=50).sum()
        stable_ends = (rolling_sum == 50).to_numpy().nonzero()[0]
        voltages = prelim["Voltage(V)"].values
        for end_iloc in stable_ends:
            start_iloc = end_iloc - 49
            if voltages[end_iloc] > voltages[start_iloc]:
                continue
            subset = prelim.iloc[start_iloc:].copy()
            td = subset["Time(s)"].diff()
            gaps = td[td > 30.0].index
            if not gaps.empty:
                subset = subset.iloc[: subset.index.get_loc(gaps[0])]
            return subset
        return pd.DataFrame()

    @staticmethod
    def _build_ic_config(cell_id: str, battery_data: Any) -> Dict[str, Any]:
        special_mode2 = ["01", "02", "03"]
        high_range = ["13R", "14C", "15H", "16R", "18H"]
        mid_range = ["01", "02", "03"]
        peak_mode = 2 if any(x in cell_id for x in special_mode2) else 1
        if any(x in cell_id for x in high_range):
            ic_area_range = (3.8, 3.9)
        elif any(x in cell_id for x in mid_range):
            ic_area_range = (3.5, 3.7)
        else:
            ic_area_range = (3.4, 3.6)
        force_fwhm_zero = True
        force_icv_zero = not any(x in cell_id for x in ["01", "02", "03"])
        nominal_cap = battery_data.get("nominal_capacity_in_Ah", 2.0)
        return {
            "peak_mode": peak_mode,
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
            "icv_search_offset_upper": 0.2,
            "plot_interval": 50,
            "ic_area_voltage_range": ic_area_range,
            "force_icp_fwhm_zero": force_fwhm_zero,
            "force_icv_zero": force_icv_zero,
        }

    @staticmethod
    def _get_tevi_tevd_overrides(cell_id: str):
        high_range = ["13R", "14C", "15H", "16R", "18H"]
        if any(x in cell_id for x in high_range):
            return [(3.7, 3.85), (3.85, 4.00), (4.00, 4.15)], [(4.15, 4.00), (4.00, 3.85), (3.85, 3.7)]
        return None, None

    @staticmethod
    def _parse_temperature(cell_id: str) -> int:
        m = re.search(r"NMC_(\d+)C", cell_id)
        return int(m.group(1)) if m else 0

    @staticmethod
    def _parse_soc(cell_id: str) -> int:
        high_range = ["13R", "14C", "15H", "16R", "17C", "18H"]
        if any(x in cell_id for x in high_range):
            return 50
        if "0-100" in cell_id:
            return 100
        if "50-100" in cell_id:
            return 50
        return 100


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_cycles", type=int, default=None, help="Number of cycles to process (default: all)")
    args = parser.parse_args()
    input_dir = resolve_dataset_input_dir(project_root, "MICH_EXP", "data/MICH_EXP")
    output_dir = project_root / "results" / "features" / "MICH_EXP"
    MICHEXPFeatureExtractor().run(input_dir, output_dir, num_cycles=args.num_cycles)



if __name__ == "__main__":
    main()

"""
Feature Extraction Script for MATR Battery Dataset.

MATR uses A123 LFP 1.1Ah cells. Unique aspects:
1. Multi-stage CC charging with robust histogram-based stage detection
   (handles b1/b2 batch noise differences).
2. Enhanced thermal features from Temperature(C) column.
3. AttrDict-style pkl format (dot-notation metadata access).
"""

import argparse
import warnings
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from scipy.signal import find_peaks, medfilt, savgol_filter

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


class MATRFeatureExtractor(BaseFeatureExtractor):
    CHARGE_SLOPES = [(0.10, 0.30), (0.10, 0.50), (0.10, 0.80)]
    DISCHARGE_SLOPES = [(0.10, 0.30), (0.10, 0.50), (0.10, 0.80)]
    TEVI_INTERVALS = [(3.0, 3.2), (3.2, 3.4), (3.4, 3.55)]
    TEVD_INTERVALS = [(3.5, 3.3), (3.3, 3.1), (3.1, 2.9)]

    _LFP_IC_CONFIG = {
        "peak_mode": 1,
        "nominal_capacity": 1.1,
        "voltage_range_ic": (2.8, 3.8),
        "prominence_ic": 0.05,
        "window_length_ic": 31,
        "window_length_dv": 31,
        "plot_interval": 50,
        "disable_dvv": False,
        "search_window_dvv": 0.1,
        "search_window_dvp": 0.1,
        "ic_step_size": 0.001,
        "dv_step_size": 1.1 * 0.005,
        "initial_capacity_cut_fraction": 0.02,
        "icv_search_offset_lower": 0.05,
        "icv_search_offset_upper": 0.1,
        "ic_area_config": {"method": "fixed_width", "width_v": 0.03},
    }

    def load_battery_data(self, file_path: Path) -> Dict[str, Any]:
        import pickle
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        battery = _AttrDict(data)
        if battery.get("cycle_data"):
            battery["cycle_data"] = [_AttrDict(c) for c in battery["cycle_data"]]
        if battery.get("charge_protocol"):
            battery["charge_protocol"] = (
                [_AttrDict(p) for p in battery["charge_protocol"]]
                if isinstance(battery["charge_protocol"], list)
                else _AttrDict(battery["charge_protocol"])
            )
        if battery.get("discharge_protocol"):
            battery["discharge_protocol"] = (
                [_AttrDict(p) for p in battery["discharge_protocol"]]
                if isinstance(battery["discharge_protocol"], list)
                else _AttrDict(battery["discharge_protocol"])
            )
        return battery

    def build_cycle_frame(self, cycle_data) -> pd.DataFrame:
        df = pd.DataFrame({
            "Time(s)": cycle_data.time_in_s,
            "Current(A)": cycle_data.current_in_A,
            "Voltage(V)": cycle_data.voltage_in_V,
            "Charge_Capacity(Ah)": cycle_data.charge_capacity_in_Ah,
            "Discharge_Capacity(Ah)": cycle_data.discharge_capacity_in_Ah,
        })
        if hasattr(cycle_data, "temperature_in_C") and cycle_data.temperature_in_C is not None:
            df["Temperature(C)"] = cycle_data.temperature_in_C
        return df

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

        lvp = getattr(battery_data, "min_voltage_limit_in_V", 2.0)
        uvp = getattr(battery_data, "max_voltage_limit_in_V", 3.6)
        discharge_df = self.trim_discharge_to_voltage_limit(discharge_df, lvp)
        cell_id = getattr(battery_data, "cell_id", "")

        direct_features = self.calculate_capacity_energy_features(
            cycle_num, charge_df, discharge_df
        )
        # C-rates (AttrDict access)
        try:
            cp = battery_data.charge_protocol
            direct_features["charge_c_rate"] = (
                cp[0].rate_in_C if isinstance(cp, list) else cp.rate_in_C
            )
        except (AttributeError, IndexError):
            direct_features["charge_c_rate"] = 0.0
        try:
            dp = battery_data.discharge_protocol
            direct_features["discharge_c_rate"] = (
                dp[0].rate_in_C if isinstance(dp, list) else dp.rate_in_C
            )
        except (AttributeError, IndexError):
            direct_features["discharge_c_rate"] = 0.0

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

        # MATR multi-stage charge features
        chg_stages = self._detect_stages_robust(charge_df, cell_id)
        direct_features.update(self._stage_features(chg_stages, n_stages=3))
        # Override TCCC/TCVC based on detected stages
        if chg_stages and not charge_df.empty:
            v_max_limit = charge_df["Voltage(V)"].max()
            cc_dur = 0.0
            for s in chg_stages:
                vseg = charge_df["Voltage(V)"].iloc[s["start_iloc"]: s["end_iloc"]]
                if not vseg.empty and not (vseg.mean() > v_max_limit - 0.03 and vseg.std() < 0.015):
                    cc_dur += s["duration"]
            total_chg_time = charge_df["Time(s)"].iloc[-1] - charge_df["Time(s)"].iloc[0]
            direct_features["TCCC"] = cc_dur
            direct_features["TCVC"] = max(0.0, total_chg_time - cc_dur)

        # Thermal features
        direct_features.update(self._thermal_features(charge_df, "charge"))
        direct_features.update(self._thermal_features(discharge_df, "discharge"))

        derivative_features = extract_ic_dv_features(
            discharge_df,
            config=self._LFP_IC_CONFIG,
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

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _detect_stages_robust(
        df: pd.DataFrame,
        cell_id: str,
        min_duration: float = 60.0,
    ) -> List[Dict[str, Any]]:
        """Histogram clustering approach to detect CC charge stages robustly."""
        if df.empty:
            return []
        time = df["Time(s)"].values
        curr = df["Current(A)"].values
        is_b2 = "b2" in cell_id
        cluster_tol = 0.5 if is_b2 else 0.3
        stitch_gap = 120.0
        curr_smooth = medfilt(curr, kernel_size=15)
        valid_curr = curr_smooth[curr_smooth > 0.1]
        if len(valid_curr) == 0:
            return []
        max_val = max(valid_curr.max(), 1.0) * 1.1
        hist, bin_edges = np.histogram(valid_curr, bins=100, range=(0, max_val))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        peaks, _ = find_peaks(hist, distance=5, prominence=len(valid_curr) * 0.01)
        if len(peaks) == 0:
            return []
        dom_currents = np.sort(bin_centers[peaks])[::-1]
        labels = np.full(len(curr), -1, dtype=int)
        for i, tgt in enumerate(dom_currents):
            mask = np.abs(curr_smooth - tgt) < cluster_tol
            labels[mask] = i
        change_pts = np.where(np.diff(labels, prepend=labels[0] - 1) != 0)[0]
        raw_segs = []
        for k in range(len(change_pts)):
            s = change_pts[k]
            e = change_pts[k + 1] if k < len(change_pts) - 1 else len(labels)
            if labels[s] != -1:
                raw_segs.append({
                    "label": labels[s], "target_I": dom_currents[labels[s]],
                    "start_iloc": s, "end_iloc": e,
                    "start_time": time[s], "end_time": time[e - 1],
                })
        if not raw_segs:
            return []
        # Stitch segments
        merged = []
        cur_seg = raw_segs[0]
        for i in range(1, len(raw_segs)):
            nxt = raw_segs[i]
            gap = nxt["start_time"] - cur_seg["end_time"]
            same = nxt["label"] == cur_seg["label"]
            both_low = cur_seg["target_I"] < 0.5 and nxt["target_I"] < 0.5
            if (same and gap < stitch_gap) or (both_low and gap < 120.0):
                cur_seg["end_iloc"] = nxt["end_iloc"]
                cur_seg["end_time"] = nxt["end_time"]
            else:
                merged.append(cur_seg)
                cur_seg = nxt
        merged.append(cur_seg)
        final = []
        for seg in merged:
            duration = seg["end_time"] - seg["start_time"]
            if duration < min_duration:
                continue
            seg_curr = curr[seg["start_iloc"]: seg["end_iloc"]]
            valid = np.abs(seg_curr - seg["target_I"]) < cluster_tol * 2
            avg = float(np.mean(seg_curr[valid])) if valid.sum() > 0 else seg["target_I"]
            final.append({
                "start_iloc": seg["start_iloc"], "end_iloc": seg["end_iloc"],
                "duration": duration, "current": avg, "start_time": seg["start_time"],
            })
        final.sort(key=lambda x: x["start_time"])
        return final

    @staticmethod
    def _stage_features(stages: List[Dict], n_stages: int = 3) -> Dict[str, Any]:
        feats: Dict[str, Any] = {}
        for i in range(1, n_stages + 1):
            if i <= len(stages):
                feats[f"charge_current_{i}"] = abs(stages[i - 1]["current"])
                feats[f"charge_time_{i}"] = stages[i - 1]["duration"]
            else:
                feats[f"charge_current_{i}"] = 0.0
                feats[f"charge_time_{i}"] = 0.0
        return feats

    @staticmethod
    def _thermal_features(df: pd.DataFrame, phase: str) -> Dict[str, float]:
        keys = [f"MAT_{phase}", f"MET_{phase}", f"MinT_{phase}", f"T_rise_{phase}",
                f"Max_HeatRate_{phase}", f"Mean_HeatRate_{phase}", f"Thermal_Load_{phase}"]
        default: Dict[str, float] = {k: 0.0 for k in keys}
        if "Temperature(C)" not in df.columns or df.empty:
            return default
        try:
            temp_raw = df["Temperature(C)"].astype(float).values
        except ValueError:
            return default
        time_arr = df["Time(s)"].values
        if len(temp_raw) < 5 or np.isnan(temp_raw).all():
            return default
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
            dT = np.gradient(temp_sm)
            dt = np.gradient(time_arr)
            dt[dt == 0] = 1e-6
            dT_dt = dT / dt
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
    input_dir = resolve_dataset_input_dir(project_root, "MATR", "data/MATR")
    output_dir = project_root / "results" / "features" / "MATR"
    MATRFeatureExtractor().run(input_dir, output_dir, num_cycles=args.num_cycles)



if __name__ == "__main__":
    main()

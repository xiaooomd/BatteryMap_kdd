"""Shared template helpers for dataset feature extractors."""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from scipy.stats import skew
from tqdm import tqdm

from src.utils.math_tools import fit_cv_decay


class BaseFeatureExtractor:
    """Template-method base class for per-dataset feature extraction scripts."""

    DATASET_NAME = "base"

    def get_file_pattern(self) -> str:
        return "*.pkl"

    def load_battery_data(self, file_path: Path) -> Dict[str, Any]:
        with open(file_path, "rb") as handle:
            battery_data = pickle.load(handle)
        if "cycle_data" in battery_data and battery_data["cycle_data"] is not None:
            battery_data["cycle_data"] = list(battery_data["cycle_data"])
        return battery_data

    def get_cell_id(self, battery_data: Dict[str, Any], file_path: Path) -> str:
        return battery_data.get("cell_id", file_path.stem)

    def get_cycles_to_process(
        self,
        battery_data: Dict[str, Any],
        num_cycles: Optional[int],
    ) -> Iterable[Dict[str, Any]]:
        cycles = battery_data.get("cycle_data") or []
        if num_cycles is not None and num_cycles > 0:
            cycles = cycles[:num_cycles]
        return cycles

    def build_feature_frame(self, all_cycle_features: List[Dict[str, Any]]) -> pd.DataFrame:
        return pd.DataFrame(all_cycle_features)

    def order_columns(self, features_df: pd.DataFrame) -> pd.DataFrame:
        return features_df

    def build_cycle_frame(self, cycle_data: Dict[str, Any]) -> pd.DataFrame:
        return pd.DataFrame({
            "Time(s)": cycle_data["time_in_s"],
            "Current(A)": cycle_data["current_in_A"],
            "Voltage(V)": cycle_data["voltage_in_V"],
            "Charge_Capacity(Ah)": cycle_data.get("charge_capacity_in_Ah", []),
            "Discharge_Capacity(Ah)": cycle_data.get("discharge_capacity_in_Ah", []),
        })

    def split_phase_frames(
        self,
        cycle_df: pd.DataFrame,
        charge_threshold: float = 0.0,
        discharge_threshold: float = 0.0,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        charge_df = cycle_df[cycle_df["Current(A)"] > charge_threshold].copy()
        discharge_df = cycle_df[cycle_df["Current(A)"] < discharge_threshold].copy()
        rest_df = pd.DataFrame(columns=cycle_df.columns)

        if not charge_df.empty and not discharge_df.empty:
            charge_end_time = charge_df["Time(s)"].iloc[-1]
            discharge_start_time = discharge_df["Time(s)"].iloc[0]
            if discharge_start_time > charge_end_time:
                rest_df = cycle_df[
                    (cycle_df["Time(s)"] > charge_end_time) &
                    (cycle_df["Time(s)"] < discharge_start_time)
                ].copy()

        return charge_df, discharge_df, rest_df

    def trim_discharge_to_voltage_limit(
        self,
        discharge_df: pd.DataFrame,
        min_voltage_limit: Optional[float],
    ) -> pd.DataFrame:
        if discharge_df.empty or min_voltage_limit is None:
            return discharge_df

        cutoff_idx = discharge_df.index[discharge_df["Voltage(V)"] <= min_voltage_limit]
        if cutoff_idx.empty:
            return discharge_df
        return discharge_df.loc[:cutoff_idx[0]].copy()

    def get_protocol_rate(self, protocol: Any, default: float = 0.0) -> float:
        if isinstance(protocol, list) and protocol:
            return protocol[0].get("rate_in_C", default)
        if isinstance(protocol, dict):
            return protocol.get("rate_in_C", default)
        return default

    def get_time_span(self, df: pd.DataFrame) -> float:
        if df.empty:
            return 0.0
        return float(df["Time(s)"].iloc[-1] - df["Time(s)"].iloc[0])

    def calculate_capacity_energy_features(
        self,
        cycle_num: int,
        charge_df: pd.DataFrame,
        discharge_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        features: Dict[str, Any] = {"Cycle_Number": cycle_num}

        has_valid_charge = not charge_df.empty and len(charge_df) >= 2
        has_valid_discharge = not discharge_df.empty and len(discharge_df) >= 2

        q_chg = (
            trapezoid(np.abs(charge_df["Current(A)"]), x=charge_df["Time(s)"]) / 3600.0
            if has_valid_charge
            else np.nan
        )
        q_dis = (
            trapezoid(np.abs(discharge_df["Current(A)"]), x=discharge_df["Time(s)"]) / 3600.0
            if has_valid_discharge
            else np.nan
        )
        features["Charge_Capacity"] = q_chg
        features["Discharge_Capacity"] = q_dis
        if np.isfinite(q_chg) and np.isfinite(q_dis) and q_chg > 1e-6:
            ce_raw = q_dis / q_chg
            if ce_raw > 1.0 + 1e-4:
                features["Coulombic_Efficiency"] = 1.0
            else:
                features["Coulombic_Efficiency"] = ce_raw
        else:
            features["Coulombic_Efficiency"] = np.nan

        charge_energy = 0.0
        if not charge_df.empty:
            charge_power = charge_df["Voltage(V)"].values * np.abs(charge_df["Current(A)"].values)
            charge_energy = trapezoid(charge_power, x=charge_df["Time(s)"].values) / 3600.0

        discharge_energy = 0.0
        if not discharge_df.empty:
            discharge_power = discharge_df["Voltage(V)"].values * np.abs(discharge_df["Current(A)"].values)
            discharge_energy = trapezoid(discharge_power, x=discharge_df["Time(s)"].values) / 3600.0

        features["Charge_Energy"] = charge_energy
        features["Discharge_Energy"] = discharge_energy
        features["Energy_Efficiency"] = (discharge_energy / charge_energy) if charge_energy > 1e-6 else 0.0
        return features

    def calculate_charge_phase_features(
        self,
        charge_df: pd.DataFrame,
        uvp: float,
        time_mode: str = "duration",
        force_no_cv: bool = False,
        cv_voltage_tolerance: float = 0.01,
        cv_current_drop_ratio: Optional[float] = None,
    ) -> Dict[str, Any]:
        if charge_df.empty:
            return {
                "ICHV": 0.0,
                "UVP": uvp,
                "UVP_time": 0.0,
                "TCCC": 0.0,
                "TCVC": 0.0,
            }

        charge_start = charge_df["Time(s)"].iloc[0]
        charge_end = charge_df["Time(s)"].iloc[-1]
        uvp_time = charge_end - charge_start if time_mode == "duration" else charge_end
        features: Dict[str, Any] = {
            "ICHV": float(charge_df["Voltage(V)"].iloc[0]),
            "UVP": uvp,
            "UVP_time": float(uvp_time),
        }

        if force_no_cv or uvp <= 0:
            features["TCCC"] = float(uvp_time)
            features["TCVC"] = 0.0
            return features

        cv_mask = charge_df["Voltage(V)"] >= (uvp - cv_voltage_tolerance)
        if cv_current_drop_ratio is not None and not charge_df.empty:
            current_mask = charge_df["Current(A)"] < (charge_df["Current(A)"].max() * cv_current_drop_ratio)
            cv_mask = cv_mask & current_mask

        if cv_mask.any():
            cv_start_time = float(charge_df.loc[cv_mask.idxmax(), "Time(s)"])
            features["TCCC"] = cv_start_time - charge_start
            features["TCVC"] = charge_end - cv_start_time
        else:
            features["TCCC"] = float(uvp_time)
            features["TCVC"] = 0.0

        return features

    def calculate_discharge_phase_features(
        self,
        discharge_df: pd.DataFrame,
        lvp: float,
        time_mode: str = "duration",
    ) -> Dict[str, Any]:
        if discharge_df.empty:
            return {
                "IDV": 0.0,
                "LVP": lvp,
                "LVP_time": 0.0,
                "var_I_discharge": 0.0,
                "var_V_discharge": 0.0,
                "median_V_discharge": 0.0,
                "total_discharge_time": 0.0,
            }

        discharge_start = discharge_df["Time(s)"].iloc[0]
        discharge_end = discharge_df["Time(s)"].iloc[-1]
        lvp_time = discharge_end - discharge_start if time_mode == "duration" else discharge_end
        return {
            "IDV": float(discharge_df["Voltage(V)"].iloc[0]),
            "LVP": lvp,
            "LVP_time": float(lvp_time),
            "var_I_discharge": float(discharge_df["Current(A)"].var()),
            "var_V_discharge": float(discharge_df["Voltage(V)"].var()),
            "median_V_discharge": float(discharge_df["Voltage(V)"].median()),
            "total_discharge_time": float(discharge_end - discharge_start),
        }

    def calculate_advanced_features_common(
        self,
        cycle_df: pd.DataFrame,
        charge_df: pd.DataFrame,
        discharge_df: pd.DataFrame,
        direct_features: Dict[str, Any],
        rest_df: Optional[pd.DataFrame] = None,
        compute_cv_tau: bool = False,
        uvp_key: str = "UVP",
        tccc_key: str = "TCCC",
        tcvc_key: str = "TCVC",
        cv_voltage_tolerance: float = 0.01,
        cv_current_threshold: float = 0.001,
    ) -> Dict[str, Any]:
        adv_features: Dict[str, Any] = {}
        rest_df = rest_df if rest_df is not None else pd.DataFrame(columns=cycle_df.columns)

        if not charge_df.empty and not discharge_df.empty:
            v_dis_start = discharge_df["Voltage(V)"].iloc[0]
            i_dis_start = abs(discharge_df["Current(A)"].iloc[0])
            if rest_df.empty:
                charge_end_time = charge_df["Time(s)"].iloc[-1]
                discharge_start_time = discharge_df["Time(s)"].iloc[0]
                if discharge_start_time > charge_end_time:
                    rest_df = cycle_df[
                        (cycle_df["Time(s)"] > charge_end_time) &
                        (cycle_df["Time(s)"] < discharge_start_time)
                    ].copy()
            v_pre_dis = rest_df["Voltage(V)"].iloc[-1] if not rest_df.empty else charge_df["Voltage(V)"].iloc[-1]
            adv_features["Internal_Resistance"] = ((v_pre_dis - v_dis_start) / i_dis_start) if i_dis_start > 1e-6 else 0.0
        else:
            adv_features["Internal_Resistance"] = 0.0

        tcvc = direct_features.get(tcvc_key, 0.0)
        tccc = direct_features.get(tccc_key, 0.0)
        adv_features["RCV"] = (tccc / tcvc) if tcvc > 1e-6 else 0.0
        adv_features["skew_V_discharge"] = (
            float(skew(discharge_df["Voltage(V)"]))
            if not discharge_df.empty and discharge_df["Voltage(V)"].std() > 1e-6
            else 0.0
        )

        if compute_cv_tau and not charge_df.empty and tcvc > 10.0:
            uvp = direct_features.get(uvp_key, 0.0)
            cv_df = charge_df[charge_df["Voltage(V)"] >= (uvp - cv_voltage_tolerance)]
            if len(cv_df) > 10 and cv_df["Current(A)"].max() > cv_current_threshold:
                adv_features["CV_Current_Tau"] = fit_cv_decay(cv_df["Time(s)"].values, cv_df["Current(A)"].values)
            else:
                adv_features["CV_Current_Tau"] = 0.0
        elif compute_cv_tau:
            adv_features["CV_Current_Tau"] = 0.0

        return adv_features

    def calculate_anchor_features_common(
        self,
        charge_df: pd.DataFrame,
        discharge_df: pd.DataFrame,
        charge_intervals: List[tuple[float, float]],
        discharge_intervals: List[tuple[float, float]],
        tevi_intervals: List[tuple[float, float]],
        tevd_intervals: List[tuple[float, float]],
    ) -> Dict[str, Any]:
        anchor_features: Dict[str, Any] = {}

        def get_voltage_at_relative_time(df: pd.DataFrame, relative_time: float) -> Optional[float]:
            if df.empty:
                return None
            absolute_time = df["Time(s)"].iloc[0] + relative_time
            index = np.searchsorted(df["Time(s)"].values, absolute_time)
            index = min(index, len(df) - 1)
            return float(df["Voltage(V)"].iloc[index])

        def get_time_at_voltage(df: pd.DataFrame, voltage: float, direction: str) -> Optional[float]:
            if df.empty:
                return None
            mask = df["Voltage(V)"] >= voltage if direction == "charge" else df["Voltage(V)"] <= voltage
            if not mask.any():
                return None
            return float(df.loc[mask.idxmax(), "Time(s)"])

        phase_configs = [
            ("charge", charge_df, charge_intervals),
            ("discharge", discharge_df, discharge_intervals),
        ]
        for phase_name, df, intervals in phase_configs:
            duration = self.get_time_span(df)
            for idx, (start_ratio, end_ratio) in enumerate(intervals, start=1):
                feature_key = f"{phase_name}_slope_{idx}"
                dt = duration * (end_ratio - start_ratio)
                if dt <= 1e-6:
                    anchor_features[feature_key] = 0.0
                    continue
                v_start = get_voltage_at_relative_time(df, duration * start_ratio)
                v_end = get_voltage_at_relative_time(df, duration * end_ratio)
                anchor_features[feature_key] = ((v_end - v_start) / dt) if v_start is not None and v_end is not None else 0.0

        for idx, (v_start, v_end) in enumerate(tevi_intervals, start=1):
            t_start = get_time_at_voltage(charge_df, v_start, "charge")
            t_end = get_time_at_voltage(charge_df, v_end, "charge")
            anchor_features[f"TEVI_{idx}"] = (t_end - t_start) if t_start is not None and t_end is not None and t_end > t_start else 0.0

        for idx, (v_start, v_end) in enumerate(tevd_intervals, start=1):
            t_start = get_time_at_voltage(discharge_df, v_start, "discharge")
            t_end = get_time_at_voltage(discharge_df, v_end, "discharge")
            anchor_features[f"TEVD_{idx}"] = (t_end - t_start) if t_start is not None and t_end is not None and t_end > t_start else 0.0

        return anchor_features

    def build_plot_params(
        self,
        battery_data: Dict[str, Any],
        cycle_num: int,
        output_dir: Optional[Path],
    ) -> Optional[Dict[str, Any]]:
        if output_dir is None:
            return None
        return {
            "cell_id": battery_data.get("cell_id", "unknown"),
            "cycle_num": cycle_num,
            "output_dir": output_dir,
        }

    def handle_cycle_error(self, cell_id: str, exc: Exception) -> None:
        print(f"Warning: failed to extract features for {cell_id}: {exc}")

    def extract_cycle_features(
        self,
        cycle_data: Dict[str, Any],
        battery_data: Dict[str, Any],
        output_dir: Path,
    ) -> Dict[str, Any]:
        raise NotImplementedError

    def process_battery(
        self,
        file_path: Path,
        output_dir: Path,
        num_cycles: Optional[int] = None,
    ) -> None:
        try:
            battery_data = self.load_battery_data(file_path)
        except Exception as exc:
            print(f"Error loading {file_path}: {exc}")
            return

        cell_id = self.get_cell_id(battery_data, file_path)
        cycles_to_process = self.get_cycles_to_process(battery_data, num_cycles)

        all_cycle_features: List[Dict[str, Any]] = []
        for cycle_data in tqdm(cycles_to_process, desc=f"Processing {cell_id}"):
            if not cycle_data.get("time_in_s"):
                continue
            try:
                features = self.extract_cycle_features(cycle_data, battery_data, output_dir)
            except Exception as exc:
                self.handle_cycle_error(cell_id, exc)
                continue
            if not features:
                continue
            all_cycle_features.append(features)

        if not all_cycle_features:
            print(f"Warning: No features extracted for {cell_id}")
            return

        features_df = self.order_columns(self.build_feature_frame(all_cycle_features))
        output_file = output_dir / f"{cell_id}.csv"
        features_df.to_csv(output_file, index=False)
        print(f"Features for {cell_id} saved to {output_file}")

    def run(
        self,
        input_dir: Path,
        output_dir: Path,
        num_cycles: Optional[int] = None,
    ) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)

        if not input_dir.exists():
            print(f"Error: Directory not found at '{input_dir}'.")
            return

        files = list(input_dir.glob(self.get_file_pattern()))
        if not files:
            print(f"Error: No {self.get_file_pattern()} files found in '{input_dir}'.")
            return

        for file_path in files:
            self.process_battery(file_path, output_dir, num_cycles=num_cycles)


def resolve_dataset_input_dir(project_root: Path, dataset_name: str, relative_default: str) -> Path:
    dataset_env = f"{dataset_name.upper().replace('-', '_')}_DATA_DIR"
    explicit = os.environ.get(dataset_env)
    if explicit:
        return Path(explicit)

    data_root = os.environ.get("BATTERY_DATA_ROOT")
    if data_root:
        root = Path(data_root)
        canonical = root / dataset_name
        if canonical.exists():
            return canonical

        # Compatibility fallback for datasets where folder names use underscores.
        underscore_alias = root / dataset_name.replace("-", "_")
        if underscore_alias.exists():
            return underscore_alias

        return canonical

    return project_root / relative_default

"""
Feature Extraction Script for Stanford Battery Dataset
======================================================

Refactored to use shared utilities (src.utils) for consistent algorithms.
The local `_calculate_derivative_features` using Savitzky-Golay is replaced
by the more robust, standardized `extract_ic_dv_features` from the shared library.
"""
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import skew
from scipy.integrate import trapezoid  # Standardized
from tqdm import tqdm
import warnings
import sys
from typing import List, Dict, Any, Tuple, Optional

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.utils.math_tools import fit_cv_decay
from src.utils.feature_tools import extract_ic_dv_features

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


def _calculate_direct_features(
    cycle_df: pd.DataFrame,
    charge_df: pd.DataFrame,
    discharge_df: pd.DataFrame,
    cycle_num: int,
    battery_data: Dict[str, Any]
) -> Dict[str, Any]:
    """Calculates direct features from raw data."""
    features = {}

    # --- 1. Capacity & Energy ---
    features['Cycle_Number'] = cycle_num

    if not discharge_df.empty:
        dis_cap = trapezoid(discharge_df['Current(A)'].abs(), x=discharge_df['Time(s)']) / 3600.0
    else:
        dis_cap = 0.0

    if not charge_df.empty:
        chg_cap = trapezoid(charge_df['Current(A)'].abs(), x=charge_df['Time(s)']) / 3600.0
    else:
        chg_cap = 0.0

    features['Discharge_Capacity(Ah)'] = dis_cap
    features['Charge_Capacity(Ah)'] = chg_cap

    if not charge_df.empty:
        p_charge = charge_df['Voltage(V)'] * charge_df['Current(A)'].abs()
        e_charge_ws = trapezoid(p_charge, x=charge_df['Time(s)'])
        features['Charge_Energy(Wh)'] = e_charge_ws / 3600.0
    else:
        features['Charge_Energy(Wh)'] = 0

    if not discharge_df.empty:
        p_discharge = discharge_df['Voltage(V)'] * discharge_df['Current(A)'].abs()
        e_discharge_ws = trapezoid(p_discharge, x=discharge_df['Time(s)'])
        features['Discharge_Energy(Wh)'] = e_discharge_ws / 3600.0
    else:
        features['Discharge_Energy(Wh)'] = 0

    features['Coulombic_Efficiency'] = (dis_cap / chg_cap) if chg_cap > 1e-6 else 0
    features['Energy_Efficiency'] = (
        features['Discharge_Energy(Wh)'] / features['Charge_Energy(Wh)']
    ) if features['Charge_Energy(Wh)'] > 1e-6 else 0

    # --- 2. Rest Time & C-rates ---
    features['Rest_Time(s)'] = 0

    charge_proto = battery_data.get('charge_protocol', [{}])
    features['charge_c_rate'] = charge_proto[0].get('rate_in_C', 0) if charge_proto else 0

    discharge_proto = battery_data.get('discharge_protocol', [{}])
    features['discharge_c_rate'] = discharge_proto[0].get('rate_in_C', 0) if discharge_proto else 0

    # --- 3. Charge Phase & CV Dynamics ---
    features['CV_Current_Tau'] = 0.0
    if not charge_df.empty:
        features['ICHV(V)'] = charge_df['Voltage(V)'].iloc[0]
        features['UVP_time(s)'] = charge_df['Time(s)'].iloc[-1] - charge_df['Time(s)'].iloc[0]
        v_upper = battery_data.get('max_voltage_limit_in_V', 0)
        features['UVP(V)'] = v_upper

        if v_upper > 0 and charge_df['Voltage(V)'].max() >= (v_upper - 0.01):
            cv_mask = charge_df['Voltage(V)'] >= (v_upper - 0.01)
            cv_df = charge_df[cv_mask]
            if not cv_df.empty:
                t_cv_start = cv_df['Time(s)'].iloc[0]
                features['TCCC(s)'] = t_cv_start - charge_df['Time(s)'].iloc[0]
                features['TCVC(s)'] = charge_df['Time(s)'].iloc[-1] - t_cv_start

                if np.sum(cv_df['Current(A)'] > 0.001) > 10:
                    features['CV_Current_Tau'] = fit_cv_decay(
                        cv_df['Time(s)'].values, cv_df['Current(A)'].values
                    )
            else: # Should not happen if max >= thresh, but for safety
                features['TCCC(s)'] = charge_df['Time(s)'].iloc[-1] - charge_df['Time(s)'].iloc[0]
                features['TCVC(s)'] = 0
        else: # Pure CC
            features['TCCC(s)'] = charge_df['Time(s)'].iloc[-1] - charge_df['Time(s)'].iloc[0]
            features['TCVC(s)'] = 0
    else:
        features.update({'ICHV(V)': 0, 'UVP_time(s)': 0, 'UVP(V)': 0, 'TCCC(s)': 0, 'TCVC(s)': 0})

    # --- 4. Discharge Phase ---
    if not discharge_df.empty:
        features['IDV(V)'] = discharge_df['Voltage(V)'].iloc[0]
        features['LVP_time(s)'] = discharge_df['Time(s)'].iloc[-1] - discharge_df['Time(s)'].iloc[0]
        features['LVP(V)'] = battery_data.get('min_voltage_limit_in_V', 0)
        features['var_I_discharge'] = discharge_df['Current(A)'].var()
        features['var_V_discharge'] = discharge_df['Voltage(V)'].var()
        features['median_V_discharge(V)'] = discharge_df['Voltage(V)'].median()
        features['total_discharge_time(s)'] = (
            discharge_df['Time(s)'].iloc[-1] - discharge_df['Time(s)'].iloc[0]
        )
    else:
        features.update({
            'IDV(V)': 0, 'LVP_time(s)': 0, 'LVP(V)': 0, 'var_I_discharge': 0,
            'var_V_discharge': 0, 'median_V_discharge(V)': 0, 'total_discharge_time(s)': 0
        })

    return features


def _calculate_advanced_features(
    cycle_df: pd.DataFrame,
    charge_df: pd.DataFrame,
    discharge_df: pd.DataFrame,
    features: Dict[str, Any]
) -> Dict[str, Any]:
    """Calculates IR, RCV, and statistical features."""
    adv_features = {}

    # Internal Resistance
    if not charge_df.empty and not discharge_df.empty:
        v_dis_start = discharge_df['Voltage(V)'].iloc[0]
        i_dis_start = abs(discharge_df['Current(A)'].iloc[0])
        rest_df = cycle_df[
            (cycle_df['Time(s)'] > charge_df['Time(s)'].iloc[-1]) &
            (cycle_df['Time(s)'] < discharge_df['Time(s)'].iloc[0])
        ]
        v_ocv = rest_df['Voltage(V)'].iloc[-1] if not rest_df.empty else charge_df['Voltage(V)'].iloc[-1]
        adv_features['Internal_Resistance(Ohm)'] = (v_ocv - v_dis_start) / i_dis_start if i_dis_start > 1e-3 else 0
    else:
        adv_features['Internal_Resistance(Ohm)'] = 0

    # RCV Ratio
    tcvc = features.get('TCVC', 0)
    adv_features['RCV(V)'] = features.get('TCCC', 0) / tcvc if tcvc > 1e-3 else 0

    # Skewness
    if not discharge_df.empty:
        adv_features['skew_V_discharge'] = skew(discharge_df['Voltage(V)'])
    else:
        adv_features['skew_V_discharge'] = 0

    return adv_features


def _calculate_anchor_features(
    charge_df: pd.DataFrame,
    discharge_df: pd.DataFrame,
    charge_slope_intervals: List[Tuple],
    discharge_slope_intervals: List[Tuple],
    tevi_intervals: List[Tuple],
    tevd_intervals: List[Tuple]
) -> Dict[str, Any]:
    """Calculates anchor features (slopes, time-at-voltage)."""
    anchor_features = {}

    # --- Helper: Get Voltage at Relative Time ---
    def get_voltage_at_relative_time(df: pd.DataFrame, rel_time: float) -> Optional[float]:
        if df.empty: return None
        abs_time = df['Time(s)'].iloc[0] + rel_time
        idx = np.searchsorted(df['Time(s)'].values, abs_time)
        idx = min(idx, len(df) - 1)
        return df['Voltage(V)'].iloc[idx]

    # --- Charge/Discharge Slopes ---
    for phase, intervals in [('charge', charge_slope_intervals), ('discharge', discharge_slope_intervals)]:
        df = charge_df if phase == 'charge' else discharge_df
        duration = df['Time(s)'].iloc[-1] - df['Time(s)'].iloc[0] if not df.empty else 0

        for i, (p_start, p_end) in enumerate(intervals):
            key = f'{phase}_slope_{i+1}'
            if duration > 1.0:
                t1, t2 = duration * p_start, duration * p_end
                v1, v2 = get_voltage_at_relative_time(df, t1), get_voltage_at_relative_time(df, t2)
                dt = t2 - t1
                if v1 is not None and v2 is not None and dt > 1e-3:
                    anchor_features[key] = (v2 - v1) / dt
                else:
                    anchor_features[key] = 0
            else:
                anchor_features[key] = 0

    # --- Helper: Get Time at Voltage ---
    def get_time_at_voltage(df: pd.DataFrame, voltage: float, direction: str) -> Optional[float]:
        if df.empty: return None
        mask = df['Voltage(V)'] >= voltage if direction == 'charge' else df['Voltage(V)'] <= voltage
        if mask.any():
            return df.loc[mask.idxmax(), 'Time(s)']
        return None

    # --- TEVI/TEVD ---
    for i, (v_start, v_end) in enumerate(tevi_intervals):
        t1, t2 = get_time_at_voltage(charge_df, v_start, 'charge'), get_time_at_voltage(charge_df, v_end, 'charge')
        anchor_features[f'TEVI_{i+1}'] = (t2 - t1) if t1 and t2 and t2 > t1 else 0

    for i, (v_start, v_end) in enumerate(tevd_intervals):
        t1, t2 = get_time_at_voltage(discharge_df, v_start, 'discharge'), get_time_at_voltage(discharge_df, v_end, 'discharge')
        anchor_features[f'TEVD_{i+1}'] = (t2 - t1) if t1 and t2 and t2 > t1 else 0

    return anchor_features


def extract_features_for_cycle(
    cycle_data: Dict[str, Any],
    battery_data: Dict[str, Any],
    charge_slopes: List[Tuple],
    discharge_slopes: List[Tuple],
    tevi_ints: List[Tuple],
    tevd_ints: List[Tuple],
    output_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """Main orchestrator for single-cycle feature extraction."""
    # 1. Prepare Data
    cycle_df = pd.DataFrame({
        'Time(s)': cycle_data['time_in_s'],
        'Current(A)': cycle_data['current_in_A'],
        'Voltage(V)': cycle_data['voltage_in_V'],
        'Charge_Capacity(Ah)': cycle_data['charge_capacity_in_Ah'],
        'Discharge_Capacity(Ah)': cycle_data['discharge_capacity_in_Ah'],
    })
    cycle_num = cycle_data.get('cycle_number', 0)

    # 2. Split Phases
    charge_df = cycle_df[cycle_df['Current(A)'] > 0].copy()
    discharge_df = cycle_df[cycle_df['Current(A)'] < 0].copy()

    # 3. Apply Cutoff
    v_lower = battery_data.get('min_voltage_limit_in_V', 0)
    if not discharge_df.empty and v_lower > 0:
        cutoff_idx = discharge_df.index[discharge_df['Voltage(V)'] <= v_lower]
        if not cutoff_idx.empty:
            discharge_df = discharge_df.loc[:cutoff_idx[0]]

    # 4. Extract Features
    direct_feats = _calculate_direct_features(
        cycle_df, charge_df, discharge_df, cycle_num, battery_data
    )
    nominal_cap = battery_data.get('nominal_capacity_in_Ah', 2.0)
    ncm_config = {
        'peak_mode': 2,
        'nominal_capacity': nominal_cap,
        'window_length_ic': 21,
        'window_length_dv': 21,
        'peak_height_ic': 0.01,
        # 'peak_height_dv': 0.01,
        'voltage_range_ic': (3.4, 4.1),
        # 'voltage_range_dv': (3.4, 4.1),
        'prominence_ic': 0.01,
        # 'prominence_dv': 0.005,
        'ic_step_size': 0.002,
        'dv_step_size': nominal_cap *0.005,
        'search_window_dvv': 0.1,
        'search_window_dvp': 0.1,
        'initial_capacity_cut_fraction': 0.02,
        'icv_search_offset_lower': 0.05,
        'icv_search_offset_upper': 0.5
    }

    plot_params = None
    if output_dir:
        plot_params = {
            'cell_id': battery_data.get('cell_id', 'unknown'),
            'cycle_num': cycle_num,
            'output_dir': output_dir
        }

    derivative_feats = extract_ic_dv_features(
        discharge_df,
        config=ncm_config,
        plot_params=plot_params
    )

    advanced_feats = _calculate_advanced_features(
        cycle_df, charge_df, discharge_df, direct_feats
    )
    anchor_feats = _calculate_anchor_features(
        charge_df, discharge_df, charge_slopes, discharge_slopes, tevi_ints, tevd_ints
    )

    return {**direct_feats, **derivative_feats, **advanced_feats, **anchor_feats}


def process_battery(
    file_path: Path,
    output_dir: Path,
    charge_slopes: List[Tuple],
    discharge_slopes: List[Tuple],
    tevi_ints: List[Tuple],
    tevd_ints: List[Tuple],
    num_cycles: Optional[int] = None
):
    try:
        with open(file_path, 'rb') as f:
            battery_data = pickle.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return

    cycles_to_proc = battery_data.get('cycle_data', [])
    if num_cycles:
        cycles_to_proc = cycles_to_proc[:num_cycles]

    cell_id = battery_data.get('cell_id', file_path.stem)
    all_feats = []

    for c_data in tqdm(cycles_to_proc, desc=f"Processing {cell_id}", leave=False):
        if not c_data.get('time_in_s'): continue
        try:
            feats = extract_features_for_cycle(
                c_data, battery_data, charge_slopes, discharge_slopes, tevi_ints, tevd_ints,
                output_dir=output_dir
            )
            all_feats.append(feats)
        except Exception:
            continue

    if not all_feats:
        print(f"No features extracted for {cell_id}")
        return

    features_df = pd.DataFrame(all_feats)
    out_path = output_dir / f"{cell_id}.csv"
    features_df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


def main():
    datasets = [
        (Path('F:/datasets/battery/Stanford_2'), 'Stanford_2'),
        (Path('F:/datasets/battery/Stanford'), 'Stanford')
    ]

    # Common Intervals
    charge_slope_intervals = [(0.10, 0.30), (0.10, 0.50), (0.10, 0.80)]
    discharge_slope_intervals = [(0.10, 0.30), (0.10, 0.50), (0.10, 0.80)]
    tevi_intervals = [(3.5, 3.7), (3.7, 3.9), (3.9, 4.1)]
    tevd_intervals = [(4.1, 3.9), (3.9, 3.7), (3.7, 3.5)]

    num_cycles_to_extract = 100

    for input_path, output_name in datasets:
        print(f"\n{'='*40}")
        print(f"Processing Dataset: {output_name}")
        print(f"Input: {input_path}")
        print(f"{'='*40}")

        output_dir = project_root / 'results' / output_name
        output_dir.mkdir(parents=True, exist_ok=True)

        if not input_path.exists():
            print(f"Error: Directory not found at '{input_path}'. Skipping...")
            continue

        pkl_files = list(input_path.glob('*.pkl'))
        if not pkl_files:
            print(f"Error: No .pkl files found in '{input_path}'. Skipping...")
            continue

        for file_path in pkl_files:
            process_battery(
                file_path, output_dir,
                charge_slope_intervals, discharge_slope_intervals,
                tevi_intervals, tevd_intervals,
                num_cycles=num_cycles_to_extract
            )


if __name__ == '__main__':
    main()

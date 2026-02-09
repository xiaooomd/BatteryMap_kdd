"""
Feature Extraction Script for XJTU Battery Dataset
===================================================

Refactored to use shared utilities (src.utils) for consistent algorithms.
- Replaced local IC/DV feature calculation with `extract_ic_dv_features`.
- Replaced local CV phase detection (`_detect_cv_entry_time`) with the
  standardized logic embedded within the feature calculation functions.
"""
import pickle
import warnings
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd
import numpy as np
from scipy.integrate import trapezoid
from scipy.stats import skew
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.utils.math_tools import fit_cv_decay
from src.utils.feature_tools import extract_ic_dv_features

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# --- Constants for Robust Physics Logic ---
CV_VOLTAGE_TOLERANCE_V = 0.05
CC_CURRENT_DROP_RATIO = 0.95


def _calculate_direct_features(
    cycle_df: pd.DataFrame, charge_df: pd.DataFrame, discharge_df: pd.DataFrame,
    cycle_num: int, battery_data: Dict[str, Any]
) -> Dict[str, Any]:
    """Calculates direct features from raw cycle data."""
    features = {}

    features['Cycle_Number'] = cycle_num
    # [FIX] Recalculate Capacity via Integration for accuracy and consistency
    # XJTU dataset columns might have issues leading to CE > 1
    # We use trapezoid integration of Current over Time
    if not charge_df.empty:
        q_chg = trapezoid(y=charge_df['Current(A)'].values, x=charge_df['Time(s)'].values) / 3600.0
    else:
        q_chg = 0.0

    if not discharge_df.empty:
        q_dis = trapezoid(y=np.abs(discharge_df['Current(A)'].values), x=discharge_df['Time(s)'].values) / 3600.0
    else:
        q_dis = 0.0

    features['Discharge_Capacity(Ah)'] = q_dis
    features['Charge_Capacity(Ah)'] = q_chg
    features['Coulombic_Efficiency'] = (q_dis / q_chg) if q_chg > 1e-6 else 0.0

    if not charge_df.empty:
        p_chg = charge_df['Voltage(V)'].values * charge_df['Current(A)'].values
        features['Charge_Energy(Wh)'] = trapezoid(y=p_chg, x=charge_df['Time(s)'].values) / 3600.0
    else: features['Charge_Energy(Wh)'] = 0.0

    if not discharge_df.empty:
        p_dis = discharge_df['Voltage(V)'].values * np.abs(discharge_df['Current(A)'].values)
        features['Discharge_Energy(Wh)'] = trapezoid(y=p_dis, x=discharge_df['Time(s)'].values) / 3600.0
    else: features['Discharge_Energy(Wh)'] = 0.0

    features['Energy_Efficiency'] = (features['Discharge_Energy(Wh)'] / features['Charge_Energy(Wh)']) if features['Charge_Energy(Wh)'] > 1e-6 else 0.0

    # [FIX] Rest Time Calculation for XJTU (Time resets at each step)
    # We look for the gap between Charge end and Discharge start indices in the original cycle_df
    features['Rest_Time(s)'] = 0.0
    if not charge_df.empty and not discharge_df.empty:
        last_chg_idx = charge_df.index[-1]
        first_dis_idx = discharge_df.index[0]

        # Ensure discharge comes after charge
        if first_dis_idx > last_chg_idx + 1:
            # Extract the segment between charge and discharge
            rest_segment = cycle_df.loc[last_chg_idx+1 : first_dis_idx-1]
            if not rest_segment.empty:
                # Calculate duration. Since time might reset to 0 or be continuous,
                # (max - min) is a robust way to get duration for a single monotonic segment.
                t_rest = rest_segment['Time(s)'].max() - rest_segment['Time(s)'].min()
                features['Rest_Time(s)'] = max(0.0, t_rest)

    features['charge_c_rate'] = battery_data['charge_protocol'][0]['rate_in_C']
    features['discharge_c_rate'] = battery_data['discharge_protocol'][0]['rate_in_C']

    v_upper_limit = battery_data['max_voltage_limit_in_V']
    features['UVP(V)'] = v_upper_limit
    if not charge_df.empty:
        features['ICHV(V)'] = charge_df['Voltage(V)'].iloc[0]
        features['UVP_time(s)'] = charge_df['Time(s)'].iloc[-1]

        # Robust CV detection
        cv_mask = charge_df['Voltage(V)'] >= (v_upper_limit - CV_VOLTAGE_TOLERANCE_V)
        current_mask = charge_df['Current(A)'] < (charge_df['Current(A)'].max() * CC_CURRENT_DROP_RATIO)
        combined_mask = cv_mask & current_mask

        cv_start_time = None
        if combined_mask.any():
            cv_start_time = charge_df.loc[combined_mask.idxmax(), 'Time(s)']

        if cv_start_time is not None:
            features['TCCC(s)'] = cv_start_time - charge_df['Time(s)'].iloc[0]
            features['TCVC(s)'] = charge_df['Time(s)'].iloc[-1] - cv_start_time
        else:
            features['TCCC(s)'] = charge_df['Time(s)'].iloc[-1] - charge_df['Time(s)'].iloc[0]
            features['TCVC(s)'] = 0.0
    else:
        features.update({'ICHV(V)': 0, 'UVP_time(s)': 0, 'TCCC(s)': 0, 'TCVC(s)': 0})

    features['LVP(V)'] = battery_data['min_voltage_limit_in_V']
    if not discharge_df.empty:
        features['IDV(V)'] = discharge_df['Voltage(V)'].iloc[0]
        features['LVP_time(s)'] = discharge_df['Time(s)'].iloc[-1]
        features['var_I_discharge'] = discharge_df['Current(A)'].var()
        features['var_V_discharge'] = discharge_df['Voltage(V)'].var()
        features['median_V_discharge(V)'] = discharge_df['Voltage(V)'].median()
        features['total_discharge_time(s)'] = discharge_df['Time(s)'].iloc[-1] - discharge_df['Time(s)'].iloc[0]
    else:
        features.update({'IDV(V)': 0, 'LVP_time(s)': 0, 'var_I_discharge': 0, 'var_V_discharge': 0, 'median_V_discharge(V)': 0, 'total_discharge_time(s)': 0})
    return features


def _calculate_advanced_features(
    charge_df: pd.DataFrame, discharge_df: pd.DataFrame, rest_df: pd.DataFrame,
    features: Dict[str, Any]
) -> Dict[str, Any]:
    """Calculates advanced features (IR, RCV, Skewness, CV Tau)."""
    adv_features = {}

    if not charge_df.empty and not discharge_df.empty:
        v_dis_start = discharge_df['Voltage(V)'].iloc[0]
        i_dis_start = abs(discharge_df['Current(A)'].iloc[0])
        v_pre_dis = rest_df['Voltage(V)'].iloc[-1] if not rest_df.empty else charge_df['Voltage(V)'].iloc[-1]
        adv_features['Internal_Resistance(Ohm)'] = ((v_pre_dis - v_dis_start) / i_dis_start) if i_dis_start > 1e-3 else 0.0
    else: adv_features['Internal_Resistance(Ohm)'] = 0.0

    tcvc = features.get('TCVC', 0)
    adv_features['RCV(V)'] = (features.get('TCCC', 0) / tcvc) if tcvc > 1.0 else 0.0
    adv_features['skew_V_discharge'] = skew(discharge_df['Voltage(V)']) if not discharge_df.empty else 0.0

    if not charge_df.empty and tcvc > 10.0:
        cv_mask = charge_df['Voltage(V)'] >= (features.get('UVP', 4.2) - 0.05)
        cv_df = charge_df[cv_mask]
        if len(cv_df) > 10 and cv_df['Current(A)'].max() > 0.01:
            adv_features['CV_Current_Tau'] = fit_cv_decay(cv_df['Time(s)'].values, cv_df['Current(A)'].values)
        else: adv_features['CV_Current_Tau'] = 0.0
    else: adv_features['CV_Current_Tau'] = 0.0

    return adv_features


def _calculate_anchor_features(
    charge_df: pd.DataFrame, discharge_df: pd.DataFrame,
    charge_intervals: List[Tuple], discharge_intervals: List[Tuple],
    tevi_intervals: List[Tuple], tevd_intervals: List[Tuple]
) -> Dict[str, Any]:
    """Calculates anchor point features."""
    anchor_features = {}

    def get_v_at_rel_time(df: pd.DataFrame, rel_time: float) -> Optional[float]:
        if df.empty: return None
        abs_time = df['Time(s)'].iloc[0] + rel_time
        idx = np.searchsorted(df['Time(s)'].values, abs_time)
        idx = min(idx, len(df) - 1)
        return df['Voltage(V)'].iloc[idx]

    c_dur = (charge_df['Time(s)'].iloc[-1] - charge_df['Time(s)'].iloc[0]) if not charge_df.empty else 0
    for i, (ps, pe) in enumerate(charge_intervals):
        dt = c_dur * (pe - ps)
        if dt > 1e-6:
            vs, ve = get_v_at_rel_time(charge_df, c_dur*ps), get_v_at_rel_time(charge_df, c_dur*pe)
            anchor_features[f'charge_slope_{i+1}'] = (ve - vs) / dt if vs and ve else 0.0
        else: anchor_features[f'charge_slope_{i+1}'] = 0.0

    d_dur = (discharge_df['Time(s)'].iloc[-1] - discharge_df['Time(s)'].iloc[0]) if not discharge_df.empty else 0
    for i, (ps, pe) in enumerate(discharge_intervals):
        dt = d_dur * (pe - ps)
        if dt > 1e-6:
            vs, ve = get_v_at_rel_time(discharge_df, d_dur*ps), get_v_at_rel_time(discharge_df, d_dur*pe)
            anchor_features[f'discharge_slope_{i+1}'] = (ve - vs) / dt if vs and ve else 0.0
        else: anchor_features[f'discharge_slope_{i+1}'] = 0.0

    def get_t_for_v(df: pd.DataFrame, v: float, direction: str) -> Optional[float]:
        if df.empty: return None
        mask = df['Voltage(V)'] >= v if direction == 'charge' else df['Voltage(V)'] <= v
        return df.loc[mask.idxmax(), 'Time(s)'] if mask.any() else None

    for i, (vs, ve) in enumerate(tevi_intervals):
        ts, te = get_t_for_v(charge_df, vs, 'charge'), get_t_for_v(charge_df, ve, 'charge')
        anchor_features[f'TEVI_{i+1}'] = (te - ts) if ts and te and te > ts else 0.0

    for i, (vs, ve) in enumerate(tevd_intervals):
        ts, te = get_t_for_v(discharge_df, vs, 'discharge'), get_t_for_v(discharge_df, ve, 'discharge')
        anchor_features[f'TEVD_{i+1}'] = (te - ts) if ts and te and te > ts else 0.0

    return anchor_features


def extract_features_for_cycle(
    cycle_data: Dict[str, Any], battery_data: Dict[str, Any],
    charge_slopes: List[Tuple], discharge_slopes: List[Tuple],
    tevi_ints: List[Tuple], tevd_ints: List[Tuple],
    output_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """Orchestrates feature extraction for a single cycle."""
    cycle_df = pd.DataFrame({
        'Time(s)': cycle_data['time_in_s'], 'Current(A)': cycle_data['current_in_A'],
        'Voltage(V)': cycle_data['voltage_in_V'],
        'Charge_Capacity(Ah)': cycle_data['charge_capacity_in_Ah'],
        'Discharge_Capacity(Ah)': cycle_data['discharge_capacity_in_Ah']
    })
    cycle_num = cycle_data['cycle_number']

    charge_df = cycle_df[cycle_df['Current(A)'] > 0].copy()
    discharge_df = cycle_df[cycle_df['Current(A)'] < 0].copy()

    if not discharge_df.empty:
        v_lower = battery_data['min_voltage_limit_in_V']
        cutoff_idx = discharge_df.index[discharge_df['Voltage(V)'] <= v_lower]
        if not cutoff_idx.empty:
            discharge_df = discharge_df.loc[:cutoff_idx[0]]

    rest_df = pd.DataFrame()
    if not charge_df.empty and not discharge_df.empty:
        if discharge_df['Time(s)'].iloc[0] > charge_df['Time(s)'].iloc[-1]:
            rest_df = cycle_df[(cycle_df['Time(s)'] > charge_df['Time(s)'].iloc[-1]) & (cycle_df['Time(s)'] < discharge_df['Time(s)'].iloc[0])].copy()

    direct_feats = _calculate_direct_features(cycle_df, charge_df, discharge_df, cycle_num, battery_data)

    # [MODIFIED] Use shared tool for IC/DV with config
    nominal_cap = battery_data.get('nominal_capacity_in_Ah', 2.0)
    ncm_config = {
        'peak_mode': 1,
        'nominal_capacity': nominal_cap,
        'window_length_ic': 21,
        'window_length_dv': 21,
        'peak_height_ic': 0.01,
        # 'peak_height_dv': 0.01,
        'voltage_range_ic': (3.3, 4.2),
        # 'voltage_range_dv': (3.3, 4.2),
        'prominence_ic': 0.02,
        # 'prominence_dv': 0.02,
        'ic_step_size': 0.01,
        'dv_step_size': nominal_cap * 0.005,
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

    advanced_feats = _calculate_advanced_features(charge_df, discharge_df, rest_df, direct_feats)
    anchor_feats = _calculate_anchor_features(charge_df, discharge_df, charge_slopes, discharge_slopes, tevi_ints, tevd_ints)

    return {**direct_feats, **derivative_feats, **advanced_feats, **anchor_feats}


def process_battery(
    file_path: Path, output_dir: Path, charge_slopes: List[Tuple], discharge_slopes: List[Tuple],
    tevi_ints: List[Tuple], tevd_ints: List[Tuple], num_cycles: Optional[int] = None
):
    try:
        with open(file_path, 'rb') as f:
            data_dict = pickle.load(f)
    except Exception as e:
        print(f"Error loading {file_path.name}: {e}")
        return

    cell_id = data_dict.get('cell_id', file_path.stem)
    cycles = list(data_dict.get('cycle_data', []))[:num_cycles] if num_cycles else list(data_dict.get('cycle_data', []))
    all_feats = []

    for cycle_data in tqdm(cycles, desc=f"Processing {cell_id}", unit="cycle", leave=False):
        if not cycle_data.get('time_in_s'): continue
        try:
            feats = extract_features_for_cycle(cycle_data, data_dict, charge_slopes, discharge_slopes, tevi_ints, tevd_ints, output_dir=output_dir)
            all_feats.append(feats)
        except Exception:
            continue

    if not all_feats:
        print(f"Warning: No valid features extracted for {cell_id}")
        return

    df_out = pd.DataFrame(all_feats)
    out_path = output_dir / f"{cell_id}.csv"
    df_out.to_csv(out_path, index=False)
    print(f"Features saved: {out_path}")


def main():
    PROCESSED_DATA_DIR = project_root / 'data' / 'XJTU'
    OUTPUT_DIR = project_root / 'results' / 'features' / 'XJTU'
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    CHARGE_SLOPES = [(0.1, 0.3), (0.1, 0.5), (0.1, 0.8)]
    DISCHARGE_SLOPES = [(0.1, 0.3), (0.1, 0.5), (0.1, 0.8)]
    TEVI_INTERVALS = [(3.6, 3.7), (3.7, 3.9), (3.9, 4.1)]
    TEVD_INTERVALS = [(4.1, 3.9), (3.9, 3.7), (3.7, 3.5)]
    NUM_CYCLES = 100

    if not PROCESSED_DATA_DIR.exists():
        print(f"Error: Input directory '{PROCESSED_DATA_DIR}' does not exist.")
        return

    pkl_files = list(PROCESSED_DATA_DIR.glob('*.pkl'))
    if not pkl_files:
        print(f"Error: No .pkl files found in '{PROCESSED_DATA_DIR}'.")
        return

    print(f"Found {len(pkl_files)} battery files. Starting extraction...")
    for pkl_file in pkl_files:
        process_battery(pkl_file, OUTPUT_DIR, CHARGE_SLOPES, DISCHARGE_SLOPES, TEVI_INTERVALS, TEVD_INTERVALS, num_cycles=NUM_CYCLES)

if __name__ == '__main__':
    main()

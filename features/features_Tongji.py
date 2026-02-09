"""
Feature Extraction Script for Tongji University Battery Dataset
===============================================================

Refactored to use shared utilities (src.utils) for consistent algorithms.
The local `_calculate_derivative_features` using Savitzky-Golay is replaced
by the more robust, standardized `extract_ic_dv_features` from the shared library.
Specific logic for C-rate inference is preserved.
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
from typing import List, Dict, Any, Optional, Tuple

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.utils.math_tools import fit_cv_decay
from src.utils.feature_tools import extract_ic_dv_features

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

class AttrDict(dict):
    """A dictionary that allows attribute-style access."""
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def _calculate_direct_features(
    cycle_df: pd.DataFrame,
    charge_df: pd.DataFrame,
    discharge_df: pd.DataFrame,
    rest_df: pd.DataFrame,
    cycle_num: int,
    battery_data: Any
) -> Dict[str, Any]:
    """Calculates direct features from raw cycle data."""
    features = {}

    # --- A. Overall Cycle Features ---
    features['Cycle_Number'] = cycle_num

    if not discharge_df.empty:
        q_dis = trapezoid(discharge_df['Current(A)'].abs(), x=discharge_df['Time(s)']) / 3600.0
    else:
        q_dis = 0.0

    if not charge_df.empty:
        q_chg = trapezoid(charge_df['Current(A)'].abs(), x=charge_df['Time(s)']) / 3600.0
    else:
        q_chg = 0.0

    features['Discharge_Capacity(Ah)'] = q_dis
    features['Charge_Capacity(Ah)'] = q_chg

    features['Coulombic_Efficiency'] = (q_dis / q_chg) if q_chg > 1e-6 else 0

    if not charge_df.empty:
        p_chg = charge_df['Voltage(V)'].values * charge_df['Current(A)'].values
        e_chg_j = trapezoid(y=p_chg, x=charge_df['Time(s)'].values)
        features['Charge_Energy(Wh)'] = e_chg_j / 3600.0
    else:
        features['Charge_Energy(Wh)'] = 0

    if not discharge_df.empty:
        p_dis = discharge_df['Voltage(V)'].values * np.abs(discharge_df['Current(A)'].values)
        e_dis_j = trapezoid(y=p_dis, x=discharge_df['Time(s)'].values)
        features['Discharge_Energy(Wh)'] = e_dis_j / 3600.0
    else:
        features['Discharge_Energy(Wh)'] = 0

    features['Energy_Efficiency'] = (
        features['Discharge_Energy(Wh)'] / features['Charge_Energy(Wh)']
    ) if features['Charge_Energy(Wh)'] > 1e-6 else 0

    # --- Rest Phase ---
    if not rest_df.empty:
        features['Rest_Time(s)'] = rest_df['Time(s)'].iloc[-1] - rest_df['Time(s)'].iloc[0]
        features['V_rest_end(V)'] = rest_df['Voltage(V)'].iloc[-1]
    else:
        features['Rest_Time(s)'] = 0
        features['V_rest_end(V)'] = 0

    # --- C-rate Inference (Preserved Tongji Specific Logic) ---
    nominal_cap = getattr(battery_data, 'nominal_capacity_in_Ah', 0)

    if not charge_df.empty and nominal_cap > 0:
        cc_charge = charge_df.iloc[:len(charge_df) // 2]
        if not cc_charge.empty:
            charge_c_rate = cc_charge['Current(A)'].median() / nominal_cap
            features['charge_c_rate'] = float(f"{charge_c_rate:.2f}")
        else: features['charge_c_rate'] = 0.0
    else: features['charge_c_rate'] = 0.0

    if not discharge_df.empty and nominal_cap > 0:
        discharge_c_rate = abs(discharge_df['Current(A)'].median()) / nominal_cap
        features['discharge_c_rate'] = float(f"{discharge_c_rate:.2f}")
    else: features['discharge_c_rate'] = 0.0

    # --- B. Charge/Discharge Phase Limits & Durations ---
    v_upper = getattr(battery_data, 'max_voltage_limit_in_V', 0)
    v_lower = getattr(battery_data, 'min_voltage_limit_in_V', 0)

    if not charge_df.empty:
        features['ICHV(V)'] = charge_df['Voltage(V)'].iloc[0]
        # [FIX] Use relative time (duration) instead of absolute timestamp
        features['UVP_time(s)'] = charge_df['Time(s)'].iloc[-1] - charge_df['Time(s)'].iloc[0]
        features['UVP(V)'] = v_upper

        t_at_v_limit = charge_df[charge_df['Voltage(V)'] >= v_upper - 0.01]['Time(s)']
        if not t_at_v_limit.empty:
            t_cv_start = t_at_v_limit.iloc[0]
            features['TCCC(s)'] = t_cv_start - charge_df['Time(s)'].iloc[0]
            features['TCVC(s)'] = charge_df['Time(s)'].iloc[-1] - t_cv_start
        else:
            features['TCCC(s)'] = charge_df['Time(s)'].iloc[-1] - charge_df['Time(s)'].iloc[0]
            features['TCVC(s)'] = 0
    else:
        features.update({'ICHV(V)': 0, 'UVP_time(s)': 0, 'UVP(V)': v_upper, 'TCCC(s)': 0, 'TCVC(s)': 0})

    if not discharge_df.empty:
        features['IDV(V)'] = discharge_df['Voltage(V)'].iloc[0]
        # [FIX] Use relative time (duration) instead of absolute timestamp
        features['LVP_time(s)'] = discharge_df['Time(s)'].iloc[-1] - discharge_df['Time(s)'].iloc[0]
        features['LVP(V)'] = v_lower
        features['var_I_discharge'] = discharge_df['Current(A)'].var()
        features['var_V_discharge'] = discharge_df['Voltage(V)'].var()
        features['median_V_discharge(V)'] = discharge_df['Voltage(V)'].median()
        features['total_discharge_time(s)'] = discharge_df['Time(s)'].iloc[-1] - discharge_df['Time(s)'].iloc[0]
    else:
        features.update({
            'IDV(V)': 0, 'LVP_time(s)': 0, 'LVP(V)': v_lower, 'var_I_discharge': 0,
            'var_V_discharge': 0, 'median_V_discharge(V)': 0, 'total_discharge_time(s)': 0
        })

    return features


def _calculate_advanced_features(
    charge_df: pd.DataFrame,
    discharge_df: pd.DataFrame,
    rest_df: pd.DataFrame,
    features: Dict[str, Any],
    cycle_df: pd.DataFrame
) -> Dict[str, Any]:
    """Calculates Resistance, RCV, Skewness, and CV Tau."""
    adv_features = {}

    # Internal Resistance
    # Refined logic: Trace back from Discharge Start to find the true Rest End Voltage
    adv_features['Internal_Resistance(Ohm)'] = 0.0

    if not discharge_df.empty:
        # Create a localized State column for logic processing
        # 1: Charge, -1: Discharge, 0: Rest
        # Note: We rely on cycle_df being sorted by time (which it is by construction)
        cycle_df = cycle_df.copy() # Avoid modifying original if not needed, but here it's local
        cycle_df['State'] = 0
        cycle_df.loc[cycle_df['Current(A)'] > 1e-3, 'State'] = 1
        cycle_df.loc[cycle_df['Current(A)'] < -1e-3, 'State'] = -1

        discharge_start_indices = cycle_df.index[cycle_df['State'] == -1].tolist()

        if discharge_start_indices:
            first_discharge_idx = discharge_start_indices[0]

            # Get discharge start values
            v_dis_start = cycle_df.loc[first_discharge_idx, 'Voltage(V)']
            i_dis_start = abs(cycle_df.loc[first_discharge_idx, 'Current(A)'])

            # Look backwards for Rest phase (State 0)
            prev_idx = first_discharge_idx - 1
            v_rest_end = 0.0

            if prev_idx >= 0:
                # Case 1: Ideal transition Rest -> Discharge
                if cycle_df.loc[prev_idx, 'State'] == 0:
                    v_rest_end = cycle_df.loc[prev_idx, 'Voltage(V)']

                    # Phantom Drop Correction
                    # Check if the very last point of Rest has a sudden drop compared to the one before
                    if prev_idx > 0 and cycle_df.loc[prev_idx-1, 'State'] == 0:
                        v_rest_prev = cycle_df.loc[prev_idx-1, 'Voltage(V)']
                        if (v_rest_prev - v_rest_end) > 0.02: # 20mV threshold
                             v_rest_end = v_rest_prev

                # Case 2: Immediate transition Charge -> Discharge (Rare but possible)
                elif cycle_df.loc[prev_idx, 'State'] == 1:
                    v_rest_end = cycle_df.loc[prev_idx, 'Voltage(V)']

            if v_rest_end > 0 and i_dis_start > 1e-3:
                adv_features['Internal_Resistance(Ohm)'] = abs(v_rest_end - v_dis_start) / i_dis_start

    # RCV Ratio
    tcvc = features.get('TCVC', 0)
    adv_features['RCV(V)'] = features.get('TCCC', 0) / tcvc if tcvc > 1e-3 else 0

    # Skewness
    adv_features['skew_V_discharge'] = skew(discharge_df['Voltage(V)']) if not discharge_df.empty else 0

    # CV Current Decay Tau
    if features.get('TCVC', 0) > 10.0 and not charge_df.empty:
        v_limit = features.get('UVP', 4.2)
        cv_df = charge_df[charge_df['Voltage(V)'] >= (v_limit - 0.05)]
        if len(cv_df) > 10 and cv_df['Current(A)'].max() > 0.001:
            adv_features['CV_Current_Tau'] = fit_cv_decay(
                cv_df['Time(s)'].values, cv_df['Current(A)'].values
            )
        else: adv_features['CV_Current_Tau'] = 0.0
    else: adv_features['CV_Current_Tau'] = 0.0

    return adv_features


def _calculate_anchor_features(
    charge_df: pd.DataFrame,
    discharge_df: pd.DataFrame,
    charge_slope_intervals: List[Tuple],
    discharge_slope_intervals: List[Tuple],
    tevi_intervals: List[Tuple],
    tevd_intervals: List[Tuple]
) -> Dict[str, Any]:
    """Calculates anchor features (slopes and time-at-voltage)."""
    anchor_features = {}

    def get_voltage_at_relative_time(df, rel_time):
        if df.empty: return None
        abs_time = df['Time(s)'].iloc[0] + rel_time
        idx = np.searchsorted(df['Time(s)'].values, abs_time)
        idx = min(idx, len(df) - 1)
        return df['Voltage(V)'].iloc[idx]

    # Slopes
    c_dur = charge_df['Time(s)'].iloc[-1] - charge_df['Time(s)'].iloc[0] if not charge_df.empty else 0
    for i, (ps, pe) in enumerate(charge_slope_intervals):
        if c_dur > 1.0:
            v1, v2 = get_voltage_at_relative_time(charge_df, c_dur * ps), get_voltage_at_relative_time(charge_df, c_dur * pe)
            anchor_features[f'charge_slope_{i+1}'] = (v2 - v1) / (c_dur * (pe - ps)) if v1 and v2 and pe > ps else 0
        else: anchor_features[f'charge_slope_{i+1}'] = 0

    d_dur = discharge_df['Time(s)'].iloc[-1] - discharge_df['Time(s)'].iloc[0] if not discharge_df.empty else 0
    for i, (ps, pe) in enumerate(discharge_slope_intervals):
        if d_dur > 1.0:
            v1, v2 = get_voltage_at_relative_time(discharge_df, d_dur * ps), get_voltage_at_relative_time(discharge_df, d_dur * pe)
            anchor_features[f'discharge_slope_{i+1}'] = (v2 - v1) / (d_dur * (pe - ps)) if v1 and v2 and pe > ps else 0
        else: anchor_features[f'discharge_slope_{i+1}'] = 0

    # TEVI/TEVD
    def get_time_for_voltage(df, voltage, direction):
        if df.empty: return None
        mask = df['Voltage(V)'] >= voltage if direction == 'charge' else df['Voltage(V)'] <= voltage
        return df.loc[mask.idxmax(), 'Time(s)'] if mask.any() else None

    for i, (vs, ve) in enumerate(tevi_intervals):
        t1, t2 = get_time_for_voltage(charge_df, vs, 'charge'), get_time_for_voltage(charge_df, ve, 'charge')
        anchor_features[f'TEVI_{i+1}'] = (t2 - t1) if t1 and t2 and t2 > t1 else 0

    for i, (vs, ve) in enumerate(tevd_intervals):
        t1, t2 = get_time_for_voltage(discharge_df, vs, 'discharge'), get_time_for_voltage(discharge_df, ve, 'discharge')
        anchor_features[f'TEVD_{i+1}'] = (t2 - t1) if t1 and t2 and t2 > t1 else 0

    return anchor_features


def extract_features_for_cycle(
    cycle_data: Any, battery_data: Any,
    charge_slopes: List[Tuple], discharge_slopes: List[Tuple],
    tevi_ints: List[Tuple], tevd_ints: List[Tuple],
    output_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """Orchestrates all feature extraction for a single cycle."""

    cycle_df = pd.DataFrame({
        'Time(s)': cycle_data.time_in_s, 'Current(A)': cycle_data.current_in_A,
        'Voltage(V)': cycle_data.voltage_in_V,
        'Charge_Capacity(Ah)': cycle_data.charge_capacity_in_Ah,
        'Discharge_Capacity(Ah)': cycle_data.discharge_capacity_in_Ah,
    })
    cycle_num = cycle_data.cycle_number

    charge_df = cycle_df[cycle_df['Current(A)'] > 0].copy()
    discharge_df = cycle_df[cycle_df['Current(A)'] < 0].copy()
    rest_df = cycle_df[np.isclose(cycle_df['Current(A)'], 0, atol=1e-3)].copy()

    if not discharge_df.empty:
        v_lower = getattr(battery_data, 'min_voltage_limit_in_V', 0)
        cutoff_idx = discharge_df.index[discharge_df['Voltage(V)'] <= v_lower]
        if not cutoff_idx.empty:
            discharge_df = discharge_df.loc[:cutoff_idx[0]]

    direct_feats = _calculate_direct_features(
        cycle_df, charge_df, discharge_df, rest_df, cycle_num, battery_data
    )
    # [MODIFIED] Use shared tool
    nominal_cap = getattr(battery_data, 'nominal_capacity_in_Ah', 2.0)
    ncm_config = {
        'peak_mode': 2,
        'nominal_capacity': nominal_cap,
        'window_length_ic': 21,
        'window_length_dv': 21,
        'peak_height_ic': 0.01,
        # 'peak_height_dv': 0.01,
        'voltage_range_ic': (3.8, 4.0),
        # 'voltage_range_dv': (3.4, 4.1),
        'prominence_ic': 0.01,
        # 'prominence_dv': 0.005,
        'ic_step_size': 0.005,
        'dv_step_size': nominal_cap * 0.005,
        'search_window_dvv': 0.1,
        'search_window_dvp': 0.1,
        'initial_capacity_cut_fraction': 0.02,
        'icv_search_offset_lower': 0.05,
        'icv_search_offset_upper': 0.2,
        'icv_search_direction': 'left',
        'fwhm_method': 'valley_limited',
        'ic_area_config': {
            'method': 'fixed_width',
            'width_v': 0.05
        },
        'aux_peak_config': {
            'voltage_range': (3.1, 3.75),
            'selection': 'first'
        }
    }

    # [FIX] Special Handling for High-Rate (3C) Files
    # Issue: High polarization shifts ICP from ~3.9V down to ~3.76V, falling outside (3.8, 4.0)
    # Target: Tongji3_CY25-05_4--1 / 2 / 3
    # Solution: Widen range to (3.6, 4.05) specifically for these files
    if 'CY25-05_4--' in getattr(battery_data, 'cell_id', ''):
        ncm_config['voltage_range_ic'] = (3.6, 4.05)

    plot_params = None
    if output_dir:
        plot_params = {
            'cell_id': battery_data.cell_id,
            'cycle_num': cycle_num,
            'output_dir': output_dir
        }

    derivative_feats = extract_ic_dv_features(
        discharge_df,
        config=ncm_config,
        plot_params=plot_params
    )

    # [NEW] Calculate Peak 1 vs Peak 3 relationships
    # Note: 'ICP' is Peak 3 (Main Peak in 3.8-4.0), 'ICP_Aux' is Peak 1 (First peak in 3.1-3.75)
    icpl_v_peak3 = derivative_feats.get('ICPL_V', np.nan)
    icpl_v_peak1 = derivative_feats.get('ICPL_V_Aux', np.nan)
    icp_peak3 = derivative_feats.get('ICP', 0.0)
    icp_peak1 = derivative_feats.get('ICP_Aux', 0.0)

    if not np.isnan(icpl_v_peak3) and not np.isnan(icpl_v_peak1):
        derivative_feats['V_diff_Peak3_Peak1'] = icpl_v_peak3 - icpl_v_peak1
    else:
        derivative_feats['V_diff_Peak3_Peak1'] = 0.0

    if icp_peak3 > 1e-6 and not np.isnan(icp_peak1):
        derivative_feats['Ratio_Peak1_Peak3'] = icp_peak1 / icp_peak3
    else:
        derivative_feats['Ratio_Peak1_Peak3'] = 0.0

    advanced_feats = _calculate_advanced_features(charge_df, discharge_df, rest_df, direct_feats, cycle_df)
    anchor_feats = _calculate_anchor_features(
        charge_df, discharge_df, charge_slopes, discharge_slopes, tevi_ints, tevd_ints
    )

    return {**direct_feats, **derivative_feats, **advanced_feats, **anchor_feats}


def process_battery(
    file_path: Path, output_dir: Path,
    charge_slopes: List[Tuple], discharge_slopes: List[Tuple],
    tevi_ints: List[Tuple], tevd_ints: List[Tuple],
    num_cycles: Optional[int] = None
):
    try:
        with open(file_path, 'rb') as f:
            battery_data = AttrDict(pickle.load(f))
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return

    if 'cycle_data' in battery_data and battery_data.cycle_data:
        battery_data.cycle_data = [AttrDict(c) for c in battery_data.cycle_data]

    cell_id = battery_data.cell_id
    cycles_to_proc = battery_data.cycle_data[:num_cycles] if num_cycles else battery_data.cycle_data
    all_feats = []

    for cycle in tqdm(cycles_to_proc, desc=f"Processing {cell_id}", leave=False):
        if not hasattr(cycle, 'time_in_s') or not cycle.time_in_s: continue
        try:
            feats = extract_features_for_cycle(
                cycle, battery_data, charge_slopes, discharge_slopes, tevi_ints, tevd_ints,
                output_dir=output_dir
            )
            all_feats.append(feats)
        except Exception:
            continue

    if not all_feats:
        print(f"Warning: No features for {cell_id}")
        return

    features_df = pd.DataFrame(all_feats)
    out_path = output_dir / f"{cell_id}.csv"
    features_df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


def main():
    processed_data_dir = Path('F:/datasets/battery/Tongji')
    output_dir = project_root / 'results' / 'Tongji'
    output_dir.mkdir(parents=True, exist_ok=True)

    charge_slope_intervals = [(0.1, 0.3), (0.1, 0.5), (0.1, 0.8)]
    discharge_slope_intervals = [(0.1, 0.3), (0.1, 0.5), (0.1, 0.8)]
    tevi_intervals = [(3.5, 3.7), (3.7, 3.9), (3.9, 4.1)]
    tevd_intervals = [(4.1, 3.9), (3.9, 3.7), (3.7, 3.5)]
    num_cycles_to_extract = 100

    if not processed_data_dir.exists():
        print(f"Error: Dir not found '{processed_data_dir}'")
        return

    pkl_files = list(processed_data_dir.glob('*.pkl'))
    if not pkl_files:
        print(f"Error: No .pkl files in '{processed_data_dir}'")
        return

    for file_path in pkl_files:
        process_battery(
            file_path, output_dir,
            charge_slope_intervals, discharge_slope_intervals,
            tevi_intervals, tevd_intervals,
            num_cycles=num_cycles_to_extract
        )


if __name__ == '__main__':
    main()

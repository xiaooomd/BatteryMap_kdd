"""
Feature Extraction Script for RWTH Battery Dataset
==================================================

Refactored to use shared utilities (src.utils) for consistent algorithms.
"""
import pickle
import warnings
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from scipy.stats import skew
from tqdm import tqdm

# Add project root to path to allow importing src
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.utils.math_tools import fit_cv_decay
from src.utils.feature_tools import extract_ic_dv_features

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# --- Type Hints ---
CycleData = Dict[str, Any]
BatteryData = Dict[str, Any]
Intervals = List[Tuple[float, float]]
Features = Dict[str, Any]


# def _calculate_discharge_cv_features(
#     cv_df: pd.DataFrame,
#     cc_capacity: float
# ) -> Features:
#     """Calculates features specifically for the discharge CV tail."""
#     feats = {
#         'Discharge_CV_Capacity': 0.0,
#         'Discharge_CV_Time': 0.0,
#         'Discharge_CV_Peak_Current': 0.0,
#         'Ratio_CC_CV_Capacity': 0.0
#     }
#
#     if cv_df.empty:
#         return feats
#
#     time_s = cv_df['Time(s)'].values
#     curr_a = cv_df['Current(A)'].abs().values
#
#     if len(time_s) > 1:
#         cv_cap_ah = trapezoid(y=curr_a, x=time_s) / 3600.0
#     else:
#         cv_cap_ah = 0.0
#
#     feats['Discharge_CV_Capacity'] = cv_cap_ah
#     feats['Discharge_CV_Time'] = time_s[-1] - time_s[0]
#     feats['Discharge_CV_Peak_Current'] = curr_a.min()
#
#     if cv_cap_ah > 1e-6:
#         feats['Ratio_CC_CV_Capacity'] = cc_capacity / cv_cap_ah
#     else:
#         feats['Ratio_CC_CV_Capacity'] = 0
#
#     return feats


def _calculate_direct_features(
    cycle_df: pd.DataFrame,
    charge_df: pd.DataFrame,
    discharge_df: pd.DataFrame,
    cycle_num: int,
    battery_data: BatteryData
) -> Features:
    """Calculates direct features (Capacity, Energy, Times)."""
    features: Features = {}

    features['Cycle_Number'] = cycle_num

    if not discharge_df.empty:
        q_dis_max = trapezoid(discharge_df['Current(A)'].abs(), x=discharge_df['Time(s)']) / 3600.0
    else:
        q_dis_max = 0.0

    if not charge_df.empty:
        q_chg_max = trapezoid(charge_df['Current(A)'].abs(), x=charge_df['Time(s)']) / 3600.0
    else:
        q_chg_max = 0.0

    features['Discharge_Capacity(Ah)'] = q_dis_max
    features['Charge_Capacity(Ah)'] = q_chg_max

    features['Coulombic_Efficiency'] = (q_dis_max / q_chg_max) if q_chg_max > 1e-6 else 0

    # Energy (Wh)
    if not charge_df.empty:
        p_chg = charge_df['Voltage(V)'] * charge_df['Current(A)'].abs()
        features['Charge_Energy(Wh)'] = trapezoid(y=p_chg, x=charge_df['Time(s)']) / 3600.0
    else:
        features['Charge_Energy(Wh)'] = 0

    if not discharge_df.empty:
        p_dis = discharge_df['Voltage(V)'] * discharge_df['Current(A)'].abs()
        features['Discharge_Energy(Wh)'] = trapezoid(y=p_dis, x=discharge_df['Time(s)']) / 3600.0
    else:
        features['Discharge_Energy(Wh)'] = 0

    if features['Charge_Energy(Wh)'] > 1e-6:
        features['Energy_Efficiency'] = features['Discharge_Energy(Wh)'] / features['Charge_Energy(Wh)']
    else:
        features['Energy_Efficiency'] = 0

    # Protocols
    c_proto = battery_data.get('charge_protocol', [{}])
    features['charge_current(A)'] = c_proto[0].get('current_in_A', 0) if c_proto else 0
    d_proto = battery_data.get('discharge_protocol', [{}])
    features['discharge_current(A)'] = d_proto[0].get('current_in_A', 0) if d_proto else 0

    # --- Charge Phase ---
    features['CV_Current_Tau'] = 0.0
    if not charge_df.empty:
        features['ICHV(V)'] = charge_df['Voltage(V)'].iloc[0]
        features['UVP_time(s)'] = charge_df['Time(s)'].iloc[-1] - charge_df['Time(s)'].iloc[0]
        v_limit = battery_data.get('max_voltage_limit_in_V', 0)
        features['UVP(V)'] = v_limit

        CV_THRESH = v_limit - 0.01
        if v_limit > 0 and charge_df['Voltage(V)'].max() >= CV_THRESH:
            cv_mask = charge_df['Voltage(V)'] >= CV_THRESH
            cv_df = charge_df[cv_mask]

            if not cv_df.empty:
                t_cv_start = cv_df['Time(s)'].iloc[0]
                features['TCCC(s)'] = t_cv_start - charge_df['Time(s)'].iloc[0]
                features['TCVC(s)'] = charge_df['Time(s)'].iloc[-1] - t_cv_start

                # Tau
                valid_i = cv_df['Current(A)'] > 0.001
                if valid_i.sum() > 10:
                    features['CV_Current_Tau'] = fit_cv_decay(
                        cv_df.loc[valid_i, 'Time(s)'].values,
                        cv_df.loc[valid_i, 'Current(A)'].values
                    )
            else:
                features['TCCC(s)'] = features['UVP_time(s)']
                features['TCVC(s)'] = 0
        else:
            features['TCCC(s)'] = features['UVP_time(s)']
            features['TCVC(s)'] = 0
    else:
        features.update({'ICHV(V)': 0, 'UVP_time(s)': 0, 'UVP(V)': 0, 'TCCC(s)': 0, 'TCVC(s)': 0})

    # --- Discharge Phase ---
    if not discharge_df.empty:
        features['IDV(V)'] = discharge_df['Voltage(V)'].iloc[0]
        features['LVP_time(s)'] = discharge_df['Time(s)'].iloc[-1] - discharge_df['Time(s)'].iloc[0]
        features['LVP(V)'] = battery_data.get('min_voltage_limit_in_V', 0)
        features['var_I_discharge'] = discharge_df['Current(A)'].var()
        features['var_V_discharge'] = discharge_df['Voltage(V)'].var()
        features['median_V_discharge(V)'] = discharge_df['Voltage(V)'].median()
        features['total_discharge_time(s)'] = features['LVP_time(s)']
    else:
        features.update({
            'IDV(V)': 0, 'LVP_time(s)': 0, 'LVP(V)': 0,
            'var_I_discharge': 0, 'var_V_discharge': 0,
            'median_V_discharge(V)': 0, 'total_discharge_time(s)': 0
        })

    return features


def _calculate_advanced_features(
    charge_df: pd.DataFrame,
    discharge_df: pd.DataFrame,
    features: Features
) -> Features:
    """Calculates IR, RCV, Skewness."""
    adv_features: Features = {}

    if not charge_df.empty and not discharge_df.empty:
        v_dis_end = discharge_df['Voltage(V)'].iloc[-1]
        v_chg_start = charge_df['Voltage(V)'].iloc[0]
        i_chg_start = charge_df['Current(A)'].iloc[0]

        if i_chg_start > 0.001:
            adv_features['Internal_Resistance(Ohm)'] = (v_chg_start - v_dis_end) / i_chg_start
        else:
            adv_features['Internal_Resistance(Ohm)'] = 0.0
    else:
        adv_features['Internal_Resistance(Ohm)'] = 0.0

    tcvc = features.get('TCVC', 0)
    tccc = features.get('TCCC', 0)
    adv_features['RCV(V)'] = (tccc / tcvc) if tcvc > 0.001 else 0.0

    if not discharge_df.empty:
        adv_features['skew_V_discharge'] = skew(discharge_df['Voltage(V)'])
    else:
        adv_features['skew_V_discharge'] = 0.0

    return adv_features


def _get_voltage_at_relative_time(
    df: pd.DataFrame, relative_time: float
) -> Optional[float]:
    """Helper for anchor features using searchsorted."""
    if df.empty:
        return None

    start_time = df['Time(s)'].iloc[0]
    target_abs_time = start_time + relative_time

    times = df['Time(s)'].values
    volts = df['Voltage(V)'].values

    idx = np.searchsorted(times, target_abs_time, side='left')

    if idx == 0:
        return volts[0]
    if idx >= len(times):
        return volts[-1]

    return volts[idx]


def _get_time_for_voltage(
    df: pd.DataFrame, voltage: float, direction: str
) -> Optional[float]:
    """Helper: First time voltage crosses threshold."""
    if df.empty:
        return None

    if direction == 'charge':
        mask = df['Voltage(V)'] >= voltage
    else:
        mask = df['Voltage(V)'] <= voltage

    if mask.any():
        idx = mask.idxmax()
        return df.loc[idx, 'Time(s)']
    return None


def _calculate_anchor_features(
    charge_df: pd.DataFrame,
    discharge_df: pd.DataFrame,
    charge_intervals: Intervals,
    discharge_intervals: Intervals,
    tevi_intervals: Intervals,
    tevd_intervals: Intervals
) -> Features:
    """Calculates slope and time-interval features."""
    features: Features = {}

    # Charge Slopes
    chg_dur = 0.0
    if not charge_df.empty:
        chg_dur = charge_df['Time(s)'].iloc[-1] - charge_df['Time(s)'].iloc[0]

    for i, (p_start, p_end) in enumerate(charge_intervals):
        key = f'charge_slope_{i+1}'
        if not charge_df.empty and chg_dur > 1.0:
            t1, t2 = chg_dur * p_start, chg_dur * p_end
            v1 = _get_voltage_at_relative_time(charge_df, t1)
            v2 = _get_voltage_at_relative_time(charge_df, t2)
            if v1 is not None and v2 is not None:
                features[key] = (v2 - v1) / (t2 - t1)
            else:
                features[key] = 0.0
        else:
            features[key] = 0.0

    # Discharge Slopes
    dis_dur = 0.0
    if not discharge_df.empty:
        dis_dur = discharge_df['Time(s)'].iloc[-1] - discharge_df['Time(s)'].iloc[0]

    for i, (p_start, p_end) in enumerate(discharge_intervals):
        key = f'discharge_slope_{i+1}'
        if not discharge_df.empty and dis_dur > 1.0:
            t1, t2 = dis_dur * p_start, dis_dur * p_end
            v1 = _get_voltage_at_relative_time(discharge_df, t1)
            v2 = _get_voltage_at_relative_time(discharge_df, t2)
            if v1 is not None and v2 is not None:
                features[key] = (v2 - v1) / (t2 - t1)
            else:
                features[key] = 0.0
        else:
            features[key] = 0.0

    # TEVI / TEVD
    for i, (v_s, v_e) in enumerate(tevi_intervals):
        t1 = _get_time_for_voltage(charge_df, v_s, 'charge')
        t2 = _get_time_for_voltage(charge_df, v_e, 'charge')
        features[f'TEVI_{i+1}'] = (t2 - t1) if (t1 and t2 and t2 > t1) else 0.0

    for i, (v_s, v_e) in enumerate(tevd_intervals):
        t1 = _get_time_for_voltage(discharge_df, v_s, 'discharge')
        t2 = _get_time_for_voltage(discharge_df, v_e, 'discharge')
        features[f'TEVD_{i+1}'] = (t2 - t1) if (t1 and t2 and t2 > t1) else 0.0

    return features


def extract_features_for_cycle(
    cycle_data: CycleData,
    battery_data: BatteryData,
    charge_slopes: Intervals,
    discharge_slopes: Intervals,
    tevi_ints: Intervals,
    tevd_ints: Intervals,
    output_dir: Optional[Path] = None
) -> Features:
    """Master function to extract all features for a single cycle."""

    # 1. Prepare Data
    time_s = np.array(cycle_data['time_in_s']) / 1000.0
    cycle_df = pd.DataFrame({
        'Time(s)': time_s,
        'Current(A)': cycle_data['current_in_A'],
        'Voltage(V)': cycle_data['voltage_in_V'],
        'Charge_Capacity(Ah)': cycle_data['charge_capacity_in_Ah'],
        'Discharge_Capacity(Ah)': cycle_data['discharge_capacity_in_Ah'],
    })
    cycle_num = cycle_data['cycle_number']

    # Phase Splitting
    charge_df = cycle_df[cycle_df['Current(A)'] > 0].copy()
    discharge_df = cycle_df[cycle_df['Current(A)'] < 0].copy()

    # Basic Features
    direct_feats = _calculate_direct_features(
        cycle_df, charge_df, discharge_df, cycle_num, battery_data
    )

    # 2. Derivative Features (using shared tool)
    # Modified [2024-01] per need_fixed.md: Use Charge Data (CC Phase only)
    if not charge_df.empty:
        # --- Robust CC Phase Extraction ---
        # Logic: Identify the transition from CC to CV by monitoring Current drop
        # while Voltage is high.

        # 1. Estimate Target CC Current (I_cc)
        # Try to get from protocol first
        c_proto = battery_data.get('charge_protocol', [{}])
        i_cc_target = c_proto[0].get('current_in_A', 0) if c_proto else 0

        # If protocol is missing or invalid, estimate from data
        # Use a high percentile (e.g. 90th) to capture the plateau value
        if i_cc_target <= 0.001:
            i_cc_target = charge_df['Current(A)'].quantile(0.90)

        # 2. Find CC-CV Transition Point
        # We look for the point where:
        #   a) Voltage is near the maximum (to avoid false triggers at start)
        #   b) Current drops significantly below the steady CC value

        v_max = charge_df['Voltage(V)'].max()
        # "High Voltage" region: within 50mV of max voltage
        v_threshold = v_max - 0.05

        # "Current Drop" threshold: drops below 98% of target (sensitive detection)
        i_threshold = i_cc_target * 0.98

        v_vals = charge_df['Voltage(V)'].values
        i_vals = charge_df['Current(A)'].values

        # Find indices that satisfy BOTH conditions
        # Note: We assume current is positive for charge
        is_cv_region = (v_vals >= v_threshold) & (i_vals < i_threshold)
        cv_indices = np.where(is_cv_region)[0]

        if len(cv_indices) > 0:
            # The start of CV is the first point satisfying the condition
            cut_idx = cv_indices[0]
            # Ensure we have enough points (at least 10)
            if cut_idx > 10:
                cc_charge_df = charge_df.iloc[:cut_idx].copy()
            else:
                cc_charge_df = charge_df.copy()
        else:
            # No CV detected (Pure CC charge), use full data
            cc_charge_df = charge_df.copy()

        # --- Pre-processing: Cut Initial Voltage Jump (IR Effect) ---
        # Requirement: Skip initial 20mV rise to avoid fake dQ/dV peaks.
        if not cc_charge_df.empty:
            v_min_start = cc_charge_df['Voltage(V)'].min()
            v_cut_threshold = v_min_start + 0.02  # 20mV offset

            filtered_df = cc_charge_df[cc_charge_df['Voltage(V)'] > v_cut_threshold]

            # Ensure we don't filter everything out (safety check)
            if len(filtered_df) > 10:
                cc_charge_df = filtered_df.copy()

        # NCM Config (Compatible with RWTH Charge curve)
        nominal_cap = battery_data.get('nominal_capacity_in_Ah', 2.0)
        ncm_config = {
            'peak_mode': 1,
            'nominal_capacity': nominal_cap,
            'window_length_ic': 51,
            'window_length_dv': 21,
            'peak_height_ic': 0.01,
            # 'peak_height_dv': 0.01,
            'voltage_range_ic': (3.7, 3.88), # Extended upper bound for charge
            # 'voltage_range_dv': (3.4, 4.15),
            'prominence_ic': 0.01,
            # 'prominence_dv': 0.005,
            'ic_step_size': 0.002,
            'dv_step_size': nominal_cap * 0.005,
            'search_window_dvv': 0.1,
            'search_window_dvp': 0.1,
            'initial_capacity_cut_fraction': 0.02,
            'icv_search_offset_lower': 0.05,
            'icv_search_offset_upper': 0.1,
            'plot_interval': 50, # Debug: Plot every cycle

            # --- Need Fixed: Custom RWTH Logic ---
            'ic_area_voltage_range': (3.75, 3.85), # Task 1 & 4
            'disable_dvv': True, # Task 2
            'dvp_capacity_range': (0.1, 0.4), # Task 3 & 4
            'dvpl_v_capacity_fraction': 0.5, # Task 3
        }

        plot_params = None
        if output_dir:
            plot_params = {
                'cell_id': battery_data.get('cell_id', 'unknown'),
                'cycle_num': cycle_num,
                'output_dir': output_dir
            }

        # Prepare data for tool (map Charge Cap -> Discharge Cap)
        df_tool = cc_charge_df.copy()
        if 'Charge_Capacity(Ah)' in df_tool.columns:
            df_tool['Discharge_Capacity(Ah)'] = df_tool['Charge_Capacity(Ah)']

        deriv_feats = extract_ic_dv_features(
            df_tool,
            config=ncm_config,
            plot_params=plot_params
        )
    else:
        deriv_feats = extract_ic_dv_features(pd.DataFrame(), config={}) # Returns defaults

    # 3. Calculate CV specific features (RWTH custom logic retained roughly)
    # We attempt to isolate CV tail from discharge if needed, but for simplicity
    # and consistency with other scripts, we might skip the 'Strict CC' logic unless crucial.
    # However, direct_feats already calculates standard CC/CV split for Charge.
    # RWTH specifically looked for Discharge CV tail.
    # Let's keep a simplified version if possible, or just default 0 if not critical.
    # The original script had complex splitting. We'll simplify:
    # If there is a tail where voltage is constant-ish at end of discharge?
    # For now, let's assume standard behavior.
    # cv_feats = {'Discharge_CV_Capacity': 0, 'Discharge_CV_Time': 0, 'Ratio_CC_CV_Capacity': 0}

    # Other features
    adv_feats = _calculate_advanced_features(charge_df, discharge_df, direct_feats)
    anchor_feats = _calculate_anchor_features(
        charge_df, discharge_df,
        charge_slopes, discharge_slopes,
        tevi_ints, tevd_ints
    )

    # return {**direct_feats, **deriv_feats, **cv_feats, **adv_feats, **anchor_feats}
    return {**direct_feats, **deriv_feats, **adv_feats, **anchor_feats}


def process_battery(
    file_path: Path,
    output_dir: Path,
    charge_slopes: Intervals,
    discharge_slopes: Intervals,
    tevi_ints: Intervals,
    tevd_ints: Intervals,
    num_cycles: Optional[int] = None
):
    """Processes a single .pkl file."""
    try:
        with open(file_path, 'rb') as f:
            battery_data: BatteryData = pickle.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return

    cycles = battery_data.get('cycle_data', [])
    if not cycles:
        print(f"No cycle data in {file_path}")
        return

    if num_cycles:
        cycles = cycles[:num_cycles]

    cell_id = battery_data.get('cell_id', file_path.stem)
    all_features = []

    for c_data in tqdm(cycles, desc=f"Processing {cell_id}", leave=False):
        if not c_data.get('time_in_s'):
            continue
        try:
            feats = extract_features_for_cycle(
                c_data, battery_data,
                charge_slopes, discharge_slopes,
                tevi_ints, tevd_ints,
                output_dir=output_dir
            )
            all_features.append(feats)
        except Exception as e:
            print(f"Error in cycle {c_data.get('cycle_number')}: {e}")
            continue

    if not all_features:
        return

    df_out = pd.DataFrame(all_features)
    out_path = output_dir / f"{cell_id}.csv"
    df_out.to_csv(out_path, index=False)
    print(f"Saved {out_path}")


def main():
    # --- Configuration ---
    processed_dir = Path('F:/datasets/battery/RWTH')
    output_dir = project_root / 'results' / 'RWTH'

    output_dir.mkdir(parents=True, exist_ok=True)

    # Hyperparameters
    charge_slopes = [(0.1, 0.3), (0.1, 0.5), (0.1, 0.8)]
    discharge_slopes = [(0.1, 0.3), (0.1, 0.5), (0.1, 0.8)]
    tevi_ints = [(3.55, 3.7), (3.7, 3.85), (3.55, 3.85)]
    tevd_ints = [(3.85, 3.7), (3.7, 3.55), (3.85, 3.55)]

    num_cycles_to_extract = 100

    # --- Execution ---
    if not processed_dir.exists():
         print(f"Data directory not found: {processed_dir}")
         return

    files = list(processed_dir.glob('*.pkl'))
    if not files:
        print(f"No .pkl files found in {processed_dir}")
        return

    print(f"Found {len(files)} files. Starting extraction...")

    for pkl_file in files:
        process_battery(
            pkl_file, output_dir,
            charge_slopes, discharge_slopes,
            tevi_ints, tevd_ints,
            num_cycles = num_cycles_to_extract
        )

    print("All tasks completed.")


if __name__ == '__main__':
    main()

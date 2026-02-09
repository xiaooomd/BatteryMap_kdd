"""
Feature Extraction Script for ZN-ion Battery Dataset
======================================================

Refactored to use shared utilities (src.utils).
- Replaced local IC/DV feature calculation with `extract_ic_dv_features`.
- Standardized integration to `scipy.integrate.trapezoid`.
- Preserved Zn-ion specific voltage intervals.
"""
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import sys
from typing import List, Dict, Any, Optional, Tuple

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


def _calculate_direct_features(
    cycle_df: pd.DataFrame, charge_df: pd.DataFrame, discharge_df: pd.DataFrame,
    cycle_num: int, battery_data: Dict[str, Any]
) -> Dict[str, Any]:
    """Calculates direct features from raw cycle data."""
    features = {}
    features['Cycle_Number'] = cycle_num

    # Workload Type
    if not charge_df.empty and not discharge_df.empty:
        features['Workload_Type'] = '0' if charge_df['Time(s)'].iloc[0] < discharge_df['Time(s)'].iloc[0] else '1'
    elif not charge_df.empty: features['Workload_Type'] = '2'
    elif not discharge_df.empty: features['Workload_Type'] = '3'
    else: features['Workload_Type'] = '-1'

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
    features['Coulombic_Efficiency'] = (q_dis / q_chg) if q_chg > 1e-6 else 0.0

    # Energy
    if not charge_df.empty:
        p_chg = charge_df['Voltage(V)'].values * charge_df['Current(A)'].values
        features['Charge_Energy(Wh)'] = trapezoid(y=p_chg, x=charge_df['Time(s)'].values) / 3600.0
    else: features['Charge_Energy(Wh)'] = 0.0

    if not discharge_df.empty:
        p_dis = discharge_df['Voltage(V)'].values * np.abs(discharge_df['Current(A)'].values)
        features['Discharge_Energy(Wh)'] = trapezoid(y=p_dis, x=discharge_df['Time(s)'].values) / 3600.0
    else: features['Discharge_Energy(Wh)'] = 0.0

    features['Energy_Efficiency'] = (features['Discharge_Energy(Wh)'] / features['Charge_Energy(Wh)']) if features['Charge_Energy(Wh)'] > 1e-6 else 0.0

    # C-Rates
    charge_proto = battery_data.get('charge_protocol', [{}])
    features['charge_c_rate'] = charge_proto[0].get('rate_in_C', 0) if charge_proto else 0.0
    discharge_proto = battery_data.get('discharge_protocol', [{}])
    features['discharge_c_rate'] = discharge_proto[0].get('rate_in_C', 0) if discharge_proto else 0.0

    # Charge Phase
    v_upper = battery_data.get('max_voltage_limit_in_V', 0.0)
    features['UVP(V)'] = v_upper
    if not charge_df.empty:
        features['ICHV(V)'] = charge_df['Voltage(V)'].iloc[0]
        # Fix: Use relative time duration instead of absolute timestamp (need_fixed.md)
        features['UVP_time(s)'] = charge_df['Time(s)'].iloc[-1] - charge_df['Time(s)'].iloc[0]

        # Fix: Force TCVC to 0 as there is no CV phase in this dataset (need_fixed.md)
        features['TCCC(s)'] = features['UVP_time(s)']
        features['TCVC(s)'] = 0.0
    else:
        features.update({'ICHV(V)': 0, 'UVP_time(s)': 0, 'TCCC(s)': 0, 'TCVC(s)': 0})

    # Discharge Phase
    features['LVP(V)'] = battery_data.get('min_voltage_limit_in_V', 0.0)
    if not discharge_df.empty:
        features['IDV(V)'] = discharge_df['Voltage(V)'].iloc[0]
        # Fix: Use relative time duration instead of absolute timestamp (need_fixed.md)
        features['LVP_time(s)'] = discharge_df['Time(s)'].iloc[-1] - discharge_df['Time(s)'].iloc[0]
        features['var_I_discharge'] = discharge_df['Current(A)'].var()
        features['var_V_discharge'] = discharge_df['Voltage(V)'].var()
        features['median_V_discharge(V)'] = discharge_df['Voltage(V)'].median()
        features['total_discharge_time(s)'] = discharge_df['Time(s)'].iloc[-1] - discharge_df['Time(s)'].iloc[0]
    else:
        features.update({'IDV(V)': 0, 'LVP_time(s)': 0, 'var_I_discharge': 0, 'var_V_discharge': 0, 'median_V_discharge(V)': 0, 'total_discharge_time(s)': 0})

    return features


def _calculate_advanced_features(
    cycle_df: pd.DataFrame, charge_df: pd.DataFrame, discharge_df: pd.DataFrame,
    direct_features: Dict[str, Any], battery_data: Dict[str, Any]
) -> Dict[str, Any]:
    """Calculates IR, RCV, Skew, CV Tau."""
    adv_features = {}

    if not charge_df.empty and not discharge_df.empty:
        v_dis_start = discharge_df['Voltage(V)'].iloc[0]
        i_dis_start = abs(discharge_df['Current(A)'].iloc[0])
        rest_df = cycle_df[(cycle_df['Time(s)'] > charge_df['Time(s)'].iloc[-1]) & (cycle_df['Time(s)'] < discharge_df['Time(s)'].iloc[0])]
        v_ocv = rest_df['Voltage(V)'].iloc[-1] if not rest_df.empty else charge_df['Voltage(V)'].iloc[-1]
        adv_features['Internal_Resistance(Ohm)'] = ((v_ocv - v_dis_start) / i_dis_start) if i_dis_start > 1e-6 else 0.0
    else: adv_features['Internal_Resistance(Ohm)'] = 0.0

    tcvc = direct_features.get('TCVC', 0)
    adv_features['RCV(V)'] = (direct_features.get('TCCC', 0) / tcvc) if tcvc > 1e-6 else 0.0
    adv_features['skew_V_discharge'] = skew(discharge_df['Voltage(V)']) if not discharge_df.empty and discharge_df['Voltage(V)'].std() > 1e-6 else 0.0

    v_limit = battery_data.get('max_voltage_limit_in_V', np.inf)
    if not charge_df.empty and tcvc > 10.0:
        cv_df = charge_df[charge_df['Voltage(V)'] >= (v_limit - 0.01)]
        if len(cv_df) > 10 and cv_df['Current(A)'].max() > 0.001:
            adv_features['CV_Current_Tau'] = fit_cv_decay(cv_df['Time(s)'].values, cv_df['Current(A)'].values)
        else: adv_features['CV_Current_Tau'] = 0.0
    else: adv_features['CV_Current_Tau'] = 0.0

    return adv_features


def _calculate_anchor_features(
    charge_df: pd.DataFrame, discharge_df: pd.DataFrame,
    charge_slope_intervals: List[Tuple], discharge_slope_intervals: List[Tuple],
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
    for i, (ps, pe) in enumerate(charge_slope_intervals):
        dt = c_dur * (pe - ps)
        if dt > 1e-6:
            vs, ve = get_v_at_rel_time(charge_df, c_dur*ps), get_v_at_rel_time(charge_df, c_dur*pe)
            anchor_features[f'charge_slope_{i+1}'] = (ve - vs) / dt if vs and ve else 0.0
        else: anchor_features[f'charge_slope_{i+1}'] = 0.0

    d_dur = (discharge_df['Time(s)'].iloc[-1] - discharge_df['Time(s)'].iloc[0]) if not discharge_df.empty else 0
    for i, (ps, pe) in enumerate(discharge_slope_intervals):
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
) -> Optional[Dict[str, Any]]:
    """Orchestrates feature extraction for a single cycle."""
    cycle_df = pd.DataFrame({
        'Time(s)': cycle_data['time_in_s'], 'Current(A)': cycle_data['current_in_A'],
        'Voltage(V)': cycle_data['voltage_in_V'],
        'Charge_Capacity(Ah)': cycle_data['charge_capacity_in_Ah'],
        'Discharge_Capacity(Ah)': cycle_data['discharge_capacity_in_Ah']
    })
    cycle_num = cycle_data.get('cycle_number', 0)

    charge_df = cycle_df[cycle_df['Current(A)'] > 0].copy()
    discharge_df = cycle_df[cycle_df['Current(A)'] < 0].copy()

    if not discharge_df.empty:
        v_lower = battery_data.get('min_voltage_limit_in_V')
        if v_lower is not None:
            cutoff_idx = discharge_df.index[discharge_df['Voltage(V)'] <= v_lower]
            if not cutoff_idx.empty:
                discharge_df = discharge_df.loc[:cutoff_idx[0]]

    direct_feats = _calculate_direct_features(cycle_df, charge_df, discharge_df, cycle_num, battery_data)
    total_time = direct_feats.get('TCCC', 0) + direct_feats.get('TCVC', 0) + direct_feats.get('total_discharge_time', 0)
    if total_time > 10000: return None

    nominal_cap = battery_data.get('nominal_capacity_in_Ah', 0.5)
    zn_ion_config = {
        'peak_mode': 2,
        'nominal_capacity': nominal_cap,
        'window_length_ic': 21,
        'window_length_dv': 21,
        'peak_height_ic': 0.0001,
        # 'peak_height_dv': 0.01,
        'voltage_range_ic': (1.0, 1.6),
        # 'voltage_range_dv': (1.0, 1.6),
        'prominence_ic': 0.0001,
        # 'prominence_dv': 0.01,
        'ic_step_size': 0.005,
        'dv_step_size': nominal_cap * 0.005,
        'search_window_dvv': 0.3,
        'search_window_dvp': 0.3,
        'initial_capacity_cut_fraction': 0.02,
        'icv_search_offset_lower': 0.01,
        'icv_search_offset_upper': 0.25,
        'icv_search_direction': 'left'
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
        config=zn_ion_config,
        plot_params=plot_params
    )

    advanced_feats = _calculate_advanced_features(cycle_df, charge_df, discharge_df, direct_feats, battery_data)
    anchor_feats = _calculate_anchor_features(charge_df, discharge_df, charge_slopes, discharge_slopes, tevi_ints, tevd_ints)

    return {**direct_feats, **derivative_feats, **advanced_feats, **anchor_feats}


def process_battery(
    file_path: Path, output_dir: Path, charge_slopes: List[Tuple],
    discharge_slopes: List[Tuple], tevi_ints: List[Tuple], tevd_ints: List[Tuple],
    num_cycles: Optional[int] = None
):
    try:
        with open(file_path, 'rb') as f:
            data_dict = pickle.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return

    battery_data = data_dict
    if 'cycle_data' in battery_data and battery_data['cycle_data']:
        battery_data['cycle_data'] = [c if isinstance(c, dict) else dict(c) for c in battery_data['cycle_data']]
    else:
        print(f"No cycle data for {battery_data.get('cell_id', 'Unknown')}")
        return

    cycles = battery_data.get('cycle_data', [])
    cell_id = battery_data.get('cell_id', file_path.stem)
    valid_cycles_count = 0
    all_feats = []

    for cycle_data in tqdm(cycles, desc=f"Processing {cell_id}", leave=False):
        if not cycle_data.get('time_in_s'): continue
        try:
            feats = extract_features_for_cycle(cycle_data, battery_data, charge_slopes, discharge_slopes, tevi_ints, tevd_ints, output_dir=output_dir)
            if feats:
                valid_cycles_count += 1
                feats['Cycle_Number'] = valid_cycles_count
                all_feats.append(feats)
                if num_cycles and valid_cycles_count >= num_cycles: break
        except Exception:
            continue

    if not all_feats:
        print(f"No valid features for {cell_id}")
        return

    df_out = pd.DataFrame(all_feats)
    out_path = output_dir / f"{cell_id}.csv"
    df_out.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


def main():
    processed_data_dir = Path('F:/datasets/battery/ZN-coin')
    output_dir = project_root / 'results' / 'ZNion'
    output_dir.mkdir(parents=True, exist_ok=True)

    # ZN-ion Specific Intervals
    charge_slopes = [(0.1, 0.3), (0.1, 0.5), (0.1, 0.8)]
    discharge_slopes = [(0.1, 0.3), (0.1, 0.5), (0.1, 0.8)]
    tevi_ints = [(1.0, 1.2), (1.2, 1.4), (1.4, 1.6)]
    tevd_ints = [(1.6, 1.4), (1.4, 1.2), (1.2, 1.0)]
    num_cycles = 100

    if not processed_data_dir.exists():
        print(f"Error: Directory '{processed_data_dir}' not found.")
        return

    pkl_files = list(processed_data_dir.glob('*.pkl'))
    if not pkl_files:
        print(f"Error: No .pkl files in '{processed_data_dir}'.")
        return

    for pkl_file in pkl_files:
        process_battery(pkl_file, output_dir, charge_slopes, discharge_slopes, tevi_ints, tevd_ints, num_cycles=num_cycles)

if __name__ == '__main__':
    main()

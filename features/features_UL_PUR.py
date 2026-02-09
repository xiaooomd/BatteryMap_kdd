"""
Feature Extraction Script for UL-PUR Battery Dataset
=====================================================

Refactored to use shared utilities (src.utils) for consistent algorithms.
- Replaced local IC/DV feature calculation with `extract_ic_dv_features`.
- Preserved the enhanced thermal feature calculation (`_calculate_enhanced_thermal_features`)
  as it provides specific value for this dataset.
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
from scipy.signal import savgol_filter
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.utils.math_tools import fit_cv_decay
from src.utils.feature_tools import extract_ic_dv_features

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


def _calculate_enhanced_thermal_features(
    df: pd.DataFrame,
    phase_name: str
) -> Dict[str, float]:
    """
    Calculates enhanced temperature features (smoothed, heat rate, thermal load).
    This function is retained locally for its specific detailed analysis.
    """
    default_keys = [
        f'MAT_{phase_name}', f'MET_{phase_name}', f'MinT_{phase_name}',
        f'T_rise_{phase_name}', f'Max_HeatRate_{phase_name}',
        f'Mean_HeatRate_{phase_name}', f'Thermal_Load_{phase_name}'
    ]
    features = {k: 0.0 for k in default_keys}

    if 'Temperature(C)' not in df.columns or df.empty or len(df) < 5:
        return features

    temp_raw = df['Temperature(C)'].values
    time_arr = df['Time(s)'].values

    # Savitzky-Golay smoothing
    window_len = min(len(temp_raw), 51)
    if window_len % 2 == 0: window_len -= 1

    if window_len < 5:
        temp_smooth = temp_raw
    else:
        try:
            temp_smooth = savgol_filter(temp_raw, window_length=window_len, polyorder=3)
        except ValueError:
            temp_smooth = temp_raw

    # Basic stats
    features[f'MAT_{phase_name}'] = float(np.max(temp_smooth))
    features[f'MET_{phase_name}'] = float(np.mean(temp_smooth))
    features[f'MinT_{phase_name}'] = float(np.min(temp_smooth))
    features[f'T_rise_{phase_name}'] = float(temp_smooth[-1] - temp_smooth[0])

    # Heat Rate (dT/dt)
    if len(time_arr) > 5:
        dT_dt = np.gradient(temp_smooth, time_arr)
        features[f'Max_HeatRate_{phase_name}'] = float(np.max(dT_dt))
        features[f'Mean_HeatRate_{phase_name}'] = float(np.mean(dT_dt))

    # Thermal Load (Integral of Temp over Time)
    try:
        thermal_load = trapezoid(y=temp_smooth, x=time_arr)
        features[f'Thermal_Load_{phase_name}'] = float(thermal_load)
    except Exception:
        features[f'Thermal_Load_{phase_name}'] = 0.0

    return features


def _calculate_direct_features(
    cycle_df: pd.DataFrame, charge_df: pd.DataFrame, discharge_df: pd.DataFrame,
    cycle_num: int, battery_data: Dict[str, Any]
) -> Dict[str, Any]:
    """Calculates direct features (Capacity, Energy, Times, etc.)."""
    features: Dict[str, Any] = {}

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
    features['Coulombic_Efficiency'] = (q_dis / q_chg) if q_chg > 1e-6 else 0.0

    if not charge_df.empty and not discharge_df.empty:
        features['Rest_Time(s)'] = max(0, discharge_df['Time(s)'].iloc[0] - charge_df['Time(s)'].iloc[-1])
    else:
        features['Rest_Time(s)'] = 0.0

    features['charge_c_rate'] = battery_data['charge_protocol'][0]['rate_in_C']
    features['discharge_c_rate'] = battery_data['discharge_protocol'][0]['rate_in_C']

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

    # Charge Phase
    v_upper_limit = battery_data['max_voltage_limit_in_V']
    if not charge_df.empty:
        features['ICHV(V)'] = charge_df['Voltage(V)'].iloc[0]
        features['UVP_time(s)'] = charge_df['Time(s)'].iloc[-1]
        features['UVP(V)'] = v_upper_limit
        if charge_df['Voltage(V)'].max() >= v_upper_limit - 0.01:
            t_at_v_limit = charge_df[charge_df['Voltage(V)'] >= v_upper_limit - 0.01]['Time(s)'].iloc[0]
            features['TCCC(s)'] = t_at_v_limit - charge_df['Time(s)'].iloc[0]
            features['TCVC(s)'] = charge_df['Time(s)'].iloc[-1] - t_at_v_limit
        else:
            features['TCCC(s)'] = charge_df['Time(s)'].iloc[-1] - charge_df['Time(s)'].iloc[0]
            features['TCVC(s)'] = 0.0
        features.update(_calculate_enhanced_thermal_features(charge_df, 'charge'))
    else:
        features.update({'ICHV(V)': 0, 'UVP_time(s)': 0, 'UVP(V)': v_upper_limit, 'TCCC(s)': 0, 'TCVC(s)': 0})
        features.update(_calculate_enhanced_thermal_features(pd.DataFrame(), 'charge'))

    # Discharge Phase
    if not discharge_df.empty:
        features['IDV(V)'] = discharge_df['Voltage(V)'].iloc[0]
        features['LVP_time(s)'] = discharge_df['Time(s)'].iloc[-1]
        features['LVP(V)'] = battery_data['min_voltage_limit_in_V']
        features['var_I_discharge'] = discharge_df['Current(A)'].var()
        features['var_V_discharge'] = discharge_df['Voltage(V)'].var()
        features['median_V_discharge(V)'] = discharge_df['Voltage(V)'].median()
        features['total_discharge_time(s)'] = discharge_df['Time(s)'].iloc[-1] - discharge_df['Time(s)'].iloc[0]
        features.update(_calculate_enhanced_thermal_features(discharge_df, 'discharge'))
    else:
        features.update({
            'IDV(V)': 0, 'LVP_time(s)': 0, 'LVP(V)': battery_data['min_voltage_limit_in_V'],
            'var_I_discharge': 0, 'var_V_discharge': 0, 'median_V_discharge(V)': 0, 'total_discharge_time(s)': 0
        })
        features.update(_calculate_enhanced_thermal_features(pd.DataFrame(), 'discharge'))

    return features


def _calculate_advanced_features(
    cycle_df: pd.DataFrame, charge_df: pd.DataFrame, discharge_df: pd.DataFrame,
    features: Dict[str, Any]
) -> Dict[str, Any]:
    """Calculates advanced features (IR, RCV, Skew, CV Tau)."""
    adv_features: Dict[str, Any] = {}

    # Internal Resistance
    if not charge_df.empty and not discharge_df.empty:
        v_dis_start = discharge_df['Voltage(V)'].iloc[0]
        i_dis_start = abs(discharge_df['Current(A)'].iloc[0])
        pre_dis_df = cycle_df[cycle_df['Time(s)'] < discharge_df['Time(s)'].iloc[0]]
        v_ocv = pre_dis_df['Voltage(V)'].iloc[-1] if not pre_dis_df.empty else charge_df['Voltage(V)'].iloc[-1]
        adv_features['Internal_Resistance(Ohm)'] = ((v_ocv - v_dis_start) / i_dis_start) if i_dis_start > 1e-6 else 0.0
    else:
        adv_features['Internal_Resistance(Ohm)'] = 0.0

    # RCV, Skewness
    adv_features['RCV(V)'] = (features.get('TCCC', 0) / features['TCVC(s)']) if features.get('TCVC', 0) > 1e-6 else 0.0
    adv_features['skew_V_discharge'] = skew(discharge_df['Voltage(V)']) if not discharge_df.empty else 0.0

    # CV Tau
    tcvc = features.get('TCVC', 0)
    if not charge_df.empty and tcvc > 10.0:
        cv_mask = charge_df['Voltage(V)'] >= (features.get('UVP', 4.2) - 0.02)
        cv_df = charge_df[cv_mask]
        if len(cv_df) > 10 and cv_df['Current(A)'].max() > 0.001:
            adv_features['CV_Current_Tau'] = fit_cv_decay(cv_df['Time(s)'].values, cv_df['Current(A)'].values)
        else: adv_features['CV_Current_Tau'] = 0.0
    else: adv_features['CV_Current_Tau'] = 0.0

    return adv_features


def _calculate_anchor_features(
    charge_df: pd.DataFrame, discharge_df: pd.DataFrame,
    charge_slope_intervals: List[Tuple[float, float]],
    discharge_slope_intervals: List[Tuple[float, float]],
    tevi_intervals: List[Tuple[float, float]],
    tevd_intervals: List[Tuple[float, float]]
) -> Dict[str, float]:
    """Calculates anchor point features (slopes, time-at-voltage)."""
    anchor_features: Dict[str, float] = {}

    def get_voltage_at_relative_time(df: pd.DataFrame, relative_time: float) -> Optional[float]:
        if df.empty: return None
        abs_time = df['Time(s)'].iloc[0] + relative_time
        idx = np.searchsorted(df['Time(s)'].values, abs_time, side='left')
        idx = min(idx, len(df) - 1)
        return df['Voltage(V)'].iloc[idx]

    # Slopes
    c_dur = (charge_df['Time(s)'].iloc[-1] - charge_df['Time(s)'].iloc[0]) if not charge_df.empty else 0
    for i, (ps, pe) in enumerate(charge_slope_intervals):
        dt = c_dur * (pe - ps)
        if dt > 1e-6:
            vs, ve = get_voltage_at_relative_time(charge_df, c_dur*ps), get_voltage_at_relative_time(charge_df, c_dur*pe)
            anchor_features[f'charge_slope_{i+1}'] = (ve - vs) / dt if vs and ve else 0.0
        else: anchor_features[f'charge_slope_{i+1}'] = 0.0

    d_dur = (discharge_df['Time(s)'].iloc[-1] - discharge_df['Time(s)'].iloc[0]) if not discharge_df.empty else 0
    for i, (ps, pe) in enumerate(discharge_slope_intervals):
        dt = d_dur * (pe - ps)
        if dt > 1e-6:
            vs, ve = get_voltage_at_relative_time(discharge_df, d_dur*ps), get_voltage_at_relative_time(discharge_df, d_dur*pe)
            anchor_features[f'discharge_slope_{i+1}'] = (ve - vs) / dt if vs and ve else 0.0
        else: anchor_features[f'discharge_slope_{i+1}'] = 0.0

    # TEVI/TEVD
    def get_time_for_voltage(df: pd.DataFrame, v: float, direction: str) -> Optional[float]:
        if df.empty: return None
        mask = df['Voltage(V)'] >= v if direction == 'charge' else df['Voltage(V)'] <= v
        return df.loc[mask.idxmax(), 'Time(s)'] if mask.any() else None

    for i, (vs, ve) in enumerate(tevi_intervals):
        ts, te = get_time_for_voltage(charge_df, vs, 'charge'), get_time_for_voltage(charge_df, ve, 'charge')
        anchor_features[f'TEVI_{i+1}'] = (te - ts) if ts and te and te > ts else 0.0

    for i, (vs, ve) in enumerate(tevd_intervals):
        ts, te = get_time_for_voltage(discharge_df, vs, 'discharge'), get_time_for_voltage(discharge_df, ve, 'discharge')
        anchor_features[f'TEVD_{i+1}'] = (te - ts) if ts and te and te > ts else 0.0

    return anchor_features


def _preprocess_discharge_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses discharge data to handle glitches and interruptions.
    Logic:
    1. Detects gaps in the index (where current >= -0.01A was filtered out).
    2. Small gaps (< 10s): Stitches the voltage curve by removing the relaxation recovery (voltage rebound).
    3. Large gaps (>= 10s): Truncates the data, keeping only the longest continuous segment to avoid capacity integration errors.
    """
    if df.empty:
        return df

    # Detect index jumps
    # diff > 1 means there was a gap in the original cycle_df (filtered out data)
    idx_series = df.index.to_series()
    gaps = idx_series.diff() > 1

    if not gaps.any():
        return df

    # Find the locations of gaps
    gap_indices = df.index[gaps]

    # We will reconstruct the dataframe
    processed_df = df.copy()

    # To handle multiple gaps correctly, we process them sequentially and maintain a cumulative voltage offset
    cum_voltage_offset = 0.0

    # Identify segments
    # A segment is defined by [start_idx, end_idx]
    # We can split the dataframe based on gap locations

    # Get integer locations (iloc) of the gaps
    # gap_indices contains the Index (label) of the start of the NEW segment
    # We need integer positions
    gap_ilocs = [df.index.get_loc(idx) for idx in gap_indices]

    # Add 0 as start and len(df) as end
    boundaries = [0] + gap_ilocs + [len(df)]

    segments = []
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i+1]
        segments.append(df.iloc[start:end].copy())

    # Now analyze gaps between segments
    final_segments = []

    # Start with the first segment
    current_segment = segments[0]
    final_segments.append(current_segment)

    # Track if we have encountered a "fatal" gap (large gap) that forces a reset
    # Actually, if we have a large gap, we should probably pick the longest segment group
    # But for now, let's try to stitch if possible, and break if not.

    # Strategy:
    # We form "Chains" of stitched segments.
    # If a gap is large, we break the chain and start a new one.
    # Finally, we return the longest chain.

    chains = []
    current_chain = [segments[0]]

    for i in range(len(segments) - 1):
        # Gap is between segments[i] (pre) and segments[i+1] (post)
        pre_seg = segments[i]
        post_seg = segments[i+1]

        t_pre = pre_seg['Time(s)'].iloc[-1]
        t_post = post_seg['Time(s)'].iloc[0]
        dt = t_post - t_pre

        if dt < 10.0:
            # Small gap: Stitch
            # Calculate voltage rebound
            v_pre = pre_seg['Voltage(V)'].iloc[-1]
            v_post = post_seg['Voltage(V)'].iloc[0]
            dv = v_post - v_pre

            # If voltage rebounded (increased) during discharge, it's a glitch
            if dv > 0:
                # We need to shift the POST segment down by dv to match the PRE segment
                # But wait, we need to shift EVERYTHING after this.
                # So we apply an offset to the post segment (and it will accumulate)

                # However, since we are building a chain, we can just fix the current post_seg relative to pre_seg
                # But post_seg might be pre_seg for the next iteration.
                # So we modify post_seg in place.
                post_seg['Voltage(V)'] -= dv

            current_chain.append(post_seg)
        else:
            # Large gap: Break chain
            chains.append(current_chain)
            current_chain = [post_seg]

    chains.append(current_chain)

    # Find the longest chain (by total data points)
    best_chain = max(chains, key=lambda c: sum(len(s) for s in c))

    # Concatenate the best chain
    result_df = pd.concat(best_chain)

    if len(chains) > 1:
        # Warn or log if needed, but for now just return the best chain
        pass

    return result_df


def extract_features_for_cycle(
    cycle_data: Dict[str, Any], battery_data: Dict[str, Any],
    charge_slopes: List[Tuple], discharge_slopes: List[Tuple],
    tevi_ints: List[Tuple], tevd_ints: List[Tuple],
    output_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """Main orchestrator for a single cycle."""
    cycle_df = pd.DataFrame({
        'Time(s)': cycle_data['time_in_s'], 'Current(A)': cycle_data['current_in_A'],
        'Voltage(V)': cycle_data['voltage_in_V'],
        'Charge_Capacity(Ah)': cycle_data['charge_capacity_in_Ah'],
        'Discharge_Capacity(Ah)': cycle_data['discharge_capacity_in_Ah'],
        'Temperature(C)': cycle_data.get('temperature_in_C')
    })
    cycle_num = cycle_data['cycle_number']

    # Phase splitting
    charge_df = cycle_df[cycle_df['Current(A)'] > 0.01].copy()
    discharge_df = cycle_df[cycle_df['Current(A)'] < -0.01].copy()

    # [NEW] Preprocess discharge data to fix glitches (voltage rebound stitching)
    if not discharge_df.empty:
        discharge_df = _preprocess_discharge_data(discharge_df)

    # Discharge cutoff
    if not discharge_df.empty:
        v_lower = battery_data['min_voltage_limit_in_V']
        cutoff_idx = discharge_df.index[discharge_df['Voltage(V)'] <= v_lower]
        if not cutoff_idx.empty:
            discharge_df = discharge_df.loc[:cutoff_idx[0]]

    direct_feats = _calculate_direct_features(cycle_df, charge_df, discharge_df, cycle_num, battery_data)

    # [MODIFIED] Use shared tool for IC/DV with config
    nominal_cap = battery_data.get('nominal_capacity_in_Ah', 2.0)
    ncm_config = {
        'peak_mode': 2,
        'nominal_capacity': nominal_cap,
        'window_length_ic': 21,
        'window_length_dv': 31,
        'peak_height_ic': 0.01,
        # 'peak_height_dv': 0.01,
        'voltage_range_ic': (3.1, 3.7),
        # 'voltage_range_dv': (3.3, 4.2),
        'prominence_ic': 0.02,
        # 'prominence_dv': 0.02,
        'ic_step_size': 0.002,
        'dv_step_size': nominal_cap * 0.005,
        'search_window_dvv': 0.1,
        'search_window_dvp': 0.1,
        'initial_capacity_cut_fraction': 0.02,
        'icv_search_offset_lower': 0.02,
        'icv_search_offset_upper': 0.1,
        'ic_area_config': {
            'method': 'fixed_width',
            'width_v': 0.05
        }
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

    # [Task Fix] Disable ICP_FWHM and Zero out DVP features due to high deviation
    if 'ICP_FWHM' in derivative_feats:
        derivative_feats['ICP_FWHM'] = 0.0

    for k in list(derivative_feats.keys()):
        if 'DVP' in k:
            derivative_feats[k] = 0.0

    advanced_feats = _calculate_advanced_features(cycle_df, charge_df, discharge_df, direct_feats)
    anchor_feats = _calculate_anchor_features(charge_df, discharge_df, charge_slopes, discharge_slopes, tevi_ints, tevd_ints)

    return {**direct_feats, **derivative_feats, **advanced_feats, **anchor_feats}


def process_battery(
    file_path: Path, output_dir: Path, charge_slopes: List[Tuple],
    discharge_slopes: List[Tuple], tevi_ints: List[Tuple], tevd_ints: List[Tuple],
    num_cycles: Optional[int] = None
):
    try:
        with open(file_path, 'rb') as f:
            battery_data = pickle.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return

    cycles = battery_data.get('cycle_data', [])
    if num_cycles: cycles = cycles[:num_cycles]
    cell_id = battery_data.get('cell_id', file_path.stem)
    all_feats = []

    for c_data in tqdm(cycles, desc=f"Processing {cell_id}", leave=False):
        if not c_data.get('time_in_s'): continue
        try:
            feats = extract_features_for_cycle(c_data, battery_data, charge_slopes, discharge_slopes, tevi_ints, tevd_ints, output_dir=output_dir)
            all_feats.append(feats)
        except Exception:
            continue

    if not all_feats:
        print(f"No features for {cell_id}")
        return

    df_out = pd.DataFrame(all_feats)
    out_path = output_dir / f"{cell_id}.csv"
    df_out.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


def main():
    processed_data_dir = Path('F:/datasets/battery/UL_PUR')
    output_dir = project_root / 'results' / 'UL_PUR'
    output_dir.mkdir(parents=True, exist_ok=True)

    charge_slopes = [(0.10, 0.30), (0.10, 0.50), (0.10, 0.80)]
    discharge_slopes = [(0.10, 0.30), (0.10, 0.50), (0.10, 0.80)]
    tevi_ints = [(3.5, 3.7), (3.7, 3.9), (3.9, 4.1)]
    tevd_ints = [(4.1, 3.9), (3.9, 3.7), (3.7, 3.5)]
    num_cycles = 100  # Set to None to process all cycles

    if not processed_data_dir.exists():
        print(f"Error: Dir not found '{processed_data_dir}'")
        return

    pkl_files = list(processed_data_dir.glob('*.pkl'))
    if not pkl_files:
        print(f"Error: No .pkl files in '{processed_data_dir}'")
        return

    for pkl_file in pkl_files:
        process_battery(pkl_file, output_dir, charge_slopes, discharge_slopes, tevi_ints, tevd_ints, num_cycles)

if __name__ == '__main__':
    main()

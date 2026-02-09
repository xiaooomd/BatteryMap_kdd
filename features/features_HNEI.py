import pickle
import warnings
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# --- Configuration Constants ---
# HNEI Dataset Specifics (Generic defaults, usually overwritten by file metadata if avail)
BATTERY_NOMINAL_CAPACITY = 2.8  # Ah (Typical for HNEI LG 18650)
CHARGE_C_RATE = 2.0
DISCHARGE_C_RATE = 1.0
VOLTAGE_UPPER_LIMIT = 4.3       # V
VOLTAGE_LOWER_LIMIT = 3.0       # V


def _calculate_direct_features(
    cycle_df: pd.DataFrame,
    charge_df: pd.DataFrame,
    discharge_df: pd.DataFrame,
    cycle_num: int,
    cell_id: str
) -> Dict[str, Any]:
    """Calculates basic features (Capacity, Energy, CV Dynamics).

    Computes standard battery metrics including capacity, energy, efficiency,
    and voltage/current dynamics. Modifies time features to be relative
    to the start of their respective phases rather than absolute timestamps.

    Args:
        cycle_df: Full cycle dataframe.
        charge_df: Charge phase dataframe.
        discharge_df: Discharge phase dataframe.
        cycle_num: Current cycle number.
        cell_id: Identifier for the cell.

    Returns:
        Dict[str, Any]: Basic calculated features with relative timing.
    """
    features = {}

    # --- A. Basic Capacity Features ---
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

    # Coulombic Efficiency
    if chg_cap > 0:
        features['Coulombic_Efficiency'] = dis_cap / chg_cap
    else:
        features['Coulombic_Efficiency'] = 0.0

    # --- B. Energy & Efficiency (Integration) ---
    # Charge Energy (Wh)
    if not charge_df.empty:
        p_charge = charge_df['Voltage(V)'] * charge_df['Current(A)'].abs()
        e_charge_j = trapezoid(y=p_charge, x=charge_df['Time(s)'])
        features['Charge_Energy(Wh)'] = e_charge_j / 3600.0
    else:
        features['Charge_Energy(Wh)'] = 0.0

    # Discharge Energy (Wh)
    if not discharge_df.empty:
        p_discharge = discharge_df['Voltage(V)'] * discharge_df['Current(A)'].abs()
        e_discharge_j = trapezoid(y=p_discharge, x=discharge_df['Time(s)'])
        features['Discharge_Energy(Wh)'] = e_discharge_j / 3600.0
    else:
        features['Discharge_Energy(Wh)'] = 0.0

    # Energy Efficiency
    if features['Charge_Energy(Wh)'] > 0:
        features['Energy_Efficiency'] = (
            features['Discharge_Energy(Wh)'] / features['Charge_Energy(Wh)']
        )
    else:
        features['Energy_Efficiency'] = 0.0

    features['charge_c_rate'] = CHARGE_C_RATE
    features['discharge_c_rate'] = DISCHARGE_C_RATE

    # --- C. Charging Phase & CV Dynamics ---
    features['CV_Current_Tau'] = 0.0

    if not charge_df.empty:
        features['ICHV(V)'] = charge_df['Voltage(V)'].iloc[0]

        # [MODIFIED]: Calculated as relative time (Duration from start of charge)
        t_start_charge = charge_df['Time(s)'].iloc[0]
        features['UVP_time(s)'] = charge_df['Time(s)'].iloc[-1] - t_start_charge

        features['UVP(V)'] = VOLTAGE_UPPER_LIMIT

        # CC-CV Split logic
        # Use tolerance for float comparison
        mask_cv = charge_df['Voltage(V)'] >= (VOLTAGE_UPPER_LIMIT - 0.02)

        if mask_cv.any():
            cv_df = charge_df[mask_cv]
            time_at_v_limit = cv_df['Time(s)'].iloc[0]

            # Relative times for CC and CV phases
            features['TCCC(s)'] = time_at_v_limit - t_start_charge
            features['TCVC(s)'] = charge_df['Time(s)'].iloc[-1] - time_at_v_limit

            # CV Tau Fitting
            cv_current = cv_df['Current(A)'].values
            cv_time = cv_df['Time(s)'].values

            # Simple filter for valid fitting data
            valid_mask = cv_current > 0.001
            if np.sum(valid_mask) > 15:
                features['CV_Current_Tau'] = fit_cv_decay(
                    cv_time[valid_mask],
                    cv_current[valid_mask]
                )
        else:
            # Pure CC charge
            features['TCCC(s)'] = features['UVP_time(s)'] # Same as total duration
            features['TCVC(s)'] = 0
    else:
        features.update({
            'ICHV(V)': 0, 'UVP_time(s)': 0, 'UVP(V)': 0, 'TCCC(s)': 0, 'TCVC(s)': 0
        })

    # --- D. Discharging Phase Features ---
    if not discharge_df.empty:
        features['IDV(V)'] = discharge_df['Voltage(V)'].iloc[0]

        # [MODIFIED]: Calculated as relative time (Duration from start of discharge)
        t_start_discharge = discharge_df['Time(s)'].iloc[0]
        features['LVP_time(s)'] = discharge_df['Time(s)'].iloc[-1] - t_start_discharge

        features['LVP(V)'] = VOLTAGE_LOWER_LIMIT
        features['var_I_discharge'] = discharge_df['Current(A)'].var()
        features['var_V_discharge'] = discharge_df['Voltage(V)'].var()
        features['median_V_discharge(V)'] = discharge_df['Voltage(V)'].median()

        # Total discharge time (redundant with LVP_time usually, but kept for clarity)
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
    features: Dict[str, Any]
) -> Dict[str, Any]:
    """Computes advanced derived features like Resistance and RCV."""
    adv_features = {}

    # Internal Resistance
    # Approximation: (V_end_charge - V_start_discharge) / I_discharge
    if not charge_df.empty and not discharge_df.empty:
        v_charge_end = charge_df['Voltage(V)'].iloc[-1]
        v_discharge_start = discharge_df['Voltage(V)'].iloc[0]
        i_discharge_start = abs(discharge_df['Current(A)'].iloc[0])

        if i_discharge_start > 1e-3:
            adv_features['Internal_Resistance(Ohm)'] = (
                (v_charge_end - v_discharge_start) / i_discharge_start
            )
        else:
            adv_features['Internal_Resistance(Ohm)'] = 0.0
    else:
        adv_features['Internal_Resistance(Ohm)'] = 0.0

    # RCV Ratio
    if features.get('TCVC', 0) > 0:
        adv_features['RCV(V)'] = features.get('TCCC', 0) / features['TCVC(s)']
    else:
        adv_features['RCV(V)'] = 0.0

    # Skewness of Discharge Curve
    if not discharge_df.empty:
        adv_features['skew_V_discharge'] = skew(discharge_df['Voltage(V)'])
    else:
        adv_features['skew_V_discharge'] = 0.0

    return adv_features


def _calculate_anchor_features(
    charge_df: pd.DataFrame,
    discharge_df: pd.DataFrame,
    charge_slope_intervals: List[Tuple[float, float]],
    discharge_slope_intervals: List[Tuple[float, float]],
    tevi_intervals: List[Tuple[float, float]],
    tevd_intervals: List[Tuple[float, float]]
) -> Dict[str, Any]:
    """Calculates anchor interval features (Slope, TEVI, TEVD)."""
    anchor_features = {}

    def get_voltage_at_relative_time(
        df: pd.DataFrame,
        relative_time: float
    ) -> Optional[float]:
        if df.empty:
            return None
        start_time = df['Time(s)'].iloc[0]
        target_absolute_time = start_time + relative_time
        time_array = df['Time(s)'].values
        idx = np.searchsorted(time_array, target_absolute_time)

        if idx == 0:
            closest_df_index = df.index[0]
        elif idx == len(time_array):
            closest_df_index = df.index[-1]
        else:
            # Check which neighbor is closer
            diff_prev = target_absolute_time - time_array[idx - 1]
            diff_curr = time_array[idx] - target_absolute_time
            if diff_prev < diff_curr:
                closest_df_index = df.index[idx - 1]
            else:
                closest_df_index = df.index[idx]
        return df.loc[closest_df_index, 'Voltage(V)']

    # --- Charge Slopes ---
    if not charge_df.empty:
        c_dur = charge_df['Time(s)'].iloc[-1] - charge_df['Time(s)'].iloc[0]
        for i, (p_start, p_end) in enumerate(charge_slope_intervals):
            if c_dur > 1e-5:
                v_s = get_voltage_at_relative_time(charge_df, c_dur * p_start)
                v_e = get_voltage_at_relative_time(charge_df, c_dur * p_end)
                dt = c_dur * (p_end - p_start)
                if v_s is not None and v_e is not None and dt > 1e-9:
                    anchor_features[f'charge_slope_{i + 1}'] = (v_e - v_s) / dt
                else:
                    anchor_features[f'charge_slope_{i + 1}'] = 0
            else:
                anchor_features[f'charge_slope_{i + 1}'] = 0
    else:
        for i in range(len(charge_slope_intervals)):
            anchor_features[f'charge_slope_{i + 1}'] = 0

    # --- Discharge Slopes ---
    if not discharge_df.empty:
        d_dur = (
            discharge_df['Time(s)'].iloc[-1] - discharge_df['Time(s)'].iloc[0]
        )
        for i, (p_start, p_end) in enumerate(discharge_slope_intervals):
            if d_dur > 1e-5:
                v_s = get_voltage_at_relative_time(discharge_df, d_dur * p_start)
                v_e = get_voltage_at_relative_time(discharge_df, d_dur * p_end)
                dt = d_dur * (p_end - p_start)
                if v_s is not None and v_e is not None and dt > 1e-9:
                    anchor_features[f'discharge_slope_{i + 1}'] = (v_e - v_s) / dt
                else:
                    anchor_features[f'discharge_slope_{i + 1}'] = 0
            else:
                anchor_features[f'discharge_slope_{i + 1}'] = 0
    else:
        for i in range(len(discharge_slope_intervals)):
            anchor_features[f'discharge_slope_{i + 1}'] = 0

    # --- TEVI / TEVD ---
    def get_time_for_voltage(
        df: pd.DataFrame,
        voltage: float,
        direction: str
    ) -> Optional[float]:
        if df.empty:
            return None
        if direction == 'charge':
            target_rows = df[df['Voltage(V)'] >= voltage]
        else:
            target_rows = df[df['Voltage(V)'] <= voltage]
        return target_rows['Time(s)'].iloc[0] if not target_rows.empty else None

    for i, (v_start, v_end) in enumerate(tevi_intervals):
        t_start = get_time_for_voltage(charge_df, v_start, 'charge')
        t_end = get_time_for_voltage(charge_df, v_end, 'charge')
        if t_start is not None and t_end is not None and t_end > t_start:
            anchor_features[f'TEVI_{i + 1}'] = t_end - t_start
        else:
            anchor_features[f'TEVI_{i + 1}'] = 0

    for i, (v_start, v_end) in enumerate(tevd_intervals):
        t_start = get_time_for_voltage(discharge_df, v_start, 'discharge')
        t_end = get_time_for_voltage(discharge_df, v_end, 'discharge')
        if t_start is not None and t_end is not None and t_end > t_start:
            anchor_features[f'TEVD_{i + 1}'] = t_end - t_start
        else:
            anchor_features[f'TEVD_{i + 1}'] = 0

    return anchor_features


def _calculate_personalized_features(
    cycle_df: pd.DataFrame,
    cell_id: str
) -> Dict[str, Any]:
    return {}


def extract_features_for_cycle(
    cycle_data: Dict[str, Any],
    cell_id: str,
    charge_slope_intervals: List[Tuple[float, float]],
    discharge_slope_intervals: List[Tuple[float, float]],
    tevi_intervals: List[Tuple[float, float]],
    tevd_intervals: List[Tuple[float, float]],
    output_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """Main feature extraction for a single cycle."""

    cycle_df = pd.DataFrame({
        'Time(s)': cycle_data['time_in_s'],
        'Current(A)': cycle_data['current_in_A'],
        'Voltage(V)': cycle_data['voltage_in_V'],
        'Charge_Capacity(Ah)': cycle_data['charge_capacity_in_Ah'],
        'Discharge_Capacity(Ah)': cycle_data['discharge_capacity_in_Ah']
    })
    cycle_num = cycle_data['cycle_number']

    # Phase Separation
    charge_df = cycle_df[cycle_df['Current(A)'] > 0].copy()
    discharge_df = cycle_df[cycle_df['Current(A)'] < 0].copy()

    # Cutoff Logic (Clean tail of discharge)
    if not discharge_df.empty:
        cutoff_indices = discharge_df.index[
            discharge_df['Voltage(V)'] <= VOLTAGE_LOWER_LIMIT
        ]
        if not cutoff_indices.empty:
            discharge_df = discharge_df.loc[:cutoff_indices[0]]

    # Feature Extraction
    direct_features = _calculate_direct_features(
        cycle_df, charge_df, discharge_df, cycle_num, cell_id
    )
    # [MODIFIED] Use shared tool for IC/DV
    ncm_config = {
        'peak_mode': 1,
        'nominal_capacity': BATTERY_NOMINAL_CAPACITY,
        'window_length_ic': 31,
        'window_length_dv': 21,
        'peak_height_ic': 0.01,
        # 'peak_height_dv': 0.01,
        'voltage_range_ic': (3.2, 4.2),
        # 'voltage_range_dv': (3.4, 4.2),
        'prominence_ic': 0.01,
        # 'prominence_dv': 0.005,
        'ic_step_size': 0.002,
        'dv_step_size': BATTERY_NOMINAL_CAPACITY * 0.005,
        'search_window_dvv': 0.2,
        'search_window_dvp': 0.2,
        'initial_capacity_cut_fraction': 0.02,
        'icv_search_offset_lower': 0.05,
        'icv_search_offset_upper': 0.5
    }

    plot_params = None
    if output_dir:
        plot_params = {
            'cell_id': cell_id,
            'cycle_num': cycle_num,
            'output_dir': output_dir
        }

    derivative_features = extract_ic_dv_features(
        discharge_df,
        config=ncm_config,
        plot_params=plot_params
    )
    advanced_features = _calculate_advanced_features(
        charge_df, discharge_df, direct_features
    )
    anchor_features = _calculate_anchor_features(
        charge_df, discharge_df,
        charge_slope_intervals, discharge_slope_intervals,
        tevi_intervals, tevd_intervals
    )
    personalized_features = _calculate_personalized_features(cycle_df, cell_id)

    return {
        **direct_features,
        **derivative_features,
        **advanced_features,
        **anchor_features,
        **personalized_features
    }


def process_battery(
    file_path: Path,
    output_dir: Path,
    charge_slope_intervals: List[Tuple[float, float]],
    discharge_slope_intervals: List[Tuple[float, float]],
    tevi_intervals: List[Tuple[float, float]],
    tevd_intervals: List[Tuple[float, float]],
    num_cycles: Optional[int] = None
):
    try:
        with open(file_path, 'rb') as f:
            data_dict = pickle.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return

    battery_data = data_dict

    if 'cycle_data' in battery_data and battery_data['cycle_data'] is not None:
        cycles_to_process = list(battery_data['cycle_data'])
    else:
        print(f"No cycle data found for {battery_data.get('cell_id', 'Unknown')}")
        return

    if num_cycles is not None and num_cycles > 0:
        cycles_to_process = cycles_to_process[:num_cycles]

    all_cycle_features = []
    cell_id = battery_data.get('cell_id', file_path.stem)

    for cycle_data in tqdm(cycles_to_process, desc=f"Processing {cell_id}"):
        if not cycle_data.get('time_in_s'):
            continue

        try:
            features = extract_features_for_cycle(
                cycle_data, cell_id,
                charge_slope_intervals, discharge_slope_intervals,
                tevi_intervals, tevd_intervals,
                output_dir=output_dir
            )
            all_cycle_features.append(features)
        except Exception:
            # Silently skip bad cycles to maintain pipeline flow
            continue

    if not all_cycle_features:
        print(f"Warning: No features extracted for {cell_id}")
        return

    features_df = pd.DataFrame(all_cycle_features)

    # Final Column Ordering
    base_cols = [
        'Cycle_Number',
        'Discharge_Capacity(Ah)', 'Charge_Capacity(Ah)',
        'Discharge_Energy(Wh)', 'Charge_Energy(Wh)',
        'Coulombic_Efficiency', 'Energy_Efficiency',
        'charge_c_rate', 'discharge_c_rate',
        'ICHV(V)', 'UVP_time(s)', 'TCCC(s)', 'TCVC(s)', 'CV_Current_Tau', 'UVP(V)',
        'IDV(V)', 'LVP_time(s)', 'var_I_discharge', 'var_V_discharge',
        'median_V_discharge(V)', 'total_discharge_time(s)', 'LVP(V)',
        'ICP', 'ICPL_V', 'ICP_FWHM', 'ICP_Area',
        'ICV', 'ICVL_V',
        'DVP', 'DVPL_V', #'DVP_FWHM', 'DVP_Area',
        'DVV', 'DVVL_V',
        'Internal_Resistance(Ohm)', 'RCV(V)', 'skew_V_discharge'
    ]

    existing_base_cols = [c for c in base_cols if c in features_df.columns]
    remaining_cols = [c for c in features_df.columns if c not in existing_base_cols]
    remaining_cols.sort()

    final_cols = existing_base_cols + remaining_cols
    features_df = features_df[final_cols]

    output_file = output_dir / f"{cell_id}.csv"
    features_df.to_csv(output_file, index=False)
    print(f"Features for {cell_id} saved to {output_file}")


def main():
    processed_data_dir = project_root / 'data' / 'HNEI'
    output_dir = project_root / 'results' / 'features' / 'HNEI'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Intervals Setup
    charge_slope_intervals = [(0.10, 0.30), (0.10, 0.50), (0.10, 0.80)]
    discharge_slope_intervals = [(0.10, 0.30), (0.10, 0.50), (0.10, 0.80)]
    tevi_intervals = [(3.5, 3.8), (3.8, 4.1), (4.1, 4.25)]
    tevd_intervals = [(4.2, 3.9), (3.9, 3.6), (3.6, 3.3)]

    num_cycles_to_extract = 100

    if not processed_data_dir.exists():
        print(f"Error: Directory not found at '{processed_data_dir}'.")
        return

    pkl_files = list(processed_data_dir.glob('*.pkl'))
    if not pkl_files:
        print(f"Error: No .pkl files found in '{processed_data_dir}'.")
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

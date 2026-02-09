import pickle
import re
import warnings
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

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
warnings.filterwarnings('ignore', category=np.RankWarning)

# --- Configuration Constants ---
# ISU-ILCC Dataset Specifics
# Rates map: cell_group -> [Charge Rate, Discharge Rate]
CYCLING_RATES = {
    'ISU-ILCC_G1': [0.5, 0.5], 'ISU-ILCC_G2': [0.5, 0.5], 'ISU-ILCC_G3': [0.5, 0.5],
    'ISU-ILCC_G4': [1, 0.5], 'ISU-ILCC_G5': [1, 0.5], 'ISU-ILCC_G6': [2, 0.5],
    'ISU-ILCC_G7': [2, 0.5], 'ISU-ILCC_G8': [2, 0.5], 'ISU-ILCC_G9': [2, 0.5],
    'ISU-ILCC_G10': [2.5, 0.5], 'ISU-ILCC_G12': [3, 0.5], 'ISU-ILCC_G13': [3, 0.5],
    'ISU-ILCC_G14': [3, 0.5], 'ISU-ILCC_G15': [3, 0.5], 'ISU-ILCC_G16': [0.5, 0.5],
    'ISU-ILCC_G17': [1, 0.5], 'ISU-ILCC_G18': [2.5, 0.5], 'ISU-ILCC_G19': [2.5, 0.5],
    'ISU-ILCC_G20': [0.8, 0.5], 'ISU-ILCC_G21': [1.2, 0.5], 'ISU-ILCC_G22': [1.4, 0.5],
    'ISU-ILCC_G23': [1.6, 0.5], 'ISU-ILCC_G24': [1.8, 0.5], 'ISU-ILCC_G25': [1.8, 0.6],
    'ISU-ILCC_G26': [1.4, 2.2], 'ISU-ILCC_G27': [0.6, 2.4], 'ISU-ILCC_G28': [2.4, 1.6],
    'ISU-ILCC_G29': [1.6, 1.8], 'ISU-ILCC_G30': [0.8, 0.8], 'ISU-ILCC_G31': [1.2, 1],
    'ISU-ILCC_G32': [1, 1.4], 'ISU-ILCC_G33': [2, 1.2], 'ISU-ILCC_G34': [2.2, 2],
    'ISU-ILCC_G35': [1.825, 0.5], 'ISU-ILCC_G36': [2.075, 0.5], 'ISU-ILCC_G37': [0.725, 0.5],
    'ISU-ILCC_G38': [1.875, 0.5], 'ISU-ILCC_G39': [1.475, 0.5], 'ISU-ILCC_G40': [1.825, 1.025],
    'ISU-ILCC_G41': [2.075, 1.775], 'ISU-ILCC_G42': [0.725, 2.375], 'ISU-ILCC_G43': [1.875, 2.325],
    'ISU-ILCC_G44': [0.775, 1.275], 'ISU-ILCC_G45': [1.125, 1.725], 'ISU-ILCC_G46': [1.225, 2.025],
    'ISU-ILCC_G47': [2.325, 1.925], 'ISU-ILCC_G48': [2.375, 2.225], 'ISU-ILCC_G49': [0.975, 0.675],
    'ISU-ILCC_G50': [2.425, 1.625], 'ISU-ILCC_G51': [2.275, 1.875], 'ISU-ILCC_G52': [1.425, 0.875],
    'ISU-ILCC_G53': [2.025, 0.825], 'ISU-ILCC_G54': [0.925, 1.125], 'ISU-ILCC_G55': [1.025, 2.475],
    'ISU-ILCC_G56': [2.175, 0.975], 'ISU-ILCC_G57': [1.775, 1.175], 'ISU-ILCC_G58': [2.475, 0.575],
    'ISU-ILCC_G59': [1.325, 1.825], 'ISU-ILCC_G60': [0.675, 1.325], 'ISU-ILCC_G61': [2.125, 1.975],
    'ISU-ILCC_G62': [1.575, 2.425], 'ISU-ILCC_G63': [1.975, 1.675], 'ISU-ILCC_G64': [1.175, 1.425],
}

# Static Metadata
CHARGE_CUTOFF_V = 4.2  # V
DISCHARGE_CUTOFF_V = 3.0  # V

# Type Hints
Interval = Tuple[float, float]
IntervalList = List[Interval]


def _normalize_time_array(times: Union[List[Any], np.ndarray]) -> np.ndarray:
    """Normalizes time array to relative seconds (float)."""
    if times is None:
        return np.array([])

    arr = np.array(times)
    if arr.size == 0:
        return np.array([])

    if np.issubdtype(arr.dtype, np.datetime64):
        ns = arr.astype('datetime64[ns]').astype('int64')
        s = ns.astype('float64') / 1e9
        return s - s[0]

    if arr.dtype == 'O':
        try:
            ts = pd.to_datetime(arr, errors='coerce')
            if not ts.isna().all():
                ns = ts.values.astype('int64')
                s = ns.astype('float64') / 1e9
                return s - s[0]
        except Exception:
            pass

    try:
        arrf = arr.astype('float64')
        if arrf.size > 0 and np.nanmax(np.abs(arrf)) > 1e11:
            arrf = arrf / 1e9
        if arrf.size > 0:
            arrf = arrf - arrf[0]
        return arrf
    except ValueError:
        return np.array([])


def _get_cell_group_key(cell_id: str) -> str:
    """Extracts group key (e.g., 'ISU-ILCC_G57') from cell_id."""
    match = re.search(r'(ISU-ILCC_G\d+)', cell_id)
    if match:
        return match.group(1)
    return cell_id


def _calculate_direct_features(
    cycle_df: pd.DataFrame,
    charge_df: pd.DataFrame,
    discharge_df: pd.DataFrame,
    cycle_num: int,
    cell_id: str
) -> Dict[str, Any]:
    """Calculates basic capacity, energy, and efficiency features."""
    features = {}
    cell_group_key = _get_cell_group_key(cell_id)
    rates = CYCLING_RATES.get(cell_group_key, [np.nan, np.nan])

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

    if chg_cap > 0:
        features['Coulombic_Efficiency'] = dis_cap / chg_cap
    else:
        features['Coulombic_Efficiency'] = 0.0

    features['charge_c_rate'] = rates[0]
    features['discharge_c_rate'] = rates[1]

    # Energy (Wh)
    if not charge_df.empty:
        p_charge = charge_df['Voltage(V)'] * charge_df['Current(A)'].abs()
        e_charge_j = trapezoid(y=p_charge, x=charge_df['Time(s)'])
        features['Charge_Energy(Wh)'] = e_charge_j / 3600.0
    else:
        features['Charge_Energy(Wh)'] = 0.0

    if not discharge_df.empty:
        p_discharge = discharge_df['Voltage(V)'] * discharge_df['Current(A)'].abs()
        e_discharge_j = trapezoid(y=p_discharge, x=discharge_df['Time(s)'])
        features['Discharge_Energy(Wh)'] = e_discharge_j / 3600.0
    else:
        features['Discharge_Energy(Wh)'] = 0.0

    if features['Charge_Energy(Wh)'] > 0:
        features['Energy_Efficiency'] = features['Discharge_Energy(Wh)'] / features['Charge_Energy(Wh)']
    else:
        features['Energy_Efficiency'] = 0.0

    # Charge Dynamics (CV Tau)
    features['CV_Current_Tau'] = 0.0

    if not charge_df.empty:
        features['ICHV(V)'] = charge_df['Voltage(V)'].iloc[0]
        features['UVP_time(s)'] = charge_df['Time(s)'].iloc[-1]
        features['UVP(V)'] = CHARGE_CUTOFF_V

        v_upper_limit = CHARGE_CUTOFF_V
        charge_voltage = charge_df['Voltage(V)']

        # Identify CV phase (using tolerance)
        mask_cv = charge_voltage >= (v_upper_limit - 0.01)

        if mask_cv.any():
            cv_start_idx = mask_cv.idxmax()
            time_at_v_limit = charge_df.loc[cv_start_idx, 'Time(s)']

            features['TCCC(s)'] = time_at_v_limit - charge_df['Time(s)'].iloc[0]
            features['TCVC(s)'] = charge_df['Time(s)'].iloc[-1] - time_at_v_limit

            # CV Fit
            cv_df = charge_df.loc[cv_start_idx:]
            cv_current = cv_df['Current(A)'].values
            cv_time = cv_df['Time(s)'].values

            valid_mask = cv_current > 0.001
            if np.sum(valid_mask) > 15:
                features['CV_Current_Tau'] = fit_cv_decay(
                    cv_time[valid_mask],
                    cv_current[valid_mask]
                )
        else:
            features['TCCC(s)'] = charge_df['Time(s)'].iloc[-1] - charge_df['Time(s)'].iloc[0]
            features['TCVC(s)'] = 0
    else:
        features.update({'ICHV(V)': 0, 'UVP_time(s)': 0, 'UVP(V)': 0, 'TCCC(s)': 0, 'TCVC(s)': 0})

    # Discharge Dynamics
    if not discharge_df.empty:
        features['IDV(V)'] = discharge_df['Voltage(V)'].iloc[0]
        features['LVP_time(s)'] = discharge_df['Time(s)'].iloc[-1]
        features['LVP(V)'] = DISCHARGE_CUTOFF_V
        features['var_I_discharge'] = discharge_df['Current(A)'].var()
        features['var_V_discharge'] = discharge_df['Voltage(V)'].var()
        features['median_V_discharge(V)'] = discharge_df['Voltage(V)'].median()
        features['total_discharge_time(s)'] = (
            discharge_df['Time(s)'].iloc[-1] - discharge_df['Time(s)'].iloc[0]
        )
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
    rest_df: pd.DataFrame,
    features: Dict[str, Any]
) -> Dict[str, Any]:
    """Calculates internal resistance and statistical features."""
    adv_features = {}

    # Internal Resistance
    if not charge_df.empty and not discharge_df.empty:
        v_charge_end = charge_df['Voltage(V)'].iloc[-1]
        v_discharge_start = discharge_df['Voltage(V)'].iloc[0]
        i_discharge_start = abs(discharge_df['Current(A)'].iloc[0])

        if i_discharge_start > 0:
            resistance = (v_charge_end - v_discharge_start) / i_discharge_start
            adv_features['Internal_Resistance(Ohm)'] = max(0.0, resistance)
        else:
            adv_features['Internal_Resistance(Ohm)'] = 0.0
    else:
        adv_features['Internal_Resistance(Ohm)'] = 0.0

    # RCV Ratio
    if features.get('TCVC', 0) > 0:
        adv_features['RCV(V)'] = features.get('TCCC', 0) / features['TCVC(s)']
    else:
        adv_features['RCV(V)'] = 0.0

    # Skewness
    if not discharge_df.empty:
        v_data = discharge_df['Voltage(V)']
        if len(v_data) > 2 and v_data.std() > 1e-6:
            adv_features['skew_V_discharge'] = skew(v_data)
        else:
            adv_features['skew_V_discharge'] = 0.0
    else:
        adv_features['skew_V_discharge'] = 0.0
    return adv_features


def _get_time_by_interpolation(
    df: pd.DataFrame,
    voltage: float,
    direction: str
) -> Optional[float]:
    """Interpolates time at a specific voltage."""
    if df.empty or len(df) < 2:
        return None

    volt_array = df['Voltage(V)'].values
    time_array = df['Time(s)'].values

    try:
        if direction == 'charge':
            unique_volts, idx = np.unique(volt_array, return_index=True)
            unique_times = time_array[idx]
            if voltage < unique_volts[0] or voltage > unique_volts[-1]:
                return None
            return np.interp(voltage, unique_volts, unique_times)
        else:
            unique_volts, idx = np.unique(volt_array, return_index=True)
            unique_times = time_array[idx]
            if voltage < unique_volts[0] or voltage > unique_volts[-1]:
                return None
            return np.interp(voltage, unique_volts, unique_times)
    except Exception:
        return None


def _calculate_anchor_features(
    charge_df: pd.DataFrame,
    discharge_df: pd.DataFrame,
    charge_slope_intervals_pct: IntervalList,
    discharge_slope_intervals_pct: IntervalList,
    tevi_intervals: List[Tuple[float, float]],
    tevd_intervals: List[Tuple[float, float]]
) -> Dict[str, Any]:
    """Calculates anchor interval features (Slope, TEVI, TEVD)."""
    anchor_features = {}

    def get_voltage_at_relative_time(df, relative_time):
        if df.empty: return None
        time_array = df['Time(s)'].values
        volt_array = df['Voltage(V)'].values
        idx = np.searchsorted(time_array, relative_time, side='left')
        if idx == 0: return volt_array[0]
        if idx == len(time_array): return volt_array[-1]

        time_before = time_array[idx - 1]
        time_after = time_array[idx]
        if (relative_time - time_before) < (time_after - relative_time):
            return volt_array[idx - 1]
        else:
            return volt_array[idx]

    # Charge Slopes
    charge_start = charge_df['Time(s)'].iloc[0] if not charge_df.empty else 0
    charge_dur = (
        charge_df['Time(s)'].iloc[-1] - charge_start
        if not charge_df.empty else 0
    )
    for i, (p_start, p_end) in enumerate(charge_slope_intervals_pct):
        if charge_dur > 0:
            t_s = charge_start + (charge_dur * p_start)
            t_e = charge_start + (charge_dur * p_end)
            v_s = get_voltage_at_relative_time(charge_df, t_s)
            v_e = get_voltage_at_relative_time(charge_df, t_e)
            dt = t_e - t_s
            if v_s is not None and v_e is not None and dt > 1e-6:
                anchor_features[f'charge_slope_{i + 1}'] = (v_e - v_s) / dt
            else:
                anchor_features[f'charge_slope_{i + 1}'] = 0
        else:
            anchor_features[f'charge_slope_{i + 1}'] = 0

    # Discharge Slopes
    dis_start = discharge_df['Time(s)'].iloc[0] if not discharge_df.empty else 0
    dis_dur = (
        discharge_df['Time(s)'].iloc[-1] - dis_start
        if not discharge_df.empty else 0
    )
    for i, (p_start, p_end) in enumerate(discharge_slope_intervals_pct):
        if dis_dur > 0:
            t_s = dis_start + (dis_dur * p_start)
            t_e = dis_start + (dis_dur * p_end)
            v_s = get_voltage_at_relative_time(discharge_df, t_s)
            v_e = get_voltage_at_relative_time(discharge_df, t_e)
            dt = t_e - t_s
            if v_s is not None and v_e is not None and dt > 1e-6:
                anchor_features[f'discharge_slope_{i + 1}'] = (v_e - v_s) / dt
            else:
                anchor_features[f'discharge_slope_{i + 1}'] = 0
        else:
            anchor_features[f'discharge_slope_{i + 1}'] = 0

    # TEVI / TEVD
    for i, (v_start, v_end) in enumerate(tevi_intervals):
        t_start = _get_time_by_interpolation(charge_df, v_start, 'charge')
        t_end = _get_time_by_interpolation(charge_df, v_end, 'charge')
        if t_start is not None and t_end is not None:
            anchor_features[f'TEVI_{i + 1}'] = t_end - t_start
        else:
            anchor_features[f'TEVI_{i + 1}'] = 0

    for i, (v_start, v_end) in enumerate(tevd_intervals):
        t_start = _get_time_by_interpolation(discharge_df, v_start, 'discharge')
        t_end = _get_time_by_interpolation(discharge_df, v_end, 'discharge')
        if t_start is not None and t_end is not None:
            anchor_features[f'TEVD_{i + 1}'] = t_end - t_start
        else:
            anchor_features[f'TEVD_{i + 1}'] = 0

    return anchor_features


def _calculate_personalized_features(
    cycle_df: pd.DataFrame,
    cell_id: str
) -> Dict[str, Any]:
    return {}



def _get_voltage_intervals(cell_id: str) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    """
    Returns dynamic TEVI and TEVD intervals based on cell chemistry/voltage range.
    Groups determined by analysis of 10th cycle voltage ranges.
    """
    # Regex Patterns for Groups
    # Group 1: Low Voltage (Charge Start ~3.2V, Discharge End 3.0V) - LFP-like
    # Cells: G5, G16, G20-G24, G35-G39
    pattern_low_v = r'(ISU-ILCC_)?(G5C|G16C|G2[0-4]C|G3[5-9]C)'

    # Group 3: High Voltage (Charge Start ~4.0V, Discharge End ~3.8V)
    # Cells: G1, G2, G4, G6, G7, G10, G12, G13, G18, G25, G33, G34, G40, G49, G50, G54, G56-G59
    pattern_high_v = r'(ISU-ILCC_)?(G1C|G2C|G4C|G6C|G7C|G10C|G12C|G13C|G18C|G25C|G33C|G34C|G40C|G49C|G50C|G54C|G5[6-9]C)'

    if re.search(pattern_low_v, cell_id):
        # Low Voltage Intervals
        tevi = [(3.25, 3.35), (3.35, 3.45), (3.45, 3.55)]
        tevd = [(3.5, 3.4), (3.3, 3.2), (3.2, 3.05)]
    elif re.search(pattern_high_v, cell_id):
        # High Voltage Intervals
        tevi = [(4.0, 4.05), (4.05, 4.1), (4.1, 4.15)]
        tevd = [(4.15, 4.1), (4.0, 3.9), (3.9, 3.8)]
    else:
        # Default / Mid Voltage Intervals (NMC Standard)
        # Adjusted slightly to accommodate 3.7V start
        tevi = [(3.6, 3.75), (3.8, 3.95), (4.0, 4.15)]
        tevd = [(4.1, 3.9), (3.8, 3.6), (3.5, 3.3)]

    return tevi, tevd


def _get_isu_ilcc_config(cell_id: str) -> Dict[str, Any]:
    """
    Generates dynamic configuration for feature extraction based on Cell ID Groups.

    Args:
        cell_id: The cell identifier (e.g., 'ISU-ILCC_G40C3').

    Returns:
        Dict[str, Any]: Configuration dictionary.
    """
    nominal_cap = 2.0  # Default nominal capacity

    # Base Configuration
    config = {
        'peak_mode': 2,
        'nominal_capacity': nominal_cap,
        'window_length_ic': 21,
        'window_length_dv': 21,
        'peak_height_ic': 0.01,
        'voltage_range_ic': (3.85, 4.2),  # Default Main Peak Range
        'prominence_ic': 0.02,
        'ic_step_size': 0.01,
        'dv_step_size': nominal_cap * 0.005,
        'search_window_dvv': 0.1,
        'search_window_dvp': 0.1,
        'initial_capacity_cut_fraction': 0.02,
        'icv_search_offset_lower': 0.02,
        'icv_search_offset_upper': 0.2,
        'ic_area_config': {
            'method': 'fixed_width',
            'width_v': 0.03
        },
        'icv_method': 'first_valley_left',
        'force_icp_fwhm_zero': True,  # [MODIFIED] Disable FWHM calculation
        'aux_peak_config': {
            'voltage_range': (3.75, 3.85),  # Default Aux Peak Range
            'selection': 'max',
            'default_value': 0.0
        }
    }

    # --- Group Match Logic ---
    # Define regex patterns for groups
    # Group A: (3.8, 4.0) | Aux: (3.7, 3.8)
    # G40C3, G41(C3/C4), G42(C1-C3), G47C1, G59C4, G60C2, G61(C2/C4), G62C4, G63(C1/C2), G64(C1-C4)
    pattern_group_a = r'G40C3|G41C[34]|G42C[1-3]|G43C[12]|G46C1|G46C3|G47C1|G48C[1-4]|G51C4|G59C4|G60C2|G61C[24]|G62C4|G63C[12]|G64C[1-4]'

    # Group B: (3.7, 3.9) | Aux: (3.6, 3.7)
    # G27C1
    pattern_group_b = r'G27C1'

    # Group C: (3.6, 3.7) | Aux: (3.5, 3.6)
    # G27C2, G27C3, G27C4
    pattern_group_c = r'G27C[2-4]'

    # Group D: (3.75, 3.9) | Aux: (3.65, 3.75)
    # G34C2, G51C2, G55(C1-C4), G62C3
    pattern_group_d = r'G34C[2-4]|G43C[34]|G51C2|G55C[1-4]|G62C1|G62C2|G62C3'

    # G57 Series: Disable ICV
    pattern_g57 = r'G57'

    # G49 Series: Force Zero for ICP and ICV
    pattern_g49 = r'G49C3'

    # Special Case: G40C3 window length
    pattern_g40c3 = r'G40C3'

    # Apply Configurations
    if re.search(pattern_group_a, cell_id):
        config['voltage_range_ic'] = (3.8, 4.0)
        config['aux_peak_config']['voltage_range'] = (3.7, 3.8)

    elif re.search(pattern_group_b, cell_id):
        config['voltage_range_ic'] = (3.7, 3.9)
        config['aux_peak_config']['voltage_range'] = (3.6, 3.7)

    elif re.search(pattern_group_c, cell_id):
        config['voltage_range_ic'] = (3.75, 3.9)
        config['aux_peak_config']['voltage_range'] = (3.5, 3.75)

    elif re.search(pattern_group_d, cell_id):
        config['voltage_range_ic'] = (3.75, 3.9)
        config['aux_peak_config']['voltage_range'] = (3.65, 3.75)

    # G57 Specifics
    if re.search(pattern_g57, cell_id):
        config['force_icv_zero'] = True

    # G49 Specifics
    # if re.search(pattern_g49, cell_id):
    #    config['force_icp_zero'] = True
    #    config['force_icv_zero'] = True

    # G40C3 Specifics
    if re.search(pattern_g40c3, cell_id):
        config['window_length_ic'] = 51
        config['window_length_dv'] = 51

    return config



def extract_features_for_cycle(
    cycle_data: Dict[str, Any],
    cell_id: str,
    charge_slope_intervals_pct: IntervalList,
    discharge_slope_intervals_pct: IntervalList,
    output_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """Main feature extraction for a single cycle."""

    # 1. Prepare Data
    relative_seconds = _normalize_time_array(cycle_data['time_in_s'])
    if relative_seconds.size == 0:
        return {}

    cycle_df = pd.DataFrame({
        'Time(s)': relative_seconds,
        'Current(A)': cycle_data['current_in_A'],
        'Voltage(V)': cycle_data['voltage_in_V'],
        'Charge_Capacity(Ah)': cycle_data['charge_capacity_in_Ah'],
        'Discharge_Capacity(Ah)': cycle_data['discharge_capacity_in_Ah']
    })
    cycle_num = cycle_data['cycle_number']

    # 2. Phase Separation
    charge_df = cycle_df[cycle_df['Current(A)'] > 0].copy()
    discharge_df = cycle_df[cycle_df['Current(A)'] < 0].copy()
    rest_df = cycle_df[cycle_df['Current(A)'] == 0].copy()

    # Cutoff Logic
    if not discharge_df.empty:
        cutoff_indices = discharge_df.index[
            discharge_df['Voltage(V)'] <= DISCHARGE_CUTOFF_V
        ]
        if not cutoff_indices.empty:
            first_cutoff_index = cutoff_indices[0]
            discharge_df = discharge_df.loc[:first_cutoff_index]

    # 3. Feature Extraction
    direct_features = _calculate_direct_features(
        cycle_df, charge_df, discharge_df, cycle_num, cell_id
    )

    # [MODIFIED] Use dynamic configuration based on cell_id
    ncm_config = _get_isu_ilcc_config(cell_id)

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
        charge_df, discharge_df, rest_df, direct_features
    )

    # [MODIFIED] Get dynamic voltage intervals
    tevi_intervals, tevd_intervals = _get_voltage_intervals(cell_id)

    anchor_features = _calculate_anchor_features(
        charge_df, discharge_df,
        charge_slope_intervals_pct,
        discharge_slope_intervals_pct,
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
    charge_slope_intervals_pct: IntervalList,
    discharge_slope_intervals_pct: IntervalList,
    num_cycles: Optional[int] = None
):
    try:
        with open(file_path, 'rb') as f:
            data_dict = pickle.load(f)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return

    if 'cycle_data' not in data_dict or not data_dict['cycle_data']:
        print(f"Warning: No 'cycle_data' found in {file_path}")
        return

    # Handle iterator vs list
    cycles_to_process = list(data_dict['cycle_data'])
    all_cycle_features = []

    if num_cycles is not None and num_cycles > 0:
        cycles_to_process = cycles_to_process[:num_cycles]

    cell_id = data_dict.get('cell_id', file_path.stem)

    for cycle_data in tqdm(cycles_to_process, desc=f"Processing {cell_id}"):
        if not cycle_data.get('time_in_s'):
            continue
        try:
            features = extract_features_for_cycle(
                cycle_data, cell_id,
                charge_slope_intervals_pct,
                discharge_slope_intervals_pct,
                output_dir=output_dir
            )
            if features:
                all_cycle_features.append(features)
        except Exception:
            # Skip corrupted cycles to maintain flow
            continue

    if not all_cycle_features:
        print(f"Warning: No features extracted for {cell_id}")
        return

    features_df = pd.DataFrame(all_cycle_features)

    # Column Ordering
    ordered_cols = [col for col in [
        'Cycle_Number',
        'Discharge_Capacity', 'Charge_Capacity',
        'Discharge_Energy', 'Charge_Energy',
        'Coulombic_Efficiency', 'Energy_Efficiency',
        'charge_c_rate', 'discharge_c_rate', 'ICHV', 'UVP_time',
        'TCCC', 'TCVC', 'CV_Current_Tau', 'UVP',
        'IDV', 'LVP_time', 'var_I_discharge',
        'var_V_discharge', 'median_V_discharge', 'total_discharge_time',
        'LVP',
        'ICP', 'ICPL_V', 'ICP_FWHM', 'ICP_Area',
        'ICV', 'ICVL_V',
        'DVP', 'DVPL_V', 'DVP_FWHM', 'DVP_Area',
        'DVV', 'DVVL_V',
        'Internal_Resistance', 'RCV', 'skew_V_discharge',
        'charge_slope_1', 'charge_slope_2', 'charge_slope_3',
        'discharge_slope_1', 'discharge_slope_2', 'discharge_slope_3',
        'TEVI_1', 'TEVI_2', 'TEVI_3', 'TEVD_1', 'TEVD_2', 'TEVD_3'
    ] if col in features_df.columns]

    final_cols = ordered_cols + [
        col for col in features_df.columns if col not in ordered_cols
    ]
    features_df = features_df[final_cols]

    output_file = output_dir / f"{cell_id}.csv"
    features_df.to_csv(output_file, index=False)
    print(f"Features for {cell_id} saved to {output_file}")


def main():
    processed_data_dir = Path('F:/datasets/battery/ISU_ILCC')
    output_dir = project_root / 'results' / 'ISU_ILCC'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Intervals
    charge_slope_intervals_pct = [(0.10, 0.30), (0.10, 0.50), (0.10, 0.80)]
    discharge_slope_intervals_pct = [(0.10, 0.30), (0.10, 0.50), (0.10, 0.80)]

    # Set to None to extract all cycles
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
            file_path,
            output_dir,
            charge_slope_intervals_pct,
            discharge_slope_intervals_pct,
            num_cycles=num_cycles_to_extract
        )


if __name__ == '__main__':
    main()
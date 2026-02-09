import pickle
import warnings
import sys
import traceback
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

from src.utils.math_tools import fit_cv_decay, get_interp_val
from src.utils.feature_tools import extract_ic_dv_features

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# --- Configuration Constants ---
# HUST Dataset Specifics
CHARGE_CUTOFF_V = 3.6
DISCHARGE_CUTOFF_V = 2.0

# Reference Table for Discharge Protocols (Provided in need_fixed.md)
# C1/C2/C3/C4 represent different discharge stages
TABLE_S1 = {
    "#1": {"Cycle life": 1504, "C1": "5C", "C2": "1C", "C3": "1C", "C4": "1C"},
    "#2": {"Cycle life": 2678, "C1": "5C", "C2": "1C", "C3": "2C", "C4": "1C"},
    "#3": {"Cycle life": 1858, "C1": "5C", "C2": "1C", "C3": "3C", "C4": "1C"},
    "#4": {"Cycle life": 1500, "C1": "5C", "C2": "1C", "C3": "4C", "C4": "1C"},
    "#5": {"Cycle life": 1971, "C1": "5C", "C2": "1C", "C3": "5C", "C4": "1C"},
    "#6": {"Cycle life": 1143, "C1": "5C", "C2": "2C", "C3": "1C", "C4": "1C"},
    "#7": {"Cycle life": 1678, "C1": "5C", "C2": "2C", "C3": "2C", "C4": "1C"},
    "#8": {"Cycle life": 2285, "C1": "5C", "C2": "2C", "C3": "3C", "C4": "1C"},
    "#9": {"Cycle life": 2651, "C1": "5C", "C2": "2C", "C3": "5C", "C4": "1C"},
    "#10": {"Cycle life": 1751, "C1": "5C", "C2": "3C", "C3": "1C", "C4": "1C"},
    "#11": {"Cycle life": 1499, "C1": "5C", "C2": "3C", "C3": "2C", "C4": "1C"},
    "#12": {"Cycle life": 1386, "C1": "5C", "C2": "3C", "C3": "3C", "C4": "1C"},
    "#13": {"Cycle life": 1572, "C1": "5C", "C2": "3C", "C3": "4C", "C4": "1C"},
    "#14": {"Cycle life": 2202, "C1": "5C", "C2": "3C", "C3": "5C", "C4": "1C"},
    "#15": {"Cycle life": 1481, "C1": "5C", "C2": "4C", "C3": "1C", "C4": "1C"},
    "#16": {"Cycle life": 1938, "C1": "5C", "C2": "4C", "C3": "2C", "C4": "1C"},
    "#17": {"Cycle life": 2283, "C1": "5C", "C2": "4C", "C3": "3C", "C4": "1C"},
    "#18": {"Cycle life": 1649, "C1": "5C", "C2": "4C", "C3": "4C", "C4": "1C"},
    "#19": {"Cycle life": 1766, "C1": "5C", "C2": "4C", "C3": "5C", "C4": "1C"},
    "#20": {"Cycle life": 2657, "C1": "5C", "C2": "5C", "C3": "1C", "C4": "1C"},
    "#21": {"Cycle life": 2491, "C1": "5C", "C2": "5C", "C3": "2C", "C4": "1C"},
    "#22": {"Cycle life": 2479, "C1": "5C", "C2": "5C", "C3": "3C", "C4": "1C"},
    "#23": {"Cycle life": 2342, "C1": "5C", "C2": "5C", "C3": "4C", "C4": "1C"},
    "#24": {"Cycle life": 2217, "C1": "5C", "C2": "5C", "C3": "5C", "C4": "1C"},
    "#25": {"Cycle life": 1782, "C1": "4C", "C2": "1C", "C3": "1C", "C4": "1C"},
    "#26": {"Cycle life": 1142, "C1": "4C", "C2": "1C", "C3": "2C", "C4": "1C"},
    "#27": {"Cycle life": 1491, "C1": "4C", "C2": "1C", "C3": "3C", "C4": "1C"},
    "#28": {"Cycle life": 1561, "C1": "4C", "C2": "1C", "C3": "4C", "C4": "1C"},
    "#29": {"Cycle life": 1380, "C1": "4C", "C2": "1C", "C3": "5C", "C4": "1C"},
    "#30": {"Cycle life": 2216, "C1": "4C", "C2": "2C", "C3": "1C", "C4": "1C"},
    "#31": {"Cycle life": 1706, "C1": "4C", "C2": "2C", "C3": "2C", "C4": "1C"},
    "#32": {"Cycle life": 2507, "C1": "4C", "C2": "2C", "C3": "3C", "C4": "1C"},
    "#33": {"Cycle life": 1926, "C1": "4C", "C2": "2C", "C3": "4C", "C4": "1C"},
    "#34": {"Cycle life": 2689, "C1": "4C", "C2": "2C", "C3": "5C", "C4": "1C"},
    "#35": {"Cycle life": 1962, "C1": "4C", "C2": "3C", "C3": "1C", "C4": "1C"},
    "#36": {"Cycle life": 1583, "C1": "4C", "C2": "3C", "C3": "2C", "C4": "1C"},
    "#37": {"Cycle life": 2460, "C1": "4C", "C2": "3C", "C3": "3C", "C4": "1C"},
    "#38": {"Cycle life": 1448, "C1": "4C", "C2": "3C", "C3": "4C", "C4": "1C"},
    "#39": {"Cycle life": 1609, "C1": "4C", "C2": "4C", "C3": "1C", "C4": "1C"},
    "#40": {"Cycle life": 1908, "C1": "4C", "C2": "4C", "C3": "2C", "C4": "1C"},
    "#41": {"Cycle life": 1804, "C1": "4C", "C2": "4C", "C3": "3C", "C4": "1C"},
    "#42": {"Cycle life": 1717, "C1": "4C", "C2": "4C", "C3": "4C", "C4": "1C"},
    "#43": {"Cycle life": 2178, "C1": "4C", "C2": "4C", "C3": "5C", "C4": "1C"},
    "#44": {"Cycle life": 2468, "C1": "4C", "C2": "5C", "C3": "1C", "C4": "1C"},
    "#45": {"Cycle life": 2450, "C1": "4C", "C2": "5C", "C3": "3C", "C4": "1C"},
    "#46": {"Cycle life": 1690, "C1": "4C", "C2": "5C", "C3": "4C", "C4": "1C"},
    "#47": {"Cycle life": 2030, "C1": "4C", "C2": "5C", "C3": "5C", "C4": "1C"},
    "#48": {"Cycle life": 1295, "C1": "3C", "C2": "1C", "C3": "1C", "C4": "1C"},
    "#49": {"Cycle life": 1393, "C1": "3C", "C2": "1C", "C3": "2C", "C4": "1C"},
    "#50": {"Cycle life": 1875, "C1": "3C", "C2": "1C", "C3": "3C", "C4": "1C"},
    "#51": {"Cycle life": 1419, "C1": "3C", "C2": "1C", "C3": "4C", "C4": "1C"},
    "#52": {"Cycle life": 1685, "C1": "3C", "C2": "1C", "C3": "5C", "C4": "1C"},
    "#53": {"Cycle life": 1938, "C1": "3C", "C2": "2C", "C3": "1C", "C4": "1C"},
    "#54": {"Cycle life": 1308, "C1": "3C", "C2": "2C", "C3": "2C", "C4": "1C"},
    "#55": {"Cycle life": 2041, "C1": "3C", "C2": "2C", "C3": "3C", "C4": "1C"},
    "#56": {"Cycle life": 2290, "C1": "3C", "C2": "2C", "C3": "4C", "C4": "1C"},
    "#57": {"Cycle life": 1885, "C1": "3C", "C2": "2C", "C3": "5C", "C4": "1C"},
    "#58": {"Cycle life": 1348, "C1": "3C", "C2": "3C", "C3": "1C", "C4": "1C"},
    "#59": {"Cycle life": 2365, "C1": "3C", "C2": "3C", "C3": "2C", "C4": "1C"},
    "#60": {"Cycle life": 2047, "C1": "3C", "C2": "3C", "C3": "3C", "C4": "1C"},
    "#61": {"Cycle life": 1679, "C1": "3C", "C2": "3C", "C3": "4C", "C4": "1C"},
    "#62": {"Cycle life": 2057, "C1": "3C", "C2": "3C", "C3": "5C", "C4": "1C"},
    "#63": {"Cycle life": 2143, "C1": "3C", "C2": "4C", "C3": "1C", "C4": "1C"},
    "#64": {"Cycle life": 1905, "C1": "3C", "C2": "4C", "C3": "2C", "C4": "1C"},
    "#65": {"Cycle life": 1975, "C1": "3C", "C2": "4C", "C3": "3C", "C4": "1C"},
    "#66": {"Cycle life": 2168, "C1": "3C", "C2": "4C", "C3": "4C", "C4": "1C"},
    "#67": {"Cycle life": 1742, "C1": "3C", "C2": "4C", "C3": "5C", "C4": "1C"},
    "#68": {"Cycle life": 2012, "C1": "3C", "C2": "5C", "C3": "1C", "C4": "1C"},
    "#69": {"Cycle life": 2308, "C1": "3C", "C2": "5C", "C3": "2C", "C4": "1C"},
    "#70": {"Cycle life": 1702, "C1": "3C", "C2": "5C", "C3": "3C", "C4": "1C"},
    "#71": {"Cycle life": 1697, "C1": "3C", "C2": "5C", "C3": "4C", "C4": "1C"},
    "#72": {"Cycle life": 1848, "C1": "3C", "C2": "5C", "C3": "5C", "C4": "1C"},
    "#73": {"Cycle life": 1811, "C1": "2C", "C2": "4C", "C3": "1C", "C4": "1C"},
    "#74": {"Cycle life": 2030, "C1": "2C", "C2": "5C", "C3": "2C", "C4": "1C"},
    "#75": {"Cycle life": 2285, "C1": "2C", "C2": "3C", "C3": "3C", "C4": "1C"},
    "#76": {"Cycle life": 1783, "C1": "2C", "C2": "2C", "C3": "4C", "C4": "1C"},
    "#77": {"Cycle life": 1400, "C1": "2C", "C2": "1C", "C3": "5C", "C4": "1C"},
}

def _detect_stages(
    df: pd.DataFrame,
    i_thresh: float = 0.5,
    v_thresh: float = 0.01,
    min_len: int = 5
) -> List[Dict[str, Any]]:
    """
    Detects operational stages based on Current and Voltage changes.
    Handles cases where Current remains same but Voltage jumps (e.g. C1->C2 with same C-rate).
    """
    if df.empty:
        return []

    curr = df['Current(A)'].values
    volt = df['Voltage(V)'].values
    time = df['Time(s)'].values

    # Calculate absolute differences
    # Use prepend to align indices with original array
    dI = np.abs(np.diff(curr, prepend=curr[0]))
    dV = np.abs(np.diff(volt, prepend=volt[0]))

    # Identify Step Change Points:
    # 1. Current magnitude changes significantly (> i_thresh)
    # 2. Current is stable (diff < 0.1) BUT Voltage jumps (> v_thresh) - for same C-rate transition
    is_step = (dI > i_thresh) | ((dI < 0.1) & (dV > v_thresh))

    # Get indices where step happens
    step_indices = np.where(is_step)[0]

    # Add start(0) and end(len) to boundaries
    # Filter step_indices to remove index 0 if present (to avoid duplicate)
    step_indices = step_indices[step_indices > 0]
    boundaries = [0] + sorted(list(set(step_indices))) + [len(df)]

    stages = []
    # Merge consecutive boundaries that are too close (noise) is handled by min_len check below
    # But strictly we iterate intervals

    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i+1]

        # Filter noise segments
        if end - start < min_len:
            continue

        segment_time = time[start:end]
        segment_curr = curr[start:end]

        duration = segment_time[-1] - segment_time[0]
        avg_curr = np.mean(segment_curr)

        stages.append({
            'start_iloc': start,
            'end_iloc': end,
            'duration': duration,
            'current': avg_curr
        })

    return stages


def _calculate_direct_features(
    cycle_df: pd.DataFrame,
    charge_df: pd.DataFrame,
    discharge_df: pd.DataFrame,
    cycle_num: int,
    cell_id: str
) -> Dict[str, Any]:
    """Calculates basic features (Capacity, Energy, CV Dynamics)."""
    features = {}

    # --- A. Overall Cycle Features ---
    features['Cycle_Number'] = cycle_num

    # [FIX] Calculate Capacity via Integration to ensure consistency and avoid CE > 1
    # caused by potential offsets in pre-calculated columns
    if not charge_df.empty:
        # Integrate Current over Time: Q = ∫ I dt
        q_charge_as = trapezoid(y=charge_df['Current(A)'].abs(), x=charge_df['Time(s)'])
        chg_cap = q_charge_as / 3600.0
    else:
        chg_cap = 0.0

    if not discharge_df.empty:
        q_discharge_as = trapezoid(y=discharge_df['Current(A)'].abs(), x=discharge_df['Time(s)'])
        dis_cap = q_discharge_as / 3600.0
    else:
        dis_cap = 0.0

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

    # --- C. Charging Phase Features ---
    features['CV_Current_Tau'] = 0.0

    if not charge_df.empty:
        features['ICHV(V)'] = charge_df['Voltage(V)'].iloc[0]
        features['UVP_time(s)'] = charge_df['Time(s)'].iloc[-1]
        features['UVP(V)'] = CHARGE_CUTOFF_V

        # [NEW] Multi-stage Charge Detection (C1, C2)
        chg_stages = _detect_stages(charge_df)

        # Record C1
        if len(chg_stages) >= 1:
            features['charge_current_1(A)'] = abs(chg_stages[0]['current'])
            features['charge_time_1(s)'] = chg_stages[0]['duration']
        else:
            features['charge_current_1(A)'] = 0.0
            features['charge_time_1(s)'] = 0.0

        # Record C2
        if len(chg_stages) >= 2:
            features['charge_current_2(A)'] = abs(chg_stages[1]['current'])
            features['charge_time_2(s)'] = chg_stages[1]['duration']
        else:
            features['charge_current_2(A)'] = 0.0
            features['charge_time_2(s)'] = 0.0

        # [NEW] Delta V (5C -> 1C) and Time Ratio
        if len(chg_stages) >= 2:
             # Assuming Stage 0 is 5C, Stage 1 is 1C
             idx_transition = chg_stages[1]['start_iloc']
             # Voltage just before transition (end of C1)
             v_c1_end = charge_df['Voltage(V)'].iloc[idx_transition - 1] if idx_transition > 0 else charge_df['Voltage(V)'].iloc[0]
             # Voltage just after transition (start of C2)
             v_c2_start = charge_df['Voltage(V)'].iloc[idx_transition]

             features['resistance_jump_dV'] = abs(v_c1_end - v_c2_start)
        else:
             features['resistance_jump_dV'] = 0.0

        if features['charge_time_2(s)'] > 0:
            features['charge_time_ratio_1_2'] = features['charge_time_1(s)'] / features['charge_time_2(s)']
        else:
            features['charge_time_ratio_1_2'] = 0.0

        # HUST Protocol: CC -> CV
        # Identify CV start
        v_upper_limit = CHARGE_CUTOFF_V
        charge_voltage = charge_df['Voltage(V)']
        mask_cv = charge_voltage >= (v_upper_limit - 0.01)

        if mask_cv.any():
            cv_start_idx = mask_cv.idxmax()
            time_at_v_limit = charge_df.loc[cv_start_idx, 'Time(s)']

            features['TCCC(s)'] = time_at_v_limit - charge_df['Time(s)'].iloc[0]
            features['TCVC(s)'] = charge_df['Time(s)'].iloc[-1] - time_at_v_limit

            # CV Tau Fitting
            cv_df = charge_df.loc[cv_start_idx:]
            cv_current = cv_df['Current(A)'].values
            cv_time = cv_df['Time(s)'].values

            # Robust filter for fitting
            valid_mask = cv_current > 0.001
            if np.sum(valid_mask) > 15:
                features['CV_Current_Tau'] = fit_cv_decay(
                    cv_time[valid_mask],
                    cv_current[valid_mask]
                )
        else:
            features['TCCC(s)'] = (
                charge_df['Time(s)'].iloc[-1] - charge_df['Time(s)'].iloc[0]
            )
            features['TCVC(s)'] = 0
    else:
        features.update({
            'ICHV(V)': 0, 'UVP_time(s)': 0, 'UVP(V)': 0, 'TCCC(s)': 0, 'TCVC(s)': 0,
            'charge_current_1(A)': 0, 'charge_time_1(s)': 0,
            'charge_current_2(A)': 0, 'charge_time_2(s)': 0
        })

    # --- D. Discharging Phase Features ---
    if not discharge_df.empty:
        features['IDV(V)'] = discharge_df['Voltage(V)'].iloc[0]
        features['LVP_time(s)'] = discharge_df['Time(s)'].iloc[-1]
        features['LVP(V)'] = DISCHARGE_CUTOFF_V

        # [MODIFIED] Commented out as requested
        # features['var_I_discharge'] = discharge_df['Current(A)'].var()
        # features['var_V_discharge'] = discharge_df['Voltage(V)'].var()
        # features['median_V_discharge(V)'] = discharge_df['Voltage(V)'].median()

        # features['var_I_discharge'] = 0.0 # Placeholder
        # features['var_V_discharge'] = 0.0 # Placeholder
        # features['median_V_discharge(V)'] = 0.0 # Placeholder

        features['total_discharge_time(s)'] = (
            discharge_df['Time(s)'].iloc[-1] -
            discharge_df['Time(s)'].iloc[0]
        )

        # [NEW] Multi-stage Discharge Detection (C1, C2, C3, C4)
        dis_stages = _detect_stages(discharge_df)

        # Initialize C1-C4 features
        for i in range(1, 5):
            features[f'discharge_current_{i}'] = 0.0
            features[f'discharge_time_{i}'] = 0.0

        # Fill detected stages
        for i, stage in enumerate(dis_stages[:4]): # Max 4 stages
            features[f'discharge_current_{i+1}'] = abs(stage['current'])
            features[f'discharge_time_{i+1}'] = stage['duration']

    else:
        features.update({
            'IDV(V)': 0, 'LVP_time(s)': 0, 'LVP(V)': 0,
            'var_I_discharge': 0, 'var_V_discharge': 0,
            'median_V_discharge(V)': 0, 'total_discharge_time(s)': 0,
            'discharge_current_1': 0, 'discharge_time_1': 0,
            'discharge_current_2': 0, 'discharge_time_2': 0,
            'discharge_current_3': 0, 'discharge_time_3': 0,
            'discharge_current_4': 0, 'discharge_time_4': 0
        })

    return features


def _calculate_advanced_features(
    charge_df: pd.DataFrame,
    discharge_df: pd.DataFrame,
    features: Dict[str, Any],
    v_rest_end_for_ir: float = np.nan
) -> Dict[str, Any]:
    """Calculates internal resistance and statistical features."""
    adv_features = {}

    # Internal Resistance
    if not charge_df.empty and not discharge_df.empty:
        # Use provided rest voltage or fallback to end of charge
        v_pre_discharge = (
            v_rest_end_for_ir if not pd.isna(v_rest_end_for_ir)
            else charge_df['Voltage(V)'].iloc[-1]
        )
        v_discharge_start = discharge_df['Voltage(V)'].iloc[0]
        i_discharge_start = abs(discharge_df['Current(A)'].iloc[0])

        if i_discharge_start > 0:
            adv_features['Internal_Resistance(Ohm)'] = (
                (v_pre_discharge - v_discharge_start) / i_discharge_start
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

    # [MODIFIED] Commented out as requested
    # if not discharge_df.empty:
    #     adv_features['skew_V_discharge'] = skew(discharge_df['Voltage(V)'])
    # else:
    #     adv_features['skew_V_discharge'] = 0.0
    # adv_features['skew_V_discharge'] = 0.0

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

        # Binary search for speed
        idx = np.searchsorted(time_array, target_absolute_time, side='left')

        if idx == 0:
            closest_iloc = 0
        elif idx == len(time_array):
            closest_iloc = len(time_array) - 1
        else:
            if (target_absolute_time - time_array[idx - 1]) < (time_array[idx] - target_absolute_time):
                closest_iloc = idx - 1
            else:
                closest_iloc = idx
        return df['Voltage(V)'].iloc[closest_iloc]

    # Charge Slopes
    charge_dur = (
        charge_df['Time(s)'].iloc[-1] - charge_df['Time(s)'].iloc[0]
        if not charge_df.empty else 0
    )
    for i, (pct_start, pct_end) in enumerate(charge_slope_intervals):
        if charge_dur > 0:
            v_start = get_voltage_at_relative_time(charge_df, charge_dur * pct_start)
            v_end = get_voltage_at_relative_time(charge_df, charge_dur * pct_end)
            dt = charge_dur * (pct_end - pct_start)

            if v_start is not None and v_end is not None and dt > 0:
                anchor_features[f'charge_slope_{i + 1}'] = (v_end - v_start) / dt
            else:
                anchor_features[f'charge_slope_{i + 1}'] = 0
        else:
            anchor_features[f'charge_slope_{i + 1}'] = 0

    # [MODIFIED] Commented out as requested
    # for i in range(len(discharge_slope_intervals)):
    #    anchor_features[f'discharge_slope_{i + 1}'] = 0.0

    # TEVI / TEVD
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

    # [MODIFIED] Commented out as requested
    # for i in range(len(tevd_intervals)):
    #     anchor_features[f'TEVD_{i + 1}'] = 0.0

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
    """Extracts features for a single cycle using Charge phase for IC/DV."""

    # 1. Prepare Data
    cycle_df = pd.DataFrame({
        'Time(s)': cycle_data['time_in_s'],
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

    # --- HUST Cutoff (2.0V) ---
    if not discharge_df.empty:
        cutoff_indices = discharge_df.index[
            discharge_df['Voltage(V)'] <= DISCHARGE_CUTOFF_V
        ]
        if not cutoff_indices.empty:
            discharge_df = discharge_df.loc[:cutoff_indices[0]]

    # Find Rest Voltage for IR
    v_rest_end = np.nan
    if not charge_df.empty and not discharge_df.empty and not rest_df.empty:
        try:
            charge_end_time = charge_df['Time(s)'].iloc[-1]
            discharge_start_time = discharge_df['Time(s)'].iloc[0]

            intermediate_rest_df = rest_df[
                (rest_df['Time(s)'] > charge_end_time) &
                (rest_df['Time(s)'] < discharge_start_time)
            ]

            if not intermediate_rest_df.empty:
                v_rest_end = intermediate_rest_df['Voltage(V)'].iloc[-1]
        except IndexError:
            v_rest_end = np.nan

    # 3. Extract Features
    direct_features = _calculate_direct_features(
        cycle_df, charge_df, discharge_df, cycle_num, cell_id
    )

    # REFACTORED: Use shared module for IC/DV
    # HUST specific: Use Charge DF for derivative features
    # [MODIFIED] Use C2 Charge Data if available

    # Identify C2 phase using the same logic
    chg_stages = _detect_stages(charge_df)
    target_ic_df = pd.DataFrame()

    if len(chg_stages) >= 2:
        # Use Stage 2 (Index 1)
        s, e = chg_stages[1]['start_iloc'], chg_stages[1]['end_iloc']
        # Map back to original indices using iloc
        target_ic_df = charge_df.iloc[s:e].copy()
    elif len(chg_stages) == 1:
        # Fallback to Stage 1 if only 1 exists
        target_ic_df = charge_df.copy()
    else:
        target_ic_df = charge_df.copy()

    # Config for HUST (LFP)
    lfp_config = {
        'peak_mode': 1,
        'nominal_capacity': 1.1, # Approx for HUST A123
        'window_length_ic': 51,
        'window_length_dv': 21,
        'peak_height_ic': 0.01,
        # 'peak_height_dv': 0.01,
        'voltage_range_ic': (3.0, 3.6),
        # 'voltage_range_dv': (3.0, 3.6),
        'prominence_ic': 0.02,
        # 'prominence_dv': 0.02,
        'ic_step_size': 0.001,
        'dv_step_size': 1.1 * 0.005, # 0.01 Ah
        'search_window_dvv': 0.1,
        'search_window_dvp': 0.1,
        'initial_capacity_cut_fraction': 0.05,
        'icv_search_offset_lower': 0.05,
        'icv_search_offset_upper': 0.5,
        # [MODIFIED] Fixes for Issues 2, 3, 5
        'disable_dvv': True,
        'dvp_capacity_range': (1.05, 1.15),
        'ic_area_config': {'method': 'fixed_width', 'width_v': 0.05}
    }

    plot_params = None
    if output_dir:
        plot_params = {
            'cell_id': cell_id,
            'cycle_num': cycle_num,
            'output_dir': output_dir
        }

    # Pass the targeted C2 dataframe
    # [FIX] Map Charge Capacity to Discharge Capacity column as required by the generic tool
    ic_input_df = target_ic_df.copy()
    if not ic_input_df.empty and 'Charge_Capacity(Ah)' in ic_input_df.columns:
        ic_input_df['Discharge_Capacity(Ah)'] = ic_input_df['Charge_Capacity(Ah)']

    derivative_features = extract_ic_dv_features(
        ic_input_df,
        config=lfp_config,
        plot_params=plot_params
    )

    advanced_features = _calculate_advanced_features(
        charge_df, discharge_df, direct_features, v_rest_end_for_ir=v_rest_end
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
            battery_data = pickle.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return

    if 'cycle_data' in battery_data and battery_data['cycle_data'] is not None:
        battery_data['cycle_data'] = [c for c in battery_data['cycle_data']]

    all_cycle_features = []

    cycles_to_process = battery_data['cycle_data']
    if battery_data['cell_id'] == 'HUST_7-5':
        cycles_to_process = battery_data['cycle_data'][2:]  # Skip bad cycles

    if num_cycles is not None and num_cycles > 0:
        cycles_to_process = cycles_to_process[:num_cycles]

    for cycle_data in tqdm(cycles_to_process, desc=f"Processing {battery_data['cell_id']}"):
        if not cycle_data.get('time_in_s'):
            continue
        try:
            features = extract_features_for_cycle(
                cycle_data, battery_data['cell_id'],
                charge_slope_intervals, discharge_slope_intervals,
                tevi_intervals, tevd_intervals,
                output_dir=output_dir
            )
            all_cycle_features.append(features)
        except Exception:
            # [FIX] Print traceback to diagnose why features are not extracted
            traceback.print_exc()
            continue

    if not all_cycle_features:
        print(f"Warning: No features extracted for {battery_data['cell_id']}")
        return

    features_df = pd.DataFrame(all_cycle_features)

    ordered_cols = [col for col in [
        # Overall
        'Cycle_Number',
        'Discharge_Capacity', 'Charge_Capacity',
        'Discharge_Energy', 'Charge_Energy',
        'Coulombic_Efficiency', 'Energy_Efficiency',
        # [NEW] Multi-stage
        'charge_current_1', 'charge_time_1',
        'charge_current_2', 'charge_time_2',
        'charge_time_ratio_1_2', 'resistance_jump_dV',
        'discharge_current_1', 'discharge_time_1',
        'discharge_current_2', 'discharge_time_2',
        'discharge_current_3', 'discharge_time_3',
        'discharge_current_4', 'discharge_time_4',
        # Charge
        'ICHV', 'UVP_time', 'TCCC', 'TCVC', 'CV_Current_Tau', 'UVP',
        # Discharge
        'IDV', 'LVP_time', 'total_discharge_time', 'LVP',
        # Curves (Calculated from CHARGE C2)
        'ICP', 'ICPL_V', 'ICP_FWHM', 'ICP_Area',
        'ICV', 'ICVL_V',
        'DVP', 'DVPL_V', 'DVP_FWHM', 'DVP_Area',
        # 'DVV', 'DVVL_V', # Abandoned
        # Advanced
        'Internal_Resistance', 'RCV',
        # Anchor
        'charge_slope_1', 'charge_slope_2', 'charge_slope_3',
        'TEVI_1', 'TEVI_2', 'TEVI_3'
    ] if col in features_df.columns]

    final_cols = ordered_cols + [
        col for col in features_df.columns if col not in ordered_cols
    ]
    features_df = features_df[final_cols]

    output_file = output_dir / f"{battery_data['cell_id']}.csv"
    features_df.to_csv(output_file, index=False)
    print(f"Features for {battery_data['cell_id']} saved to {output_file}")


def main():
    # Keep input paths but ensure output logic matches current refactor state
    processed_data_dir = project_root / 'data' / 'HUST'
    output_dir = project_root / 'results' / 'features' / 'HUST'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Intervals Setup
    charge_slope_intervals = [(0.10, 0.30), (0.10, 0.50), (0.10, 0.80)]
    discharge_slope_intervals = [(0.10, 0.30), (0.10, 0.50), (0.10, 0.80)]

    # HUST Voltage Intervals (2.0V - 3.6V range)
    tevi_intervals = [(3.0, 3.2), (3.2, 3.4), (3.4, 3.5)]
    tevd_intervals = [(3.0, 2.8), (2.8, 2.5), (2.5, 2.2)]

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

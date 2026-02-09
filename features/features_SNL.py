"""
Feature Extraction Script for SNL Battery Dataset
=================================================

Refactored to use shared utilities (src.utils).
"""
import pickle
import sys
import warnings
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from scipy.stats import skew
from tqdm import tqdm

# Add project root
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.utils.math_tools import fit_cv_decay
from src.utils.feature_tools import extract_ic_dv_features, extract_charge_ic_dv_features

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


# ==========================================
# --- Cleaning Functions (Simplified) ---
# ==========================================

def _get_shape_stream_charge(raw_charge_df: pd.DataFrame, v_thresh: float = 2.5) -> pd.DataFrame:
    """Pre-process/Clean charge dataframe."""
    if raw_charge_df.empty:
        return pd.DataFrame()

    mask = (raw_charge_df['Voltage(V)'] >= v_thresh) & \
           (raw_charge_df['Current(A)'] > 0.01)

    clean_df = raw_charge_df[mask].copy()

    if clean_df.empty:
        return pd.DataFrame()

    # Reset Time Axis
    start_time = clean_df['Time(s)'].iloc[0]
    clean_df['Time(s)'] = clean_df['Time(s)'] - start_time

    return clean_df


def _get_shape_stream_discharge(raw_discharge_df: pd.DataFrame) -> pd.DataFrame:
    """Pre-process/Clean discharge dataframe.

    Includes tail truncation for abnormal voltage rise (physically impossible during CC discharge).
    """
    if raw_discharge_df.empty:
        return pd.DataFrame()

    # Strict Current Filter
    mask = raw_discharge_df['Current(A)'] < -0.005 # Relaxed from -0.05 to -0.005 (5mA) to avoid false empty

    if not mask.any():
        return pd.DataFrame()

    # Block Identification (Longest Continuous Segment)
    df_temp = raw_discharge_df.copy()
    df_temp['block_id'] = (mask != mask.shift()).cumsum()
    valid_blocks = df_temp[mask]

    if valid_blocks.empty:
        return pd.DataFrame()

    block_counts = valid_blocks['block_id'].value_counts()
    best_block_id = block_counts.idxmax()

    clean_df = valid_blocks[valid_blocks['block_id'] == best_block_id].copy()

    # [NEW] Head Truncation: Detect ghost points (abnormal low voltage at start)
    # Some SNL files have a ghost point (e.g., 1.99V) at index 0, followed by real discharge (3.2V)
    # Strategy: Find the maximum voltage point and truncate everything before it.
    # Discharge should be monotonically decreasing, so starting from max V is safe.
    if not clean_df.empty:
        max_v_idx_loc = clean_df['Voltage(V)'].argmax()
        # Only truncate if max voltage is not at the very end (unlikely for discharge)
        if max_v_idx_loc < len(clean_df) - 1:
             clean_df = clean_df.iloc[max_v_idx_loc:].copy()

    # [NEW] Tail Truncation: Detect voltage rise at end of discharge
    # Scan from end backwards. If V[i] > V[i-1] (rising towards end), it's bad.
    # We want to keep the segment where V is monotonically decreasing (or at least not rising sharply).
    # Simple heuristic: Find the minimum voltage point. Anything after that (if V rises) is suspect.
    if not clean_df.empty:
        # Find index of minimum voltage
        min_v_idx_loc = clean_df['Voltage(V)'].argmin()
        # Truncate everything after the minimum voltage point
        # Only truncate if we have a reasonable amount of data before the minimum (e.g., > 10 points)
        if min_v_idx_loc > 10:
             clean_df = clean_df.iloc[:min_v_idx_loc + 1].copy()

    if clean_df.empty:
         return pd.DataFrame()

    # Reset Time Axis
    start_time = clean_df['Time(s)'].iloc[0]
    clean_df['Time(s)'] = clean_df['Time(s)'] - start_time

    return clean_df


# ==========================================
# --- Feature Extraction Functions ---
# ==========================================

def _calculate_direct_features(
    charge_df: pd.DataFrame,
    discharge_df: pd.DataFrame,
    rest_df: pd.DataFrame,
    cycle_num: int,
    battery_data: Any,
    work_condition: int,
    cycle_df: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    """Calculates Capacity, Energy from RAW Data.

    Args:
        cycle_df: Full cycle dataframe for masked integration (robust to gaps).
    """
    features = {}
    features['Cycle_Number'] = cycle_num

    # Helper for masked integration over full cycle time (Fixes gap interpolation issue)
    def integrate_masked(c_df, mask_condition):
        if c_df is None or c_df.empty: return 0.0
        y = c_df['Current(A)'].values.copy()
        y[~mask_condition] = 0.0
        return trapezoid(y=np.abs(y), x=c_df['Time(s)'].values) / 3600.0

    def integrate_energy_masked(c_df, mask_condition):
        if c_df is None or c_df.empty: return 0.0
        i_vals = c_df['Current(A)'].values.copy()
        v_vals = c_df['Voltage(V)'].values
        i_vals[~mask_condition] = 0.0
        p = v_vals * np.abs(i_vals)
        return trapezoid(y=p, x=c_df['Time(s)'].values) / 3600.0

    # Discharge (Raw)
    if not discharge_df.empty:
        features['Discharge_Capacity(Ah)'] = discharge_df['Discharge_Capacity(Ah)'].max()

        # [Fix] Use Masked Integration if cycle_df provided
        if cycle_df is not None:
            # Mask: Current < -0.005 (match _get_shape_stream_discharge filter approx)
            mask = cycle_df['Current(A)'] < -0.005
            features['Discharge_Energy(Wh)'] = integrate_energy_masked(cycle_df, mask)
        else:
            p_dis = discharge_df['Voltage(V)'] * discharge_df['Current(A)'].abs()
            features['Discharge_Energy(Wh)'] = trapezoid(y=p_dis, x=discharge_df['Time(s)']) / 3600.0

        features['IDV(V)'] = discharge_df['Voltage(V)'].iloc[0]
        features['LVP_time(s)'] = discharge_df['Time(s)'].iloc[-1] - discharge_df['Time(s)'].iloc[0]
        features['total_discharge_time(s)'] = discharge_df['Time(s)'].iloc[-1] - discharge_df['Time(s)'].iloc[0]

        features['MAT_discharge(C)'] = discharge_df['Temperature(C)'].max()
        features['MET_discharge(s)'] = discharge_df['Temperature(C)'].mean()
        features['var_I_discharge'] = discharge_df['Current(A)'].var()
        features['median_V_discharge(V)'] = discharge_df['Voltage(V)'].median()
    else:
        features.update({
            'Discharge_Capacity(Ah)': 0, 'Discharge_Energy(Wh)': 0, 'IDV(V)': 0, 'LVP_time(s)': 0,
            'total_discharge_time(s)': 0, 'MAT_discharge(C)': 0, 'MET_discharge(s)': 0,
            'var_I_discharge': 0, 'median_V_discharge(V)': 0
        })

    # Charge (Raw)
    if not charge_df.empty:
        features['Charge_Capacity(Ah)'] = charge_df['Charge_Capacity(Ah)'].max()

        if cycle_df is not None:
             mask = cycle_df['Current(A)'] > 0.005
             features['Charge_Energy(Wh)'] = integrate_energy_masked(cycle_df, mask)
        else:
            p_chg = charge_df['Voltage(V)'] * charge_df['Current(A)'].abs()
            features['Charge_Energy(Wh)'] = trapezoid(y=p_chg, x=charge_df['Time(s)']) / 3600.0

        features['MAT_charge(C)'] = charge_df['Temperature(C)'].max()
        features['MET_charge(s)'] = charge_df['Temperature(C)'].mean()
    else:
        features.update({
            'Charge_Capacity(Ah)': 0, 'Charge_Energy(Wh)': 0, 'MAT_charge(C)': 0, 'MET_charge(s)': 0
        })

    # Efficiencies
    # [Fix] Always use integration for Coulombic Efficiency as requested
    q_dis_int = 0.0
    q_chg_int = 0.0

    if cycle_df is not None:
        q_dis_int = integrate_masked(cycle_df, cycle_df['Current(A)'] < -0.005)
        q_chg_int = integrate_masked(cycle_df, cycle_df['Current(A)'] > 0.005)
    else:
        if not discharge_df.empty:
            q_dis_int = trapezoid(y=discharge_df['Current(A)'].abs(), x=discharge_df['Time(s)']) / 3600.0
        if not charge_df.empty:
            q_chg_int = trapezoid(y=charge_df['Current(A)'].abs(), x=charge_df['Time(s)']) / 3600.0

    if q_chg_int > 1e-6:
        features['Coulombic_Efficiency'] = q_dis_int / q_chg_int
    else:
        features['Coulombic_Efficiency'] = 0

    if features['Charge_Energy(Wh)'] > 1e-6:
        features['Energy_Efficiency'] = features['Discharge_Energy(Wh)'] / features['Charge_Energy(Wh)']
    else:
        features['Energy_Efficiency'] = 0

    # Rest Time
    features['Rest_Time(s)'] = 0.0
    if not rest_df.empty and len(rest_df) > 1:
        features['Rest_Time(s)'] = rest_df['Time(s)'].iloc[-1] - rest_df['Time(s)'].iloc[0]

    # Meta
    features['charge_c_rate'] = battery_data.charge_protocol[0]['rate_in_C']
    features['discharge_c_rate'] = battery_data.discharge_protocol[0]['rate_in_C']
    features['work_condition'] = work_condition
    features['UVP(V)'] = battery_data.max_voltage_limit_in_V
    features['LVP(V)'] = battery_data.min_voltage_limit_in_V

    return features


def _calculate_cv_features(charge_df: pd.DataFrame, uvp: float, work_condition: int) -> Dict[str, Any]:
    """Calculates CV phase features."""
    features = {'TCCC': 0, 'TCVC': 0, 'CV_Current_Tau': 0, 'UVP_time': 0}
    if charge_df.empty:
        return features

    # [Fix] Use relative time for UVP_time (Duration from start of charge)
    features['UVP_time(s)'] = charge_df['Time(s)'].iloc[-1] - charge_df['Time(s)'].iloc[0]
    v_data = charge_df['Voltage(V)']
    v_limit = uvp

    # Heuristic for WC3 or low voltage
    if v_data.max() < (uvp - 0.05) and work_condition != 3:
        features['TCCC(s)'] = charge_df['Time(s)'].iloc[-1] - charge_df['Time(s)'].iloc[0]
        return features

    # [Fix] Robust CV detection: Scan backwards to find end of CC phase
    v_thresh = v_limit - 0.01

    # Check if we have any points below threshold
    low_v_mask = v_data < v_thresh

    if low_v_mask.any():
        # Find index of last point in CC phase
        last_cc_idx = low_v_mask[::-1].idxmax()
        # Get time boundary
        t_switch = charge_df.loc[last_cc_idx, 'Time(s)']

        # Select CV segment (after switch time)
        cv_segment = charge_df[charge_df['Time(s)'] > t_switch]

        if not cv_segment.empty:
            time_at_cv = cv_segment['Time(s)'].iloc[0]
            features['TCCC(s)'] = time_at_cv - charge_df['Time(s)'].iloc[0]
            features['TCVC(s)'] = charge_df['Time(s)'].iloc[-1] - time_at_cv

            if len(cv_segment) > 10:
                features['CV_Current_Tau'] = fit_cv_decay(
                    cv_segment['Time(s)'].values, cv_segment['Current(A)'].values
                )
        else:
            # Reached UVP but no subsequent data?
            features['TCCC(s)'] = charge_df['Time(s)'].iloc[-1] - charge_df['Time(s)'].iloc[0]

    else:
        # No points below threshold? Entirely CV?
        # Likely implies start of charge is already high voltage
        features['TCCC(s)'] = 0.0
        features['TCVC(s)'] = charge_df['Time(s)'].iloc[-1] - charge_df['Time(s)'].iloc[0]
        cv_segment = charge_df
        if len(cv_segment) > 10:
            features['CV_Current_Tau'] = fit_cv_decay(
                cv_segment['Time(s)'].values, cv_segment['Current(A)'].values
            )

    return features


def _calculate_anchor_features(
    charge_df: pd.DataFrame,
    discharge_df: pd.DataFrame,
    charge_slopes: List[Tuple],
    discharge_slopes: List[Tuple],
    tevi_intervals: List[Tuple],
    tevd_intervals: List[Tuple],
    is_dirty_file: bool
) -> Dict[str, Any]:
    """Calculates Slopes and TEVI/TEVD."""
    features = {}

    # --- Helper functions (kept local for specific logic if needed, or inline) ---
    def get_slope(df, p_start, p_end):
        if df.empty or len(df) < 5: return 0.0
        duration = df['Time(s)'].iloc[-1]
        t_start, t_end = duration * p_start, duration * p_end

        times = df['Time(s)'].values
        idx_start = np.searchsorted(times, t_start)
        idx_end = np.searchsorted(times, t_end)

        idx_start = min(idx_start, len(df) - 1)
        idx_end = min(idx_end, len(df) - 1)

        v_start = df['Voltage(V)'].iloc[idx_start]
        v_end = df['Voltage(V)'].iloc[idx_end]
        delta_t = t_end - t_start
        if delta_t < 1e-3: return 0.0
        return (v_end - v_start) / delta_t

    def get_time_for_voltage(df, voltage, direction):
        if df.empty: return None
        if direction == 'charge':
            rows = df[df['Voltage(V)'] >= voltage]
        else:
            rows = df[df['Voltage(V)'] <= voltage]
        return rows['Time(s)'].iloc[0] if not rows.empty else None

    # 1. Slopes
    for i, (s, e) in enumerate(charge_slopes):
        features[f'charge_slope_{i+1}'] = get_slope(charge_df, s, e)
    for i, (s, e) in enumerate(discharge_slopes):
        features[f'discharge_slope_{i+1}'] = get_slope(discharge_df, s, e)

    # 2. TEVI / TEVD
    if is_dirty_file:
        for i in range(len(tevi_intervals)):
            features[f'TEVI_{i+1}'] = 0
        for i in range(len(tevd_intervals)):
            features[f'TEVD_{i+1}'] = 0
    else:
        for i, (v_start, v_end) in enumerate(tevi_intervals):
            t1 = get_time_for_voltage(charge_df, v_start, 'charge')
            t2 = get_time_for_voltage(charge_df, v_end, 'charge')
            if t1 is not None and t2 is not None and t2 > t1:
                features[f'TEVI_{i+1}'] = t2 - t1
            else:
                features[f'TEVI_{i+1}'] = 0

        for i, (v_start, v_end) in enumerate(tevd_intervals):
            t1 = get_time_for_voltage(discharge_df, v_start, 'discharge')
            t2 = get_time_for_voltage(discharge_df, v_end, 'discharge')
            if t1 is not None and t2 is not None and t2 > t1:
                features[f'TEVD_{i+1}'] = t2 - t1
            else:
                features[f'TEVD_{i+1}'] = 0

    return features


def _calculate_advanced_features(
    shape_charge_df: pd.DataFrame,
    shape_discharge_df: pd.DataFrame,
    raw_discharge_df: pd.DataFrame,
    direct_feats: Dict,
    is_dirty_file: bool
) -> Dict[str, Any]:
    """Calculates IR, RCV, Temp Rise, and Skewness."""
    adv = {}

    # 1. Internal Resistance
    if not shape_charge_df.empty and not shape_discharge_df.empty:
        v_c_end = shape_charge_df['Voltage(V)'].iloc[-1]
        v_d_start = shape_discharge_df['Voltage(V)'].iloc[0]
        i_d_start = abs(shape_discharge_df['Current(A)'].iloc[0])
        adv['Internal_Resistance'] = (v_c_end - v_d_start) / i_d_start if i_d_start > 0.001 else 0
    else:
        adv['Internal_Resistance'] = 0

    # 2. RCV
    tccc = direct_feats.get('TCCC', 0)
    tcvc = direct_feats.get('TCVC', 0)
    adv['RCV'] = tccc / tcvc if tcvc > 0.1 else 0

    # 3. Temperature Rise
    if not raw_discharge_df.empty:
        adv['Temperature_Rise'] = raw_discharge_df['Temperature(C)'].max() - raw_discharge_df['Temperature(C)'].iloc[0]
    else:
        adv['Temperature_Rise'] = 0

    # 4. Skewness
    NOISE_FLOOR = 1e-3
    if is_dirty_file or shape_discharge_df.empty:
        adv['skew_V_discharge'] = 0.0
        adv['skew_T_discharge'] = 0.0
    else:
        v_data = shape_discharge_df['Voltage(V)']
        if v_data.std() < NOISE_FLOOR:
            adv['skew_V_discharge'] = 0.0
        else:
            try:
                adv['skew_V_discharge'] = skew(v_data, nan_policy='omit')
            except Exception:
                adv['skew_V_discharge'] = 0.0

        t_data = shape_discharge_df['Temperature(C)']
        if t_data.std() < NOISE_FLOOR:
            adv['skew_T_discharge'] = 0.0
        else:
            try:
                adv['skew_T_discharge'] = skew(t_data, nan_policy='omit')
            except Exception:
                adv['skew_T_discharge'] = 0.0

    return adv


def _get_adaptive_intervals(
    base_tevi: List[Tuple],
    base_tevd: List[Tuple],
    avg_start_v_chg: float,
    avg_start_v_dis: float,
    uvp: float,
    lvp: float
) -> Tuple[List[Tuple], List[Tuple]]:
    """Adjusts TEVI/TEVD intervals based on actual start voltages."""

    new_tevi = list(base_tevi)
    new_tevd = list(base_tevd)

    # 1. Adapt TEVI (Charge)
    # If starting voltage is too high (e.g. IR rise), shift intervals up
    if new_tevi:
        first_lower = new_tevi[0][0]
        # If start voltage is uncomfortably close to or above the lower bound
        # [Fix] Increased margin from 0.02 to 0.05 to avoid edge instability
        if avg_start_v_chg > (first_lower - 0.05):
            # Strategy: Start from avg_start_v_chg + 0.1 (Increased from 0.05)
            # This moves the anchor point away from the flat plateau edge
            start_v = avg_start_v_chg + 0.1
            # Generate 3 intervals of width 0.2V (or slightly smaller if compressed)
            intervals = []
            curr = start_v
            for _ in range(3):
                next_v = curr + 0.2
                if next_v >= uvp:
                    # If we hit UVP, try to fit a smaller interval or stop
                    if uvp - curr > 0.05:
                        intervals.append((curr, uvp - 0.01))
                    break
                intervals.append((curr, next_v))
                curr = next_v

            if len(intervals) >= 1: # Only replace if we generated valid intervals
                new_tevi = intervals

    # 2. Adapt TEVD (Discharge)
    # If starting voltage is too low (e.g. IR drop), shift intervals down
    if new_tevd:
        first_upper = new_tevd[0][0]
        # If start voltage is below the upper bound
        # [Fix] Increased margin from 0.02 to 0.05
        if avg_start_v_dis < (first_upper + 0.02):
            # Strategy: Start from avg_start_v_dis - 0.1 (Increased from 0.05)
            # This avoids "ghost points" and unstable initial voltage drops
            start_v = avg_start_v_dis - 0.1
            # Generate 3 intervals of width 0.2V downwards
            intervals = []
            curr = start_v
            for _ in range(3):
                next_v = curr - 0.2
                if next_v <= lvp:
                    if curr - lvp > 0.05:
                        intervals.append((curr, lvp + 0.01))
                    break
                intervals.append((curr, next_v))
                curr = next_v

            if len(intervals) >= 1:
                new_tevd = intervals

    return new_tevi, new_tevd


def extract_features_for_cycle(
    cycle_data: Any,
    battery_data: Any,
    charge_slope_intervals: List[Tuple],
    discharge_slope_intervals: List[Tuple],
    tevi_intervals: List[Tuple],
    tevd_intervals: List[Tuple],
    work_condition: int,
    is_dirty_file: bool,
    soc: int,
    is_20_80: bool = False,
    output_dir: Optional[Path] = None,
    calibration_mode: bool = False,  # New: Calibration mode
    static_feats: Optional[Dict[str, float]] = None  # New: Static features
) -> Dict[str, Any]:
    """Main function combining Raw and Shape streams.

    Args:
        calibration_mode: If True, only calculates basic features (Cap, IR) for calibration.
        static_feats: Dictionary of static features (avg_cap, avg_ir) to append.
    """

    cycle_df = pd.DataFrame({
        'Time(s)': cycle_data.time_in_s,
        'Current(A)': cycle_data.current_in_A,
        'Voltage(V)': cycle_data.voltage_in_V,
        'Charge_Capacity(Ah)': cycle_data.charge_capacity_in_Ah,
        'Discharge_Capacity(Ah)': cycle_data.discharge_capacity_in_Ah,
        'Temperature(C)': cycle_data.temperature_in_C
    })
    cycle_num = cycle_data.cycle_number

    # Stream 1: Raw
    raw_charge_df = cycle_df[cycle_df['Current(A)'] > 0].copy()
    raw_discharge_df = cycle_df[cycle_df['Current(A)'] < 0].copy()

    rest_df = pd.DataFrame()
    if not raw_charge_df.empty and not raw_discharge_df.empty:
        c_end = raw_charge_df.index[-1]
        d_start = raw_discharge_df.index[0]
        if d_start > c_end:
            rest_df = cycle_df.loc[c_end+1:d_start-1]

    # Stream 2: Shape (Cleaned) - Used for cleaner slope/skew calculations
    shape_charge_df = _get_shape_stream_charge(raw_charge_df, v_thresh=2.5)
    shape_discharge_df = _get_shape_stream_discharge(raw_discharge_df)

    # Calculation
    feat_direct = _calculate_direct_features(
        raw_charge_df, raw_discharge_df, rest_df,
        cycle_num, battery_data, work_condition,
        cycle_df=cycle_df
    )

    # Advanced features (IR is needed for calibration)
    feat_adv = _calculate_advanced_features(
        shape_charge_df, shape_discharge_df, raw_discharge_df,
        feat_direct, is_dirty_file
    )

    # --- Calibration Mode Return ---
    if calibration_mode:
        return {
            'Discharge_Capacity': feat_direct.get('Discharge_Capacity', 0),
            'Internal_Resistance': feat_adv.get('Internal_Resistance', 0)
        }

    feat_cv = _calculate_cv_features(
        raw_charge_df, battery_data.max_voltage_limit_in_V, work_condition
    )

    # [MODIFIED] Use Shared Tool for IC/DV with dynamic config
    v_range = (3.2, 4.2)

    # Infer Battery Type
    cell_id_str = battery_data.cell_id.upper()
    if 'LFP' in cell_id_str:
        bat_type = 'LFP'
    elif 'NCA' in cell_id_str:
        bat_type = 'NCA'
    elif 'NMC' in cell_id_str:
        bat_type = 'NMC'
    else:
        # Fallback based on WC
        if work_condition == 1:
            bat_type = 'LFP'
        else:
            bat_type = 'NMC'  # Default to NMC for unknown high voltage

    # Set Ranges based on Type/WC
    if bat_type == 'LFP':
        v_range = (3.2, 3.6) # Tighter range for LFP Charge peak
    elif work_condition == 3: # Low voltage NCA/NMC
         v_range = (3.1, 3.9)
    else:
         v_range = (3.5, 4.1) # NCM/NCA Charge Peak range

    nominal_cap = battery_data.charge_protocol[0].get('nominal_capacity_in_Ah', 2.0) if battery_data.charge_protocol else 2.0

    # Base Configuration
    config = {
        'nominal_capacity': nominal_cap,
        'window_length_ic': 21,
        'window_length_dv': 21,
        'peak_height_ic': 0.01,
        'voltage_range_ic': v_range,
        'prominence_ic': 0.01,
        'ic_step_size': 0.005, # Finer step for charge
        'cutoff_voltage': battery_data.max_voltage_limit_in_V, # For NCA check
        'ic_area_config': {'method': 'fixed_width', 'width_v': 0.03},
        'soc': soc, # [NEW] Pass SOC
        'icv_search_range': (0.05, 0.5) # [NEW] Default range
    }

    # [Specific Config Adjustments]
    if bat_type == 'LFP':
        config['icv_search_range'] = (0.02, 0.2) # Widened range from 0.1 to 0.2

    if bat_type == 'NCA':
        # Larger smoothing for NCA DV curve
        config['window_length_dv'] = 51
        config['dv_step_size'] = nominal_cap * 0.01 # Coarser grid

    plot_params = None
    plot_interval=50  # Plot every 10 cycles
    if output_dir:
        plot_params = {
            'cell_id': battery_data.cell_id,
            'cycle_num': cycle_num,
            'output_dir': output_dir,
            'plot_interval': plot_interval
        }

    # [MODIFIED] Use new extract_charge_ic_dv_features
    if not shape_charge_df.empty:
        feat_deriv = extract_charge_ic_dv_features(
            shape_charge_df,
            battery_type=bat_type,
            config=config,
            plot_params=plot_params
        )
    else:
        feat_deriv = extract_charge_ic_dv_features(pd.DataFrame(), bat_type, config=config)

    feat_anchor = _calculate_anchor_features(
        shape_charge_df, shape_discharge_df,
        charge_slope_intervals, discharge_slope_intervals,
        tevi_intervals, tevd_intervals,
        is_dirty_file
    )

    # Merge
    final_features = {
        **feat_direct,
        **feat_cv,
        **feat_deriv,
        **feat_anchor,
        **feat_adv,
        'soc': soc  # Add SOC
    }

    # Add Static Features
    if static_feats:
        final_features.update(static_feats)

    if not shape_charge_df.empty:
        final_features['ICHV(V)'] = shape_charge_df['Voltage(V)'].iloc[0]
    if not shape_discharge_df.empty:
        final_features['LVP(V)'] = shape_discharge_df['Voltage(V)'].iloc[-1]

    return final_features


def process_battery(
    file_path: Path,
    output_dir: Path,
    charge_intervals: List[Tuple],
    discharge_intervals: List[Tuple],
    num_cycles: int
):
    try:
        with open(file_path, 'rb') as f:
            data_dict = pickle.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return

    bat = AttrDict(data_dict)
    if 'cycle_data' in bat and bat['cycle_data']:
        bat.cycle_data = [AttrDict(c) for c in bat.cycle_data]

    cell_id = bat.cell_id

    # is_dirty = any(d_key in cell_id for d_key in DIRTY_FILES)
    # if is_dirty:
    #     print(f"Notice: {cell_id} identified as 'Dirty File'. TEVI/Skewness will be 0.")

    # Work Condition Logic
    wc = 0
    uvp = bat.max_voltage_limit_in_V

    tevi_intervals = [(3.5, 3.7), (3.7, 3.9), (3.9, 4.1)]
    tevd_intervals = [(4.1, 3.9), (3.9, 3.7), (3.7, 3.5)]

    # 20-80 cycling check
    is_20_80 = "20-80" in cell_id
    soc = 100 # Default

    # Try parsing SOC from filename (e.g., 0-100 -> 100, 20-80 -> 80)
    # Matches patterns like "0-100", "20-80"
    soc_match = re.search(r'(\d+)-(\d+)', cell_id)
    if soc_match:
        try:
             soc = int(soc_match.group(2)) # Take the second number as upper SOC limit
        except ValueError:
             pass

    if is_20_80:
        wc = 3
        tevi_intervals = [(3.1, 3.3), (3.3, 3.5), (3.5, 3.8)]
        tevd_intervals = [(3.8, 3.6), (3.6, 3.4), (3.4, 3.2)]
    elif 3.5 <= uvp <= 3.7:
        wc = 1
        # [FIX] LFP Intervals based on data analysis (Max V ~3.28V)
        tevi_intervals = [(3.0, 3.25), (3.25, 3.4), (3.4, 3.55)]
        tevd_intervals = [(3.25, 3.15), (3.15, 3.0), (3.0, 2.5)]
    elif 4.1 <= uvp <= 4.3:
        wc = 2
        tevi_intervals = [(3.5, 3.7), (3.7, 3.9), (3.9, 4.1)]
        tevd_intervals = [(4.1, 3.9), (3.9, 3.7), (3.7, 3.5)]
    elif 3.8 <= uvp <= 4.0:
        wc = 3
        tevi_intervals = [(3.1, 3.3), (3.3, 3.5), (3.5, 3.8)]
        tevd_intervals = [(3.8, 3.6), (3.6, 3.4), (3.4, 3.2)]

    all_feats = []
    cycles = bat.cycle_data[:num_cycles] if num_cycles else bat.cycle_data

    # --- Calibration Phase (First 4 Cycles) ---
    calibration_cycles = cycles[:4]
    calib_caps = []
    calib_irs = []

    # [NEW] Collect Start Voltages for Adaptive Intervals
    calib_start_v_chg = []
    calib_start_v_dis = []

    for cycle in calibration_cycles:
        # Extract start voltages directly from raw data to be safe
        try:
             # Construct minimal DF for cleaning
             df_temp = pd.DataFrame({
                 'Time(s)': cycle.time_in_s,
                 'Voltage(V)': cycle.voltage_in_V,
                 'Current(A)': cycle.current_in_A
             })

             raw_chg = df_temp[df_temp['Current(A)'] > 0.01].copy()
             raw_dis = df_temp[df_temp['Current(A)'] < -0.005].copy() # Relaxed threshold

             # Apply cleaning to remove ghost points (essential for high rate discharge)
             clean_chg = _get_shape_stream_charge(raw_chg, v_thresh=2.0)
             clean_dis = _get_shape_stream_discharge(raw_dis)

             if not clean_chg.empty:
                 calib_start_v_chg.append(clean_chg['Voltage(V)'].iloc[0])
             if not clean_dis.empty:
                 calib_start_v_dis.append(clean_dis['Voltage(V)'].iloc[0])
        except Exception:
            pass

        try:
            calib_res = extract_features_for_cycle(
                cycle, bat,
                charge_intervals, discharge_intervals,
                tevi_intervals, tevd_intervals,
                wc,
                False, # is_dirty_file
                soc=soc,
                is_20_80=is_20_80,
                calibration_mode=True  # Enable calibration mode
            )
            if calib_res['Discharge_Capacity'] > 0:
                calib_caps.append(calib_res['Discharge_Capacity'])
            if calib_res['Internal_Resistance'] > 0:
                calib_irs.append(calib_res['Internal_Resistance'])
        except Exception:
            continue

    avg_cap_first4 = np.mean(calib_caps) if calib_caps else 0.0
    avg_ir_first4 = np.mean(calib_irs) if calib_irs else 0.0

    # [NEW] Determine Adaptive Intervals
    avg_start_chg = np.mean(calib_start_v_chg) if calib_start_v_chg else 0.0
    avg_start_dis = np.mean(calib_start_v_dis) if calib_start_v_dis else 0.0

    if avg_start_chg > 0 and avg_start_dis > 0:
        tevi_intervals, tevd_intervals = _get_adaptive_intervals(
            tevi_intervals, tevd_intervals,
            avg_start_chg, avg_start_dis,
            bat.max_voltage_limit_in_V,
            bat.min_voltage_limit_in_V
        )

    static_feats = {
        'Avg_Cap_First4': avg_cap_first4,
        'Avg_IR_First4': avg_ir_first4
    }

    # --- Analysis Phase (From Cycle 5 onwards) ---
    analysis_cycles = cycles[4:]

    for cycle in tqdm(analysis_cycles, desc=f"Proc {cell_id}", leave=False):
        try:
            feats = extract_features_for_cycle(
                cycle, bat,
                charge_intervals, discharge_intervals,
                tevi_intervals, tevd_intervals,
                wc,
                False, # is_dirty_file (fixed via tail truncation)
                soc=soc, # Pass SOC
                is_20_80=is_20_80, # Pass new arg
                output_dir=output_dir,
                calibration_mode=False,
                static_feats=static_feats # Pass static feats
            )
            all_feats.append(feats)
        except Exception:
            continue

    if all_feats:
        df_out = pd.DataFrame(all_feats)

        preferred_order = [
            'Cycle_Number', 'Discharge_Capacity', 'Internal_Resistance', 'soc',
            'Avg_Cap_First4', 'Avg_IR_First4',
            'TEVI_1', 'TEVI_2', 'TEVI_3', 'TEVD_1', 'TEVD_2', 'TEVD_3',
            'skew_V_discharge', 'skew_T_discharge'
        ]
        cols = preferred_order + [c for c in df_out.columns if c not in preferred_order]
        cols = [c for c in cols if c in df_out.columns]

        df_out = df_out[cols]
        out_path = output_dir / f"{cell_id}.csv"
        df_out.to_csv(out_path, index=False)
        print(f"Saved {out_path}")


def main():
    input_dir = Path('F:/datasets/battery/SNL')
    output_dir = project_root / 'results' / 'SNL'
    output_dir.mkdir(parents=True, exist_ok=True)

    c_intervals = [(0.1, 0.3), (0.1, 0.5), (0.1, 0.8)]
    d_intervals = [(0.1, 0.3), (0.1, 0.5), (0.1, 0.8)]
    num_cycles_to_extract = 100

    if not input_dir.exists():
         print(f"Data directory not found: {input_dir}")
         return

    pkl_files = list(input_dir.glob('*.pkl'))
    if not pkl_files:
        print("No files found.")
        return

    for pkl in pkl_files:
        process_battery(pkl, output_dir, c_intervals, d_intervals, num_cycles_to_extract)

if __name__ == '__main__':
    main()

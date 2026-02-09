import pickle
import warnings
import sys
import re
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

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


"""
# Anomaly Report

## 1. Core Conclusion
After in-depth diagnosis and comparative analysis of the raw `.pkl` files, it was confirmed that the "abnormal" phenomena of `UVP_time`, `ICHV`, and `TCCC` in specific cycles of the **15H, 16R, 17C, 18H** files reported by users are **not code calculation logic errors**, but are due to the inclusion of **0-100% full capacity check cycles (Capacity Check / RPT Cycles)** during the experiment.

The physical characteristics of these specific cycles are completely different from the regular 50-100% aging cycles, leading to significant jumps in feature values (e.g., doubling of charging time, decrease in starting voltage).

## 2. Detailed Data Evidence

### 2.1 Comparative Analysis (Example: 15H)

We selected "abnormal" Cycle 9 and "normal" Cycle 10 for comparison:

| Feature | Cycle 9 (Anomaly) | Cycle 10 (Normal) | Physical Meaning Interpretation |
| :--- | :--- | :--- | :--- |
| **ICHV (Start Voltage)** | **3.306 V** | 3.685 V | Cycle 9 starts charging from empty (0% SOC); Cycle 10 starts from half empty (50% SOC). |
| **UVP_time (Charge Duration)** | **17451 s** (~4.85h) | 9406 s (~2.61h) | At 0.2C rate, 0-100% takes 5 hours; 50-100% takes 2.5 hours. Data fully conforms to physical laws. |
| **Pre-Charge Gap** | 40.00 s | 0.00 s | There may be a rest step before RPT cycles. |
| **Judgment Result** | **0-100% Full Charge/Discharge Cycle** | **50-100% Cycle** | Belongs to regular capacity calibration in the experimental protocol. |

### 2.2 Confirmation of Anomalies in Other Files

All reported anomalies exhibit the same **0-100% full cycle characteristics**:

*   **15H (Cycle 9, 18, 23, 36, 45, 59, 92)**
    *   **Phenomenon**: Charge duration is **16000s - 17500s** (approx. 4.5-5 hours).
    *   **Start Voltage**: All are **around 3.3V** (0% SOC).
    *   **Conclusion**: These are regular RPT cycles.

*   **16R (Cycle 41, 82)**
    *   **Phenomenon**: Charge duration approx. **17500s**.
    *   **Start Voltage**: **3.33V**.
    *   **Conclusion**: Full capacity calibration every 41 cycles.

*   **17C (Cycle 41, 82)**
    *   **Phenomenon**: Charge duration approx. **18500s**.
    *   **Start Voltage**: **3.46V** (higher voltage at low temperature, as expected).
    *   **Conclusion**: Full capacity calibration every 41 cycles.

*   **18H (Cycle 16, 31, 45, 66, 80)**
    *   **Phenomenon**: Charge duration approx. **16500s - 17500s**.
    *   **Start Voltage**: **3.30V - 3.34V**.
    *   **Conclusion**: Regular RPT cycles.

## 3. Root Cause

1.  **Mixed operating conditions not separated**: The dataset contains two distinct operating conditions (50-100% aging cycles and 0-100% calibration cycles).
2.  **Misunderstanding of feature consistency**: The code currently extracts features for all cycles equally. When an RPT cycle is encountered, its physical duration is naturally twice that of a regular cycle. This manifests numerically as an "anomaly" (jump), but it is actually a real physical response.
3.  **SOC label conflict**: The current feature extraction logic assigns a `soc=50` label to all rows based on the filename (e.g., `50-100`). For these RPT cycles, although they are in the `50-100` file, they are actually `soc=100` (0-100 range) data.

## 4. Next Steps

Since this is not a code bug but a data nature issue, it is suggested to discuss the following solutions:

1.  **Solution A (Cleaning)**: Identify and **exclude** these RPT cycles during the feature extraction stage, keeping only the cycles that match the filename (50-100%) to ensure distribution consistency of the training data.
2.  **Solution B (Marking)**: Add a new column `is_rpt` or dynamically calculate `soc_range` to mark these cycles, for downstream models to decide whether to use them.
3.  **Solution C (Correction)**: If the data must be retained, the physical reality of these values needs to be accepted and not treated as "errors" to be forcibly truncated.
"""


class AttrDict(dict):
    """A dictionary that allows attribute-style access."""
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def _calculate_direct_features(
    cycle_df: pd.DataFrame,
    charge_df: pd.DataFrame,
    discharge_df: pd.DataFrame,
    cycle_num: int,
    battery_data: AttrDict
) -> Dict[str, Any]:
    """Calculates direct features, including energy integration and CV dynamics."""
    features = {}

    features['Cycle_Number'] = cycle_num

    # Define cycle start time for relative time calculations (Issue 4)
    cycle_start_time = cycle_df['Time(s)'].iloc[0] if not cycle_df.empty else 0

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
        e_charge_ws = trapezoid(y=p_charge, x=charge_df['Time(s)'])
        features['Charge_Energy(Wh)'] = e_charge_ws / 3600.0
    else:
        features['Charge_Energy(Wh)'] = 0

    if not discharge_df.empty:
        p_discharge = discharge_df['Voltage(V)'] * discharge_df['Current(A)'].abs()
        e_discharge_ws = trapezoid(y=p_discharge, x=discharge_df['Time(s)'])
        features['Discharge_Energy(Wh)'] = e_discharge_ws / 3600.0
    else:
        features['Discharge_Energy(Wh)'] = 0

    features['Coulombic_Efficiency'] = (dis_cap / chg_cap) if chg_cap > 0 else 0
    features['Energy_Efficiency'] = (features['Discharge_Energy(Wh)'] / features['Charge_Energy(Wh)']) if features['Charge_Energy(Wh)'] > 0 else 0

    if not charge_df.empty and not discharge_df.empty:
        time_charge_end = charge_df['Time(s)'].iloc[-1]
        time_discharge_start = discharge_df['Time(s)'].iloc[0]
        features['Rest_Time(s)'] = time_discharge_start - time_charge_end if time_discharge_start > time_charge_end else 0
    else:
        features['Rest_Time(s)'] = 0

    features['charge_c_rate'] = battery_data.charge_protocol[0].rate_in_C if hasattr(battery_data, 'charge_protocol') and battery_data.charge_protocol else 0.0
    features['discharge_c_rate'] = battery_data.discharge_protocol[0].rate_in_C if hasattr(battery_data, 'discharge_protocol') and battery_data.discharge_protocol else 0.0

    features['CV_Current_Tau'] = 0
    if not charge_df.empty:
        features['ICHV(V)'] = charge_df['Voltage(V)'].iloc[0]
        # Calculate UVP time relative to charge start (Fix for massive initial gaps/rests)
        features['UVP_time(s)'] = charge_df['Time(s)'].iloc[-1] - charge_df['Time(s)'].iloc[0]
        features['UVP(V)'] = battery_data.max_voltage_limit_in_V

        v_upper_limit = battery_data.max_voltage_limit_in_V
        charge_voltage = charge_df['Voltage(V)']

        if charge_voltage.max() >= (v_upper_limit - 0.01):
            cv_mask = charge_voltage >= (v_upper_limit - 0.01)
            cv_df = charge_df[cv_mask]
            if not cv_df.empty:
                time_at_v_limit = cv_df['Time(s)'].iloc[0]
                features['TCCC(s)'] = time_at_v_limit - charge_df['Time(s)'].iloc[0]
                features['TCVC(s)'] = charge_df['Time(s)'].iloc[-1] - time_at_v_limit

                cv_current = cv_df['Current(A)'].values
                cv_time = cv_df['Time(s)'].values
                valid_mask = cv_current > 0.001
                if np.sum(valid_mask) > 10:
                    features['CV_Current_Tau'] = fit_cv_decay(cv_time[valid_mask], cv_current[valid_mask])
            else:
                features['TCCC(s)'] = charge_df['Time(s)'].iloc[-1] - charge_df['Time(s)'].iloc[0]
                features['TCVC(s)'] = 0
        else:
            features['TCCC(s)'] = charge_df['Time(s)'].iloc[-1] - charge_df['Time(s)'].iloc[0]
            features['TCVC(s)'] = 0
    else:
        features.update({'ICHV(V)': 0, 'UVP_time(s)': 0, 'UVP(V)': battery_data.max_voltage_limit_in_V, 'TCCC(s)': 0, 'TCVC(s)': 0})

    if not discharge_df.empty:
        features['IDV(V)'] = discharge_df['Voltage(V)'].iloc[0]
        # Calculate LVP time relative to discharge start
        features['LVP_time(s)'] = discharge_df['Time(s)'].iloc[-1] - discharge_df['Time(s)'].iloc[0]
        features['LVP(V)'] = battery_data.min_voltage_limit_in_V
        features['var_I_discharge'] = discharge_df['Current(A)'].var()
        features['var_V_discharge'] = discharge_df['Voltage(V)'].var()
        features['median_V_discharge(V)'] = discharge_df['Voltage(V)'].median()
        features['total_discharge_time(s)'] = discharge_df['Time(s)'].iloc[-1] - discharge_df['Time(s)'].iloc[0]
        # features['MAT_discharge(C)'] = discharge_df['Cell_Temperature(C)'].max()
        # features['MET_discharge(s)'] = discharge_df['Cell_Temperature(C)'].mean()
    else:
        features.update({'IDV(V)': 0, 'LVP_time(s)': 0, 'var_I_discharge': 0, 'var_V_discharge': 0, 'median_V_discharge(V)': 0, 'total_discharge_time(s)': 0, 'LVP(V)': battery_data.min_voltage_limit_in_V}) # , 'MAT_discharge': 0, 'MET_discharge': 0})

    return features

def _calculate_advanced_features(charge_df: pd.DataFrame, discharge_df: pd.DataFrame, features: Dict[str, Any], rest_between_cd_df: pd.DataFrame) -> Dict[str, Any]:
    adv_features = {}
    adv_features['Internal_Resistance(Ohm)'] = 0.0

    if not discharge_df.empty and discharge_df.shape[0] > 0:
        v_discharge_start = discharge_df['Voltage(V)'].iloc[0]
        i_discharge_start = abs(discharge_df['Current(A)'].iloc[0])
        v_start_for_calc = 0.0

        if not rest_between_cd_df.empty:
            v_start_for_calc = rest_between_cd_df['Voltage(V)'].iloc[-1]
        elif not charge_df.empty:
            v_start_for_calc = charge_df['Voltage(V)'].iloc[-1]

        if i_discharge_start > 1e-3 and v_start_for_calc > 0:
            delta_v = v_start_for_calc - v_discharge_start
            if delta_v > 0:
                adv_features['Internal_Resistance(Ohm)'] = delta_v / i_discharge_start

    adv_features['RCV(V)'] = features.get('TCCC', 0) / features['TCVC(s)'] if features.get('TCVC', 0) > 0 else 0
    if not discharge_df.empty:
        adv_features['skew_V_discharge'] = skew(discharge_df['Voltage(V)'])
        # adv_features['Temperature_Rise(C)'] = discharge_df['Cell_Temperature(C)'].max() - discharge_df['Cell_Temperature(C)'].iloc[0] if discharge_df.shape[0] > 1 else 0
        # adv_features['skew_T_discharge'] = skew(discharge_df['Cell_Temperature(C)'])
    else:
        adv_features['skew_V_discharge'] = 0
        # adv_features['Temperature_Rise(C)'] = 0
        # adv_features['skew_T_discharge'] = 0
    return adv_features

def _calculate_anchor_features(charge_df: pd.DataFrame, discharge_df: pd.DataFrame, charge_slope_intervals: List, discharge_slope_intervals: List, tevi_intervals: List, tevd_intervals: List) -> Dict[str, Any]:
    anchor_features = {}

    def get_voltage_at_relative_time(df: pd.DataFrame, relative_time: float, time_series_values: np.ndarray, start_time: float) -> float:
        if df.empty: return 0.0
        absolute_time = start_time + relative_time
        idx = np.searchsorted(time_series_values, absolute_time, side='left')
        if idx == 0: return df['Voltage(V)'].iloc[0]
        if idx == len(time_series_values): return df['Voltage(V)'].iloc[-1]
        left_time = time_series_values[idx - 1]
        right_time = time_series_values[idx]
        return df['Voltage(V)'].iloc[idx-1] if (absolute_time - left_time) < (right_time - absolute_time) else df['Voltage(V)'].iloc[idx]

    charge_duration = (charge_df['Time(s)'].iloc[-1] - charge_df['Time(s)'].iloc[0]) if not charge_df.empty else 0
    charge_time_series = charge_df['Time(s)'].values if not charge_df.empty else np.array([])
    charge_start_time = charge_df['Time(s)'].iloc[0] if not charge_df.empty else 0
    for i, (ratio_start, ratio_end) in enumerate(charge_slope_intervals):
        key = f'charge_slope_{i+1}'
        if not charge_df.empty and charge_duration > 1e-6:
            v_start = get_voltage_at_relative_time(charge_df, charge_duration * ratio_start, charge_time_series, charge_start_time)
            v_end = get_voltage_at_relative_time(charge_df, charge_duration * ratio_end, charge_time_series, charge_start_time)
            time_delta = charge_duration * (ratio_end - ratio_start)
            anchor_features[key] = (v_end - v_start) / time_delta if time_delta != 0 else 0
        else:
            anchor_features[key] = 0

    discharge_duration = (discharge_df['Time(s)'].iloc[-1] - discharge_df['Time(s)'].iloc[0]) if not discharge_df.empty else 0
    discharge_time_series = discharge_df['Time(s)'].values if not discharge_df.empty else np.array([])
    discharge_start_time = discharge_df['Time(s)'].iloc[0] if not discharge_df.empty else 0
    for i, (ratio_start, ratio_end) in enumerate(discharge_slope_intervals):
        key = f'discharge_slope_{i+1}'
        if not discharge_df.empty and discharge_duration > 1e-6:
            v_start = get_voltage_at_relative_time(discharge_df, discharge_duration * ratio_start, discharge_time_series, discharge_start_time)
            v_end = get_voltage_at_relative_time(discharge_df, discharge_duration * ratio_end, discharge_time_series, discharge_start_time)
            time_delta = discharge_duration * (ratio_end - ratio_start)
            anchor_features[key] = (v_end - v_start) / time_delta if time_delta != 0 else 0
        else:
            anchor_features[key] = 0

    def get_time_for_voltage(df, voltage, direction):
        if df.empty: return None
        target_rows = df[df['Voltage(V)'] >= voltage] if direction == 'charge' else df[df['Voltage(V)'] <= voltage]
        return target_rows['Time(s)'].iloc[0] if not target_rows.empty else None

    for i, (v_start, v_end) in enumerate(tevi_intervals):
        t_start = get_time_for_voltage(charge_df, v_start, 'charge')
        t_end = get_time_for_voltage(charge_df, v_end, 'charge')
        anchor_features[f'TEVI_{i+1}'] = t_end - t_start if t_start is not None and t_end is not None and t_end > t_start else 0

    for i, (v_start, v_end) in enumerate(tevd_intervals):
        t_start = get_time_for_voltage(discharge_df, v_start, 'discharge')
        t_end = get_time_for_voltage(discharge_df, v_end, 'discharge')
        anchor_features[f'TEVD_{i+1}'] = t_end - t_start if t_start is not None and t_end is not None and t_end > t_start else 0
    return anchor_features

def _calculate_personalized_features(cycle_df: pd.DataFrame, cell_id: str) -> Dict[str, Any]:
    return {}

def extract_features_for_cycle(cycle_data: AttrDict, battery_data: AttrDict, charge_slope_intervals: List, discharge_slope_intervals: List, tevi_intervals: List, tevd_intervals: List, output_dir: Optional[Path] = None) -> Dict[str, Any]:
    cycle_df = pd.DataFrame({'Time(s)': cycle_data.time_in_s, 'Current(A)': cycle_data.current_in_A, 'Voltage(V)': cycle_data.voltage_in_V, 'Charge_Capacity(Ah)': cycle_data.charge_capacity_in_Ah, 'Discharge_Capacity(Ah)': cycle_data.discharge_capacity_in_Ah, 'Cell_Temperature(C)': cycle_data.temperature_in_C})
    cycle_num = cycle_data.cycle_number

    current_threshold = 1e-3
    v_min_limit = 2.5
    prelim_charge_df = cycle_df[(cycle_df['Current(A)'] > current_threshold) & (cycle_df['Voltage(V)'] > v_min_limit)].copy()
    charge_df = pd.DataFrame()
    if not prelim_charge_df.empty and prelim_charge_df.shape[0] >= 50:
        nominal_charge_current = prelim_charge_df['Current(A)'].max()
        if nominal_charge_current > 0.1:
            real_current_threshold = nominal_charge_current * 0.90
            is_stable_current = (prelim_charge_df['Current(A)'] >= real_current_threshold)
            rolling_sum = is_stable_current.rolling(window=50, min_periods=50).sum()
            current_stable_end_ilocs = (rolling_sum == 50).to_numpy().nonzero()[0]
            voltages = prelim_charge_df['Voltage(V)'].values
            times = prelim_charge_df['Time(s)'].values
            for end_iloc in current_stable_end_ilocs:
                start_iloc = end_iloc - 49
                if (times[end_iloc] - times[start_iloc]) > 1e-3 and (voltages[end_iloc] - voltages[start_iloc]) / (times[end_iloc] - times[start_iloc]) > 1e-5:
                    subset = prelim_charge_df.iloc[start_iloc:]
                    # Fix: Truncate at large time jumps (Issue 3/4 Fix)
                    time_diff = subset['Time(s)'].diff()
                    gap_indices = time_diff[time_diff > 30.0].index

                    if not gap_indices.empty:
                        # Truncate before the first gap
                        first_gap = gap_indices[0]
                        # Need to find integer location relative to subset start to slice properly
                        # gap_indices contains original index labels.
                        # subset.index.get_loc(first_gap) gives integer pos of the gap row
                        cutoff_pos = subset.index.get_loc(first_gap)
                        charge_df = subset.iloc[:cutoff_pos].copy()
                    else:
                        charge_df = subset.copy()

                    # Fix 2: Truncate if voltage drops significantly (Handle noise bridging)
                    # If current noise > threshold exists during rest, Time Gap check won't work.
                    # Use Voltage Drop > 0.05V from running max as a secondary cutoff.
                    if not charge_df.empty:
                        v_vals = charge_df['Voltage(V)'].values
                        running_max = np.maximum.accumulate(v_vals)
                        # Find first point where voltage drops more than 0.05V below peak so far
                        drop_indices = np.where((running_max - v_vals) > 0.05)[0]
                        if len(drop_indices) > 0:
                            first_drop = drop_indices[0]
                            charge_df = charge_df.iloc[:first_drop].copy()

                    break

    # Debug for Issue 3
    if charge_df.empty:
        print(f"DEBUG [Cycle {cycle_num}]: charge_df is empty.")
        print(f"  - Prelim size: {len(prelim_charge_df)}")
        if not prelim_charge_df.empty:
            print(f"  - Max Current: {prelim_charge_df['Current(A)'].max()}")

    prelim_discharge_df = cycle_df[(cycle_df['Current(A)'] < -current_threshold) & (cycle_df['Voltage(V)'] > v_min_limit)].copy()
    discharge_df = pd.DataFrame()
    if not prelim_discharge_df.empty and prelim_discharge_df.shape[0] >= 10:
        nominal_discharge_current = prelim_discharge_df['Current(A)'].min()
        if nominal_discharge_current < -0.1:
            real_current_threshold = nominal_discharge_current * 0.90
            is_stable_current = (prelim_discharge_df['Current(A)'] <= real_current_threshold)
            rolling_sum = is_stable_current.rolling(window=50, min_periods=50).sum()
            current_stable_end_ilocs = (rolling_sum == 50).to_numpy().nonzero()[0]
            voltages = prelim_discharge_df['Voltage(V)'].values
            for end_iloc in current_stable_end_ilocs:
                start_iloc = end_iloc - 49
                if voltages[end_iloc] <= voltages[start_iloc]:
                    subset = prelim_discharge_df.iloc[start_iloc:]
                    # Fix: Truncate at large time jumps
                    time_diff = subset['Time(s)'].diff()
                    gap_indices = time_diff[time_diff > 30.0].index

                    if not gap_indices.empty:
                        first_gap = gap_indices[0]
                        cutoff_pos = subset.index.get_loc(first_gap)
                        discharge_df = subset.iloc[:cutoff_pos].copy()
                    else:
                        discharge_df = subset.copy()
                    break

    # Debug for Issue 3
    if discharge_df.empty:
        print(f"DEBUG [Cycle {cycle_num}]: discharge_df is empty.")
        print(f"  - Prelim size: {len(prelim_discharge_df)}")
        if not prelim_discharge_df.empty:
            print(f"  - Min Current: {prelim_discharge_df['Current(A)'].min()}")

    if not discharge_df.empty:
        cutoff_voltage = battery_data.min_voltage_limit_in_V
        cutoff_indices = discharge_df.index[discharge_df['Voltage(V)'] <= cutoff_voltage]
        if not cutoff_indices.empty:
            discharge_df = discharge_df.loc[:cutoff_indices[0]]

    rest_between_cd_df = pd.DataFrame()
    if not charge_df.empty and not discharge_df.empty:
        time_charge_end = charge_df['Time(s)'].iloc[-1]
        time_discharge_start = discharge_df['Time(s)'].iloc[0]
        if time_discharge_start > time_charge_end:
            rest_between_cd_df = cycle_df[(cycle_df['Time(s)'] > time_charge_end) & (cycle_df['Time(s)'] < time_discharge_start) & (cycle_df['Current(A)'].abs() <= current_threshold)].copy()

    direct_features = _calculate_direct_features(cycle_df, charge_df, discharge_df, cycle_num, battery_data)

    # --- Cell-Specific Configuration for MICH_EXP ---
    cell_id = battery_data.get('cell_id', 'Unknown')

    # 1. Peak Mode Logic (Issue 1)
    # 01/02/03 -> Mode 2, Others -> Mode 1
    special_cells_mode_2 = ['01', '02', '03']
    if any(x in cell_id for x in special_cells_mode_2):
        peak_mode = 2
    else:
        peak_mode = 1

    # 2. ICP Area Range Logic (Issue 2)
    ic_area_range = (3.4, 3.6) # Default

    high_range_cells = ['13R', '14C', '15H', '16R', '18H']
    mid_range_cells = ['01', '02', '03']

    if any(x in cell_id for x in high_range_cells):
        ic_area_range = (3.8, 3.9)
    elif any(x in cell_id for x in mid_range_cells):
        ic_area_range = (3.5, 3.7)

    # 4. Force Zero Logic (Issue 4)
    # ICP_FWHM always 0 for MICH_EXP due to low temp plateau
    force_fwhm_zero = True

    # ICV valid only for 01/02/03, else 0
    force_icv_zero = not any(x in cell_id for x in ['01', '02', '03'])

    # 5. Extract Temperature from Cell ID
    # Pattern: NMC_XXC -> XX is temperature
    temp_match = re.search(r'NMC_(\d+)C', cell_id)
    temperature_val = int(temp_match.group(1)) if temp_match else 0

    # 6. Extract SOC window from Cell ID
    # Pattern: 0-100 -> 100, 50-100 -> 50
    soc_val = 100 # Default

    # Specific overrides for known 50-100% datasets that might be mislabeled or default to 100
    if any(x in cell_id for x in ['13R', '14C', '15H', '16R', '17C', '18H']):
        soc_val = 50
    elif '0-100' in cell_id:
        soc_val = 100
    elif '50-100' in cell_id:
        soc_val = 50

    nominal_cap = battery_data.get('nominal_capacity_in_Ah', 2.0)
    ncm_config = {
        'peak_mode': peak_mode,
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
        'icv_search_offset_upper': 0.2,
        'plot_interval': 50,

        # MICH_EXP Specific Overrides
        'ic_area_voltage_range': ic_area_range,
        'force_icp_fwhm_zero': force_fwhm_zero,
        'force_icv_zero': force_icv_zero
    }

    plot_params = None
    if output_dir:
        plot_params = {
            'cell_id': battery_data.cell_id,
            'cycle_num': cycle_num,
            'output_dir': output_dir
        }

    derivative_features = extract_ic_dv_features(discharge_df, config=ncm_config, plot_params=plot_params)

    # 7. Handle soc=50 logic (Issue 2)
    # ICP/ICV/DVV/DVP and positions all set to 0
    if soc_val == 50:
        for k in ['ICP', 'ICPL_V', 'ICV', 'ICVL_V', 'DVP', 'DVPL_V', 'DVV', 'DVVL_V']:
            if k in derivative_features:
                derivative_features[k] = 0.0

    advanced_features = _calculate_advanced_features(charge_df, discharge_df, direct_features, rest_between_cd_df)
    anchor_features = _calculate_anchor_features(charge_df, discharge_df, charge_slope_intervals, discharge_slope_intervals, tevi_intervals, tevd_intervals)
    personalized_features = _calculate_personalized_features(cycle_df, battery_data.cell_id)

    # Add temperature and SOC
    final_features = {**direct_features, **derivative_features, **advanced_features, **anchor_features, **personalized_features}
    final_features['temperature(C)'] = temperature_val
    final_features['soc'] = soc_val

    return final_features

def process_battery(file_path: Path, output_dir: Path, charge_slope_intervals: List, discharge_slope_intervals: List, tevi_intervals: List, tevd_intervals: List, num_cycles: int = None):
    try:
        with open(file_path, 'rb') as f:
            data_dict = pickle.load(f)
    except Exception as e:
        print(f"Error loading pickle file {file_path}: {e}")
        return

    battery_data = AttrDict(data_dict)
    cell_id = battery_data.get('cell_id', 'Unknown_Cell')

    # Special handling for high voltage cells (4.2V - 3.7V range)
    high_voltage_cells = ['13R', '14C', '15H', '16R', '18H']
    if any(x in cell_id for x in high_voltage_cells):
        # Adjust intervals to fit the 3.7V - 4.2V range
        # Default TEVI: [(3.5, 3.7), (3.7, 3.9), (3.9, 4.1)]
        tevi_intervals = [(3.7, 3.85), (3.85, 4.00), (4.00, 4.15)]

        # Default TEVD: [(4.1, 3.9), (3.9, 3.7), (3.7, 3.5)]
        tevd_intervals = [(4.15, 4.00), (4.00, 3.85), (3.85, 3.7)]

    if 'cycle_data' in battery_data and battery_data.cycle_data is not None:
        battery_data.cycle_data = [AttrDict(c) for c in battery_data.cycle_data]
    else:
        print(f"Warning: 'cycle_data' is missing or None in {file_path}")
        return

    if 'charge_protocol' in battery_data and battery_data.charge_protocol is not None:
        battery_data.charge_protocol = [AttrDict(p) for p in battery_data.charge_protocol]
    if 'discharge_protocol' in battery_data and battery_data.discharge_protocol is not None:
        battery_data.discharge_protocol = [AttrDict(p) for p in battery_data.discharge_protocol]

    cycles_to_process = battery_data.cycle_data[:num_cycles] if num_cycles is not None and num_cycles > 0 else battery_data.cycle_data

    all_cycle_features = []
    for cycle_data in tqdm(cycles_to_process, desc=f"Extracting features for {cell_id}"):
        if not hasattr(cycle_data, 'time_in_s') or not cycle_data.time_in_s:
            continue
        try:
            features = extract_features_for_cycle(cycle_data, battery_data, charge_slope_intervals, discharge_slope_intervals, tevi_intervals, tevd_intervals, output_dir=output_dir)

            # --- RPT Filtering Logic (Added based on Anomaly Report) ---
            # Issue: MICH_EXP 50-100% datasets contain periodic 0-100% RPT cycles.
            # These cause feature jumps (e.g., UVP_time doubles, ICHV drops to ~3.3V).
            # We filter them out to ensure data consistency for the 50-100% label.
            # Threshold: ICHV < 3.55V indicates a start from 0% SOC (RPT), whereas 50% SOC starts > 3.6V.
            if features.get('soc') == 50 and features.get('ICHV', 100) < 3.55:
                # 3.55V is a safe threshold (RPT starts ~3.30-3.46V, Normal 50% starts ~3.68V)
                # We log this to stdout so the user knows why cycles are missing.
                print(f"  [Skipping Cycle {cycle_data.cycle_number}] RPT Detected (ICHV={features.get('ICHV'):.3f}V < 3.55V) in 50-100% dataset.")
                continue

            all_cycle_features.append(features)
        except Exception as e:
            print(f"Error extracting features for cycle {cycle_data.cycle_number}: {e}")
            continue

    if not all_cycle_features:
        print(f"Warning: No features extracted for {cell_id}")
        return

    features_df = pd.DataFrame(all_cycle_features)
    # Removed: 'MAT_charge', 'MET_charge', 'MAT_discharge', 'MET_discharge', 'Temperature_Rise', 'skew_T_discharge'
    # Added: 'temperature', 'soc'
    ordered_cols = ['Cycle_Number', 'Discharge_Capacity', 'Charge_Capacity', 'Discharge_Energy', 'Charge_Energy', 'Coulombic_Efficiency', 'Energy_Efficiency', 'Rest_Time', 'charge_c_rate', 'discharge_c_rate', 'ICHV', 'UVP_time', 'TCCC', 'TCVC', 'CV_Current_Tau', 'UVP', 'IDV', 'LVP_time', 'var_I_discharge', 'var_V_discharge', 'median_V_discharge', 'total_discharge_time', 'LVP', 'temperature', 'soc', 'ICP', 'ICPL_V', 'ICP_FWHM', 'ICP_Area', 'ICV', 'ICVL_V', 'DVP', 'DVPL_V', 'DVP_FWHM', 'DVP_Area', 'DVV', 'DVVL_V', 'DTP', 'DTPL_V', 'DTV', 'DTVL_V', 'Internal_Resistance', 'RCV', 'skew_V_discharge', 'charge_slope_1', 'charge_slope_2', 'charge_slope_3', 'discharge_slope_1', 'discharge_slope_2', 'discharge_slope_3', 'TEVI_1', 'TEVI_2', 'TEVI_3', 'TEVD_1', 'TEVD_2', 'TEVD_3']
    final_ordered_cols = [col for col in ordered_cols if col in features_df.columns]
    extra_cols = [col for col in features_df.columns if col not in final_ordered_cols]
    features_df = features_df[final_ordered_cols + extra_cols]
    output_file = output_dir / f"{cell_id}.csv"
    features_df.to_csv(output_file, index=False)
    print(f"Features for {cell_id} saved to {output_file}")

def main():
    processed_data_dir = Path('F:/datasets/battery/MICH_EXP')
    output_dir = project_root / 'results' / 'MICH_EXP'
    output_dir.mkdir(parents=True, exist_ok=True)
    charge_slope_intervals = [(0.10, 0.30), (0.10, 0.50), (0.10, 0.80)]
    discharge_slope_intervals = [(0.10, 0.30), (0.10, 0.50), (0.10, 0.80)]
    tevi_intervals = [(3.5, 3.7), (3.7, 3.9), (3.9, 4.1)]
    tevd_intervals = [(4.1, 3.9), (3.9, 3.7), (3.7, 3.5)]
    num_cycles_to_extract = 100
    pkl_files = list(processed_data_dir.glob('*.pkl'))
    if not pkl_files:
        print(f"Error: No .pkl files found in '{processed_data_dir}'.")
        return
    for file_path in pkl_files:
        process_battery(file_path, output_dir, charge_slope_intervals, discharge_slope_intervals, tevi_intervals, tevd_intervals, num_cycles=num_cycles_to_extract)

if __name__ == '__main__':
    main()

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
from scipy.signal import find_peaks, savgol_filter, peak_widths, medfilt
from scipy.optimize import curve_fit
from tqdm import tqdm


project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))
from src.utils.feature_tools import extract_ic_dv_features

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# --- Configuration Constants ---
# MATR dataset typically has nominal capacity around 1.1Ah (A123 cells)


def _get_interp_val(arr: np.ndarray, idx_float: float) -> float:
    """
    Helper function: Linearly interpolate value in array at floating point index.
    """
    low = int(np.floor(idx_float))
    high = int(np.ceil(idx_float))
    
    # Boundary checks
    low = max(0, min(low, len(arr)-1))
    high = max(0, min(high, len(arr)-1))
    
    if low == high:
        return arr[low]
    
    frac = idx_float - low
    return arr[low] * (1 - frac) + arr[high] * frac


def _fit_cv_decay(time_series: np.ndarray, current_series: np.ndarray) -> float:
    """
    Fits an exponential decay model to the CV phase current.
    Model: I(t) = a * exp(-t / tau) + c
    Safe version with overflow protection.
    """
    def exponential_decay(t, a, tau, c):
        # 1. Avoid tau being 0 (prevent division by zero error)
        if tau == 0:
            tau = 1e-10
            
        # 2. Calculate exponential term parameters
        arg = -t / tau
        
        # 3. Parameter range clipping
        # np.exp(709) is approximately the maximum value for float64, limiting to +/- 700 is safe
        arg = np.clip(arg, -700, 700)
        
        return a * np.exp(arg) + c

    if len(time_series) < 15:
        return 0.0

    # Normalize time to start at 0
    t_norm = time_series - time_series[0]
    
    # Initial guess: a=start_current, tau=100s, c=0
    try:
        p0 = [current_series[0], 100, 0]
        
        # [Modification] Change tau's lower bound from 0 to 1e-5 to prevent overflow caused by division by zero or extremely small values
        bounds = ([0, 1e-5, -np.inf], [np.inf, 10000, np.inf])
        
        popt, _ = curve_fit(
            exponential_decay, 
            t_norm, 
            current_series, 
            p0=p0, 
            bounds=bounds, 
            maxfev=2000
        )
        return popt[1]  # Return tau
    except (RuntimeError, ValueError):
        return 0.0


def _calculate_enhanced_thermal_features(
    df: pd.DataFrame,
    phase_name: str
) -> Dict[str, float]:
    """
    Calculate enhanced thermal features (Smooth, Heat Rate, Thermal Load).
    Uses Savitzky-Golay filter for noise reduction.
    
    Args:
        df: DataFrame containing 'Time(s)' and 'Temperature(C)'.
        phase_name: 'charge' or 'discharge'.
        
    Returns:
        Dictionary of thermal features. Returns 0.0 for missing data.
    """
    # Initialize defaults
    default_keys = [
        f'MAT_{phase_name}', f'MET_{phase_name}', f'MinT_{phase_name}',
        f'T_rise_{phase_name}',
        f'Max_HeatRate_{phase_name}', f'Mean_HeatRate_{phase_name}',
        f'Thermal_Load_{phase_name}'
    ]
    features = {k: 0.0 for k in default_keys}

    if 'Temperature(C)' not in df.columns or df.empty:
        return features

    # MATR data can sometimes have NaNs or empty strings in Temperature
    try:
        temp_raw = df['Temperature(C)'].astype(float).values
    except ValueError:
        return features
        
    time_arr = df['Time(s)'].values

    # Check for valid data length
    if len(temp_raw) < 5 or np.isnan(temp_raw).all():
        return features

    # 1. Preprocessing: Savitzky-Golay Smoothing
    # Dynamic window length: max 51, must be odd
    window_len = min(len(temp_raw), 51)
    if window_len % 2 == 0:
        window_len -= 1
    
    if window_len < 5:
        temp_smooth = temp_raw
    else:
        try:
            # polyorder=3 preserves peaks better than 2
            temp_smooth = savgol_filter(temp_raw, window_length=window_len, polyorder=3)
        except ValueError:
            temp_smooth = temp_raw

    # 2. Statistical Features
    features[f'MAT_{phase_name}'] = float(np.max(temp_smooth))
    features[f'MET_{phase_name}'] = float(np.mean(temp_smooth))
    features[f'MinT_{phase_name}'] = float(np.min(temp_smooth))
    features[f'T_rise_{phase_name}'] = float(temp_smooth[-1] - temp_smooth[0])

    # 3. Kinetic Features (dT/dt - Heat Rate)
    if len(time_arr) > 5:
        # Separately calculate temperature difference (dT) and time difference (dt)
        dT = np.gradient(temp_smooth)
        dt = np.gradient(time_arr)
        
        # Forcing zero time differences to an extremely small value to prevent division by zero
        dt[dt == 0] = 1e-6 
        
        # Manually calculate derivatives
        dT_dt = dT / dt
        
        features[f'Max_HeatRate_{phase_name}'] = float(np.max(dT_dt))
        features[f'Mean_HeatRate_{phase_name}'] = float(np.mean(dT_dt))

    # 4. Integral Features (Thermal Load: degC * s)
    try:
        thermal_load = trapezoid(y=temp_smooth, x=time_arr)
        features[f'Thermal_Load_{phase_name}'] = float(thermal_load)
    except Exception:
        features[f'Thermal_Load_{phase_name}'] = 0.0

    return features


def _detect_stages_robust(
    df: pd.DataFrame,
    cell_id: str,
    min_duration: float = 60.0
) -> List[Dict[str, Any]]:
    """
    Robustly detects charging stages using Histogram Clustering + Time Stitching.
    Handles current fluctuations, dropouts, and batch-specific noise.

    Args:
        df: Charge phase DataFrame.
        cell_id: Used to determine batch-specific parameters (b1, b2, etc).
        min_duration: Minimum duration (seconds) to keep a stage.

    Returns:
        List of dicts with 'current' and 'duration', sorted by time.
    """
    if df.empty:
        return []

    time = df['Time(s)'].values
    curr = df['Current(A)'].values

    # 0. Batch Adaptive Parameters
    # b2 batch has larger fluctuations
    is_b2 = 'b2' in cell_id

    # Tolerance for clustering: +/- X Amps
    # b2 needs wider tolerance
    cluster_tol = 0.5 if is_b2 else 0.3

    # Gap stitching threshold (seconds)
    # Allow stitching if gap is less than this
    stitch_gap_thresh = 120.0

    # 1. Median Filtering (Remove point-source dropouts/spikes)
    # kernel_size must be odd. 21 points corresponds to ~20s if 1Hz
    curr_smooth = medfilt(curr, kernel_size=15)

    # 2. Histogram Clustering (Identify Dominant Levels)
    # Filter out near-zero currents for histogram
    valid_curr = curr_smooth[curr_smooth > 0.1]

    if len(valid_curr) == 0:
        return []

    # Use histogram to find peaks
    # bins=100 covers 0-10A range with 0.1A precision roughly
    # [FIX] Extend range slightly beyond max to ensure the highest current peak is not at the edge
    max_val = max(valid_curr.max(), 1.0) * 1.1
    hist, bin_edges = np.histogram(valid_curr, bins=100, range=(0, max_val))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Find peaks in histogram
    # distance=5 (0.5A separation), prominence=relative
    peaks, _ = find_peaks(hist, distance=5, prominence=len(valid_curr)*0.01)

    dominant_currents = bin_centers[peaks]
    dominant_currents = np.sort(dominant_currents)[::-1] # Descending order

    if len(dominant_currents) == 0:
        # Fallback if no distinct peaks found
        return []

    # 3. Classification & Segmentation
    # Map each time point to a dominant current cluster
    labels = np.full(len(curr), -1, dtype=int) # -1 = Noise/Gap

    for i, target_I in enumerate(dominant_currents):
        # Find points belonging to this cluster (priority to higher currents if overlap?)
        # Since we sorted descending, higher currents claim first?
        # Actually overlap shouldn't happen with distance=5 (0.5A) and tol=0.3
        mask = np.abs(curr_smooth - target_I) < cluster_tol
        labels[mask] = i

    # 4. Create initial segments
    # Find contiguous regions of same label
    # diff != 0 marks boundaries
    change_points = np.where(np.diff(labels, prepend=labels[0]-1) != 0)[0]
    segments = []

    for k in range(len(change_points)):
        start = change_points[k]
        end = change_points[k+1] if k < len(change_points)-1 else len(labels)
        lbl = labels[start]

        if lbl != -1: # Skip noise segments
            segments.append({
                'label': lbl,
                'target_I': dominant_currents[lbl],
                'start_iloc': start,
                'end_iloc': end,
                'start_time': time[start],
                'end_time': time[end-1]
            })

    if not segments:
        return []

    # 5. Stitching (Merge broken segments)
    # Enhanced Stitching: Merge across small glitches, especially for low current
    merged_segments = []
    if len(segments) > 0:
        current_seg = segments[0]

        i = 1
        while i < len(segments):
            next_seg = segments[i]

            # Check 1: Simple adjacency with same label
            is_same_label = (next_seg['label'] == current_seg['label'])
            gap = next_seg['start_time'] - current_seg['end_time']
            is_small_gap = (gap < stitch_gap_thresh)

            should_merge = False

            if is_same_label and is_small_gap:
                should_merge = True

            # Check 2: Low Current Merge (Cross-Glitch or Drift)
            # If both are low current (< 0.5A), we are more aggressive
            # Even if labels are different (drift) or gap is bridged by a short glitch
            if not should_merge and \
               current_seg['target_I'] < 0.5 and \
               next_seg['target_I'] < 0.5:

                # Allow merge if they are close in time (gap < 120s)
                # This handles the case where a short high-current glitch might have been skipped
                # or just noise caused them to be separate
                if gap < 120.0:
                    should_merge = True

            if should_merge:
                # Merge
                current_seg['end_iloc'] = next_seg['end_iloc']
                current_seg['end_time'] = next_seg['end_time']
                # Update target_I to be weighted average? Or just keep first?
                # For robust clustering, keep first is okay, we calc real mean later.
            else:
                merged_segments.append(current_seg)
                current_seg = next_seg

            i += 1

        merged_segments.append(current_seg)

    # 6. Final Calculation & Filtering
    final_stages = []

    for seg in merged_segments:
        duration = seg['end_time'] - seg['start_time']

        if duration < min_duration:
            continue

        # Calculate robust mean current
        # Use ORIGINAL current, but only points that are somewhat close to target
        # to exclude the deep dropouts included in the gap
        seg_curr = curr[seg['start_iloc']:seg['end_iloc']]

        # Valid points for mean calculation: within 2x tolerance of target
        # This keeps the mean physically meaningful (Setting Current)
        valid_mask = np.abs(seg_curr - seg['target_I']) < (cluster_tol * 2.0)

        if np.sum(valid_mask) > 0:
            avg_curr = np.mean(seg_curr[valid_mask])
        else:
            avg_curr = seg['target_I'] # Fallback

        final_stages.append({
            'start_iloc': seg['start_iloc'],
            'end_iloc': seg['end_iloc'],
            'duration': duration,
            'current': avg_curr,
            'start_time': seg['start_time']
        })

    # Sort by start time
    final_stages.sort(key=lambda x: x['start_time'])

    return final_stages


def _calculate_direct_features(
    cycle_df: pd.DataFrame,
    charge_df: pd.DataFrame,
    discharge_df: pd.DataFrame,
    cycle_num: int,
    battery_metadata: Any
) -> Dict[str, Any]:
    """
    Calculate direct features (Capacity, Energy, Efficiency, CV Tau, Temp).
    """
    features = {}

    # --- A. Overall Cycle Features ---
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
    if chg_cap > 1e-6:
        features['Coulombic_Efficiency'] = dis_cap / chg_cap
    else:
        features['Coulombic_Efficiency'] = 0.0

    # C-Rates
    try:
        if hasattr(battery_metadata, 'charge_protocol') and battery_metadata.charge_protocol:
            # Handle potential list or single object
            if isinstance(battery_metadata.charge_protocol, list):
                features['charge_c_rate'] = battery_metadata.charge_protocol[0].rate_in_C
            else:
                features['charge_c_rate'] = battery_metadata.charge_protocol.rate_in_C
        else:
            features['charge_c_rate'] = 0.0

        if hasattr(battery_metadata, 'discharge_protocol') and battery_metadata.discharge_protocol:
            if isinstance(battery_metadata.discharge_protocol, list):
                features['discharge_c_rate'] = battery_metadata.discharge_protocol[0].rate_in_C
            else:
                features['discharge_c_rate'] = battery_metadata.discharge_protocol.rate_in_C
        else:
            features['discharge_c_rate'] = 0.0
    except AttributeError:
        features['charge_c_rate'] = 0.0
        features['discharge_c_rate'] = 0.0

    # --- B. Energy & Efficiency (Integration) ---
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

    if features['Charge_Energy(Wh)'] > 1e-6:
        features['Energy_Efficiency'] = features['Discharge_Energy(Wh)'] / features['Charge_Energy(Wh)']
    else:
        features['Energy_Efficiency'] = 0.0

    # --- C. Charging Phase Features & CV Dynamics ---
    features['CV_Current_Tau'] = 0.0

    if not charge_df.empty:
        features['ICHV(V)'] = charge_df['Voltage(V)'].iloc[0]
        features['UVP_time(s)'] = charge_df['Time(s)'].iloc[-1]
        features['UVP(V)'] = battery_metadata.max_voltage_limit_in_V

        # [NEW] Multi-stage Charge Detection (Robust)
        chg_stages = _detect_stages_robust(charge_df, battery_metadata.cell_id)

        # Initialize defaults for up to 3 stages
        for i in range(1, 4):
            features[f'charge_current_{i}'] = 0.0
            features[f'charge_time_{i}'] = 0.0

        # Fill detected stages
        for i, stage in enumerate(chg_stages[:3]):
            features[f'charge_current_{i+1}'] = abs(stage['current'])
            features[f'charge_time_{i+1}'] = stage['duration']

        # [FIX] TCCC calculation: Sum of all Constant Current stages
        # Use detected stages to define TCCC, as they represent the physical CC phases
        # [Refinement] Filter out CV stages that might be misclassified as CC stages
        # Criteria: CV stage has high voltage (near max) and low voltage variance (flat)
        if len(chg_stages) > 0:
            v_max_limit = charge_df['Voltage(V)'].max()
            cc_duration_sum = 0.0

            for stage in chg_stages:
                # Retrieve voltage data for the stage
                # Note: iloc is 0-indexed relative to charge_df if we use detected indices directly?
                # _detect_stages_robust returns indices relative to the passed df
                start = stage['start_iloc']
                end = stage['end_iloc']
                stage_volt = charge_df['Voltage(V)'].iloc[start:end]

                if stage_volt.empty:
                    continue

                v_mean = stage_volt.mean()
                v_std = stage_volt.std()

                # CV Criteria: Voltage is effectively constant at the top
                # Thresholds: Mean > Max - 0.03V, Std < 0.015V
                is_cv = (v_mean > (v_max_limit - 0.03)) and (v_std < 0.015)

                if not is_cv:
                    cc_duration_sum += stage['duration']

            features['TCCC(s)'] = cc_duration_sum

            # TCVC is the remaining time (Total Charge Time - TCCC)
            total_charge_time = charge_df['Time(s)'].iloc[-1] - charge_df['Time(s)'].iloc[0]
            features['TCVC(s)'] = max(0.0, total_charge_time - features['TCCC(s)'])

            # --- CV Tau Fitting ---
            # Attempt to fit Tau on the remaining CV part
            # CV part starts roughly after TCCC
            # We find the index where time > start + TCCC
            time_start = charge_df['Time(s)'].iloc[0]
            mask_cv_time = charge_df['Time(s)'] > (time_start + features['TCCC(s)'])

            if mask_cv_time.any():
                cv_df = charge_df[mask_cv_time]
                cv_current = cv_df['Current(A)'].values
                cv_time = cv_df['Time(s)'].values

                valid_mask = cv_current > 0.001
                if np.sum(valid_mask) > 15:
                    features['CV_Current_Tau'] = _fit_cv_decay(
                        cv_time[valid_mask],
                        cv_current[valid_mask]
                    )
        else:
            # Fallback: Determine CV phase by voltage threshold if no stages detected
            v_upper_limit = battery_metadata.max_voltage_limit_in_V
            charge_voltage = charge_df['Voltage(V)']

            mask_cv = charge_voltage >= (v_upper_limit - 0.01)

            if mask_cv.any():
                cv_start_idx = mask_cv.idxmax()
                time_at_v_limit = charge_df.loc[cv_start_idx, 'Time(s)']

                features['TCCC(s)'] = time_at_v_limit - charge_df['Time(s)'].iloc[0]
                features['TCVC(s)'] = charge_df['Time(s)'].iloc[-1] - time_at_v_limit

                # --- CV Tau Fitting ---
                cv_df = charge_df.loc[cv_start_idx:]
                cv_current = cv_df['Current(A)'].values
                cv_time = cv_df['Time(s)'].values

                valid_mask = cv_current > 0.001
                if np.sum(valid_mask) > 15:
                    features['CV_Current_Tau'] = _fit_cv_decay(
                        cv_time[valid_mask],
                        cv_current[valid_mask]
                    )
            else:
                features['TCCC(s)'] = charge_df['Time(s)'].iloc[-1] - charge_df['Time(s)'].iloc[0]
                features['TCVC(s)'] = 0.0

        # [NEW] Enhanced Charge Temp Features
        features.update(_calculate_enhanced_thermal_features(charge_df, 'charge'))

    else:
        features.update({
            'ICHV(V)': 0, 'UVP_time(s)': 0, 'UVP(V)': 0, 'TCCC(s)': 0, 'TCVC(s)': 0,
            'charge_current_1(A)': 0, 'charge_time_1(s)': 0,
            'charge_current_2(A)': 0, 'charge_time_2(s)': 0,
            'charge_current_3(A)': 0, 'charge_time_3(s)': 0
        })
        features.update(_calculate_enhanced_thermal_features(pd.DataFrame(), 'charge'))

    # --- D. Discharging Phase Features ---
    if not discharge_df.empty:
        features['IDV(V)'] = discharge_df['Voltage(V)'].iloc[0]
        features['LVP_time(s)'] = discharge_df['Time(s)'].iloc[-1]
        features['LVP(V)'] = battery_metadata.min_voltage_limit_in_V
        features['var_I_discharge'] = discharge_df['Current(A)'].var()
        features['var_V_discharge'] = discharge_df['Voltage(V)'].var()
        features['median_V_discharge(V)'] = discharge_df['Voltage(V)'].median()
        features['total_discharge_time(s)'] = discharge_df['Time(s)'].iloc[-1] - discharge_df['Time(s)'].iloc[0]
        
        # [NEW] Enhanced Discharge Temp Features
        features.update(_calculate_enhanced_thermal_features(discharge_df, 'discharge'))
    else:
        features.update({
            'IDV(V)': 0, 'LVP_time(s)': 0, 'LVP(V)': 0, 'var_I_discharge': 0, 
            'var_V_discharge': 0, 'median_V_discharge(V)': 0, 'total_discharge_time(s)': 0,
        })
        features.update(_calculate_enhanced_thermal_features(pd.DataFrame(), 'discharge'))

    return features


def _calculate_derivative_features(
    charge_df: pd.DataFrame,
    discharge_df: pd.DataFrame,
    cycle_num: int = 0,
    cell_id: str = "unknown",
    output_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Calculate features derived from IC, DV curves.
    Refactored to use src.utils.feature_tools.extract_ic_dv_features.
    """
    # 1. Use standardized extraction from util
    # MATR uses A123 LFP cells (1.1Ah nominal)
    config = {
        'peak_mode': 1, # LFP Single Peak
        'nominal_capacity': 1.1,
        'voltage_range_ic': (2.8, 3.8), # A123 specific range
        'prominence_ic': 0.05,
        'window_length_ic': 31,
        'window_length_dv': 31,
        'plot_interval': 50,  # Plot every 50 cycles
        # [FIX] Explicitly enable DVV search and set windows
        'disable_dvv': False,
        'search_window_dvv': 0.1,
        'search_window_dvp': 0.1,
        'ic_step_size': 0.001,
        'dv_step_size': 1.1 * 0.005, # 0.01 Ah
        'initial_capacity_cut_fraction': 0.02,
        'icv_search_offset_lower': 0.05,
        'icv_search_offset_upper': 0.1,
        # [NEW] Fixed width integration for ICP Area (+/- 30mV)
        'ic_area_config': {
            'method': 'fixed_width',
            'width_v': 0.03
        }
    }

    plot_params = None
    if output_dir:
        plot_params = {
            'cell_id': cell_id,
            'cycle_num': cycle_num,
            'output_dir': output_dir
        }

    # The util expects 'Discharge_Capacity(Ah)' and 'Voltage(V)'
    features = extract_ic_dv_features(discharge_df, config, plot_params=plot_params)

    # 2. Add DTP (Derivative Temperature Peak) - Unique to MATR, not in utils yet
    # We must preserve this logic as it's not in the standard tool
    features['DTP'] = 0.0
    features['DTPL_V'] = 0.0

    df = discharge_df.copy()
    if 'Temperature(C)' in df.columns and not df['Temperature(C)'].isnull().all():
        # Clean duplicates for gradient calculation
        df = df.drop_duplicates(subset=['Discharge_Capacity(Ah)'])
        if len(df) > 15:
            temp_vals = df['Temperature(C)'].values
            cap_vals = df['Discharge_Capacity(Ah)'].values
            volt_vals = df['Voltage(V)'].values

            # Smooth Temp
            win_len_t = min(len(temp_vals), 31)
            if win_len_t % 2 == 0: win_len_t -= 1
            if win_len_t > 3:
                temp_smooth = savgol_filter(temp_vals, window_length=win_len_t, polyorder=2)
            else:
                temp_smooth = temp_vals

            dQ = np.gradient(cap_vals)
            dT = np.gradient(temp_smooth)

            with np.errstate(divide='ignore', invalid='ignore'):
                dt_curve = np.abs(np.divide(dT, dQ, out=np.zeros_like(dT), where=dQ!=0))

            # Smooth DT curve
            win_len = min(len(dt_curve), 15)
            if win_len % 2 == 0: win_len -= 1
            if win_len > 3:
                dt_smooth = savgol_filter(dt_curve, window_length=win_len, polyorder=2)
            else:
                dt_smooth = dt_curve

            dt_peaks, _ = find_peaks(dt_smooth, height=0.1)
            if len(dt_peaks) > 0:
                idx_max = np.argmax(dt_smooth[dt_peaks])
                peak_idx = dt_peaks[idx_max]
                features['DTP'] = dt_smooth[peak_idx]
                features['DTPL_V'] = volt_vals[peak_idx]

    return features


def _calculate_advanced_features(
    charge_df: pd.DataFrame, 
    discharge_df: pd.DataFrame, 
    features: Dict[str, Any], 
    cycle_data: Any
) -> Dict[str, Any]:
    """Calculates internal resistance, RCV, and Temp stats."""
    adv_features = {}
    
    # Internal Resistance
    try:
        # Some MATR files use direct object access, some might be dicts
        if isinstance(cycle_data, dict):
            adv_features['Internal_Resistance(Ohm)'] = cycle_data.get('internal_resistance_in_ohm', 0.0)
        else:
            adv_features['Internal_Resistance(Ohm)'] = getattr(cycle_data, 'internal_resistance_in_ohm', 0.0)
    except AttributeError:
        adv_features['Internal_Resistance(Ohm)'] = 0.0

    # [FIX] Fallback calculation if Internal Resistance is 0 or extremely small
    # This happens in some MATR batch files where the field is missing or zeroed.
    # Calculate DCIR using the voltage drop at the start of discharge.
    if adv_features['Internal_Resistance(Ohm)'] <= 1e-6:
        try:
            # We need the full cycle arrays to find the transition
            # cycle_data is an AttrDict wrapping the original struct/dict
            # Check if we have the arrays
            if hasattr(cycle_data, 'current_in_A') and hasattr(cycle_data, 'voltage_in_V'):
                curr = np.array(cycle_data.current_in_A)
                volt = np.array(cycle_data.voltage_in_V)

                # Find start of discharge: first point where Current < -0.1A (assuming reasonable C-rate)
                # Discharge usually follows Charge or Rest (Current >= 0)
                dis_indices = np.where(curr < -0.1)[0]

                if len(dis_indices) > 0:
                    idx = dis_indices[0]
                    # Ensure we have a previous point to compare against
                    if idx > 0:
                        v_load = volt[idx]
                        i_load = curr[idx]

                        v_rest = volt[idx-1]
                        i_rest = curr[idx-1]

                        delta_v = abs(v_rest - v_load)
                        delta_i = abs(i_rest - i_load)

                        # Prevent division by zero or noise
                        if delta_i > 0.1:
                            adv_features['Internal_Resistance(Ohm)'] = delta_v / delta_i
        except Exception:
            # If calculation fails, keep it as 0
            pass

    # RCV
    if features.get('TCVC', 0) > 0:
        adv_features['RCV(V)'] = features.get('TCCC', 0) / features['TCVC(s)']
    else:
        adv_features['RCV(V)'] = 0.0

    # Skewness
    if not discharge_df.empty:
        adv_features['skew_V_discharge'] = skew(discharge_df['Voltage(V)'])
    else:
        adv_features['skew_V_discharge'] = 0.0

    # Temperature Statistics (Skewness mainly, others moved to direct/enhanced)
    if not discharge_df.empty and 'Temperature(C)' in discharge_df.columns:
        adv_features['skew_T_discharge'] = skew(discharge_df['Temperature(C)'])
    else:
        adv_features['skew_T_discharge'] = 0.0

    return adv_features


def _calculate_anchor_features(
    charge_df: pd.DataFrame, 
    discharge_df: pd.DataFrame, 
    charge_slope_intervals: List[Tuple[float, float]], 
    discharge_slope_intervals: List[Tuple[float, float]], 
    tevi_intervals: List[Tuple[float, float]], 
    tevd_intervals: List[Tuple[float, float]]
) -> Dict[str, Any]:
    """
    Calculate anchor interval features using percentage-based time intervals.
    """
    anchor_features = {}

    def get_voltage_at_relative_time(df, relative_time):
        if df.empty: return None
        start_time = df['Time(s)'].iloc[0]
        target_absolute_time = start_time + relative_time
        time_array = df['Time(s)'].values 
        
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

    # --- Slopes ---
    # Charge (Disabled due to unstable multi-step charging)
    # c_dur = charge_df['Time(s)'].iloc[-1] - charge_df['Time(s)'].iloc[0] if not charge_df.empty else 0
    # for i, (p_start, p_end) in enumerate(charge_slope_intervals):
    #     if c_dur > 0:
    #         t_start = c_dur * p_start if p_start < 1.0 else p_start
    #         t_end = c_dur * p_end if p_end < 1.0 else p_end
    #
    #         if c_dur > t_end:
    #             v_start = get_voltage_at_relative_time(charge_df, t_start)
    #             v_end = get_voltage_at_relative_time(charge_df, t_end)
    #             if v_start is not None and v_end is not None:
    #                 anchor_features[f'charge_slope_{i+1}'] = (v_end - v_start) / (t_end - t_start)
    #             else:
    #                 anchor_features[f'charge_slope_{i+1}'] = 0.0
    #         else:
    #              anchor_features[f'charge_slope_{i+1}'] = 0.0
    #     else:
    #         anchor_features[f'charge_slope_{i+1}'] = 0.0

    # Discharge Slope - Linear Fit Implementation
    d_dur = discharge_df['Time(s)'].iloc[-1] - discharge_df['Time(s)'].iloc[0] if not discharge_df.empty else 0
    t_base = discharge_df['Time(s)'].iloc[0] if not discharge_df.empty else 0

    for i, (p_start, p_end) in enumerate(discharge_slope_intervals):
        if d_dur > 10.0: # Ensure enough duration
            t_start_rel = d_dur * p_start if p_start < 1.0 else p_start
            t_end_rel = d_dur * p_end if p_end < 1.0 else p_end

            t_start_abs = t_base + t_start_rel
            t_end_abs = t_base + t_end_rel

            # Slice data
            mask = (discharge_df['Time(s)'] >= t_start_abs) & (discharge_df['Time(s)'] <= t_end_abs)
            segment = discharge_df[mask]

            if len(segment) > 5: # Need enough points for fit
                try:
                    # np.polyfit(x, y, 1) returns [slope, intercept]
                    slope, _ = np.polyfit(segment['Time(s)'], segment['Voltage(V)'], 1)

                    # Physical constraint: Discharge slope should be negative.
                    # If positive (noise), clamp to 0.
                    if slope > 0:
                         slope = 0.0

                    anchor_features[f'discharge_slope_{i+1}'] = slope
                except Exception:
                    anchor_features[f'discharge_slope_{i+1}'] = 0.0
            else:
                 anchor_features[f'discharge_slope_{i+1}'] = 0.0
        else:
            anchor_features[f'discharge_slope_{i+1}'] = 0.0

    # --- TEVI / TEVD ---
    def get_time_for_voltage(df, voltage, direction):
        if df.empty: return None
        if direction == 'charge':
            target_rows = df[df['Voltage(V)'] >= voltage]
        else:
            target_rows = df[df['Voltage(V)'] <= voltage]
        return target_rows['Time(s)'].iloc[0] if not target_rows.empty else None

    # (Disabled TEVI)
    # for i, (v_start, v_end) in enumerate(tevi_intervals):
    #     t_start = get_time_for_voltage(charge_df, v_start, 'charge')
    #     t_end = get_time_for_voltage(charge_df, v_end, 'charge')
    #     if t_start is not None and t_end is not None and t_end > t_start:
    #         anchor_features[f'TEVI_{i+1}'] = t_end - t_start
    #     else:
    #         anchor_features[f'TEVI_{i+1}'] = 0.0

    for i, (v_start, v_end) in enumerate(tevd_intervals):
        t_start = get_time_for_voltage(discharge_df, v_start, 'discharge')
        t_end = get_time_for_voltage(discharge_df, v_end, 'discharge')
        if t_start is not None and t_end is not None and t_end > t_start:
            anchor_features[f'TEVD_{i+1}'] = t_end - t_start
        else:
            anchor_features[f'TEVD_{i+1}'] = 0.0

    return anchor_features


def _calculate_personalized_features(cycle_df, cell_id):
    return {}


def extract_features_for_cycle(
    cycle_data: Any,
    battery_metadata: Any,
    charge_slope_intervals: List[Tuple[float, float]],
    discharge_slope_intervals: List[Tuple[float, float]],
    tevi_intervals: List[Tuple[float, float]],
    tevd_intervals: List[Tuple[float, float]],
    output_dir: Optional[Path] = None  # Added output_dir
) -> Dict[str, Any]:

    # 1. Prepare Data
    cycle_df = pd.DataFrame({
        'Time(s)': cycle_data.time_in_s,
        'Current(A)': cycle_data.current_in_A,
        'Voltage(V)': cycle_data.voltage_in_V,
        'Charge_Capacity(Ah)': cycle_data.charge_capacity_in_Ah,
        'Discharge_Capacity(Ah)': cycle_data.discharge_capacity_in_Ah,
        'Temperature(C)': cycle_data.temperature_in_C 
    })
    cycle_num = cycle_data.cycle_number

    # 2. Phase Separation
    charge_df = cycle_df[cycle_df['Current(A)'] > 0].copy()
    discharge_df = cycle_df[cycle_df['Current(A)'] < 0].copy()

    # Cutoff Logic
    if not discharge_df.empty:
        cutoff_voltage = battery_metadata.min_voltage_limit_in_V 
        cutoff_indices = discharge_df.index[discharge_df['Voltage(V)'] <= cutoff_voltage]
        if not cutoff_indices.empty:
            discharge_df = discharge_df.loc[:cutoff_indices[0]]

    # 3. Extract Features
    direct_features = _calculate_direct_features(
        cycle_df, charge_df, discharge_df, cycle_num, battery_metadata
    )
    derivative_features = _calculate_derivative_features(
        charge_df, discharge_df, cycle_num, battery_metadata.cell_id, output_dir
    )
    advanced_features = _calculate_advanced_features(
        charge_df, discharge_df, direct_features, cycle_data
    )
    anchor_features = _calculate_anchor_features(
        charge_df, discharge_df, 
        charge_slope_intervals, discharge_slope_intervals, 
        tevi_intervals, tevd_intervals
    )
    personalized_features = _calculate_personalized_features(
        cycle_df, battery_metadata.cell_id
    )

    return {
        **direct_features, 
        **derivative_features, 
        **advanced_features, 
        **anchor_features, 
        **personalized_features
    }

# Helper class for dot notation access if dict is loaded
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

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

    battery_data = AttrDict(data_dict)
    
    # Ensure nested structures are AttrDicts for dot notation
    if 'cycle_data' in battery_data and battery_data['cycle_data'] is not None:
        battery_data.cycle_data = [AttrDict(c) for c in battery_data.cycle_data]
    if 'charge_protocol' in battery_data and battery_data['charge_protocol'] is not None:
        battery_data.charge_protocol = [AttrDict(p) for p in battery_data.charge_protocol]
    if 'discharge_protocol' in battery_data and battery_data['discharge_protocol'] is not None:
        if isinstance(battery_data.discharge_protocol, list) and len(battery_data.discharge_protocol) > 0:
            battery_data.discharge_protocol = AttrDict(battery_data.discharge_protocol[0])
        elif isinstance(battery_data.discharge_protocol, dict):
            battery_data.discharge_protocol = AttrDict(battery_data.discharge_protocol)

    all_cycle_features = []
    
    cycles_to_process = battery_data.cycle_data
    if num_cycles is not None and num_cycles > 0:
        cycles_to_process = battery_data.cycle_data[:num_cycles]
        
    for cycle_data in tqdm(cycles_to_process, desc=f"Processing {battery_data.cell_id}"):
        if not hasattr(cycle_data, 'time_in_s') or not cycle_data.time_in_s: 
            continue
        
        try:
            features = extract_features_for_cycle(
                cycle_data, battery_data,
                charge_slope_intervals, discharge_slope_intervals,
                tevi_intervals, tevd_intervals,
                output_dir=output_dir  # Pass output_dir
            )
            all_cycle_features.append(features)
        except Exception as e:
            print(f"Error processing cycle {cycle_data.cycle_number}: {e}")
            traceback.print_exc()
            continue

    if not all_cycle_features:
        print(f"Warning: No features extracted for {battery_data.cell_id}")
        return

    features_df = pd.DataFrame(all_cycle_features)
    
    ordered_cols_priority = [
        # Overall
        'Cycle_Number',
        'Discharge_Capacity', 'Charge_Capacity',
        'Discharge_Energy', 'Charge_Energy',
        'Coulombic_Efficiency', 'Energy_Efficiency',
        'charge_c_rate', 'discharge_c_rate',
        # [NEW] Multi-stage Charge
        'charge_current_1', 'charge_time_1',
        'charge_current_2', 'charge_time_2',
        'charge_current_3', 'charge_time_3',
        # Charge Dynamics
        'ICHV', 'UVP_time', 'TCCC', 'TCVC', 'CV_Current_Tau', 'UVP',
        # Charge Temp (Enhanced)
        'MAT_charge', 'MET_charge', 'MinT_charge', 'T_rise_charge',
        'Max_HeatRate_charge', 'Mean_HeatRate_charge', 'Thermal_Load_charge',
        # Discharge Dynamics
        'IDV', 'LVP_time', 'var_I_discharge', 'var_V_discharge', 
        'median_V_discharge', 'total_discharge_time', 'LVP',
        # Discharge Temp (Enhanced)
        'MAT_discharge', 'MET_discharge', 'MinT_discharge', 'T_rise_discharge',
        'Max_HeatRate_discharge', 'Mean_HeatRate_discharge', 'Thermal_Load_discharge',
        # Advanced
        'Internal_Resistance', 'RCV', 'skew_V_discharge', 'skew_T_discharge',
        # Curves
        'ICP', 'ICPL_V', 'ICP_FWHM', 'ICP_Area',
        'ICV', 'ICVL_V', 
        'DVP', 'DVPL_V', 'DVP_FWHM', 'DVP_Area',
        'DVV', 'DVVL_V',
        'DTP', 'DTPL_V',
        # Anchors
        # 'charge_slope_1', 'charge_slope_2', 'charge_slope_3',
        'discharge_slope_1', 'discharge_slope_2', 'discharge_slope_3',
        # 'TEVI_1', 'TEVI_2', 'TEVI_3',
        'TEVD_1', 'TEVD_2', 'TEVD_3'
    ]
    
    # Filter only columns that actually exist
    ordered_cols = [c for c in ordered_cols_priority if c in features_df.columns]
    remaining_cols = [c for c in features_df.columns if c not in ordered_cols]
    
    features_df = features_df[ordered_cols + remaining_cols]

    output_file = output_dir / f"{battery_data.cell_id}.csv"
    features_df.to_csv(output_file, index=False)
    print(f"Features for {battery_data.cell_id} saved to {output_file}")


def main():
    processed_data_dir = Path('F:/datasets/battery/MATR')
    output_dir = project_root / 'results' / 'MATR'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use Percentage for slopes (0.1 = 10% of time)
    charge_slope_intervals = [(0.10, 0.30), (0.10, 0.50), (0.10, 0.80)]
    discharge_slope_intervals = [(0.10, 0.30), (0.10, 0.50), (0.10, 0.80)]
    
    # Voltage Intervals (MATR: 2.0V - 3.6V range)
    tevi_intervals = [(2.5, 2.8), (2.8, 3.1), (3.1, 3.4)]
    tevd_intervals = [(3.4, 3.1), (3.1, 2.8), (2.8, 2.5)]

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
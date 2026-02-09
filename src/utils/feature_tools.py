"""
Feature extraction tools for battery data.
Contains logic for phase identification and IC/DV curve analysis.

Data Source Usage Guide:
------------------------
This utility is designed to work with generic capacity/voltage dataframes.
Most datasets use **Discharge Data** for IC/DV analysis (e.g., CALB, CALCE, Stanford, SNL).

Exceptions (Charge Data):
- **HUST**: Uses `charge_df` directly.
- **RWTH**: Uses `cc_charge_df` (mapped to mimic discharge columns).

Ensure input DataFrames contain 'Voltage(V)' and 'Discharge_Capacity(Ah)' columns.
For charge data, map 'Charge_Capacity(Ah)' to 'Discharge_Capacity(Ah)' before calling.
"""

from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.signal import find_peaks, savgol_filter, peak_widths
from scipy.integrate import trapezoid

# Import from local utils
from .math_tools import (
    equidistant_resample,
    calculate_area_with_baseline,
    get_interp_val,
    find_curvature_boundaries
)
from .plot_tools import plot_ic_dv_curves


def identify_phases(
    cycle_df: pd.DataFrame,
    rest_threshold_a: float = 0.01,
    change_threshold_a: float = 0.05
) -> List[Dict[str, Any]]:
    """
    Scans the cycle chronologically to identify continuous operating phases.

    Args:
        cycle_df: DataFrame containing cycle data.
        rest_threshold_a: Current threshold to define rest phase.
        change_threshold_a: Current change threshold to detect steps.

    Returns:
        List[Dict[str, Any]]: List of phases with type and dataframe.
    """
    def get_state(current: float) -> str:
        if abs(current) < rest_threshold_a:
            return 'Rest'
        if current >= rest_threshold_a:
            return 'Charge'
        return 'Discharge'

    phases = []
    if cycle_df.empty:
        return phases

    start_idx = 0
    current_state = get_state(cycle_df['Current(A)'].iloc[0])

    for i in range(1, len(cycle_df)):
        new_state = get_state(cycle_df['Current(A)'].iloc[i])

        state_changed = (new_state != current_state)
        step_changed = False

        # Detect step changes within the same state
        if not state_changed and current_state != 'Rest':
            current_diff = abs(
                cycle_df['Current(A)'].iloc[i] - cycle_df['Current(A)'].iloc[i - 1]
            )
            if current_diff > change_threshold_a:
                step_changed = True

        if state_changed or step_changed:
            if i > start_idx:
                phases.append({
                    'type': current_state,
                    'df': cycle_df.iloc[start_idx:i].copy()
                })
            start_idx = i
            current_state = new_state

    if start_idx < len(cycle_df):
        phases.append({
            'type': current_state,
            'df': cycle_df.iloc[start_idx:].copy()
        })

    return phases


def extract_ic_dv_features(
    discharge_df: pd.DataFrame,
    config: Dict[str, Any],
    plot_params: Optional[Dict[str, Any]] = None,
    include_curves: bool = False
) -> Dict[str, Any]:
    """
    Calculates IC/DV curve features using Adaptive Resampling, Absolute Range Search,
    and Curvature Boundary Integration.

    Refactored to meet 'Problem Fix' requirements (Issues 1-5).

    Args:
        discharge_df: DataFrame for discharge phase.
        config: Dictionary containing battery-specific parameters:
            - peak_mode (int): 1 for Single Peak (LFP), 2 for Multi Peak (NCM). Default 2.
            - nominal_capacity (float): Battery capacity in Ah.
            - voltage_range_ic (Tuple[float, float]): Absolute voltage range for IC search (min, max).
            - prominence_ic (float): Prominence threshold for IC.
            - ic_step_size (float): Step size for Voltage resampling (default 0.002 V).
            - dv_step_size (float): Step size for Capacity resampling (default 0.01 Ah).
            - window_length_ic (int): Smoothing window for IC (default 25).
            - window_length_dv (int): Smoothing window for DV (default 25).
            - search_window_v (float): Voltage window for targeted DV search (default 0.2 V).
            - icv_search_offset_lower (float): Lower bound offset for ICV search relative to ICP (default 0.05 V).
            - icv_search_offset_upper (float): Upper bound offset for ICV search relative to ICP (default 0.5 V).
        plot_params: Optional dict for plotting. If provided, plots are generated.
            - cell_id (str)
            - cycle_num (int)
            - output_dir (Path)
        include_curves: If True, includes raw curve data in the returned dictionary.

    Returns:
        Dict[str, Any]: Dictionary of IC/DV features.
    """
    # Standardize output keys (ensure consistency across modes)
    features = {
        'ICP': np.nan, 'ICPL_V': np.nan, 'ICV': np.nan, 'ICVL_V': np.nan,
        'DVP': np.nan, 'DVPL_V': np.nan, 'DVP_Q': np.nan,
        'DVV': np.nan, 'DVVL_V': np.nan, 'DVV_Q': np.nan,
        'ICP_Area': np.nan, 'ICP_FWHM': np.nan,
        # 'DVP_Area': np.nan, 'DVP_FWHM': np.nan,
        'centroid_voltage': np.nan,
        'peak_mode': np.nan,
        'dvp_type': np.nan,  # 1: Real Peak, 0: Inflection Point
        'ICP_is_missing': 1  # 1: Missing, 0: Found
    }

    if len(discharge_df) < 15:
        return features

    # --- Config Defaults ---
    peak_mode = config.get('peak_mode', 2)
    features['peak_mode'] = peak_mode

    nominal_cap = config.get('nominal_capacity', 1.0)
    ic_v_range = config.get('voltage_range_ic', (2.8, 4.2))
    prom_ic = config.get('prominence_ic', 0.01)

    # Adaptive Step Sizes (Issue 2)
    step_v = config.get('ic_step_size', 0.002) # Voltage step for IC
    step_q = config.get('dv_step_size', nominal_cap * 0.005) # Cap step for DV (e.g. 0.5% SoC)

    # Window Lengths (Adaptive based on config, but defaulting to safe values)
    win_ic = config.get('window_length_ic', 21)
    win_dv = config.get('window_length_dv', 21)

    # Peak Heights (Issue 1 extension: configurable height threshold)
    height_ic = config.get('peak_height_ic', 0.01)

    # Search Window (New parameter for IC-guided DV search)
    search_win = config.get('search_window_v', 0.2)
    search_win_dvv = config.get('search_window_dvv', search_win)
    search_win_dvp = config.get('search_window_dvp', search_win)

    # ICV Search Offsets
    icv_offset_lower = config.get('icv_search_offset_lower', 0.05)
    icv_offset_upper = config.get('icv_search_offset_upper', 0.5)

    # Force Zero Overrides (Specific physical constraint handling)
    force_icp_zero = config.get('force_icp_zero', False)
    force_icp_fwhm_zero = config.get('force_icp_fwhm_zero', False)
    force_icv_zero = config.get('force_icv_zero', False)

    # Clean Data
    # De-noising: Cut initial fraction of capacity (optional)
    cut_fraction = config.get('initial_capacity_cut_fraction', 0.0)
    if cut_fraction > 0 and not discharge_df.empty:
        max_cap = discharge_df['Discharge_Capacity(Ah)'].max()
        cut_val = max_cap * cut_fraction
        df = discharge_df[discharge_df['Discharge_Capacity(Ah)'] > cut_val].copy()
    else:
        df = discharge_df.copy()

    df = df.drop_duplicates(subset=['Voltage(V)'])
    df = df.drop_duplicates(subset=['Discharge_Capacity(Ah)'])

    cap_vals = df['Discharge_Capacity(Ah)'].values
    volt_vals = df['Voltage(V)'].values
    v_min, v_max = np.min(volt_vals), np.max(volt_vals)
    q_max = np.max(cap_vals)

    # ==========================================
    # 1. IC Calculation (dQ/dV)
    # ==========================================
    # Adaptive Resampling
    num_points_ic = int(abs(v_max - v_min) / step_v) if step_v > 0 else 1000
    num_points_ic = max(100, min(5000, num_points_ic)) # Safety clamp

    v_grid_ic, q_mapped_ic = equidistant_resample(volt_vals, cap_vals, num_points=num_points_ic)

    dv_ic_grid = np.gradient(v_grid_ic)
    dq_ic_grid = np.gradient(q_mapped_ic)

    with np.errstate(divide='ignore', invalid='ignore'):
        ic_curve = np.abs(np.divide(dq_ic_grid, dv_ic_grid, out=np.zeros_like(dq_ic_grid), where=dv_ic_grid != 0))

    ic_curve[ic_curve < 0] = 0
    ic_smooth = savgol_filter(ic_curve, window_length=min(win_ic, len(ic_curve)), polyorder=2)
    ic_smooth = np.maximum(ic_smooth, 0)  # [Fix] Clip negative values after smoothing

    # --- New Feature: Centroid Voltage ---
    if np.sum(ic_smooth) > 0:
        centroid_v = np.sum(v_grid_ic * ic_smooth) / np.sum(ic_smooth)
        features['centroid_voltage'] = centroid_v
    else:
        features['centroid_voltage'] = np.nan

    # --- Feature Override: Fixed Range IC Area (Need Fixed 1 & 4) ---
    ic_area_range = config.get('ic_area_voltage_range')
    ic_area_calculated = False

    if ic_area_range:
        v_start, v_end = ic_area_range
        # Handle case where v_start > v_end
        if v_start > v_end:
            v_start, v_end = v_end, v_start

        mask_area = (v_grid_ic >= v_start) & (v_grid_ic <= v_end)
        if np.any(mask_area):
            # Calculate absolute area under the curve using trapezoidal rule
            area_val = trapezoid(y=ic_smooth[mask_area], x=v_grid_ic[mask_area])
            features['ICP_Area'] = area_val
            ic_area_calculated = True
        else:
            features['ICP_Area'] = 0.0
            ic_area_calculated = True

    # Search in Absolute Range (Issue 1)
    if force_icp_zero:
        features['ICP'] = 0.0
        features['ICPL_V'] = 0.0
        features['ICP_Area'] = 0.0
        features['ICP_FWHM'] = 0.0
        features['ICP_is_missing'] = 1 # Treated as missing/invalid
        peak_idx = -1
    else:
        # Mask for search region
        ic_search_mask = (v_grid_ic >= ic_v_range[0]) & (v_grid_ic <= ic_v_range[1])
        # Apply find_peaks globally, then filter by mask
    ic_peaks, props_ic = find_peaks(ic_smooth, height=height_ic, prominence=prom_ic)

    # Filter peaks within range
    valid_ic_peaks = [p for p in ic_peaks if ic_search_mask[p]]

    if valid_ic_peaks:
        # Score Peaks (Issue 1: Score = Height * Width * Prominence)
        scores = []
        for p in valid_ic_peaks:
            idx_orig = np.where(ic_peaks == p)[0][0]
            prom = props_ic['prominences'][idx_orig]
            height = ic_smooth[p]
            w = peak_widths(ic_smooth, [p], rel_height=0.5)[0][0]
            scores.append(height * w * prom)

        best_idx = np.argmax(scores)
        peak_idx = valid_ic_peaks[best_idx]
    else:
        # No peaks found: Fallback to Max in Range (User Requirement)
        range_indices = np.where(ic_search_mask)[0]
        if len(range_indices) > 0:
            best_idx = range_indices[np.argmax(ic_smooth[range_indices])]
            peak_idx = best_idx
        else:
            peak_idx = -1

    if peak_idx != -1:
        # Boundary Truncation Check (Issue 4)
        if 5 < peak_idx < len(ic_smooth) - 5:
            features['ICP'] = ic_smooth[peak_idx]
            features['ICPL_V'] = v_grid_ic[peak_idx]
            features['ICP_is_missing'] = 0

            # --- FWHM Calculation ---
            fwhm_method = config.get('fwhm_method', 'standard')

            if force_icp_fwhm_zero:
                features['ICP_FWHM'] = 0.0
            elif fwhm_method == 'valley_limited':
                # Robust FWHM logic: Search outward from peak until Target reached or Valley encountered
                y_peak = ic_smooth[peak_idx]

                # Estimate base level using local window (e.g., +/- 50 points or full range if small)
                window_base = 50
                l_ctx = max(0, peak_idx - window_base)
                r_ctx = min(len(ic_smooth), peak_idx + window_base)
                y_base = np.min(ic_smooth[l_ctx:r_ctx])

                y_target = y_base + (y_peak - y_base) / 2.0

                # Search Left
                l_idx_fwhm = peak_idx
                for i in range(peak_idx - 1, -1, -1):
                    y_curr = ic_smooth[i]
                    y_prev = ic_smooth[i+1] # Moving left from peak

                    # Check for valley: If we start rising (going left), we hit a valley bottom at i+1
                    if y_curr > y_prev:
                        l_idx_fwhm = i + 1
                        break

                    if y_curr <= y_target:
                        l_idx_fwhm = i
                        break
                    l_idx_fwhm = i

                # Interpolate Left
                v_left = v_grid_ic[l_idx_fwhm]
                if l_idx_fwhm < peak_idx and ic_smooth[l_idx_fwhm] <= y_target and ic_smooth[l_idx_fwhm+1] > y_target:
                    y0, y1 = ic_smooth[l_idx_fwhm], ic_smooth[l_idx_fwhm+1]
                    x0, x1 = v_grid_ic[l_idx_fwhm], v_grid_ic[l_idx_fwhm+1]
                    # Linear interp for x at y_target
                    if abs(y1 - y0) > 1e-9:
                         v_left = x0 + (x1 - x0) * ((y_target - y0) / (y1 - y0))

                # Search Right
                r_idx_fwhm = peak_idx
                for i in range(peak_idx + 1, len(ic_smooth)):
                    y_curr = ic_smooth[i]
                    y_prev = ic_smooth[i-1] # Moving right from peak

                    # Check for valley: If we start rising (going right), we hit a valley bottom at i-1
                    if y_curr > y_prev:
                        r_idx_fwhm = i - 1
                        break

                    if y_curr <= y_target:
                        r_idx_fwhm = i
                        break
                    r_idx_fwhm = i

                # Interpolate Right
                v_right = v_grid_ic[r_idx_fwhm]
                if r_idx_fwhm > peak_idx and ic_smooth[r_idx_fwhm] <= y_target and ic_smooth[r_idx_fwhm-1] > y_target:
                    y0, y1 = ic_smooth[r_idx_fwhm-1], ic_smooth[r_idx_fwhm] # y0 > target >= y1
                    x0, x1 = v_grid_ic[r_idx_fwhm-1], v_grid_ic[r_idx_fwhm]
                    if abs(y1 - y0) > 1e-9:
                        v_right = x0 + (x1 - x0) * ((y_target - y0) / (y1 - y0))

                features['ICP_FWHM'] = abs(v_right - v_left)

            else:
                # Standard scipy peak_widths
                w_res = peak_widths(ic_smooth, [peak_idx], rel_height=0.5)
                l_idx_fwhm, r_idx_fwhm = w_res[2][0], w_res[3][0]
                features['ICP_FWHM'] = abs(get_interp_val(v_grid_ic, r_idx_fwhm) - get_interp_val(v_grid_ic, l_idx_fwhm))

            # --- Area Calculation ---
            if not ic_area_calculated:
                ic_area_conf = config.get('ic_area_config')

                if ic_area_conf and ic_area_conf.get('method') == 'fixed_width':
                    # Fixed width integration around peak
                    width_v = ic_area_conf.get('width_v', 0.05)
                    v_peak_val = features['ICPL_V']
                    v_start = v_peak_val - width_v
                    v_end = v_peak_val + width_v

                    mask_area = (v_grid_ic >= v_start) & (v_grid_ic <= v_end)
                    if np.any(mask_area):
                        features['ICP_Area'] = trapezoid(y=ic_smooth[mask_area], x=v_grid_ic[mask_area])
                    else:
                        features['ICP_Area'] = 0.0
                else:
                    # Default Curvature Boundary
                    l_bound, r_bound = find_curvature_boundaries(ic_smooth, peak_idx, window_length=11)
                    features['ICP_Area'] = calculate_area_with_baseline(
                        v_grid_ic[l_bound:r_bound+1],
                        ic_smooth[l_bound:r_bound+1]
                    )

            # --- Logic Branch: ICV Search (Valley between peaks) ---
            icv_direction = config.get('icv_search_direction', 'right')
            icv_method = config.get('icv_method', 'standard')

            if force_icv_zero:
                features['ICV'] = 0.0
                features['ICVL_V'] = 0.0
            elif icv_method == 'first_valley_left':
                # New Logic: Search LEFT from ICP for the FIRST valley (local minimum)
                # Requirement: "从当前ICP的左侧开始查找谷值，从左侧找第一个一阶导数为0的点定义为ICV"
                # Respect Offsets: Start search at (Peak - Lower) and stop at (Peak - Upper)

                v_peak = features['ICPL_V']
                v_start_search = v_peak - icv_offset_lower
                v_stop_search = v_peak - icv_offset_upper

                # Find indices corresponding to these voltages
                # v_grid_ic is typically sorted ascending

                # Start Index: closest to v_start_search (from below)
                idx_start = np.searchsorted(v_grid_ic, v_start_search, side='right') - 1
                idx_start = min(idx_start, peak_idx - 1) # Ensure we don't start at peak

                # Stop Index: closest to v_stop_search
                idx_stop = np.searchsorted(v_grid_ic, v_stop_search, side='left')
                idx_stop = max(idx_stop, 1) # Ensure valid index range

                found_valley = False

                if idx_start > idx_stop:
                    # Search backwards from idx_start down to idx_stop
                    for i in range(idx_start, idx_stop, -1):
                        # Local minimum check: y[i-1] > y[i] < y[i+1]
                        if ic_smooth[i] < ic_smooth[i+1] and ic_smooth[i] < ic_smooth[i-1]:
                            features['ICV'] = ic_smooth[i]
                            features['ICVL_V'] = v_grid_ic[i]
                            found_valley = True
                            break

                if not found_valley:
                    # Fallback Logic (User Requirement)
                    l_bound = idx_stop
                    r_bound = idx_start

                    if r_bound > l_bound:
                        segment = ic_smooth[l_bound : r_bound + 1]
                        seg_indices = np.arange(l_bound, r_bound + 1)

                        # Fallback 1: Derivative Zero Crossing (Inflection/Flat)
                        grad = np.gradient(segment)
                        zero_crossings = np.where(np.diff(np.sign(grad)))[0]

                        if len(zero_crossings) > 0:
                            # Pick the one with minimum IC value
                            best_local_idx = zero_crossings[np.argmin(segment[zero_crossings])]
                            global_idx = seg_indices[best_local_idx]

                            features['ICV'] = ic_smooth[global_idx]
                            features['ICVL_V'] = v_grid_ic[global_idx]
                            found_valley = True

                        # Fallback 2: Global Minimum in Range
                        if not found_valley:
                            min_local_idx = np.argmin(segment)
                            global_idx = seg_indices[min_local_idx]

                            features['ICV'] = ic_smooth[global_idx]
                            features['ICVL_V'] = v_grid_ic[global_idx]
                            found_valley = True

            elif icv_method == 'first_valley_right_constrained':
                # New Logic: Search RIGHT from ICP but constrained by offsets
                # Requirement: Use 'icv_search_offset_lower' and 'icv_search_offset_upper'

                v_peak = features['ICPL_V']
                v_start_search = v_peak + icv_offset_lower
                v_end_search = v_peak + icv_offset_upper

                idx_start = np.searchsorted(v_grid_ic, v_start_search, side='left')
                idx_end = np.searchsorted(v_grid_ic, v_end_search, side='left')

                start_search = max(peak_idx + 1, min(idx_start, len(ic_smooth) - 1))
                end_search = max(start_search, min(idx_end, len(ic_smooth) - 1))

                found_valley = False
                if start_search < end_search:
                    for i in range(start_search, end_search):
                        y_prev = ic_smooth[i-1]
                        y_curr = ic_smooth[i]
                        y_next = ic_smooth[i+1]

                        if y_curr < y_prev and y_curr < y_next:
                            features['ICV'] = y_curr
                            features['ICVL_V'] = v_grid_ic[i]
                            found_valley = True
                            break

            elif peak_mode == 2:
                # NCM / Na-ion Logic: Search for Valley
                if icv_direction == 'left':
                    # --- Optimized Left Search: Try to find two peaks first ---
                    # 1. Identify all valid peaks to the left of main peak
                    left_peak_candidates = [p for p in valid_ic_peaks if p < peak_idx]

                    if left_peak_candidates:
                        # 2. Get the nearest neighbor peak on the left
                        nearest_left_peak_idx = left_peak_candidates[-1]

                        # 3. Search for minimum between the two peaks
                        search_indices = np.arange(nearest_left_peak_idx, peak_idx + 1)
                        if len(search_indices) > 0:
                            abs_min_idx = search_indices[np.argmin(ic_smooth[search_indices])]
                            features['ICV'] = ic_smooth[abs_min_idx]
                            features['ICVL_V'] = v_grid_ic[abs_min_idx]

                    # 4. Fallback: If no left peak found OR search failed, use original window logic
                    if np.isnan(features['ICV']):
                        icv_mask = (v_grid_ic < features['ICPL_V'] - icv_offset_lower) & \
                                   (v_grid_ic > features['ICPL_V'] - icv_offset_upper)
                        if np.any(icv_mask):
                            sub_indices = np.where(icv_mask)[0]
                            abs_min_idx = sub_indices[np.argmin(ic_smooth[sub_indices])]
                            features['ICV'] = ic_smooth[abs_min_idx]
                            features['ICVL_V'] = v_grid_ic[abs_min_idx]
                else:
                    # Default: Search RIGHT of the main peak
                    # REFACTORED: Three-stage Fallback Strategy based on need_fixed.md

                    # 1. Define Search Window based on Offsets
                    v_peak = features['ICPL_V']
                    v_start_search = v_peak + icv_offset_lower
                    v_end_search = v_peak + icv_offset_upper

                    # Map Voltage Range to Indices
                    # v_grid_ic is sorted ascending
                    idx_start = np.searchsorted(v_grid_ic, v_start_search, side='left')
                    idx_end = np.searchsorted(v_grid_ic, v_end_search, side='left')

                    # Safety Bounds: Ensure we stay within array and to the right of peak
                    start_search = max(peak_idx + 1, min(idx_start, len(ic_smooth) - 1))
                    end_search = max(start_search, min(idx_end, len(ic_smooth) - 1))

                    found_icv = False

                    if start_search < end_search:
                        # --- Strategy 1: Valley Between Peaks (within Search Range) ---
                        # Identify all peaks to the right of the main peak
                        all_right_peaks = [p for p in valid_ic_peaks if p > peak_idx]

                        # Collect all local minima (valleys) within the search range
                        range_valleys = []
                        for i in range(start_search, end_search):
                            if i >= len(ic_smooth) - 1: break
                            if ic_smooth[i] < ic_smooth[i-1] and ic_smooth[i] < ic_smooth[i+1]:
                                range_valleys.append(i)

                        if range_valleys and all_right_peaks:
                            # Check if any valley in range is between peaks (has a peak to its right)
                            for v_idx in range_valleys:
                                if any(p > v_idx for p in all_right_peaks):
                                    features['ICV'] = ic_smooth[v_idx]
                                    features['ICVL_V'] = v_grid_ic[v_idx]
                                    found_icv = True
                                    break

                        # --- Strategy 2: First Local Minimum (Derivative Zero Point) ---
                        if not found_icv and range_valleys:
                            # Just take the first local minimum in the range
                            v_idx = range_valleys[0]
                            features['ICV'] = ic_smooth[v_idx]
                            features['ICVL_V'] = v_grid_ic[v_idx]
                            found_icv = True

                        # --- Strategy 3: Global Minimum in Range (Fallback) ---
                        if not found_icv:
                            # Force find the absolute minimum value within the defined search range
                            search_segment = ic_smooth[start_search : end_search + 1]
                            if len(search_segment) > 0:
                                min_rel_idx = np.argmin(search_segment)
                                icv_idx = start_search + min_rel_idx

                                features['ICV'] = ic_smooth[icv_idx]
                                features['ICVL_V'] = v_grid_ic[icv_idx]
                                found_icv = True

                    if not found_icv:
                        # Absolute fallback if range is invalid or empty
                        features['ICV'] = 0.0
                        features['ICVL_V'] = 0.0
            # else: peak_mode == 1, ICV remains NaN (as initialized)

    # --- Logic Branch: Auxiliary Peak Search (Optional) ---
    aux_config = config.get('aux_peak_config')
    if aux_config:
        v_min_aux, v_max_aux = aux_config.get('voltage_range', (0, 0))
        selection = aux_config.get('selection', 'max')  # 'max', 'first', 'last'

        # Filter global peaks by aux range
        mask_aux = (v_grid_ic >= v_min_aux) & (v_grid_ic <= v_max_aux)
        valid_aux_peaks = [p for p in ic_peaks if mask_aux[p]]

        if valid_aux_peaks:
            peak_idx = -1
            if selection == 'first':
                # Sort by index (voltage is monotonic increasing)
                peak_idx = sorted(valid_aux_peaks)[0]
            elif selection == 'last':
                peak_idx = sorted(valid_aux_peaks)[-1]
            else:  # 'max' height
                heights = [ic_smooth[p] for p in valid_aux_peaks]
                best_local = np.argmax(heights)
                peak_idx = valid_aux_peaks[best_local]

            features['ICP_Aux'] = ic_smooth[peak_idx]
            features['ICPL_V_Aux'] = v_grid_ic[peak_idx]
        else:
            default_val = aux_config.get('default_value', np.nan)
            features['ICP_Aux'] = default_val
            features['ICPL_V_Aux'] = np.nan if np.isnan(default_val) else 0.0

    # ==========================================
    # 2. DV Calculation & Targeted Search
    # ==========================================
    num_points_dv = int(q_max / step_q) if step_q > 0 else 1000
    num_points_dv = max(100, min(5000, num_points_dv))

    q_grid_dv, v_mapped_dv = equidistant_resample(cap_vals, volt_vals, num_points=num_points_dv)

    dq_dv_grid = np.gradient(q_grid_dv)
    dv_dv_grid = np.gradient(v_mapped_dv)

    with np.errstate(divide='ignore', invalid='ignore'):
        dv_curve = np.abs(np.divide(dv_dv_grid, dq_dv_grid, out=np.zeros_like(dv_dv_grid), where=dq_dv_grid != 0))

    dv_curve[dv_curve < 0] = 0
    dv_smooth = savgol_filter(dv_curve, window_length=min(win_dv, len(dv_curve)), polyorder=2)
    dv_smooth = np.maximum(dv_smooth, 0)  # [Fix] Clip negative values after smoothing

    # ==========================================
    # Logic Branch: DV Search Strategy
    # ==========================================

    # Common Search setup (Used by both modes now)
    half_window_dvv = search_win_dvv / 2.0
    half_window_dvp = search_win_dvp / 2.0

    # --- Feature Override: Disable DVV (Need Fixed 2) ---
    disable_dvv = config.get('disable_dvv', False)
    if disable_dvv:
        # Force DVV related features to NaN
        features['DVV'] = np.nan
        features['DVVL_V'] = np.nan
        features['DVV_Q'] = np.nan

    # --- Feature Override: Fixed Range DVP (Need Fixed 3 & 4) ---
    dvp_cap_range = config.get('dvp_capacity_range')
    dvp_calculated = False

    if dvp_cap_range:
        q_start, q_end = dvp_cap_range
        # Handle case where q_start > q_end
        if q_start > q_end:
            q_start, q_end = q_end, q_start

        mask_dvp = (q_grid_dv >= q_start) & (q_grid_dv <= q_end)
        if np.any(mask_dvp):
            features['DVP'] = np.mean(dv_smooth[mask_dvp])
            features['DVP_Q'] = np.mean(q_grid_dv[mask_dvp]) # Rough location
            features['dvp_type'] = 2 # 2: Mean Value
            dvp_calculated = True
        else:
            features['DVP'] = np.nan
            dvp_calculated = True

    # --- Feature Override: DVPL_V at Fixed Fraction (Need Fixed 3) ---
    dvpl_fraction = config.get('dvpl_v_capacity_fraction')
    if dvpl_fraction is not None:
        target_q = q_max * dvpl_fraction
        # Find voltage at target_q from (q_grid_dv, v_mapped_dv)
        idx = (np.abs(q_grid_dv - target_q)).argmin()
        features['DVPL_V'] = v_mapped_dv[idx]
        # Note: We do NOT set dvp_calculated = True here because DVPL_V is independent of DVP peak finding
        # But if DVP was calculated via range, we don't need to do peak finding for it either.

    if peak_mode == 1:
        # --- Single Peak (LFP) Logic ---

        # 1. DVV (Valley) - Search near ICP (Requirement: Unified with mode 2)
        if not disable_dvv:
            if not np.isnan(features['ICPL_V']) and features['ICPL_V'] > 0:
                v_target = features['ICPL_V']
                v_upper = v_target + half_window_dvv
                v_lower = v_target - half_window_dvv
                mask_dvv = (v_mapped_dv <= v_upper) & (v_mapped_dv >= v_lower)

                if np.any(mask_dvv):
                    indices_dvv = np.where(mask_dvv)[0]
                    dvv_idx = indices_dvv[np.argmin(dv_smooth[indices_dvv])]

                    features['DVV'] = dv_smooth[dvv_idx]
                    features['DVVL_V'] = v_mapped_dv[dvv_idx]
                    features['DVV_Q'] = q_grid_dv[dvv_idx]
            else:
                # Fallback to global minimum if ICP not found
                dvv_idx = np.argmin(dv_smooth)
                features['DVV'] = dv_smooth[dvv_idx]
                features['DVVL_V'] = v_mapped_dv[dvv_idx]
                features['DVV_Q'] = q_grid_dv[dvv_idx]

        # 2. DVP (Peak) - Safe Search Zone (Original Logic)
        if not dvp_calculated:
            # Range: (0, capacity * 0.7)
            safe_q_max = q_max * 0.7
            mask_dvp_safe = (q_grid_dv > 0) & (q_grid_dv < safe_q_max)

            if np.any(mask_dvp_safe):
                indices_safe = np.where(mask_dvp_safe)[0]
                dv_safe = dv_smooth[indices_safe]

                # Try to find a real peak
                dv_peaks, props_dv = find_peaks(dv_safe, prominence=0.01) # Original threshold

                if len(dv_peaks) > 0:
                    # Found a real peak
                    best_peak_local_idx = np.argmax(props_dv['prominences'])
                    best_peak_idx = indices_safe[dv_peaks[best_peak_local_idx]]

                    features['DVP'] = dv_smooth[best_peak_idx]
                    if dvpl_fraction is None: # Only set if not overridden
                        features['DVPL_V'] = v_mapped_dv[best_peak_idx]
                    features['DVP_Q'] = q_grid_dv[best_peak_idx]
                    features['dvp_type'] = 1 # Real Peak
                else:
                    # Fallback: Inflection Point (Original Logic)
                    d2v = savgol_filter(dv_safe, window_length=min(11, len(dv_safe)), polyorder=3, deriv=2)
                    inflection_local_idx = np.argmax(np.abs(d2v))
                    inflection_idx = indices_safe[inflection_local_idx]

                    features['DVP'] = dv_smooth[inflection_idx]
                    if dvpl_fraction is None:
                        features['DVPL_V'] = v_mapped_dv[inflection_idx]
                    features['DVP_Q'] = q_grid_dv[inflection_idx]
                    features['dvp_type'] = 0 # Inflection Point

    else:
        # --- Multi Peak (NCM) Logic (Legacy) ---

        # We use IC features (ICPL_V and ICVL_V) to guide DV search.
        # half_window_dvv/dvp already defined above

        # -- Find DVV (Valley) near ICP --
        if not disable_dvv and not np.isnan(features['ICPL_V']) and features['ICPL_V'] > 0:
            v_target = features['ICPL_V']
            v_upper = v_target + half_window_dvv
            v_lower = v_target - half_window_dvv
            mask_dvv = (v_mapped_dv <= v_upper) & (v_mapped_dv >= v_lower)

            if np.any(mask_dvv):
                indices_dvv = np.where(mask_dvv)[0]
                local_min_idx = np.argmin(dv_smooth[indices_dvv])
                dvv_idx = indices_dvv[local_min_idx]

                # Check boundary/inflection fallback (as per original logic)
                is_at_boundary = (local_min_idx == 0) or (local_min_idx == len(indices_dvv) - 1)
                if is_at_boundary and len(indices_dvv) > 5:
                    d2v = savgol_filter(dv_smooth[indices_dvv], window_length=min(5, len(indices_dvv)), polyorder=2, deriv=2)
                    inflections = np.where(np.diff(np.sign(d2v)))[0]
                    if len(inflections) > 0:
                        dvv_idx = indices_dvv[inflections[0]]

                features['DVV'] = dv_smooth[dvv_idx]
                features['DVVL_V'] = v_mapped_dv[dvv_idx]
                features['DVV_Q'] = q_grid_dv[dvv_idx]

        # -- Find DVP (Peak) near ICV --
        if not dvp_calculated and not np.isnan(features['ICVL_V']) and features['ICVL_V'] > 0:
            v_target = features['ICVL_V']
            v_upper = v_target + half_window_dvp
            v_lower = v_target - half_window_dvp
            mask_dvp = (v_mapped_dv <= v_upper) & (v_mapped_dv >= v_lower)

            if np.any(mask_dvp):
                indices_dvp = np.where(mask_dvp)[0]
                local_max_idx = np.argmax(dv_smooth[indices_dvp])
                dvp_idx = indices_dvp[local_max_idx]

                features['DVP'] = dv_smooth[dvp_idx]
                if dvpl_fraction is None:
                    features['DVPL_V'] = v_mapped_dv[dvp_idx]
                features['DVP_Q'] = q_grid_dv[dvp_idx]
                features['dvp_type'] = 1 # Always assumed peak in this mode

    # Plotting Hook
    if plot_params:
        try:
            plot_ic_dv_curves(
                cycle_num=plot_params.get('cycle_num', 0),
                v_grid_ic=v_grid_ic,
                ic_curve=ic_smooth,
                q_grid_dv=q_grid_dv,
                dv_curve=dv_smooth,
                features=features,
                output_dir=plot_params.get('output_dir', Path('.')),
                cell_id=plot_params.get('cell_id', 'unknown'),
                plot_interval=config.get('plot_interval', 50)
            )
        except Exception as e:
            print(f"Plotting error for {plot_params.get('cell_id')}: {e}")

    if include_curves:
        features['curves'] = {
            'v_grid_ic': v_grid_ic,
            'ic_smooth': ic_smooth,
            'q_grid_dv': q_grid_dv,
            'dv_smooth': dv_smooth
        }

    return features


def extract_charge_ic_dv_features(
    charge_df: pd.DataFrame,
    battery_type: str,
    config: Dict[str, Any],
    plot_params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Specialized IC/DV feature extraction for charge data.

    Logical branches (Based on need_fixed.md):
    1. LFP:
       - ICV: Find the first local minimum to the right of ICP, must be within config['icv_search_range'] (min, max).
       - DVV: Directly take the reciprocal of ICP (DVV = 1 / ICP).
       - DVP: Search within the first 70% of capacity.
    2. NCA:
       - SOC != 100: ICV = 0 (no search).
       - ICV (SOC=100): Prioritize finding the valley between double peaks, otherwise find the first derivative zero point to the right.
    3. NMC:
       - Maintain existing logic or refer to NCA.

    Args:
        charge_df: DataFrame containing 'Voltage(V)' and 'Charge_Capacity(Ah)'.
        battery_type: 'LFP', 'NCA', 'NMC'.
        config: Configuration dictionary containing 'icv_search_range', 'soc', etc.
        plot_params: Plotting parameters.
    """
    features = {
        'ICP': np.nan, 'ICPL_V': np.nan,
        'ICV': np.nan, 'ICVL_V': np.nan,
        'DVV': np.nan, 'DVVL_V': np.nan,
        'DVP': np.nan, 'DVPL_V': np.nan, 'DVP_Q': np.nan,
        'ICP_Area': np.nan, 'ICP_FWHM': np.nan,
        'centroid_voltage': np.nan,
        'ICP_is_missing': 1
    }

    # 1. Data Cleaning & Preprocessing
    if len(charge_df) < 15:
        return features

    # Ensure columns exist (Charge Data usually has Charge_Capacity)
    cap_col = 'Charge_Capacity(Ah)'
    if cap_col not in charge_df.columns:
        # Fallback if mapped
        if 'Discharge_Capacity(Ah)' in charge_df.columns:
            cap_col = 'Discharge_Capacity(Ah)'
        else:
            return features

    df = charge_df.drop_duplicates(subset=['Voltage(V)'])
    df = df.drop_duplicates(subset=[cap_col])

    # Sort by Voltage (Charge: V increases)
    df = df.sort_values(by='Voltage(V)', ascending=True)

    volt_vals = df['Voltage(V)'].values
    cap_vals = df[cap_col].values

    v_min, v_max = np.min(volt_vals), np.max(volt_vals)

    # 2. Resampling & Calculation
    step_v = config.get('ic_step_size', 0.002)
    win_ic = config.get('window_length_ic', 21)

    # Grid for IC (Voltage based)
    num_points = int(abs(v_max - v_min) / step_v)
    num_points = max(100, min(5000, num_points))

    v_grid, q_mapped = equidistant_resample(volt_vals, cap_vals, num_points=num_points)

    # Calculate IC = dQ/dV (Charge: Q increases with V, so dQ/dV > 0)
    dq = np.gradient(q_mapped)
    dv = np.gradient(v_grid)

    with np.errstate(divide='ignore', invalid='ignore'):
        ic_curve = np.divide(dq, dv, out=np.zeros_like(dq), where=dv != 0)

    ic_curve = np.abs(ic_curve) # Ensure positive
    ic_smooth = savgol_filter(ic_curve, window_length=min(win_ic, len(ic_curve)), polyorder=2)
    ic_smooth = np.maximum(ic_smooth, 0)  # [Fix] Clip negative values after smoothing

    # --- New Feature: Centroid Voltage ---
    if np.sum(ic_smooth) > 0:
        centroid_v = np.sum(v_grid * ic_smooth) / np.sum(ic_smooth)
        features['centroid_voltage'] = centroid_v
    else:
        features['centroid_voltage'] = np.nan

    # 2.1 Calculate DV Curve (Needed for plotting and DV features)
    step_q = config.get('dv_step_size', config.get('nominal_capacity', 1.0) * 0.005)
    max_q = np.max(cap_vals)
    num_points_dv = int(max_q / step_q) if step_q > 0 else 1000
    num_points_dv = max(100, min(5000, num_points_dv))

    q_grid_dv, v_mapped_dv = equidistant_resample(cap_vals, volt_vals, num_points=num_points_dv)

    dq_dv = np.gradient(q_grid_dv)
    dv_dv = np.gradient(v_mapped_dv)

    with np.errstate(divide='ignore', invalid='ignore'):
        dv_curve = np.abs(np.divide(dv_dv, dq_dv, out=np.zeros_like(dv_dv), where=dq_dv != 0))

    win_dv = config.get('window_length_dv', 21)
    dv_smooth = savgol_filter(dv_curve, window_length=min(win_dv, len(dv_curve)), polyorder=2)
    dv_smooth = np.maximum(dv_smooth, 0)  # [Fix] Clip negative values after smoothing

    # 3. Find ICP (Main Peak)
    ic_v_range = config.get('voltage_range_ic', (v_min, v_max))
    prom_ic = config.get('prominence_ic', 0.01)
    height_ic = config.get('peak_height_ic', 0.01)

    mask_search = (v_grid >= ic_v_range[0]) & (v_grid <= ic_v_range[1])
    peaks, props = find_peaks(ic_smooth, height=height_ic, prominence=prom_ic)
    valid_peaks = [p for p in peaks if mask_search[p]]

    peak_idx = -1
    if valid_peaks:
        # Score = Height * Width * Prominence
        scores = []
        for p in valid_peaks:
            idx_orig = np.where(peaks == p)[0][0]
            prom = props['prominences'][idx_orig]
            h = ic_smooth[p]
            w = peak_widths(ic_smooth, [p], rel_height=0.5)[0][0]
            scores.append(h * w * prom)
        peak_idx = valid_peaks[np.argmax(scores)]
    else:
        # Fallback: Max in range
        indices = np.where(mask_search)[0]
        if len(indices) > 0:
            peak_idx = indices[np.argmax(ic_smooth[indices])]

    if peak_idx != -1:
        features['ICP'] = ic_smooth[peak_idx]
        features['ICPL_V'] = v_grid[peak_idx]
        features['ICP_is_missing'] = 0

        # --- Calculate ICP Area and FWHM ---
        # FWHM
        w_res = peak_widths(ic_smooth, [peak_idx], rel_height=0.5)
        l_idx_fwhm, r_idx_fwhm = w_res[2][0], w_res[3][0]
        features['ICP_FWHM'] = abs(get_interp_val(v_grid, r_idx_fwhm) - get_interp_val(v_grid, l_idx_fwhm))

        # Area
        ic_area_conf = config.get('ic_area_config')
        if ic_area_conf and ic_area_conf.get('method') == 'fixed_width':
             width_v = ic_area_conf.get('width_v', 0.03)
             v_start = features['ICPL_V'] - width_v
             v_end = features['ICPL_V'] + width_v
             mask_area = (v_grid >= v_start) & (v_grid <= v_end)
             if np.any(mask_area):
                 features['ICP_Area'] = trapezoid(y=ic_smooth[mask_area], x=v_grid[mask_area])
             else:
                 features['ICP_Area'] = 0.0
        else:
             # Default to Curvature if not specified (or add simple range fallback)
             l_bound, r_bound = find_curvature_boundaries(ic_smooth, peak_idx, window_length=11)
             features['ICP_Area'] = calculate_area_with_baseline(
                v_grid[l_bound:r_bound+1],
                ic_smooth[l_bound:r_bound+1]
             )

        # 4. Battery Specific Logic for ICV (Valley) & DVV

        # Helper: Find first valley to the RIGHT
        def find_first_valley_right(start_idx, curve, limit_idx=None):
            end = limit_idx if limit_idx is not None else len(curve) - 1
            for i in range(start_idx, end):
                if curve[i] < curve[i-1] and curve[i] < curve[i+1]:
                    return i
            return -1

        # Helper: Find zero crossing of derivative (Neg -> Pos)
        def find_deriv_zero_crossing_right(start_idx, curve):
            grad = np.gradient(curve)
            for i in range(start_idx, len(grad) - 1):
                if grad[i] < 0 and grad[i+1] > 0:
                    return i
            return -1

        # Helper: Map Voltage to DV features
        def get_dv_at_voltage(target_v):
            # Find closest index in v_mapped_dv
            idx = (np.abs(v_mapped_dv - target_v)).argmin()
            return dv_smooth[idx], q_grid_dv[idx]

        soc_val = config.get('soc', 100)  # Default 100 if not provided

        if battery_type == 'LFP':
            # --- LFP Logic ---
            # 1. ICV: Find the first local minimum to the right of the peak, restricted to the [min_offset, max_offset] range
            icv_range = config.get('icv_search_range', (0.05, 0.5))
            min_offset, max_offset = icv_range

            v_peak = features['ICPL_V']
            v_start_search = v_peak + min_offset
            v_end_search = v_peak + max_offset

            # Convert Voltage range to indices
            idx_start = np.searchsorted(v_grid, v_start_search, side='left')
            idx_end = np.searchsorted(v_grid, v_end_search, side='left')

            # Ensure valid indices
            start_search_idx = max(peak_idx + 1, min(idx_start, len(ic_smooth) - 2))
            end_search_idx = max(start_search_idx + 1, min(idx_end, len(ic_smooth) - 1))

            icv_idx = find_first_valley_right(start_search_idx, ic_smooth, limit_idx=end_search_idx)

            if icv_idx == -1:
                # Fallback 1: Derivative Zero Crossing (Inflection/Flat)
                # Look for transition from negative slope to positive slope
                grad = np.gradient(ic_smooth)
                for i in range(start_search_idx, end_search_idx):
                    if i < len(grad) - 1 and grad[i] < 0 and grad[i+1] > 0:
                        icv_idx = i
                        break

            if icv_idx != -1:
                features['ICV'] = ic_smooth[icv_idx]
                features['ICVL_V'] = v_grid[icv_idx]
            else:
                # Fallback 2: Use minimum value in search range
                if start_search_idx < end_search_idx:
                    range_vals = ic_smooth[start_search_idx:end_search_idx+1]
                    if len(range_vals) > 0:
                        min_local_idx = np.argmin(range_vals)
                        icv_idx = start_search_idx + min_local_idx

                        features['ICV'] = ic_smooth[icv_idx]
                        features['ICVL_V'] = v_grid[icv_idx]
                    else:
                        features['ICV'] = 0.0
                        features['ICVL_V'] = 0.0
                else:
                    features['ICV'] = 0.0 # Not found in range
                    features['ICVL_V'] = 0.0

            # 2. DVV: Take the reciprocal of ICP (DV = 1/IC)
            if features['ICP'] > 1e-6:
                features['DVV'] = 1.0 / features['ICP']
            else:
                features['DVV'] = 0.0
            features['DVVL_V'] = features['ICPL_V'] # Same location as ICP
            # Map Q location
            _, dvv_q = get_dv_at_voltage(features['ICPL_V'])
            features['DVV_Q'] = dvv_q

            # 3. DVP: Search within the first 70% of the capacity
            safe_q_max = max_q * 0.7
            mask_dvp = (q_grid_dv > 0) & (q_grid_dv < safe_q_max)

            if np.any(mask_dvp):
                indices_safe = np.where(mask_dvp)[0]
                dv_safe = dv_smooth[indices_safe]
                # Find Peak in this segment
                dv_peaks, props_dv = find_peaks(dv_safe, prominence=0.005) # Lower prominence allowed

                if len(dv_peaks) > 0:
                    best_local = np.argmax(props_dv['prominences'])
                    best_idx_dv = indices_safe[dv_peaks[best_local]]
                    features['DVP'] = dv_smooth[best_idx_dv]
                    features['DVPL_V'] = v_mapped_dv[best_idx_dv]
                    features['DVP_Q'] = q_grid_dv[best_idx_dv]

        elif battery_type == 'NCA' or battery_type == 'NMC':
            # --- NCA/NMC Logic ---

            # 1. ICV
            if soc_val != 100:
                features['ICV'] = 0.0
                features['ICVL_V'] = 0.0
            else:
                # Priority strategy: If there is another peak to the right, find the lowest point between the two peaks
                # Alternative strategy: first derivative zero point to the right

                # Check for secondary peaks to the right
                right_peaks = [p for p in valid_peaks if p > peak_idx]

                if right_peaks:
                    # Found a peak to the right, search valley between current peak and next peak
                    next_peak_idx = right_peaks[0]
                    # Search range: peak_idx to next_peak_idx
                    valley_segment = ic_smooth[peak_idx:next_peak_idx+1]
                    local_min_idx = np.argmin(valley_segment)
                    icv_idx = peak_idx + local_min_idx

                    features['ICV'] = ic_smooth[icv_idx]
                    features['ICVL_V'] = v_grid[icv_idx]
                else:
                    # No peak to the right, use zero-crossing
                    zero_cross_idx = find_deriv_zero_crossing_right(peak_idx + 1, ic_smooth)
                    if zero_cross_idx != -1:
                        features['ICV'] = ic_smooth[zero_cross_idx]
                        features['ICVL_V'] = v_grid[zero_cross_idx]
                    else:
                        # Final fallback: First valley search
                        icv_idx = find_first_valley_right(peak_idx + 1, ic_smooth)
                        if icv_idx != -1:
                             features['ICV'] = ic_smooth[icv_idx]
                             features['ICVL_V'] = v_grid[icv_idx]
                        else:
                             features['ICV'] = 0.0
                             features['ICVL_V'] = 0.0

            # 2. DVP/DVV Mapping (Default behavior for NMC/NCA)
            # DVP corresponds to ICV (if found)
            if features['ICV'] > 0:
                dvp_val, dvp_q = get_dv_at_voltage(features['ICVL_V'])
                features['DVP'] = dvp_val
                features['DVPL_V'] = features['ICVL_V']
                features['DVP_Q'] = dvp_q

            # DVV corresponds to ICP
            dvv_val, dvv_q = get_dv_at_voltage(features['ICPL_V'])
            features['DVV'] = dvv_val
            features['DVVL_V'] = features['ICPL_V']
            features['DVV_Q'] = dvv_q

    # Plotting Hook
    if plot_params:
        try:
            plot_ic_dv_curves(
                cycle_num=plot_params.get('cycle_num', 0),
                v_grid_ic=v_grid,
                ic_curve=ic_smooth,
                q_grid_dv=q_grid_dv,
                dv_curve=dv_smooth,
                features=features,
                output_dir=plot_params.get('output_dir', Path('.')),
                cell_id=plot_params.get('cell_id', 'unknown'),
                plot_interval=plot_params.get('plot_interval', 50)
            )
        except Exception as e:
            print(f"Plotting error for {plot_params.get('cell_id')}: {e}")

    return features

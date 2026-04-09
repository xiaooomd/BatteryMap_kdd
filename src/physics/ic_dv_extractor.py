"""
Physics-oriented IC/DV feature extraction helpers.

This module is the authority for phase identification and IC/DV curve analysis.
Legacy imports from ``src.utils.feature_tools`` remain supported via re-export.
"""

from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path

import pandas as pd
import numpy as np
from scipy.signal import find_peaks, savgol_filter, peak_widths
from scipy.integrate import trapezoid

from src.utils.math_tools import (
    equidistant_resample,
    calculate_area_with_baseline,
    get_interp_val,
    find_curvature_boundaries,
)
from src.utils.plot_tools import plot_ic_dv_curves


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

    Args:
        discharge_df: DataFrame for discharge phase.
        config: Dictionary containing battery-specific parameters.
        plot_params: Optional dict for plotting.
        include_curves: If True, includes raw curve data in the returned dictionary.

    Returns:
        Dict[str, Any]: Dictionary of IC/DV features.
    """
    features = {
        'ICP': np.nan, 'ICPL_V': np.nan, 'ICV': np.nan, 'ICVL_V': np.nan,
        'DVP': np.nan, 'DVPL_V': np.nan, 'DVP_Q': np.nan,
        'DVV': np.nan, 'DVVL_V': np.nan, 'DVV_Q': np.nan,
        'ICP_Area': np.nan, 'ICP_FWHM': np.nan,
        'centroid_voltage': np.nan,
        'peak_mode': np.nan,
        'dvp_type': np.nan,
        'ICP_is_missing': 1
    }

    if len(discharge_df) < 15:
        return features

    peak_mode = config.get('peak_mode', 2)
    features['peak_mode'] = peak_mode

    nominal_cap = config.get('nominal_capacity', 1.0)
    ic_v_range = config.get('voltage_range_ic', (2.8, 4.2))
    prom_ic = config.get('prominence_ic', 0.01)
    step_v = config.get('ic_step_size', 0.002)
    step_q = config.get('dv_step_size', nominal_cap * 0.005)
    win_ic = config.get('window_length_ic', 21)
    win_dv = config.get('window_length_dv', 21)
    height_ic = config.get('peak_height_ic', 0.01)
    search_win = config.get('search_window_v', 0.2)
    search_win_dvv = config.get('search_window_dvv', search_win)
    search_win_dvp = config.get('search_window_dvp', search_win)
    icv_offset_lower = config.get('icv_search_offset_lower', 0.05)
    icv_offset_upper = config.get('icv_search_offset_upper', 0.5)
    force_icp_zero = config.get('force_icp_zero', False)
    force_icp_fwhm_zero = config.get('force_icp_fwhm_zero', False)
    force_icv_zero = config.get('force_icv_zero', False)

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
    max_q = np.max(cap_vals)

    num_points_ic = int(abs(v_max - v_min) / step_v) if step_v > 0 else 1000
    num_points_ic = max(100, min(5000, num_points_ic))
    v_grid, q_mapped_ic = equidistant_resample(volt_vals, cap_vals, num_points=num_points_ic)

    dv_ic_grid = np.gradient(v_grid)
    dq_ic_grid = np.gradient(q_mapped_ic)
    with np.errstate(divide='ignore', invalid='ignore'):
        ic_curve = np.abs(np.divide(dq_ic_grid, dv_ic_grid, out=np.zeros_like(dq_ic_grid), where=dv_ic_grid != 0))

    ic_curve[ic_curve < 0] = 0
    ic_window = min(win_ic, len(ic_curve))
    if ic_window % 2 == 0:
        ic_window -= 1
    ic_window = max(ic_window, 3)
    ic_smooth = savgol_filter(ic_curve, window_length=ic_window, polyorder=2)
    ic_smooth = np.maximum(ic_smooth, 0)

    if np.sum(ic_smooth) > 0:
        centroid_v = np.sum(v_grid * ic_smooth) / np.sum(ic_smooth)
        features['centroid_voltage'] = centroid_v

    num_points_dv = int(max_q / step_q) if step_q > 0 else 1000
    num_points_dv = max(100, min(5000, num_points_dv))
    q_grid_dv, v_mapped_dv = equidistant_resample(cap_vals, volt_vals, num_points=num_points_dv)

    dq_dv_grid = np.gradient(q_grid_dv)
    dv_dv_grid = np.gradient(v_mapped_dv)
    with np.errstate(divide='ignore', invalid='ignore'):
        dv_curve = np.abs(np.divide(dv_dv_grid, dq_dv_grid, out=np.zeros_like(dv_dv_grid), where=dq_dv_grid != 0))

    dv_curve[dv_curve < 0] = 0
    dv_window = min(win_dv, len(dv_curve))
    if dv_window % 2 == 0:
        dv_window -= 1
    dv_window = max(dv_window, 3)
    dv_smooth = savgol_filter(dv_curve, window_length=dv_window, polyorder=2)
    dv_smooth = np.maximum(dv_smooth, 0)

    battery_type = config.get('battery_type', 'NMC')
    if peak_mode == 1:
        battery_type = 'LFP'
    elif battery_type not in ['LFP', 'NCA', 'NMC']:
        battery_type = 'NMC'

    peak_idx = -1
    if not force_icp_zero:
        mask_search = (v_grid >= ic_v_range[0]) & (v_grid <= ic_v_range[1])
        peaks, props = find_peaks(ic_smooth, height=height_ic, prominence=prom_ic)
        valid_peaks = [p for p in peaks if mask_search[p]]

        if valid_peaks:
            scores = []
            for p in valid_peaks:
                idx_orig = np.where(peaks == p)[0][0]
                prom = props['prominences'][idx_orig]
                h = ic_smooth[p]
                w = peak_widths(ic_smooth, [p], rel_height=0.5)[0][0]
                scores.append(h * w * prom)
            peak_idx = valid_peaks[np.argmax(scores)]
        else:
            indices = np.where(mask_search)[0]
            if len(indices) > 0:
                peak_idx = indices[np.argmax(ic_smooth[indices])]

    if peak_idx != -1:
        features['ICP'] = ic_smooth[peak_idx]
        features['ICPL_V'] = v_grid[peak_idx]
        features['ICP_is_missing'] = 0

        w_res = peak_widths(ic_smooth, [peak_idx], rel_height=0.5)
        l_idx_fwhm, r_idx_fwhm = w_res[2][0], w_res[3][0]
        features['ICP_FWHM'] = abs(get_interp_val(v_grid, r_idx_fwhm) - get_interp_val(v_grid, l_idx_fwhm))

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
            l_bound, r_bound = find_curvature_boundaries(ic_smooth, peak_idx, window_length=11)
            features['ICP_Area'] = calculate_area_with_baseline(
                v_grid[l_bound:r_bound + 1],
                ic_smooth[l_bound:r_bound + 1]
            )

        def find_first_valley_right(start_idx, curve, limit_idx=None):
            end = limit_idx if limit_idx is not None else len(curve) - 1
            for i in range(start_idx, end):
                if curve[i] < curve[i - 1] and curve[i] < curve[i + 1]:
                    return i
            return -1

        def find_deriv_zero_crossing_right(start_idx, curve):
            grad = np.gradient(curve)
            for i in range(start_idx, len(grad) - 1):
                if grad[i] < 0 and grad[i + 1] > 0:
                    return i
            return -1

        def get_dv_at_voltage(target_v):
            idx = (np.abs(v_mapped_dv - target_v)).argmin()
            return dv_smooth[idx], q_grid_dv[idx]

        soc_val = config.get('soc', 100)

        if battery_type == 'LFP':
            icv_range = config.get('icv_search_range', (0.05, 0.5))
            min_offset, max_offset = icv_range

            v_peak = features['ICPL_V']
            v_start_search = v_peak + min_offset
            v_end_search = v_peak + max_offset

            idx_start = np.searchsorted(v_grid, v_start_search, side='left')
            idx_end = np.searchsorted(v_grid, v_end_search, side='left')

            start_search_idx = max(peak_idx + 1, min(idx_start, len(ic_smooth) - 2))
            end_search_idx = max(start_search_idx + 1, min(idx_end, len(ic_smooth) - 1))

            icv_idx = find_first_valley_right(start_search_idx, ic_smooth, limit_idx=end_search_idx)

            if icv_idx == -1:
                grad = np.gradient(ic_smooth)
                for i in range(start_search_idx, end_search_idx):
                    if i < len(grad) - 1 and grad[i] < 0 and grad[i + 1] > 0:
                        icv_idx = i
                        break

            if icv_idx != -1:
                features['ICV'] = ic_smooth[icv_idx]
                features['ICVL_V'] = v_grid[icv_idx]
            else:
                if start_search_idx < end_search_idx:
                    range_vals = ic_smooth[start_search_idx:end_search_idx + 1]
                    if len(range_vals) > 0:
                        min_local_idx = np.argmin(range_vals)
                        icv_idx = start_search_idx + min_local_idx
                        features['ICV'] = ic_smooth[icv_idx]
                        features['ICVL_V'] = v_grid[icv_idx]
                    else:
                        features['ICV'] = 0.0
                        features['ICVL_V'] = 0.0
                else:
                    features['ICV'] = 0.0
                    features['ICVL_V'] = 0.0

            if features['ICP'] > 1e-6:
                features['DVV'] = 1.0 / features['ICP']
            else:
                features['DVV'] = 0.0
            features['DVVL_V'] = features['ICPL_V']
            _, dvv_q = get_dv_at_voltage(features['ICPL_V'])
            features['DVV_Q'] = dvv_q

            safe_q_max = max_q * 0.7
            mask_dvp = (q_grid_dv > 0) & (q_grid_dv < safe_q_max)

            if np.any(mask_dvp):
                indices_safe = np.where(mask_dvp)[0]
                dv_safe = dv_smooth[indices_safe]
                dv_peaks, props_dv = find_peaks(dv_safe, prominence=0.005)

                if len(dv_peaks) > 0:
                    best_local = np.argmax(props_dv['prominences'])
                    best_idx_dv = indices_safe[dv_peaks[best_local]]
                    features['DVP'] = dv_smooth[best_idx_dv]
                    features['DVPL_V'] = v_mapped_dv[best_idx_dv]
                    features['DVP_Q'] = q_grid_dv[best_idx_dv]

        elif battery_type == 'NCA' or battery_type == 'NMC':
            if soc_val != 100:
                features['ICV'] = 0.0
                features['ICVL_V'] = 0.0
            else:
                right_peaks = [p for p in valid_peaks if p > peak_idx]

                if right_peaks:
                    next_peak_idx = right_peaks[0]
                    valley_segment = ic_smooth[peak_idx:next_peak_idx + 1]
                    local_min_idx = np.argmin(valley_segment)
                    icv_idx = peak_idx + local_min_idx

                    features['ICV'] = ic_smooth[icv_idx]
                    features['ICVL_V'] = v_grid[icv_idx]
                else:
                    zero_cross_idx = find_deriv_zero_crossing_right(peak_idx + 1, ic_smooth)
                    if zero_cross_idx != -1:
                        features['ICV'] = ic_smooth[zero_cross_idx]
                        features['ICVL_V'] = v_grid[zero_cross_idx]
                    else:
                        icv_idx = find_first_valley_right(peak_idx + 1, ic_smooth)
                        if icv_idx != -1:
                            features['ICV'] = ic_smooth[icv_idx]
                            features['ICVL_V'] = v_grid[icv_idx]
                        else:
                            features['ICV'] = 0.0
                            features['ICVL_V'] = 0.0

            if features['ICV'] > 0:
                dvp_val, dvp_q = get_dv_at_voltage(features['ICVL_V'])
                features['DVP'] = dvp_val
                features['DVPL_V'] = features['ICVL_V']
                features['DVP_Q'] = dvp_q

            dvv_val, dvv_q = get_dv_at_voltage(features['ICPL_V'])
            features['DVV'] = dvv_val
            features['DVVL_V'] = features['ICPL_V']
            features['DVV_Q'] = dvv_q

    if force_icp_fwhm_zero:
        features['ICP_FWHM'] = 0.0
    if force_icv_zero:
        features['ICV'] = 0.0
        features['ICVL_V'] = 0.0

    if include_curves:
        features['v_grid_ic'] = v_grid
        features['ic_curve'] = ic_smooth
        features['q_grid_dv'] = q_grid_dv
        features['dv_curve'] = dv_smooth

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
        except Exception as exc:
            print(f"Plotting error for {plot_params.get('cell_id')}: {exc}")

    return features


def extract_charge_ic_dv_features(
    charge_df: pd.DataFrame,
    config: Dict[str, Any],
    plot_params: Optional[Dict[str, Any]] = None,
    include_curves: bool = False
) -> Dict[str, Any]:
    """
    Calculates IC/DV features for charge curves.
    """
    features = {
        'ICP': np.nan, 'ICPL_V': np.nan, 'ICV': np.nan, 'ICVL_V': np.nan,
        'DVP': np.nan, 'DVPL_V': np.nan, 'DVP_Q': np.nan,
        'DVV': np.nan, 'DVVL_V': np.nan, 'DVV_Q': np.nan,
        'ICP_Area': np.nan, 'ICP_FWHM': np.nan,
        'centroid_voltage': np.nan,
        'peak_mode': np.nan,
        'dvp_type': np.nan,
        'ICP_is_missing': 1
    }

    if len(charge_df) < 15:
        return features

    peak_mode = config.get('peak_mode', 2)
    features['peak_mode'] = peak_mode
    battery_type = config.get('battery_type', 'NMC')
    if peak_mode == 1:
        battery_type = 'LFP'

    nominal_cap = config.get('nominal_capacity', 1.0)
    ic_v_range = config.get('voltage_range_ic', (2.8, 4.2))
    prom_ic = config.get('prominence_ic', 0.01)
    step_v = config.get('ic_step_size', 0.002)
    step_q = config.get('dv_step_size', nominal_cap * 0.005)
    win_ic = config.get('window_length_ic', 21)
    win_dv = config.get('window_length_dv', 21)
    height_ic = config.get('peak_height_ic', 0.01)
    icv_offset_lower = config.get('icv_search_offset_lower', 0.05)
    icv_offset_upper = config.get('icv_search_offset_upper', 0.5)

    df = charge_df.copy().drop_duplicates(subset=['Voltage(V)']).drop_duplicates(subset=['Charge_Capacity(Ah)'])
    cap_vals = df['Charge_Capacity(Ah)'].values
    volt_vals = df['Voltage(V)'].values
    v_min, v_max = np.min(volt_vals), np.max(volt_vals)
    max_q = np.max(cap_vals)

    num_points_ic = int(abs(v_max - v_min) / step_v) if step_v > 0 else 1000
    num_points_ic = max(100, min(5000, num_points_ic))
    v_grid, q_mapped_ic = equidistant_resample(volt_vals, cap_vals, num_points=num_points_ic)

    dv_ic_grid = np.gradient(v_grid)
    dq_ic_grid = np.gradient(q_mapped_ic)
    with np.errstate(divide='ignore', invalid='ignore'):
        ic_curve = np.abs(np.divide(dq_ic_grid, dv_ic_grid, out=np.zeros_like(dq_ic_grid), where=dv_ic_grid != 0))
    ic_curve[ic_curve < 0] = 0

    ic_window = min(win_ic, len(ic_curve))
    if ic_window % 2 == 0:
        ic_window -= 1
    ic_window = max(ic_window, 3)
    ic_smooth = savgol_filter(ic_curve, window_length=ic_window, polyorder=2)
    ic_smooth = np.maximum(ic_smooth, 0)

    if np.sum(ic_smooth) > 0:
        features['centroid_voltage'] = np.sum(v_grid * ic_smooth) / np.sum(ic_smooth)

    num_points_dv = int(max_q / step_q) if step_q > 0 else 1000
    num_points_dv = max(100, min(5000, num_points_dv))
    q_grid_dv, v_mapped_dv = equidistant_resample(cap_vals, volt_vals, num_points=num_points_dv)
    dq_dv_grid = np.gradient(q_grid_dv)
    dv_dv_grid = np.gradient(v_mapped_dv)
    with np.errstate(divide='ignore', invalid='ignore'):
        dv_curve = np.abs(np.divide(dv_dv_grid, dq_dv_grid, out=np.zeros_like(dv_dv_grid), where=dq_dv_grid != 0))
    dv_curve[dv_curve < 0] = 0

    dv_window = min(win_dv, len(dv_curve))
    if dv_window % 2 == 0:
        dv_window -= 1
    dv_window = max(dv_window, 3)
    dv_smooth = savgol_filter(dv_curve, window_length=dv_window, polyorder=2)
    dv_smooth = np.maximum(dv_smooth, 0)

    peaks, props = find_peaks(ic_smooth, height=height_ic, prominence=prom_ic)
    valid_peaks = [p for p in peaks if ic_v_range[0] <= v_grid[p] <= ic_v_range[1]]
    peak_idx = -1

    if valid_peaks:
        scores = []
        for p in valid_peaks:
            idx_orig = np.where(peaks == p)[0][0]
            prom = props['prominences'][idx_orig]
            h = ic_smooth[p]
            w = peak_widths(ic_smooth, [p], rel_height=0.5)[0][0]
            scores.append(h * w * prom)
        peak_idx = valid_peaks[np.argmax(scores)]
    else:
        mask_search = (v_grid >= ic_v_range[0]) & (v_grid <= ic_v_range[1])
        indices = np.where(mask_search)[0]
        if len(indices) > 0:
            peak_idx = indices[np.argmax(ic_smooth[indices])]

    if peak_idx != -1:
        features['ICP'] = ic_smooth[peak_idx]
        features['ICPL_V'] = v_grid[peak_idx]
        features['ICP_is_missing'] = 0

        w_res = peak_widths(ic_smooth, [peak_idx], rel_height=0.5)
        l_idx_fwhm, r_idx_fwhm = w_res[2][0], w_res[3][0]
        features['ICP_FWHM'] = abs(get_interp_val(v_grid, r_idx_fwhm) - get_interp_val(v_grid, l_idx_fwhm))

        l_bound, r_bound = find_curvature_boundaries(ic_smooth, peak_idx, window_length=11)
        features['ICP_Area'] = calculate_area_with_baseline(
            v_grid[l_bound:r_bound + 1],
            ic_smooth[l_bound:r_bound + 1]
        )

        def find_first_valley_right(start_idx, curve, limit_idx=None):
            end = limit_idx if limit_idx is not None else len(curve) - 1
            for i in range(start_idx, end):
                if curve[i] < curve[i - 1] and curve[i] < curve[i + 1]:
                    return i
            return -1

        def find_deriv_zero_crossing_right(start_idx, curve):
            grad = np.gradient(curve)
            for i in range(start_idx, len(grad) - 1):
                if grad[i] < 0 and grad[i + 1] > 0:
                    return i
            return -1

        def get_dv_at_voltage(target_v):
            idx = (np.abs(v_mapped_dv - target_v)).argmin()
            return dv_smooth[idx], q_grid_dv[idx]

        if battery_type == 'LFP':
            v_peak = features['ICPL_V']
            idx_start = np.searchsorted(v_grid, v_peak + icv_offset_lower, side='left')
            idx_end = np.searchsorted(v_grid, v_peak + icv_offset_upper, side='left')
            start_search_idx = max(peak_idx + 1, min(idx_start, len(ic_smooth) - 2))
            end_search_idx = max(start_search_idx + 1, min(idx_end, len(ic_smooth) - 1))

            icv_idx = find_first_valley_right(start_search_idx, ic_smooth, limit_idx=end_search_idx)
            if icv_idx == -1:
                zero_cross_idx = find_deriv_zero_crossing_right(start_search_idx, ic_smooth)
                if zero_cross_idx != -1 and zero_cross_idx <= end_search_idx:
                    icv_idx = zero_cross_idx

            if icv_idx != -1:
                features['ICV'] = ic_smooth[icv_idx]
                features['ICVL_V'] = v_grid[icv_idx]

            if features['ICP'] > 1e-6:
                features['DVV'] = 1.0 / features['ICP']
            else:
                features['DVV'] = 0.0
            features['DVVL_V'] = features['ICPL_V']
            _, dvv_q = get_dv_at_voltage(features['ICPL_V'])
            features['DVV_Q'] = dvv_q

            safe_q_max = max_q * 0.7
            mask_dvp = (q_grid_dv > 0) & (q_grid_dv < safe_q_max)
            if np.any(mask_dvp):
                indices_safe = np.where(mask_dvp)[0]
                dv_safe = dv_smooth[indices_safe]
                dv_peaks, props_dv = find_peaks(dv_safe, prominence=0.005)
                if len(dv_peaks) > 0:
                    best_local = np.argmax(props_dv['prominences'])
                    best_idx_dv = indices_safe[dv_peaks[best_local]]
                    features['DVP'] = dv_smooth[best_idx_dv]
                    features['DVPL_V'] = v_mapped_dv[best_idx_dv]
                    features['DVP_Q'] = q_grid_dv[best_idx_dv]
        else:
            right_peaks = [p for p in valid_peaks if p > peak_idx]
            if right_peaks:
                next_peak_idx = right_peaks[0]
                valley_segment = ic_smooth[peak_idx:next_peak_idx + 1]
                local_min_idx = np.argmin(valley_segment)
                icv_idx = peak_idx + local_min_idx
                features['ICV'] = ic_smooth[icv_idx]
                features['ICVL_V'] = v_grid[icv_idx]
            else:
                zero_cross_idx = find_deriv_zero_crossing_right(peak_idx + 1, ic_smooth)
                if zero_cross_idx != -1:
                    features['ICV'] = ic_smooth[zero_cross_idx]
                    features['ICVL_V'] = v_grid[zero_cross_idx]
                else:
                    icv_idx = find_first_valley_right(peak_idx + 1, ic_smooth)
                    if icv_idx != -1:
                        features['ICV'] = ic_smooth[icv_idx]
                        features['ICVL_V'] = v_grid[icv_idx]

            if features['ICV'] > 0:
                dvp_val, dvp_q = get_dv_at_voltage(features['ICVL_V'])
                features['DVP'] = dvp_val
                features['DVPL_V'] = features['ICVL_V']
                features['DVP_Q'] = dvp_q

            dvv_val, dvv_q = get_dv_at_voltage(features['ICPL_V'])
            features['DVV'] = dvv_val
            features['DVVL_V'] = features['ICPL_V']
            features['DVV_Q'] = dvv_q

    if include_curves:
        features['v_grid_ic'] = v_grid
        features['ic_curve'] = ic_smooth
        features['q_grid_dv'] = q_grid_dv
        features['dv_curve'] = dv_smooth

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
        except Exception as exc:
            print(f"Plotting error for {plot_params.get('cell_id')}: {exc}")

    return features


__all__ = [
    "identify_phases",
    "extract_ic_dv_features",
    "extract_charge_ic_dv_features",
]

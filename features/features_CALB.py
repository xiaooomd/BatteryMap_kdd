import pickle
import warnings
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import skew
from scipy.integrate import trapezoid
from tqdm import tqdm

# Add project root to path to allow importing src
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.utils.math_tools import fit_cv_decay, get_interp_val
from src.utils.feature_tools import identify_phases, extract_ic_dv_features

# Suppress FutureWarnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)


def _calculate_direct_features(
    charge_df: pd.DataFrame,
    discharge_df: pd.DataFrame,
    rest_df_after_discharge: pd.DataFrame,
    cycle_num: int,
    battery_data: Dict[str, Any]
) -> Dict[str, Any]:
    """Calculates direct capacity, energy, and efficiency features.

    Args:
        charge_df: DataFrame for the charge phase.
        discharge_df: DataFrame for the discharge phase.
        rest_df_after_discharge: DataFrame for rest after discharge.
        cycle_num: The current cycle number.
        battery_data: Dictionary containing battery metadata.

    Returns:
        Dict[str, Any]: Calculated direct features.
    """
    features = {}

    # --- A. Basic Capacity Features ---
    features['Cycle_Number'] = cycle_num

    # Workload Type
    if not charge_df.empty and not discharge_df.empty:
        if charge_df['Time(s)'].iloc[0] < discharge_df['Time(s)'].iloc[0]:
            features['Workload_Type'] = '0'  # Charge First
        else:
            features['Workload_Type'] = '1'  # Discharge First
    elif not charge_df.empty:
        features['Workload_Type'] = '0'
    elif not discharge_df.empty:
        features['Workload_Type'] = '1'
    else:
        features['Workload_Type'] = '-1'

    # Calculate Capacity using Ampere-hour Integration (current over time)
    # Replaced np.trapz with scipy.integrate.trapezoid
    if not discharge_df.empty:
        # Filter noise current at boundaries
        valid_dis = discharge_df[discharge_df['Current(A)'].abs() > 1e-4]
        if not valid_dis.empty:
            q_dis = trapezoid(valid_dis['Current(A)'].abs(), x=valid_dis['Time(s)']) / 3600.0
        else:
            q_dis = trapezoid(discharge_df['Current(A)'].abs(), x=discharge_df['Time(s)']) / 3600.0
    else:
        q_dis = 0

    if not charge_df.empty:
        # Filter noise current at boundaries
        valid_chg = charge_df[charge_df['Current(A)'].abs() > 1e-4]
        if not valid_chg.empty:
            q_chg = trapezoid(valid_chg['Current(A)'].abs(), x=valid_chg['Time(s)']) / 3600.0
        else:
            q_chg = trapezoid(charge_df['Current(A)'].abs(), x=charge_df['Time(s)']) / 3600.0
    else:
        q_chg = 0

    features['Discharge_Capacity(Ah)'] = q_dis
    features['Charge_Capacity(Ah)'] = q_chg

    # Coulombic Efficiency
    if q_chg > 0:
        features['Coulombic_Efficiency'] = q_dis / q_chg
    else:
        features['Coulombic_Efficiency'] = 0

    # --- Energy & Efficiency (Integration) ---
    # Charge Energy
    if not charge_df.empty:
        p_charge = charge_df['Voltage(V)'] * charge_df['Current(A)'].abs()
        e_charge_ws = trapezoid(p_charge, x=charge_df['Time(s)'])
        features['Charge_Energy(Wh)'] = e_charge_ws / 3600.0
    else:
        features['Charge_Energy(Wh)'] = 0

    # Discharge Energy
    if not discharge_df.empty:
        p_discharge = discharge_df['Voltage(V)'] * discharge_df['Current(A)'].abs()
        e_discharge_ws = trapezoid(p_discharge, x=discharge_df['Time(s)'])
        features['Discharge_Energy(Wh)'] = e_discharge_ws / 3600.0
    else:
        features['Discharge_Energy(Wh)'] = 0

    # Energy Efficiency
    if features['Charge_Energy(Wh)'] > 0:
        features['Energy_Efficiency'] = features['Discharge_Energy(Wh)'] / features['Charge_Energy(Wh)']
    else:
        features['Energy_Efficiency'] = 0

    # Rest Time
    if not rest_df_after_discharge.empty:
        rest_times = rest_df_after_discharge['Time(s)']
        features['Rest_Time(s)'] = rest_times.iloc[-1] - rest_times.iloc[0]
    else:
        features['Rest_Time(s)'] = 0

    # C-Rates & Current Stats
    nominal_capacity = battery_data.get('nominal_capacity_in_Ah', 2.0)

    # Discharge C-rate (Calculated from real data)
    if not discharge_df.empty:
        # Use valid discharge data for stats
        valid_dis = discharge_df[discharge_df['Current(A)'].abs() > 1e-4]
        if valid_dis.empty:
            valid_dis = discharge_df

        # Time-weighted Average Discharge C-rate
        # I_avg = Q_dis / Total_Time
        total_dis_time = valid_dis['Time(s)'].iloc[-1] - valid_dis['Time(s)'].iloc[0]
        if total_dis_time > 1.0:
            avg_dis_current = (q_dis * 3600.0) / total_dis_time
            features['discharge_c_rate'] = avg_dis_current / nominal_capacity
        else:
            features['discharge_c_rate'] = 0.0
    else:
        features['discharge_c_rate'] = 0.0

    # Charge Current Features (New Logic)
    if not charge_df.empty:
        # Use valid charge data for stats to avoid zero-padding effects
        valid_chg = charge_df[charge_df['Current(A)'].abs() > 1e-4]
        if valid_chg.empty:
            valid_chg = charge_df

        # 1. Time-weighted Average Charge C-rate
        # I_avg = Q_chg / Total_Time
        total_chg_time = valid_chg['Time(s)'].iloc[-1] - valid_chg['Time(s)'].iloc[0]
        if total_chg_time > 1.0:
            avg_current = (q_chg * 3600.0) / total_chg_time
            features['avg_charge_c_rate'] = avg_current / nominal_capacity
        else:
            features['avg_charge_c_rate'] = 0.0

        # 2. Max Charge Current (Lithium Plating Risk)
        features['max_I_charge(A)'] = valid_chg['Current(A)'].abs().max()

        # 3. Current Variance
        features['var_I_charge'] = valid_chg['Current(A)'].var()
    else:
        features['avg_charge_c_rate'] = 0.0
        features['max_I_charge(A)'] = 0.0
        features['var_I_charge'] = 0.0

    # --- B. Charging Phase & CV Dynamics ---
    features['CV_Current_Tau'] = 0

    if not charge_df.empty:
        features['ICHV(V)'] = charge_df['Voltage(V)'].iloc[0]
        features['UVP_time(s)'] = charge_df['Time(s)'].iloc[-1]
        features['UVP(V)'] = battery_data['max_voltage_limit_in_V']

        v_upper_limit = battery_data['max_voltage_limit_in_V']
        charge_voltage = charge_df['Voltage(V)']
        cv_threshold = v_upper_limit - 0.01

        # Check for CV phase
        # Strictly require Voltage >= Limit AND Current > Threshold (to exclude rest)
        if charge_voltage.max() >= cv_threshold:
            cv_mask = (charge_df['Voltage(V)'] >= cv_threshold) & (charge_df['Current(A)'] > 0.005)
            cv_df = charge_df[cv_mask]

            if not cv_df.empty:
                # Time of CV
                t_cv_start = cv_df['Time(s)'].iloc[0]
                t_cv_end = cv_df['Time(s)'].iloc[-1]
                features['TCVC(s)'] = t_cv_end - t_cv_start

                # Time of CC (Approximate as Total - CV)
                features['TCCC(s)'] = (charge_df['Time(s)'].iloc[-1] - charge_df['Time(s)'].iloc[0]) - features['TCVC(s)']

                # Robust CV Tau Fitting
                cv_current = cv_df['Current(A)'].values
                cv_time = cv_df['Time(s)'].values

                valid_tau_mask = cv_current > 0.001
                if np.sum(valid_tau_mask) > 10:
                    features['CV_Current_Tau'] = fit_cv_decay(
                        cv_time[valid_tau_mask], cv_current[valid_tau_mask]
                    )
            else:
                features['TCCC(s)'] = charge_df['Time(s)'].iloc[-1] - charge_df['Time(s)'].iloc[0]
                features['TCVC(s)'] = 0
        else:
            features['TCCC(s)'] = charge_df['Time(s)'].iloc[-1] - charge_df['Time(s)'].iloc[0]
            features['TCVC(s)'] = 0
    else:
        features.update({
            'ICHV(V)': 0, 'UVP_time(s)': 0, 'TCCC(s)': 0, 'TCVC(s)': 0,
            'UVP(V)': battery_data['max_voltage_limit_in_V']
        })

    # --- C. Discharging Phase Features ---
    if not discharge_df.empty:
        features['IDV(V)'] = discharge_df['Voltage(V)'].iloc[0]
        features['LVP_time(s)'] = discharge_df['Time(s)'].iloc[-1]
        features['var_I_discharge'] = discharge_df['Current(A)'].var()
        features['var_V_discharge'] = discharge_df['Voltage(V)'].var()
        features['median_V_discharge(V)'] = discharge_df['Voltage(V)'].median()
        features['total_discharge_time(s)'] = (
            discharge_df['Time(s)'].iloc[-1] - discharge_df['Time(s)'].iloc[0]
        )
    else:
        features.update({
            'IDV(V)': 0, 'LVP_time(s)': 0, 'var_I_discharge': 0,
            'var_V_discharge': 0, 'median_V_discharge(V)': 0,
            'total_discharge_time(s)': 0,
        })

    features['LVP(V)'] = battery_data['min_voltage_limit_in_V']

    return features


def _calculate_advanced_features(
    discharge_df: pd.DataFrame,
    v_rest_end_before_discharge: Optional[float],
    tccc_val: float,
    tcvc_val: float
) -> Dict[str, Any]:
    """Calculates internal resistance and statistical features.

    Args:
        discharge_df: DataFrame of the discharge phase.
        v_rest_end_before_discharge: Voltage at end of pre-discharge rest.
        tccc_val: Time of Constant Current Charge.
        tcvc_val: Time of Constant Voltage Charge.

    Returns:
        Dict[str, Any]: Advanced features including Resistance and RCV.
    """
    adv_features = {}

    # Internal Resistance
    if v_rest_end_before_discharge is not None and not discharge_df.empty:
        v_rest_end = v_rest_end_before_discharge
        v_discharge_start = discharge_df['Voltage(V)'].iloc[0]
        i_discharge_start = abs(discharge_df['Current(A)'].iloc[0])

        if i_discharge_start > 0.01:
            adv_features['Internal_Resistance(Ohm)'] = (
                (v_rest_end - v_discharge_start) / i_discharge_start
            )
        else:
            adv_features['Internal_Resistance(Ohm)'] = 0
    else:
        adv_features['Internal_Resistance(Ohm)'] = 0

    # RCV Ratio
    if tcvc_val > 0:
        adv_features['RCV(V)'] = tccc_val / tcvc_val
    else:
        adv_features['RCV(V)'] = 0

    # Statistical Features
    if not discharge_df.empty:
        adv_features['skew_V_discharge'] = skew(discharge_df['Voltage(V)'])
    else:
        adv_features['skew_V_discharge'] = 0

    return adv_features


def _calculate_anchor_features(
    charge_df: pd.DataFrame,
    discharge_df: pd.DataFrame,
    charge_slope_intervals: List[Tuple[float, float]],
    discharge_slope_intervals: List[Tuple[float, float]],
    tevi_intervals: List[Tuple[float, float]],
    tevd_intervals: List[Tuple[float, float]]
) -> Dict[str, Any]:
    """Calculates anchor features (slopes, TEVI, TEVD)."""
    anchor_features = {}

    def get_voltage_at_relative_time(
        df: pd.DataFrame,
        relative_time: float
    ) -> Optional[float]:
        if df.empty:
            return None
        start_time = df['Time(s)'].iloc[0]
        absolute_time = start_time + relative_time
        times = df['Time(s)'].values

        idx = np.searchsorted(times, absolute_time)

        if idx == 0:
            closest_iloc = 0
        elif idx == len(times):
            closest_iloc = len(times) - 1
        else:
            if (absolute_time - times[idx - 1]) < (times[idx] - absolute_time):
                closest_iloc = idx - 1
            else:
                closest_iloc = idx
        return df['Voltage(V)'].iloc[closest_iloc]

    # Charge Slopes
    c_dur = (charge_df['Time(s)'].iloc[-1] - charge_df['Time(s)'].iloc[0]
             if not charge_df.empty else 0)
    for i, (p_start, p_end) in enumerate(charge_slope_intervals):
        key = f'charge_slope_{i + 1}'
        if c_dur > 0:
            v_s = get_voltage_at_relative_time(charge_df, c_dur * p_start)
            v_e = get_voltage_at_relative_time(charge_df, c_dur * p_end)
            dt = c_dur * (p_end - p_start)
            if v_s is not None and v_e is not None and dt > 0:
                anchor_features[key] = (v_e - v_s) / dt
            else:
                anchor_features[key] = 0
        else:
            anchor_features[key] = 0

    # Discharge Slopes
    d_dur = (discharge_df['Time(s)'].iloc[-1] - discharge_df['Time(s)'].iloc[0]
             if not discharge_df.empty else 0)
    for i, (p_start, p_end) in enumerate(discharge_slope_intervals):
        key = f'discharge_slope_{i + 1}'
        if d_dur > 0:
            v_s = get_voltage_at_relative_time(discharge_df, d_dur * p_start)
            v_e = get_voltage_at_relative_time(discharge_df, d_dur * p_end)
            dt = d_dur * (p_end - p_start)
            if v_s is not None and v_e is not None and dt > 0:
                anchor_features[key] = (v_e - v_s) / dt
            else:
                anchor_features[key] = 0
        else:
            anchor_features[key] = 0

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
        key = f'TEVI_{i + 1}'
        if t_start is not None and t_end is not None and t_end > t_start:
            anchor_features[key] = t_end - t_start
        else:
            anchor_features[key] = 0

    for i, (v_start, v_end) in enumerate(tevd_intervals):
        t_start = get_time_for_voltage(discharge_df, v_start, 'discharge')
        t_end = get_time_for_voltage(discharge_df, v_end, 'discharge')
        key = f'TEVD_{i + 1}'
        if t_start is not None and t_end is not None and t_end > t_start:
            anchor_features[key] = t_end - t_start
        else:
            anchor_features[key] = 0

    return anchor_features


def _calculate_personalized_features(
    cycle_df: pd.DataFrame,
    cell_id: str
) -> Dict[str, Any]:
    return {}


def extract_features_for_cycle(
    cycle_data: Dict[str, Any],
    battery_data: Dict[str, Any],
    charge_slope_intervals: List[Tuple[float, float]],
    discharge_slope_intervals: List[Tuple[float, float]],
    tevi_intervals: List[Tuple[float, float]],
    tevd_intervals: List[Tuple[float, float]],
    v_rest_end_of_prev_cycle: Optional[float],
    output_dir: Optional[Path] = None
) -> Tuple[Dict[str, Any], Optional[float]]:
    """Main function to extract features for a single cycle."""

    # 1. Prepare Data
    cycle_df = pd.DataFrame({
        'Time(s)': cycle_data['time_in_s'],
        'Current(A)': cycle_data['current_in_A'],
        'Voltage(V)': cycle_data['voltage_in_V'],
        'Charge_Capacity(Ah)': cycle_data['charge_capacity_in_Ah'],
        'Discharge_Capacity(Ah)': cycle_data['discharge_capacity_in_Ah'],
        'Temperature(C)': cycle_data['temperature_in_C']
    })
    cycle_num = cycle_data['cycle_number']

    # 2. Phase Identification (Refactored to shared module)
    phases = identify_phases(cycle_df)

    # 3. Locate Key Phases
    discharge_idx = next(
        (i for i, p in enumerate(phases) if p['type'] == 'Discharge'), -1
    )
    discharge_df = phases[discharge_idx]['df'] if discharge_idx != -1 else pd.DataFrame()

    charge_dfs = [p['df'] for p in phases if p['type'] == 'Charge']
    charge_df = pd.concat(charge_dfs, ignore_index=True) if charge_dfs else pd.DataFrame()

    # Voltage memory for Resistance calc
    v_rest_end_before_discharge: Optional[float] = None
    if discharge_idx > 0 and phases[discharge_idx - 1]['type'] == 'Rest':
        rest_df_before_discharge = phases[discharge_idx - 1]['df']
        if not rest_df_before_discharge.empty:
            v_rest_end_before_discharge = rest_df_before_discharge['Voltage(V)'].iloc[-1]

    if v_rest_end_before_discharge is None:
        v_rest_end_before_discharge = v_rest_end_of_prev_cycle

    # Save voltage for next cycle
    v_rest_end_for_next_cycle: Optional[float] = None
    if phases and phases[-1]['type'] == 'Rest':
        rest_df_at_end_of_cycle = phases[-1]['df']
        if not rest_df_at_end_of_cycle.empty:
            v_rest_end_for_next_cycle = rest_df_at_end_of_cycle['Voltage(V)'].iloc[-1]

    # Post-discharge rest
    rest_df_after_discharge = pd.DataFrame()
    if (discharge_idx != -1 and (discharge_idx + 1) < len(phases) and
            phases[discharge_idx + 1]['type'] == 'Rest'):
        rest_df_after_discharge = phases[discharge_idx + 1]['df']

    # Cutoff Logic
    if not discharge_df.empty:
        cutoff_voltage = battery_data['min_voltage_limit_in_V']
        cutoff_indices = discharge_df.index[
            discharge_df['Voltage(V)'] <= cutoff_voltage
        ]
        if not cutoff_indices.empty:
            first_cutoff_index = cutoff_indices[0]
            discharge_df = discharge_df.loc[:first_cutoff_index]

    # 4. Feature Extraction
    direct_features = _calculate_direct_features(
        charge_df, discharge_df, rest_df_after_discharge,
        cycle_num, battery_data
    )

    # REFACTORED: Use shared module for IC/DV
    # Config for CALB (NCM)
    ncm_config = {
        'peak_mode': 2,
        'nominal_capacity': battery_data.get('nominal_capacity_in_Ah', 2.0),
        'window_length_ic': 41,
        'window_length_dv': 25,
        'peak_height_ic': 0.5,
        # 'peak_height_dv': 0.01,
        'voltage_range_ic': (3.2, 3.6),
        # 'voltage_range_dv': (3.4, 3.8),
        'prominence_ic': 0.1,
        # 'prominence_dv': 0.005,
        'ic_step_size': 0.002,
        'dv_step_size': battery_data.get('nominal_capacity_in_Ah', 2.0) * 0.005,
        'search_window_dvv': 0.1,
        'search_window_dvp': 0.04,
        'initial_capacity_cut_fraction': 0.02,
        'icv_search_offset_lower': 0.05,
        'icv_search_offset_upper': 0.5
    }

    plot_params = None
    if output_dir:
        plot_params = {
            'cell_id': battery_data.get('cell_id', 'unknown'),
            'cycle_num': cycle_num,
            'output_dir': output_dir
        }

    derivative_features = extract_ic_dv_features(
        discharge_df,
        config=ncm_config,
        plot_params=plot_params
    )

    advanced_features = _calculate_advanced_features(
        discharge_df,
        v_rest_end_before_discharge,
        direct_features.get('TCCC', 0),
        direct_features.get('TCVC', 0)
    )

    anchor_features = _calculate_anchor_features(
        charge_df, discharge_df,
        charge_slope_intervals, discharge_slope_intervals,
        tevi_intervals, tevd_intervals
    )

    personalized_features = _calculate_personalized_features(
        cycle_df, battery_data['cell_id']
    )

    all_features = {
        **direct_features,
        **derivative_features,
        **advanced_features,
        **anchor_features,
        **personalized_features
    }

    return all_features, v_rest_end_for_next_cycle


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

    all_cycle_features = []

    # Safety check for cycle_data
    if 'cycle_data' not in battery_data or battery_data['cycle_data'] is None:
        print(f"No cycle data for {battery_data.get('cell_id', 'Unknown')}")
        return

    cycles_to_process = battery_data['cycle_data']
    if num_cycles is not None and num_cycles > 0:
        cycles_to_process = battery_data['cycle_data'][:num_cycles]

    v_rest_end_of_prev_cycle: Optional[float] = None
    cell_id = battery_data.get('cell_id', file_path.stem)

    pbar = tqdm(cycles_to_process, desc=f"Processing {cell_id}")
    for cycle_data in pbar:
        if not cycle_data.get('time_in_s'):
            continue

        try:
            features, v_rest_end_for_next_cycle = extract_features_for_cycle(
                cycle_data, battery_data,
                charge_slope_intervals, discharge_slope_intervals,
                tevi_intervals, tevd_intervals,
                v_rest_end_of_prev_cycle,
                output_dir=output_dir
            )
            all_cycle_features.append(features)

            if v_rest_end_for_next_cycle is not None:
                v_rest_end_of_prev_cycle = v_rest_end_for_next_cycle

        except Exception:
            # Keep previous rest voltage to avoid cascading failures
            v_rest_end_of_prev_cycle = None

    if not all_cycle_features:
        print(f"Warning: No features extracted for {cell_id}")
        return

    features_df = pd.DataFrame(all_cycle_features)

    # Extract temperature from filename (Format: CALB_Temp_ID)
    try:
        ambient_temp = float(file_path.stem.split('_')[1])
    except (IndexError, ValueError):
        print(f"Warning: Could not parse temperature from filename {file_path.stem}")
        ambient_temp = None

    features_df['Ambient_Temperature'] = ambient_temp

    # Final Column Ordering
    base_cols = [
        'Cycle_Number', 'Workload_Type', 'Ambient_Temperature',
        'Discharge_Capacity(Ah)', 'Charge_Capacity(Ah)',
        'Discharge_Energy(Wh)', 'Charge_Energy(Wh)',
        'Coulombic_Efficiency', 'Energy_Efficiency',
        'Rest_Time(s)', 'avg_charge_c_rate', 'discharge_c_rate',
        'max_I_charge(A)', 'var_I_charge',
        'ICHV(V)', 'UVP_time(s)', 'TCCC(s)', 'TCVC(s)', 'CV_Current_Tau',
        'UVP(V)', # 'MAT_charge(C)', 'MET_charge(s)',
        'IDV(V)', 'LVP_time(s)', 'var_I_discharge', 'var_V_discharge',
        'median_V_discharge(V)', 'total_discharge_time(s)', 'LVP(V)',
        # 'MAT_discharge(C)', 'MET_discharge(s)',
        'ICP', 'ICPL_V', 'ICP_FWHM', 'ICP_Area',
        'ICV', 'ICVL_V',
        'DVP', 'DVPL_V', 'DVP_FWHM', 'DVP_Area',
        'DVV', 'DVVL_V',
        'Internal_Resistance(Ohm)', 'RCV(V)', 'skew_V_discharge', # 'Temperature_Rise(C)',
        # 'skew_T_discharge'
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
    # Keep input paths but ensure output logic matches current refactor state

    # NOTE: Use relative path assuming data is at project root 'data/calb_pkl'
    # or passed via args. For now, we update to project_root / 'data' / 'calb_pkl'
    processed_data_dir = project_root / 'data' / 'calb_pkl'

    output_dir = project_root / 'results' / 'features' / 'CALB'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Standard feature extraction intervals
    charge_slope_intervals = [(0.10, 0.30), (0.10, 0.50), (0.10, 0.80)]
    discharge_slope_intervals = [(0.10, 0.30), (0.10, 0.50), (0.10, 0.80)]
    tevi_intervals = [(3.5, 3.7), (3.7, 3.9), (3.9, 4.1)]
    tevd_intervals = [(4.1, 3.9), (3.9, 3.7), (3.7, 3.5)]

    num_cycles_to_extract = 100

    if not processed_data_dir.exists():
        print(f"Error: Directory not found at '{processed_data_dir}'.")
        print("Please ensure data is placed in 'data/calb_pkl' relative to project root.")
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

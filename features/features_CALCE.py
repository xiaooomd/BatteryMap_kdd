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

from src.utils.math_tools import fit_cv_decay, get_interp_val
from src.utils.feature_tools import extract_ic_dv_features

# Suppress FutureWarnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# CALCE Dataset Battery Conditions
BATTERY_CONDITIONS = {
    # 0.5C discharge rate
    'CS2_33': {'discharge_rate': 0.5, 'nominal_capacity': 1.1},
    'CS2_34': {'discharge_rate': 0.5, 'nominal_capacity': 1.1},
    'CX2_16': {'discharge_rate': 0.5, 'nominal_capacity': 1.35},
    'CX2_33': {'discharge_rate': 0.5, 'nominal_capacity': 1.35},
    'CX2_34': {'discharge_rate': 0.5, 'nominal_capacity': 1.35},
    'CX2_35': {'discharge_rate': 0.5, 'nominal_capacity': 1.35},
    'CX2_36': {'discharge_rate': 0.5, 'nominal_capacity': 1.35},
    'CX2_37': {'discharge_rate': 0.5, 'nominal_capacity': 1.35},
    'CX2_38': {'discharge_rate': 0.5, 'nominal_capacity': 1.35},
    # 1C discharge rate
    'CS2_35': {'discharge_rate': 1.0, 'nominal_capacity': 1.1},
    'CS2_36': {'discharge_rate': 1.0, 'nominal_capacity': 1.1},
    'CS2_37': {'discharge_rate': 1.0, 'nominal_capacity': 1.1},
    'CS2_38': {'discharge_rate': 1.0, 'nominal_capacity': 1.1},
}


def _find_main_phases(
    cycle_df: pd.DataFrame,
    charge_current_threshold: float = 0.01,
    discharge_current_threshold: float = -0.01
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Identifies the main 'Charge', 'Discharge', and 'Rest' phases.

    Modified to be robust against fragmentation. Instead of selecting only the
    single longest block, it selects the range from the start of the first
    significant block to the end of the last significant block, effectively
    bridging small gaps caused by cycler mode switching.

    Args:
        cycle_df: The dataframe for the specific cycle.
        charge_current_threshold: Current > this is Charge.
        discharge_current_threshold: Current < this is Discharge.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            (Charge DF, Discharge DF, Rest DF). Returns empty DFs if not found.
    """
    empty_df = pd.DataFrame(columns=cycle_df.columns)

    if cycle_df.empty:
        return empty_df, empty_df, empty_df

    # 1. Assign states: 1 (Charge), -1 (Discharge), 0 (Rest)
    # Using strict inequality to avoid floating point noise around 0
    cond_charge = cycle_df['Current(A)'] > charge_current_threshold
    cond_discharge = cycle_df['Current(A)'] < discharge_current_threshold

    # Create a lightweight view for state analysis
    states = np.zeros(len(cycle_df), dtype=int)
    states[cond_charge] = 1
    states[cond_discharge] = -1

    # 2. Helper to extract phases based on first and last significant occurrence
    def extract_phase(
        target_state: int,
        min_len: int = 50
    ) -> pd.DataFrame:
        """Extracts the full range of a phase, bridging gaps.

        Modified: Instead of just picking the largest cluster, we look for clusters
        and combine them if they are not separated by a significant opposite phase.
        This ensures CC and CV phases are both captured if they are split.
        """
        indices = np.where(states == target_state)[0]

        if len(indices) < min_len:
            return empty_df

        # Find continuous blocks of the target state
        # A gap of more than 500 points is likely a real state change (Rest/Discharge)
        # while small gaps are just instrumentation/sampling artifacts.
        change_points = np.where(np.diff(indices) > 500)[0]

        if len(change_points) == 0:
            start_idx = indices[0]
            end_idx = indices[-1]
        else:
            # Split into blocks and filter by size
            split_indices = np.split(indices, change_points + 1)
            valid_blocks = [b for b in split_indices if len(b) > 20] # Filter out tiny noise

            if not valid_blocks:
                return empty_df

            # Use the absolute first and last points of significant blocks
            # This bridges small Rests or switching gaps between CC and CV
            start_idx = valid_blocks[0][0]
            end_idx = valid_blocks[-1][-1]

        return cycle_df.iloc[start_idx : end_idx + 1].copy()

    clean_charge_df = extract_phase(1)
    clean_discharge_df = extract_phase(-1)

    # 3. Handle Rest
    # Rest is typically between Charge end and Discharge start
    clean_rest_df = empty_df
    if not clean_charge_df.empty and not clean_discharge_df.empty:
        chg_end = clean_charge_df.index[-1]
        dch_start = clean_discharge_df.index[0]

        if dch_start > chg_end:
            # Extract everything in between
            clean_rest_df = cycle_df.loc[chg_end + 1 : dch_start - 1].copy()

    return clean_charge_df, clean_discharge_df, clean_rest_df

def _calculate_direct_features(
    charge_df: pd.DataFrame,
    discharge_df: pd.DataFrame,
    cycle_num: int,
    cell_id: str,
    uvp: float,
    lvp: float
) -> Dict[str, Any]:
    """Calculates direct features (Capacity, Energy, CV Dynamics).

    Fixes:
        - Normalizes time to be relative to the start of the step/cycle.
        - Ensures UVP/LVP times represent duration/relative time, not absolute timestamp.
    """
    features = {}
    # Extract model name
    cell_model = cell_id.replace('CALCE_', '')
    conditions = BATTERY_CONDITIONS.get(cell_model, {})

    # --- A. Overall Cycle Features ---
    features['Cycle_Number'] = cycle_num

    def calc_capacity_integration(df: pd.DataFrame) -> float:
        if df.empty or len(df) < 2:
            return 0.0
        q_as = np.trapz(y=df['Current(A)'].abs(), x=df['Time(s)'])
        return q_as / 3600.0 # Convert Ampere-seconds to Ampere-hours

    dis_cap_integrated = calc_capacity_integration(discharge_df)
    chg_cap_integrated = calc_capacity_integration(charge_df)

    # Override the capacity features with integrated values for consistency
    features['Discharge_Capacity(Ah)'] = dis_cap_integrated
    features['Charge_Capacity(Ah)'] = chg_cap_integrated

    if chg_cap_integrated > 0.01:
        features['Coulombic_Efficiency'] = dis_cap_integrated / chg_cap_integrated
    else:
        features['Coulombic_Efficiency'] = 0

    # 3. Energy & Efficiency
    # Helper to calculate energy
    def calc_energy(df: pd.DataFrame) -> float:
        if df.empty:
            return 0.0
        p = df['Voltage(V)'] * df['Current(A)'].abs()
        e_ws = np.trapz(y=p, x=df['Time(s)'])
        return e_ws / 3600.0

    features['Charge_Energy(Wh)'] = calc_energy(charge_df)
    features['Discharge_Energy(Wh)'] = calc_energy(discharge_df)

    if features['Charge_Energy(Wh)'] > 0:
        features['Energy_Efficiency'] = (
            features['Discharge_Energy(Wh)'] / features['Charge_Energy(Wh)']
        )
    else:
        features['Energy_Efficiency'] = 0

    features['charge_c_rate'] = 0.5
    features['discharge_c_rate'] = conditions.get('discharge_rate', np.nan)
    features['UVP(V)'] = uvp
    features['LVP(V)'] = lvp

    # Charge Current Features
    if not charge_df.empty:
        # Use valid charge data for stats to avoid zero-padding effects
        valid_chg = charge_df[charge_df['Current(A)'].abs() > 1e-4]
        if valid_chg.empty:
            valid_chg = charge_df

        # Max Charge Current (Lithium Plating Risk)
        features['max_I_charge(A)'] = valid_chg['Current(A)'].abs().max()

        # Current Variance
        features['var_I_charge'] = valid_chg['Current(A)'].var()
    else:
        features['max_I_charge(A)'] = 0.0
        features['var_I_charge'] = 0.0

    # --- B. Charging Phase Features & CV Dynamics ---
    features['CV_Current_Tau'] = 0.0

    if not charge_df.empty:
        # Time Normalization: Relative to Charge Start
        t_start_charge = charge_df['Time(s)'].iloc[0]

        features['ICHV(V)'] = charge_df['Voltage(V)'].iloc[0]
        # UVP_time: Duration from start of charge until end of charge (hitting UVP/Cutoff)
        features['UVP_time(s)'] = charge_df['Time(s)'].iloc[-1] - t_start_charge

        # CV Phase Logic
        # Using a slightly looser threshold to detect CV entry robustly
        voltage_tolerance = 0.01
        # Find index where voltage first crosses (UVP - tolerance)
        cv_candidates = charge_df[charge_df['Voltage(V)'] >= (uvp - voltage_tolerance)]

        if not cv_candidates.empty:
            cv_start_idx = cv_candidates.index[0]
            time_at_cv_start = charge_df.loc[cv_start_idx, 'Time(s)']

            features['TCCC(s)'] = time_at_cv_start - t_start_charge
            features['TCVC(s)'] = charge_df['Time(s)'].iloc[-1] - time_at_cv_start

            # Robust CV Fit
            # Only fit if we have enough data points
            cv_df = charge_df.loc[cv_start_idx:].copy()
            if len(cv_df) > 10:
                cv_current = cv_df['Current(A)'].values
                cv_time = cv_df['Time(s)'].values
                # Use normalized time for fitting to avoid overflow
                features['CV_Current_Tau'] = fit_cv_decay(cv_time, cv_current)
        else:
            # Did not reach CV voltage
            features['TCCC(s)'] = features['UVP_time(s)']
            features['TCVC(s)'] = 0
    else:
        features.update({'ICHV(V)': 0, 'UVP_time(s)': 0, 'TCCC(s)': 0, 'TCVC(s)': 0})

    # --- C. Discharging Phase Features ---
    if not discharge_df.empty:
        t_start_discharge = discharge_df['Time(s)'].iloc[0]

        features['IDV(V)'] = discharge_df['Voltage(V)'].iloc[0]
        # LVP_time: Duration from start of discharge until LVP/Cutoff
        features['LVP_time(s)'] = discharge_df['Time(s)'].iloc[-1] - t_start_discharge

        features['var_I_discharge'] = discharge_df['Current(A)'].var()
        features['var_V_discharge'] = discharge_df['Voltage(V)'].var()
        features['median_V_discharge(V)'] = discharge_df['Voltage(V)'].median()
        features['total_discharge_time(s)'] = features['LVP_time(s)'] # Same logic
    else:
        features.update({
            'IDV(V)': 0, 'LVP_time(s)': 0, 'var_I_discharge': 0,
            'var_V_discharge': 0, 'median_V_discharge(V)': 0,
            'total_discharge_time(s)': 0
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

    # --- A. Internal Resistance ---
    # Using the voltage drop at the start of discharge relative to rest
    if not rest_df.empty and not discharge_df.empty and len(discharge_df) > 5:
        v_rest_end = rest_df['Voltage(V)'].iloc[-1]
        # Average first few points to reduce noise sensitivity
        v_discharge_stable = discharge_df['Voltage(V)'].iloc[0:5].mean()
        i_discharge_stable = abs(discharge_df['Current(A)'].iloc[0:5].mean())

        if i_discharge_stable > 1e-3:
            resistance = (v_rest_end - v_discharge_stable) / i_discharge_stable
            adv_features['Internal_Resistance(Ohm)'] = max(0, resistance)
        else:
            adv_features['Internal_Resistance(Ohm)'] = 0
    else:
        adv_features['Internal_Resistance(Ohm)'] = 0

    # --- B. Other Calculated Features ---
    min_time_tolerance = 1.0
    tcvc = features.get('TCVC', 0)
    tccc = features.get('TCCC', 0)

    if tcvc > min_time_tolerance:
        adv_features['RCV(V)'] = tccc / tcvc
    else:
        adv_features['RCV(V)'] = 0

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
    """Calculates anchor interval features (Slope, TEVI, TEVD)."""
    anchor_features = {}

    def get_voltage_at_relative_time(
        df: pd.DataFrame,
        relative_time: float
    ) -> float:
        if df.empty:
            return 0.0

        start_time = df['Time(s)'].iloc[0]
        absolute_time = start_time + relative_time
        time_array = df['Time(s)'].values

        idx = np.searchsorted(time_array, absolute_time, side='left')

        if idx == 0:
            return df['Voltage(V)'].iloc[0]
        if idx == len(time_array):
            return df['Voltage(V)'].iloc[-1]

        t_left = time_array[idx - 1]
        t_right = time_array[idx]

        if (absolute_time - t_left) < (t_right - absolute_time):
            return df['Voltage(V)'].iloc[idx - 1]
        else:
            return df['Voltage(V)'].iloc[idx]

    # --- A. Charge Slopes ---
    charge_duration = (
        charge_df['Time(s)'].iloc[-1] - charge_df['Time(s)'].iloc[0]
        if not charge_df.empty else 0
    )
    for i, (p_start, p_end) in enumerate(charge_slope_intervals):
        if not charge_df.empty and charge_duration > 1e-6:
            t_start_rel = charge_duration * p_start
            t_end_rel = charge_duration * p_end
            v_start = get_voltage_at_relative_time(charge_df, t_start_rel)
            v_end = get_voltage_at_relative_time(charge_df, t_end_rel)
            time_delta = t_end_rel - t_start_rel
            if v_start is not None and v_end is not None and time_delta > 1e-6:
                anchor_features[f'charge_slope_{i + 1}'] = (v_end - v_start) / time_delta
            else:
                anchor_features[f'charge_slope_{i + 1}'] = 0
        else:
            anchor_features[f'charge_slope_{i + 1}'] = 0

    # --- B. Discharge Slopes ---
    discharge_duration = (
        discharge_df['Time(s)'].iloc[-1] - discharge_df['Time(s)'].iloc[0]
        if not discharge_df.empty else 0
    )
    for i, (p_start, p_end) in enumerate(discharge_slope_intervals):
        if not discharge_df.empty and discharge_duration > 1e-6:
            t_start_rel = discharge_duration * p_start
            t_end_rel = discharge_duration * p_end
            v_start = get_voltage_at_relative_time(discharge_df, t_start_rel)
            v_end = get_voltage_at_relative_time(discharge_df, t_end_rel)
            time_delta = t_end_rel - t_start_rel
            if v_start is not None and v_end is not None and time_delta > 1e-6:
                anchor_features[f'discharge_slope_{i + 1}'] = (v_end - v_start) / time_delta
            else:
                anchor_features[f'discharge_slope_{i + 1}'] = 0
        else:
            anchor_features[f'discharge_slope_{i + 1}'] = 0

    # --- C. TEVI / TEVD ---
    def get_time_for_voltage(df, voltage, direction):
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


def _calculate_personalized_features(cycle_df: pd.DataFrame, cell_id: str) -> Dict[str, Any]:
    return {}


def extract_features_for_cycle(
    cycle_data: Dict[str, Any],
    cell_id: str,
    charge_slope_intervals: List[Tuple[float, float]],
    discharge_slope_intervals: List[Tuple[float, float]],
    tevi_intervals: List[Tuple[float, float]],
    tevd_intervals: List[Tuple[float, float]],
    uvp: float,
    lvp: float,
    output_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """Main function to extract all features for a single cycle."""
    cycle_df = pd.DataFrame({
        'Time(s)': cycle_data['time_in_s'],
        'Current(A)': cycle_data['current_in_A'],
        'Voltage(V)': cycle_data['voltage_in_V'],
        'Charge_Capacity(Ah)': cycle_data['charge_capacity_in_Ah'],
        'Discharge_Capacity(Ah)': cycle_data['discharge_capacity_in_Ah']
    })
    cycle_num = cycle_data['cycle_number']

    charge_df, discharge_df, rest_df = _find_main_phases(
        cycle_df,
        charge_current_threshold=0.01,
        discharge_current_threshold=-0.01
    )

    direct_features = _calculate_direct_features(
        charge_df, discharge_df, cycle_num, cell_id, uvp, lvp
    )

    if not discharge_df.empty:
        cutoff_voltage = direct_features.get('LVP', 2.7)
        cutoff_indices = discharge_df.index[
            discharge_df['Voltage(V)'] <= cutoff_voltage
        ]
        if not cutoff_indices.empty:
            first_cutoff_index = cutoff_indices[0]
            discharge_df = discharge_df.loc[:first_cutoff_index]

    cell_model = cell_id.replace('CALCE_', '')
    conditions = BATTERY_CONDITIONS.get(cell_model, {})
    nominal_cap = conditions.get('nominal_capacity', 1.1)

    ncm_config = {
        'peak_mode': 1,
        'nominal_capacity': nominal_cap,
        'window_length_ic': 25,
        'window_length_dv': 31,
        'peak_height_ic': 0.05,
        # 'peak_height_dv': 0.01,
        'voltage_range_ic': (3.4, 4.2),
        # 'voltage_range_dv': (3.4, 4.15),
        'prominence_ic': 0.01,
        # 'prominence_dv': 0.005,
        'ic_step_size': 0.001,
        'dv_step_size': nominal_cap * 0.005,
        'search_window_dvv': 0.05,
        'search_window_dvp': 0.05,
        'initial_capacity_cut_fraction': 0.01,
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
        charge_df, discharge_df, rest_df, direct_features
    )
    anchor_features = _calculate_anchor_features(
        charge_df, discharge_df,
        charge_slope_intervals, discharge_slope_intervals,
        tevi_intervals, tevd_intervals
    )
    personalized_features = _calculate_personalized_features(
        cycle_df, cell_id
    )

    all_features = {
        **direct_features,
        **derivative_features,
        **advanced_features,
        **anchor_features,
        **personalized_features
    }

    return all_features


def process_battery(
    file_path: Path,
    output_dir: Path,
    charge_slope_intervals: List[Tuple[float, float]],
    discharge_slope_intervals: List[Tuple[float, float]],
    tevi_intervals: List[Tuple[float, float]],
    tevd_intervals: List[Tuple[float, float]],
    num_cycles: Optional[int] = None
):
    """Processes a single battery .pkl file."""
    try:
        with open(file_path, 'rb') as f:
            data_dict = pickle.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return

    battery_data = data_dict
    # Ensure cycle_data is iterable
    if 'cycle_data' in battery_data and battery_data['cycle_data'] is not None:
        if not isinstance(battery_data['cycle_data'], list):
            # In case it's a numpy array or other iterable
            battery_data['cycle_data'] = list(battery_data['cycle_data'])
    else:
        print(f"No cycle data found in {file_path}")
        return

    # Defaults for CALCE if limits are missing
    uvp = battery_data.get('max_voltage_limit_in_V', 4.2)
    lvp = battery_data.get('min_voltage_limit_in_V', 2.7)

    all_cycle_features = []

    cycles_to_process = battery_data['cycle_data']
    if num_cycles is not None and num_cycles > 0:
        cycles_to_process = battery_data['cycle_data'][:num_cycles]

    cell_id = battery_data.get('cell_id', file_path.stem)

    for cycle_data in tqdm(cycles_to_process, desc=f"Processing {cell_id}"):
        if not cycle_data.get('time_in_s'):
            continue

        try:
            features = extract_features_for_cycle(
                cycle_data, cell_id,
                charge_slope_intervals, discharge_slope_intervals,
                tevi_intervals, tevd_intervals,
                uvp, lvp,
                output_dir=output_dir
            )
            all_cycle_features.append(features)
        except Exception:
            continue

    if not all_cycle_features:
        print(f"Warning: No features extracted for {cell_id}")
        return

    features_df = pd.DataFrame(all_cycle_features)

    # Reorder columns for clarity
    ordered_cols = [col for col in [
        # Overall
        'Cycle_Number', 'Discharge_Capacity', 'Charge_Capacity',
        'Coulombic_Efficiency',
        'Discharge_Energy', 'Charge_Energy', 'Energy_Efficiency',
        'charge_c_rate', 'discharge_c_rate',
        'max_I_charge', 'var_I_charge',
        # Charge
        'ICHV', 'UVP_time', 'TCCC', 'TCVC', 'CV_Current_Tau',
        'UVP',
        # Discharge
        'IDV', 'LVP_time', 'var_I_discharge', 'var_V_discharge',
        'median_V_discharge', 'total_discharge_time', 'LVP',
        # Curves
        'ICP', 'ICPL_V', 'ICP_FWHM', 'ICP_Area',
        # 'ICV', 'ICVL_V',
        'DVP', 'DVPL_V', 'DVV', 'DVVL_V',
        # Advanced
        'Internal_Resistance', 'RCV', 'skew_V_discharge',
        # Anchor
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
    processed_data_dir = Path('F:/datasets/battery/CALCE')
    output_dir = project_root / 'results' / 'CALCE'
    output_dir.mkdir(parents=True, exist_ok=True)

    charge_slope_intervals = [(0.10, 0.30), (0.10, 0.50), (0.10, 0.80)]
    discharge_slope_intervals = [(0.10, 0.30), (0.10, 0.50), (0.10, 0.80)]
    # Typical Voltage range for LCO/NMC: ~2.7V to 4.2V
    tevi_intervals = [(3.5, 3.7), (3.7, 3.9), (3.9, 4.1)]
    tevd_intervals = [(4.1, 3.9), (3.9, 3.7), (3.7, 3.5)]

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

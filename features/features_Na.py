import pickle
import sys
import warnings
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
from src.utils.feature_tools import extract_ic_dv_features

# Suppress FutureWarnings
warnings.filterwarnings('ignore', category=FutureWarning)


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
    """Calculates direct features from raw data."""
    features: Dict[str, Any] = {}

    t0 = cycle_df['Time(s)'].iloc[0] if not cycle_df.empty else 0.0

    # --- A. Basic Capacity & Cycle Info ---
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

    # --- B. Energy & Efficiency ---
    if not charge_df.empty:
        p_charge = charge_df['Voltage(V)'] * charge_df['Current(A)'].abs()
        e_charge_ws = trapezoid(y=p_charge, x=charge_df['Time(s)'])
        features['Charge_Energy(Wh)'] = e_charge_ws / 3600.0
    else:
        features['Charge_Energy(Wh)'] = 0.0

    if not discharge_df.empty:
        p_discharge = discharge_df['Voltage(V)'] * discharge_df['Current(A)'].abs()
        e_discharge_ws = trapezoid(y=p_discharge, x=discharge_df['Time(s)'])
        features['Discharge_Energy(Wh)'] = e_discharge_ws / 3600.0
    else:
        features['Discharge_Energy(Wh)'] = 0.0

    if features['Charge_Energy(Wh)'] > 1e-6:
        features['Energy_Efficiency'] = features['Discharge_Energy(Wh)'] / features['Charge_Energy(Wh)']
    else:
        features['Energy_Efficiency'] = 0.0

    # Rest Time
    if not charge_df.empty and not discharge_df.empty:
        time_charge_end = charge_df['Time(s)'].iloc[-1]
        time_discharge_start = discharge_df['Time(s)'].iloc[0]
        features['Rest_Time(s)'] = time_discharge_start - time_charge_end
    else:
        features['Rest_Time(s)'] = 0.0

    # Protocol C-rates
    try:
        features['charge_c_rate'] = battery_data.charge_protocol[0].rate_in_C
        features['discharge_c_rate'] = battery_data.discharge_protocol[0].rate_in_C
    except (AttributeError, IndexError):
        features['charge_c_rate'] = 0.0
        features['discharge_c_rate'] = 0.0

    # --- C. Charging Phase Features (NO CV PHASE for NA-ion protocol) ---
    if not charge_df.empty:
        features['ICHV(V)'] = charge_df['Voltage(V)'].iloc[0]
        features['UVP_time(s)'] = charge_df['Time(s)'].iloc[-1] - t0
        features['UVP(V)'] = battery_data.max_voltage_limit_in_V
        # TCCC is total charge duration here
        features['TCCC(s)'] = charge_df['Time(s)'].iloc[-1] - charge_df['Time(s)'].iloc[0]
    else:
        features.update({
            'ICHV(V)': 0, 'UVP_time(s)': 0,
            'UVP(V)': battery_data.max_voltage_limit_in_V, 'TCCC(s)': 0
        })

    # --- D. Discharging Phase Features ---
    if not discharge_df.empty:
        features['IDV(V)'] = discharge_df['Voltage(V)'].iloc[0]
        features['LVP_time(s)'] = discharge_df['Time(s)'].iloc[-1] - t0
        features['LVP(V)'] = battery_data.min_voltage_limit_in_V
        features['var_I_discharge'] = discharge_df['Current(A)'].var()
        features['var_V_discharge'] = discharge_df['Voltage(V)'].var()
        features['median_V_discharge(V)'] = discharge_df['Voltage(V)'].median()
        features['total_discharge_time(s)'] = (
            discharge_df['Time(s)'].iloc[-1] - discharge_df['Time(s)'].iloc[0]
        )
    else:
        features.update({
            'IDV(V)': 0, 'LVP_time(s)': 0,
            'LVP(V)': battery_data.min_voltage_limit_in_V,
            'var_I_discharge': 0, 'var_V_discharge': 0,
            'median_V_discharge(V)': 0, 'total_discharge_time(s)': 0
        })

    return features


def _calculate_advanced_features(
    cycle_df: pd.DataFrame,
    charge_df: pd.DataFrame,
    discharge_df: pd.DataFrame,
    rest_df: pd.DataFrame,
    features: Dict[str, Any]
) -> Dict[str, Any]:
    """Calculates advanced features (Internal Resistance, Skewness)."""
    adv_features = {}

    # --- Internal Resistance ---
    if not discharge_df.empty:
        time_discharge_start = discharge_df['Time(s)'].iloc[0]
        v_on_load = discharge_df['Voltage(V)'].iloc[0]
        i_on_load = abs(discharge_df['Current(A)'].iloc[0])

        pre_discharge_rest = rest_df[rest_df['Time(s)'] < time_discharge_start]

        if not pre_discharge_rest.empty:
            v_ocv = pre_discharge_rest['Voltage(V)'].iloc[-1]
        else:
            all_points_before = cycle_df[cycle_df['Time(s)'] < time_discharge_start]
            if not all_points_before.empty:
                v_ocv = all_points_before['Voltage(V)'].iloc[-1]
            else:
                v_ocv = v_on_load

        if i_on_load > 0.001 and v_ocv > v_on_load:
            adv_features['Internal_Resistance(Ohm)'] = (v_ocv - v_on_load) / i_on_load
        else:
            adv_features['Internal_Resistance(Ohm)'] = 0.0
    else:
        adv_features['Internal_Resistance(Ohm)'] = 0.0

    # --- Skewness ---
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
    """Calculates anchor features (Slopes, TEVI, TEVD)."""
    anchor_features = {}

    def get_voltage_at_relative_time(
        df: pd.DataFrame,
        relative_time: float
    ) -> Optional[float]:
        if df.empty: return None
        start_time = df['Time(s)'].iloc[0]
        absolute_time = start_time + relative_time
        time_series = df['Time(s)'].values
        idx = np.searchsorted(time_series, absolute_time)

        if idx == 0: return float(df['Voltage(V)'].iloc[0])
        if idx == len(time_series): return float(df['Voltage(V)'].iloc[-1])

        time_before = time_series[idx - 1]
        time_after = time_series[idx]
        if (absolute_time - time_before) < (time_after - absolute_time):
            return float(df['Voltage(V)'].iloc[idx - 1])
        else:
            return float(df['Voltage(V)'].iloc[idx])

    # --- Charge Slopes ---
    charge_duration = (
        charge_df['Time(s)'].iloc[-1] - charge_df['Time(s)'].iloc[0]
        if not charge_df.empty else 0
    )
    for i, (p_start, p_end) in enumerate(charge_slope_intervals):
        key = f'charge_slope_{i+1}'
        if not charge_df.empty and charge_duration > 0 and p_end > p_start:
            t_start_rel = p_start * charge_duration
            t_end_rel = p_end * charge_duration
            v_start = get_voltage_at_relative_time(charge_df, t_start_rel)
            v_end = get_voltage_at_relative_time(charge_df, t_end_rel)

            if v_start is not None and v_end is not None:
                 anchor_features[key] = (v_end - v_start) / (t_end_rel - t_start_rel)
            else:
                anchor_features[key] = 0.0
        else:
            anchor_features[key] = 0.0

    # --- Discharge Slopes ---
    discharge_duration = (
        discharge_df['Time(s)'].iloc[-1] - discharge_df['Time(s)'].iloc[0]
        if not discharge_df.empty else 0
    )
    for i, (p_start, p_end) in enumerate(discharge_slope_intervals):
        key = f'discharge_slope_{i+1}'
        if not discharge_df.empty and discharge_duration > 0 and p_end > p_start:
            t_start_rel = p_start * discharge_duration
            t_end_rel = p_end * discharge_duration
            v_start = get_voltage_at_relative_time(discharge_df, t_start_rel)
            v_end = get_voltage_at_relative_time(discharge_df, t_end_rel)

            if v_start is not None and v_end is not None:
                anchor_features[key] = (v_end - v_start) / (t_end_rel - t_start_rel)
            else:
                anchor_features[key] = 0.0
        else:
            anchor_features[key] = 0.0

    # --- TEVI / TEVD ---
    def get_time_for_voltage(
        df: pd.DataFrame,
        voltage: float,
        direction: str
    ) -> Optional[float]:
        if df.empty: return None
        if direction == 'charge':
            target_rows = df[df['Voltage(V)'] >= voltage]
        else:
            target_rows = df[df['Voltage(V)'] <= voltage]
        return float(target_rows['Time(s)'].iloc[0]) if not target_rows.empty else None

    for i, (v_start, v_end) in enumerate(tevi_intervals):
        t_start = get_time_for_voltage(charge_df, v_start, 'charge')
        t_end = get_time_for_voltage(charge_df, v_end, 'charge')
        if t_start is not None and t_end is not None and t_end > t_start:
            anchor_features[f'TEVI_{i+1}'] = t_end - t_start
        else:
            anchor_features[f'TEVI_{i+1}'] = 0.0

    for i, (v_start, v_end) in enumerate(tevd_intervals):
        t_start = get_time_for_voltage(discharge_df, v_start, 'discharge')
        t_end = get_time_for_voltage(discharge_df, v_end, 'discharge')
        if t_start is not None and t_end is not None and t_end > t_start:
            anchor_features[f'TEVD_{i+1}'] = t_end - t_start
        else:
            anchor_features[f'TEVD_{i+1}'] = 0.0

    return anchor_features


def extract_features_for_cycle(
    cycle_data: AttrDict,
    battery_data: AttrDict,
    charge_slope_intervals: List[Tuple[float, float]],
    discharge_slope_intervals: List[Tuple[float, float]],
    tevi_intervals: List[Tuple[float, float]],
    tevd_intervals: List[Tuple[float, float]],
    output_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """Master function to extract all features for a single cycle."""
    cycle_df = pd.DataFrame({
        'Time(s)': cycle_data.time_in_s,
        'Current(A)': cycle_data.current_in_A,
        'Voltage(V)': cycle_data.voltage_in_V,
        'Charge_Capacity(Ah)': cycle_data.charge_capacity_in_Ah,
        'Discharge_Capacity(Ah)': cycle_data.discharge_capacity_in_Ah,
    })
    cycle_num = cycle_data.cycle_number

    charge_df = cycle_df[cycle_df['Current(A)'] > 0].copy()
    discharge_df = cycle_df[cycle_df['Current(A)'] < 0].copy()
    rest_df = cycle_df[cycle_df['Current(A)'] == 0].copy()

    if not discharge_df.empty:
        cutoff_voltage = battery_data.min_voltage_limit_in_V
        cutoff_indices = discharge_df.index[
            discharge_df['Voltage(V)'] <= cutoff_voltage
        ]
        if not cutoff_indices.empty:
            first_cutoff_index = cutoff_indices[0]
            discharge_df = discharge_df.loc[:first_cutoff_index]

    direct_features = _calculate_direct_features(
        cycle_df, charge_df, discharge_df, cycle_num, battery_data
    )
    nominal_cap = battery_data.get('nominal_capacity_in_Ah', 1.0)

    # --- Cell-Specific Configuration ---
    # Define a list of special cells that require configuration overrides
    special_cells = [
        'NA-ion_270040-6-5-27',
        'NA-ion_270040-1-1-64',
        'NA-ion_270040-8-3-18'
    ]

    # Base configuration for all Na-ion cells
    na_ion_config = {
        'peak_mode': 2,
        'nominal_capacity': nominal_cap,
        'window_length_ic': 21,
        'window_length_dv': 21,
        'peak_height_ic': 0.01,
        'voltage_range_ic': (2.4, 3.0),  # Default voltage range
        'prominence_ic': 0.05,
        'ic_step_size': 0.01,
        'dv_step_size': nominal_cap * 0.005,
        'search_window_dvv': 0.1,
        'search_window_dvp': 0.1,
        'initial_capacity_cut_fraction': 0.02,
        'icv_search_offset_lower': 0.05,
        'icv_search_offset_upper': 0.5,
        'icv_search_direction': 'right'  # Default search direction
    }

    # Apply overrides for special cells
    if battery_data.cell_id in special_cells:
        na_ion_config['voltage_range_ic'] = (2.4, 3.5)
        na_ion_config['icv_search_direction'] = 'left'
        na_ion_config['icv_search_offset_upper'] = 1.0

    plot_params = None
    if output_dir:
        plot_params = {
            'cell_id': battery_data.cell_id,
            'cycle_num': cycle_num,
            'output_dir': output_dir
        }

    if not discharge_df.empty:
        derivative_features = extract_ic_dv_features(
            discharge_df,
            config=na_ion_config,
            plot_params=plot_params
        )
    else:
        derivative_features = extract_ic_dv_features(pd.DataFrame(), config=na_ion_config)

    advanced_features = _calculate_advanced_features(
        cycle_df, charge_df, discharge_df, rest_df, direct_features
    )
    anchor_features = _calculate_anchor_features(
        charge_df, discharge_df, charge_slope_intervals,
        discharge_slope_intervals, tevi_intervals, tevd_intervals
    )

    all_features = {
        **direct_features,
        **derivative_features,
        **advanced_features,
        **anchor_features,
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
) -> None:
    """Process a single battery .pkl file."""
    try:
        with open(file_path, 'rb') as f:
            data_dict = pickle.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return

    battery_data = AttrDict(data_dict)

    if 'cycle_data' in battery_data and battery_data['cycle_data'] is not None:
        battery_data.cycle_data = [
            AttrDict(c) for c in battery_data.cycle_data
        ]
    if ('charge_protocol' in battery_data and
            battery_data['charge_protocol'] is not None):
        battery_data.charge_protocol = [
            AttrDict(p) for p in battery_data.charge_protocol
        ]
    if ('discharge_protocol' in battery_data and
            battery_data['discharge_protocol'] is not None):
        battery_data.discharge_protocol = [
            AttrDict(p) for p in battery_data.discharge_protocol
        ]

    all_cycle_features = []
    cycles_to_process = battery_data.cycle_data

    if num_cycles is not None and num_cycles > 0:
        cycles_to_process = battery_data.cycle_data[:num_cycles]

    pbar_desc = f"Processing {battery_data.cell_id}"
    for cycle_data in tqdm(cycles_to_process, desc=pbar_desc):
        if not cycle_data.time_in_s:
            continue

        try:
            features = extract_features_for_cycle(
                cycle_data, battery_data,
                charge_slope_intervals, discharge_slope_intervals,
                tevi_intervals, tevd_intervals,
                output_dir=output_dir
            )
            all_cycle_features.append(features)
        except Exception:
            continue

    if not all_cycle_features:
        print(f"Warning: No features extracted for {battery_data.cell_id}")
        return

    features_df = pd.DataFrame(all_cycle_features)

    # Column Ordering
    ordered_cols_base = [
        'Cycle_Number',
        'Discharge_Capacity', 'Charge_Capacity',
        'Discharge_Energy', 'Charge_Energy',
        'Coulombic_Efficiency', 'Energy_Efficiency',
        'Rest_Time',
        'charge_c_rate', 'discharge_c_rate',
        'ICHV', 'UVP_time',
        'TCCC',
        'UVP',
        'IDV', 'LVP_time', 'var_I_discharge',
        'var_V_discharge', 'median_V_discharge', 'total_discharge_time',
        'LVP',
        'ICP', 'ICPL_V', 'ICP_FWHM', 'ICP_Area',
        'ICV', 'ICVL_V',
        'DVP', 'DVPL_V', #'DVP_FWHM', 'DVP_Area',
        'DVV', 'DVVL_V',
        'Internal_Resistance',
        'skew_V_discharge',
        'charge_slope_1', 'charge_slope_2', 'charge_slope_3',
        'discharge_slope_1', 'discharge_slope_2', 'discharge_slope_3',
        'TEVI_1', 'TEVI_2', 'TEVI_3', 'TEVD_1', 'TEVD_2', 'TEVD_3'
    ]

    existing_cols = [col for col in ordered_cols_base if col in features_df.columns]
    remaining_cols = [col for col in features_df.columns if col not in existing_cols]
    final_cols = existing_cols + sorted(remaining_cols)

    features_df = features_df[final_cols]

    output_file = output_dir / f"{battery_data.cell_id}.csv"
    features_df.to_csv(output_file, index=False)
    print(f"Features for {battery_data.cell_id} saved to {output_file}")


def main() -> None:
    """Main execution entry point."""
    processed_data_dir = Path('F:/datasets/battery/NA-ion')
    output_dir = project_root / 'results' / 'NA'
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Percentage-based slope intervals
    charge_slope_intervals = [(0.10, 0.30), (0.10, 0.50), (0.10, 0.80)]
    discharge_slope_intervals = [(0.10, 0.30), (0.10, 0.50), (0.10, 0.80)]

    # 2. Voltage-based anchor intervals (Specifically for Na-ion typical ranges)
    tevi_intervals = [(3.0, 3.2), (3.3, 3.5), (3.6, 3.8)]
    tevd_intervals = [(3.9, 3.6), (3.5, 3.2), (3.1, 2.8)]

    num_cycles_to_extract = 100

    if not processed_data_dir.exists():
        print(f"Error: Directory not found at '{processed_data_dir}'.")
        return

    pkl_files = list(processed_data_dir.glob('*.pkl'))
    if not pkl_files:
        print(f"Error: No .pkl files found in '{processed_data_dir}'.")
        return

    for file_path in pkl_files:
        try:
            process_battery(
                file_path, output_dir,
                charge_slope_intervals,
                discharge_slope_intervals,
                tevi_intervals,
                tevd_intervals,
                num_cycles=num_cycles_to_extract
            )
        except Exception as e:
            print(f"Failed to process {file_path.name}. Error: {e}")
            continue


if __name__ == '__main__':
    main()

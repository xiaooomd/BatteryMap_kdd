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

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


def _calculate_direct_features(
    cycle_df: pd.DataFrame,
    charge_df: pd.DataFrame,
    discharge_df: pd.DataFrame, 
    cycle_num: int, 
    battery_data: Any
) -> Dict[str, Any]:
    """
    Calculate direct features (Capacity, Energy, Efficiency, CV Dynamics).
    """
    features = {}

    # --- A. Overall Cycle and Operating Condition Features ---
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
    if chg_cap > 0:
        features['Coulombic_Efficiency'] = dis_cap / chg_cap
    else:
        features['Coulombic_Efficiency'] = 0.0
    
    # Rest Time
    if not charge_df.empty and not discharge_df.empty:
        time_charge_end = charge_df['Time(s)'].iloc[-1]
        time_discharge_start = discharge_df['Time(s)'].iloc[0]
        if time_discharge_start > time_charge_end:
            features['Rest_Time(s)'] = time_discharge_start - time_charge_end
        else:
            features['Rest_Time(s)'] = 0
    else:
        features['Rest_Time(s)'] = 0

    # C-Rates (Handle AttrDict structure)
    try:
        features['charge_c_rate'] = battery_data.charge_protocol[0].rate_in_C
        features['discharge_c_rate'] = battery_data.discharge_protocol[0].rate_in_C
    except (AttributeError, IndexError):
        features['charge_c_rate'] = np.nan
        features['discharge_c_rate'] = np.nan

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
        features['Energy_Efficiency'] = features['Discharge_Energy(Wh)'] / features['Charge_Energy(Wh)']
    else:
        features['Energy_Efficiency'] = 0.0

    # --- C. Charging Phase Features & CV Dynamics ---
    features['CV_Current_Tau'] = 0.0
    
    if not charge_df.empty:
        features['ICHV(V)'] = charge_df['Voltage(V)'].iloc[0]
        features['UVP_time(s)'] = charge_df['Time(s)'].iloc[-1]
        features['UVP(V)'] = battery_data.max_voltage_limit_in_V
        
        v_upper_limit = battery_data.max_voltage_limit_in_V
        charge_voltage = charge_df['Voltage(V)']
        
        # Check for CV Phase (Voltage >= Limit - Tolerance)
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
                features['CV_Current_Tau'] = fit_cv_decay(
                    cv_time[valid_mask],
                    cv_current[valid_mask]
                )
        else:
            features['TCCC(s)'] = charge_df['Time(s)'].iloc[-1] - charge_df['Time(s)'].iloc[0]
            features['TCVC(s)'] = 0
            
        # [Temp Removed] Temperature Features
        # if 'Temperature(C)' in charge_df.columns:
        #     features['MAT_charge(C)'] = charge_df['Temperature(C)'].max()
        #     features['MET_charge(s)'] = charge_df['Temperature(C)'].mean()
        # else:
        #     features['MAT_charge(C)'] = 0
        #     features['MET_charge(s)'] = 0
    else:
        features.update({'ICHV(V)': 0, 'UVP_time(s)': 0, 'UVP(V)': 0, 'TCCC(s)': 0, 'TCVC(s)': 0})
        # features.update({'MAT_charge(C)': 0, 'MET_charge(s)': 0}) # [Temp Removed]

    # --- D. Discharging Phase Features ---
    if not discharge_df.empty:
        features['IDV(V)'] = discharge_df['Voltage(V)'].iloc[0]
        features['LVP_time(s)'] = discharge_df['Time(s)'].iloc[-1]
        features['LVP(V)'] = battery_data.min_voltage_limit_in_V
        features['var_I_discharge'] = discharge_df['Current(A)'].var()
        features['var_V_discharge'] = discharge_df['Voltage(V)'].var()
        features['median_V_discharge(V)'] = discharge_df['Voltage(V)'].median()
        features['total_discharge_time(s)'] = discharge_df['Time(s)'].iloc[-1] - discharge_df['Time(s)'].iloc[0]
        
        # [Temp Removed] Temperature Features
        # if 'Temperature(C)' in discharge_df.columns:
        #     features['MAT_discharge(C)'] = discharge_df['Temperature(C)'].max()
        #     features['MET_discharge(s)'] = discharge_df['Temperature(C)'].mean()
        # else:
        #     features['MAT_discharge(C)'] = 0
        #     features['MET_discharge(s)'] = 0
    else:
        features.update({
            'IDV(V)': 0, 'LVP_time(s)': 0, 'LVP(V)': 0, 'var_I_discharge': 0, 
            'var_V_discharge': 0, 'median_V_discharge(V)': 0, 'total_discharge_time(s)': 0,
            # 'MAT_discharge(C)': 0, 'MET_discharge(s)': 0 # [Temp Removed]
        })

    return features




def _calculate_advanced_features(
    charge_df: pd.DataFrame, 
    discharge_df: pd.DataFrame, 
    rest_df: pd.DataFrame, 
    features: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Calculate derived features like Resistance, RCV.
    """
    adv_features = {}
    
    # Internal Resistance
    # Using (V_rest_end - V_discharge_start) / I_discharge
    if not charge_df.empty and not discharge_df.empty:
        v_discharge_start = discharge_df['Voltage(V)'].iloc[0]
        i_discharge_start = abs(discharge_df['Current(A)'].iloc[0])
        
        # Fallback logic for V_before_discharge
        if not rest_df.empty:
            v_before_discharge = rest_df['Voltage(V)'].iloc[-1]
        else:
            v_before_discharge = charge_df['Voltage(V)'].iloc[-1]

        if i_discharge_start > 0.001:
            adv_features['Internal_Resistance(Ohm)'] = max(0, (v_before_discharge - v_discharge_start) / i_discharge_start)
        else:
            adv_features['Internal_Resistance(Ohm)'] = 0.0
    else:
        adv_features['Internal_Resistance(Ohm)'] = 0.0

    # RCV Ratio
    if features.get('TCVC', 0) > 0:
        adv_features['RCV(V)'] = features.get('TCCC', 0) / features['TCVC(s)']
    else:
        adv_features['RCV(V)'] = 0.0

    # Discharge Statistics
    if not discharge_df.empty:
        adv_features['skew_V_discharge'] = skew(discharge_df['Voltage(V)'])
        # [Temp Removed] Temperature Stats
        # if 'Temperature(C)' in discharge_df.columns:
        #     adv_features['Temperature_Rise(C)'] = (
        #         discharge_df['Temperature(C)'].max() - 
        #         discharge_df['Temperature(C)'].iloc[0]
        #     )
        #     adv_features['skew_T_discharge'] = skew(discharge_df['Temperature(C)'])
        # else:
        #     adv_features['Temperature_Rise(C)'] = 0.0
        #     adv_features['skew_T_discharge'] = 0.0
    else:
        adv_features['skew_V_discharge'] = 0.0
        # adv_features['Temperature_Rise(C)'] = 0.0 # [Temp Removed]
        # adv_features['skew_T_discharge'] = 0.0 # [Temp Removed]

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
    Calculate anchor interval features using robust percentage-based logic.
    """
    anchor_features = {}

    def get_voltage_at_relative_time(df, relative_time):
        if df.empty: return None
        time_array = df['Time(s)'].values
        volt_array = df['Voltage(V)'].values
        start_time = time_array[0]
        abs_target_time = start_time + relative_time
        
        # Binary search for O(logN)
        idx = np.searchsorted(time_array, abs_target_time, side='left')
        
        if idx == 0: return volt_array[0]
        if idx == len(time_array): return volt_array[-1]
        
        # Check nearest neighbor
        if (abs_target_time - time_array[idx - 1]) < (time_array[idx] - abs_target_time):
            return volt_array[idx - 1]
        else:
            return volt_array[idx]

    # --- Charge Slopes (Percentage) ---
    c_dur = charge_df['Time(s)'].iloc[-1] - charge_df['Time(s)'].iloc[0] if not charge_df.empty else 0
    for i, (p_start, p_end) in enumerate(charge_slope_intervals):
        key = f'charge_slope_{i+1}'
        if c_dur > 0 and p_end > p_start:
            v_start = get_voltage_at_relative_time(charge_df, c_dur * p_start)
            v_end = get_voltage_at_relative_time(charge_df, c_dur * p_end)
            dt = c_dur * (p_end - p_start)
            if v_start is not None and v_end is not None and dt > 1e-6:
                anchor_features[key] = (v_end - v_start) / dt
            else:
                anchor_features[key] = 0
        else:
            anchor_features[key] = 0

    # --- Discharge Slopes (Percentage) ---
    d_dur = discharge_df['Time(s)'].iloc[-1] - discharge_df['Time(s)'].iloc[0] if not discharge_df.empty else 0
    for i, (p_start, p_end) in enumerate(discharge_slope_intervals):
        key = f'discharge_slope_{i+1}'
        if d_dur > 0 and p_end > p_start:
            v_start = get_voltage_at_relative_time(discharge_df, d_dur * p_start)
            v_end = get_voltage_at_relative_time(discharge_df, d_dur * p_end)
            dt = d_dur * (p_end - p_start)
            if v_start is not None and v_end is not None and dt > 1e-6:
                anchor_features[key] = (v_end - v_start) / dt
            else:
                anchor_features[key] = 0
        else:
            anchor_features[key] = 0

    # --- TEVI / TEVD ---
    def get_time_for_voltage(df, voltage, direction):
        if df.empty: return None
        if direction == 'charge':
            target_rows = df[df['Voltage(V)'] >= voltage]
        else:
            target_rows = df[df['Voltage(V)'] <= voltage]
        return target_rows['Time(s)'].iloc[0] if not target_rows.empty else None

    for i, (v_start, v_end) in enumerate(tevi_intervals):
        t_start = get_time_for_voltage(charge_df, v_start, 'charge')
        t_end = get_time_for_voltage(charge_df, v_end, 'charge')
        if t_start is not None and t_end is not None and t_end > t_start:
            anchor_features[f'TEVI_{i+1}'] = t_end - t_start
        else:
            anchor_features[f'TEVI_{i+1}'] = 0

    for i, (v_start, v_end) in enumerate(tevd_intervals):
        t_start = get_time_for_voltage(discharge_df, v_start, 'discharge')
        t_end = get_time_for_voltage(discharge_df, v_end, 'discharge')
        if t_start is not None and t_end is not None and t_end > t_start:
            anchor_features[f'TEVD_{i+1}'] = t_end - t_start
        else:
            anchor_features[f'TEVD_{i+1}'] = 0

    return anchor_features


def _calculate_personalized_features(cycle_df: pd.DataFrame, cell_id: str) -> Dict[str, Any]:
    return {}


def extract_features_for_cycle(
    cycle_data: Any,
    battery_data: Any,
    charge_slope_intervals: List[Tuple[float, float]],
    discharge_slope_intervals: List[Tuple[float, float]],
    tevi_intervals: List[Tuple[float, float]],
    tevd_intervals: List[Tuple[float, float]],
    output_dir: Optional[Path] = None
) -> Dict[str, Any]:
    
    # 1. Prepare Data
    cycle_df = pd.DataFrame({
        'Time(s)': cycle_data.time_in_s,
        'Current(A)': cycle_data.current_in_A,
        'Voltage(V)': cycle_data.voltage_in_V,
        'Charge_Capacity(Ah)': cycle_data.charge_capacity_in_Ah,
        'Discharge_Capacity(Ah)': cycle_data.discharge_capacity_in_Ah,
        # 'Temperature(C)': cycle_data.temperature_in_C # [Temp Removed]
    })
    cycle_num = cycle_data.cycle_number

    # 2. Phase Separation
    charge_df = cycle_df[cycle_df['Current(A)'] > 0].copy()
    discharge_df = cycle_df[cycle_df['Current(A)'] < 0].copy()

    # Cutoff Logic
    if not discharge_df.empty:
        cutoff_voltage = battery_data.min_voltage_limit_in_V
        cutoff_indices = discharge_df.index[discharge_df['Voltage(V)'] <= cutoff_voltage]
        if not cutoff_indices.empty:
            discharge_df = discharge_df.loc[:cutoff_indices[0]]

    # 3. Extract Rest Phase (for Internal Resistance)
    rest_df = pd.DataFrame()
    if not charge_df.empty and not discharge_df.empty:
        t_c_end = charge_df['Time(s)'].iloc[-1]
        t_d_start = discharge_df['Time(s)'].iloc[0]
        if t_d_start > t_c_end:
            rest_df = cycle_df[
                (cycle_df['Time(s)'] > t_c_end) & 
                (cycle_df['Time(s)'] < t_d_start)
            ].copy()

    # 4. Feature Extraction
    direct_features = _calculate_direct_features(
        cycle_df, charge_df, discharge_df, cycle_num, battery_data
    )
    nominal_cap = battery_data.get('nominal_capacity_in_Ah', 1.1)
    # Config for NCM (MICH/LG HG2)
    ncm_config = {
        'peak_mode': 1,
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
        'dv_step_size': nominal_cap*0.005,
        'search_window_dvv': 0.1,
        'search_window_dvp': 0.1,
        'initial_capacity_cut_fraction': 0.02,
        'icv_search_offset_lower': 0.05,
        'icv_search_offset_upper': 0.5
    }

    plot_params = None
    if output_dir:
        plot_params = {
            'cell_id': battery_data.cell_id,
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
        cycle_df, battery_data.cell_id
    )

    return {
        **direct_features, 
        **derivative_features, 
        **advanced_features, 
        **anchor_features, 
        **personalized_features
    }

# Helper class for dot notation access
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
    if 'cycle_data' in battery_data and battery_data['cycle_data'] is not None:
        battery_data.cycle_data = [AttrDict(c) for c in battery_data.cycle_data]
    if 'charge_protocol' in battery_data and battery_data['charge_protocol'] is not None:
        battery_data.charge_protocol = [AttrDict(p) for p in battery_data.charge_protocol]
    if 'discharge_protocol' in battery_data and battery_data['discharge_protocol'] is not None:
         battery_data.discharge_protocol = [AttrDict(p) for p in battery_data.discharge_protocol]

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
                output_dir=output_dir
            )
            all_cycle_features.append(features)
        except Exception:
            continue

    if not all_cycle_features:
        print(f"Warning: No features extracted for {battery_data.cell_id}")
        return

    features_df = pd.DataFrame(all_cycle_features)
    
    ordered_cols = [col for col in [
        # Overall
        'Cycle_Number', 
        'Discharge_Capacity', 'Charge_Capacity', 
        'Discharge_Energy', 'Charge_Energy', 
        'Coulombic_Efficiency', 'Energy_Efficiency',
        'Rest_Time', 'charge_c_rate', 'discharge_c_rate',
        # Charge
        'ICHV', 'UVP_time', 'TCCC', 'TCVC', 'CV_Current_Tau', 'UVP', 
        # 'MAT_charge', 'MET_charge', # [Temp Removed]
        # Discharge
        'IDV', 'LVP_time', 'var_I_discharge', 'var_V_discharge', 
        'median_V_discharge', 'total_discharge_time', 'LVP', 
        # 'MAT_discharge', 'MET_discharge', # [Temp Removed]
        # Curves
        'ICP', 'ICPL_V', 'ICP_FWHM', 'ICP_Area',
        'ICV', 'ICVL_V', 
        'DVP', 'DVPL_V', 'DVP_FWHM', 'DVP_Area',
        'DVV', 'DVVL_V', 
        # 'DTP', 'DTPL_V', 'DTV', 'DTVL_V', # [Temp Removed]
        # Advanced
        'Internal_Resistance', 'RCV', 'skew_V_discharge', 
        # 'Temperature_Rise', 'skew_T_discharge', # [Temp Removed]
        # Anchor
        'charge_slope_1', 'charge_slope_2', 'charge_slope_3',
        'discharge_slope_1', 'discharge_slope_2', 'discharge_slope_3',
        'TEVI_1', 'TEVI_2', 'TEVI_3', 'TEVD_1', 'TEVD_2', 'TEVD_3'
    ] if col in features_df.columns]
    
    final_cols = ordered_cols + [col for col in features_df.columns if col not in ordered_cols]
    features_df = features_df[final_cols]

    output_file = output_dir / f"{battery_data.cell_id}.csv"
    features_df.to_csv(output_file, index=False)
    print(f"Features for {battery_data.cell_id} saved to {output_file}")


def main():
    processed_data_dir = Path('F:/datasets/battery/MICH')
    output_dir = project_root / 'results' / 'MICH'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Intervals (Percentage for slopes)
    charge_slope_intervals = [(0.10, 0.30), (0.10, 0.50), (0.10, 0.80)]
    discharge_slope_intervals = [(0.10, 0.30), (0.10, 0.50), (0.10, 0.80)]
    
    # Voltage Intervals (MICH: 3.0V - 4.2V range)
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
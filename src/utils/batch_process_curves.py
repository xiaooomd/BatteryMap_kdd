import pickle
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src.physics.ic_dv_extractor import extract_ic_dv_features, identify_phases
from src.utils.plot_tools import plot_aggregated_icdv

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# --- Configuration & Helpers ---

def get_default_config(nominal_capacity: float = 1.1) -> Dict[str, Any]:
    """Returns default NCM/LCO config for IC/DV extraction."""
    return {
        'peak_mode': 2, # Multi-peak
        'nominal_capacity': nominal_capacity,
        'window_length_ic': 25,
        'window_length_dv': 25,
        'peak_height_ic': 0.01,
        'voltage_range_ic': (3.0, 4.3), # Broad range
        'prominence_ic': 0.01,
        'ic_step_size': 0.002,
        'dv_step_size': nominal_capacity * 0.005,
        'search_window_dvv': 0.1,
        'search_window_dvp': 0.1,
        'initial_capacity_cut_fraction': 0.0,
        'icv_search_offset_lower': 0.05,
        'icv_search_offset_upper': 0.5
    }

def get_lfp_config(nominal_capacity: float = 1.1) -> Dict[str, Any]:
    """Returns default LFP config."""
    return {
        'peak_mode': 1, # Single-peak
        'nominal_capacity': nominal_capacity,
        'window_length_ic': 51,
        'window_length_dv': 21,
        'peak_height_ic': 0.01,
        'voltage_range_ic': (2.8, 3.6),
        'prominence_ic': 0.02,
        'ic_step_size': 0.001,
        'dv_step_size': nominal_capacity * 0.005,
        'search_window_dvv': 0.1,
        'search_window_dvp': 0.1,
        'initial_capacity_cut_fraction': 0.02,
        'icv_search_offset_lower': 0.05,
        'icv_search_offset_upper': 0.5,
        'disable_dvv': True # Usually disabled for LFP in this project
    }

def get_na_config(nominal_capacity: float = 1.0) -> Dict[str, Any]:
    """Returns default Na-ion config."""
    return {
        'peak_mode': 2,
        'nominal_capacity': nominal_capacity,
        'window_length_ic': 21,
        'window_length_dv': 21,
        'peak_height_ic': 0.01,
        'voltage_range_ic': (2.4, 3.5),
        'prominence_ic': 0.05,
        'ic_step_size': 0.01,
        'dv_step_size': nominal_capacity * 0.005,
        'search_window_dvv': 0.1,
        'search_window_dvp': 0.1,
        'initial_capacity_cut_fraction': 0.02,
        'icv_search_offset_lower': 0.05,
        'icv_search_offset_upper': 0.5,
        'icv_search_direction': 'right'
    }

def get_zn_config(nominal_capacity: float = 0.5) -> Dict[str, Any]:
    """Returns default Zn-ion config."""
    return {
        'peak_mode': 2,
        'nominal_capacity': nominal_capacity,
        'window_length_ic': 21,
        'window_length_dv': 21,
        'peak_height_ic': 0.0001,
        'voltage_range_ic': (1.0, 1.6),
        'prominence_ic': 0.0001,
        'ic_step_size': 0.005,
        'dv_step_size': nominal_capacity * 0.005,
        'search_window_dvv': 0.3,
        'search_window_dvp': 0.3,
        'initial_capacity_cut_fraction': 0.02,
        'icv_search_offset_lower': 0.01,
        'icv_search_offset_upper': 0.25,
        'icv_search_direction': 'left'
    }

# --- HUST Specific Helper ---
def detect_hust_stages(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Detects operational stages (C1, C2...) for HUST dataset."""
    if df.empty: return []
    curr = df['Current(A)'].values
    dI = np.abs(np.diff(curr, prepend=curr[0]))
    is_step = dI > 0.5 # Threshold
    step_indices = np.where(is_step)[0]
    step_indices = step_indices[step_indices > 0]
    boundaries = [0] + sorted(list(set(step_indices))) + [len(df)]
    stages = []
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i+1]
        if end - start < 5: continue
        stages.append({'start': start, 'end': end, 'current': np.mean(curr[start:end])})
    return stages

# --- Dataset Processors ---

def preprocess_standard(cycle_data: Dict[str, Any]) -> pd.DataFrame:
    """Standard preprocessing: extract Discharge dataframe."""
    df = pd.DataFrame({
        'Time(s)': cycle_data['time_in_s'],
        'Current(A)': cycle_data['current_in_A'],
        'Voltage(V)': cycle_data['voltage_in_V'],
        'Discharge_Capacity(Ah)': cycle_data['discharge_capacity_in_Ah']
    })
    # Filter Discharge (Current < -0.001 roughly)
    discharge_df = df[df['Current(A)'] < -0.001].copy()

    # Simple Cutoff (if needed, but extract_ic_dv handles some noise)
    return discharge_df

def preprocess_hust(cycle_data: Dict[str, Any]) -> pd.DataFrame:
    """HUST preprocessing: extract Charge C2 phase."""
    df = pd.DataFrame({
        'Time(s)': cycle_data['time_in_s'],
        'Current(A)': cycle_data['current_in_A'],
        'Voltage(V)': cycle_data['voltage_in_V'],
        'Charge_Capacity(Ah)': cycle_data['charge_capacity_in_Ah'] # Use Charge Cap
    })
    # Filter Charge
    charge_df = df[df['Current(A)'] > 0.001].copy()

    stages = detect_hust_stages(charge_df)
    target_df = pd.DataFrame()

    if len(stages) >= 2:
        # Use Stage 2 (Index 1) for HUST
        s, e = stages[1]['start'], stages[1]['end']
        target_df = charge_df.iloc[s:e].copy()
    elif len(stages) == 1:
        target_df = charge_df.copy()
    else:
        target_df = charge_df.copy()

    # Map Charge Cap to Discharge Cap column for compatibility
    if not target_df.empty:
        target_df['Discharge_Capacity(Ah)'] = target_df['Charge_Capacity(Ah)']

    return target_df

def preprocess_rwth(cycle_data: Dict[str, Any]) -> pd.DataFrame:
    """RWTH preprocessing: Uses constant current charge or discharge."""
    # Assuming standard discharge for now, based on previous exploration
    return preprocess_standard(cycle_data)

def preprocess_coin_cell(cycle_data: Dict[str, Any]) -> pd.DataFrame:
    """Coin cell preprocessing: Handles small currents (< 0)."""
    df = pd.DataFrame({
        'Time(s)': cycle_data['time_in_s'],
        'Current(A)': cycle_data['current_in_A'],
        'Voltage(V)': cycle_data['voltage_in_V'],
        'Discharge_Capacity(Ah)': cycle_data['discharge_capacity_in_Ah']
    })
    # Filter Discharge (Current < 0, sensitive for coin cells)
    discharge_df = df[df['Current(A)'] < 0].copy()
    return discharge_df

# --- Main Processing Logic ---

def process_dataset_curves(
    dataset_name: str,
    input_dir: Path,
    output_root: Path,
    config: Dict[str, Any],
    preprocess_func: Callable[[Dict[str, Any]], pd.DataFrame],
    file_pattern: str = "*.pkl",
    num_cycles: int = 100
):
    print(f"\nProcessing Dataset: {dataset_name}")
    print(f"Input: {input_dir}")

    if not input_dir.exists():
        print(f"Error: Directory not found: {input_dir}")
        return

    files = list(input_dir.glob(file_pattern))
    if not files:
        print(f"No files found matching {file_pattern}")
        return

    # Create dataset specific output dir
    curve_output_dir = output_root / dataset_name
    curve_output_dir.mkdir(parents=True, exist_ok=True)

    for file_path in tqdm(files, desc=f"{dataset_name}"):
        try:
            with open(file_path, 'rb') as f:
                battery_data = pickle.load(f)
        except Exception as e:
            print(f"Failed to load {file_path.name}: {e}")
            continue

        cell_id = battery_data.get('cell_id', file_path.stem)
        cycle_data_list = battery_data.get('cycle_data', [])

        # Handle HUST skipped cycles
        if dataset_name == 'HUST' and cell_id == 'HUST_7-5':
            cycle_data_list = cycle_data_list[2:]

        # Limit cycles
        cycles_to_process = cycle_data_list[:num_cycles]

        all_curves = []

        for i, c_data in enumerate(cycles_to_process):
            if not c_data.get('time_in_s'):
                continue

            try:
                # 1. Preprocess (Extract relevant phase)
                phase_df = preprocess_func(c_data)

                if phase_df.empty or len(phase_df) < 10:
                    continue

                # 2. Extract Curves
                # We only need the curves, so we can ignore the feature dict mostly
                features = extract_ic_dv_features(
                    phase_df,
                    config=config,
                    include_curves=True
                )

                if 'curves' in features:
                    curve_data = features['curves']
                    curve_data['cycle'] = i + 1
                    all_curves.append(curve_data)

            except Exception:
                continue

        # 3. Save and Plot
        if all_curves:
            # Save CSV
            flattened_curves = []
            for curve in all_curves:
                cycle_num = curve['cycle']
                # IC data
                ic_df = pd.DataFrame({
                    'cycle': cycle_num,
                    'type': 'IC',
                    'x_voltage': curve['v_grid_ic'],
                    'y_value': curve['ic_smooth']
                })
                # DV data
                dv_df = pd.DataFrame({
                    'cycle': cycle_num,
                    'type': 'DV',
                    'x_capacity': curve['q_grid_dv'],
                    'y_value': curve['dv_smooth']
                })
                flattened_curves.extend([ic_df, dv_df])

            if flattened_curves:
                curves_df = pd.concat(flattened_curves, ignore_index=True)
                csv_path = curve_output_dir / f"{cell_id}.csv"
                curves_df.to_csv(csv_path, index=False)

            # Plot Aggregated
            plot_aggregated_icdv(
                cell_id=cell_id,
                dataset_name=dataset_name,
                all_curves=all_curves,
                output_dir=output_root
            )
        else:
            # print(f"No curves extracted for {cell_id}")
            pass

def main():
    # Define Output Root
    output_root = project_root / 'results' / 'icdv_curves'
    output_root.mkdir(parents=True, exist_ok=True)

    # Define Datasets Configuration
    # Paths now relative to project root or injected via ENV/Args
    # TODO: These paths should ideally be passed as arguments or loaded from a config file
    # For now, we assume a standard structure or rely on user to mount data there.
    # To fix hardcoded paths, we check project_root/data/... first

    data_root = project_root / 'data'

    datasets = [
        {
            'name': 'NA',
            'path': data_root / 'NA-ion',
            'config': get_na_config(nominal_capacity=1.0),
            'preprocess': preprocess_coin_cell
        },
        {
            'name': 'ZNion',
            'path': data_root / 'ZN-coin',
            'config': get_zn_config(nominal_capacity=0.5),
            'preprocess': preprocess_coin_cell
        },
        # {
        #     'name': 'CALB',
        #     'path': data_root / 'calb_pkl',
        #     'config': get_default_config(nominal_capacity=2.0), # NCM?
        #     'preprocess': preprocess_standard
        # },
        # ... (Other datasets commented out in original)
    ]

    for ds in datasets:
        process_dataset_curves(
            dataset_name=ds['name'],
            input_dir=ds['path'],
            output_root=output_root,
            config=ds['config'],
            preprocess_func=ds['preprocess'],
            num_cycles=100
        )

    print("\nAll datasets processed.")

if __name__ == "__main__":
    main()

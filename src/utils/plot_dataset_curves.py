import argparse
import sys
import pickle
from pathlib import Path
from typing import List, Optional, Dict
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

# ==========================================
# --- Configuration: Dataset Paths ---
# ==========================================
# Updated to use relative paths under 'data/' directory.
# This assumes the user places datasets in a 'data' folder at the project root.

DATA_ROOT = project_root / 'data'

DATASET_PATHS: Dict[str, Path] = {
    'CALB': DATA_ROOT / 'calb_pkl',
    'CALCE': DATA_ROOT / 'CALCE',
    'HNEI': DATA_ROOT / 'HNEI',
    'HUST': DATA_ROOT / 'HUST',
    'ISU-ILCC': DATA_ROOT / 'ISU_ILCC',
    'MATR': DATA_ROOT / 'MATR',
    'MICH': DATA_ROOT / 'MICH',
    'MICH_EXP': DATA_ROOT / 'MICH_EXP',
    'NA': DATA_ROOT / 'NA-ion',
    'NA-ION': DATA_ROOT / 'NA-ion',
    'RWTH': DATA_ROOT / 'RWTH',
    'SNL': DATA_ROOT / 'SNL',
    'SNL_LFP': DATA_ROOT / 'SNL',
    'STANFORD': DATA_ROOT / 'Stanford',
    'STANFORD2': DATA_ROOT / 'Stanford_2',
    'TONGJI': DATA_ROOT / 'Tongji',
    'UL_PUR': DATA_ROOT / 'UL_PUR',
    'XJTU': DATA_ROOT / 'XJTU',
    'ZNION': DATA_ROOT / 'ZN-coin'
}

def get_processed_data_dir(dataset_name: str) -> Optional[Path]:
    """
    Retrieves the data directory from the configuration.
    """
    key = dataset_name.upper()
    if key in DATASET_PATHS:
        return DATASET_PATHS[key]
    return None

def plot_single_cell(file_path: Path, output_dir: Path, target_cycles: List[int], dataset_name: str = ""):
    """
    Plots V/I curves for a single cell.

    Args:
        file_path: Path to the .pkl file
        output_dir: Directory to save the plot
        target_cycles: List of cycles to plot. If empty, plots ALL cycles.
        dataset_name: Name of the dataset (e.g., 'XJTU') to apply specific fixes.
    """
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"Error loading {file_path.name}: {e}")
        return

    cell_id = data.get('cell_id', file_path.stem)
    cycle_data_list = data.get('cycle_data', [])

    if not cycle_data_list:
        # Some datasets might structure it differently, but sticking to standard schema for now
        return

    # Determine which cycles to plot
    total_cycles = len(cycle_data_list)
    cycles_to_plot = []

    if not target_cycles:
        # If target_cycles is empty (passed as empty list), plot ALL cycles
        cycles_to_plot = range(1, total_cycles + 1)
        title_suffix = "(All Cycles)"
    else:
        # Filter target cycles to those that exist
        cycles_to_plot = [c for c in target_cycles if 1 <= c <= total_cycles]
        title_suffix = f"(Cycles: {cycles_to_plot})"

    if not cycles_to_plot:
        return

    # Setup Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, layout='constrained')

    # Use a colormap
    cmap = plt.get_cmap('viridis')
    norm = plt.Normalize(vmin=min(cycles_to_plot), vmax=max(cycles_to_plot))

    plotted_count = 0

    # Optimization: if plotting ALL and total is huge, maybe downsample?
    # For now, we plot all as requested.

    for cycle_num in cycles_to_plot:
        idx = cycle_num - 1
        if idx >= len(cycle_data_list):
            continue

        cycle = cycle_data_list[idx]

        # Check integrity (dict vs object access depending on dataset, assuming dict based on prev code)
        # Some scripts wrap in AttrDict. Let's handle both.
        if isinstance(cycle, dict):
            t = cycle.get('time_in_s')
            v = cycle.get('voltage_in_V')
            i = cycle.get('current_in_A')
        else: # Object/AttrDict access
            t = getattr(cycle, 'time_in_s', None)
            v = getattr(cycle, 'voltage_in_V', None)
            i = getattr(cycle, 'current_in_A', None)

        if t is None or v is None or i is None:
            continue

        t = np.array(t)
        v = np.array(v)
        i = np.array(i)

        # --- FIX: XJTU Time Continuity ---
        if dataset_name.upper() == 'XJTU' and len(t) > 1:
            # XJTU dataset resets time to 0 at each step (Charge/Rest/Discharge/Rest)
            # We need to stitch them together to form a continuous timeline for the cycle.
            diffs = np.diff(t)
            # Find indices where time jumps backwards (e.g. 3000 -> 0)
            reset_indices = np.where(diffs < -1.0)[0]

            if len(reset_indices) > 0:
                # Split into segments
                # split_indices should be reset_indices + 1 because diff[i] corresponds to t[i+1] - t[i]
                # If diff[i] is negative, it means t[i+1] < t[i], so split at i+1
                split_indices = reset_indices + 1
                segments = np.split(t, split_indices)

                # Reconstruct time
                current_offset = 0.0
                fixed_segments = []
                for seg in segments:
                    if len(seg) == 0: continue
                    fixed_segments.append(seg + current_offset)
                    # Next segment starts at 0, so we add the last time of the current fixed segment
                    # to the offset.
                    # Wait, seg is raw.
                    # Correct logic:
                    # fixed_seg_i = raw_seg_i + offset_i
                    # offset_{i+1} = fixed_seg_i[-1]
                    # because raw_seg_{i+1} starts at 0.
                    current_offset += seg[-1]

                t = np.concatenate(fixed_segments)

        # Shift time to start at 0 (for the whole cycle)
        if len(t) > 0:
            t = t - t[0]

        color = cmap(norm(cycle_num))

        # Visual Optimization for many lines
        alpha = 0.3 if len(cycles_to_plot) > 50 else 0.8
        lw = 0.5 if len(cycles_to_plot) > 50 else 1.5

        ax1.plot(t, v, color=color, alpha=alpha, linewidth=lw, label=f'Cycle {cycle_num}' if len(cycles_to_plot) <= 10 else None)
        ax2.plot(t, i, color=color, alpha=alpha, linewidth=lw)
        plotted_count += 1

    if plotted_count == 0:
        plt.close(fig)
        return

    ax1.set_ylabel('Voltage (V)', fontsize=12)
    ax1.set_title(f'{cell_id} - Voltage Curves {title_suffix}', fontsize=14)
    ax1.grid(True, alpha=0.3)
    if len(cycles_to_plot) <= 10:
        ax1.legend(loc='upper right', fontsize='small')

    ax2.set_ylabel('Current (A)', fontsize=12)
    ax2.set_title(f'{cell_id} - Current Curves', fontsize=14)
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.grid(True, alpha=0.3)

    # Add Colorbar if many cycles
    if len(cycles_to_plot) > 10:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        # Using constrained_layout (via subplots layout='constrained')
        # colorbar will be automatically placed to the right of the axes
        cbar = fig.colorbar(sm, ax=[ax1, ax2], label='Cycle Number')

    # plt.tight_layout() is not needed and incompatible with constrained_layout

    output_path = output_dir / f"{cell_id}.png"
    plt.savefig(output_path, dpi=150)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="Plot Voltage and Current curves for a dataset.")
    parser.add_argument("dataset_name", help="Name of the dataset (e.g., CALB, HUST)")
    parser.add_argument("--cycles", nargs='*', type=int, help="Specific cycles to plot. If flag is present but no numbers (or passed explicitly empty), plots ALL cycles. Default if flag omitted: [1, 50, 100].")

    args = parser.parse_args()
    dataset_name = args.dataset_name

    # Logic for target_cycles
    target_cycles = []

    # Check if --cycles was provided
    if args.cycles is not None:
        if len(args.cycles) == 0:
            # Case: --cycles passed with no args -> Plot ALL
            target_cycles = []
            print(f"Mode: Plotting ALL cycles for {dataset_name}")
        else:
            # Case: --cycles 10 20 -> Plot specific
            target_cycles = args.cycles
            print(f"Mode: Plotting specified cycles: {target_cycles}")
    else:
        # Case: --cycles NOT passed -> Default behavior
        target_cycles = [1, 50, 100]
        print(f"Mode: Plotting default cycles: {target_cycles}")

    # 1. Find Data Directory
    data_dir = get_processed_data_dir(dataset_name)
    if not data_dir:
        print(f"Error: Unknown dataset '{dataset_name}'. Please check the name or update DATASET_PATHS.")
        print("Available datasets:", ", ".join(sorted(DATASET_PATHS.keys())))
        return

    if not data_dir.exists():
        print(f"Data directory does not exist on disk: {data_dir}")
        return

    # 2. Setup Output Directory
    output_dir = project_root / 'results' / f'{dataset_name}_vol_curr_curves'
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # 3. Scan Files
    pkl_files = list(data_dir.glob('*.pkl'))
    if not pkl_files:
        print(f"No .pkl files found in {data_dir}")
        return

    print(f"Found {len(pkl_files)} files in {data_dir}. Starting processing...")

    # 4. Process Loop
    for pkl_file in tqdm(pkl_files):
        plot_single_cell(pkl_file, output_dir, target_cycles, dataset_name=dataset_name)

    print(f"\nCompleted! Results saved to {output_dir}")

if __name__ == "__main__":
    main()

"""
Plotting utilities for battery feature extraction.
Handles visualization of IC/DV curves and detected features.
"""

from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np
import matplotlib.pyplot as plt

def plot_aggregated_icdv(
    cell_id: str,
    dataset_name: str,
    all_curves: List[Dict[str, Any]],
    output_dir: Path
) -> None:
    """
    Plots aggregated IC and DV curves for multiple cycles on two separate plots.

    Args:
        cell_id: Cell identifier.
        dataset_name: Name of the dataset.
        all_curves: List of dictionaries containing curve data for each cycle.
        output_dir: Root directory for saving plots.
    """
    if not all_curves:
        return

    plot_dir = output_dir / f"{dataset_name}_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Use a colormap to distinguish cycles (cool to warm)
    cmap = plt.get_cmap('jet')
    num_cycles = len(all_curves)

    # 1. Aggregated IC Plot
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    for i, curve_data in enumerate(all_curves):
        color = cmap(i / num_cycles)
        ax1.plot(
            curve_data['v_grid_ic'],
            curve_data['ic_smooth'],
            color=color,
            alpha=0.6,
            linewidth=1
        )
    ax1.set_xlabel('Voltage (V)')
    ax1.set_ylabel('dQ/dV (Ah/V)')
    ax1.set_title(f'Aggregated IC Curves - {cell_id} ({num_cycles} cycles)')
    ax1.grid(True, alpha=0.3)

    save_path_ic = plot_dir / f"{cell_id}_aggregated_IC.png"
    plt.savefig(save_path_ic, dpi=150, bbox_inches='tight')
    plt.close(fig1)

    # 2. Aggregated DV Plot
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    for i, curve_data in enumerate(all_curves):
        color = cmap(i / num_cycles)
        ax2.plot(
            curve_data['q_grid_dv'],
            curve_data['dv_smooth'],
            color=color,
            alpha=0.6,
            linewidth=1
        )
    ax2.set_xlabel('Capacity (Ah)')
    ax2.set_ylabel('dV/dQ (V/Ah)')
    ax2.set_title(f'Aggregated DV Curves - {cell_id} ({num_cycles} cycles)')
    ax2.grid(True, alpha=0.3)

    save_path_dv = plot_dir / f"{cell_id}_aggregated_DV.png"
    plt.savefig(save_path_dv, dpi=150, bbox_inches='tight')
    plt.close(fig2)

def plot_ic_dv_curves(
    cycle_num: int,
    v_grid_ic: np.ndarray,
    ic_curve: np.ndarray,
    q_grid_dv: np.ndarray,
    dv_curve: np.ndarray,
    features: Dict[str, Any],
    output_dir: Path,
    cell_id: str,
    plot_interval: int = 20
) -> None:
    """
    Plots and saves IC and DV curves for specific cycles.

    Args:
        cycle_num: Current cycle number.
        v_grid_ic: Voltage grid for IC curve.
        ic_curve: IC (dQ/dV) values.
        q_grid_dv: Capacity grid for DV curve.
        dv_curve: DV (dV/dQ) values.
        features: Dictionary of extracted features (peaks, areas, etc.).
        output_dir: Directory to save plots.
        cell_id: Cell identifier.
        plot_interval: Interval for saving plots (e.g., every 50 cycles).
                       Always plots cycle 1.
    """
    # Option B: Plot at intervals and first cycle
    # Note: We might not know if it's the "last" cycle here easily,
    # but covering start and intervals is usually sufficient.
    if cycle_num != 1 and cycle_num % plot_interval != 0:
        return

    # Create subfolder for plots if it doesn't exist
    plot_dir = output_dir.parent / f"{output_dir.name}_curves"
    plot_dir.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'Cycle {cycle_num} Analysis - {cell_id}')

    # --- 1. IC Curve (dQ/dV vs V) ---
    ax1.plot(v_grid_ic, ic_curve, 'b-', label='IC Curve')
    ax1.set_xlabel('Voltage (V)')
    ax1.set_ylabel('dQ/dV (Ah/V)')
    ax1.set_title('Incremental Capacity (IC)')
    ax1.grid(True, alpha=0.3)

    # Mark ICP
    if features.get('ICP', 0) > 0:
        ax1.plot(features['ICPL_V'], features['ICP'], 'ro', label='ICP')
        # Mark FWHM if available (approximate visual)
        # Note: We don't have exact FWHM x-coordinates passed here, just the width
        # So we just mark the peak.

    # Mark ICV
    if features.get('ICV', 0) > 0:
         ax1.plot(features['ICVL_V'], features['ICV'], 'go', label='ICV')

    ax1.legend()

    # --- 2. DV Curve (dV/dQ vs Q) ---
    ax2.plot(q_grid_dv, dv_curve, 'r-', label='DV Curve')
    ax2.set_xlabel('Capacity (Ah)')
    ax2.set_ylabel('dV/dQ (V/Ah)')
    ax2.set_title('Differential Voltage (DV)')
    ax2.grid(True, alpha=0.3)

    # Mark DVP
    if features.get('DVP', 0) > 0:
        ax2.plot(features.get('DVP_Q', 0), features['DVP'], 'bo', label='DVP')

    # Mark DVV
    if features.get('DVV', 0) > 0:
        ax2.plot(features.get('DVV_Q', 0), features['DVV'], 'go', label='DVV')

    # Ideally we should pass Q locations of peaks if we want to plot them vs Q.
    # For now, just showing the curve is helpful enough for debugging shape.
    ax2.legend()

    # Save
    save_path = plot_dir / f"{cell_id}_cyc{cycle_num:04d}.png"
    plt.savefig(save_path, dpi=100)
    plt.close(fig)

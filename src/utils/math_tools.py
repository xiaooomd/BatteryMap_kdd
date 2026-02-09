"""
Math utilities for battery feature extraction.
Contains integration, differentiation, fitting, and interpolation tools.
"""

from typing import Tuple, Optional, List
import numpy as np
from scipy.optimize import curve_fit
from scipy.integrate import trapezoid, simpson
from scipy.signal import savgol_filter

def get_interp_val(arr: np.ndarray, idx_float: float) -> float:
    """
    Linearly interpolate value in array at floating point index.

    Args:
        arr: The data array (e.g., voltage or capacity).
        idx_float: The floating point index.

    Returns:
        float: Interpolated value.
    """
    if len(arr) == 0:
        return 0.0

    low = int(np.floor(idx_float))
    high = int(np.ceil(idx_float))

    # Boundary checks
    low = max(0, min(low, len(arr) - 1))
    high = max(0, min(high, len(arr) - 1))

    if low == high:
        return float(arr[low])

    frac = idx_float - low
    return float(arr[low] * (1 - frac) + arr[high] * frac)


def fit_cv_decay(time_series: np.ndarray, current_series: np.ndarray) -> float:
    """
    Fits an exponential decay model to the CV phase current.
    Model: I(t) = a * exp(-t / tau) + c

    Args:
        time_series: Array of time stamps.
        current_series: Array of current values.

    Returns:
        float: The time constant (tau). Returns 0.0 if fit fails.
    """
    def exponential_decay(t, a, tau, c):
        return a * np.exp(-t / tau) + c

    if len(time_series) < 10:
        return 0.0

    # Normalize time to start at 0 to avoid overflow
    t_norm = time_series - time_series[0]

    # Check if data is constant (or near constant) to avoid fitting errors
    if np.std(current_series) < 1e-6:
        return 0.0

    # Initial guess: a=range, tau=100s, c=end_val
    p0 = [
        current_series[0] - current_series[-1],
        100,
        current_series[-1]
    ]

    # Bounds: a>0, tau>0, c can be whatever
    bounds = ([0, 0, -np.inf], [np.inf, 10000, np.inf])

    try:
        popt, _ = curve_fit(
            exponential_decay,
            t_norm,
            current_series,
            p0=p0,
            bounds=bounds,
            maxfev=1000
        )
        return float(popt[1])  # Return tau
    except (RuntimeError, ValueError):
        return 0.0


def equidistant_resample(
    x: np.ndarray,
    y: np.ndarray,
    num_points: int = 1000
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resamples data onto an equidistant grid to reduce noise in differentiation.
    Fixes issue 2.2 in CLAUDE.md.

    Args:
        x: Original x-axis data (e.g., Voltage).
        y: Original y-axis data (e.g., Capacity).
        num_points: Number of points in the new grid.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (x_new, y_new)
    """
    if len(x) < 2:
        return x, y

    x_min, x_max = np.min(x), np.max(x)
    x_new = np.linspace(x_min, x_max, num_points)

    # Sort x for interpolation if not sorted
    if not np.all(np.diff(x) >= 0):
        sorted_idx = np.argsort(x)
        x_sorted = x[sorted_idx]
        y_sorted = y[sorted_idx]
        # Remove duplicates
        unique_mask = np.concatenate(([True], np.diff(x_sorted) > 1e-9))
        x_sorted = x_sorted[unique_mask]
        y_sorted = y_sorted[unique_mask]

        y_new = np.interp(x_new, x_sorted, y_sorted)
    else:
        y_new = np.interp(x_new, x, y)

    return x_new, y_new


def calculate_area_with_baseline(
    x: np.ndarray,
    y: np.ndarray
) -> float:
    """
    Calculates area under curve with Sloped Baseline subtraction.
    Fixes issue 2.1 in CLAUDE.md.

    Algorithm:
    1. Define Sloped Baseline: Line connecting (x[0], y[0]) and (x[-1], y[-1]).
    2. Area = Integral(y - Baseline)

    Args:
        x: x-axis data (voltage).
        y: y-axis data (dQ/dV).

    Returns:
        float: The calculated area.
    """
    if len(x) < 2:
        return 0.0

    # Calculate linear baseline values at each x point
    if len(x) >= 2:
        slope = (y[-1] - y[0]) / (x[-1] - x[0]) if (x[-1] - x[0]) != 0 else 0
        baseline = y[0] + slope * (x - x[0])
    else:
        baseline = np.full_like(y, y[0])

    # Subtract baseline
    y_corrected = y - baseline

    # Integrate
    area = trapezoid(y_corrected, x=x)

    return float(abs(area))  # Return absolute area


def find_curvature_boundaries(
    y_curve: np.ndarray,
    peak_idx: int,
    window_length: int = 11,
    polyorder: int = 3,
    threshold: float = 1e-5
) -> Tuple[int, int]:
    """
    Finds the boundaries of a peak based on the curvature (2nd derivative).
    Implements 'Curvature Boundary Recognition' from problem fix plan.

    Args:
        y_curve: The signal array (e.g., dQ/dV).
        peak_idx: Index of the peak.
        window_length: Window length for Savitzky-Golay derivative.
        polyorder: Polynomial order for SG.
        threshold: Threshold for 2nd derivative to be considered 'flat'.

    Returns:
        Tuple[int, int]: (left_idx, right_idx)
    """
    if len(y_curve) < window_length:
        return max(0, peak_idx-1), min(len(y_curve)-1, peak_idx+1)

    # Calculate 2nd derivative (curvature)
    d2y = savgol_filter(y_curve, window_length=window_length, polyorder=polyorder, deriv=2)

    # Search Left
    l_idx = peak_idx
    for i in range(peak_idx - 1, 0, -1):
        # Stop if curvature crosses zero or becomes very small (inflection point approach)
        # Or if the signal itself starts rising again (valley)
        if d2y[i] * d2y[i+1] <= 0: # Zero crossing
            l_idx = i
            break
        if y_curve[i] < y_curve[i+1] and y_curve[i] < y_curve[i-1]: # Local minimum
            l_idx = i
            break

    # Search Right
    r_idx = peak_idx
    for i in range(peak_idx + 1, len(y_curve) - 1):
        if d2y[i] * d2y[i-1] <= 0: # Zero crossing
            r_idx = i
            break
        if y_curve[i] < y_curve[i+1] and y_curve[i] < y_curve[i-1]: # Local minimum
            r_idx = i
            break

    return l_idx, r_idx

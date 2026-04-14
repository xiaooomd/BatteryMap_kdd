# Battery Feature Definitions

This document provides detailed explanations of battery features and their physical meanings used in the feature extraction scripts. Features are categorized into Energy, Kinetics, Thermodynamics, Curve Analysis, and Statistical Geometry.

## 1. Energy (Energy and Capacity Features)
These features reflect the battery's macroscopic energy storage capacity and conversion efficiency, directly related to the battery's SOH (State of Health).

*   **Discharge_Capacity (Ah)**: Discharge capacity. Total charge released during a complete discharge in a single cycle.
    *   *Formula*: $\int |I_{dis}| dt$
*   **Charge_Capacity (Ah)**: Charge capacity. Total charge input during a complete charge in a single cycle.
*   **Discharge_Energy (Wh)**: Discharge energy. Total energy released during a single cycle.
    *   *Formula*: $\int V \cdot |I_{dis}| dt$
*   **Charge_Energy (Wh)**: Charge energy. Total energy input during a single cycle.
*   **Coulombic_Efficiency (%)**: Coulombic efficiency. Ratio of discharge capacity to charge capacity.
    *   *Formula*: $Q_{dis} / Q_{chg}$
*   **Energy_Efficiency (%)**: Energy efficiency. Ratio of discharge energy to charge energy.

## 2. Kinetics (Kinetic Features)
These features reflect the battery's internal reaction rate, impedance, and electrochemical polarization state.

*   **Internal_Resistance (Ohm)**: DC Internal Resistance (DCIR). Typically calculated from voltage transients during charge/discharge switching or after rest periods.
    *   *Formula*: $\Delta V / \Delta I$
*   **CV_Current_Tau (s)**: Current decay time constant during the Constant Voltage (CV) charge phase. Reflects lithium-ion diffusion rate.
    *   *Fitting*: $\tau$ from $I(t) = A \cdot e^{-t/\tau} + C$.
*   **TCCC (s)**: Time of Constant Current Charge.
*   **TCVC (s)**: Time of Constant Voltage Charge.
*   **RCV**: Ratio of constant current time to constant voltage time, reflecting the increase in battery polarization.
    *   *Formula*: $TCCC / TCVC$
*   **charge_c_rate / discharge_c_rate**: Charge/discharge C-rate. Normalized current intensity ($I / Q_{nominal}$).
*   **Rest_Time (s)**: Rest time. Relaxation time between charge and discharge.
*   **total_discharge_time (s)**: Total discharge duration.
*   **UVP_time (s)**: Duration before reaching over-voltage protection (Upper Voltage Limit).
*   **LVP_time (s)**: Duration before reaching under-voltage protection (Lower Voltage Limit).
*   **V_rest_end (V)**: Voltage at the end of rest, approximating open-circuit voltage (OCV).
*   **charge_time_1 / charge_time_2 ...**: Duration of each constant current phase in multi-stage charge protocols.
*   **discharge_time_1 / discharge_time_2 ...**: Duration of each phase in multi-stage discharge protocols.
*   **charge_time_ratio_1_2**: Ratio of first-phase charge time to second-phase charge time, used to capture phase shift during aging.
*   **TEVI (Time in Every Voltage Interval)**: Time spent in specific voltage intervals (e.g., 3.0-3.1V) during charge.
*   **TEVD (Time in Every Voltage Interval - Discharge)**: Time spent in specific voltage intervals during discharge.

## 3. Thermodynamics (Thermodynamic Features)
These features are derived from battery surface temperature monitoring, reflecting the battery's heat generation and thermal management state.

*   **DTP (Difference in Temperature Peak)**: Difference between peak temperatures during charge and discharge phases.
*   **DTPL_V (Voltage at DTP)**: Terminal voltage at which peak temperature difference occurs.
*   **MAT_charge / MAT_discharge (°C)**: Maximum Actual Temperature during charge/discharge.
*   **MET_charge / MET_discharge (°C)**: Mean Evaluation Temperature during charge/discharge.
*   **MinT_charge / MinT_discharge (°C)**: Minimum temperature during charge/discharge.
*   **T_rise_charge / T_rise_discharge (°C)**: Temperature rise. End temperature minus start temperature of the phase.
*   **Thermal_Load_charge / Thermal_Load_discharge (°C·s)**: Thermal load. Integral of temperature over time, representing cumulative heat exposure.
*   **skew_T_discharge**: Skewness of the temperature distribution during discharge, reflecting non-uniformity of heat generation.

## 4. Curve (Curve Features: IC/DV)
Features extracted from incremental capacity (dQ/dV) and differential voltage (dV/dQ) curves, core to electrochemical mechanism analysis.

*   **ICP (Incremental Capacity Peak)**: Peak height of the IC curve ($dQ/dV$). Represents reaction rate at the most active phase transition point.
*   **ICPL_V (V)**: Voltage position corresponding to ICP.
*   **ICV (Incremental Capacity Valley)**: Valley height between the two peaks of the IC curve.
*   **ICVL_V (V)**: Voltage position corresponding to ICV.
*   **ICP_Area (Ah/V * V = Ah)**: Integrated area under the IC peak. Typically corresponds to active lithium content of specific electrochemical reactions.
*   **ICP_FWHM (V)**: Full Width at Half Maximum of the IC peak. Reflects uniformity of phase transitions; aging leads to peak broadening.
*   **DVP (Differential Voltage Peak)**: Peak height of the DV curve ($dV/dQ$).
*   **DVPL_V (V)**: Voltage position corresponding to DVP.
*   **DVP_Q (Ah)**: Capacity position corresponding to DVP.
*   **DVV (Differential Voltage Valley)**: Valley height of the DV curve.
*   **DVVL_V (V)**: Voltage position corresponding to DVV.
*   **DVV_Q (Ah)**: Capacity position corresponding to DVV.
*   **centroid_voltage (V)**: Centroid voltage of the IC curve.
    *   *Formula*: $\frac{\sum V \cdot IC}{\sum IC}$
*   **Ratio_Peak1_Peak3**: Height ratio between different IC peaks (e.g., main peak and secondary peak), reflecting relative degradation rates of different cathode phase transitions.
*   **V_diff_Peak3_Peak1 (V)**: Voltage spacing between the two main IC peaks.

## 5. Geometric & Statistical (Geometric and Statistical Features)
Features obtained from statistical analysis of raw voltage and current curves.

*   **skew_V_discharge**: Skewness of the discharge voltage curve. Reflects asymmetry in the voltage decline trend.
*   **var_I_charge / var_I_discharge**: Variance of current. Used to detect stability or fluctuations in constant current phases.
*   **var_V_charge / var_V_discharge**: Variance of voltage.
*   **median_V_discharge (V)**: Median voltage during discharge.
*   **charge_slope_n**: Voltage rise slope ($dV/dt$) of the charge curve over a specific time interval.
*   **discharge_slope_n**: Voltage decline slope of the discharge curve over a specific time interval.

## 6. Metadata & Conditions (Metadata and Conditions)
Records experimental conditions, status flags, and boundary conditions of battery cycling.

*   **Workload_Type**: Charge-discharge sequence flag. Identifies whether the current cycle is charge-first, discharge-first, charge-only, or discharge-only.
    *   *Values*: '0' (Charge First), '1' (Discharge First), '2' (Charge Only), '3' (Discharge Only), '-1' (Unknown).
*   **ICP_is_missing (bool)**: Flag. If True, indicates the automatic peak search algorithm failed to find a valid IC peak (usually due to poor data quality or extreme aging).
*   **dvp_type**: DVP calculation mode flag. Indicates whether standard DV peak or a simplified alternative calculation (e.g., 1/ICP) was used.
*   **peak_mode**: Peak search configuration mode. Corresponds to preset parameter sets for different battery chemistries (e.g., NCM, LFP, ZN-ion).
*   **ICHV (V)**: Initial Charge Voltage. Voltage at the instant charge starts.
*   **IDV (V)**: Initial Discharge Voltage. Voltage at the instant discharge starts.
*   **UVP (V)**: Upper Voltage Protection. Experimentally set charge cutoff voltage.
*   **LVP (V)**: Lower Voltage Protection. Experimentally set discharge cutoff voltage.
*   **soc (%)**: State of Charge. Estimated percentage of remaining charge, typically based on discharge capacity or open-circuit voltage.

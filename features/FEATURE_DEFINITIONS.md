# Battery Feature Definitions

本文档详细解释了特征提取脚本中使用的各类电池特征及其物理含义。特征按能量、动力学、热力学、曲线分析及统计几何分类。

## 1. Energy (能量与容量特征)
此类特征反映电池的宏观储能能力及转换效率，直接关联电池的 SOH (State of Health)。

*   **Discharge_Capacity (Ah)**: 放电容量。单次循环中完全放电释放的总电荷量。
    *   *公式*: $\int |I_{dis}| dt$
*   **Charge_Capacity (Ah)**: 充电容量。单次循环中完全充电输入的总电荷量。
*   **Discharge_Energy (Wh)**: 放电能量。单次循环中释放的总能量。
    *   *公式*: $\int V \cdot |I_{dis}| dt$
*   **Charge_Energy (Wh)**: 充电能量。单次循环中输入的总能量。
*   **Coulombic_Efficiency (%)**: 库伦效率。放电容量与充电容量之比。
    *   *公式*: $Q_{dis} / Q_{chg}$
*   **Energy_Efficiency (%)**: 能量效率。放电能量与充电能量之比。

## 2. Kinetics (动力学特征)
此类特征反映电池内部的反应速率、阻抗及电化学极化状态。

*   **Internal_Resistance (Ohm)**: 直流内阻 (DCIR)。通常通过充放电切换瞬间或静置后的电压跳变计算。
    *   *公式*: $\Delta V / \Delta I$
*   **CV_Current_Tau (s)**: 恒压充电 (CV) 阶段的电流衰减时间常数。反映锂离子扩散速率。
    *   *拟合*: $I(t) = A \cdot e^{-t/\tau} + C$ 中的 $\tau$。
*   **TCCC (s)**: 恒流充电时间 (Time of Constant Current Charge)。
*   **TCVC (s)**: 恒压充电时间 (Time of Constant Voltage Charge)。
*   **RCV**: 恒流时间与恒压时间之比，反映电池极化程度的增加。
    *   *公式*: $TCCC / TCVC$
*   **charge_c_rate / discharge_c_rate**: 充/放电倍率。标准化电流强度 ($I / Q_{nominal}$)。
*   **Rest_Time (s)**: 静置时间。充放电之间的弛豫时间。
*   **total_discharge_time (s)**: 总放电时长。
*   **UVP_time (s)**: 达到过压保护 (Upper Voltage Limit) 前的持续时间。
*   **LVP_time (s)**: 达到欠压保护 (Lower Voltage Limit) 前的持续时间。
*   **V_rest_end (V)**: 静置结束时的电压，近似于开路电压 (OCV)。
*   **charge_time_1 / charge_time_2 ...**: 多阶段充电协议中，各恒流阶段的持续时间。
*   **discharge_time_1 / discharge_time_2 ...**: 多阶段放电协议中，各阶段的持续时间。
*   **charge_time_ratio_1_2**: 第一阶段充电时间与第二阶段充电时间之比，用于捕捉老化过程中的阶段偏移。
*   **TEVI (Time in Every Voltage Interval)**: 充电过程中经过特定电压区间（如 3.0-3.1V）所需的时间。
*   **TEVD (Time in Every Voltage Interval - Discharge)**: 放电过程中经过特定电压区间所需的时间。

## 3. Thermodynamics (热力学特征)
此类特征来源于电池表面温度监测，反映电池的热产出与热管理状态。

*   **DTP (Difference in Temperature Peak)**: 充放电阶段最高温度的差值。
*   **DTPL_V (Voltage at DTP)**: 出现峰值温差时的端电压。
*   **MAT_charge / MAT_discharge (°C)**: 充/放电过程中的最大实际温度 (Maximum Actual Temperature)。
*   **MET_charge / MET_discharge (°C)**: 充/放电过程中的平均评估温度 (Mean Evaluation Temperature)。
*   **MinT_charge / MinT_discharge (°C)**: 充/放电过程中的最低温度。
*   **T_rise_charge / T_rise_discharge (°C)**: 温升。阶段结束温度减去开始温度。
*   **Thermal_Load_charge / Thermal_Load_discharge (°C·s)**: 热负荷。温度随时间的积分，代表累积热暴露。
*   **skew_T_discharge**: 放电过程温度分布的偏度，反映产热的不均匀性。

## 4. Curve (曲线特征: IC/DV)
基于增量容量 (dQ/dV) 和微分电压 (dV/dQ) 曲线提取的特征，是电化学机理分析的核心。

*   **ICP (Incremental Capacity Peak)**: IC 曲线 ($dQ/dV$) 的峰值高度。代表相变最活跃点的反应速率。
*   **ICPL_V (V)**: ICP 对应的电压位置。
*   **ICV (Incremental Capacity Valley)**: IC 曲线两峰之间的谷值高度。
*   **ICVL_V (V)**: ICV 对应的电压位置。
*   **ICP_Area (Ah/V * V = Ah)**: IC 峰下的积分面积。通常对应特定电化学反应的活性锂含量。
*   **ICP_FWHM (V)**: IC 峰的半高宽 (Full Width at Half Maximum)。反映相变的均一性，老化会导致峰宽化。
*   **DVP (Differential Voltage Peak)**: DV 曲线 ($dV/dQ$) 的峰值高度。
*   **DVPL_V (V)**: DVP 对应的电压位置。
*   **DVP_Q (Ah)**: DVP 对应的容量位置。
*   **DVV (Differential Voltage Valley)**: DV 曲线的谷值高度。
*   **DVVL_V (V)**: DVV 对应的电压位置。
*   **DVV_Q (Ah)**: DVV 对应的容量位置。
*   **centroid_voltage (V)**: IC 曲线的质心电压。
    *   *公式*: $\frac{\sum V \cdot IC}{\sum IC}$
*   **Ratio_Peak1_Peak3**: 不同 IC 峰（如主峰与次峰）的高度比，反映不同正极相变的相对衰退速度。
*   **V_diff_Peak3_Peak1 (V)**: 两个主要 IC 峰之间的电压间距。

## 5. Geometric & Statistical (几何与统计特征)
对原始电压、电流曲线进行统计学分析得到的特征。

*   **skew_V_discharge**: 放电电压曲线的偏度 (Skewness)。反映电压下降趋势的不对称性。
*   **var_I_charge / var_I_discharge**: 电流的方差。用于检测恒流阶段的稳定性或波动。
*   **var_V_charge / var_V_discharge**: 电压的方差。
*   **median_V_discharge (V)**: 放电过程的中位电压。
*   **charge_slope_n**: 充电曲线在特定时间段内的电压上升斜率 ($dV/dt$)。
*   **discharge_slope_n**: 放电曲线在特定时间段内的电压下降斜率。

## 6. Metadata & Conditions (元数据与工况)
记录电池循环的实验条件、状态标志及边界条件。

*   **Workload_Type**: 充放电顺序标志。识别当前循环是先充后放、先放后充，还是仅充或仅放。
    *   *取值*: '0' (Charge First), '1' (Discharge First), '2' (Charge Only), '3' (Discharge Only), '-1' (Unknown)。
*   **ICP_is_missing (bool)**: 标志位。如果为 True，表示自动峰值搜索算法未能找到有效的 IC 峰值（通常由于数据质量差或极端老化）。
*   **dvp_type**: DVP 计算模式标志。指示使用的是标准 DV 峰值还是简化的替代计算（如 1/ICP）。
*   **peak_mode**: 峰值搜索配置模式。对应不同的电池化学体系（如 NCM, LFP, ZN-ion）的预设参数集。
*   **ICHV (V)**: 初始充电电压 (Initial Charge Voltage)。充电开始瞬间的电压。
*   **IDV (V)**: 初始放电电压 (Initial Discharge Voltage)。放电开始瞬间的电压。
*   **UVP (V)**: 上限截止电压 (Upper Voltage Protection)。实验设置的充电截止电压。
*   **LVP (V)**: 下限截止电压 (Lower Voltage Protection)。实验设置的放电截止电压。
*   **soc (%)**: 荷电状态 (State of Charge)。通常基于放电容量或开路电压估算出的当前剩余电量百分比。

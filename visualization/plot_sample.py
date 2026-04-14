import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
font = {'family' : 'Arial'}

matplotlib.rcParams['mathtext.fontset'] = 'custom'

matplotlib.rcParams['mathtext.rm'] = 'Arial'

matplotlib.rcParams['mathtext.it'] = 'Arial'

matplotlib.rc('font', **font)
matplotlib.rcParams['pdf.fonttype'] = 42 # make the text editable for Adobe Illustrator
matplotlib.rcParams['ps.fonttype'] = 42
#matplotlib.rc('font', **font0)
def set_ax_linewidth(ax, bw=1.5):
    ax.spines['bottom'].set_linewidth(bw)
    ax.spines['left'].set_linewidth(bw)
    ax.spines['top'].set_linewidth(bw)
    ax.spines['right'].set_linewidth(bw)

def set_ax_font_size(ax, fontsize=10):
    ax.tick_params(axis='y',
                 labelsize=fontsize  # y-axis font size setting
                  )
    ax.tick_params(axis='x',
                 labelsize=fontsize  # x-axis font size setting
                  ) 
    
def resample_charge_discharge_curves(voltages, currents, capacity_in_battery):
    '''
    resample the charge and discharge curves based on the natural records
    :param voltages:charge or dicharge voltages
    :param currents: charge or discharge current
    :param capacity_in_battery: remaining capacities in the battery
    :return:interploted records
    '''
    charge_discharge_len = 300
    charge_discharge_len = charge_discharge_len // 2
    raw_bases = np.arange(1, len(voltages)+1)
    interp_bases = np.linspace(1, len(voltages)+1, num=charge_discharge_len,
                                    endpoint=True)
    interp_voltages = np.interp(interp_bases, raw_bases, voltages)
    interp_currents = np.interp(interp_bases, raw_bases, currents)
    interp_capacity_in_battery = np.interp(interp_bases, raw_bases, capacity_in_battery)
    return interp_voltages, interp_currents, interp_capacity_in_battery


data_path = './dataset/MICH/MICH_BLForm2_pouch_NMC_45C_0-100_1-1C_b.pkl'
data = pickle.load(open(data_path, 'rb'))
is_discharge = False
# data = pickle.load(open('../dataset/HUST/HUST_7-5.pkl', 'rb'))
cycle_data = data['cycle_data']
nominal_capacity = data['nominal_capacity_in_Ah']
need_keys = ['current_in_A', 'voltage_in_V', 'charge_capacity_in_Ah', 'discharge_capacity_in_Ah', 'time_in_s']
file_name = data_path.split('/')[-1]
prefix = file_name.split('_')[0]
if prefix == 'CALB':
    prefix = file_name.split('_')[:2]
    prefix = '_'.join(prefix)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 5))
for correct_cycle_index, sub_cycle_data in enumerate(cycle_data[:100]):
    cycle_df = pd.DataFrame()
    for key in need_keys:
        cycle_df[key] = sub_cycle_data[key]
    cycle_df['cycle_number'] = correct_cycle_index + 1
    cycle_df['cycle_number'] = correct_cycle_index + 1
    cycle_df.loc[cycle_df['charge_capacity_in_Ah'] < 0] = np.nan
    cycle_df.bfill(inplace=True)  # deal with NaN
    voltage_records = cycle_df['voltage_in_V'].values
    current_records = cycle_df['current_in_A'].values
    current_records_in_C = current_records / nominal_capacity
    charge_capacity_records = cycle_df['charge_capacity_in_Ah'].values
    discharge_capacity_records = cycle_df['discharge_capacity_in_Ah'].values

    cutoff_voltage_indices = np.nonzero(
        current_records_in_C >= 0.01)  # This includes constant-voltage charge data, 49th cycle of MATR_b1c18 has some abnormal voltage records
    charge_end_index = cutoff_voltage_indices[0][
        -1]  # after charge_end_index, there are rest after charge, discharge, and rest after discharge data

    cutoff_voltage_indices = np.nonzero(current_records_in_C <= -0.01)
    discharge_end_index = cutoff_voltage_indices[0][-1]

    # tmp_discharge_capacity_records = max(charge_capacity_records) - discharge_capacity_records
    # 'CALB_0', 'CALB_35', 'CALB_45'
    if prefix in ['RWTH', 'OX', 'ZN-coin', 'CALB_0', 'CALB_35', 'CALB_45']:
        # Every cycle first discharge and then charge
        #capacity_in_battery = np.where(charge_capacity_records==0, discharge_capacity_records, charge_capacity_records)
        discharge_voltages = voltage_records[:discharge_end_index]
        discharge_capacities = discharge_capacity_records[:discharge_end_index]
        discharge_currents = current_records[:discharge_end_index]
        
        charge_voltages = voltage_records[discharge_end_index:]
        charge_capacities = charge_capacity_records[discharge_end_index:]
        charge_currents = current_records[discharge_end_index:]
        charge_current_in_C = charge_currents / nominal_capacity
        
        charge_voltages = charge_voltages[np.abs(charge_current_in_C)>0.01]
        charge_capacities = charge_capacities[np.abs(charge_current_in_C)>0.01]
        charge_currents = charge_currents[np.abs(charge_current_in_C)>0.01]
    else:
        # Every cycle first charge and then discharge
        #capacity_in_battery = np.where(np.logical_and(current_records>=-(nominal_capacity*0.01), discharge_capacity_records<=nominal_capacity*0.01), charge_capacity_records, discharge_capacity_records)
        discharge_voltages = voltage_records[charge_end_index:]
        discharge_capacities = discharge_capacity_records[charge_end_index:]
        discharge_currents = current_records[charge_end_index:]
        discharge_current_in_C = discharge_currents / nominal_capacity
        
        discharge_voltages = discharge_voltages[np.abs(discharge_current_in_C)>0.01]
        discharge_capacities = discharge_capacities[np.abs(discharge_current_in_C)>0.01]
        discharge_currents = discharge_currents[np.abs(discharge_current_in_C)>0.01]
        discharge_times = discharge_times[np.abs(discharge_current_in_C)>0.01]
        
        charge_voltages = voltage_records[:charge_end_index]
        charge_capacities = charge_capacity_records[:charge_end_index]
        charge_currents = current_records[:charge_end_index]

    if is_discharge:
        ax1.plot(discharge_capacities, discharge_voltages, marker='o')
        #ax1.plot(discharge_voltages, marker='o')
        discharge_voltages, discharge_currents, discharge_capacities = resample_charge_discharge_curves(discharge_voltages, discharge_currents,
                                                                                    discharge_capacities)
        ax2.plot(discharge_capacities, discharge_voltages, marker='x')
    else:
        ax1.plot(charge_capacities, charge_voltages, marker='o')
        # ax1.plot(charge_voltages, marker='o')
        charge_voltages, charge_currents, charge_capacities = resample_charge_discharge_curves(charge_voltages, charge_currents, charge_capacities)
        ax2.plot(charge_capacities, charge_voltages, marker='x')
    # plt.plot(current_records, marker='o', label='current')
    # plt.plot(cycle_df['charge_capacity_in_Ah'].values, marker='o', label='charge Q')
    # plt.plot(cycle_df['discharge_capacity_in_Ah'].values, marker='o', label='discharge Q')

ax2.set_xlabel('Normalized capacity')
ax2.set_ylabel('Voltage (V)')
ax1.set_xlabel('Capacity (Ah)')
ax1.set_ylabel('Voltage (V)')
fig.tight_layout()
plt.subplots_adjust(wspace=0, hspace=0.3)  # Adjust subplot spacing
plt.savefig('./figures/111.png')
# plt.show()
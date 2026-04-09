import os
import sys
import math
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))  # allow importing from project root
from data_provider.data_split_recorder import split_recorder

# ---------------------------------------------------------------------------
# Dataset split lookup (P1-1/P1-2): single dict replaces the old if-elif chain.
# Keys are (dataset_name, random_seed) for seed-dependent datasets,
# or just dataset_name for the rest. Values are (train, val, test) tuples.
# ---------------------------------------------------------------------------
_SR = split_recorder  # short alias

_SPLITS_FIXED = {
    'CALCE':      (_SR.CALCE_train_files,     _SR.CALCE_val_files,      _SR.CALCE_test_files),
    'HNEI':       (_SR.HNEI_train_files,      _SR.HNEI_val_files,       _SR.HNEI_test_files),
    'HUST':       (_SR.HUST_train_files,      _SR.HUST_val_files,       _SR.HUST_test_files),
    'ISU_ILCC':   (_SR.ISU_ILCC_train_files,  _SR.ISU_ILCC_val_files,   _SR.ISU_ILCC_test_files),
    'MATR':       (_SR.MATR_train_files,      _SR.MATR_val_files,       _SR.MATR_test_files),
    'total_MICH': (_SR.MICH_train_files + _SR.MICH_EXP_train_files,
                   _SR.MICH_val_files  + _SR.MICH_EXP_val_files,
                   _SR.MICH_test_files + _SR.MICH_EXP_test_files),
    'RWTH':       (_SR.RWTH_train_files,      _SR.RWTH_val_files,       _SR.RWTH_test_files),
    'SNL':        (_SR.SNL_train_files,       _SR.SNL_val_files,        _SR.SNL_test_files),
    'Stanford':   (_SR.Stanford_train_files,  _SR.Stanford_val_files,   _SR.Stanford_test_files),
    'Tongji':     (_SR.Tongji_train_files,    _SR.Tongji_val_files,     _SR.Tongji_test_files),
    'XJTU':       (_SR.XJTU_train_files,      _SR.XJTU_val_files,       _SR.XJTU_test_files),
    'UL_PUR':     (_SR.UL_PUR_train_files,    _SR.UL_PUR_val_files,     _SR.UL_PUR_test_files),
}

# Seed-keyed datasets: (dataset_name, seed) -> (train, val, test)
_SPLITS_SEEDED = {
    ('ZN-coin', 2021): (_SR.ZNcoin_train_files,    _SR.ZNcoin_val_files,      _SR.ZNcoin_test_files),
    ('ZN-coin', 2024): (_SR.ZN_2024_train_files,   _SR.ZN_2024_val_files,     _SR.ZN_2024_test_files),
    ('ZN-coin', 42):   (_SR.ZN_42_train_files,     _SR.ZN_42_val_files,       _SR.ZN_42_test_files),
    ('CALB',    2021): (_SR.CALB_train_files,       _SR.CALB_val_files,        _SR.CALB_test_files),
    ('CALB',    2024): (_SR.CALB_2024_train_files,  _SR.CALB_2024_val_files,   _SR.CALB_2024_test_files),
    ('CALB',    42):   (_SR.CALB_42_train_files,    _SR.CALB_42_val_files,     _SR.CALB_42_test_files),
    ('NA-ion',  2021): (_SR.NAion_2021_train_files, _SR.NAion_2021_val_files,  _SR.NAion_2021_test_files),
    ('NA-ion',  2024): (_SR.NAion_2024_train_files, _SR.NAion_2024_val_files,  _SR.NAion_2024_test_files),
    ('NA-ion',  42):   (_SR.NAion_42_train_files,   _SR.NAion_42_val_files,    _SR.NAion_42_test_files),
}

_TYPE_INDEX = {'train': 0, 'vali': 1, 'val': 1, 'test': 2}


def find_dataset(dataset_name, random_seed, type):
    """Return the file list for a given dataset / seed / split.

    Replaces a 130-line if-elif chain; adding a new dataset only requires
    updating data_split_recorder.py and _SPLITS_FIXED / _SPLITS_SEEDED above.
    """
    dataset_name = str(dataset_name)
    random_seed = int(random_seed)
    idx = _TYPE_INDEX.get(type)
    if idx is None:
        raise ValueError(f"Unknown split type '{type}'. Expected one of: train, vali, test.")

    # Try seed-keyed lookup first
    seeded = _SPLITS_SEEDED.get((dataset_name, random_seed))
    if seeded is not None:
        return seeded[idx]

    # Fall back to seed-independent lookup
    fixed = _SPLITS_FIXED.get(dataset_name)
    if fixed is not None:
        return fixed[idx]

    raise ValueError(
        f"No split defined for dataset='{dataset_name}', seed={random_seed}. "
        f"Available fixed datasets: {sorted(_SPLITS_FIXED.keys())}; "
        f"seed-keyed: {sorted(set(k[0] for k in _SPLITS_SEEDED))}"
    )



def cal_loss(dataset_name, random_seed, file, path):
    dataset_name = str(dataset_name)
    random_seed = int(random_seed)

    vali_loss = 0
    test_loss = 0
    vali_sample = 0
    test_sample = 0
    train_vali_avg_list = []
    train_test_avg_list = []
    vali_data = []
    test_data = []
    total_seen_unseen_label = []
    if file.startswith(f'{dataset_name}'):
        if 'ISU-ILCC' in dataset_name:
            dataset_name = 'ISU_ILCC'
        elif 'UL-PUR' in dataset_name:
            dataset_name = 'UL_PUR'

        train_data_names = [i for i in find_dataset(dataset_name, random_seed, type='train')]
        vali_data_names = [i for i in find_dataset(dataset_name, random_seed, type='vali')]
        test_data_names = [i for i in find_dataset(dataset_name, random_seed, type='test')]
        label_data = pd.read_json(os.path.join(path, file), lines=True)

        if random_seed == 42 and dataset_name == 'ZN-coin':
            seen_unseen_label = pd.read_json('./dataset/seen_unseen_labels/cal_for_test_ZN42.json', lines=True)
        elif random_seed == 2024 and dataset_name == 'ZN-coin':
            seen_unseen_label = pd.read_json('./dataset/seen_unseen_labels/cal_for_test_ZN2024.json', lines=True)
        elif random_seed == 42 and dataset_name == 'CALB':
            seen_unseen_label = pd.read_json('./dataset/seen_unseen_labels/cal_for_test_CALB42.json', lines=True)
        elif random_seed == 2024 and dataset_name == 'CALB':
            seen_unseen_label = pd.read_json('./dataset/seen_unseen_labels/cal_for_test_CALB2024.json', lines=True)
        elif random_seed == 42 and dataset_name == 'NA-ion':
            seen_unseen_label = pd.read_json('./dataset/seen_unseen_labels/cal_for_test_NA42.json', lines=True)
        elif random_seed == 2021 and dataset_name == 'NA-ion':
            seen_unseen_label = pd.read_json('./dataset/seen_unseen_labels/cal_for_test_NA2021.json', lines=True)
        elif random_seed == 2024 and dataset_name == 'NA-ion':
            seen_unseen_label = pd.read_json('./dataset/seen_unseen_labels/cal_for_test_NA2024.json', lines=True)
        else:
            seen_unseen_label = pd.read_json('./dataset/seen_unseen_labels/cal_for_test.json', lines=True)

        train_data = []
        for train_name in train_data_names:
            if 'Tongji' in train_name:
                train_name = train_name.replace('--', '-#')
            
            if train_name not in label_data:
                # print('Missing label for train file: ', train_name)
                continue
            train_data.append(label_data[train_name].values[0])
        for vali_name in vali_data_names:
            if 'Tongji' in vali_name:
                vali_name = vali_name.replace('--', '-#')

            if vali_name not in label_data:
                # print('Missing label for vali file: ', vali_name)
                continue
            vali_data.append(label_data[vali_name].values[0])
        for test_name in test_data_names:
            if 'Tongji' in test_name:
                test_name = test_name.replace('--', '-#')

            if test_name not in label_data:
                # print('Missing label for test file: ', test_name)
                continue

            test_name = test_name.replace('-#', '--')
            label = seen_unseen_label[test_name].values
            if label == 'seen':
                total_seen_unseen_label.append(1)
            else:
                total_seen_unseen_label.append(0)

            test_name = test_name.replace('--', '-#')
            test_data.append(label_data[test_name].values[0])

        train_avg = np.average(train_data, axis=0)
        train_vali_avg_list = [train_avg] * len(vali_data)
        train_test_avg_list = [train_avg] * len(test_data)

        for vali in vali_data:
            vali_loss = vali_loss + abs(vali - train_avg)
        for test in test_data:
            test_loss = test_loss + abs(test - train_avg)
        vali_sample = len(vali_data)
        test_sample = len(test_data)

    return vali_loss, test_loss, vali_sample, test_sample, vali_data, test_data, train_vali_avg_list, train_test_avg_list, total_seen_unseen_label


if __name__ == '__main__' :
    path = './dataset/Life labels/'
    files_path = os.listdir(path)
    files = [i for i in files_path if i.endswith('.json')]

    total_test_loss = 0
    total_vali_loss = 0
    total_vali_sample = 0
    total_test_sample = 0
    total_vali_data = []
    total_test_data = []
    total_train_vali_avg_list = []
    total_train_test_avg_list = []
    total_seen_unseen_labels = []
    seen_alpha_acc1 = 0
    seen_alpha_acc2 = 0
    unseen_alpha_acc1 = 0
    unseen_alpha_acc2 = 0
    target_dataset = sys.argv[1]
    random_seed = sys.argv[2]

    for file in tqdm(files):
        if target_dataset == 'MIX_large':
            if 'ZN-coin' in file:
                continue
            elif 'CALB' in file:
                continue
            elif 'NA-ion' in file:
                continue
            dataset_name = file.split('_')[0]
            if 'total' in file:
                dataset_name = 'total_MICH'
        elif target_dataset == 'MICH' or target_dataset == 'MICH_EXP':
            print('Please use [total_MICH] to evaluate.')
            break
        elif target_dataset == 'total_MICH':
            dataset_name = 'total_MICH'
        elif target_dataset == 'UL_PUR':
            print('UL_PUR has no enough samples to evaluate, try another dataset.')
            break
        elif target_dataset in file:
            dataset_name = file.split('_')[0]
        elif target_dataset == 'ISU-ILCC' or target_dataset == 'ISU_ILCC':
            dataset_name = 'ISU_ILCC'
        elif target_dataset == 'NAion':
            dataset_name = 'NA-ion'
        else:
            continue

        vali_loss, test_loss, vali_sample, test_sample, vali_data, test_data, train_vali_avg_list, train_test_avg_list, total_seen_unseen_label = cal_loss(dataset_name, random_seed, file, path)
        total_test_loss = total_test_loss + test_loss
        total_vali_loss = total_vali_loss + vali_loss
        total_test_sample = total_test_sample + test_sample
        total_vali_sample = total_vali_sample + vali_sample
        total_vali_data = total_vali_data + vali_data
        total_test_data = total_test_data + test_data
        total_train_vali_avg_list = total_train_vali_avg_list + train_vali_avg_list
        total_train_test_avg_list = total_train_test_avg_list + train_test_avg_list
        total_seen_unseen_labels = total_seen_unseen_labels + total_seen_unseen_label

        # print(f'Dataset {dataset_name}: Vali loss: {vali_loss} | Vali Sample: {vali_sample} | Test loss: {test_loss} | Test sample: {test_sample}')
    
    if target_dataset != 'UL_PUR' and target_dataset != 'MICH' and target_dataset != 'MICH_EXP':
        vali_mae = mean_absolute_error(total_vali_data, total_train_vali_avg_list)
        test_mae = mean_absolute_error(total_test_data, total_train_test_avg_list)
        vali_mape = mean_absolute_percentage_error(total_vali_data, total_train_vali_avg_list)
        test_mape = mean_absolute_percentage_error(total_test_data, total_train_test_avg_list)
        vali_mse = np.square(np.subtract(total_vali_data, total_train_vali_avg_list)).mean()
        vali_rmse = math.sqrt(vali_mse)
        test_mse = np.square(np.subtract(total_test_data, total_train_test_avg_list)).mean()
        test_rmse = math.sqrt(test_mse)

        hit_num = 0
        for pred, refer in zip(total_test_data, total_train_test_avg_list):
            relative_error = abs(pred - refer) / refer
            if relative_error <= 0.15:
                hit_num += 1
        alpha_acc1 = hit_num / len(total_train_test_avg_list) * 100

        indices_seen = [index for index, value in enumerate(total_seen_unseen_labels) if value == 1]
        indices_unseen = [index for index, value in enumerate(total_seen_unseen_labels) if value == 0]
        seen_preds = [total_test_data[i] for i in indices_seen]
        unseen_preds = [total_test_data[i] for i in indices_unseen]
        seen_references = [total_train_test_avg_list[i] for i in indices_seen]
        unseen_references = [total_train_test_avg_list[i] for i in indices_unseen]
        # seen
        hit_num = 0
        for pred, refer in zip(seen_preds, seen_references):
            relative_error = abs(pred - refer) / refer
            if relative_error <= 0.15:
                hit_num += 1
        if len(seen_references) == 0:
            seen_alpha_acc1 = 'No seen sample'
        else:
            seen_alpha_acc1 = hit_num / len(seen_references) * 100


        hit_num = 0
        for pred, refer in zip(seen_preds, seen_references):
            relative_error = abs(pred - refer) / refer
            if relative_error <= 0.1:
                hit_num += 1
        if len(seen_references) == 0:
            seen_alpha_acc2 = 'No seen sample'
        else:
            seen_alpha_acc2 = hit_num / len(seen_references) * 100


        # unseen
        hit_num = 0
        for pred, refer in zip(unseen_preds, unseen_references):
            relative_error = abs(pred - refer) / refer
            if relative_error <= 0.15:
                hit_num += 1
        if len(unseen_references) == 0:
            unseen_alpha_acc1 = 'No unseen sample'
        else:
            unseen_alpha_acc1 = hit_num / len(unseen_references) * 100

        hit_num = 0
        for pred, refer in zip(unseen_preds, unseen_references):
            relative_error = abs(pred - refer) / refer
            if relative_error <= 0.1:
                hit_num += 1
        if len(unseen_references) == 0:
            unseen_alpha_acc2 = 'No unseen sample'
        else:
            unseen_alpha_acc2 = hit_num / len(unseen_references) * 100

        print(f'Dummy Model : \nTest MAE: {test_mae} | Test MAPE: {test_mape} | Test RMSE: {test_rmse} | Vali MAE: {vali_mae} | Vali MAPE: {vali_mape} | Vali RMSE: {vali_rmse} | 0.15_acc: {alpha_acc1} | 0.15_acc_seen: {seen_alpha_acc1} | 0.1_acc_seen {seen_alpha_acc2} | 0.15_acc_unseen: {unseen_alpha_acc1} | 0.1_acc_unseen {unseen_alpha_acc2}')

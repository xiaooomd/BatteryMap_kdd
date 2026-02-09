from data_provider.data_loader import Dataset_original, Dataset_CustomFeatures
from data_provider.data_loader import my_collate_fn_baseline, my_collate_fn_withId
from torch.utils.data import DataLoader, RandomSampler, Dataset
import numpy as np
import random
import torch

data_dict = {
    'Dataset_original': Dataset_original,
    'Dataset_CustomFeatures': Dataset_CustomFeatures
}

def worker_init_fn(worker_id):
    """
    Initialize each DataLoader worker with a unique random seed.
    This provides additional randomness for multi-GPU training,
    helping to balance load across GPUs.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def data_provider_baseline_DA(args, flag, tokenizer=None, label_scaler=None, eval_cycle_min=None, eval_cycle_max=None, total_prompts=None,
                 total_charge_discharge_curves=None, total_curve_attn_masks=None, total_labels=None, unique_labels=None,
                 class_labels=None, life_class_scaler=None, sample_weighted=False, target_dataset='None', feature_scaler=None):
    if args.feature_type == 'extracted_features':
        Data = Dataset_CustomFeatures
    else:
        Data = data_dict[args.data]

    if flag == 'test' or flag == 'val':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size

    if flag == 'test' or flag == 'val':
        data_set = Data(args=args,
                flag=flag,
                tokenizer=tokenizer,
                label_scaler=label_scaler,
                eval_cycle_min=eval_cycle_min,
                eval_cycle_max=eval_cycle_max,
                total_prompts=total_prompts,
                total_charge_discharge_curves=total_charge_discharge_curves,
                total_curve_attn_masks=total_curve_attn_masks, total_labels=total_labels, unique_labels=unique_labels,
                class_labels=class_labels,
                life_class_scaler=life_class_scaler,
                use_target_dataset=True
            )
    else:
        data_set = Data(args=args,
                flag=flag,
                tokenizer=tokenizer,
                label_scaler=label_scaler,
                eval_cycle_min=eval_cycle_min,
                eval_cycle_max=eval_cycle_max,
                total_prompts=total_prompts,
                total_charge_discharge_curves=total_charge_discharge_curves,
                total_curve_attn_masks=total_curve_attn_masks, total_labels=total_labels, unique_labels=unique_labels,
                class_labels=class_labels,
                life_class_scaler=life_class_scaler,
                use_target_dataset=False
            )

    data_loader = DataLoader(
                data_set,
                batch_size=batch_size,
                shuffle=shuffle_flag,
                num_workers=args.num_workers,
                drop_last=drop_last,
                collate_fn=my_collate_fn_baseline,
                pin_memory=True,
                persistent_workers=True if args.num_workers > 0 else False,
                worker_init_fn=worker_init_fn if args.num_workers > 0 else None)

    if target_dataset != 'None' and flag=='train':
        target_data_set = Data(args=args,
                flag=flag,
                tokenizer=tokenizer,
                label_scaler=data_set.return_label_scaler(),
                eval_cycle_min=eval_cycle_min,
                eval_cycle_max=eval_cycle_max,
                total_prompts=total_prompts,
                total_charge_discharge_curves=total_charge_discharge_curves,
                total_curve_attn_masks=total_curve_attn_masks, total_labels=total_labels, unique_labels=unique_labels,
                class_labels=class_labels,
                life_class_scaler=data_set.return_life_class_scaler(),
                use_target_dataset=True
            )

        target_data_loader = DataLoader(
                    target_data_set,
                    batch_size=batch_size,
                    shuffle=shuffle_flag,
                    num_workers=args.num_workers,
                    drop_last=drop_last,
                    collate_fn=my_collate_fn_baseline,
                    pin_memory=True,
                    persistent_workers=True if args.num_workers > 0 else False,
                    worker_init_fn=worker_init_fn if args.num_workers > 0 else None)
        target_sampler = RandomSampler(target_data_loader.dataset, replacement=True, num_samples=len(data_loader.dataset))
        target_resampled_dataloader = DataLoader(target_data_loader.dataset, batch_size=batch_size, sampler=target_sampler, collate_fn=my_collate_fn_baseline)
        return data_set, data_loader, target_data_set, target_resampled_dataloader
    else:
        return data_set, data_loader

def data_provider_baseline(args, flag, tokenizer=None, label_scaler=None, eval_cycle_min=None, eval_cycle_max=None, total_prompts=None,
                 total_charge_discharge_curves=None, total_curve_attn_masks=None, total_labels=None, unique_labels=None,
                 class_labels=None, life_class_scaler=None, sample_weighted=False, feature_scaler=None):

    if args.feature_type == 'extracted_features':
        Data = Dataset_CustomFeatures
    else:
        Data = data_dict[args.data]

    if flag == 'test' or flag == 'val':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
    else:
        shuffle_flag = True
        drop_last = True

        # CRITICAL FIX: For small datasets in extracted_features mode, drop_last=True kills the only batch
        if args.feature_type == 'extracted_features':
            drop_last = False

        batch_size = args.batch_size

    if args.feature_type == 'extracted_features':
        # New Dataset with feature_scaler support
        data_set = Data(args=args,
            flag=flag,
            feature_scaler=feature_scaler,
            label_scaler=label_scaler
        )
    else:
        # Original Dataset
        data_set = Data(args=args,
            flag=flag,
            tokenizer=tokenizer,
            label_scaler=label_scaler,
            eval_cycle_min=eval_cycle_min,
            eval_cycle_max=eval_cycle_max,
            total_prompts=total_prompts,
            total_charge_discharge_curves=total_charge_discharge_curves,
            total_curve_attn_masks=total_curve_attn_masks, total_labels=total_labels, unique_labels=unique_labels,
            class_labels=class_labels,
            life_class_scaler=life_class_scaler
        )

    data_loader = DataLoader(
                data_set,
                batch_size=batch_size,
                shuffle=shuffle_flag,
                num_workers=args.num_workers,
                drop_last=drop_last,
                collate_fn=my_collate_fn_baseline,
                worker_init_fn=worker_init_fn if args.num_workers > 0 else None)

    return data_set, data_loader


def data_provider_evaluate(args, flag, tokenizer=None, label_scaler=None, eval_cycle_min=None, eval_cycle_max=None, total_prompts=None,
                 total_charge_discharge_curves=None, total_curve_attn_masks=None, total_labels=None, unique_labels=None,
                 class_labels=None, life_class_scaler=None, sample_weighted=False):
    Data = data_dict[args.data]

    if flag == 'test' or flag == 'val':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size

    data_set = Data(args=args,
            flag=flag,
            tokenizer=tokenizer,
            label_scaler=label_scaler,
            eval_cycle_min=eval_cycle_min,
            eval_cycle_max=eval_cycle_max,
            total_prompts=total_prompts,
            total_charge_discharge_curves=total_charge_discharge_curves,
            total_curve_attn_masks=total_curve_attn_masks, total_labels=total_labels, unique_labels=unique_labels,
            class_labels=class_labels,
            life_class_scaler=life_class_scaler
        )


    data_loader = DataLoader(
                data_set,
                batch_size=batch_size,
                shuffle=shuffle_flag,
                num_workers=args.num_workers,
                drop_last=drop_last,
                collate_fn=my_collate_fn_withId,
                worker_init_fn=worker_init_fn if args.num_workers > 0 else None)
    return data_set, data_loader

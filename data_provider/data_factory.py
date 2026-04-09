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

def data_provider(args, flag, use_da=False, use_id=False, **kwargs):
    """
    Unified data provider (P1-3). Consolidates baseline, DA, and evaluate providers.
    Args:
        args: configuration object
        flag: 'train', 'val', or 'test'
        use_da: whether to return Domain Adaptation (DA) resampled target loaders
        use_id: whether to use my_collate_fn_withId (returns indices)
        **kwargs: passed to Dataset constructor (tokenizer, label_scaler, etc.)
    """
    # 1. Determine local variables from flag
    is_eval = (flag in ['test', 'val'])
    shuffle_flag = not is_eval
    drop_last = not is_eval
    batch_size = args.batch_size
    
    # Special fix for small datasets in extracted features mode (from old baseline)
    if not is_eval and args.feature_type == 'extracted_features':
        drop_last = False

    # 2. Select Dataset class
    if args.feature_type == 'extracted_features':
        Data = Dataset_CustomFeatures
    else:
        Data = data_dict.get(args.data, Dataset_original)

    # 3. Create main dataset
    # We pass use_target_dataset=True if we are in eval mode (consistent with old DA provider)
    # and if kwargs didn't specify it.
    if 'use_target_dataset' not in kwargs:
        kwargs['use_target_dataset'] = is_eval

    data_set = Data(args=args, flag=flag, **kwargs)

    # 4. Create main DataLoader
    collate_fn = my_collate_fn_withId if use_id else my_collate_fn_baseline
    
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False,
        worker_init_fn=worker_init_fn if args.num_workers > 0 else None
    )

    # 5. Handle Domain Adaptation (DA)
    target_dataset_val = kwargs.get('target_dataset', 'None')
    if use_da and target_dataset_val != 'None' and flag == 'train':
        target_kwargs = kwargs.copy()
        target_kwargs['use_target_dataset'] = True
        target_kwargs['label_scaler'] = data_set.return_label_scaler()
        target_kwargs['life_class_scaler'] = data_set.return_life_class_scaler()

        target_data_set = Data(args=args, flag=flag, **target_kwargs)
        target_data_loader = DataLoader(
            target_data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            collate_fn=collate_fn,
            pin_memory=True,
            persistent_workers=True if args.num_workers > 0 else False,
            worker_init_fn=worker_init_fn if args.num_workers > 0 else None
        )
        
        target_sampler = RandomSampler(target_data_set, replacement=True, num_samples=len(data_set))
        target_resampled_dataloader = DataLoader(
            target_data_set, 
            batch_size=batch_size, 
            sampler=target_sampler, 
            collate_fn=collate_fn
        )
        return data_set, data_loader, target_data_set, target_resampled_dataloader

    return data_set, data_loader


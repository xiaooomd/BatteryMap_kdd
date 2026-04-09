import numpy as np
import torch
import matplotlib.pyplot as plt
import shutil
import random
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm
import evaluate
import torch.nn.functional as F
from scipy.signal.windows import gaussian
from sklearn.metrics import root_mean_squared_error, mean_absolute_percentage_error
import time
from torch import nn
import os
import pandas as pd

plt.switch_backend('agg')


def set_seed(seed):
    """Set random seed for reproducibility across all libraries."""
    try:
        import accelerate
        accelerate.utils.set_seed(seed)
    except (ImportError, AttributeError):
        pass
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def check_csv_integrity(file_path, early_cycle_threshold=100):
    """
    Checks if a CSV file is valid for feature extraction.
    Args:
        file_path: Path to the CSV file.
        early_cycle_threshold: Minimum number of rows required (optional warning).
    Returns:
        bool: True if valid, False otherwise.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return False

    try:
        # Check if file is empty
        if os.path.getsize(file_path) == 0:
            print(f"Error: Empty file: {file_path}")
            return False

        # Read header and a few lines to check structure
        df = pd.read_csv(file_path, nrows=early_cycle_threshold + 10)

        # Check for numeric data
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            print(f"Error: No numeric columns found in {file_path}")
            return False

        # Check row count (optional warning)
        if len(df) < early_cycle_threshold:
             # This is not strictly an error as we pad, but good to know
             # print(f"Warning: File {file_path} has fewer rows ({len(df)}) than threshold ({early_cycle_threshold}). Padding will be used.")
             pass

        return True

    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return False

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num, 'Percent': trainable_num/total_num}

class Augment_time_series_family(object):
    '''
    This is a set of augmentation for methods for time series
    '''
    def __init__(self, n_holes, mean=0, std=0.02):
        pass

class Downsample_Expand_aug(object):
    '''
    '''
    def __init__(self, rate=0.1):
        pass

class Masking_aug(object):
    def __init__(self, drop_rate):
        self.drop_rate = drop_rate

    def __call__(self, seq):
        '''
        Params:
            seq: Tensor sequence of size (B, num_var, L)
        '''
        seq = F.dropout(seq, self.drop_rate)
        return seq


def sample_top_p(probs, p):
    """
    Perform top-p (nucleus) sampling on a probability distribution.

    Args:
        probs (torch.Tensor): Probability distribution tensor.
        p (float): Probability threshold for top-p sampling.

    Returns:
        torch.Tensor: Sampled token indices.

    Note:
        Top-p sampling selects the smallest set of tokens whose cumulative probability mass
        exceeds the threshold p. The distribution is renormalized based on the selected tokens.

    """
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token

def adjust_learning_rate(accelerator, optimizer, scheduler, epoch, args, printout=True):
    if args.lradj == 'type1':
        # lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
        if epoch > args.least_epochs:
            lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - args.least_epochs) // 1))}
        else:
            lr_adjust = {epoch: args.learning_rate}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == 'PEMS':
        lr_adjust = {epoch: args.learning_rate * (0.95 ** (epoch // 1))}
    elif args.lradj == 'TST':
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}
    elif args.lradj == 'constant':
        lr_adjust = {epoch: args.learning_rate}

    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if printout:
            if accelerator is not None:
                accelerator.print(f'{args.lradj}| Updating learning rate to {lr}')
            else:
                print(f'{args.lradj}| Updating learning rate to {lr}')


class EarlyStopping:
    def __init__(self, accelerator=None, patience=7, verbose=False, delta=0, save_mode=True, least_epochs=5):
        self.accelerator = accelerator
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_vali_mae = None
        self.best_test_mae = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.save_mode = save_mode
        self.least_epochs = least_epochs

    def __call__(self, epoch, val_loss, vali_mae_loss, test_mae_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            if self.save_mode:
                self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            if epoch > self.least_epochs:
                # the early stopping won't count before some epoches are trained
                self.counter += 1
            if self.accelerator is None:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            else:
                self.accelerator.print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            # self.best_vali_mae = vali_mae_loss
            # self.best_test_mae = test_mae_loss
            if self.save_mode:
                self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            if self.accelerator is not None:
                self.accelerator.print(
                    f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            else:
                print(
                    f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        if self.accelerator is not None:

            # model = self.accelerator.unwrap_model(model)
            # self.accelerator.save(model.state_dict(), path + '/' + 'checkpoint')
            # self.accelerator.wait_for_everyone()
            self.accelerator.save_model(model, path)
            #self.accelerator.save(model, path + '/' + 'checkpoint.pth')
            self.accelerator.print(f'The checkpoint is saved in {path}!')
            # self.accelerator.save_state(path + '/')
        else:
            #torch.save(model, path + '/' + 'checkpoint.pth')
            torch.save(model.state_dict(), path + '/' + 'checkpoint')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean=0, std=1):
        self.mean = mean
        self.std = std

    def fit(self, data):
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)
        self.std = np.where(self.std == 0, 1.0, self.std) # Avoid division by zero

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean

def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)


def del_files(dir_path):
    shutil.rmtree(dir_path, ignore_errors=True)

def vali_baseline(args, accelerator, model, vali_data, vali_loader, criterion, compute_seen_unseen=False):
    total_preds, total_references = [], []
    total_seen_unseen_ids = []
    model.eval()
    with torch.no_grad():
        for i, (cycle_curve_data, curve_attn_mask,  labels, life_class, scaled_life_class, weights, seen_unseen_ids) in tqdm(enumerate(vali_loader)):
            cycle_curve_data = cycle_curve_data.float().to(accelerator.device)# [B, S, N]
            curve_attn_mask = curve_attn_mask.float().to(accelerator.device)
            labels = labels.float().to(accelerator.device)

            # encoder - decoder
            # Mirror training logic: Transformer-family needs decoder inputs in extracted_features mode
            if args.feature_type == 'extracted_features' and args.model in ['Transformer', 'Informer', 'Autoformer', 'Reformer']:
                batch_size, seq_len, _ = cycle_curve_data.shape
                pred_len = args.pred_len
                x_mark_enc = torch.zeros(batch_size, seq_len, 4).float().to(accelerator.device)
                x_mark_dec = torch.zeros(batch_size, pred_len, 4).float().to(accelerator.device)
                x_dec = torch.zeros(batch_size, pred_len, args.dec_in).float().to(accelerator.device)
                outputs = model(cycle_curve_data, x_mark_enc, x_dec, x_mark_dec)
            else:
                outputs = model(cycle_curve_data, curve_attn_mask)

            # Handling scalar output for early_prediction
            if args.task_type == 'early_prediction':
                if outputs.ndim == 2 and outputs.shape[1] == 1:
                    outputs = outputs.squeeze()
                if labels.ndim == 2 and labels.shape[1] == 1:
                    labels = labels.squeeze()

            # self.accelerator.wait_for_everyone()

            # Inverse Transform Logic
            # Check if label_scaler is simple mean/std (list/array) or object
            # Original code accessed .var_[-1] which implies sklearn scaler or similar structure
            # Our custom StandardScaler mimics basic behavior but let's be safe

            if hasattr(vali_data.label_scaler, 'var_'):
                 std = np.sqrt(vali_data.label_scaler.var_[-1])
                 mean_value = vali_data.label_scaler.mean_[-1]
            elif hasattr(vali_data.label_scaler, 'std'):
                 # Custom StandardScaler or sklearn fitted
                 # If it's 1D array (scalar regression), take the scalar
                 if np.size(vali_data.label_scaler.std) == 1:
                     std = vali_data.label_scaler.std
                     mean_value = vali_data.label_scaler.mean
                 else:
                     # Multidimensional, take last? Or assuming scalar label is last?
                     # For early prediction, label is [N, 1]. std is [1].
                     std = vali_data.label_scaler.std[-1] if np.size(vali_data.label_scaler.std) > 1 else float(vali_data.label_scaler.std)
                     mean_value = vali_data.label_scaler.mean[-1] if np.size(vali_data.label_scaler.mean) > 1 else float(vali_data.label_scaler.mean)
            else:
                 std = 1.0
                 mean_value = 0.0

            transformed_preds = outputs * std + mean_value
            transformed_labels = labels * std + mean_value

            all_predictions, all_targets, seen_unseen_ids = accelerator.gather_for_metrics((transformed_preds, transformed_labels, seen_unseen_ids))


            total_preds = total_preds + all_predictions.detach().cpu().numpy().reshape(-1).tolist()
            total_references = total_references + all_targets.detach().cpu().numpy().reshape(-1).tolist()
            if compute_seen_unseen:
                total_seen_unseen_ids = total_seen_unseen_ids + seen_unseen_ids.detach().cpu().numpy().reshape(-1).tolist()

    total_preds = np.array(total_preds)
    total_references = np.array(total_references)
    total_seen_unseen_ids = np.array(total_seen_unseen_ids)

    # Error handling for empty predictions
    if len(total_preds) == 0:
        return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

    rmse = root_mean_squared_error(total_references, total_preds)
    mae = mean_absolute_error(total_references, total_preds)
    mape = mean_absolute_percentage_error(total_references, total_preds)

    relative_error = abs(total_preds - total_references) / (total_references + 1e-5) # Avoid div by zero
    hit_num = sum(relative_error<=args.alpha1)
    alpha_acc1 = hit_num / len(total_references) * 100

    relative_error = abs(total_preds - total_references) / (total_references + 1e-5)
    hit_num = sum(relative_error<=args.alpha2)
    alpha_acc2 = hit_num / len(total_references) * 100

    if compute_seen_unseen:
        # calculate the model performance on the samples from the seen and unseen aging conditions
        seen_references = total_references[total_seen_unseen_ids==1] if np.any(total_seen_unseen_ids==1) else np.array([0])
        unseen_references = total_references[total_seen_unseen_ids==0] if np.any(total_seen_unseen_ids==0) else np.array([0])
        seen_preds = total_preds[total_seen_unseen_ids==1] if np.any(total_seen_unseen_ids==1) else np.array([1])
        unseen_preds = total_preds[total_seen_unseen_ids==0] if np.any(total_seen_unseen_ids==0) else np.array([1])

        # MAPE
        seen_mape = mean_absolute_percentage_error(seen_references, seen_preds)
        if len(unseen_preds) > 0 and len(unseen_preds) == len(unseen_references):
             # Ensure unseen_references is not empty or zero-sum if possible, but MAPE handles it via sklearn usually or we added epsilon
             unseen_mape = mean_absolute_percentage_error(unseen_references, unseen_preds)
        else:
            unseen_mape = -10000

        # alpha-acc1
        relative_error = abs(seen_preds - seen_references) / (seen_references + 1e-5)
        hit_num = sum(relative_error<=args.alpha1)
        seen_alpha_acc1 = hit_num / len(seen_references) * 100


        if len(unseen_preds) > 0:
            relative_error = abs(unseen_preds - unseen_references) / (unseen_references + 1e-5)
            hit_num = sum(relative_error<=args.alpha1)
            unseen_alpha_acc1 = hit_num / len(unseen_references) * 100
        else:
            unseen_alpha_acc1 = -10000

        # alpha-acc2
        relative_error = abs(seen_preds - seen_references) / (seen_references + 1e-5)
        hit_num = sum(relative_error<=args.alpha2)
        seen_alpha_acc2 = hit_num / len(seen_references) * 100

        if len(unseen_preds) > 0:
            relative_error = abs(unseen_preds - unseen_references) / (unseen_references + 1e-5)
            hit_num = sum(relative_error<=args.alpha2)
            unseen_alpha_acc2 = hit_num / len(unseen_references) * 100
        else:
            unseen_alpha_acc2 = -10000

        model.train()
        return  rmse, mae, mape, alpha_acc1, alpha_acc2, unseen_mape, seen_mape, unseen_alpha_acc1, seen_alpha_acc1, unseen_alpha_acc2, seen_alpha_acc2

    model.train()
    return rmse, mae, mape, alpha_acc1, alpha_acc2


def load_content(args):
    if 'ETT' in args.data:
        file = 'ETT'
    else:
        file = args.data
    with open('./dataset/prompt_bank/{0}.txt'.format(file), 'r') as f:
        content = f.read()
    return content
import os

# RTX 4000 series compatibility
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

import argparse
import torch
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
import json
import numpy as np
import pandas as pd
import datetime
from modules.predictor.trainer import BatteryLifeTrainer
from utils.tools import set_seed

CAPACITY_LAST_COLUMN_MODELS = {'DACNet', 'DegradeBEATS', 'TrendSpec'}

# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["CUDA_VISIBLE_DEVICES"] = '4,5,6,7'

parser = argparse.ArgumentParser(description='BatteryLife')

# basic config
parser.add_argument('--task_name', type=str, required=False, default='long_term_forecast',
                    help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
parser.add_argument('--is_training', type=int, required=False, default=1, help='status')
parser.add_argument('--model_id', type=str, required=False, default='test', help='model id')
parser.add_argument('--model_comment', type=str, required=False, default='none', help='prefix when saving test results')
parser.add_argument('--model', type=str, required=False, default='Autoformer',
                    help='model name, options: [Autoformer, DLinear]')
parser.add_argument('--seed', type=int, default=2021, help='random seed')

# data loader
parser.add_argument('--charge_discharge_length', type=int, default=100, help='The resampled length for charge and discharge curves')
parser.add_argument('--dataset', type=str, default='HUST', help='dataset used for pretrained model')
parser.add_argument('--data', type=str, required=False, default='BatteryLife', help='dataset type')
parser.add_argument('--root_path', type=str, default='./dataset/HUST_dataset/', help='root path of the data file')
parser.add_argument('--label_dir', type=str, default=None, help='optional directory containing dataset label JSON files')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--task_type', type=str, default='forecasting', help='task type, options:[forecasting, early_prediction]')
parser.add_argument('--feature_type', type=str, default='curve', help='feature type, options:[curve, extracted_features]')
parser.add_argument('--n_selected', type=int, default=-1, help='number of selected features from csv, -1 for all')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; '
                         'M:multivariate predict multivariate, S: univariate predict univariate, '
                         'MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--loader', type=str, default='modal', help='dataset type')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, '
                         'options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], '
                         'you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# forecasting task
parser.add_argument('--early_cycle_threshold', type=int, default=100, help='when to stop model training')
parser.add_argument('--seq_len', type=int, default=5, help='input sequence length')
parser.add_argument('--pred_len', type=int, default=5, help='prediction sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')

# model define
parser.add_argument('--enc_in', type=int, default=1, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=1, help='decoder input size')
parser.add_argument('--c_out', type=int, default=1, help='output size')
parser.add_argument('--d_model', type=int, default=16, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=4, help='num of heads')
parser.add_argument('--lstm_layers', type=int, default=1, help='num of LSTM layers')
parser.add_argument('--e_layers', type=int, default=2, help='num of intra-cycle layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of inter-cycle layers')
parser.add_argument('--d_ff', type=int, default=32, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='relu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
parser.add_argument('--patch_len', type=int, default=10, help='patch length')
parser.add_argument('--stride', type=int, default=10, help='stride')
parser.add_argument('--patch_len2', type=int, default=10, help='patch length for inter-cycle patching')
parser.add_argument('--stride2', type=int, default=10, help='stride for inter-cycle patching')
parser.add_argument('--prompt_domain', type=int, default=0, help='')
parser.add_argument('--output_num', type=int, default=1, help='The number of prediction targets')
parser.add_argument('--class_num', type=int, default=8, help='The number of life classes')
# DACNet TAACM hyperparameters
parser.add_argument('--seg_len', type=int, default=12, help='DACNet TAACM segment length')
parser.add_argument('--n_basis', type=int, default=3, help='DACNet TAACM basis mappings')
parser.add_argument('--use_taacm', type=int, default=1, help='DACNet ablation: use TAACM module')
parser.add_argument('--use_mdctb', type=int, default=1, help='DACNet ablation: use MDCTB module')
# DegradeBEATS hyperparameters
parser.add_argument('--encoder_type', type=str, default='cnn', help='DegradeBEATS encoder type')
parser.add_argument('--cnn_channels', type=int, default=16, help='DegradeBEATS CNN channels')
parser.add_argument('--n_mixer_layers', type=int, default=2, help='DegradeBEATS number of mixer layers')
# TrendSpec hyperparameters
parser.add_argument('--top_k', type=int, default=4, help='TrendSpec ULSR top-k frequencies')
parser.add_argument('--w3_min', type=float, default=0.05, help='TrendSpec ED-DPBF minimum w3')
parser.add_argument('--w3_max', type=float, default=5.0, help='TrendSpec ED-DPBF maximum w3')
parser.add_argument('--ridge_lambda', type=float, default=1e-4, help='TrendSpec ED-DPBF ridge regularization lambda')

# optimization
parser.add_argument('--weighted_loss', action='store_true', default=False, help='use weighted loss')
parser.add_argument('--weighted_sampling', action='store_true', default=False, help='use weighted sampling')
parser.add_argument('--num_workers', type=int, default=1, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--least_epochs', type=int, default=5, help='The model is trained at least some epoches before the early stopping is used')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--wd', type=float, default=0.0, help='weight decay')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='MSE', help='loss function, [MSE, BMSE, MAPE]')
parser.add_argument('--lradj', type=str, default='constant', help='adjust learning rate')
parser.add_argument('--pct_start', type=float, default=0.2, help='pct_start')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--percent', type=int, default=100)
parser.add_argument('--accumulation_steps', type=int, default=1, help='gradient accumulation steps')
parser.add_argument('--mlp', type=int, default=0)

# Evaluation alpha-accuracy
parser.add_argument('--alpha1', type=float, default=0.15, help='the 10 percent alpha for alpha-accuracy')
parser.add_argument('--alpha2', type=float, default=0.1, help='the 15 percent alpha for alpha-accuracy')
parser.add_argument('--config', type=str, required=False, help='Path to hyperparameter config JSON file')
parser.add_argument('--swanlab_project', type=str, default='paper4_benchmark_baselines', help='SwanLab project name')
parser.add_argument('--swanlab_experiment_name', type=str, default=None, help='Optional SwanLab run name override')
parser.add_argument('--swanlab_mode', type=str, default=None, help='SwanLab mode, e.g. online/offline/disabled')
parser.add_argument('--disable_swanlab', action='store_true', default=False, help='Disable SwanLab tracking')

if __name__ == '__main__':
    args = parser.parse_args()

    # Load params from config file if provided (overwrites command line defaults)
    if args.config and os.path.exists(args.config):
        print(f"Loading configuration from {args.config}")
        with open(args.config, 'r') as f:
            config_args = json.load(f)
            # Update args with config values, but ONLY if they exist in argparse
            # This prevents overwriting critical system args like 'local_rank' unless explicitly intended
            for key, value in config_args.items():
                if hasattr(args, key):
                    # Convert float to int for integer parameters
                    if isinstance(value, float) and isinstance(getattr(args, key), int):
                        value = int(value)
                    setattr(args, key, value)
                    print(f"  Overwriting {key} = {value}")
                else:
                    print(f"  Warning: Config key '{key}' not found in argparse, ignoring.")

    if args.feature_type == 'extracted_features':
        print(f"DEBUG: n_selected={args.n_selected}")
        uses_capacity_last_column = args.model in CAPACITY_LAST_COLUMN_MODELS
        if args.n_selected > 0:
            args.enc_in = args.n_selected + 1 if uses_capacity_last_column else args.n_selected
        else:
            # Recursively scan for the first CSV to count numeric feature columns.
            if os.path.isdir(args.root_path):
                sample_path = None
                for current_root, _, files in os.walk(args.root_path):
                    csv_files = sorted([f for f in files if f.endswith('.csv')])
                    if csv_files:
                        sample_path = os.path.join(current_root, csv_files[0])
                        break

                if sample_path:
                    # Read just header and first rows to determine numeric columns.
                    try:
                        sample_df = pd.read_csv(sample_path, nrows=5)
                        # For the new models, keep the final numeric column as capacity.
                        # Legacy extracted feature models keep the historical behavior.
                        numeric_cols = sample_df.select_dtypes(include=[np.number]).columns
                        args.enc_in = len(numeric_cols) if uses_capacity_last_column else len(numeric_cols) - 1
                        print(f"Auto-detected enc_in: {args.enc_in} from {sample_path}")
                    except Exception as e:
                        print(f"Error reading CSV for enc_in detection: {e}")
                        # Fallback to default value
                        if args.enc_in is None or args.enc_in <= 0:
                            args.enc_in = 1
                            print(f"Warning: Using default enc_in=1")
                else:
                    print(f"Warning: No CSV files found under {args.root_path} for enc_in detection.")
                    # Fallback to default value
                    if args.enc_in is None or args.enc_in <= 0:
                        args.enc_in = 1
                        print(f"Warning: Using default enc_in=1")
            else:
                print(f"Warning: root_path {args.root_path} is not a directory.")
                # Fallback to default value
                if args.enc_in is None or args.enc_in <= 0:
                    args.enc_in = 1
                    print(f"Warning: Using default enc_in=1")

        if uses_capacity_last_column:
            args.seq_len = args.early_cycle_threshold
            print(f"Aligned seq_len to early_cycle_threshold for {args.model}: {args.seq_len}")

            if args.model == 'DACNet' and args.seq_len % args.seg_len != 0:
                valid_divisors = [d for d in range(1, args.seq_len + 1) if args.seq_len % d == 0]
                fallback_seg_len = max([d for d in valid_divisors if d <= args.seg_len], default=valid_divisors[0])
                print(
                    f"Adjusted seg_len for DACNet extracted_features mode: "
                    f"{args.seg_len} -> {fallback_seg_len}"
                )
                args.seg_len = fallback_seg_len

        # Sync dec_in and c_out with enc_in for feature tasks to prevent mismatch
        args.dec_in = args.enc_in
        args.c_out = args.enc_in

    # Sync task_name with task_type for model compatibility
    if args.task_type == 'early_prediction':
        args.task_name = 'early_prediction'

    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    # Timestamp for folder name (filesystem safe)
    folder_timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

    set_seed(args.seed)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    # deepspeed_plugin = DeepSpeedPlugin(hf_ds_config='./ds_config_zero2_baseline.json')
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], gradient_accumulation_steps=args.accumulation_steps)
    accelerator.print(args.__dict__)
    BatteryLifeTrainer(args, accelerator, nowtime, folder_timestamp).run()

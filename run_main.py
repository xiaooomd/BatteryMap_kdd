import os

# RTX 4000 series compatibility
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

import argparse
import torch
import accelerate
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from torch import nn, optim
from torch.optim import lr_scheduler
import evaluate
from utils.tools import get_parameter_number
from models import CPGRU, CPLSTM, CPMLP, CPBiGRU, CPBiLSTM, CPTransformer, PatchTST, iTransformer, Transformer, \
    DLinear, Autoformer, MLP, MICN, CNN,  \
    BiLSTM, BiGRU, GRU, LSTM
import wandb
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from data_provider.data_factory import data_provider_baseline
import time
import json
import random
import numpy as np
import pandas as pd
import os
import json
import datetime
import joblib
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error

def list_of_ints(arg):
	return list(map(int, arg.split(',')))
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["CUDA_VISIBLE_DEVICES"] = '4,5,6,7'

from utils.tools import del_files, EarlyStopping, adjust_learning_rate, vali_baseline, load_content
parser = argparse.ArgumentParser(description='BatteryLife')

def set_seed(seed):
    """Set random seed for reproducibility."""
    accelerate.utils.set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    if torch.cuda.is_available() > 0:
        torch.cuda.manual_seed_all(seed)

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
        if args.n_selected > 0:
            args.enc_in = args.n_selected
        else:
            # Scan first CSV to count features
            if os.path.isdir(args.root_path):
                csv_files = [f for f in os.listdir(args.root_path) if f.endswith('.csv')]
                if csv_files:
                    # Read just header and first row to determine numeric columns
                    sample_path = os.path.join(args.root_path, csv_files[0])
                    try:
                        sample_df = pd.read_csv(sample_path, nrows=5)
                        # Assuming label is the last column, we subtract 1 from numeric columns
                        numeric_cols = sample_df.select_dtypes(include=[np.number]).columns
                        args.enc_in = len(numeric_cols) - 1
                        print(f"Auto-detected enc_in: {args.enc_in} from {sample_path}")
                    except Exception as e:
                        print(f"Error reading CSV for enc_in detection: {e}")
                        # Fallback to default value
                        if args.enc_in is None or args.enc_in <= 0:
                            args.enc_in = 1
                            print(f"Warning: Using default enc_in=1")
                else:
                    print(f"Warning: No CSV files found in {args.root_path} for enc_in detection.")
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
    for ii in range(args.itr):
        # Hyper-parameter string (excluding model and dataset)
        hyper_params = 'sl{}_lr{}_dm{}_nh{}_el{}_dl{}_df{}_lradj{}_loss{}_wd{}_wl{}_bs{}_s{}'.format(
            args.seq_len,
            args.learning_rate,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.lradj,
            args.loss,
            args.wd,
            args.weighted_loss,
            args.batch_size,
            args.seed)

        # Construct folder name parts
        folder_parts = [args.model, args.dataset, hyper_params]

        # Add model_comment if it exists and is not 'none'
        if args.model_comment and args.model_comment != 'none':
            folder_parts.append(args.model_comment)

        # Add timestamp
        folder_parts.append(folder_timestamp)

        # Join with underscores
        folder_name = "_".join(folder_parts)

        # Validate critical parameters before model creation
        if args.enc_in is None or args.enc_in <= 0:
            raise ValueError(f"Invalid enc_in value: {args.enc_in}. Must be a positive integer.")
        if args.d_model is None or args.d_model <= 0:
            raise ValueError(f"Invalid d_model value: {args.d_model}. Must be a positive integer.")
        if args.dec_in is None or args.dec_in <= 0:
            raise ValueError(f"Invalid dec_in value: {args.dec_in}. Must be a positive integer.")
        if args.c_out is None or args.c_out <= 0:
            raise ValueError(f"Invalid c_out value: {args.c_out}. Must be a positive integer.")

        print(f"Model parameters: enc_in={args.enc_in}, d_model={args.d_model}, dec_in={args.dec_in}, c_out={args.c_out}")

        data_provider_func = data_provider_baseline
        if args.model == 'Transformer':
            model = Transformer.Model(args).float()
        elif args.model == 'CPBiLSTM':
            model = CPBiLSTM.Model(args).float()
        elif args.model == 'CPBiGRU':
            model = CPBiGRU.Model(args).float()
        elif args.model == 'CPGRU':
            model = CPGRU.Model(args).float()
        elif args.model == 'CPLSTM':
            model = CPLSTM.Model(args).float()
        elif args.model == 'BiLSTM':
            model = BiLSTM.Model(args).float()
        elif args.model == 'BiGRU':
            model = BiGRU.Model(args).float()
        elif args.model == 'LSTM':
            model = LSTM.Model(args).float()
        elif args.model == 'GRU':
            model = GRU.Model(args).float()
        elif args.model == 'PatchTST':
            model = PatchTST.Model(args).float()
        elif args.model == 'iTransformer':
            model = iTransformer.Model(args).float()
        elif args.model == 'DLinear':
            model = DLinear.Model(args).float()
        elif args.model == 'CPMLP':
            model = CPMLP.Model(args).float()
        elif args.model == 'Autoformer':
            model = Autoformer.Model(args).float()
        elif args.model == 'MLP':
            model = MLP.Model(args).float()
        elif args.model == 'MICN':
            model = MICN.Model(args).float()
        elif args.model == 'CNN':
            model = CNN.Model(args).float()
        elif args.model == 'MLP':
            model = MLP.Model(args).float()
        elif args.model == 'MICN':
            model = MICN.Model(args).float()
        elif args.model == 'CNN':
            model = CNN.Model(args).float()
        elif args.model == 'CPTransformer':
            model = CPTransformer.Model(args).float()
        else:
            raise Exception(f'The {args.model} is not an implemented baseline!')


        path = os.path.join(args.checkpoints, folder_name)  # unique checkpoint saving path


        accelerator.print("Loading training samples......")
        train_data, train_loader = data_provider_func(args, 'train', None, sample_weighted=args.weighted_sampling)
        label_scaler = train_data.return_label_scaler()
        life_class_scaler = train_data.return_life_class_scaler()

        # Get feature scaler if available (for Feature Mode)
        feature_scaler = None
        if hasattr(train_data, 'return_feature_scaler'):
            feature_scaler = train_data.return_feature_scaler()

        accelerator.print("Loading vali samples......")
        vali_data, vali_loader = data_provider_func(args, 'val', None, label_scaler, life_class_scaler=life_class_scaler, sample_weighted=args.weighted_sampling, feature_scaler=feature_scaler)
        accelerator.print("Loading test samples......")
        test_data, test_loader = data_provider_func(args, 'test', None, label_scaler, life_class_scaler=life_class_scaler, sample_weighted=args.weighted_sampling, feature_scaler=feature_scaler)

        if accelerator.is_local_main_process and os.path.exists(path):
            del_files(path)  # delete checkpoint files
            accelerator.print(f'success delete {path}')

        os.makedirs(path, exist_ok=True)
        accelerator.wait_for_everyone()
        joblib.dump(label_scaler, f'{path}/label_scaler')
        joblib.dump(life_class_scaler, f'{path}/life_class_scaler')
        with open(path+'/args.json', 'w') as f:
            json.dump(args.__dict__, f)
        
        # Track whether wandb is available
        wandb_available = False
        if accelerator.is_local_main_process:
            try:
                wandb.init(
                    # set the wandb project where this run will be logged
                    project="paper4_benchmark_baselines",

                    # track hyperparameters and run metadata
                    config=args.__dict__,
                    name=nowtime
                )
                wandb_available = True
            except Exception as e:
                accelerator.print(f"WandB initialization failed: {e}. Proceeding without WandB.")
                wandb_available = False


        para_res = get_parameter_number(model)
        accelerator.print(para_res)

        for name, module in model._modules.items():
            accelerator.print(name," : ",module)


        time_now = time.time()

        train_steps = len(train_loader)
        # CRITICAL FIX: Ensure train_steps is at least 1 to prevent scheduler crashes
        if train_steps == 0:
            accelerator.print("WARNING: train_loader is empty. Setting train_steps to 1 to avoid scheduler error.")
            train_steps = 1

        early_stopping = EarlyStopping(accelerator=accelerator, patience=args.patience)

        trained_parameters = []
        trained_parameters_names = []
        for name, p in model.named_parameters():
            if p.requires_grad is True:
                trained_parameters_names.append(name)
                trained_parameters.append(p)

        accelerator.print(f'Trainable parameters are: {trained_parameters_names}')
        model_optim = optim.Adam(trained_parameters, lr=args.learning_rate)

        if args.lradj == 'COS':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=20, eta_min=1e-8)
        else:
            scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                                steps_per_epoch=train_steps,
                                                pct_start=args.pct_start,
                                                epochs=args.train_epochs,
                                                max_lr=args.learning_rate)

        life_classes = json.load(open('data_provider/life_classes.json'))
        class_numbers = len(list(life_classes.keys()))

        criterion = nn.MSELoss(reduction='none')


        life_class_criterion = nn.MSELoss()

        train_loader, vali_loader, test_loader, model, model_optim, scheduler = accelerator.prepare(
            train_loader, vali_loader, test_loader, model, model_optim, scheduler)
        best_vali_loss = float('inf')
        best_vali_MAE, best_test_MAE = 0, 0
        best_vali_RMSE, best_test_RMSE = 0, 0
        best_vali_alpha_acc1, best_test_alpha_acc1 = 0, 0
        best_vali_alpha_acc2, best_test_alpha_acc2 = 0, 0

        best_seen_vali_alpha_acc1, best_seen_test_alpha_acc1 = 0, 0
        best_seen_vali_alpha_acc2, best_seen_test_alpha_acc2 = 0, 0
        best_unseen_vali_alpha_acc1, best_unseen_test_alpha_acc1 = 0, 0
        best_unseen_vali_alpha_acc2, best_unseen_test_alpha_acc2 = 0, 0

        best_vali_MAPE, best_test_MAPE = 0, 0
        best_seen_vali_MAPE, best_seen_test_MAPE = 0, 0
        best_unseen_vali_MAPE, best_unseen_test_MAPE = 0, 0

        for epoch in range(args.train_epochs):
            mae_metric = evaluate.load('./utils/mae')
            mape_metric = evaluate.load('./utils/mape')
            iter_count = 0
            total_loss = 0
            total_cl_loss = 0
            total_lc_loss = 0

            model.train()
            epoch_time = time.time()
            print_cl_loss = 0
            print_life_class_loss = 0
            std, mean_value = np.sqrt(train_data.label_scaler.var_[-1]), train_data.label_scaler.mean_[-1]
            total_preds, total_references = [], []
            for i, (cycle_curve_data, curve_attn_mask,  labels, life_class, scaled_life_class, weights, seen_unseen_ids) in enumerate(train_loader):
                with accelerator.accumulate(model):
                    model_optim.zero_grad()
                    iter_count += 1

                    life_class = life_class.to(accelerator.device)
                    scaled_life_class = scaled_life_class.float().to(accelerator.device)
                    cycle_curve_data = cycle_curve_data.float().to(accelerator.device)
                    curve_attn_mask = curve_attn_mask.float().to(accelerator.device) # [B, L]
                    labels = labels.float().to(accelerator.device)


                    # encoder - decoder
                    if args.feature_type == 'extracted_features' and args.model in ['Transformer', 'Informer', 'Autoformer', 'Reformer']:
                        # Special handling for standard time-series models requiring 4 inputs
                        # x_enc = cycle_curve_data
                        # x_mark_enc = zeros
                        # x_dec = zeros
                        # x_mark_dec = zeros

                        B, L, D = cycle_curve_data.shape
                        pred_len = args.pred_len

                        # Mock time features (marks) as zeros [B, L, 4] (assuming 4 dims for timeF)
                        x_mark_enc = torch.zeros(B, L, 4).float().to(accelerator.device)
                        x_mark_dec = torch.zeros(B, pred_len, 4).float().to(accelerator.device)

                        # Mock decoder input
                        # x_dec = torch.zeros(B, pred_len, D).float().to(accelerator.device)
                        x_dec = torch.zeros(B, pred_len, args.dec_in).float().to(accelerator.device) # Fix dimension to dec_in

                        outputs = model(cycle_curve_data, x_mark_enc, x_dec, x_mark_dec)
                    else:
                        outputs = model(cycle_curve_data, curve_attn_mask)


                    cut_off = labels.shape[0]
                    
                    # DEBUG: Print shapes to understand the issue
                    if (i + 1) % 50 == 0:
                        accelerator.print(f"DEBUG BEFORE SQUEEZE: outputs.shape={outputs.shape}, labels.shape={labels.shape}, args.loss={args.loss}, args.task_type={args.task_type}")

                    if args.task_type == 'early_prediction':
                        # Early Prediction (Scalar Regression)
                        if outputs.ndim == 2 and outputs.shape[1] == 1:
                            outputs = outputs.squeeze()
                        if labels.ndim == 2 and labels.shape[1] == 1:
                            labels = labels.squeeze()

                        loss = criterion(outputs, labels)
                        if args.weighted_loss:
                            # Ensure weights shape matches loss
                            if weights.ndim > 1: weights = weights.squeeze()
                            loss = torch.mean(loss * weights)
                        else:
                            loss = torch.mean(loss)

                    elif args.loss == 'MSE':
                        # Ensure output shape matches label shape for MSE loss
                        if outputs.ndim == 2 and outputs.shape[1] == 1:
                            outputs = outputs.squeeze()
                        if labels.ndim == 2 and labels.shape[1] == 1:
                            labels = labels.squeeze()
                        
                        loss = criterion(outputs[:cut_off], labels)
                        loss = torch.mean(loss * weights)
                    elif args.loss == 'MAPE':
                        tmp_outputs = outputs[:cut_off] * std + mean_value
                        tmp_labels = labels * std + mean_value
                        loss = criterion(tmp_outputs/tmp_labels, tmp_labels/tmp_labels)
                        loss = torch.mean(loss * weights)

                    label_loss = loss.detach().float()


                    print_loss = loss.detach().float()

                    if torch.isnan(loss) or torch.isinf(loss):
                        accelerator.print(f"\n[CRITICAL ERROR] Loss is {loss.item()} at Epoch {epoch+1}, Step {iter_count}")
                        accelerator.print(f"Input  (cycle_curve_data): min={cycle_curve_data.min().item():.4f}, max={cycle_curve_data.max().item():.4f}, mean={cycle_curve_data.mean().item():.4f}")
                        accelerator.print(f"Output (outputs)         : min={outputs.min().item():.4f}, max={outputs.max().item():.4f}, mean={outputs.mean().item():.4f}")
                        accelerator.print(f"Target (labels)          : min={labels.min().item():.4f}, max={labels.max().item():.4f}, mean={labels.mean().item():.4f}")
                        if torch.isnan(cycle_curve_data).any(): accelerator.print("Input contains NaN!")
                        if torch.isinf(cycle_curve_data).any(): accelerator.print("Input contains Inf!")

                    total_loss += loss.detach().float()
                    total_cl_loss += print_cl_loss
                    total_lc_loss += print_life_class_loss

                    # Ensure correct shape before gathering
                    # Make sure both outputs and labels are properly sliced and squeezed
                    transformed_preds = outputs[:cut_off] * std + mean_value
                    transformed_labels = labels[:cut_off] * std + mean_value
                    
                    # Debug: print shapes before any transformation
                    if (i + 1) % 50 == 0:
                        accelerator.print(f"DEBUG: outputs shape: {outputs.shape}, transformed_preds shape: {transformed_preds.shape}")
                    
                    # Ensure 1D tensors - squeeze ALL dimensions
                    transformed_preds = transformed_preds.squeeze(-1)  # Remove last dimension if size 1
                    transformed_labels = transformed_labels.squeeze(-1)
                    
                    # If still multi-dimensional, squeeze again
                    if transformed_preds.ndim > 1:
                        transformed_preds = transformed_preds.squeeze()
                    if transformed_labels.ndim > 1:
                        transformed_labels = transformed_labels.squeeze()
                    
                    # Final reshape to ensure 1D
                    transformed_preds = transformed_preds.reshape(-1)
                    transformed_labels = transformed_labels.reshape(-1)
                    
                    if (i + 1) % 50 == 0:
                        accelerator.print(f"DEBUG: After squeeze - transformed_preds shape: {transformed_preds.shape}, transformed_labels shape: {transformed_labels.shape}")
                    
                    all_predictions, all_targets = accelerator.gather_for_metrics((transformed_preds, transformed_labels))

                    if accelerator.is_main_process:
                        total_preds = total_preds + all_predictions.detach().cpu().numpy().tolist()
                        total_references = total_references + all_targets.detach().cpu().numpy().tolist()
                    accelerator.backward(loss)

                    # Gradient Clipping to prevent explosion (Inf/NaN loss)
                    accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)

                    model_optim.step()
                    if args.lradj == 'TST':
                        adjust_learning_rate(accelerator, model_optim, scheduler, epoch + 1, args, printout=False)
                        scheduler.step()

                    if (i + 1) % 5 == 0:
                        accelerator.print(f'\titeras: {i+1}, epoch: {epoch+1} | loss:{print_loss:.7f} | label_loss: {label_loss:.7f} | cl_loss: {print_cl_loss:.7f} | lc_loss: {print_life_class_loss:.7f}')
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * ((args.train_epochs - epoch) * train_steps - i)
                        accelerator.print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                        iter_count = 0
                        time_now = time.time()

            if accelerator.is_main_process:
                train_rmse = root_mean_squared_error(total_references, total_preds)
                train_mape = mean_absolute_percentage_error(total_references, total_preds)
            else:
                train_rmse = 0.0
                train_mape = 0.0
            accelerator.print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

            vali_rmse, vali_mae_loss, vali_mape, vali_alpha_acc1, vali_alpha_acc2 = vali_baseline(args, accelerator, model, vali_data, vali_loader, criterion, compute_seen_unseen=False)
            test_rmse, test_mae_loss, test_mape, test_alpha_acc1, test_alpha_acc2, test_unseen_mape, test_seen_mape, test_unseen_alpha_acc1, test_seen_alpha_acc1, test_unseen_alpha_acc2, test_seen_alpha_acc2 = vali_baseline(args, accelerator, model, test_data, test_loader, criterion, compute_seen_unseen=True)
            vali_loss = vali_mape


            if vali_loss < best_vali_loss:
                best_vali_loss = vali_loss
                best_vali_MAE = vali_mae_loss
                best_test_MAE = test_mae_loss
                best_vali_RMSE = vali_rmse
                best_test_RMSE = test_rmse
                best_vali_MAPE = vali_mape
                best_test_MAPE = test_mape

                # alpha-accuracy
                best_vali_alpha_acc1 = vali_alpha_acc1
                best_vali_alpha_acc2 = vali_alpha_acc2
                best_test_alpha_acc1 = test_alpha_acc1
                best_test_alpha_acc2 = test_alpha_acc2

                # seen, unseen
                best_seen_test_MAPE = test_seen_mape
                best_unseen_test_MAPE = test_unseen_mape
                best_seen_test_alpha_acc1 = test_seen_alpha_acc1
                best_unseen_test_alpha_acc1 = test_unseen_alpha_acc1
                best_seen_test_alpha_acc2 = test_seen_alpha_acc2
                best_unseen_test_alpha_acc2 = test_unseen_alpha_acc2

            train_loss = total_loss / len(train_loader)
            total_cl_loss = total_cl_loss / len(train_loader)
            total_lc_loss = total_lc_loss / len(train_loader)
            accelerator.print(
                f"Epoch: {epoch+1} | Train Loss: {train_loss:.5f}| Train cl loss: {total_cl_loss:.5f}| Train lc loss: {total_lc_loss:.5f} | Train RMSE: {train_rmse:.7f} | Train MAPE: {train_mape:.7f} | Vali RMSE: {vali_rmse:.7f}| Vali MAE: {vali_mae_loss:.7f}| Vali MAPE: {vali_mape:.7f}| "
                f"Test RMSE: {test_rmse:.7f}| Test MAE: {test_mae_loss:.7f} | Test MAPE: {test_mape:.7f}")

            if accelerator.is_local_main_process and wandb_available:
                try:
                    wandb.log({"epoch": epoch, "train_loss": train_loss, "vali_RMSE": vali_rmse, "vali_MAPE": vali_mape, "vali_acc1": vali_alpha_acc1, "vali_acc2": vali_alpha_acc2,
                            "test_RMSE": test_rmse, "test_MAPE": test_mape, "test_acc1": test_alpha_acc1, "test_acc2": test_alpha_acc2})
                except:
                    pass

            early_stopping(epoch+1, vali_loss, vali_mae_loss, test_mae_loss, model, path)
            if early_stopping.early_stop:
                accelerator.print("Early stopping")
                accelerator.set_trigger()

            if accelerator.check_trigger():
                break

            if args.lradj != 'TST':
                if args.lradj == 'COS':
                    scheduler.step()
                    accelerator.print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
                else:
                    adjust_learning_rate(accelerator, model_optim, scheduler, epoch + 1, args, printout=True)

            else:
                accelerator.print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

    accelerator.print(f'Best model performance: Test MAE: {best_test_MAE:.4f} | Test RMSE: {best_test_RMSE:.4f} | Test MAPE: {best_test_MAPE:.4f} | Test 15%-accuracy: {best_test_alpha_acc1:.4f} | Test 10%-accuracy: {best_test_alpha_acc2:.4f} | Val MAE: {best_vali_MAE:.4f} | Val RMSE: {best_vali_RMSE:.4f} | Val MAPE: {best_vali_MAPE:.4f} | Val 15%-accuracy: {best_vali_alpha_acc1:.4f} | Val 10%-accuracy: {best_vali_alpha_acc2:.4f} ')
    accelerator.print(f'Best model performance: Test Seen MAPE: {best_seen_test_MAPE:.4f} | Test Unseen MAPE: {best_unseen_test_MAPE:.4f}')
    accelerator.print(f'Best model performance: Test Seen 15%-accuracy: {best_seen_test_alpha_acc1:.4f} | Test Unseen 15%-accuracy: {best_unseen_test_alpha_acc1:.4f}')
    accelerator.print(f'Best model performance: Test Seen 10%-accuracy: {best_seen_test_alpha_acc2:.4f} | Test Unseen 10%-accuracy: {best_unseen_test_alpha_acc2:.4f}')
    accelerator.print(path)
    
    # Save best performance metrics to JSON file and CSV
    if accelerator.is_local_main_process:
        best_metrics = {
            "dataset": args.dataset,
            "model": args.model,
            "test_metrics": {
                "MAE": float(best_test_MAE),
                "RMSE": float(best_test_RMSE),
                "MAPE": float(best_test_MAPE),
                "accuracy_15pct": float(best_test_alpha_acc1),
                "accuracy_10pct": float(best_test_alpha_acc2),
            },
            "validation_metrics": {
                "MAE": float(best_vali_MAE),
                "RMSE": float(best_vali_RMSE),
                "MAPE": float(best_vali_MAPE),
                "accuracy_15pct": float(best_vali_alpha_acc1),
                "accuracy_10pct": float(best_vali_alpha_acc2),
            },
            "test_seen_unseen": {
                "seen_MAPE": float(best_seen_test_MAPE),
                "unseen_MAPE": float(best_unseen_test_MAPE),
                "seen_accuracy_15pct": float(best_seen_test_alpha_acc1),
                "unseen_accuracy_15pct": float(best_unseen_test_alpha_acc1),
                "seen_accuracy_10pct": float(best_seen_test_alpha_acc2),
                "unseen_accuracy_10pct": float(best_unseen_test_alpha_acc2),
            },
            "checkpoint_path": path,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Save JSON to checkpoint directory
        with open(os.path.join(path, "best_metrics.json"), "w") as f:
            json.dump(best_metrics, f, indent=4)
        accelerator.print(f"Best metrics saved to {os.path.join(path, 'best_metrics.json')}")
        
        # Save CSV to results directory (organized by model)
        results_dir = os.path.join("./results", args.model)
        os.makedirs(results_dir, exist_ok=True)
        
        csv_data = {
            "dataset": [args.dataset],
            "model": [args.model],
            "timestamp": [datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            "test_MAE": [float(best_test_MAE)],
            "test_RMSE": [float(best_test_RMSE)],
            "test_MAPE": [float(best_test_MAPE)],
            "test_acc_15pct": [float(best_test_alpha_acc1)],
            "test_acc_10pct": [float(best_test_alpha_acc2)],
            "val_MAE": [float(best_vali_MAE)],
            "val_RMSE": [float(best_vali_RMSE)],
            "val_MAPE": [float(best_vali_MAPE)],
            "val_acc_15pct": [float(best_vali_alpha_acc1)],
            "val_acc_10pct": [float(best_vali_alpha_acc2)],
            "test_seen_MAPE": [float(best_seen_test_MAPE)],
            "test_unseen_MAPE": [float(best_unseen_test_MAPE)],
            "test_seen_acc_15pct": [float(best_seen_test_alpha_acc1)],
            "test_unseen_acc_15pct": [float(best_unseen_test_alpha_acc1)],
            "test_seen_acc_10pct": [float(best_seen_test_alpha_acc2)],
            "test_unseen_acc_10pct": [float(best_unseen_test_alpha_acc2)],
            "checkpoint_path": [path]
        }
        
        df = pd.DataFrame(csv_data)
        csv_file = os.path.join(results_dir, f"{args.dataset}_results.csv")
        df.to_csv(csv_file, index=False, float_format='%.6f')
        accelerator.print(f"Results saved to {csv_file}")

    # ==========================================
    # Performance Profiling (Time & FLOPs)
    # ==========================================
    if accelerator.is_local_main_process:
        print("\n" + "="*40)
        print("Performance Profiling")
        print("="*40)

        try:
            # 1. FLOPs / Params (using thop if available)
            from thop import profile

            # Create a dummy input based on feature_type
            model.eval()
            if args.feature_type == 'extracted_features':
                # [1, early_cycle_threshold, enc_in]
                dummy_input = torch.randn(1, args.early_cycle_threshold, args.enc_in).float().to(accelerator.device)
                dummy_mask = torch.ones(1, args.early_cycle_threshold).float().to(accelerator.device)

                # We need to wrap model forward to handle multiple inputs if thop doesn't support kwargs easily
                # thop.profile takes inputs=(t1, t2, ...)
                if args.model in ['Transformer', 'Informer', 'Autoformer']:
                     # These need 4 inputs in this code branch: x_enc, x_mark_enc, x_dec, x_mark_dec
                     # Re-create dummy inputs similar to training loop
                     B, L, D = 1, args.early_cycle_threshold, args.enc_in
                     x_mark_enc = torch.zeros(B, L, 4).float().to(accelerator.device)
                     x_dec = torch.zeros(B, args.pred_len, args.dec_in).float().to(accelerator.device)
                     x_mark_dec = torch.zeros(B, args.pred_len, 4).float().to(accelerator.device)
                     flops, params = profile(model, inputs=(dummy_input, x_mark_enc, x_dec, x_mark_dec), verbose=False)
                else:
                     flops, params = profile(model, inputs=(dummy_input, dummy_mask), verbose=False)

            else:
                # [1, early_cycle, fixed_len, 3] or similar 4D structure
                # This part is complex due to legacy shapes. Skipping FLOPs for legacy mode to be safe unless strictly needed.
                print("FLOPs calculation skipped for legacy curve mode (complex 4D input).")
                flops, params = 0, 0

            print(f"FLOPs: {flops / 1e6:.4f} M")
            print(f"Params: {params / 1e6:.4f} M")

        except ImportError as ie:
            print(f"ImportError: {ie}")
            print("thop library not found. Install via 'pip install thop' to see FLOPs.")
            flops, params = 0, 0
        except Exception as e:
            print(f"FLOPs calculation failed: {type(e).__name__}: {e}")
            flops, params = 0, 0

        # 2. Inference Latency (Batch Size = 1)
        # Warmup
        try:
            model.eval()
            with torch.no_grad():
                if args.feature_type == 'extracted_features':
                    dummy_input = torch.randn(1, args.early_cycle_threshold, args.enc_in).float().to(accelerator.device)
                    dummy_mask = torch.ones(1, args.early_cycle_threshold).float().to(accelerator.device)
                    if args.model in ['Transformer', 'Autoformer']:
                         # Mock inputs again
                         x_mark_enc = torch.zeros(1, args.early_cycle_threshold, 4).float().to(accelerator.device)
                         x_dec = torch.zeros(1, args.pred_len, args.dec_in).float().to(accelerator.device)
                         x_mark_dec = torch.zeros(1, args.pred_len, 4).float().to(accelerator.device)
                         input_args = (dummy_input, x_mark_enc, x_dec, x_mark_dec)
                    else:
                         input_args = (dummy_input, dummy_mask)
                else:
                     # Skip for legacy
                     input_args = None

                if input_args is not None:
                    # Warmup
                    for _ in range(10):
                        _ = model(*input_args)

                    # Timing
                    iterations = 100
                    torch.cuda.synchronize() if torch.cuda.is_available() else None
                    start_time = time.time()
                    for _ in range(iterations):
                        _ = model(*input_args)
                    torch.cuda.synchronize() if torch.cuda.is_available() else None
                    end_time = time.time()

                    avg_latency = (end_time - start_time) / iterations * 1000 # ms
                    print(f"Inference Latency (BS=1): {avg_latency:.4f} ms")
                else:
                    avg_latency = 0.0
        except Exception as e:
            print(f"Latency calculation failed: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            avg_latency = 0.0

        # Save profiling results to a JSON file for the sensitivity script to read
        profiling_res = {
            "flops_M": flops / 1e6,
            "params_M": params / 1e6,
            "latency_ms": avg_latency,
            "test_mape": best_test_MAPE,
            "test_rmse": best_test_RMSE
        }
        with open(os.path.join(path, "profiling_stats.json"), "w") as f:
            json.dump(profiling_res, f)

    accelerator.set_trigger()
    if accelerator.check_trigger() and accelerator.is_local_main_process and wandb_available:
        try:
            wandb.log({"epoch": epoch+1, "train_loss": train_loss, "vali_RMSE": best_vali_RMSE, "vali_MAPE": best_vali_MAPE, "vali_acc1": best_vali_alpha_acc1, "vali_acc2": best_vali_alpha_acc2,
                    "test_RMSE": best_test_RMSE, "test_MAPE":best_test_MAPE, "test_acc1": best_test_alpha_acc1, "test_acc2": best_test_alpha_acc2})
            wandb.finish()
        except:
            pass

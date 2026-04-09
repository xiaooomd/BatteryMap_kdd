"""Training orchestration for prediction workflows."""

import json
import os
import time

import evaluate
import joblib
import numpy as np
import torch
from sklearn.metrics import mean_absolute_percentage_error, root_mean_squared_error
from torch import nn, optim
from torch.optim import lr_scheduler

from data_provider.data_factory import data_provider
from modules.predictor.evaluator import profile_model, save_best_metrics, summarize_best_metrics
from modules.predictor.model_factory import build_model
from utils.experiment_tracker import SwanLabTracker
from utils.tools import EarlyStopping, adjust_learning_rate, del_files, get_parameter_number, vali_baseline


def build_experiment_folder_name(args, folder_timestamp):
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

    folder_parts = [args.model, args.dataset, hyper_params]
    if args.model_comment and args.model_comment != 'none':
        folder_parts.append(args.model_comment)
    folder_parts.append(folder_timestamp)
    return "_".join(folder_parts)


def validate_model_args(args):
    if args.enc_in is None or args.enc_in <= 0:
        raise ValueError(f"Invalid enc_in value: {args.enc_in}. Must be a positive integer.")
    if args.d_model is None or args.d_model <= 0:
        raise ValueError(f"Invalid d_model value: {args.d_model}. Must be a positive integer.")
    if args.dec_in is None or args.dec_in <= 0:
        raise ValueError(f"Invalid dec_in value: {args.dec_in}. Must be a positive integer.")
    if args.c_out is None or args.c_out <= 0:
        raise ValueError(f"Invalid c_out value: {args.c_out}. Must be a positive integer.")


def create_optimizer_and_scheduler(args, model, train_steps):
    trained_parameters = []
    trained_parameters_names = []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad is True:
            trained_parameters_names.append(name)
            trained_parameters.append(parameter)

    model_optim = optim.Adam(trained_parameters, lr=args.learning_rate)

    if args.lradj == 'COS':
        scheduler = lr_scheduler.CosineAnnealingLR(model_optim, T_max=20, eta_min=1e-8)
    else:
        scheduler = lr_scheduler.OneCycleLR(
            optimizer=model_optim,
            steps_per_epoch=train_steps,
            pct_start=args.pct_start,
            epochs=args.train_epochs,
            max_lr=args.learning_rate)

    return trained_parameters_names, model_optim, scheduler


class BatteryLifeTrainer:
    def __init__(self, args, accelerator, nowtime, folder_timestamp):
        self.args = args
        self.accelerator = accelerator
        self.nowtime = nowtime
        self.folder_timestamp = folder_timestamp

    def run(self):
        for _ in range(self.args.itr):
            self._run_single_experiment()

    def _run_single_experiment(self):
        folder_name = build_experiment_folder_name(self.args, self.folder_timestamp)
        validate_model_args(self.args)

        print(
            f"Model parameters: enc_in={self.args.enc_in}, d_model={self.args.d_model}, "
            f"dec_in={self.args.dec_in}, c_out={self.args.c_out}"
        )

        model = build_model(self.args.model, self.args)
        path = os.path.join(self.args.checkpoints, folder_name)

        train_data, train_loader, vali_data, vali_loader, test_data, test_loader = self._load_datasets()
        label_scaler = train_data.return_label_scaler()
        life_class_scaler = train_data.return_life_class_scaler()

        if self.accelerator.is_local_main_process and os.path.exists(path):
            del_files(path)
            self.accelerator.print(f'success delete {path}')

        os.makedirs(path, exist_ok=True)
        self.accelerator.wait_for_everyone()
        joblib.dump(label_scaler, f'{path}/label_scaler')
        joblib.dump(life_class_scaler, f'{path}/life_class_scaler')
        with open(path + '/args.json', 'w') as handle:
            json.dump(self.args.__dict__, handle)

        swanlab_tracker = self._init_swanlab()

        para_res = get_parameter_number(model)
        self.accelerator.print(para_res)
        for name, module in model._modules.items():
            self.accelerator.print(name, " : ", module)

        train_steps = len(train_loader)
        if train_steps == 0:
            self.accelerator.print("WARNING: train_loader is empty. Setting train_steps to 1 to avoid scheduler error.")
            train_steps = 1

        early_stopping = EarlyStopping(accelerator=self.accelerator, patience=self.args.patience)
        trained_parameters_names, model_optim, scheduler = create_optimizer_and_scheduler(self.args, model, train_steps)
        self.accelerator.print(f'Trainable parameters are: {trained_parameters_names}')

        criterion = nn.MSELoss(reduction='none')
        train_loader, vali_loader, test_loader, model, model_optim, scheduler = self.accelerator.prepare(
            train_loader, vali_loader, test_loader, model, model_optim, scheduler)

        metrics = self._init_metrics()
        epoch = -1
        train_loss = 0.0

        for epoch in range(self.args.train_epochs):
            train_loss, epoch_state = self._train_epoch(
                epoch,
                train_data,
                train_loader,
                model,
                model_optim,
                scheduler,
                criterion,
            )

            vali_rmse, vali_mae_loss, vali_mape, vali_alpha_acc1, vali_alpha_acc2 = vali_baseline(
                self.args, self.accelerator, model, vali_data, vali_loader, criterion, compute_seen_unseen=False)
            test_metrics = vali_baseline(
                self.args, self.accelerator, model, test_data, test_loader, criterion, compute_seen_unseen=True)
            (
                test_rmse, test_mae_loss, test_mape, test_alpha_acc1, test_alpha_acc2,
                test_unseen_mape, test_seen_mape, test_unseen_alpha_acc1, test_seen_alpha_acc1,
                test_unseen_alpha_acc2, test_seen_alpha_acc2
            ) = test_metrics
            vali_loss = vali_mape

            if vali_loss < metrics['best_vali_loss']:
                metrics.update({
                    'best_vali_loss': vali_loss,
                    'best_vali_MAE': vali_mae_loss,
                    'best_test_MAE': test_mae_loss,
                    'best_vali_RMSE': vali_rmse,
                    'best_test_RMSE': test_rmse,
                    'best_vali_MAPE': vali_mape,
                    'best_test_MAPE': test_mape,
                    'best_vali_alpha_acc1': vali_alpha_acc1,
                    'best_vali_alpha_acc2': vali_alpha_acc2,
                    'best_test_alpha_acc1': test_alpha_acc1,
                    'best_test_alpha_acc2': test_alpha_acc2,
                    'best_seen_test_MAPE': test_seen_mape,
                    'best_unseen_test_MAPE': test_unseen_mape,
                    'best_seen_test_alpha_acc1': test_seen_alpha_acc1,
                    'best_unseen_test_alpha_acc1': test_unseen_alpha_acc1,
                    'best_seen_test_alpha_acc2': test_seen_alpha_acc2,
                    'best_unseen_test_alpha_acc2': test_unseen_alpha_acc2,
                })

            self.accelerator.print(
                f"Epoch: {epoch+1} | Train Loss: {train_loss:.5f}| Train cl loss: {epoch_state['total_cl_loss']:.5f}| "
                f"Train lc loss: {epoch_state['total_lc_loss']:.5f} | Train RMSE: {epoch_state['train_rmse']:.7f} | "
                f"Train MAPE: {epoch_state['train_mape']:.7f} | Vali RMSE: {vali_rmse:.7f}| "
                f"Vali MAE: {vali_mae_loss:.7f}| Vali MAPE: {vali_mape:.7f}| Test RMSE: {test_rmse:.7f}| "
                f"Test MAE: {test_mae_loss:.7f} | Test MAPE: {test_mape:.7f}"
            )

            if self.accelerator.is_local_main_process and swanlab_tracker.available:
                try:
                    swanlab_tracker.log({
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "vali_RMSE": vali_rmse,
                        "vali_MAPE": vali_mape,
                        "vali_acc1": vali_alpha_acc1,
                        "vali_acc2": vali_alpha_acc2,
                        "test_RMSE": test_rmse,
                        "test_MAPE": test_mape,
                        "test_acc1": test_alpha_acc1,
                        "test_acc2": test_alpha_acc2,
                    })
                except Exception:
                    pass

            early_stopping(epoch + 1, vali_loss, vali_mae_loss, test_mae_loss, model, path)
            if early_stopping.early_stop:
                self.accelerator.print("Early stopping")
                self.accelerator.set_trigger()

            if self.accelerator.check_trigger():
                break

            if self.args.lradj != 'TST':
                if self.args.lradj == 'COS':
                    scheduler.step()
                    self.accelerator.print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
                else:
                    adjust_learning_rate(self.accelerator, model_optim, scheduler, epoch + 1, self.args, printout=True)
            else:
                self.accelerator.print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        summarize_best_metrics(self.accelerator, path, metrics)
        save_best_metrics(self.accelerator, self.args, path, metrics)
        profile_model(self.accelerator, self.args, model, path, metrics)

        self.accelerator.set_trigger()
        if self.accelerator.check_trigger() and self.accelerator.is_local_main_process and swanlab_tracker.available:
            try:
                swanlab_tracker.log({
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "vali_RMSE": metrics['best_vali_RMSE'],
                    "vali_MAPE": metrics['best_vali_MAPE'],
                    "vali_acc1": metrics['best_vali_alpha_acc1'],
                    "vali_acc2": metrics['best_vali_alpha_acc2'],
                    "test_RMSE": metrics['best_test_RMSE'],
                    "test_MAPE": metrics['best_test_MAPE'],
                    "test_acc1": metrics['best_test_alpha_acc1'],
                    "test_acc2": metrics['best_test_alpha_acc2'],
                })
                swanlab_tracker.finish()
            except Exception:
                pass

    def _load_datasets(self):
        data_provider_func = data_provider
        self.accelerator.print("Loading training samples......")
        train_data, train_loader = data_provider_func(
            self.args, 'train', tokenizer=None, sample_weighted=self.args.weighted_sampling)
        label_scaler = train_data.return_label_scaler()
        life_class_scaler = train_data.return_life_class_scaler()

        feature_scaler = None
        if hasattr(train_data, 'return_feature_scaler'):
            feature_scaler = train_data.return_feature_scaler()

        self.accelerator.print("Loading vali samples......")
        vali_data, vali_loader = data_provider_func(
            self.args, 'val', tokenizer=None, label_scaler=label_scaler, 
            life_class_scaler=life_class_scaler, sample_weighted=self.args.weighted_sampling, 
            feature_scaler=feature_scaler)
        self.accelerator.print("Loading test samples......")
        test_data, test_loader = data_provider_func(
            self.args, 'test', tokenizer=None, label_scaler=label_scaler, 
            life_class_scaler=life_class_scaler, sample_weighted=self.args.weighted_sampling, 
            feature_scaler=feature_scaler)
        return train_data, train_loader, vali_data, vali_loader, test_data, test_loader

    def _init_swanlab(self):
        run_name = getattr(self.args, 'swanlab_experiment_name', None) or self.nowtime
        tracker = SwanLabTracker(
            project=getattr(self.args, 'swanlab_project', 'paper4_benchmark_baselines'),
            run_name=run_name,
            config=self.args.__dict__,
            enabled=self.accelerator.is_local_main_process and not getattr(self.args, 'disable_swanlab', False),
            mode=getattr(self.args, 'swanlab_mode', None),
            logger=self.accelerator.print,
        )
        return tracker

    def _init_metrics(self):
        return {
            'best_vali_loss': float('inf'),
            'best_vali_MAE': 0,
            'best_test_MAE': 0,
            'best_vali_RMSE': 0,
            'best_test_RMSE': 0,
            'best_vali_alpha_acc1': 0,
            'best_test_alpha_acc1': 0,
            'best_vali_alpha_acc2': 0,
            'best_test_alpha_acc2': 0,
            'best_seen_test_MAPE': 0,
            'best_unseen_test_MAPE': 0,
            'best_seen_test_alpha_acc1': 0,
            'best_unseen_test_alpha_acc1': 0,
            'best_seen_test_alpha_acc2': 0,
            'best_unseen_test_alpha_acc2': 0,
            'best_vali_MAPE': 0,
            'best_test_MAPE': 0,
        }

    def _train_epoch(self, epoch, train_data, train_loader, model, model_optim, scheduler, criterion):
        evaluate.load('./utils/mae')
        evaluate.load('./utils/mape')
        iter_count = 0
        total_loss = 0
        total_cl_loss = 0
        total_lc_loss = 0

        model.train()
        epoch_time = time.time()
        print_cl_loss = 0
        print_life_class_loss = 0
        std = np.sqrt(train_data.label_scaler.var_[-1])
        mean_value = train_data.label_scaler.mean_[-1]
        total_preds, total_references = [], []
        time_now = time.time()
        train_steps = max(len(train_loader), 1)

        for i, (cycle_curve_data, curve_attn_mask, labels, life_class, scaled_life_class, weights, seen_unseen_ids) in enumerate(train_loader):
            with self.accelerator.accumulate(model):
                model_optim.zero_grad()
                iter_count += 1

                life_class = life_class.to(self.accelerator.device)
                scaled_life_class = scaled_life_class.float().to(self.accelerator.device)
                cycle_curve_data = cycle_curve_data.float().to(self.accelerator.device)
                curve_attn_mask = curve_attn_mask.float().to(self.accelerator.device)
                labels = labels.float().to(self.accelerator.device)

                if self.args.feature_type == 'extracted_features' and self.args.model in ['Transformer', 'Informer', 'Autoformer', 'Reformer']:
                    batch_size, seq_len, _ = cycle_curve_data.shape
                    pred_len = self.args.pred_len
                    x_mark_enc = torch.zeros(batch_size, seq_len, 4).float().to(self.accelerator.device)
                    x_mark_dec = torch.zeros(batch_size, pred_len, 4).float().to(self.accelerator.device)
                    x_dec = torch.zeros(batch_size, pred_len, self.args.dec_in).float().to(self.accelerator.device)
                    outputs = model(cycle_curve_data, x_mark_enc, x_dec, x_mark_dec)
                else:
                    outputs = model(cycle_curve_data, curve_attn_mask)

                cut_off = labels.shape[0]
                if (i + 1) % 50 == 0:
                    self.accelerator.print(
                        f"DEBUG BEFORE SQUEEZE: outputs.shape={outputs.shape}, labels.shape={labels.shape}, "
                        f"args.loss={self.args.loss}, args.task_type={self.args.task_type}"
                    )

                if self.args.task_type == 'early_prediction':
                    if outputs.ndim == 2 and outputs.shape[1] == 1:
                        outputs = outputs.squeeze()
                    if labels.ndim == 2 and labels.shape[1] == 1:
                        labels = labels.squeeze()
                    loss = criterion(outputs, labels)
                    if self.args.weighted_loss:
                        if weights.ndim > 1:
                            weights = weights.squeeze()
                        loss = torch.mean(loss * weights)
                    else:
                        loss = torch.mean(loss)
                elif self.args.loss == 'MSE':
                    if outputs.ndim == 2 and outputs.shape[1] == 1:
                        outputs = outputs.squeeze()
                    if labels.ndim == 2 and labels.shape[1] == 1:
                        labels = labels.squeeze()
                    loss = criterion(outputs[:cut_off], labels)
                    loss = torch.mean(loss * weights)
                else:
                    tmp_outputs = outputs[:cut_off] * std + mean_value
                    tmp_labels = labels * std + mean_value
                    loss = criterion(tmp_outputs / tmp_labels, tmp_labels / tmp_labels)
                    loss = torch.mean(loss * weights)

                label_loss = loss.detach().float()
                print_loss = loss.detach().float()

                if torch.isnan(loss) or torch.isinf(loss):
                    self.accelerator.print(f"\n[CRITICAL ERROR] Loss is {loss.item()} at Epoch {epoch+1}, Step {iter_count}")
                    self.accelerator.print(
                        f"Input  (cycle_curve_data): min={cycle_curve_data.min().item():.4f}, "
                        f"max={cycle_curve_data.max().item():.4f}, mean={cycle_curve_data.mean().item():.4f}"
                    )
                    self.accelerator.print(
                        f"Output (outputs)         : min={outputs.min().item():.4f}, "
                        f"max={outputs.max().item():.4f}, mean={outputs.mean().item():.4f}"
                    )
                    self.accelerator.print(
                        f"Target (labels)          : min={labels.min().item():.4f}, "
                        f"max={labels.max().item():.4f}, mean={labels.mean().item():.4f}"
                    )
                    if torch.isnan(cycle_curve_data).any():
                        self.accelerator.print("Input contains NaN!")
                    if torch.isinf(cycle_curve_data).any():
                        self.accelerator.print("Input contains Inf!")

                total_loss += loss.detach().float()
                total_cl_loss += print_cl_loss
                total_lc_loss += print_life_class_loss

                transformed_preds = outputs[:cut_off] * std + mean_value
                transformed_labels = labels[:cut_off] * std + mean_value
                if (i + 1) % 50 == 0:
                    self.accelerator.print(
                        f"DEBUG: outputs shape: {outputs.shape}, transformed_preds shape: {transformed_preds.shape}"
                    )

                transformed_preds = transformed_preds.squeeze(-1)
                transformed_labels = transformed_labels.squeeze(-1)
                if transformed_preds.ndim > 1:
                    transformed_preds = transformed_preds.squeeze()
                if transformed_labels.ndim > 1:
                    transformed_labels = transformed_labels.squeeze()
                transformed_preds = transformed_preds.reshape(-1)
                transformed_labels = transformed_labels.reshape(-1)

                if (i + 1) % 50 == 0:
                    self.accelerator.print(
                        f"DEBUG: After squeeze - transformed_preds shape: {transformed_preds.shape}, "
                        f"transformed_labels shape: {transformed_labels.shape}"
                    )

                all_predictions, all_targets = self.accelerator.gather_for_metrics((transformed_preds, transformed_labels))
                if self.accelerator.is_main_process:
                    total_preds += all_predictions.detach().cpu().numpy().tolist()
                    total_references += all_targets.detach().cpu().numpy().tolist()

                self.accelerator.backward(loss)
                self.accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
                model_optim.step()

                if self.args.lradj == 'TST':
                    adjust_learning_rate(self.accelerator, model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

                if (i + 1) % 5 == 0:
                    self.accelerator.print(
                        f'\titeras: {i+1}, epoch: {epoch+1} | loss:{print_loss:.7f} | '
                        f'label_loss: {label_loss:.7f} | cl_loss: {print_cl_loss:.7f} | '
                        f'lc_loss: {print_life_class_loss:.7f}'
                    )
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    self.accelerator.print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

        if self.accelerator.is_main_process:
            train_rmse = root_mean_squared_error(total_references, total_preds)
            train_mape = mean_absolute_percentage_error(total_references, total_preds)
        else:
            train_rmse = 0.0
            train_mape = 0.0

        self.accelerator.print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
        return total_loss / len(train_loader), {
            'total_cl_loss': total_cl_loss / len(train_loader),
            'total_lc_loss': total_lc_loss / len(train_loader),
            'train_rmse': train_rmse,
            'train_mape': train_mape,
        }

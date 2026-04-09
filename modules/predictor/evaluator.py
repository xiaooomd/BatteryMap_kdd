"""Evaluation and profiling helpers for predictor workflows."""

import datetime
import json
import os
import time
import traceback

import pandas as pd
import torch


def summarize_best_metrics(accelerator, path, metrics):
    accelerator.print(
        f"Best model performance: Test MAE: {metrics['best_test_MAE']:.4f} | "
        f"Test RMSE: {metrics['best_test_RMSE']:.4f} | Test MAPE: {metrics['best_test_MAPE']:.4f} | "
        f"Test 15%-accuracy: {metrics['best_test_alpha_acc1']:.4f} | Test 10%-accuracy: {metrics['best_test_alpha_acc2']:.4f} | "
        f"Val MAE: {metrics['best_vali_MAE']:.4f} | Val RMSE: {metrics['best_vali_RMSE']:.4f} | "
        f"Val MAPE: {metrics['best_vali_MAPE']:.4f} | Val 15%-accuracy: {metrics['best_vali_alpha_acc1']:.4f} | "
        f"Val 10%-accuracy: {metrics['best_vali_alpha_acc2']:.4f} "
    )
    accelerator.print(
        f"Best model performance: Test Seen MAPE: {metrics['best_seen_test_MAPE']:.4f} | "
        f"Test Unseen MAPE: {metrics['best_unseen_test_MAPE']:.4f}"
    )
    accelerator.print(
        f"Best model performance: Test Seen 15%-accuracy: {metrics['best_seen_test_alpha_acc1']:.4f} | "
        f"Test Unseen 15%-accuracy: {metrics['best_unseen_test_alpha_acc1']:.4f}"
    )
    accelerator.print(
        f"Best model performance: Test Seen 10%-accuracy: {metrics['best_seen_test_alpha_acc2']:.4f} | "
        f"Test Unseen 10%-accuracy: {metrics['best_unseen_test_alpha_acc2']:.4f}"
    )
    accelerator.print(path)


def save_best_metrics(accelerator, args, path, metrics):
    if not accelerator.is_local_main_process:
        return

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    best_metrics = {
        "dataset": args.dataset,
        "model": args.model,
        "test_metrics": {
            "MAE": float(metrics["best_test_MAE"]),
            "RMSE": float(metrics["best_test_RMSE"]),
            "MAPE": float(metrics["best_test_MAPE"]),
            "accuracy_15pct": float(metrics["best_test_alpha_acc1"]),
            "accuracy_10pct": float(metrics["best_test_alpha_acc2"]),
        },
        "validation_metrics": {
            "MAE": float(metrics["best_vali_MAE"]),
            "RMSE": float(metrics["best_vali_RMSE"]),
            "MAPE": float(metrics["best_vali_MAPE"]),
            "accuracy_15pct": float(metrics["best_vali_alpha_acc1"]),
            "accuracy_10pct": float(metrics["best_vali_alpha_acc2"]),
        },
        "test_seen_unseen": {
            "seen_MAPE": float(metrics["best_seen_test_MAPE"]),
            "unseen_MAPE": float(metrics["best_unseen_test_MAPE"]),
            "seen_accuracy_15pct": float(metrics["best_seen_test_alpha_acc1"]),
            "unseen_accuracy_15pct": float(metrics["best_unseen_test_alpha_acc1"]),
            "seen_accuracy_10pct": float(metrics["best_seen_test_alpha_acc2"]),
            "unseen_accuracy_10pct": float(metrics["best_unseen_test_alpha_acc2"]),
        },
        "checkpoint_path": path,
        "timestamp": timestamp,
    }

    with open(os.path.join(path, "best_metrics.json"), "w") as handle:
        json.dump(best_metrics, handle, indent=4)
    accelerator.print(f"Best metrics saved to {os.path.join(path, 'best_metrics.json')}")

    results_dir = os.path.join("./results", args.model)
    os.makedirs(results_dir, exist_ok=True)
    csv_data = {
        "dataset": [args.dataset],
        "model": [args.model],
        "timestamp": [timestamp],
        "test_MAE": [float(metrics["best_test_MAE"])],
        "test_RMSE": [float(metrics["best_test_RMSE"])],
        "test_MAPE": [float(metrics["best_test_MAPE"])],
        "test_acc_15pct": [float(metrics["best_test_alpha_acc1"])],
        "test_acc_10pct": [float(metrics["best_test_alpha_acc2"])],
        "val_MAE": [float(metrics["best_vali_MAE"])],
        "val_RMSE": [float(metrics["best_vali_RMSE"])],
        "val_MAPE": [float(metrics["best_vali_MAPE"])],
        "val_acc_15pct": [float(metrics["best_vali_alpha_acc1"])],
        "val_acc_10pct": [float(metrics["best_vali_alpha_acc2"])],
        "test_seen_MAPE": [float(metrics["best_seen_test_MAPE"])],
        "test_unseen_MAPE": [float(metrics["best_unseen_test_MAPE"])],
        "test_seen_acc_15pct": [float(metrics["best_seen_test_alpha_acc1"])],
        "test_unseen_acc_15pct": [float(metrics["best_unseen_test_alpha_acc1"])],
        "test_seen_acc_10pct": [float(metrics["best_seen_test_alpha_acc2"])],
        "test_unseen_acc_10pct": [float(metrics["best_unseen_test_alpha_acc2"])],
        "checkpoint_path": [path],
    }
    csv_file = os.path.join(results_dir, f"{args.dataset}_results.csv")
    pd.DataFrame(csv_data).to_csv(csv_file, index=False, float_format="%.6f")
    accelerator.print(f"Results saved to {csv_file}")


def profile_model(accelerator, args, model, path, metrics):
    if not accelerator.is_local_main_process:
        return

    print("\n" + "=" * 40)
    print("Performance Profiling")
    print("=" * 40)

    try:
        from thop import profile

        model.eval()
        if args.feature_type == 'extracted_features':
            dummy_input = torch.randn(1, args.early_cycle_threshold, args.enc_in).float().to(accelerator.device)
            dummy_mask = torch.ones(1, args.early_cycle_threshold).float().to(accelerator.device)
            if args.model in ['Transformer', 'Informer', 'Autoformer']:
                batch_size, seq_len = 1, args.early_cycle_threshold
                x_mark_enc = torch.zeros(batch_size, seq_len, 4).float().to(accelerator.device)
                x_dec = torch.zeros(batch_size, args.pred_len, args.dec_in).float().to(accelerator.device)
                x_mark_dec = torch.zeros(batch_size, args.pred_len, 4).float().to(accelerator.device)
                flops, params = profile(model, inputs=(dummy_input, x_mark_enc, x_dec, x_mark_dec), verbose=False)
            else:
                flops, params = profile(model, inputs=(dummy_input, dummy_mask), verbose=False)
        else:
            print("FLOPs calculation skipped for legacy curve mode (complex 4D input).")
            flops, params = 0, 0
        print(f"FLOPs: {flops / 1e6:.4f} M")
        print(f"Params: {params / 1e6:.4f} M")
    except ImportError as exc:
        print(f"ImportError: {exc}")
        print("thop library not found. Install via 'pip install thop' to see FLOPs.")
        flops, params = 0, 0
    except Exception as exc:
        print(f"FLOPs calculation failed: {type(exc).__name__}: {exc}")
        flops, params = 0, 0

    try:
        model.eval()
        with torch.no_grad():
            if args.feature_type == 'extracted_features':
                dummy_input = torch.randn(1, args.early_cycle_threshold, args.enc_in).float().to(accelerator.device)
                dummy_mask = torch.ones(1, args.early_cycle_threshold).float().to(accelerator.device)
                if args.model in ['Transformer', 'Autoformer']:
                    x_mark_enc = torch.zeros(1, args.early_cycle_threshold, 4).float().to(accelerator.device)
                    x_dec = torch.zeros(1, args.pred_len, args.dec_in).float().to(accelerator.device)
                    x_mark_dec = torch.zeros(1, args.pred_len, 4).float().to(accelerator.device)
                    input_args = (dummy_input, x_mark_enc, x_dec, x_mark_dec)
                else:
                    input_args = (dummy_input, dummy_mask)
            else:
                input_args = None

            if input_args is not None:
                for _ in range(10):
                    _ = model(*input_args)
                iterations = 100
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start_time = time.time()
                for _ in range(iterations):
                    _ = model(*input_args)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                end_time = time.time()
                avg_latency = (end_time - start_time) / iterations * 1000
                print(f"Inference Latency (BS=1): {avg_latency:.4f} ms")
            else:
                avg_latency = 0.0
    except Exception as exc:
        print(f"Latency calculation failed: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        avg_latency = 0.0

    profiling_res = {
        "flops_M": flops / 1e6,
        "params_M": params / 1e6,
        "latency_ms": avg_latency,
        "test_mape": metrics["best_test_MAPE"],
        "test_rmse": metrics["best_test_RMSE"],
    }
    with open(os.path.join(path, "profiling_stats.json"), "w") as handle:
        json.dump(profiling_res, handle)
    accelerator.print(f"Profiling stats saved to {os.path.join(path, 'profiling_stats.json')}")

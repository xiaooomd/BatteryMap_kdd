"""
Summary script to collect all dataset training results.
Can read results from the checkpoints directory or results directory.
"""
import os
import json
import pandas as pd
from pathlib import Path
from datetime import datetime

def collect_from_results_dir(results_dir="./results", output_file="all_results_summary.csv"):
    """
    Collect all CSV result files from the results directory (organized by model).

    Args:
        results_dir: Root directory of results.
        output_file: Output CSV filename.
    """
    all_results = []

    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"Error: {results_dir} directory not found")
        return None

    # Iterate through each model directory
    for model_dir in sorted(results_path.iterdir()):
        if not model_dir.is_dir():
            continue

        model_name = model_dir.name
        print(f"\nProcessing model: {model_name}")

        # Read all CSV files under this model
        csv_files = list(model_dir.glob("*_results.csv"))
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                all_results.append(df)
                print(f"  ✓ {csv_file.name}")
            except Exception as e:
                print(f"  ✗ Failed to read {csv_file.name}: {e}")

    if not all_results:
        print("No result files found")
        return None

    # Merge all results
    combined_df = pd.concat(all_results, ignore_index=True)

    # Sort by model and dataset
    combined_df = combined_df.sort_values(["model", "dataset"])

    # Save summary results
    combined_df.to_csv(output_file, index=False, float_format='%.6f')
    print(f"\n{'='*60}")
    print(f"✓ Summary complete! Collected {len(combined_df)} results in total")
    print(f"✓ Saved to: {output_file}")
    print(f"{'='*60}")

    # Print brief statistics
    print("\nGrouped statistics by model:")
    summary = combined_df.groupby("model")[["test_MAPE", "test_RMSE", "test_acc_15pct"]].agg(['mean', 'std'])
    print(summary.to_string())

    print("\nGrouped statistics by dataset:")
    summary = combined_df.groupby("dataset")[["test_MAPE", "test_RMSE", "test_acc_15pct"]].mean()
    print(summary.to_string())

    return combined_df


def collect_from_checkpoints(checkpoints_dir="./checkpoints", output_file="all_results_summary.csv"):
    """
    Collect all best_metrics.json files from the checkpoint directories.

    Args:
        checkpoints_dir: Root directory of checkpoints.
        output_file: Output CSV filename.
    """
    results = []

    # Iterate through all checkpoint directories
    checkpoints_path = Path(checkpoints_dir)
    if not checkpoints_path.exists():
        print(f"Error: {checkpoints_dir} directory not found")
        return

    for checkpoint_dir in sorted(checkpoints_path.iterdir()):
        if not checkpoint_dir.is_dir():
            continue

        metrics_file = checkpoint_dir / "best_metrics.json"
        if not metrics_file.exists():
            continue

        try:
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)

            # Extract key information
            result = {
                "checkpoint": checkpoint_dir.name,
                "dataset": metrics.get("dataset", "N/A"),
                "model": metrics.get("model", "N/A"),
                "timestamp": metrics.get("timestamp", "N/A"),

                # Test set metrics
                "test_MAE": metrics["test_metrics"]["MAE"],
                "test_RMSE": metrics["test_metrics"]["RMSE"],
                "test_MAPE": metrics["test_metrics"]["MAPE"],
                "test_acc_15pct": metrics["test_metrics"]["accuracy_15pct"],
                "test_acc_10pct": metrics["test_metrics"]["accuracy_10pct"],

                # Validation set metrics
                "val_MAE": metrics["validation_metrics"]["MAE"],
                "val_RMSE": metrics["validation_metrics"]["RMSE"],
                "val_MAPE": metrics["validation_metrics"]["MAPE"],
                "val_acc_15pct": metrics["validation_metrics"]["accuracy_15pct"],
                "val_acc_10pct": metrics["validation_metrics"]["accuracy_10pct"],

                # Seen/Unseen
                "test_seen_MAPE": metrics["test_seen_unseen"]["seen_MAPE"],
                "test_unseen_MAPE": metrics["test_seen_unseen"]["unseen_MAPE"],
                "test_seen_acc_15pct": metrics["test_seen_unseen"]["seen_accuracy_15pct"],
                "test_unseen_acc_15pct": metrics["test_seen_unseen"]["unseen_accuracy_15pct"],
                "test_seen_acc_10pct": metrics["test_seen_unseen"]["seen_accuracy_10pct"],
                "test_unseen_acc_10pct": metrics["test_seen_unseen"]["unseen_accuracy_10pct"],
            }

            results.append(result)
            print(f"✓ Collected: {checkpoint_dir.name}")

        except Exception as e:
            print(f"✗ Failed to read {checkpoint_dir.name}: {e}")
            continue

    if not results:
        print("No result files found")
        return

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Sort by dataset name
    df = df.sort_values("dataset")

    # Save as CSV
    df.to_csv(output_file, index=False, float_format='%.4f')
    print(f"\n{'='*60}")
    print(f"✓ Summary complete! Collected {len(results)} results in total")
    print(f"✓ Saved to: {output_file}")
    print(f"{'='*60}")

    # Print brief statistics
    print("\nGrouped statistics by dataset:")
    summary = df.groupby("dataset")[["test_MAPE", "test_RMSE", "test_acc_15pct"]].mean()
    print(summary.to_string())

    return df


def generate_latex_table(df, output_file="results_table.tex"):
    """Generate a LaTeX format table."""
    if df is None or df.empty:
        return

    # Select key columns
    key_cols = ["dataset", "test_MAE", "test_RMSE", "test_MAPE", "test_acc_15pct", "test_acc_10pct"]
    table_df = df[key_cols].copy()

    # Rename columns
    table_df.columns = ["Dataset", "MAE", "RMSE", "MAPE (%)", "Acc@15%", "Acc@10%"]

    # Convert to LaTeX
    latex_str = table_df.to_latex(index=False, float_format="%.4f", caption="Model Performance on All Datasets", label="tab:results")

    with open(output_file, 'w') as f:
        f.write(latex_str)

    print(f"\n✓ LaTeX table saved to: {output_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Collect training results")
    parser.add_argument("--source", type=str, default="results", choices=["results", "checkpoints"],
                        help="Data source: results directory (organized by model) or checkpoints directory")
    parser.add_argument("--results_dir", type=str, default="./results", help="Path to results directory")
    parser.add_argument("--checkpoints_dir", type=str, default="./checkpoints", help="Path to checkpoints directory")
    parser.add_argument("--output", type=str, default="all_results_summary.csv", help="Output CSV filename")
    parser.add_argument("--latex", action="store_true", help="Also generate a LaTeX table")

    args = parser.parse_args()

    if args.source == "results":
        df = collect_from_results_dir(args.results_dir, args.output)
    else:
        df = collect_from_checkpoints(args.checkpoints_dir, args.output)

    if args.latex and df is not None:
        generate_latex_table(df, args.output.replace('.csv', '.tex'))

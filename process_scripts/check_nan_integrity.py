import os
import pandas as pd
import numpy as np
from datetime import datetime

def check_nan_in_csvs(root_path, output_report_path):
    """
    Recursively finds CSV files, checks for NaN values, and generates a Markdown report.
    """

    report_lines = []
    report_lines.append(f"# CSV NaN Integrity Report")
    report_lines.append(f"**Generated Time**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"**Root Path**: `{root_path}`")
    report_lines.append("")
    report_lines.append("## Summary")
    report_lines.append("| Dataset | File | Rows | Columns | NaN Count | Location (First 5) |")
    report_lines.append("| :--- | :--- | :--- | :--- | :--- | :--- |")

    total_files = 0
    files_with_nan = 0

    if not os.path.exists(root_path):
        print(f"Error: Path {root_path} does not exist.")
        return

    # Walk through directory
    for root, dirs, files in os.walk(root_path):
        for file in files:
            if not file.endswith('.csv'):
                continue

            total_files += 1
            file_path = os.path.join(root, file)
            rel_path = os.path.relpath(file_path, root_path)
            dataset_name = os.path.dirname(rel_path)
            file_name = os.path.basename(rel_path)

            try:
                # Read CSV
                df = pd.read_csv(file_path)

                # Check for NaNs
                if df.isnull().values.any():
                    files_with_nan += 1
                    nan_count = df.isnull().sum().sum()

                    # Find locations
                    nan_locations = []
                    # Stack to get MultiIndex (row, col) of NaNs
                    # Removed future_stack=True for compatibility
                    nan_stack = df[df.isnull()].stack(dropna=False)
                    # Use numpy to find indices where null
                    rows, cols = np.where(df.isnull())

                    for r, c in zip(rows, cols):
                        col_name = df.columns[c]
                        nan_locations.append(f"R{r}:C'{col_name}'")
                        if len(nan_locations) >= 5:
                            break

                    loc_str = ", ".join(nan_locations)
                    if nan_count > 5:
                        loc_str += "..."

                    report_lines.append(f"| {dataset_name} | {file_name} | {df.shape[0]} | {df.shape[1]} | {nan_count} | {loc_str} |")
                    print(f"[WARN] NaN found in {rel_path}")
                else:
                    # Optional: Print progress for clean files
                    # print(f"[OK] {rel_path}")
                    pass

            except Exception as e:
                report_lines.append(f"| {dataset_name} | {file_name} | - | - | ERROR | {str(e)} |")
                print(f"[ERROR] Failed to read {rel_path}: {e}")

    report_lines.append("")
    report_lines.append("## Statistics")
    report_lines.append(f"- **Total Files Scanned**: {total_files}")
    report_lines.append(f"- **Files with NaN**: {files_with_nan}")

    # Write report
    os.makedirs(os.path.dirname(output_report_path), exist_ok=True)
    with open(output_report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))

    print(f"Report generated at: {output_report_path}")

if __name__ == "__main__":
    # Define paths relative to project root (assuming script is run from project root)
    # Adjust paths if running from process_scripts folder
    current_dir = os.getcwd()

    # Path logic: Ensure we point to project root regardless of where script is run
    if os.path.basename(current_dir) == 'process_scripts':
        project_root = os.path.dirname(current_dir)
    elif os.path.basename(current_dir) == 'BatteryLife':
         project_root = current_dir
    else:
        # Fallback assuming standard structure
        project_root = os.getcwd()

    dataset_path = os.path.join(project_root, 'dataset', 'selected_result')
    output_path = os.path.join(project_root, 'docs', 'nan_check_report.md')

    print(f"Scanning: {dataset_path}")
    check_nan_in_csvs(dataset_path, output_path)

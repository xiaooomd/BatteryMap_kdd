#!/usr/bin/env bash
#
# Custom feature selection pipeline execution script
#
# Applicable to Linux/macOS systems.
# Robustness enhanced via 'set -euo pipefail', ensuring immediate exit on errors.
#

# --- Script Configuration ---
# Exit immediately if a command exits with a non-zero status.
set -e
# Treat unset variables as errors.
set -u
# Return value of a pipeline is the value of the last command to exit with non-zero status.
set -o pipefail

# --- Parameter Configuration ---

# Dataset ID(s) to process (if empty, process all datasets; multiple IDs separated by spaces)
# Example: BATTERY_ID="CALB CALCE"
BATTERY_ID=""

# -- Step 2: Initial Filter Parameters --

# Mode selection (FILTER_MODE):
# - 0: (default) Feature vs Feature (remove inter-feature collinearity).
# - 1: Feature vs Target (keep features strongly correlated with target y).
FILTER_MODE=0

# Method selection (FILTER_METHOD): Choose a correlation calculation method for the mode above.
# - p: pearson (default)
# - s: spearman
# - k: kendall
# - m: mutual_info (only effective when mode=1)
FILTER_METHOD="p"

# Threshold setting (FILTER_THRESHOLD):
# - When MODE=0, recommended 0.95 (remove features with correlation > 0.95).
# - When MODE=1, recommended 0.2 (keep features with correlation > 0.2).
FILTER_THRESHOLD=0.95

# -- Step 3: Fine Filter Parameters --
# Choose from the SELECTOR_METHOD list below
# - rfe:           Recursive Feature Elimination.
# - random_forest: Random Forest-based feature importance.
# - shap:          (Recommended) SHAP value-based feature importance.
SELECTOR_METHOD="shap"

# Final number of features to retain
TOP_K=20


# --- Execution ---
echo "================================================="
echo "  Running Feature Selection Pipeline (Advanced)"
echo "================================================="
echo "Dataset ID(s)     : ${BATTERY_ID:-"All available datasets"}"
echo "Step 1: Cleaning  : Enabled (automatic)"
echo "Step 2: Initial    : Mode=${FILTER_MODE}, Method=${FILTER_METHOD}, Threshold=${FILTER_THRESHOLD}"
echo "Step 3: Fine       : Method=${SELECTOR_METHOD}, Top_K=${TOP_K}"
echo "-------------------------------------------------"

# --- Environment Activation ---
# Important: Activate your Python environment here (e.g., Conda or venv)
# Conda example:
# if ! command -v conda &> /dev/null; then
#     echo "Error: Conda command not found. Please install and configure Conda first."
#     exit 1
# fi
# source "$(conda info --base)/etc/profile.d/conda.sh"
# conda activate your_env_name

# --- Command Assembly and Execution ---

# Safely build command arguments using arrays to avoid quoting and word splitting issues.
CMD_ARGS=()

if [[ -n "${BATTERY_ID}" ]]; then
    # Split BATTERY_ID string by spaces into array
    IFS=' ' read -r -a id_array <<< "${BATTERY_ID}"
    CMD_ARGS+=(--dataset_id "${id_array[@]}")
fi

CMD_ARGS+=(--filter_mode "${FILTER_MODE}")
CMD_ARGS+=(--filter_method "${FILTER_METHOD}")
CMD_ARGS+=(--filter_threshold "${FILTER_THRESHOLD}")
CMD_ARGS+=(--selector_method "${SELECTOR_METHOD}")
CMD_ARGS+=(--top_k "${TOP_K}")

echo "Command to execute:"
# Use printf for safer command printing
printf "python run.py"
for arg in "${CMD_ARGS[@]}"; do
  printf " %q" "$arg"
done
printf "\\n"
echo "-------------------------------------------------"

# Execute the main Python script with assembled arguments
python run.py "${CMD_ARGS[@]}"

echo "-------------------------------------------------"
echo "          Pipeline execution completed."
echo "================================================="

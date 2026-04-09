#!/usr/bin/env bash
#
# 自定义特征筛选流水线运行脚本
#
# 适用于 Linux/macOS 系统.
# 通过 'set -euo pipefail' 增强脚本的健壮性，确保在发生错误时能立即退出。
#

# --- 脚本配置 ---
# 如果命令以非零状态退出，则立即退出。
set -e
# 将未设置的变量视为错误。
set -u
# 管道命令的返回值将是最后一个以非零状态退出的命令的返回值。
set -o pipefail

# --- 参数配置 ---

# 要处理的数据集ID (如果为空, 则处理所有数据集; 可提供多个ID, 用空格隔开)
# 例如: BATTERY_ID="CALB CALCE"
BATTERY_ID=""

# -- 步骤 2: 初筛参数 --

# 模式选择 (FILTER_MODE):
# - 0: (默认) 特征 vs 特征 (移除特征间共线性)。
# - 1: 特征 vs 目标 (保留与目标y强相关的特征)。
FILTER_MODE=0

# 方法选择 (FILTER_METHOD): 为上面的模式选择一个相关性计算方法。
# - p: pearson (默认)
# - s: spearman
# - k: kendall
# - m: mutual_info (仅在 mode=1 时有效)
FILTER_METHOD="p"

# 阈值设定 (FILTER_THRESHOLD):
# - 当 MODE=0 时, 推荐 0.95 (移除相关性 > 0.95 的)。
# - 当 MODE=1 时, 推荐 0.2 (保留相关性 > 0.2 的)。
FILTER_THRESHOLD=0.95

# -- 步骤 3: 精筛参数 --
# 从下面的 SELECTOR_METHOD 列表中选择
# - rfe:           递归特征消除法。
# - random_forest: 基于随机森林的特征重要性。
# - shap:          (推荐) 基于 SHAP 值的特征重要性。
SELECTOR_METHOD="shap"

# 最终保留的特征数量
TOP_K=20


# --- 执行命令 ---
echo "================================================="
echo "  运行特征筛选流水线 (高级)"
echo "================================================="
echo "数据集 ID(s)     : ${BATTERY_ID:-"所有可用数据集"}"
echo "步骤 1: 清洗     : 已启用 (自动)"
echo "步骤 2: 初筛     : Mode=${FILTER_MODE}, Method=${FILTER_METHOD}, Threshold=${FILTER_THRESHOLD}"
echo "步骤 3: 精筛     : Method=${SELECTOR_METHOD}, Top_K=${TOP_K}"
echo "-------------------------------------------------"

# --- 环境激活 ---
# 重要: 请在此处激活您的 Python 环境 (例如 Conda 或 venv)
# Conda 示例:
# if ! command -v conda &> /dev/null; then
#     echo "错误: 未找到 Conda 命令。请先安装并配置 Conda。"
#     exit 1
# fi
# source "$(conda info --base)/etc/profile.d/conda.sh"
# conda activate your_env_name

# --- 命令组装与执行 ---

# 使用数组安全地构建命令参数，避免引用和分词问题。
CMD_ARGS=()

if [[ -n "${BATTERY_ID}" ]]; then
    # 将 BATTERY_ID 字符串按空格分割成数组
    IFS=' ' read -r -a id_array <<< "${BATTERY_ID}"
    CMD_ARGS+=(--dataset_id "${id_array[@]}")
fi

CMD_ARGS+=(--filter_mode "${FILTER_MODE}")
CMD_ARGS+=(--filter_method "${FILTER_METHOD}")
CMD_ARGS+=(--filter_threshold "${FILTER_THRESHOLD}")
CMD_ARGS+=(--selector_method "${SELECTOR_METHOD}")
CMD_ARGS+=(--top_k "${TOP_K}")

echo "即将执行命令:"
# 使用 printf 更安全地打印命令
printf "python run.py"
for arg in "${CMD_ARGS[@]}"; do
  printf " %q" "$arg"
done
printf "\\n"
echo "-------------------------------------------------"

# 使用组装好的参数执行主 Python 脚本
python run.py "${CMD_ARGS[@]}"

echo "-------------------------------------------------"
echo "          流水线执行完毕。"
echo "================================================="

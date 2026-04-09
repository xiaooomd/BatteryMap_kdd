# modules/utils.py
import logging
import os
import sys
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Dict

# 解决matplotlib中文显示问题
# plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

def setup_logger():
    """
    配置全局 Logger，支持输出到控制台和文件。
    """
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_filename = datetime.now().strftime(f"{log_dir}/%Y-%m-%d_%H-%M-%S.log")

    logger = logging.getLogger("FeatureSelection")
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger

def plot_feature_importance(importance_df: pd.DataFrame, dataset_id: str, save_dir: str, top_n: int = 20, selector_name: str = 'importance', filename_prefix: str = None):
    """
    生成并保存数据集的特征重要性排序图。
    Args:
        filename_prefix: 用于文件名的前缀（若为None，默认为 selector_name）
                         若指定（如 SNL_LFP），则文件名为 {filename_prefix}_shap_feature_importance.png
    """
    # 确定文件名
    prefix = filename_prefix if filename_prefix else selector_name
    # 保持旧逻辑兼容：如果没传 filename_prefix，原来的逻辑是 {selector_name}_feature_importance.png
    # 新逻辑下，如果不传 filename_prefix，prefix=selector_name，结果一致。
    # 但为了满足需求 "SNL_LFP_shap_feature_importance.png"，调用时需传入 filename_prefix="SNL_LFP" 且保持 selector_name="shap" 以防混淆。
    # 修改策略：文件名固定模式 {filename_prefix}_{selector_name}_feature_importance.png ?
    # 或者简单点，调用者全权负责 prefix。
    # 根据 run.py 的调用：filename_prefix=file_prefix (e.g. SNL_LFP)
    # selector_name='shap'
    # 期望文件名: SNL_LFP_shap_feature_importance.png

    if filename_prefix:
        save_path = os.path.join(save_dir, f"{filename_prefix}_{selector_name}_feature_importance.png")
    else:
        save_path = os.path.join(save_dir, f"{selector_name}_feature_importance.png")

    logger = logging.getLogger("FeatureSelection")

    plt.figure(figsize=(12, max(6, top_n // 2)))
    # 使用 Seaborn 的 barplot
    sns.set_style("whitegrid")

    top_features = importance_df.sort_values(by='importance', ascending=False).head(top_n)

    ax = sns.barplot(
        x='importance',
        y='feature',
        data=top_features,
        palette='viridis'
    )

    ax.set_title(f"Dataset: {dataset_id} - Method: {selector_name.upper()} - Top {top_n} Feature Importance", fontsize=16, weight='bold')
    ax.set_xlabel("Importance Score", fontsize=12)
    ax.set_ylabel("Feature", fontsize=12)
    plt.tight_layout()

    try:
        plt.savefig(save_path, dpi=300)
        logger.info(f"特征重要性图已保存: {save_path}")
    except Exception as e:
        logger.error(f"保存图表失败: {save_path}. 错误: {e}")
    plt.close()

def plot_correlation_heatmap(df: pd.DataFrame, save_dir: str, title_suffix: str = ""):
    """
    绘制并保存特征相关性热力图。
    """
    logger = logging.getLogger("FeatureSelection")
    save_path = os.path.join(save_dir, f"correlation_heatmap_{title_suffix}.png")

    n_cols = df.shape[1]
    if n_cols > 50:
        logger.warning(f"特征数量过多 ({n_cols})，热力图可能难以阅读。")
        plt.figure(figsize=(20, 18))
    else:
        plt.figure(figsize=(12, 10))

    sns.set_style("white")

    corr = df.corr()

    sns.heatmap(
        corr,
        annot=True if n_cols < 20 else False,
        fmt=".2f",
        cmap='coolwarm',
        vmax=1.0, vmin=-1.0,
        square=True,
        linewidths=.5,
        cbar_kws={"shrink": .5}
    )

    plt.title(f"Feature Correlation Matrix {title_suffix}", fontsize=16)
    plt.tight_layout()

    try:
        plt.savefig(save_path, dpi=300)
        logger.info(f"相关性热力图已保存: {save_path}")
    except Exception as e:
        logger.error(f"保存热力图失败: {save_path}. 错误: {e}")
    plt.close()

def plot_feature_frequency(feature_counts: pd.Series, save_dir: str, top_n: int = 30):
    """
    绘制特征在所有数据集中出现的频率。
    """
    logger = logging.getLogger("FeatureSelection")
    save_path = os.path.join(save_dir, "robust_feature_frequency.png")

    plt.figure(figsize=(14, 8))
    sns.set_style("darkgrid")

    data_to_plot = feature_counts.head(top_n)

    ax = sns.barplot(x=data_to_plot.values, y=data_to_plot.index, palette="rocket")

    for i, v in enumerate(data_to_plot.values):
        ax.text(v + 0.1, i, str(v), color='black', va='center', fontweight='bold')

    plt.title(f"Top {top_n} Most Robust Features (Frequency across Datasets)", fontsize=16, weight='bold')
    plt.xlabel("Frequency (Count of Datasets)", fontsize=12)
    plt.ylabel("Feature Name", fontsize=12)
    plt.tight_layout()

    try:
        plt.savefig(save_path, dpi=300)
        logger.info(f"特征频率分布图已保存: {save_path}")
    except Exception as e:
        logger.error(f"保存特征频率图失败: {e}")
    plt.close()

def plot_selection_heatmap(selection_matrix: pd.DataFrame, save_dir: str):
    """
    绘制特征选择的二值热力图。
    """
    logger = logging.getLogger("FeatureSelection")
    save_path = os.path.join(save_dir, "feature_selection_matrix.png")

    if selection_matrix.empty:
        return

    row_sum = selection_matrix.sum(axis=1)
    sorted_idx = row_sum.sort_values(ascending=False).index
    plot_data = selection_matrix.loc[sorted_idx]

    if plot_data.shape[0] > 50:
        logger.info(f"特征总数 ({plot_data.shape[0]}) 较多，热力图仅展示 Top 50 鲁棒特征。")
        plot_data = plot_data.head(50)

    plt.figure(figsize=(12, max(8, plot_data.shape[0] * 0.3)))
    sns.set_style("white")

    sns.heatmap(
        plot_data,
        cmap="Blues",
        cbar=False,
        linewidths=0.1,
        linecolor='lightgray',
        yticklabels=True,
        xticklabels=True
    )

    plt.title("Feature Selection Matrix (Blue = Selected)", fontsize=16, weight='bold')
    plt.xlabel("Dataset ID", fontsize=12)
    plt.ylabel("Feature (Sorted by Frequency)", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    try:
        plt.savefig(save_path, dpi=300)
        logger.info(f"特征选择矩阵热力图已保存: {save_path}")
    except Exception as e:
        logger.error(f"保存选择矩阵图失败: {e}")
    plt.close()


class MarkdownReporter:
    """
    负责生成 Markdown 格式的特征工程报告。
    """
    def __init__(self, output_dir: str):
        # 直接使用传入的目录，不再强制追加子目录
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_report(self, dataset_name: str,
                        cleaned_info: Dict[str, List[str]],
                        grouped_features: Dict[str, List[str]],
                        drop_report: pd.DataFrame,
                        shap_importance: pd.DataFrame,
                        top_k: int = 15):

        file_path = os.path.join(self.output_dir, f"{dataset_name}_report.md")

        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(f"# Feature Engineering Report: {dataset_name}\n\n")
                # f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

                # 1. Cleaning
                f.write("## 1. Data Cleaning Results\n")
                if not cleaned_info:
                    f.write("- No columns dropped during dataset-level cleaning.\n")
                else:
                    for reason, cols in cleaned_info.items():
                        f.write(f"### {reason} ({len(cols)})\n")
                        # 使用无序列表
                        for col in cols:
                            f.write(f"- {col}\n")
                f.write("\n")

                # 2. Grouping
                f.write("## 2. Feature Grouping\n")
                for group, features in grouped_features.items():
                    f.write(f"### {group} ({len(features)})\n")
                    for feat in features:
                        f.write(f"- {feat}\n")
                f.write("\n")

                # 3. Correlation Filter
                f.write("## 3. Correlation Filter Results (Physical Priority)\n")
                if drop_report.empty:
                    f.write("- No features dropped due to correlation.\n")
                else:
                    f.write("| Stage | Dropped Feature | Kept Substitute | Correlation | Reason |\n")
                    f.write("| --- | --- | --- | --- | --- |\n")
                    for _, row in drop_report.iterrows():
                        f.write(f"| {row.get('stage', 'N/A')} | {row['dropped_feature']} | {row['kept_substitute']} | {row['correlation_between_features']:.4f} | {row.get('drop_reason', 'N/A')} |\n")
                f.write("\n")

                # 4. SHAP Results
                f.write(f"## 4. SHAP Feature Selection (Top {top_k})\n")
                if shap_importance.empty:
                     f.write("- No features selected.\n")
                else:
                    top_n = shap_importance.head(top_k)
                    f.write("| Rank | Feature | Importance |\n")
                    f.write("| --- | --- | --- |\n")
                    for idx, row in top_n.iterrows():
                        f.write(f"| {idx + 1} | {row['feature']} | {row['importance']:.6f} |\n")

            logging.getLogger("FeatureSelection").info(f"Markdown report generated: {file_path}")
        except Exception as e:
            logging.getLogger("FeatureSelection").error(f"生成 Markdown 报告失败: {e}")


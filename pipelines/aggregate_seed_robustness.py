import os
import glob
import pandas as pd
import logging
from datetime import datetime
from typing import List, Dict

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def find_ranking_files(root_dir: str) -> List[str]:
    """递归查找所有的 _seed_rankings.csv 文件"""
    pattern = os.path.join(root_dir, "**", "*_seed_rankings.csv")
    files = glob.glob(pattern, recursive=True)
    return files

def analyze_single_file(file_path: str) -> pd.DataFrame:
    """
    分析单个 CSV 文件，计算特征的稳定性指标。
    输入 CSV 列: feature, rank, importance, seed, is_selected
    输出 DataFrame: feature, selection_rate, avg_rank, avg_importance, std_rank
    """
    try:
        df = pd.read_csv(file_path)
        required_cols = {'feature', 'rank', 'importance', 'seed', 'is_selected'}
        if not required_cols.issubset(df.columns):
            logger.warning(f"文件 {file_path} 缺少必要列，跳过。")
            return pd.DataFrame()

        n_seeds = df['seed'].nunique()

        # 聚合统计
        stats = df.groupby('feature').agg(
            selection_count=('is_selected', 'sum'),
            avg_rank=('rank', 'mean'),
            std_rank=('rank', 'std'),
            avg_importance=('importance', 'mean'),
            std_importance=('importance', 'std')
        ).reset_index()

        stats['selection_rate'] = stats['selection_count'] / n_seeds

        # 填充 NaN (针对 std 计算)
        stats['std_rank'] = stats['std_rank'].fillna(0)
        stats['std_importance'] = stats['std_importance'].fillna(0)

        # 排序: 优先选择率高，其次平均排名低
        stats = stats.sort_values(by=['selection_rate', 'avg_rank'], ascending=[False, True])

        return stats
    except Exception as e:
        logger.error(f"处理文件 {file_path} 时出错: {e}")
        return pd.DataFrame()

def generate_markdown_report(report_data: Dict[str, pd.DataFrame], output_path: str, top_n: int = 20):
    """生成汇总 Markdown 报告"""

    # 1. 准备全局统计 (Global Cross-Dataset Robustness)
    # 统计特征在多少个数据集中被标记为"强健" (Selection Rate > 0.8)
    global_feature_counter = {}
    dataset_count = len(report_data)

    for dataset, df in report_data.items():
        # 定义该数据集下的"强特征": 出现率 > 80% 且 平均排名前 top_n
        robust_feats = df[
            (df['selection_rate'] >= 0.8) &
            (df['avg_rank'] <= top_n)
        ]['feature'].tolist()

        for feat in robust_feats:
            global_feature_counter[feat] = global_feature_counter.get(feat, 0) + 1

    # 转为 DataFrame 并排序
    global_df = pd.DataFrame(list(global_feature_counter.items()), columns=['Feature', 'Robust_Dataset_Count'])
    global_df['Global_Frequency'] = global_df['Robust_Dataset_Count'] / dataset_count if dataset_count > 0 else 0
    global_df = global_df.sort_values(by='Robust_Dataset_Count', ascending=False)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Global Multi-Seed Robustness Analysis Report\n\n")
        f.write(f"**Generated Time**: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"**Total Datasets Analyzed**: {dataset_count}\n")
        f.write(f"**Top K Standard**: {top_n}\n\n")

        # Section 1: Global Summary
        f.write("## 1. Cross-Dataset Robust Features\n")
        f.write(f"特征在多个数据集中均表现出高鲁棒性 (Selection Rate >= 80% & Avg Rank <= {top_n})。\n\n")

        if global_df.empty:
            f.write("No globally robust features found.\n")
        else:
            f.write("| Rank | Feature | Robust in N Datasets | Global Frequency |\n")
            f.write("| --- | --- | --- | --- |\n")
            for idx, row in global_df.head(top_n).iterrows():
                f.write(f"| {idx+1} | {row['Feature']} | {row['Robust_Dataset_Count']} | {row['Global_Frequency']:.1%} |\n")
        f.write("\n")

        # Section 2: Per-Dataset Details
        f.write(f"## 2. Per-Dataset Robustness Details (Top {top_n})\n")

        if not report_data:
            f.write("No valid datasets found.\n")
        else:
            for dataset_name, stats_df in report_data.items():
                f.write(f"### {dataset_name}\n")
                f.write("| Rank | Feature | Selection Rate | Avg Rank | Avg Importance |\n")
                f.write("| --- | --- | --- | --- | --- |\n")

                # 展示 Top N
                top_features = stats_df.head(top_n)
                for idx, row in top_features.iterrows():
                    f.write(f"| {idx+1} | {row['feature']} | {row['selection_rate']:.1%} | {row['avg_rank']:.1f} | {row['avg_importance']:.4f} |\n")
                f.write("\n")

    logger.info(f"报告已生成: {output_path}")

def main():
    outputs_dir = "outputs"
    report_output_path = "feature_eng/Multi_Seed_Global_Report.md"

    if not os.path.exists(outputs_dir):
        logger.error(f"目录 {outputs_dir} 不存在。")
        return

    logger.info("开始扫描多随机种子结果文件...")
    files = find_ranking_files(outputs_dir)

    if not files:
        logger.warning("未找到任何 *_seed_rankings.csv 文件。请先运行带 --n_seeds > 1 的特征工程流程。")
        return

    logger.info(f"找到 {len(files)} 个文件。开始分析...")

    dataset_reports = {}
    EXCLUDE_DATASETS = ["Stanford_2"]

    for file_path in files:
        # 从文件名提取数据集名称 (e.g., outputs/CALB_final/CALB_seed_rankings.csv -> CALB)
        file_name = os.path.basename(file_path)
        dataset_name = file_name.replace("_seed_rankings.csv", "")

        if dataset_name in EXCLUDE_DATASETS:
            logger.info(f"忽略数据集: {dataset_name}")
            continue

        logger.info(f"正在分析: {dataset_name}")
        stats_df = analyze_single_file(file_path)

        if not stats_df.empty:
            dataset_reports[dataset_name] = stats_df

    logger.info("正在生成汇总报告...")
    generate_markdown_report(dataset_reports, report_output_path)
    logger.info("完成。")

if __name__ == "__main__":
    main()

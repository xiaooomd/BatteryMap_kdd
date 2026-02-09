# scripts/aggregate_results.py
import os
import sys
import argparse
import pandas as pd
import logging
from typing import List, Dict

# 将项目根目录添加到路径，以便导入模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.utils import setup_logger, plot_feature_frequency, plot_selection_heatmap

def aggregate_features(outputs_dir: str, summary_dir: str):
    """
    遍历 outputs 目录，聚合所有 dataset 的 selected_features.csv。
    生成汇总矩阵和频率报告。
    """
    logger = logging.getLogger("BatteryFeatureProject.Aggregator")
    logger.info(f"开始聚合特征结果，源目录: {outputs_dir}")

    if not os.path.exists(outputs_dir):
        logger.error(f"输出目录不存在: {outputs_dir}")
        return

    # 1. 扫描所有的 selected_features.csv
    feature_sets: Dict[str, List[str]] = {}
    
    # 遍历 outputs 下的一级子目录
    for folder_name in os.listdir(outputs_dir):
        folder_path = os.path.join(outputs_dir, folder_name)
        if not os.path.isdir(folder_path) or folder_name == 'summary': 
            continue
            
        # 尝试从文件夹名中解析 dataset_id (假设格式: {dataset_id}_mode...)
        # 这里简单起见，直接用文件夹名作为 ID，或者你可以 split('_')[0]
        # 建议直接用文件夹名以保证唯一性
        dataset_key = folder_name 
        
        csv_path = os.path.join(folder_path, "selected_features.csv")
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                if 'feature' in df.columns:
                    features = df['feature'].tolist()
                    feature_sets[dataset_key] = features
                    logger.info(f"已加载 {dataset_key}: {len(features)} 个特征")
                else:
                    logger.warning(f"{dataset_key} 的 CSV 缺少 'feature' 列")
            except Exception as e:
                logger.error(f"读取 {csv_path} 失败: {e}")
    
    if not feature_sets:
        logger.warning("未找到任何有效的特征结果文件。")
        return

    # 2. 构建聚合矩阵 (One-Hot 风格)
    # 获取所有出现过的特征的并集
    all_features = sorted(list(set(f for feats in feature_sets.values() for f in feats)))
    dataset_ids = sorted(list(feature_sets.keys()))
    
    logger.info(f"共发现 {len(dataset_ids)} 个结果集，涉及 {len(all_features)} 个唯一特征。")
    
    # 初始化 DataFrame: Index=Features, Columns=Datasets
    summary_df = pd.DataFrame(0, index=all_features, columns=dataset_ids)
    
    for ds_id, feats in feature_sets.items():
        summary_df.loc[feats, ds_id] = 1
        
    # 3. 计算统计量
    # 特征频率 (Frequency): 被多少个数据集选中
    feature_counts = summary_df.sum(axis=1).sort_values(ascending=False)
    summary_df['Frequency'] = feature_counts # 添加一列作为汇总
    
    # 4. 保存结果
    if not os.path.exists(summary_dir):
        os.makedirs(summary_dir)
        
    csv_save_path = os.path.join(summary_dir, "aggregated_feature_matrix.csv")
    summary_df.to_csv(csv_save_path)
    logger.info(f"聚合矩阵已保存至: {csv_save_path}")
    
    # 导出高频特征列表
    top_save_path = os.path.join(summary_dir, "top_robust_features.csv")
    feature_counts.to_csv(top_save_path, header=['count'])
    
    # 5. 绘图
    # 5.1 Top-N 频率柱状图
    plot_feature_frequency(feature_counts, summary_dir, top_n=30)
    
    # 5.2 特征选择热力图 (UpSet 替代品)
    # 绘图时去掉 'Frequency' 列，只保留 0/1 矩阵
    plot_matrix = summary_df.drop(columns=['Frequency'])
    plot_selection_heatmap(plot_matrix, summary_dir)

    logger.info("聚合与绘图任务完成。")

if __name__ == "__main__":
    setup_logger()
    
    parser = argparse.ArgumentParser(description="Aggregate Feature Selection Results")
    parser.add_argument('--outputs_dir', type=str, default='outputs/', help='特征筛选结果的根目录')
    parser.add_argument('--summary_dir', type=str, default='outputs/summary/', help='聚合结果保存目录')
    
    args = parser.parse_args()
    
    aggregate_features(args.outputs_dir, args.summary_dir)
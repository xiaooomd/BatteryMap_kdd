import os
import pandas as pd
import logging
import sys

# 添加项目根目录到路径
sys.path.append(os.getcwd())

from modules.data_processor.loader import DataLoader
from modules.data_processor.cleaner import SingleBatteryCleaner

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()

INPUT_RESULT_DIR = "data/result"
INPUT_UL_PUR_DIR = "data/result/UL_PUR"
# 排除列表 (与 li_selected_results 保持一致，但 NA/ZNion 也不算)
EXCLUDED_DATASETS = {'NA', 'ZNion', 'CALB', 'CALB1', 'CALB2',
                     'SNL_LFP', 'SNL_NCA', 'SNL_NMC',
                     'Tongji_NCA', 'Tongji_NCA_NMC', 'Tongji_NMC'}

# DataLoader 配置
# 这里的 nrows 设小一点，只为了看列名
LOADER_NROWS = 10
EARLY_CYCLE_THRESHOLD = 0 # 不过滤电池，只看特征

def get_dataset_columns(dataset_name, folder_path):
    # 尝试读取第一个有效 CSV 的列名
    # 对于 data/result 下的数据，需要先经过 SingleBatteryCleaner 的列过滤逻辑吗？
    # 理论上 SingleBatteryCleaner 主要是 smooth 和 relative，不会删除列，除非全 NaN
    # 但是 DatasetCleaner 会删列。我们这里看“原始可获取的特征交集”。

    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    if not csv_files:
        return None

    # 尝试读取几个文件，防止第一个文件恰好有问题
    for f in csv_files[:5]:
        path = os.path.join(folder_path, f)
        try:
            df = pd.read_csv(path, nrows=LOADER_NROWS)
            if df.empty:
                continue

            # 模拟 SingleBatteryCleaner 的列处理 (如果有的话)
            # 实际上 SingleBatteryCleaner 会生成 relative feature 吗？
            # 检查 cleaner 代码 -> process(df) -> 主要是 impute outliers，然后 relative = df - df.iloc[9]
            # 所以列名保持不变。

            return set(df.columns.tolist())
        except Exception:
            continue
    return None

def main():
    logger.info("Checking feature intersection from data/result (Raw Data)...")

    dataset_features = {}

    # 1. Check data/result folders
    if os.path.exists(INPUT_RESULT_DIR):
        for d in sorted(os.listdir(INPUT_RESULT_DIR)):
            if d in EXCLUDED_DATASETS or d == "UL_PUR":
                continue

            path = os.path.join(INPUT_RESULT_DIR, d)
            if not os.path.isdir(path):
                continue

            cols = get_dataset_columns(d, path)
            if cols:
                dataset_features[d] = cols
            else:
                logger.warning(f"Could not extract columns for {d}")

    # 2. Check UL_PUR
    if os.path.exists(INPUT_UL_PUR_DIR):
        cols = get_dataset_columns("UL_PUR", INPUT_UL_PUR_DIR)
        if cols:
            dataset_features["UL_PUR"] = cols

    if not dataset_features:
        logger.error("No valid datasets found.")
        return

    # 3. Intersection
    common_features = set.intersection(*dataset_features.values())

    logger.info("-" * 50)
    logger.info(f"Datasets Involved ({len(dataset_features)}): {list(dataset_features.keys())}")
    logger.info("-" * 50)
    logger.info(f"Intersection Size: {len(common_features)}")
    logger.info(f"Intersection Features: {sorted(list(common_features))}")
    logger.info("-" * 50)

    # 4. Analysis of Missing
    # 取一个并集作为参考
    all_features = set.union(*dataset_features.values())
    feature_counts = {}
    for f in all_features:
        count = sum(1 for cols in dataset_features.values() if f in cols)
        feature_counts[f] = count

    logger.info("Feature Frequency in Raw Data:")
    sorted_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)

    # 只显示 top 40 常见特征
    for f, count in sorted_features[:40]:
        status = "✅ All" if count == len(dataset_features) else f"❌ Missing in {len(dataset_features) - count}"
        if count < len(dataset_features):
             missing_ds = [k for k, v in dataset_features.items() if f not in v]
             logger.info(f"{f:<25} {count}/{len(dataset_features)} {missing_ds}")
        else:
             logger.info(f"{f:<25} {count}/{len(dataset_features)}")

if __name__ == "__main__":
    main()

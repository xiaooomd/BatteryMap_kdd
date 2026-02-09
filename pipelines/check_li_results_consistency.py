import os
import pandas as pd
import logging
from collections import Counter

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()

INPUT_DIR = "data/li_results"

def check_consistency():
    if not os.path.exists(INPUT_DIR):
        logger.error(f"Directory not found: {INPUT_DIR}")
        return

    csv_files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith('.csv')])
    if not csv_files:
        logger.error("No CSV files found.")
        return

    logger.info(f"Checking {len(csv_files)} files in {INPUT_DIR}...")

    # 记录列名指纹 (tuple of columns) -> list of files
    schema_map = {}

    for f in csv_files:
        file_path = os.path.join(INPUT_DIR, f)
        try:
            # 只读 header
            df = pd.read_csv(file_path, nrows=0)
            cols = tuple(df.columns.tolist())

            if cols not in schema_map:
                schema_map[cols] = []
            schema_map[cols].append(f)

        except Exception as e:
            logger.error(f"Error reading {f}: {e}")

    # 输出结果
    logger.info("=" * 60)
    logger.info(f"Unique Schemas Found: {len(schema_map)}")
    logger.info("=" * 60)

    for i, (cols, files) in enumerate(schema_map.items(), 1):
        logger.info(f"Schema #{i} (Count: {len(files)} files)")
        logger.info(f"Columns ({len(cols)}): {list(cols)}")
        if len(files) < 10:
            logger.info(f"Files: {files}")
        else:
            logger.info(f"Files (first 5): {files[:5]} ...")
        logger.info("-" * 60)

    if len(schema_map) == 1:
        logger.info("✅ SUCCESS: All files have identical column count and order.")
    else:
        logger.info("❌ FAILURE: Inconsistent schemas detected.")

if __name__ == "__main__":
    check_consistency()

import os
import re
import glob
import logging
import pandas as pd
from typing import List, Dict, Tuple, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("RobustFeatureAnalyzer")

# Define Thermodynamics Features (Exact Match)
THERMO_EXACT = {
    'dtp', 'dtpl_v', 'mat_charge', 'mat_discharge', 'met_charge', 'met_discharge',
    'mint_charge', 'mint_discharge', 't_rise_charge', 't_rise_discharge',
    'thermal_load_charge', 'thermal_load_discharge', 'skew_t_discharge',
    'temperature', 'heatrate'
}

# Define Thermodynamics Keywords (Partial Match)
THERMO_KEYWORDS = ['temp', 'heat', 't_rise']

class ReportParser:
    """Parses Markdown feature engineering reports to extract SHAP selection tables and metadata."""

    def __init__(self, feature_dir: str = "feature_eng/single_report"):
        self.feature_dir = feature_dir

    def get_report_files(self) -> List[str]:
        """Returns all markdown report files in the directory."""
        search_path = os.path.join(self.feature_dir, "*_report.md")
        files = glob.glob(search_path)
        return files

    def parse_file(self, file_path: str) -> Dict[str, Any]:
        """
        Parses a single report file to extract top features and check for thermodynamics.
        Returns: {'features': List[Dict], 'has_thermo': bool}
        """
        result = {'features': [], 'has_thermo': False}
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 1. Check for Thermodynamics Group
            # Look for "### Thermodynamics" within the grouping section
            # Simple check: Is "### Thermodynamics" present in the file?
            # More robust: Check if it's under "## 2. Feature Grouping"
            grouping_match = re.search(r'## 2\. Feature Grouping', content)
            if grouping_match:
                # Search after the grouping header
                post_grouping = content[grouping_match.start():]
                # Stop search at next main header "## 3."
                next_header = re.search(r'## 3\.', post_grouping)
                search_scope = post_grouping[:next_header.start()] if next_header else post_grouping

                if "### Thermodynamics" in search_scope:
                    # Check if it has items (lines starting with "- ")
                    # Split by "### Thermodynamics" and look at the part after it
                    parts = search_scope.split("### Thermodynamics")
                    if len(parts) > 1:
                        thermo_section = parts[1]
                        # Check next lines until empty line or next subheader
                        lines = thermo_section.strip().split('\n')
                        # Check if any line looks like a list item "- FeatureName"
                        for line in lines:
                            line = line.strip()
                            if not line:
                                continue
                            if line.startswith("###"): # Next group
                                break
                            if line.startswith("-"):
                                result['has_thermo'] = True
                                break

            # 2. Extract SHAP features
            section_match = re.search(r'## 4\. SHAP Feature Selection', content)
            if not section_match:
                logger.warning(f"SHAP section not found in {file_path}")
                return result

            lines = content[section_match.start():].split('\n')
            table_start = False

            extracted_features = []
            has_zero_importance = False

            for line in lines:
                if "| Rank | Feature | Importance |" in line:
                    table_start = True
                    continue
                if not table_start:
                    continue
                if "| --- |" in line:
                    continue
                if not line.strip().startswith("|"):
                    if table_start and line.strip() == "":
                        break
                    continue

                match = re.search(r'\|\s*(\d+)\s*\|\s*([^|]+)\s*\|\s*([\d\.]+)\s*\|', line)
                if match:
                    rank = int(match.group(1))
                    feature_name = match.group(2).strip()
                    importance = float(match.group(3))

                    if importance == 0.0:
                        has_zero_importance = True

                    extracted_features.append({
                        'feature': feature_name,
                        'rank': rank,
                        'importance': importance
                    })

            if has_zero_importance:
                logger.warning(f"Dataset {file_path} contains features with 0.0 importance. Excluding entire dataset from analysis.")
                result['features'] = []
            else:
                result['features'] = extracted_features

        except Exception as e:
            logger.error(f"Error parsing {file_path}: {e}")

        return result

class RobustnessAnalyzer:
    """Analyzes feature frequency and importance across datasets."""

    def is_thermo_feature(self, feature_name: str) -> bool:
        """Checks if a feature is thermodynamic based on name."""
        f_lower = feature_name.lower()
        if f_lower in THERMO_EXACT:
            return True
        for kw in THERMO_KEYWORDS:
            if kw in f_lower:
                return True
        return False

    def normalize_feature_name(self, feature_name: str) -> str:
        """Normalizes feature names by grouping specific variants."""
        # Rules:
        # charge_slope_1/2/3 -> charge_slope
        # discharge_slope_1/2/3 -> discharge_slope
        # TEVI_1/2/3 -> TEVI
        # TEVD_1/2/3 -> TEVD
        patterns = [
            (r'^charge_slope_[123]$', 'charge_slope'),
            (r'^discharge_slope_[123]$', 'discharge_slope'),
            (r'^TEVI_[123]$', 'TEVI'),
            (r'^TEVD_[123]$', 'TEVD')
        ]
        for pat, replacement in patterns:
            if re.match(pat, feature_name, re.IGNORECASE):
                return replacement
        return feature_name

    def analyze(self, dataset_feature_map: Dict[str, List[Dict]]) -> pd.DataFrame:
        """
        Aggregates feature statistics.
        Input: {'dataset_id': [{'feature': 'a', 'rank': 1, 'importance': 0.5}, ...]}
        """
        all_rows = []
        for ds_id, feats in dataset_feature_map.items():
            for f in feats:
                all_rows.append({
                    'dataset': ds_id,
                    'feature': f['feature'],
                    'rank': f['rank'],
                    'importance': f['importance']
                })

        if not all_rows:
            return pd.DataFrame()

        df = pd.DataFrame(all_rows)
        return self._calculate_stats(df, len(dataset_feature_map))

    def analyze_grouped(self, dataset_feature_map: Dict[str, List[Dict]]) -> pd.DataFrame:
        """
        Aggregates feature statistics with grouping and deduplication per dataset.
        """
        all_rows = []
        for ds_id, feats in dataset_feature_map.items():
            # Deduplicate per dataset
            # Map normalized_name -> best feature entry (highest importance)
            seen_normalized = {} # type: Dict[str, Dict]

            for f in feats:
                norm_name = self.normalize_feature_name(f['feature'])

                # If not seen, add it
                if norm_name not in seen_normalized:
                    seen_normalized[norm_name] = {
                        'feature': norm_name,
                        'rank': f['rank'],
                        'importance': f['importance']
                    }
                else:
                    # If seen, keep the one with higher importance
                    if f['importance'] > seen_normalized[norm_name]['importance']:
                        seen_normalized[norm_name]['importance'] = f['importance']
                        seen_normalized[norm_name]['rank'] = f['rank']
                    # Keep the minimum rank among the group (best rank)
                    seen_normalized[norm_name]['rank'] = min(seen_normalized[norm_name]['rank'], f['rank'])

            for norm_f in seen_normalized.values():
                all_rows.append({
                    'dataset': ds_id,
                    'feature': norm_f['feature'],
                    'rank': norm_f['rank'],
                    'importance': norm_f['importance']
                })

        if not all_rows:
            return pd.DataFrame()

        df = pd.DataFrame(all_rows)
        return self._calculate_stats(df, len(dataset_feature_map))

    def _calculate_stats(self, df: pd.DataFrame, total_datasets: int) -> pd.DataFrame:
        """Helper to calculate statistics from a DataFrame of features."""
        stats = df.groupby('feature').agg(
            frequency=('dataset', 'count'),
            avg_rank=('rank', 'mean'),
            avg_importance=('importance', 'mean'),
            datasets=('dataset', lambda x: ', '.join(sorted(x)))
        ).reset_index()

        stats['occurrence_rate'] = stats['frequency'] / total_datasets

        stats = stats.sort_values(
            by=['frequency', 'avg_rank', 'avg_importance'],
            ascending=[False, True, False]
        )

        return stats

    def run_analysis_suite(self, dataset_map: Dict[str, Dict[str, Any]]) -> Dict[str, pd.DataFrame]:
        """
        Runs three analyses: Full, No-Temp, Temp-Only.
        dataset_map: {ds_id: {'features': [...], 'has_thermo': bool}}
        """
        # 1. Full Analysis
        # Convert to simple map for analyze()
        map_full = {k: v['features'] for k, v in dataset_map.items()}
        df_full = self.analyze(map_full)

        # 2. No-Temp Analysis (Exclude Thermo Features from All Datasets)
        map_no_temp = {}
        for ds_id, data in dataset_map.items():
            filtered_feats = [f for f in data['features'] if not self.is_thermo_feature(f['feature'])]
            if filtered_feats:
                map_no_temp[ds_id] = filtered_feats

        df_no_temp = self.analyze(map_no_temp)

        # 3. Temp-Datasets Only (Include All Features, but only from Datasets with Thermo Group)
        map_temp_datasets = {}
        for ds_id, data in dataset_map.items():
            if data['has_thermo']:
                map_temp_datasets[ds_id] = data['features']

        df_temp_only = self.analyze(map_temp_datasets)

        return {
            'full': df_full,
            'no_temp': df_no_temp,
            'temp_datasets_only': df_temp_only
        }

    def run_grouped_analysis_suite(self, dataset_map: Dict[str, Dict[str, Any]]) -> Dict[str, pd.DataFrame]:
        """
        Runs grouped analyses: Full, No-Temp, Temp-Only.
        Uses analyze_grouped instead of analyze.
        """
        # 1. Full Analysis
        map_full = {k: v['features'] for k, v in dataset_map.items()}
        df_full = self.analyze_grouped(map_full)

        # 2. No-Temp Analysis
        map_no_temp = {}
        for ds_id, data in dataset_map.items():
            filtered_feats = [f for f in data['features'] if not self.is_thermo_feature(f['feature'])]
            if filtered_feats:
                map_no_temp[ds_id] = filtered_feats

        df_no_temp = self.analyze_grouped(map_no_temp)

        # 3. Temp-Datasets Only
        map_temp_datasets = {}
        for ds_id, data in dataset_map.items():
            if data['has_thermo']:
                map_temp_datasets[ds_id] = data['features']

        df_temp_only = self.analyze_grouped(map_temp_datasets)

        return {
            'full': df_full,
            'no_temp': df_no_temp,
            'temp_datasets_only': df_temp_only
        }

class SummaryGenerator:
    """Generates Markdown summary reports."""

    def generate_markdown(self, analysis_results: Dict[str, pd.DataFrame], scenario_name: str, output_path: str):
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(f"# Robust Feature Analysis Report\n\n")
                f.write(f"**Scenario**: {scenario_name}\n\n")

                # Section 1: Full Feature Set
                self._write_section(f, "1. Full Feature Set (All Datasets)", analysis_results['full'])

                # Section 2: Temperature-Excluded
                self._write_section(f, "2. Temperature-Excluded Analysis (Simulated No-Temp Sensors)", analysis_results['no_temp'])

                # Section 3: Temp-Datasets Only
                self._write_section(f, "3. Temperature-Datasets Only (Datasets with Temp Features)", analysis_results['temp_datasets_only'])

            logger.info(f"Report generated: {output_path}")

        except Exception as e:
            logger.error(f"Error generating summary {output_path}: {e}")

    def _write_section(self, f, title, df):
        f.write(f"## {title}\n\n")

        if df.empty:
            f.write("No data available for this analysis.\n\n")
            return

        total_datasets = 0
        if 'occurrence_rate' in df.columns and 'frequency' in df.columns:
             # Backward calculation of total datasets based on first row
             # frequency = total * rate => total = freq / rate
             first_row = df.iloc[0]
             if first_row['occurrence_rate'] > 0:
                 total_datasets = int(round(first_row['frequency'] / first_row['occurrence_rate']))

        f.write(f"**Total Datasets in Scope**: {total_datasets}\n\n")
        f.write("| Rank | Feature | Frequency | Occurrence Rate | Avg Rank | Avg Importance | Source Datasets |\n")
        f.write("| --- | --- | --- | --- | --- | --- | --- |\n")

        for idx, row in df.iterrows():
            occ_rate = f"{row['occurrence_rate'] * 100:.1f}%"
            imp = f"{row['avg_importance']:.4f}"
            rank = f"{row['avg_rank']:.1f}"
            f.write(f"| {idx + 1} | {row['feature']} | {row['frequency']} | {occ_rate} | {rank} | {imp} | {row['datasets']} |\n")
        f.write("\n")

def main():
    # [Modified] 默认扫描目录为 feature_eng/single_report
    feature_dir = "feature_eng/single_report"
    if not os.path.exists(feature_dir):
        # 尝试 fallback 到 feature_eng (兼容性)
        if os.path.exists("feature_eng"):
             # 检查是否存在 _report.md
             if glob.glob("feature_eng/*_report.md"):
                 feature_dir = "feature_eng"
                 logger.warning(f"目录 'feature_eng/single_report' 不存在，回退到 '{feature_dir}'")
             else:
                 logger.error(f"目录 {feature_dir} 不存在。")
                 return
        else:
            logger.error(f"目录 {feature_dir} 不存在。")
            return

    parser = ReportParser(feature_dir)
    analyzer = RobustnessAnalyzer()
    generator = SummaryGenerator()

    # Get all reports
    all_files = parser.get_report_files()
    logger.info(f"Found {len(all_files)} report files.")

    def get_id(path):
        filename = os.path.basename(path)
        return filename.replace("_report.md", "")

    # 1. Define Logic Sets
    GLOBAL_EXCLUDES = {
        "Stanford_2",
        "CALB_2024_HT", "CALB_2024_LT",
        "CALB_42_HT", "CALB_42_LT",
        "CALB1", "CALB2"
    }

    # Datasets that represent the AGGREGATED view (Parent)
    AGGREGATED_TARGETS = {"SNL", "Tongji"}

    # Datasets that represent the SPLIT view (Children)
    SPLIT_TARGETS = {
        "SNL_LFP", "SNL_NCA", "SNL_NMC",
        "Tongji_NCA", "Tongji_NMC", "Tongji_NCA_NMC"
    }

    # 2. Global Pre-Filtering
    # Filter out any files in GLOBAL_EXCLUDES
    valid_file_map = {}
    for f in all_files:
        ds_id = get_id(f)
        if ds_id in GLOBAL_EXCLUDES:
            logger.info(f"Globally excluding dataset: {ds_id}")
            continue
        valid_file_map[ds_id] = f

    # 3. Construct Scenario Maps
    dataset_map_a = {} # Aggregated Scenario
    dataset_map_b = {} # Split Scenario

    for ds_id, fpath in valid_file_map.items():
        # Parse once
        data = parser.parse_file(fpath)
        if not data['features']:
            continue

        # Logic for Scenario A (Aggregated)
        # Rule: Include if NOT a child split dataset
        if ds_id not in SPLIT_TARGETS:
            dataset_map_a[ds_id] = data

        # Logic for Scenario B (Split)
        # Rule: Include if NOT a parent aggregated dataset
        if ds_id not in AGGREGATED_TARGETS:
            dataset_map_b[ds_id] = data

    # --- Scenario A: Aggregated ---
    logger.info("--- Processing Scenario A: Aggregated (SNL/Tongji Parent Only) ---")
    results_a = analyzer.run_analysis_suite(dataset_map_a)
    output_a = os.path.join("feature_eng/analyse_report", "Robust_Features_Analysis_Aggregated.md")
    os.makedirs(os.path.dirname(output_a), exist_ok=True)
    generator.generate_markdown(results_a, "Aggregated (Parents Only)", output_a)

    # --- Scenario A: Aggregated (Grouped) ---
    logger.info("--- Processing Scenario A: Aggregated (Grouped) ---")
    results_a_grouped = analyzer.run_grouped_analysis_suite(dataset_map_a)
    output_a_grouped = os.path.join("feature_eng/analyse_report", "Robust_Features_Analysis_Aggregated_Grouped.md")
    generator.generate_markdown(results_a_grouped, "Aggregated (Parents Only, Grouped)", output_a_grouped)

    # --- Scenario B: Split ---
    logger.info("--- Processing Scenario B: Split (SNL/Tongji Children Only) ---")
    results_b = analyzer.run_analysis_suite(dataset_map_b)
    output_b = os.path.join("feature_eng/analyse_report", "Robust_Features_Analysis_Split.md")
    generator.generate_markdown(results_b, "Split (Children Only)", output_b)

    # --- Scenario B: Split (Grouped) ---
    logger.info("--- Processing Scenario B: Split (Grouped) ---")
    results_b_grouped = analyzer.run_grouped_analysis_suite(dataset_map_b)
    output_b_grouped = os.path.join("feature_eng/analyse_report", "Robust_Features_Analysis_Split_Grouped.md")
    generator.generate_markdown(results_b_grouped, "Split (Children Only, Grouped)", output_b_grouped)


if __name__ == "__main__":
    main()
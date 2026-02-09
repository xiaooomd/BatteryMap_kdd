import os
import glob
import re
import sys
import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Set, Any
from collections import Counter

# Add project root to sys.path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.feature_selector.feature_grouper import FeatureGrouper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("LiFeatureAnalyzer")

# Battery Chemistry Mapping
CHEMISTRY_MAP = {
    'LCO': ['CALCE'],
    'LFP': ['MATR', 'HUST', 'SNL_LFP'],
    'NMC': ['RWTH', 'CALB', 'ISU_ILCC', 'XJTU', 'Stanford', 'MICH', 'MICH_EXP', 'SNL_NMC', 'Tongji_NMC'],
    'NCA': ['SNL_NCA', 'UL_PUR', 'Tongji_NCA'],
    'NCA+NMC': ['Tongji_NCA_NMC'],
    # 'NMC_LCO': ['HNEI'] # Not requested in the specific list, will handle separately or include in All Li
}

# HNEI is NMC_LCO, treating as "Other" for subgroup analysis but include in All Li
OTHER_LI_DATASETS = ['HNEI']

NON_LI_DATASETS = ['Na-ion', 'Zn-ion']

class ReportParser:
    """Parser to extract dataset features from Multi_Seed_Global_Report.md."""

    def __init__(self, report_path: str):
        self.report_path = report_path

    def parse_all_datasets(self) -> Dict[str, List[Dict[str, Any]]]:
        """Parses the global report to extract features for each dataset."""
        dataset_features = {}
        if not os.path.exists(self.report_path):
            logger.error(f"Global report not found: {self.report_path}")
            return {}

        try:
            with open(self.report_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Find Section 2
            section_match = re.search(r'## 2\. Per-Dataset Robustness Details', content)
            if not section_match:
                logger.warning("Section 2 not found in global report.")
                return {}

            section_content = content[section_match.start():]
            # Split by dataset headers
            dataset_sections = re.split(r'###\s+([\w\-\.]+)', section_content)

            # re.split with capturing group returns [prefix, group1, content1, group2, content2, ...]
            for i in range(1, len(dataset_sections), 2):
                dataset_id = dataset_sections[i].strip()
                table_content = dataset_sections[i+1]

                features = []
                lines = table_content.strip().split('\n')
                table_start = False

                for line in lines:
                    if "| Rank | Feature |" in line:
                        table_start = True
                        continue
                    if not table_start or "| --- |" in line or not line.strip().startswith("|"):
                        continue

                    # | Rank | Feature | Selection Rate | Avg Rank | Avg Importance |
                    m = re.search(r'\|\s*\d+\s*\|\s*([^|]+)\s*\|\s*[\d\.]+%?\s*\|\s*[\d\.]+\s*\|\s*([\d\.]+)\s*\|', line)
                    if m:
                        features.append({
                            'feature': m.group(1).strip(),
                            'importance': float(m.group(2))
                        })

                if features:
                    dataset_features[dataset_id] = features

        except Exception as e:
            logger.error(f"Error parsing global report: {e}")

        return dataset_features

class LiAnalyzer:
    def __init__(self, report_path: str):
        self.parser = ReportParser(report_path)
        self.grouper = FeatureGrouper()

        # Load all data
        self.dataset_features = {} # {dataset_id: [feature_dict]}
        self._load_data()

    def _load_data(self):
        all_data = self.parser.parse_all_datasets()
        EXCLUDE_DATASETS = ["Stanford_2", "Na-ion", "Zn-ion", "Naion", "ZNion"]

        for dataset_id, feats in all_data.items():
            if any(ex in dataset_id for ex in EXCLUDE_DATASETS):
                continue
            self.dataset_features[dataset_id] = feats

    def get_datasets_by_chemistry(self, chem_type: str) -> List[str]:
        target_datasets = CHEMISTRY_MAP.get(chem_type, [])
        found = []
        for d in target_datasets:
            if d in self.dataset_features:
                found.append(d)
        return found

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

    def analyze_group(self, datasets: List[str], normalized: bool = True):
        """
        Analyzes a group of datasets.
        Args:
            datasets: List of dataset IDs.
            normalized: If True, normalizes feature names and deduplicates within dataset.
        """
        if not datasets:
            return None

        all_feature_lists = []
        feature_counts = Counter()
        feature_importance = {} # feature -> list of importances

        for d in datasets:
            feats = self.dataset_features[d]

            if normalized:
                # Deduplicate within dataset
                # Map normalized_name -> max_importance
                dataset_unique_feats = {} # type: Dict[str, float]

                for f in feats:
                    norm_name = self.normalize_feature_name(f['feature'])
                    imp = f['importance']

                    if norm_name not in dataset_unique_feats:
                        dataset_unique_feats[norm_name] = imp
                    else:
                        # Keep the highest importance for this dataset
                        dataset_unique_feats[norm_name] = max(dataset_unique_feats[norm_name], imp)

                # Use normalized unique features
                processed_feats = dataset_unique_feats
            else:
                # Use raw features
                processed_feats = {f['feature']: f['importance'] for f in feats}

            # Add to global stats
            all_feature_lists.append(set(processed_feats.keys()))

            for feat_name, imp in processed_feats.items():
                feature_counts[feat_name] += 1
                if feat_name not in feature_importance:
                    feature_importance[feat_name] = []
                feature_importance[feat_name].append(imp)

        # 1. Intersection
        intersection = set.intersection(*all_feature_lists) if all_feature_lists else set()

        # 2. Frequent Features (Top 30 by occurrence rate)
        # Calculate occurrence rate
        n_datasets = len(datasets)
        stats = []
        for feat, count in feature_counts.items():
            avg_imp = np.mean(feature_importance[feat])
            stats.append({
                'feature': feat,
                'count': count,
                'rate': count / n_datasets,
                'avg_importance': avg_imp
            })

        df_stats = pd.DataFrame(stats).sort_values(by=['count', 'avg_importance'], ascending=[False, False])

        # 3. Type Distribution (based on Top Frequent, e.g., appearing in > 50% datasets)
        # If dataset count is small (e.g. 1), take top 10
        if n_datasets == 1:
            top_features = df_stats.head(15)['feature'].tolist()
        else:
            # Take features appearing in at least 50% of datasets
            top_features = df_stats[df_stats['rate'] >= 0.5]['feature'].tolist()
            if not top_features: # Fallback
                top_features = df_stats.head(15)['feature'].tolist()

        # Group these top features
        grouped = self.grouper.group_features(top_features)
        # Flatten grouped to counts
        type_counts = {k: len(v) for k, v in grouped.items()}

        return {
            'intersection': list(intersection),
            'stats': df_stats,
            'type_distribution': type_counts,
            'dataset_count': n_datasets,
            'dataset_names': sorted(datasets)
        }

    def run(self):
        results = {'normalized': {}, 'raw': {}}

        # Define dataset groups
        groups = {}
        # 1. Analyze by Chemistry
        for chem in CHEMISTRY_MAP.keys():
            groups[chem] = self.get_datasets_by_chemistry(chem)

        # 2. Analyze All Li
        all_li_datasets = []
        for d in self.dataset_features.keys():
            # Include mapped ones + HNEI
            is_mapped = any(d in v for v in CHEMISTRY_MAP.values())
            is_other = d in OTHER_LI_DATASETS
            if is_mapped or is_other:
                all_li_datasets.append(d)
        groups['All_Li'] = all_li_datasets

        # Run analysis for both modes
        for group_name, datasets in groups.items():
            results['normalized'][group_name] = self.analyze_group(datasets, normalized=True)
            results['raw'][group_name] = self.analyze_group(datasets, normalized=False)

        return results

class ReportGenerator:
    def __init__(self, output_path: str):
        self.output_path = output_path
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    def generate(self, results: Dict[str, Dict[str, Any]]):
        with open(self.output_path, 'w', encoding='utf-8') as f:
            f.write("# Lithium Battery Feature Analysis Report\n\n")
            f.write("This report contains two sections:\n")
            f.write("1. **Normalized Analysis**: Features like `charge_slope_1/2/3` are grouped into `charge_slope`.\n")
            f.write("2. **Raw Analysis**: Original feature names are preserved.\n\n")

            # Part 1: Normalized
            f.write("# Part 1: Normalized Analysis (Grouped & Deduplicated)\n")
            f.write("In this section, feature variants (e.g., `charge_slope_1`, `charge_slope_2`) are merged into a single category. ")
            f.write("If a dataset contains multiple variants, it counts only once, keeping the highest importance score.\n\n")
            self._write_full_analysis(f, results['normalized'])

            f.write("\n---\n\n")

            # Part 2: Raw
            f.write("# Part 2: Raw Analysis (Original Feature Names)\n")
            f.write("In this section, all feature names are treated as distinct.\n\n")
            self._write_full_analysis(f, results['raw'])

            logger.info(f"Report written to {self.output_path}")

    def _write_full_analysis(self, f, group_results):
        # Summary of All Li
        all_res = group_results.get('All_Li')
        if all_res:
            self._write_section(f, "All Lithium Batteries", all_res)

        # Subgroups
        for chem in CHEMISTRY_MAP.keys():
            res = group_results.get(chem)
            if res:
                f.write(f"---\n")
                self._write_section(f, f"{chem} Batteries", res)
            else:
                f.write(f"## {chem} Batteries\n")
                f.write("No matching datasets found or data insufficient.\n\n")

    def _write_section(self, f, title, data):
        f.write(f"## {title} (Datasets: {data['dataset_count']})\n\n")

        ds_names = ", ".join(data['dataset_names'])
        f.write(f"**Included Datasets**: [{ds_names}]\n\n")

        # 1. Intersection
        f.write("### 1. Common Features (Intersection)\n")
        inter = data['intersection']
        if inter:
            f.write(f"Found {len(inter)} common features across all datasets in this group:\n")
            for feat in sorted(inter):
                f.write(f"- {feat}\n")
        else:
            f.write("No single feature is common to ALL datasets in this group.\n")
        f.write("\n")

        # 2. Type Distribution
        f.write("### 2. Feature Type Distribution (High Frequency Features)\n")
        f.write("Distribution of features appearing in >= 50% of datasets (or Top 15 if single):\n\n")

        types = data['type_distribution']
        if types:
            total = sum(types.values())
            for k, v in types.items():
                pct = (v / total) * 100
                f.write(f"- **{k}**: {v} ({pct:.1f}%)\n")
        else:
            f.write("No high frequency features found.\n")
        f.write("\n")

        # 3. Top Features Table
        f.write("### 3. Top Robust Features\n")
        df = data['stats'].head(20)
        f.write("| Rank | Feature | Count | Rate | Avg Importance |\n")
        f.write("| --- | --- | --- | --- | --- |\n")
        for idx, row in df.iterrows():
            rate_str = f"{row['rate']*100:.0f}%"
            f.write(f"| {idx+1} | {row['feature']} | {row['count']} | {rate_str} | {row['avg_importance']:.4f} |\n")
        f.write("\n")

def main():
    report_path = "feature_eng/Multi_Seed_Global_Report.md"
    output_path = "feature_eng/analyse_report/Li_Battery_Report.md"

    analyzer = LiAnalyzer(report_path)
    results = analyzer.run()

    generator = ReportGenerator(output_path)
    generator.generate(results)

if __name__ == "__main__":
    main()

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Add project root to sys.path
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

from modules.data_processor.cleaner import DatasetCleaner, SingleBatteryCleaner
from modules.data_processor.imputer import MissingValueImputer
from modules.data_processor.loader import DataLoader
from modules.data_processor.data_split_recorder import split_recorder
from modules.feature_selector.feature_grouper import FeatureGrouper
from modules.feature_selector.filter_methods import (
    KendallFilter,
    MutualInfoFilter,
    PearsonFilter,
    SpearmanFilter,
)
from modules.feature_selector.wrapper_methods import RFESelector, ShapSelector


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("RunFeatureSelection")

DEFAULT_SKIP_WRAPPER_MIN_TARGETS = 5
SUPPORTED_DATASETS = [
    "CALB",
    "CALCE",
    "HNEI",
    "HUST",
    "ISU-ILCC",
    "MATR",
    "MICH",
    "MICH_EXP",
    "Na",
    "RWTH",
    "SNL",
    "Stanford",
    "Stanford_2",
    "Tongji",
    "UL_PUR",
    "XJTU",
    "ZN-coin",
]


def resolve_target_datasets(dataset_id_arg: str | None) -> List[str]:
    """Resolve and validate dataset targets.

    - If dataset_id_arg is empty, process all supported datasets.
    - If provided, accept comma-separated dataset ids.
    - Any dataset outside SUPPORTED_DATASETS is rejected.
    """
    if not dataset_id_arg:
        return SUPPORTED_DATASETS.copy()

    requested = []
    for token in str(dataset_id_arg).split(","):
        dataset_name = token.strip()
        if dataset_name:
            requested.append(dataset_name)

    invalid = [name for name in requested if name not in SUPPORTED_DATASETS]
    if invalid:
        raise ValueError(
            "Unsupported dataset_id(s): "
            + ", ".join(invalid)
            + ". Allowed values are: "
            + ", ".join(SUPPORTED_DATASETS)
        )

    # Deduplicate while preserving order.
    return list(dict.fromkeys(requested))


def parse_nrows_arg(value: str) -> int | None:
    """Parse nrows argument.

    Unified semantics:
    - None/all/full/0/-1 -> load all cycles
    - positive integer N -> load first N cycles
    """
    normalized = str(value).strip().lower()
    if normalized in {"none", "all", "full"}:
        return None

    parsed = int(normalized)
    if parsed <= 0:
        return None
    return parsed


def get_filter_selector(method_code: str, threshold: float, mode: int = 0):
    method_code = method_code.lower()
    if method_code == "p":
        return PearsonFilter(mode=mode, threshold=threshold)
    if method_code == "s":
        return SpearmanFilter(mode=mode, threshold=threshold)
    if method_code == "k":
        return KendallFilter(mode=mode, threshold=threshold)
    if method_code == "m":
        return MutualInfoFilter(mode=mode, threshold=threshold)
    raise ValueError(f"Unknown filter method code: {method_code}")


def build_wrapper_selector(method_code: str, top_k: int, random_state: int):
    method_code = method_code.lower()
    if method_code == "shap":
        return ShapSelector(top_k=top_k, robust_mode=True, random_state=random_state)
    if method_code == "rfe":
        return RFESelector(top_k=top_k, robust_criterion=True, random_state=random_state)
    raise ValueError(f"Unknown wrapper method code: {method_code}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BatteryMap feature engineering pipeline")

    parser.add_argument(
        "--dataset_id",
        type=str,
        default=None,
        help=(
            "Dataset ID (e.g., CALCE). "
            "Supports comma-separated list like CALCE,HUST. "
            "If omitted, all supported datasets are processed."
        ),
    )
    parser.add_argument("--input_dir", type=str, default="./results/features", help="Directory containing extracted feature CSVs")
    parser.add_argument("--label_dir", type=str, default="./data_provider/labels", help="Directory containing label JSON files")
    parser.add_argument("--output_dir", type=str, default="./results/selected_features", help="Directory to export standardized selected battery CSVs")
    parser.add_argument("--report_dir", type=str, default="./results/feature_reports", help="Directory to save feature engineering reports")
    
    parser.add_argument("--pipeline_mode", type=str, default="full_pipeline", choices=["full_pipeline", "aggregate_legacy"], help="Feature pipeline mode")
    parser.add_argument("--nrows", type=parse_nrows_arg, default=None, help=("Number of early cycles to load per battery. " "Default is all cycles (None). " "Use a positive integer N to load first N cycles. "),)

    parser.add_argument("--filter_mode", type=int, default=0, choices=[0, 1], help="0: grouped/global filtering, 1: feature-vs-target only")
    parser.add_argument("--filter_method", type=str, default="p", choices=["p", "s", "k", "m"], help="p: Pearson, s: Spearman, k: Kendall, m: MutualInfo")
    parser.add_argument("--filter_threshold", type=float, default=0.95, help="Threshold for filter method")
    parser.add_argument("--selector_method", type=str, default="shap", choices=["shap", "rfe"], help="Wrapper selection method")
    parser.add_argument("--top_k", type=int, default=20, help="Number of top features to keep in wrapper selection")
    parser.add_argument("--n_seeds", type=int, default=1, help="Number of wrapper random seeds")

    parser.add_argument("--disable_single_clean", action="store_true", help="Skip single battery cleaning")
    parser.add_argument("--disable_dataset_clean", action="store_true", help="Skip dataset-level cleaning")
    parser.add_argument("--disable_impute", action="store_true", help="Skip missing-value imputation")
    parser.add_argument("--fit_on_full_dataset", action="store_true", help="Fit screening and scaler on all batteries instead of the training split")
    parser.add_argument("--export_scaler", action="store_true", help="Export the fitted StandardScaler with joblib")
    parser.add_argument("--skip_wrapper_if_small_sample", action="store_true", help="Skip wrapper selection if training targets are too sparse")
    parser.add_argument("--skip_heatmap", action="store_true", help="Reserved compatibility flag; heatmap export is not generated in this script")
    return parser.parse_args()


def load_dataset(args: argparse.Namespace) -> Dict[str, Dict[str, object]]:
    loader = DataLoader(args.input_dir, args.label_dir, nrows=args.nrows)
    data_gen = loader.load_dataset_generator([args.dataset_id])
    try:
        _, dataset_data = next(data_gen)
    except StopIteration:
        logger.error("No data found for dataset %s", args.dataset_id)
        raise SystemExit(1)
    return dataset_data


def aggregate_legacy_pipeline(args: argparse.Namespace, dataset_data: Dict[str, Dict[str, object]]) -> int:
    output_dir = Path(args.output_dir) / args.dataset_id
    output_dir.mkdir(parents=True, exist_ok=True)

    X_list: List[pd.DataFrame] = []
    y_list: List[float] = []
    for battery_id, data in dataset_data.items():
        df = data["X"]
        label = data["y"]
        features_mean = df.mean(numeric_only=True).to_frame().T
        features_mean["battery_id"] = battery_id
        X_list.append(features_mean)
        y_list.append(label)

    if not X_list:
        logger.error("No valid samples extracted in aggregate_legacy mode.")
        raise SystemExit(1)

    X_agg = pd.concat(X_list, ignore_index=True)
    y_agg = pd.Series(y_list, name="Cycle_Life")
    X_agg = X_agg.drop(columns=["battery_id"], errors="ignore")
    X_agg = X_agg.loc[:, (X_agg != X_agg.iloc[0]).any()]

    filter_selector = get_filter_selector(args.filter_method, args.filter_threshold, mode=args.filter_mode)
    if args.filter_mode == 0:
        selected_by_filter = filter_selector.select(X_agg, y_agg)
        filter_selector.save_report(str(output_dir), f"{args.filter_method}_drop_report.csv")
        X_filtered = X_agg[selected_by_filter]
    else:
        X_filtered = X_agg

    feature_counts: Dict[str, int] = {}
    importance_records: List[pd.DataFrame] = []
    for seed in range(args.n_seeds):
        selector = build_wrapper_selector(args.selector_method, args.top_k, 42 + seed)
        selected_feats, importance_df = selector.select(X_filtered, y_agg)
        importance_df = importance_df.copy()
        importance_df["seed"] = 42 + seed
        importance_records.append(importance_df)
        for feat in selected_feats:
            feature_counts[feat] = feature_counts.get(feat, 0) + 1

    final_df = pd.DataFrame(list(feature_counts.items()), columns=["feature", "selection_count"])
    if not final_df.empty:
        final_df = final_df.sort_values(by=["selection_count", "feature"], ascending=[False, True])
    final_df.to_csv(output_dir / f"robust_selection_{args.selector_method}.csv", index=False)

    if importance_records:
        combined = pd.concat(importance_records, ignore_index=True)
        combined.to_csv(output_dir / f"{args.selector_method}_importance.csv", index=False)

    logger.info("Legacy aggregate selection complete. Results saved to %s", output_dir)
    return 0


def apply_single_battery_cleaning(
    battery_data_dict: Dict[str, Dict[str, object]],
    disable_single_clean: bool,
) -> Dict[str, pd.DataFrame]:
    cleaner = SingleBatteryCleaner()
    cleaned_map: Dict[str, pd.DataFrame] = {}
    for battery_id, data in battery_data_dict.items():
        df = data["X"]
        df_clean = df.copy() if disable_single_clean else cleaner.process(df)
        if not df_clean.empty:
            cleaned_map[battery_id] = df_clean
    return cleaned_map


def apply_dataset_cleaning(
    dataset_id: str,
    cleaned_map: Dict[str, pd.DataFrame],
    disable_dataset_clean: bool,
) -> Tuple[List[str], Dict[str, List[str]]]:
    if not cleaned_map:
        return [], {}

    X_full = pd.concat(list(cleaned_map.values()), ignore_index=True, sort=False)
    dataset_cleaner = DatasetCleaner()
    manual_drop_cols = ["Workload_Type"] if dataset_id.upper().startswith("CALB") else []

    if disable_dataset_clean:
        X_clean = X_full.drop(columns=manual_drop_cols, errors="ignore")
        dropped_info = {"manual_drop": manual_drop_cols} if manual_drop_cols else {}
    else:
        X_clean = dataset_cleaner.process(X_full, manual_drop_cols=manual_drop_cols)
        dropped_info = dataset_cleaner.dropped_info

    valid_cols = [col for col in X_clean.columns if pd.api.types.is_numeric_dtype(X_clean[col])]
    return valid_cols, dropped_info


def apply_imputation(
    dataset_id: str,
    cleaned_map: Dict[str, pd.DataFrame],
    valid_cols: List[str],
    disable_impute: bool,
) -> Dict[str, pd.DataFrame]:
    cleaned_step2 = {}
    for battery_id, df in cleaned_map.items():
        present_cols = [col for col in valid_cols if col in df.columns]
        cleaned_step2[battery_id] = df[present_cols].copy()

    if disable_impute:
        return cleaned_step2

    imputer = MissingValueImputer()
    return imputer.process(cleaned_step2, dataset_id)


def build_processed_frame(
    dataset_data: Dict[str, Dict[str, object]],
    cleaned_data_map: Dict[str, pd.DataFrame],
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    processed_batteries = []
    for bat_id, df_imp in cleaned_data_map.items():
        label = dataset_data[bat_id]["y"]
        frame = df_imp.copy()
        frame["target"] = label
        frame["battery_id"] = bat_id
        processed_batteries.append(frame)

    if not processed_batteries:
        raise ValueError("No processed batteries available after cleaning and imputation.")

    full_df = pd.concat(processed_batteries, ignore_index=True, sort=False)
    y_full = full_df.pop("target")
    battery_ids = full_df.pop("battery_id")
    X_full = full_df.select_dtypes(include=[np.number]).copy()
    return X_full, y_full, battery_ids


def resolve_training_mask(
    dataset_id: str,
    battery_ids: pd.Series,
    fit_on_full_dataset: bool,
) -> Tuple[pd.Series, Dict[str, object]]:
    split_meta: Dict[str, object] = {"mode": "full_dataset", "used_train_split": False}
    if fit_on_full_dataset:
        return pd.Series(True, index=battery_ids.index), split_meta

    train_list, val_list, test_list = split_recorder.get_split_lists(dataset_id)
    train_ids = {Path(name).stem for name in train_list}
    val_ids = {Path(name).stem for name in val_list}
    test_ids = {Path(name).stem for name in test_list}

    if not train_ids:
        logger.warning("No split record found for %s. Falling back to full dataset fitting.", dataset_id)
        return pd.Series(True, index=battery_ids.index), split_meta

    is_train = battery_ids.isin(train_ids)
    if not is_train.any():
        logger.warning("Split record for %s matched no batteries. Falling back to full dataset fitting.", dataset_id)
        return pd.Series(True, index=battery_ids.index), split_meta

    split_meta = {
        "mode": "train_split",
        "used_train_split": True,
        "train_ids": sorted(train_ids),
        "val_ids": sorted(val_ids),
        "test_ids": sorted(test_ids),
        "matched_train_batteries": sorted(set(battery_ids[is_train].tolist())),
    }
    return is_train, split_meta


def run_filter_pipeline(
    args: argparse.Namespace,
    X_train_screen: pd.DataFrame,
    y_train_screen: pd.Series,
) -> Tuple[List[str], pd.DataFrame]:
    filter_threshold = args.filter_threshold
    if args.filter_method == "p" and filter_threshold == 0.95:
        filter_threshold = 0.98
        logger.info("Pearson filter detected; automatically raising threshold to %.2f", filter_threshold)

    filter_selector = get_filter_selector(args.filter_method, filter_threshold, mode=args.filter_mode)
    drop_reports: List[pd.DataFrame] = []

    if args.filter_mode == 1:
        selected_features = filter_selector.select(X_train_screen, y_train_screen)
        if not filter_selector.drop_report_.empty:
            report = filter_selector.drop_report_.copy()
            report["stage"] = "global_mode1"
            drop_reports.append(report)
        return selected_features, pd.concat(drop_reports, ignore_index=True) if drop_reports else pd.DataFrame()

    grouper = FeatureGrouper()
    grouped_features = grouper.group_features(X_train_screen.columns.tolist())

    stage1_selected: List[str] = []
    stage1_seen = set()
    for group_name, feats in grouped_features.items():
        valid_feats = [feat for feat in feats if feat in X_train_screen.columns]
        if not valid_feats:
            continue
        selected = filter_selector.select(X_train_screen[valid_feats], y_train_screen)
        for feat in selected:
            if feat not in stage1_seen:
                stage1_seen.add(feat)
                stage1_selected.append(feat)
        if not filter_selector.drop_report_.empty:
            report = filter_selector.drop_report_.copy()
            report["stage"] = "stage1"
            report["group_name"] = group_name
            drop_reports.append(report)

    if not stage1_selected:
        return [], pd.concat(drop_reports, ignore_index=True) if drop_reports else pd.DataFrame()

    stage2_selected = filter_selector.select(X_train_screen[stage1_selected], y_train_screen)
    if not filter_selector.drop_report_.empty:
        report = filter_selector.drop_report_.copy()
        report["stage"] = "stage2"
        report["group_name"] = "global"
        drop_reports.append(report)

    drop_report_df = pd.concat(drop_reports, ignore_index=True) if drop_reports else pd.DataFrame()
    return stage2_selected, drop_report_df


def run_wrapper_pipeline(
    args: argparse.Namespace,
    X_filtered_train: pd.DataFrame,
    y_train_screen: pd.Series,
) -> Tuple[List[str], pd.DataFrame, pd.DataFrame, bool]:
    if X_filtered_train.empty:
        return [], pd.DataFrame(), pd.DataFrame(), True

    if args.skip_wrapper_if_small_sample and y_train_screen.nunique() < DEFAULT_SKIP_WRAPPER_MIN_TARGETS:
        importance_df = pd.DataFrame({"feature": X_filtered_train.columns, "importance": np.zeros(len(X_filtered_train.columns))})
        return list(X_filtered_train.columns), importance_df, pd.DataFrame(), True

    if X_filtered_train.shape[1] <= args.top_k:
        importance_df = pd.DataFrame({"feature": X_filtered_train.columns, "importance": np.ones(len(X_filtered_train.columns))})
        return list(X_filtered_train.columns), importance_df, pd.DataFrame(), True

    per_seed_rankings: List[Dict[str, object]] = []
    combined_importances: List[pd.DataFrame] = []
    final_features: List[str] = []
    importance_df = pd.DataFrame()

    for seed_idx in range(args.n_seeds):
        seed = 42 + seed_idx
        selector = build_wrapper_selector(args.selector_method, args.top_k, seed)
        selected_feats, single_importance = selector.select(X_filtered_train, y_train_screen)
        single_importance = single_importance.copy()
        single_importance["seed"] = seed
        combined_importances.append(single_importance)

        ranking = single_importance.reset_index(drop=True)
        ranking["rank"] = ranking.index + 1
        ranking["is_selected"] = ranking["rank"] <= args.top_k
        per_seed_rankings.append(ranking[["feature", "importance", "seed", "rank", "is_selected"]])

        if seed_idx == 0:
            final_features = selected_feats
            importance_df = single_importance.drop(columns=["seed"])

    robustness_df = pd.DataFrame()
    if per_seed_rankings:
        ranking_df = pd.concat(per_seed_rankings, ignore_index=True)
        robustness_df = ranking_df.groupby("feature").agg(
            selection_count=("is_selected", "sum"),
            selection_rate=("is_selected", "mean"),
            avg_rank=("rank", "mean"),
            avg_importance=("importance", "mean"),
            std_importance=("importance", "std"),
        ).reset_index()
        robustness_df = robustness_df.sort_values(by=["selection_rate", "avg_rank"], ascending=[False, True])

        if args.n_seeds > 1:
            top_ranked = robustness_df.head(args.top_k).copy()
            final_features = top_ranked["feature"].tolist()
            merged = top_ranked[["feature", "avg_importance"]].rename(columns={"avg_importance": "importance"})
            remainder = [feat for feat in X_filtered_train.columns if feat not in final_features]
            if remainder:
                remainder_df = robustness_df[robustness_df["feature"].isin(remainder)][["feature", "avg_importance"]]
                remainder_df = remainder_df.rename(columns={"avg_importance": "importance"})
                importance_df = pd.concat([merged, remainder_df], ignore_index=True)
            else:
                importance_df = merged

    wrapper_skipped = False
    return final_features, importance_df, robustness_df, wrapper_skipped


def export_selected_dataset(
    args: argparse.Namespace,
    dataset_id: str,
    cleaned_data_map: Dict[str, pd.DataFrame],
    X_train_screen: pd.DataFrame,
    ordered_columns: List[str],
    output_dir: Path,
) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    scaler = StandardScaler()
    scaler.fit(X_train_screen[ordered_columns])

    for stale_csv in output_dir.glob("*.csv"):
        stale_csv.unlink()

    for battery_id, df in cleaned_data_map.items():
        export_df = df.reindex(columns=ordered_columns).fillna(0.0)
        norm_values = scaler.transform(export_df)
        pd.DataFrame(norm_values, columns=ordered_columns).to_csv(output_dir / f"{battery_id}.csv", index=False)

    if args.export_scaler:
        joblib.dump(scaler, output_dir / f"{dataset_id}_feature_scaler.joblib")

    return {
        "scaler_mean_shape": len(scaler.mean_),
        "scaler_scale_shape": len(scaler.scale_),
    }


def save_reports(
    dataset_id: str,
    report_dir: Path,
    summary: Dict[str, object],
    selected_features: List[str],
    ordered_columns: List[str],
    importance_df: pd.DataFrame,
    robustness_df: pd.DataFrame,
    drop_report_df: pd.DataFrame,
    cleaned_drop_info: Dict[str, List[str]],
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({"feature": selected_features}).to_csv(report_dir / "selected_features.csv", index=False)
    pd.DataFrame({"feature": ordered_columns}).to_csv(report_dir / "ordered_features.csv", index=False)

    if not importance_df.empty:
        importance_df.to_csv(report_dir / "feature_importance.csv", index=False)
    if not robustness_df.empty:
        robustness_df.to_csv(report_dir / "wrapper_robustness.csv", index=False)
    if not drop_report_df.empty:
        drop_report_df.to_csv(report_dir / "filter_drop_report.csv", index=False)

    with open(report_dir / "cleaning_drops.json", "w", encoding="utf-8") as handle:
        json.dump(cleaned_drop_info, handle, ensure_ascii=False, indent=2)

    with open(report_dir / "pipeline_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    lines = [
        f"# {dataset_id} Feature Pipeline Report",
        "",
        f"- pipeline_mode: `{summary['pipeline_mode']}`",
        f"- batteries_loaded: {summary['batteries_loaded']}",
        f"- train_rows_for_fit: {summary['train_rows_for_fit']}",
        f"- selected_feature_count: {len(selected_features)}",
        f"- exported_feature_count: {len(ordered_columns)}",
        f"- used_train_split: {summary['split_meta'].get('used_train_split', False)}",
        "",
        "## Cleaning Drops",
        "",
        "```json",
        json.dumps(cleaned_drop_info, ensure_ascii=False, indent=2),
        "```",
    ]
    with open(report_dir / "pipeline_report.md", "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def run_full_pipeline(args: argparse.Namespace, dataset_data: Dict[str, Dict[str, object]]) -> int:
    export_dir = Path(args.output_dir) / args.dataset_id
    report_dir = Path(args.report_dir) / args.dataset_id
    export_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    cleaned_map = apply_single_battery_cleaning(dataset_data, args.disable_single_clean)
    if not cleaned_map:
        logger.error("All batteries were empty after single battery cleaning.")
        raise SystemExit(1)

    valid_cols, cleaned_drop_info = apply_dataset_cleaning(args.dataset_id, cleaned_map, args.disable_dataset_clean)
    if not valid_cols:
        logger.error("No valid numeric columns remain after dataset cleaning.")
        raise SystemExit(1)

    imputed_map = apply_imputation(args.dataset_id, cleaned_map, valid_cols, args.disable_impute)
    X_full, y_full, battery_ids = build_processed_frame(dataset_data, imputed_map)
    X_full = X_full[[col for col in valid_cols if col in X_full.columns]]
    if X_full.empty:
        logger.error("No features remain after cleaning and imputation.")
        raise SystemExit(1)

    training_mask, split_meta = resolve_training_mask(args.dataset_id, battery_ids, args.fit_on_full_dataset)
    X_train_screen = X_full.loc[training_mask].copy()
    y_train_screen = y_full.loc[training_mask].copy()
    if X_train_screen.empty:
        logger.error("Training screen matrix is empty.")
        raise SystemExit(1)

    filtered_features, drop_report_df = run_filter_pipeline(args, X_train_screen, y_train_screen)
    if not filtered_features:
        logger.error("No features remain after filter pipeline.")
        raise SystemExit(1)

    X_filtered_train = X_train_screen[filtered_features].copy()
    selected_features, importance_df, robustness_df, wrapper_skipped = run_wrapper_pipeline(
        args, X_filtered_train, y_train_screen
    )

    if not selected_features:
        selected_features = filtered_features

    ordered_columns = list(dict.fromkeys(selected_features + [col for col in filtered_features if col not in selected_features]))
    scaler_meta = export_selected_dataset(
        args,
        args.dataset_id,
        imputed_map,
        X_train_screen,
        ordered_columns,
        export_dir,
    )

    summary = {
        "dataset_id": args.dataset_id,
        "pipeline_mode": args.pipeline_mode,
        "batteries_loaded": len(dataset_data),
        "batteries_exported": len(imputed_map),
        "nrows": args.nrows,
        "filter_method": args.filter_method,
        "filter_mode": args.filter_mode,
        "filter_threshold": args.filter_threshold,
        "selector_method": args.selector_method,
        "top_k": args.top_k,
        "n_seeds": args.n_seeds,
        "wrapper_skipped": wrapper_skipped,
        "train_rows_for_fit": int(len(X_train_screen)),
        "train_batteries_for_fit": sorted(set(battery_ids[training_mask].tolist())),
        "split_meta": split_meta,
        "scaler_meta": scaler_meta,
    }

    save_reports(
        args.dataset_id,
        report_dir,
        summary,
        selected_features,
        ordered_columns,
        importance_df,
        robustness_df,
        drop_report_df,
        cleaned_drop_info,
    )

    logger.info("Full feature pipeline complete for %s", args.dataset_id)
    logger.info("Selected feature export: %s", export_dir)
    logger.info("Feature reports: %s", report_dir)
    return 0


def main() -> int:
    args = parse_args()
    try:
        target_datasets = resolve_target_datasets(args.dataset_id)
    except ValueError as exc:
        logger.error(str(exc))
        return 1

    logger.info(
        "Resolved datasets (%d): %s",
        len(target_datasets),
        target_datasets,
    )

    failed_datasets: List[str] = []
    for dataset_id in target_datasets:
        args.dataset_id = dataset_id
        logger.info(
            "Starting feature pipeline for %s | mode=%s | filter=%s | wrapper=%s | seeds=%d",
            args.dataset_id,
            args.pipeline_mode,
            args.filter_method,
            args.selector_method,
            args.n_seeds,
        )

        try:
            dataset_data = load_dataset(args)
            if args.pipeline_mode == "aggregate_legacy":
                aggregate_legacy_pipeline(args, dataset_data)
            else:
                run_full_pipeline(args, dataset_data)
        except SystemExit as exc:
            logger.error("Pipeline failed for %s (exit=%s)", dataset_id, exc)
            failed_datasets.append(dataset_id)
        except Exception as exc:
            logger.exception("Pipeline failed for %s with unexpected error: %s", dataset_id, exc)
            failed_datasets.append(dataset_id)

    if failed_datasets:
        logger.error("Feature pipeline finished with failures: %s", failed_datasets)
        return 1

    logger.info("Feature pipeline finished successfully for all datasets.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

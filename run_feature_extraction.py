import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parent
FEATURES_DIR = PROJECT_ROOT / "features"

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

DATASET_SCRIPT_MAP: Dict[str, str] = {
    "CALB": "features_CALB.py",
    "CALCE": "features_CALCE.py",
    "HNEI": "features_HNEI.py",
    "HUST": "features_HUST.py",
    "ISU-ILCC": "features_ISU-ILCC.py",
    "MATR": "features_MATR.py",
    "MICH": "features_MICH.py",
    "MICH_EXP": "features_MICH_EXP.py",
    "Na": "features_Na.py",
    "RWTH": "features_RWTH.py",
    "SNL": "features_SNL.py",
    "Stanford": "features_Stanford.py",
    "Stanford_2": "features_Stanford.py",
    "Tongji": "features_Tongji.py",
    "UL_PUR": "features_UL_PUR.py",
    "XJTU": "features_XJTU.py",
    "ZN-coin": "features_ZNion.py",
}


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("RunFeatureExtraction")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BatteryMap feature extraction entrypoint")
    parser.add_argument(
        "--dataset_id",
        type=str,
        default=None,
        help=(
            "Dataset ID (e.g., CALCE). Supports comma-separated list like CALCE,HUST. "
            "If omitted, all supported datasets are processed."
        ),
    )
    parser.add_argument(
        "--num_cycles",
        type=int,
        default=None,
        help="Number of cycles to process (default: all)",
    )
    return parser.parse_args()


def resolve_target_datasets(dataset_id_arg: str | None) -> List[str]:
    if not dataset_id_arg:
        return SUPPORTED_DATASETS.copy()

    requested: List[str] = []
    for token in str(dataset_id_arg).split(","):
        dataset_name = token.strip()
        if dataset_name:
            requested.append(dataset_name)

    invalid = [name for name in requested if name not in SUPPORTED_DATASETS]
    if invalid:
        raise ValueError("当前数据集并未处理，请选择白名单中的电池")

    return list(dict.fromkeys(requested))


def build_script_args(script_name: str, dataset_names: List[str], num_cycles: int | None) -> List[str]:
    args = [sys.executable, str(FEATURES_DIR / script_name)]
    if num_cycles is not None:
        args.extend(["--num_cycles", str(num_cycles)])

    if script_name == "features_Stanford.py":
        args.extend(["--dataset_ids", ",".join(dataset_names)])

    return args


def run_extraction_for_targets(target_datasets: List[str], num_cycles: int | None) -> int:
    script_to_datasets: Dict[str, List[str]] = {}
    for dataset in target_datasets:
        script_name = DATASET_SCRIPT_MAP[dataset]
        script_to_datasets.setdefault(script_name, []).append(dataset)

    failed_groups: List[str] = []
    for script_name, datasets in script_to_datasets.items():
        script_path = FEATURES_DIR / script_name
        if not script_path.exists():
            logger.error("Feature script not found: %s", script_path)
            failed_groups.append(f"{script_name} -> {datasets}")
            continue

        logger.info("Running feature extraction script %s for datasets: %s", script_name, datasets)
        cmd = build_script_args(script_name, datasets, num_cycles)
        try:
            subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))
        except subprocess.CalledProcessError as exc:
            logger.error("Extraction failed for %s (datasets=%s): %s", script_name, datasets, exc)
            failed_groups.append(f"{script_name} -> {datasets}")

    if failed_groups:
        logger.error("Feature extraction finished with failures: %s", failed_groups)
        return 1

    logger.info("Feature extraction finished successfully for all requested datasets.")
    return 0


def main() -> int:
    args = parse_args()
    try:
        target_datasets = resolve_target_datasets(args.dataset_id)
    except ValueError as exc:
        logger.error(str(exc))
        return 1

    logger.info("Resolved datasets (%d): %s", len(target_datasets), target_datasets)
    return run_extraction_for_targets(target_datasets, args.num_cycles)


if __name__ == "__main__":
    raise SystemExit(main())
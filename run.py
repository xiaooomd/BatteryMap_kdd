"""Unified project entry point."""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

ENTRYPOINTS = {
    "predict": PROJECT_ROOT / "run_main.py",
    "extraction": PROJECT_ROOT / "run_feature_extraction.py",
    "selection": PROJECT_ROOT / "run_feature_selection.py",
    "evaluate": PROJECT_ROOT / "scripts" / "evaluate_model.py",
    "finetune": PROJECT_ROOT / "scripts" / "finetune.py",
    "domain-adaptation": PROJECT_ROOT / "scripts" / "domainAdaptation.py",
    "hyperopt": PROJECT_ROOT / "scripts" / "hyperparameter_optimization.py",
    "multi-dataset-opt": PROJECT_ROOT / "scripts" / "run_multi_dataset_optimization.py",
    "sensitivity": PROJECT_ROOT / "scripts" / "run_sensitivity_analysis.py",
    "autoformer-batch": PROJECT_ROOT / "scripts" / "run_autoformer_batch.py",
    "hyperopt-batch": PROJECT_ROOT / "scripts" / "run_hyperparam_search.py",
    "view-results": PROJECT_ROOT / "scripts" / "view_results.py",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified BatteryMap entry point.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=sorted(ENTRYPOINTS.keys()),
        help="Command to execute.",
    )
    parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to the selected command.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    parsed = parser.parse_args()

    if not parsed.command:
        parser.print_help()
        return 0

    script_path = ENTRYPOINTS[parsed.command]
    sys.argv = [str(script_path), *parsed.args]
    runpy.run_path(str(script_path), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

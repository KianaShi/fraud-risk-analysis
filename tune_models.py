"""Tune the frozen 14-feature XGBoost and CatBoost benchmarks with Optuna."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from fraud_detection.tuning import run_tuning_experiment


def run(
    input_path: Path,
    results_dir: Path,
    figures_dir: Path,
    xgb_trials: int,
    catboost_trials: int,
) -> None:
    """Run Train-only tuning, Validation selection, and final Test confirmation."""
    outputs = run_tuning_experiment(
        pd.read_csv(input_path),
        results_dir,
        figures_dir,
        xgb_trials=xgb_trials,
        catboost_trials=catboost_trials,
    )
    print("\nDefault vs tuned Validation comparison:")
    print(outputs["validation_comparison"].round(4).to_string(index=False))
    print("\nFrozen configuration Test confirmation:")
    print(outputs["final_test_comparison"].round(4).to_string(index=False))


def parse_args() -> argparse.Namespace:
    """Parse configurable trial counts and artifact paths."""
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=root / "outputs" / "p2_dataset_clean.csv")
    parser.add_argument("--results-dir", type=Path, default=root / "docs" / "results")
    parser.add_argument("--figures-dir", type=Path, default=root / "docs" / "figures")
    parser.add_argument("--trials", type=int, default=40, help="Default trials for each model.")
    parser.add_argument("--xgb-trials", type=int, default=None)
    parser.add_argument("--catboost-trials", type=int, default=None)
    args = parser.parse_args()
    args.xgb_trials = args.xgb_trials if args.xgb_trials is not None else args.trials
    args.catboost_trials = (
        args.catboost_trials if args.catboost_trials is not None else args.trials
    )
    return args


if __name__ == "__main__":
    arguments = parse_args()
    run(
        arguments.input,
        arguments.results_dir,
        arguments.figures_dir,
        arguments.xgb_trials,
        arguments.catboost_trials,
    )

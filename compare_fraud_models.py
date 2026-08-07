"""Benchmark XGBoost and CatBoost on stratified Task 2 partitions."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from fraud_detection.modeling import compare_models


def run(input_path: Path, output_path: Path, split_output_path: Path) -> None:
    """Evaluate models and save comparison and split-audit tables."""
    comparison, _, audit = compare_models(pd.read_csv(input_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    split_output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.round(4).to_csv(output_path, index=False)
    audit.round(4).to_csv(split_output_path, index=False)
    print("Split audit:")
    print(audit.round(4).to_string(index=False))
    print("\nModel comparison:")
    print(comparison.round(4).to_string(index=False))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=root / "outputs" / "p2_dataset_clean.csv")
    parser.add_argument("--output", type=Path, default=root / "outputs" / "model_comparison.csv")
    parser.add_argument("--split-output", type=Path, default=root / "outputs" / "split_audit.csv")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.input, args.output, args.split_output)
